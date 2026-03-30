import arxiv

def _get_pdf_url_patch(links) -> str:
    """
    Finds the PDF link among a result's links and returns its URL.
    Should only be called once for a given `Result`, in its constructor.
    After construction, the URL should be available in `Result.pdf_url`.
    """
    pdf_urls = [link.href for link in links if "pdf" in link.href]
    if len(pdf_urls) == 0:
        return None
    return pdf_urls[0]

arxiv.Result._get_pdf_url = _get_pdf_url_patch

import argparse
import os
import sys
import re
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import feedparser
import requests
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm

ARXIV_PAGE_SIZE = 50
ARXIV_MIN_PAGE_SIZE = 10
ARXIV_REQUEST_DELAY_SECONDS = 5.0
ARXIV_RETRY_BASE_DELAY_SECONDS = 15.0
ARXIV_MAX_BACKOFF_SECONDS = 120.0
ARXIV_MAX_PAGE_FAILURES = 5
ARXIV_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
ARXIV_USER_AGENT = "zotero-arxiv-daily/0.3.5 (+https://github.com/TideDra/zotero-arxiv-daily)"


def build_arxiv_api_query(query: str) -> str:
    categories = [category.strip() for category in query.split('+') if category.strip()]
    if not categories:
        raise Exception("Invalid ARXIV_QUERY: empty query.")

    invalid_categories = [category for category in categories if not re.fullmatch(r"[A-Za-z0-9.\-]+", category)]
    if invalid_categories:
        raise Exception(f"Invalid ARXIV_QUERY: unsupported category name(s): {', '.join(invalid_categories)}.")

    return " OR ".join(f"cat:{category}" for category in categories)

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus

def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    _,filename = mkstemp()
    with open(filename,'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename,base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def iter_arxiv_results(search: arxiv.Search):
    client = arxiv.Client(
        page_size=ARXIV_PAGE_SIZE,
        num_retries=0,
        delay_seconds=ARXIV_REQUEST_DELAY_SECONDS,
    )
    start = 0
    page_size = client.page_size
    total_results = None
    last_request_monotonic = None
    failure_count = 0

    while True:
        remaining = None if search.max_results is None else search.max_results - start
        if remaining is not None and remaining <= 0:
            break

        current_page_size = page_size if remaining is None else min(page_size, remaining)
        if last_request_monotonic is not None:
            sleep_seconds = client.delay_seconds - (time.monotonic() - last_request_monotonic)
            if sleep_seconds > 0:
                logger.info(f"Sleeping {sleep_seconds:.1f}s before the next arXiv request.")
                time.sleep(sleep_seconds)

        page_url = client._format_url(search, start, current_page_size)
        logger.info(f"Requesting arXiv page: start={start}, size={current_page_size}")
        request_started = time.monotonic()
        try:
            response = client._session.get(
                page_url,
                headers={"user-agent": ARXIV_USER_AGENT},
                timeout=60,
            )
            last_request_monotonic = request_started
            if response.status_code != requests.codes.OK:
                raise arxiv.HTTPError(page_url, failure_count, response.status_code)

            feed = feedparser.parse(response.content)
            if len(feed.entries) == 0 and start > 0:
                raise arxiv.UnexpectedEmptyPageError(page_url, failure_count, feed)
        except (
            arxiv.HTTPError,
            arxiv.UnexpectedEmptyPageError,
            requests.exceptions.RequestException,
        ) as err:
            last_request_monotonic = request_started
            status = getattr(err, "status", None)
            retryable = (
                status in ARXIV_RETRYABLE_STATUS_CODES
                or isinstance(err, arxiv.UnexpectedEmptyPageError)
                or isinstance(err, requests.exceptions.RequestException)
            )
            if not retryable:
                raise

            failure_count += 1
            if failure_count > ARXIV_MAX_PAGE_FAILURES:
                logger.error(f"arXiv request failed too many times for page starting at {start}.")
                raise

            if status == 429 and page_size > ARXIV_MIN_PAGE_SIZE:
                next_page_size = max(ARXIV_MIN_PAGE_SIZE, page_size // 2)
                if next_page_size != page_size:
                    logger.warning(
                        f"arXiv rate limited the request. Reduce page size from {page_size} to {next_page_size}."
                    )
                    page_size = next_page_size

            backoff_seconds = min(
                ARXIV_RETRY_BASE_DELAY_SECONDS * (2 ** (failure_count - 1)),
                ARXIV_MAX_BACKOFF_SECONDS,
            )
            logger.warning(
                f"Retryable arXiv error ({err}). Backing off for {backoff_seconds:.1f}s "
                f"before retry {failure_count}/{ARXIV_MAX_PAGE_FAILURES}."
            )
            time.sleep(backoff_seconds)
            continue

        failure_count = 0
        if total_results is None:
            total_results_raw = getattr(feed.feed, "opensearch_totalresults", None)
            if total_results_raw is not None:
                total_results = int(total_results_raw)
                logger.info(
                    f"Got first arXiv page: {len(feed.entries)} entries out of {total_results} total results."
                )

        if not feed.entries:
            logger.warning("arXiv API returned no results.")
            break

        for entry in feed.entries:
            try:
                yield arxiv.Result._from_feed_entry(entry)
            except arxiv.Result.MissingFieldError as err:
                logger.warning(f"Skipping partial arXiv result: {err}")

        start += len(feed.entries)
        if total_results is not None and start >= total_results:
            break


def get_arxiv_paper(query:str, debug:bool=False) -> list[ArxivPaper]:
    if not debug:
        search = arxiv.Search(
            query=build_arxiv_api_query(query),
            max_results=1000,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        papers = []
        seen_ids = set()
        latest_batch_date = None
        for result in tqdm(iter_arxiv_results(search), desc="Retrieving Arxiv papers"):
            published = getattr(result, "published", None)
            if published is None:
                continue

            if latest_batch_date is None:
                latest_batch_date = published.date()
                logger.info(f"Latest arXiv batch date: {latest_batch_date}")

            if published.date() != latest_batch_date:
                break

            paper = ArxivPaper(result)
            if paper.arxiv_id in seen_ids:
                continue
            seen_ids.add(paper.arxiv_id)
            papers.append(paper)

        if latest_batch_date is None:
            logger.warning(f"arXiv API returned no results for query: {query}")
        else:
            logger.info(f"Retrieved {len(papers)} papers from the latest arXiv batch.")

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(
            query='cat:cs.AI',
            max_results=5,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        papers = []
        for i in iter_arxiv_results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break

    return papers


def load_sent_paper_ids(history_file: str) -> set[str]:
    path = Path(history_file)
    if not path.exists():
        return set()

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        logger.warning(f"History file {path} is invalid JSON. Ignore it.")
        return set()

    ids = payload.get("sent_paper_ids", [])
    if not isinstance(ids, list):
        logger.warning(f"History file {path} has invalid sent_paper_ids. Ignore it.")
        return set()

    return {paper_id for paper_id in ids if isinstance(paper_id, str) and paper_id}


def save_sent_paper_ids(history_file: str, sent_paper_ids: set[str]) -> None:
    path = Path(history_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "sent_paper_ids": sorted(sent_paper_ids),
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def filter_sent_papers(papers: list[ArxivPaper], sent_paper_ids: set[str]) -> list[ArxivPaper]:
    if not sent_paper_ids:
        return papers

    filtered_papers = [paper for paper in papers if paper.arxiv_id not in sent_paper_ids]
    skipped_num = len(papers) - len(filtered_papers)
    if skipped_num > 0:
        logger.info(f"Skipped {skipped_num} papers already sent before.")
    return filtered_papers



parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    add_argument(
        "--history_file",
        type=str,
        help="Path of sent-paper history for deduplication across runs",
        default=".state/sent_arxiv_ids.json",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    if not args.debug:
        sent_paper_ids = load_sent_paper_ids(args.history_file)
        logger.info(f"Loaded {len(sent_paper_ids)} previously sent paper IDs from history.")
        papers = filter_sent_papers(papers, sent_paper_ids)

    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    html = render_email(papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    if not args.debug and len(papers) > 0:
        sent_paper_ids.update(p.arxiv_id for p in papers)
        save_sent_paper_ids(args.history_file, sent_paper_ids)
        logger.info(f"Saved {len(sent_paper_ids)} sent paper IDs to {args.history_file}.")
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")
