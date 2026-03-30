# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zotero-arXiv-Daily is a Python application that recommends new arXiv papers based on a user's Zotero library. It calculates paper similarity using sentence embeddings, generates TL;DR summaries with LLM, and sends daily email digests via GitHub Actions or Docker.

## Commands

```bash
# Run the application locally (requires environment variables to be set)
uv run main.py

# Run in debug mode (retrieves 5 papers regardless of date, skips history dedup)
uv run main.py --debug

# Sync dependencies
uv sync
```

## Architecture

The application is a single-entry script with four core modules:

- **main.py**: Entry point. Orchestrates the full workflow: fetch Zotero papers, retrieve arXiv papers, filter duplicates, rerank by relevance, generate TL;DRs, render HTML email, and send via SMTP.

- **paper.py**: `ArxivPaper` class wraps arxiv.Result. Handles PDF URL extraction, LaTeX source downloading/parsing (for intro/conclusion extraction), TL;DR generation via LLM, and affiliation extraction. Uses `@cached_property` for lazy evaluation.

- **recommender.py**: `rerank_paper()` computes similarity scores between candidate papers and Zotero corpus using `GIST-small-Embedding-v0` model. Applies time-decay weighting (newer Zotero papers have higher weight).

- **llm.py**: LLM abstraction. Supports either OpenAI-compatible API (when `USE_LLM_API=1`) or local GGUF model via llama-cpp-python (Qwen2.5-3B by default). Global singleton pattern.

- **construct_email.py**: Renders HTML email from paper list. Extracts affiliations and TL;DR (triggers LLM calls). `send_email()` handles SMTP with TLS/SSL fallback.

## Data Flow

1. Fetch papers from Zotero API (filtered by `ZOTERO_IGNORE` gitignore patterns)
2. Query arXiv for papers from the latest batch date
3. Filter out previously sent papers using `.state/sent_arxiv_ids.json`
4. Compute embeddings and rerank by similarity to Zotero corpus
5. Truncate to `MAX_PAPER_NUM` papers
6. For each paper: download LaTeX source, extract intro/conclusion, generate TL;DR
7. Render HTML email and send via SMTP
8. Save sent paper IDs to history file

## Key Configuration

All configuration is via environment variables (see README.md for full list). Critical ones:

| Variable | Description |
|----------|-------------|
| ZOTERO_ID, ZOTERO_KEY | Zotero API credentials |
| ARXIV_QUERY | Categories like `cs.AI+cs.CV` |
| USE_LLM_API | `1` for API, `0` for local LLM |
| OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME | LLM API config |
| SMTP_* | Email delivery config |

## GitHub Actions

Two workflows:
- `main.yml`: Runs daily at 22:00 UTC. Uses Actions cache to persist `.state/sent_arxiv_ids.json`.
- `test.yml`: Manual trigger with `--debug` flag, skips history persistence.

The workflow auto-detects entrypoint (`main.py` or `src/**/main.py`).

## Development Notes

- **Package manager**: Use `uv` only. Pip/conda are not tested.
- **Branching**: PRs should target `dev` branch, not `main`.
- **Python version**: Requires 3.11+
- **No tests**: This project has no automated test suite.
- **Local LLM**: First run downloads ~3GB model file.