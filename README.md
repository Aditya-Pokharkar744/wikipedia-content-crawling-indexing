# Wikipedia Content Crawling and Indexing System

An end-to-end search system that crawls 300K+ Wikipedia articles and indexes them through two complementary backends — BM25 keyword search via Elasticsearch and semantic search via Sentence Transformers and FAISS — served through a Flask interface with real-time mode toggling and ranked results.

## Features

- **Scrapy crawler** — seed URLs from a text file, hop-limited link traversal, URL deduplication, citation cleanup, JSON export (`title`, `url`, `content`).
- **Elasticsearch** — custom analyzer (standard tokenizer, lowercase, stop, snowball), `streaming_bulk` indexing, title-boosted fuzzy `multi_match` search.
- **Semantic search** — Sentence Transformers embeddings (`sentence-transformers/all-mpnet-base-v2`), passage splitting, FAISS index (CLI under `indexing/bert_similarity/`).
- **Flask UI** — `app.py` serves a search page; toggle between Elasticsearch and BERT-style search; JSON API for ranked results with scores.

## Architecture

```text
seed.txt  →  Scrapy spider  →  JSON corpus (title / url / content)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            Elasticsearch index              Embeddings + FAISS
            (BM25 / custom analyzer)         (Sentence Transformers)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                            Flask app (browser UI)
```

## Repository layout

| Path | Purpose |
|------|---------|
| `crawling/wikipedia_scraper/` | Scrapy project: spider, settings, `crawler.sh` |
| `indexing/elasticsearch/` | `document_indexer.py`, `evaluate.py`, `run.sh` |
| `indexing/bert_similarity/` | Embedding extraction, FAISS build, query CLI |
| `app.py` | Flask web server |
| `templates/`, `static/` | HTML/CSS/JS for the search UI |
| `Data/` | Example crawled JSON (large files may be gitignored) |
| `utils/` | Helper scripts |

## Prerequisites

- **Python** 3.8+ recommended.
- **Elasticsearch 7.x** on `http://localhost:9200` (for keyword indexing and the ES mode in the Flask app).
- **Optional** — CUDA for faster embedding; `faiss-cpu` in `requirements.txt` works without a GPU. Semantic mode in the UI may call a separate BERT service on port 5001 if you run one alongside `app.py`.

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

Core packages include Scrapy, Elasticsearch client, ijson, Flask, sentence-transformers, FAISS (CPU), PyTorch, NumPy, and Matplotlib (for the indexing evaluation script).

## Crawler

Scripts live under `crawling/wikipedia_scraper/wikipedia_scraper/spiders/`.

```bash
# From repo root (use Git Bash or WSL on Windows for sh)
cd crawling/wikipedia_scraper/wikipedia_scraper/spiders

# sh crawler.sh <seed-file> <max-pages> <hops-away> <output-dir>
sh crawler.sh seed.txt 100 6 output_dir
```

Alternatively, run the spider directly:

```bash
cd crawling/wikipedia_scraper/wikipedia_scraper/spiders
python wikipedia_spider.py seed.txt 100 6 output_dir
```

Output: JSON feed (e.g. `output_dir/output.json`) as a list of objects with `title`, `url`, and `content`.

## Elasticsearch: index and search

Run from `indexing/elasticsearch/` (the wrapper script invokes `document_indexer.py` in that directory).

```bash
cd indexing/elasticsearch

# Index all .json files in a directory (Elasticsearch must be running)
bash run.sh index /path/to/json/folder [--force] [--index-name documents_index]

# Search
bash run.sh search "your query here" --k 10
```

On Windows without Bash, use Python directly from the same folder:

```powershell
cd indexing\elasticsearch
python document_indexer.py index "D:\path\to\json\folder"
python document_indexer.py search "maximum likelihood estimate" --k 5
```

**Evaluation** — `evaluate.py` plots indexing time vs. document count (requires a JSON path and index name configured inside the script).

## Semantic search (Sentence Transformers + FAISS)

High-level pipeline (paths under `indexing/bert_similarity/`):

1. **Embeddings** — `extract_embeddings.py` reads a JSON corpus and writes a pickle of embeddings + passages + document metadata (edit input/output paths in the script or pass arguments if your copy supports them).
2. **FAISS index** — `faiss_indexer.py` loads the embeddings pickle and writes `faiss_index.bin` + `metadata.pkl` under an output directory (e.g. `index/`).
3. **Query** — `query_index.py` or `run_query.sh` runs semantic search against that index directory.

These steps are compute-heavy; adjust batch sizes and use CPU flags when resources are limited. Before running, point `extract_embeddings.py` (and CLI defaults such as `--index-dir` in `query_index.py`) at your local JSON and index paths.

## Flask web interface

```bash
# From repository root
python app.py
```

Then open `http://127.0.0.1:5000` (default). The UI posts to `/search` and can use **Elasticsearch** if the index exists, or **BERT** mode if a compatible service is available at `BERT_SERVER_URL` (default `http://localhost:5001`). Environment variable `USE_GPU` may influence local embedding code paths when using GPU-capable stacks.

## Sample data and JSON shape

Example corpora in the repo may include `Data/data.json` or crawler output under `crawling/.../spiders/output_dir/`. Each document should look like:

```json
[
  {
    "title": "Computer science",
    "url": "https://en.wikipedia.org/wiki/Computer_science",
    "content": "Computer science is the study of computer ..."
  },
  {
    "title": "Computer engineering",
    "url": "https://en.wikipedia.org/wiki/Computer_engineering",
    "content": "Computer engineering (CE or CoE or CpE) is a branch of engineering specialized in developing computer hardware and software. It integrates several fields of electrical engineer..."
  }
]
```

## Limitations

- Wikipedia HTML structure ties extraction quality to `mw-parser-output` selectors.
- Large JSON, embedding pickles, and FAISS binaries should stay out of version control; use `.gitignore` and document paths locally.
