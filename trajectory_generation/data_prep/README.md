# Trajectory Generation Data Prep

## Chunking

### How it works

**Local chunks** (`chunks_local_all.jsonl`): Produced by the making_dataset pipeline using `ContentProcessor.chunk_content()` — character-based with sentence/paragraph boundaries, default 1000 chars, 200 overlap. Local chunking uses `chunk_local.py` (target ~450 words, 50 overlap, min 300 words) on DrBench task files.

**Web chunks** (`chunks_web_drbench_urls.jsonl`): Produced by `chunk_web_drbench.py` — paragraph/heading-aware:

- Parses markdown-style headings (# ## ###) for context
- Splits at paragraph boundaries
- target_words ~260, overlap ~50, min ~140
- Adds heading prefix to each chunk (Title > H2 > H3) for retrieval

**BrowseComp-Plus** (`chunks_web.jsonl`): Doc-level chunks from Tevatron corpus; each line is a full document.

### Chunk sizes (observed)

| Source | Chars (min–max) | Avg chars |
|--------|-----------------|-----------|
| DrBench URL chunks | 784–10,783 | ~2,230 |
| Local chunks | 174–9,439 | ~2,670 |

## DrBench URLs

**Source:** Each task's `context.json` has a `url` block:

```json
"url": {
  "url": "https://example.com/article",
  "industry": "retail",
  "domain": "compliance",
  "date": "2024-08-05"
}
```

`build_url_to_task_ids(tasks_root)` scans `drbench/data/tasks/*/context.json` and maps URL → [task_ids]. Tasks without `context.json` or without a `url` field are skipped.

**Current coverage:** `chunks_web_drbench_urls.jsonl` was built from tasks DR0001–DR0015 (15 tasks). Tasks DR0016+ may have context.json with URLs but haven't been fetched/chunked yet.

## Pipeline

1. **Fetch** (external): Fetch URLs → `docs_web_drbench_urls.jsonl` (requires requests, BeautifulSoup, etc.)
2. **Chunk**: `python -m trajectory_generation.data_prep.chunk_web_drbench --docs docs.jsonl --output chunks.jsonl`
