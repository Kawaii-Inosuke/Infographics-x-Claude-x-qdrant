# FCC Infographic Ingestion → Qdrant Cloud (Gemini)

This folder ingests all `.jpg/.jpeg/.png` infographics in the **brand folder** into **Qdrant Cloud**, storing:

- embedding (`models/gemini-embedding-001`)
- Gemini Vision description (`models/gemini-flash-latest`)
- filename
- brand name (folder name)

Duplicate handling: each image is assigned a deterministic point ID derived from its **SHA-256 hash**, so re-running ingestion will **skip already-uploaded files**.

## Setup

From this folder:

```bash
cd "/home/yashodhan/91Ninjas/Infographics claude/FCC"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Fill in `.env` (template is included in this folder):

```bash
GEMINI_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
```

## Ingest

Ingest all images in the brand folder (collection name is inferred from the folder name: `FCC`):

```bash
PYTHONUNBUFFERED=1 python ingest.py --brand-dir .
```

Optional flags:

- `--max-side 1600`: resizes images before sending to Gemini (set `0` to disable)
- `--sleep 0.2`: sleep between files (helpful for rate limiting)

## Query

Search the Qdrant collection using a text query:

```bash
python query.py "infographic comparing absorption costing vs variable costing" --brand-dir . --limit 5
```

## Notes

- Qdrant collection name is **always** the brand folder name.
- Payload includes: `brand`, `filename`, `path`, `sha256`, `description`, model names, and `ingested_at_unix`.
