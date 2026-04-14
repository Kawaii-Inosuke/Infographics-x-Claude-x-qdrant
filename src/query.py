import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import google.generativeai as genai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


EMBEDDING_MODEL = "models/gemini-embedding-001"


def load_settings() -> tuple[str, str, str]:
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    missing = []
    if not gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if not qdrant_url:
        missing.append("QDRANT_URL")
    if not qdrant_api_key:
        missing.append("QDRANT_API_KEY")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)} (set them in .env)")
    return gemini_api_key, qdrant_url, qdrant_api_key


def infer_brand_name(brand_dir: Path) -> str:
    return brand_dir.resolve().name


def make_qdrant_client(qdrant_url: str, qdrant_api_key: str) -> QdrantClient:
    parsed = urlparse(qdrant_url.strip())
    scheme = parsed.scheme or "https"
    https = scheme != "http"
    host = parsed.hostname or qdrant_url.replace("https://", "").replace("http://", "").strip("/").split("/")[0]
    port = parsed.port or (443 if https else 80)
    return QdrantClient(
        host=host,
        port=port,
        https=https,
        api_key=qdrant_api_key,
        timeout=60,
        check_compatibility=False,
    )


def embed(text: str) -> list[float]:
    resp = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_query")
    if hasattr(resp, "embedding"):
        return resp.embedding
    if isinstance(resp, dict) and resp.get("embedding"):
        return resp["embedding"]
    raise RuntimeError("Gemini returned empty embedding")


def main() -> int:
    parser = argparse.ArgumentParser(description="Search similar infographics in Qdrant.")
    parser.add_argument("query", type=str, help="Search text")
    parser.add_argument(
        "--brand",
        type=str,
        default=None,
        help="Brand name (also Qdrant collection). Uses folder <base-dir>/<brand>/",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "brands"),
        help="Base directory containing brand folders (default: 'brands' folder in project root)",
    )
    parser.add_argument("--brand-dir", type=str, default=".", help="Brand folder name to infer collection (default: .)")
    parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")
    args = parser.parse_args()

    if args.brand:
        base_dir = Path(args.base_dir).expanduser().resolve()
        brand_dir = (base_dir / args.brand).resolve()
        brand = args.brand
    else:
        brand_dir = Path(args.brand_dir).expanduser().resolve()
        brand = infer_brand_name(brand_dir)

    try:
        gemini_key, qdrant_url, qdrant_key = load_settings()
    except Exception as e:
        print(f"[error] {e}")
        return 2

    qdrant = make_qdrant_client(qdrant_url, qdrant_key)
    if not qdrant.collection_exists(collection_name=brand):
        print(f"[error] Qdrant collection not found: {brand}")
        return 2

    genai.configure(api_key=gemini_key)
    query_vec = embed(args.query)

    hits = qdrant.search(
        collection_name=brand,
        query_vector=query_vec,
        limit=max(1, args.limit),
        with_payload=True,
    )

    print(f"[info] collection: {brand}")
    print(f"[info] results: {len(hits)}\n")

    for i, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        filename = payload.get("filename", "<unknown>")
        desc = (payload.get("description", "") or "").strip().replace("\n", " ")
        if len(desc) > 220:
            desc = desc[:220] + "…"
        score = getattr(hit, "score", None)
        print(f"{i}. score={score:.4f}  file={filename}")
        if desc:
            print(f"   {desc}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

