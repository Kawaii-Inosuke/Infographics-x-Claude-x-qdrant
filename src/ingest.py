import argparse
import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}
# Note: the python `google-generativeai` SDK expects model names like `models/...`.
# Your API key does not expose `models/gemini-1.5-flash`, so we use the closest
# available Flash model.
VISION_MODEL = "models/gemini-flash-latest"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 3072


PROMPT = """You are analyzing an infographic image.

Write a detailed, structured description suitable for semantic search. Focus on:
- Layout (sections, hierarchy, typography, alignment, spacing, grid)
- Color palette (primary/secondary colors, contrast, background)
- Style (illustration style, icon style, chart/table style, brand vibe)
- Visual elements (icons, charts, diagrams, callouts)
- Content theme/concept (what the infographic explains, key topics)
- Any visible headings or key phrases (quote them verbatim when possible)

Return plain text only. Be specific and concrete."""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    qdrant_url: str
    qdrant_api_key: str


def load_settings() -> Settings:
    load_dotenv()
    missing = []
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if not qdrant_url:
        missing.append("QDRANT_URL")
    if not qdrant_api_key:
        missing.append("QDRANT_API_KEY")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)} (set them in .env)")
    return Settings(
        gemini_api_key=gemini_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )


def infer_brand_name(brand_dir: Path) -> str:
    return brand_dir.resolve().name


def list_images(brand_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(brand_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def point_id_for(brand: str, file_hash: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"infographic:{brand}:{file_hash}"))


def load_image_for_gemini(path: Path, max_side: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max_side > 0:
            w, h = im.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)))
        return im


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    exists = client.collection_exists(collection_name=collection_name)
    if exists:
        try:
            info = client.get_collection(collection_name=collection_name)
            existing_size = None
            vectors = getattr(info.config.params, "vectors", None)
            if isinstance(vectors, qdrant_models.VectorParams):
                existing_size = vectors.size
            elif hasattr(vectors, "default") and isinstance(vectors.default, qdrant_models.VectorParams):
                existing_size = vectors.default.size

            if existing_size is not None and existing_size != EMBEDDING_DIM:
                cnt = client.count(collection_name=collection_name, exact=True)
                points = getattr(cnt, "count", None)
                points = int(points) if points is not None else 0
                if points == 0:
                    print(
                        f"[qdrant] collection exists but has wrong vector size "
                        f"(expected {EMBEDDING_DIM}, got {existing_size}); recreating (collection is empty)"
                    )
                    client.delete_collection(collection_name=collection_name)
                else:
                    raise RuntimeError(
                        f"Qdrant collection '{collection_name}' has vector size {existing_size}, "
                        f"but this run requires {EMBEDDING_DIM}. Collection has {points} points; "
                        "refusing to recreate automatically."
                    )
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            print(f"[qdrant] warning: could not verify collection vector size ({e}); continuing")

        if client.collection_exists(collection_name=collection_name):
            print(f"[qdrant] collection exists: {collection_name}")
            return
    print(f"[qdrant] creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(size=EMBEDDING_DIM, distance=qdrant_models.Distance.COSINE),
    )
    print(f"[qdrant] created collection: {collection_name}")


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


def already_uploaded(client: QdrantClient, collection_name: str, point_id: str) -> bool:
    got = client.retrieve(collection_name=collection_name, ids=[point_id], with_payload=False, with_vectors=False)
    return bool(got)


def describe_with_gemini(image: Image.Image) -> str:
    model = genai.GenerativeModel(VISION_MODEL)
    resp = model.generate_content([PROMPT, image])
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty description")
    return text


def embed_text(text: str) -> list[float]:
    resp = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
    emb = None
    if hasattr(resp, "embedding"):
        emb = resp.embedding
    elif isinstance(resp, dict):
        emb = resp.get("embedding")
    if not emb:
        raise RuntimeError("Gemini returned empty embedding")
    return emb


def upsert_point(
    client: QdrantClient,
    collection_name: str,
    point_id: str,
    embedding: list[float],
    payload: dict,
) -> None:
    client.upsert(
        collection_name=collection_name,
        points=[
            qdrant_models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest infographic images into Qdrant Cloud.")
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
    parser.add_argument("--brand-dir", type=str, default=None, help="Specific folder containing brand images")
    parser.add_argument("--max-side", type=int, default=1600, help="Resize image so max side <= N (0 disables)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep seconds between files")
    args = parser.parse_args()

    brand = args.brand
    if not brand:
        brand = input("Enter Brand Name (e.g., FCC): ").strip()
    
    if not brand:
        print("[error] Brand name is required.")
        return 2

    base_dir = Path(args.base_dir).expanduser().resolve()
    
    if args.brand_dir:
        brand_dir = Path(args.brand_dir).expanduser().resolve()
    else:
        brand_dir = (base_dir / brand).resolve()
    if not brand_dir.exists() or not brand_dir.is_dir():
        print(f"[error] brand folder not found: {brand_dir}")
        return 2

    print(f"[info] brand folder: {brand_dir}")
    print(f"[info] brand/collection: {brand}")

    try:
        settings = load_settings()
    except Exception as e:
        print(f"[error] {e}")
        return 2

    qdrant = make_qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    ensure_collection(qdrant, brand)
    genai.configure(api_key=settings.gemini_api_key)

    images = list_images(brand_dir)
    if not images:
        print("[info] no JPG/PNG files found")
        return 0

    total = len(images)
    ok = 0
    skipped = 0
    failed = 0

    for idx, path in enumerate(images, start=1):
        filename = path.name
        print(f"\n[{idx}/{total}] {filename}")
        try:
            file_hash = sha256_file(path)
            pid = point_id_for(brand, file_hash)

            if already_uploaded(qdrant, brand, pid):
                print("[skip] already uploaded (duplicate detected by hash)")
                skipped += 1
                continue

            print("[step] encoding image")
            image = load_image_for_gemini(path, max_side=args.max_side)

            print("[step] Gemini Vision description")
            description = describe_with_gemini(image)

            print("[step] Gemini embedding")
            embedding = embed_text(description)

            print("[step] Qdrant upsert")
            payload = {
                "brand": brand,
                "filename": filename,
                "path": str(path),
                "sha256": file_hash,
                "description": description,
                "embedding_model": EMBEDDING_MODEL,
                "vision_model": VISION_MODEL,
                "ingested_at_unix": int(time.time()),
            }
            upsert_point(qdrant, brand, pid, embedding, payload)

            print("[ok] uploaded")
            ok += 1

            if args.sleep > 0:
                time.sleep(args.sleep)
        except Exception as e:
            failed += 1
            print(f"[error] {e}")

    print("\n[done]")
    print(f"- uploaded: {ok}")
    print(f"- skipped:  {skipped}")
    print(f"- failed:   {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
