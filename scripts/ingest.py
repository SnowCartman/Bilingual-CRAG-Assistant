"""Build the Qdrant collection this app queries: PDFs in, hybrid index out.

    python scripts/ingest.py --source ./data/books
    python scripts/ingest.py --source ./data/books --recreate
    python scripts/ingest.py --source ./data/books --extractor marker

The collection it produces is exactly what ``app.services.rag_pipeline``
expects at query time:

    named dense vector  "dense"   1024-dim, cosine   (voyage-multilingual-2)
    named sparse vector "sparse"  BM25 with the IDF modifier
    payload             source_file, page, page_end, language, text

Extraction
----------
``--extractor pypdf`` (default) reads the text layer. It is fast, pure Python,
and returns nothing at all for scanned pages -- a book that comes out with zero
characters is a scan, not a failure of this script.

``--extractor marker`` runs marker-pdf, which OCRs scans and preserves markdown
structure (headings, tables). It pulls in torch and is the reason the corpus
this app was built on could include scanned dictionaries. On CPU it is slow
enough to be impractical for a whole shelf; with a CUDA GPU it is roughly an
order of magnitude faster. Install it separately -- it is deliberately NOT in
requirements-ingest.txt, because most people will not need it:

    pip install marker-pdf         # plus a CUDA build of torch for GPU use

Chunking
--------
Recursive character splitting at 2000 characters with 200 overlap, trying
markdown headings before paragraphs before sentences. The large window keeps a
grammar rule together with its examples and tables; see ARCHITECTURE.md for why
that beat the smaller windows measured against it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models as qm

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"
VECTOR_DIM = 1024
EMBED_MODEL = "voyage-multilingual-2"
SPARSE_MODEL = "Qdrant/bm25"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
# Largest semantic unit first: a heading boundary should always beat a
# paragraph boundary, and a paragraph should beat a mid-sentence cut.
SEPARATORS = ["\n#### ", "\n### ", "\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""]

# Voyage accepts up to 128 inputs per request; stay under it and under the
# per-request token ceiling for 2000-character chunks.
EMBED_BATCH = 64
UPSERT_BATCH = 128


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def extract_pypdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] from the PDF's text layer."""
    from pypdf import PdfReader

    pages: list[tuple[int, str]] = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 -- one bad page must not stop a book
            print(f"    page {i}: extraction failed ({exc})")
            text = ""
        if text.strip():
            pages.append((i, text))
    return pages


def extract_marker(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, markdown)] via marker-pdf (OCR + structure)."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError:  # pragma: no cover -- optional dependency
        sys.exit("marker-pdf is not installed. pip install marker-pdf, or use "
                 "--extractor pypdf.")

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(path))
    text, _, _ = text_from_rendered(rendered)
    # marker returns one markdown document; it marks page breaks with a rule.
    parts = text.split("\n---\n") if "\n---\n" in text else [text]
    return [(i, part) for i, part in enumerate(parts, start=1) if part.strip()]


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Cyrillic share decides. Good enough for a de/ru shelf; anything else
    that shows up is tagged by the same rule rather than guessed at."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "?"
    # Cyrillic block U+0400..U+04FF, written as code points so this file stays
    # ASCII like the rest of the codebase.
    cyrillic = sum(1 for c in letters if 0x0400 <= ord(c) <= 0x04FF)
    return "ru" if cyrillic / len(letters) > 0.3 else "de"


def chunk_pages(pages: list[tuple[int, str]], source_file: str) -> list[dict]:
    """Split a book into chunks that know which pages they came from.

    Pages are joined before splitting so a rule that runs across a page break
    stays in one chunk; the page range is recovered afterwards by walking the
    same offsets the join produced.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=SEPARATORS,
        keep_separator=True,
    )
    joined = ""
    spans: list[tuple[int, int, int]] = []      # (start, end, page)
    for page_no, text in pages:
        start = len(joined)
        joined += text + "\n\n"
        spans.append((start, len(joined), page_no))

    chunks: list[dict] = []
    cursor = 0
    for piece in splitter.split_text(joined):
        found = joined.find(piece, cursor)
        if found == -1:                          # splitter normalised whitespace
            found = joined.find(piece)
        if found == -1:
            start, end = cursor, cursor + len(piece)
        else:
            start, end = found, found + len(piece)
            cursor = max(cursor, found + 1)
        touched = [p for s, e, p in spans if s < end and e > start]
        if not piece.strip():
            continue
        chunks.append({
            "source_file": source_file,
            "page": touched[0] if touched else None,
            "page_end": touched[-1] if touched else None,
            "language": detect_language(piece),
            "text": piece,
        })
    return chunks


# --------------------------------------------------------------------------
# embedding + indexing
# --------------------------------------------------------------------------

def voyage_embed_documents(texts: list[str], api_key: str) -> list[list[float]]:
    """Embed with input_type='document' -- the query side uses 'query', and
    mixing the two costs measurable retrieval quality."""
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        for attempt in range(1, 6):
            resp = requests.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": batch, "model": EMBED_MODEL,
                      "input_type": "document"},
                timeout=120,
            )
            if resp.status_code == 200:
                out.extend(d["embedding"] for d in resp.json()["data"])
                break
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                backoff = min(30, 2 ** attempt)
                print(f"    voyage {resp.status_code}; retry in {backoff}s")
                time.sleep(backoff)
                continue
            resp.raise_for_status()
        print(f"    embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")
    return out


def ensure_collection(client: QdrantClient, name: str, recreate: bool) -> None:
    exists = client.collection_exists(name)
    if exists and recreate:
        print(f"dropping existing collection {name!r}")
        client.delete_collection(name)
        exists = False
    if exists:
        print(f"appending to existing collection {name!r}")
        return
    print(f"creating collection {name!r}")
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_NAME: qm.VectorParams(size=VECTOR_DIM,
                                        distance=qm.Distance.COSINE),
        },
        # IDF modifier: Qdrant applies inverse document frequency itself, which
        # is what makes the sparse side behave like real BM25 rather than raw
        # term counts.
        sparse_vectors_config={
            SPARSE_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF),
        },
    )


def index_chunks(client: QdrantClient, collection: str, chunks: list[dict],
                 voyage_key: str) -> None:
    print(f"embedding {len(chunks)} chunks")
    dense = voyage_embed_documents([c["text"] for c in chunks], voyage_key)
    print("building BM25 sparse vectors")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    sparse = list(sparse_model.embed([c["text"] for c in chunks]))

    points = [
        qm.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                DENSE_NAME: dense[i],
                SPARSE_NAME: qm.SparseVector(
                    indices=sparse[i].indices.tolist(),
                    values=sparse[i].values.tolist(),
                ),
            },
            payload=chunks[i],
        )
        for i in range(len(chunks))
    ]
    for i in range(0, len(points), UPSERT_BATCH):
        client.upsert(collection_name=collection,
                      points=points[i:i + UPSERT_BATCH], wait=True)
        print(f"    upserted {min(i + UPSERT_BATCH, len(points))}/{len(points)}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True,
                    help="directory containing the PDFs to index")
    ap.add_argument("--collection",
                    default=os.getenv("QDRANT_COLLECTION", "my_books_hybrid"),
                    help="Qdrant collection name (default: $QDRANT_COLLECTION)")
    ap.add_argument("--extractor", choices=("pypdf", "marker"), default="pypdf")
    ap.add_argument("--recreate", action="store_true",
                    help="drop the collection first instead of appending")
    args = ap.parse_args()

    voyage_key = os.getenv("VOYAGE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if not (voyage_key and qdrant_url):
        sys.exit("VOYAGE_API_KEY and QDRANT_URL must be set (see .env.example).")

    source = Path(args.source).expanduser().resolve()
    pdfs = sorted(source.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs found under {source}")
    print(f"{len(pdfs)} PDF(s) under {source}\n")

    extract = extract_pypdf if args.extractor == "pypdf" else extract_marker
    all_chunks: list[dict] = []
    for pdf in pdfs:
        print(f"{pdf.name}")
        pages = extract(pdf)
        if not pages:
            print("    no text layer -- likely a scan; retry with "
                  "--extractor marker")
            continue
        chunks = chunk_pages(pages, pdf.name)
        print(f"    {len(pages)} pages -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        sys.exit("nothing to index")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)
    ensure_collection(client, args.collection, args.recreate)
    index_chunks(client, args.collection, all_chunks, voyage_key)

    total = client.count(args.collection, exact=True).count
    print(f"\ndone -- collection {args.collection!r} now holds {total} chunks")
    print("set QDRANT_COLLECTION in your .env to this name, then "
          "docker compose up --build")


if __name__ == "__main__":
    main()
