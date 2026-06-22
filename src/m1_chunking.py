from __future__ import annotations

"""Module 1: Advanced Chunking Strategies."""

import glob
import os
import re
import sys
from dataclasses import dataclass, field
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer from a PDF; scanned PDFs return an empty string."""
    try:
        from pypdf import PdfReader
    except Exception:
        print(f"  Skipping {os.path.basename(path)}: pypdf is not installed.")
        return ""

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown files and text-layer PDFs from data/."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  Skipping {os.path.basename(fp)}: PDF has no text layer.")

    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Baseline paragraph chunking."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Group adjacent sentences when their similarity is above threshold."""
    metadata = metadata or {}
    sentences = _sentences(text)
    if not sentences:
        return []

    def token_vector(sentence: str) -> dict[str, float]:
        vec: dict[str, float] = {}
        for token in re.findall(r"\w+", sentence.lower(), flags=re.UNICODE):
            vec[token] = vec.get(token, 0.0) + 1.0
        return vec

    def cosine(a, b) -> float:
        try:
            from numpy import dot
            from numpy.linalg import norm

            return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))
        except Exception:
            keys = set(a) | set(b)
            numerator = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
            denom_a = sqrt(sum(v * v for v in a.values()))
            denom_b = sqrt(sum(v * v for v in b.values()))
            return numerator / (denom_a * denom_b + 1e-9)

    try:
        if os.getenv("USE_RAG_MODELS", "").lower() not in {"1", "true", "yes"}:
            raise RuntimeError("USE_RAG_MODELS is not enabled")
        from sentence_transformers import SentenceTransformer

        embeddings = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True).encode(sentences)
    except Exception:
        embeddings = [token_vector(sentence) for sentence in sentences]

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if cosine(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [
        Chunk(" ".join(group).strip(), {**metadata, "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)
        if group
    ]


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent chunks for context and child chunks for retrieval."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()] or ([text.strip()] if text.strip() else [])
    parents: list[Chunk] = []
    current = ""

    def add_parent(parent_text: str) -> None:
        pid = f"parent_{len(parents)}"
        parents.append(
            Chunk(
                parent_text.strip(),
                {**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)},
            )
        )

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > parent_size:
            add_parent(current)
            current = ""
        if len(para) > parent_size:
            for start in range(0, len(para), parent_size):
                add_parent(para[start:start + parent_size])
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    if current:
        add_parent(current)

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        units = _sentences(parent.text) or [parent.text]
        current_child = ""
        child_index = 0

        def add_child(child_text: str) -> None:
            nonlocal child_index
            children.append(
                Chunk(
                    child_text.strip(),
                    {**metadata, "chunk_type": "child", "parent_id": pid, "chunk_index": child_index},
                    parent_id=pid,
                )
            )
            child_index += 1

        for unit in units:
            if current_child and len(current_child) + len(unit) + 1 > child_size:
                add_child(current_child)
                current_child = ""
            if len(unit) > child_size:
                for start in range(0, len(unit), child_size):
                    add_child(unit[start:start + child_size])
            else:
                current_child = f"{current_child} {unit}".strip() if current_child else unit
        if current_child:
            add_child(current_child)

    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Split markdown by H1-H3 sections while preserving section headers."""
    metadata = metadata or {}
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    current_header = ""
    current_content: list[str] = []

    def flush() -> None:
        content = "\n".join(part.strip() for part in current_content if part.strip()).strip()
        if not current_header and not content:
            return
        chunk_text = f"{current_header}\n\n{content}".strip() if current_header else content
        section = re.sub(r"^#{1,3}\s+", "", current_header).strip() or "root"
        chunks.append(
            Chunk(
                chunk_text,
                {**metadata, "section": section, "strategy": "structure", "chunk_index": len(chunks)},
            )
        )

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if re.match(r"^#{1,3}\s+.+$", stripped):
            flush()
            current_header = stripped
            current_content = []
        else:
            current_content.append(part)
    flush()
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    """Run all strategies on documents and compare basic statistics."""
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
