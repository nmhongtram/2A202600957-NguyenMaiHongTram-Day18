from __future__ import annotations

"""Module 2: Hybrid Search - BM25 (Vietnamese) + Dense + RRF."""

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words, with a regex fallback."""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(chunk["text"]).split() for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25 or a simple token-overlap fallback."""
        if not self.documents:
            return []

        query_tokens = segment_vietnamese(query).split()
        if self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
        else:
            qset = set(query_tokens)
            scores = [len(qset & set(tokens)) / max(len(qset), 1) for tokens in self.corpus_tokens]

        top_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        results = []
        for i in top_indices:
            score = float(scores[i])
            if score <= 0:
                continue
            doc = self.documents[i]
            results.append(SearchResult(doc["text"], score, doc.get("metadata", {}), "bm25"))
        return results


class DenseSearch:
    def __init__(self):
        self._encoder = None
        self._memory_points: list[tuple[list[float], dict]] = []
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        except Exception:
            self.client = None

    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("USE_RAG_MODELS", "").lower() not in {"1", "true", "yes"}:
                raise RuntimeError("USE_RAG_MODELS is not enabled")
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        return self._encoder

    def _embed(self, texts):
        try:
            vectors = self._get_encoder().encode(texts, show_progress_bar=False)
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
        except Exception:
            if isinstance(texts, str):
                texts = [texts]
            return [self._hash_embed(text) for text in texts]

    @staticmethod
    def _hash_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
        vec = [0.0] * dim
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
            vec[hash(token) % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant, falling back to in-memory vectors."""
        texts = [c["text"] for c in chunks]
        vectors = self._embed(texts)
        payloads = [{**c.get("metadata", {}), "text": c["text"]} for c in chunks]
        self._memory_points = list(zip(vectors, payloads))

        if self.client is None:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            points = [
                PointStruct(id=i, vector=vector, payload=payload)
                for i, (vector, payload) in enumerate(zip(vectors, payloads))
            ]
            self.client.upsert(collection, points)
        except Exception as exc:
            print(f"  Dense index fallback to memory: {exc}")
            self.client = None

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """Search using Qdrant dense vectors or in-memory cosine fallback."""
        query_vector = self._embed([query])[0]

        if self.client is not None:
            try:
                response = self.client.query_points(collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        pt.payload.get("text", ""),
                        float(pt.score),
                        dict(pt.payload or {}),
                        "dense",
                    )
                    for pt in response.points
                ]
            except Exception as exc:
                print(f"  Dense search fallback to memory: {exc}")
                self.client = None

        scored = [
            (self._cosine(query_vector, vector), payload)
            for vector, payload in self._memory_points
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(payload.get("text", ""), float(score), dict(payload), "dense")
            for score, payload in scored[:top_k]
            if score > 0
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge ranked lists using Reciprocal Rank Fusion."""
    rrf_scores: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)
    return [
        SearchResult(
            item["result"].text,
            float(item["score"]),
            item["result"].metadata,
            "hybrid",
        )
        for item in merged[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(segment_vietnamese("Nhan vien duoc nghi phep nam"))
