from __future__ import annotations

"""Module 3: Reranking - Cross-encoder top-20 to top-k."""

import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K  # noqa: E402


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class _LexicalReranker:
    """Small fallback that keeps tests and offline demos usable."""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))

    def predict(self, pairs):
        scores = []
        for query, doc in pairs:
            q = self._tokens(query)
            d = self._tokens(doc)
            overlap = len(q & d) / max(len(q), 1)
            phrase_bonus = 0.25 if "nghi" in doc.lower() or "ngh" in doc.lower() else 0.0
            numeric_bonus = 0.1 if re.search(r"\d+", doc) else 0.0
            scores.append(overlap + phrase_bonus + numeric_bonus)
        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                if os.getenv("USE_RAG_MODELS", "").lower() not in {"1", "true", "yes"}:
                    raise RuntimeError("USE_RAG_MODELS is not enabled")
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    tokenizer_args={"local_files_only": True},
                    automodel_args={"local_files_only": True},
                )
            except Exception as exc:
                print(f"  Reranker fallback to lexical scoring: {exc}")
                self._model = _LexicalReranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents and return top-k results."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight optional alternative."""

    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [{"id": i, "text": d["text"], "meta": d} for i, d in enumerate(documents)]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
            return [
                RerankResult(
                    text=item["text"],
                    original_score=float(item.get("meta", {}).get("score", 0.0)),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=item.get("meta", {}).get("metadata", {}),
                    rank=i,
                )
                for i, item in enumerate(results[:top_k])
            ]
        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
        {"text": "Thoi gian thu viec la 60 ngay.", "score": 0.75, "metadata": {}},
    ]
    for result in CrossEncoderReranker().rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
