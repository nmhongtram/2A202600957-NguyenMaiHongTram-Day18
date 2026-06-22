from __future__ import annotations

"""Module 5: Enrichment Pipeline."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY  # noqa: E402


@dataclass
class EnrichedChunk:
    """Enriched chunk."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _use_openai() -> bool:
    return bool(OPENAI_API_KEY) and os.getenv("USE_OPENAI_API", "").lower() in {"1", "true", "yes"}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _openai_json_or_text(messages: list[dict], max_tokens: int = 200, json_mode: bool = False):
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=15)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


def summarize_chunk(text: str) -> str:
    """Create a short summary for a chunk."""
    if _use_openai():
        try:
            return _openai_json_or_text(
                [
                    {
                        "role": "system",
                        "content": "Tom tat doan van sau trong 2-3 cau ngan gon bang tieng Viet.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
        except Exception as exc:
            print(f"  OpenAI summarize fallback: {exc}")

    sentences = _sentences(text)
    return ". ".join(sentences[:2]).strip() + ("." if sentences else "")


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions that this chunk can answer."""
    if _use_openai():
        try:
            content = _openai_json_or_text(
                [
                    {
                        "role": "system",
                        "content": (
                            f"Du tren doan van, tao {n_questions} cau hoi ma doan van co the tra loi. "
                            "Tra ve moi cau hoi tren mot dong."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = [q.strip().lstrip("0123456789.-) ") for q in content.splitlines()]
            return [q for q in questions if q][:n_questions]
        except Exception as exc:
            print(f"  OpenAI HyQA fallback: {exc}")

    questions = []
    for sentence in _sentences(text)[:n_questions]:
        clean = sentence.rstrip(".!?")
        if re.search(r"\d+", clean):
            questions.append(f"Quy dinh lien quan den {clean[:60]} la gi?")
        else:
            questions.append(f"{clean}?")
    return questions


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend one context sentence while preserving the original text."""
    if _use_openai():
        try:
            context = _openai_json_or_text(
                [
                    {
                        "role": "system",
                        "content": (
                            "Viet 1 cau ngan mo ta doan van nay nam o dau trong tai lieu va noi ve chu de gi. "
                            "Chi tra ve 1 cau."
                        ),
                    },
                    {"role": "user", "content": f"Tai lieu: {document_title}\n\nDoan van:\n{text}"},
                ],
                max_tokens=80,
            )
            return f"{context}\n\n{text}"
        except Exception as exc:
            print(f"  OpenAI contextual fallback: {exc}")

    prefix = f"Trich tu {document_title}. " if document_title else "Ngu canh tai lieu. "
    return f"{prefix}{text}"


def extract_metadata(text: str) -> dict:
    """Extract lightweight metadata for filtering and diagnostics."""
    if _use_openai():
        try:
            content = _openai_json_or_text(
                [
                    {
                        "role": "system",
                        "content": (
                            'Tra ve JSON metadata: {"topic": "...", "entities": ["..."], '
                            '"category": "policy|hr|it|finance|general", "language": "vi|en"}.'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                json_mode=True,
            )
            return json.loads(content)
        except Exception as exc:
            print(f"  OpenAI metadata fallback: {exc}")

    lowered = text.lower()
    if any(word in lowered for word in ["mat khau", "vpn", "bao mat", "password"]):
        category = "it"
    elif any(word in lowered for word in ["luong", "chi phi", "tam ung", "phu cap"]):
        category = "finance"
    elif any(word in lowered for word in ["nghi", "thu viec", "dao tao", "nhan vien"]):
        category = "hr"
    else:
        category = "policy"
    entities = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text)[:5]
    topic = (_sentences(text)[:1] or ["general"])[0][:80]
    return {"topic": topic, "entities": entities, "category": category, "language": "vi"}


def _fallback_enrichment(text: str, source: str) -> dict:
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trich tu {source} ve {extract_metadata(text).get('category', 'policy')}." if source else "Ngu canh tai lieu noi bo.",
        "metadata": extract_metadata(text),
    }


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata."""
    if _use_openai():
        try:
            content = _openai_json_or_text(
                [
                    {
                        "role": "system",
                        "content": """Phan tich doan van va tra ve JSON:
{
  "summary": "tom tat 2-3 cau",
  "questions": ["cau hoi 1", "cau hoi 2", "cau hoi 3"],
  "context": "1 cau mo ta doan van nam o dau trong tai lieu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|general", "language": "vi|en"}
}""",
                    },
                    {"role": "user", "content": f"Tai lieu: {source}\n\nDoan van:\n{text}"},
                ],
                max_tokens=400,
                json_mode=True,
            )
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            print(f"  Enrichment API fallback: {exc}")
    return _fallback_enrichment(text, source)


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Run enrichment pipeline over chunks."""
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods
    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**chunk.get("metadata", {}), **auto_meta},
                method="+".join(methods),
            )
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam."
    print(summarize_chunk(sample))
    print(generate_hypothesis_questions(sample))
    print(contextual_prepend(sample, "So tay nhan vien"))
    print(extract_metadata(sample))
