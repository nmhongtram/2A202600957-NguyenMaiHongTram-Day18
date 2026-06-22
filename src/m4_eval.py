from __future__ import annotations

"""Module 4: RAGAS Evaluation - 4 metrics + failure analysis."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH  # noqa: E402


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _fallback_eval(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    per_question = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        context_text = " ".join(ctxs)
        answer_tokens = _tokens(answer)
        question_tokens = _tokens(question)
        context_tokens = _tokens(context_text)
        gt_tokens = _tokens(ground_truth)

        faithfulness = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
        answer_relevancy = len(answer_tokens & (question_tokens | gt_tokens)) / max(len(answer_tokens), 1)
        context_precision = len(context_tokens & (question_tokens | gt_tokens)) / max(len(context_tokens), 1)
        context_recall = len(gt_tokens & context_tokens) / max(len(gt_tokens), 1)

        per_question.append(
            EvalResult(
                question,
                answer,
                ctxs,
                ground_truth,
                round(faithfulness, 4),
                round(answer_relevancy, 4),
                round(context_precision, 4),
                round(context_recall, 4),
            )
        )

    def avg(metric: str) -> float:
        if not per_question:
            return 0.0
        return round(sum(getattr(item, metric) for item in per_question) / len(per_question), 4)

    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS evaluation, falling back to deterministic overlap metrics."""
    try:
        if os.getenv("USE_OPENAI_API", "").lower() not in {"1", "true", "yes"}:
            raise RuntimeError("USE_OPENAI_API is not enabled")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]

        def avg(metric: str) -> float:
            if metric in result:
                return float(result[metric])
            if not per_question:
                return 0.0
            return sum(getattr(item, metric) for item in per_question) / len(per_question)

        return {
            "faithfulness": avg("faithfulness"),
            "answer_relevancy": avg("answer_relevancy"),
            "context_precision": avg("context_precision"),
            "context_recall": avg("context_recall"),
            "per_question": per_question,
        }
    except Exception as exc:
        print(f"  RAGAS evaluation fallback: {exc}")
        return _fallback_eval(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using a simple diagnostic tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, cite context, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking, enrichment, or add BM25 coverage"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filters"),
        "answer_relevancy": ("Answer does not match the question", "Improve prompt template and answer format"),
    }

    analyzed = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append(
            {
                "question": result.question,
                "worst_metric": worst_metric,
                "score": round(avg_score, 4),
                "metric_score": round(metrics[worst_metric], 4),
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )

    analyzed.sort(key=lambda item: item["score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON."""
    per_question = results.get("per_question", [])
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(per_question),
        "per_question": [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in per_question],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
