import time
from dataclasses import dataclass, field

from rag.config import Settings
from rag.retrieval.chain import query_rag


@dataclass
class RetrievalMetrics:
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    top_k: int = 5


@dataclass
class AnswerMetrics:
    faithfulness: float = 0.0
    relevancy: float = 0.0


@dataclass
class LatencyMetrics:
    total_ms: int = 0


@dataclass
class QuestionResult:
    question_id: str
    question: str
    answer: str
    retrieval: RetrievalMetrics = field(
        default_factory=RetrievalMetrics
    )
    answer_quality: AnswerMetrics = field(
        default_factory=AnswerMetrics
    )
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    sources_count: int = 0
    category: str = ""


@dataclass
class EvaluationReport:
    questions: list[QuestionResult] = field(default_factory=list)
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_faithfulness: float = 0.0
    avg_relevancy: float = 0.0
    avg_latency_ms: float = 0.0
    provider: str = ""
    chunking_method: str = ""
    top_k: int = 5


def evaluate_retrieval_precision(
    response_sources: list[dict],
    expected_keywords: list[str],
) -> float:
    if not response_sources:
        return 0.0

    relevant_count = 0
    for source in response_sources:
        excerpt = source.get("excerpt", "").lower()
        if any(kw.lower() in excerpt for kw in expected_keywords):
            relevant_count += 1

    return relevant_count / len(response_sources)


def evaluate_retrieval_recall(
    response_sources: list[dict],
    expected_keywords: list[str],
) -> float:
    if not expected_keywords:
        return 0.0
    if not response_sources:
        return 0.0

    all_source_text = " ".join(
        s.get("excerpt", "") for s in response_sources
    ).lower()

    found = sum(
        1 for kw in expected_keywords
        if kw.lower() in all_source_text
    )
    return found / len(expected_keywords)


def evaluate_answer_relevancy(
    answer: str,
    expected_keywords: list[str],
) -> float:
    if not answer:
        return 0.0

    answer_lower = answer.lower()
    matched = sum(
        1 for kw in expected_keywords
        if kw.lower() in answer_lower
    )
    return matched / len(expected_keywords) if expected_keywords else 0.0


def evaluate_answer_faithfulness(
    answer: str,
    sources: list[dict],
) -> float:
    if not answer or not sources:
        return 0.0

    all_source_words = set(
        " ".join(
            s.get("excerpt", "") for s in sources
        ).lower().split()
    )

    answer_sentences = [
        s.strip() for s in answer.replace("$", " ").split(".")
        if s.strip() and len(s.strip()) > 10
    ]

    if not answer_sentences:
        return 0.0

    grounded = 0
    for sentence in answer_sentences:
        words = sentence.lower().split()
        key_words = [w for w in words if len(w) > 2]
        if not key_words:
            grounded += 1
            continue
        overlap = sum(
            1 for w in key_words if w in all_source_words
        )
        if overlap / len(key_words) > 0.3:
            grounded += 1

    return grounded / len(answer_sentences)


def run_evaluation(
    test_questions: list[dict],
    config: Settings,
    *,
    top_k: int = 5,
    rerank_method: str = "cross-encoder",
) -> EvaluationReport:
    results: list[QuestionResult] = []

    for q in test_questions:
        start = time.time()

        try:
            response = query_rag(
                question=q["question"],
                config=config,
                top_k=top_k,
                rerank_method=rerank_method,
            )
        except Exception:
            result = QuestionResult(
                question_id=q.get("id", ""),
                question=q["question"],
                answer="[ERROR: query failed]",
                category=q.get("category", ""),
            )
            results.append(result)
            continue

        latency_ms = int((time.time() - start) * 1000)

        sources_dicts = [
            {
                "file": s.file,
                "page": s.page,
                "section": s.section,
                "excerpt": s.excerpt,
            }
            for s in response.sources
        ]

        expected_keywords = q.get("expected_keywords", [])

        precision = evaluate_retrieval_precision(
            sources_dicts, expected_keywords
        )
        relevancy = evaluate_answer_relevancy(
            response.answer, expected_keywords
        )
        faithfulness = evaluate_answer_faithfulness(
            response.answer, sources_dicts
        )

        # Recall: fraction of expected keywords found in sources
        recall = evaluate_retrieval_recall(
            sources_dicts, expected_keywords
        )

        result = QuestionResult(
            question_id=q.get("id", ""),
            question=q["question"],
            answer=response.answer,
            retrieval=RetrievalMetrics(
                precision_at_k=precision,
                recall_at_k=recall,
                top_k=top_k,
            ),
            answer_quality=AnswerMetrics(
                faithfulness=faithfulness,
                relevancy=relevancy,
            ),
            latency=LatencyMetrics(total_ms=latency_ms),
            sources_count=len(response.sources),
            category=q.get("category", ""),
        )
        results.append(result)

    report = EvaluationReport(
        questions=results,
        provider=config.active.chat_provider,
        chunking_method=config.chunking.method,
        top_k=top_k,
    )

    if results:
        report.avg_precision = sum(
            r.retrieval.precision_at_k for r in results
        ) / len(results)
        report.avg_recall = sum(
            r.retrieval.recall_at_k for r in results
        ) / len(results)
        report.avg_relevancy = sum(
            r.answer_quality.relevancy for r in results
        ) / len(results)
        report.avg_faithfulness = sum(
            r.answer_quality.faithfulness for r in results
        ) / len(results)
        report.avg_latency_ms = sum(
            r.latency.total_ms for r in results
        ) / len(results)

    return report


def format_report(report: EvaluationReport) -> str:
    lines = [
        f"{'Metric':<30} {'Score':>10}",
        "-" * 42,
        f"{'Retrieval Precision@' + str(report.top_k):<30}"
        f" {report.avg_precision:>9.2f}",
        f"{'Retrieval Recall@' + str(report.top_k):<30}"
        f" {report.avg_recall:>9.2f}",
        f"{'Answer Relevancy':<30}"
        f" {report.avg_relevancy:>9.2f}",
        f"{'Answer Faithfulness':<30}"
        f" {report.avg_faithfulness:>9.2f}",
        f"{'Avg Latency (ms)':<30}"
        f" {report.avg_latency_ms:>9.0f}",
        "-" * 42,
        f"Provider: {report.provider}",
        f"Chunking: {report.chunking_method}",
        f"Questions: {len(report.questions)}",
    ]
    return "\n".join(lines)
