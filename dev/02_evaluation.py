"""
RAG Evaluation Runner

Usage:
    python dev/02_evaluation.py
    python dev/02_evaluation.py --top-k 10
    python dev/02_evaluation.py --provider anthropic
    python dev/02_evaluation.py --method by_title
"""
import sys
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import load_config
from rag.evaluation.metrics import (
    format_report,
    run_evaluation,
)


TEST_SET_PATH = (
    Path(__file__).parent.parent
    / "rag" / "evaluation" / "test_set.yaml"
)


@click.command()
@click.option("--test-set", default=str(TEST_SET_PATH),
              help="Path to test set YAML")
@click.option("--top-k", default=5, help="Top K for retrieval")
@click.option("--provider", default=None,
              help="Chat provider override")
@click.option("--method", default=None,
              help="Chunking method override")
@click.option("--rerank", default="cross-encoder",
              help="Rerank method")
def main(test_set, top_k, provider, method, rerank):
    """Run RAG evaluation on the test question set."""
    click.echo("Loading configuration...")
    config = load_config()

    if provider:
        config.active.chat_provider = provider
    if method:
        config.chunking.method = method

    click.echo(f"Loading test set from {test_set}...")
    with open(test_set) as f:
        data = yaml.safe_load(f)

    questions = data["questions"]
    click.echo(f"Loaded {len(questions)} test questions\n")

    click.echo(
        f"Running evaluation "
        f"(provider={config.active.chat_provider}, "
        f"top_k={top_k}, rerank={rerank})...\n"
    )

    report = run_evaluation(
        test_questions=questions,
        config=config,
        top_k=top_k,
        rerank_method=rerank,
    )

    click.echo(format_report(report))

    click.echo("\n\nPer-question breakdown:")
    click.echo(
        f"{'ID':<5} {'Category':<18} {'Prec':>6} "
        f"{'Rel':>6} {'Faith':>6} {'ms':>6} {'Srcs':>5}"
    )
    click.echo("-" * 55)
    for r in report.questions:
        click.echo(
            f"{r.question_id:<5} {r.category:<18} "
            f"{r.retrieval.precision_at_k:>6.2f} "
            f"{r.answer_quality.relevancy:>6.2f} "
            f"{r.answer_quality.faithfulness:>6.2f} "
            f"{r.latency.total_ms:>6} "
            f"{r.sources_count:>5}"
        )


if __name__ == "__main__":
    main()
