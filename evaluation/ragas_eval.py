from typing import Optional
from datasets import Dataset

try:
    # ragas >= 0.2.x
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    faithfulness      = Faithfulness()
    answer_relevancy  = AnswerRelevancy()
    context_precision = ContextPrecision()
    context_recall    = ContextRecall()
except ImportError:
    # ragas 0.1.x fallback
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """
    Evaluates RAG pipeline quality using RAGAS metrics:
      - Faithfulness: Is the answer grounded in the context?
      - Answer Relevancy: Does the answer address the question?
      - Context Precision: Are retrieved chunks relevant?
      - Context Recall: Does context cover the ground truth?
    """

    def __init__(self, metrics: list = None):
        self.metrics = metrics or [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: Optional[list[str]] = None,
    ) -> dict:
        """
        Run RAGAS evaluation.

        Args:
            questions:     List of user questions.
            answers:       List of generated answers (one per question).
            contexts:      List of context lists (retrieved chunks per question).
            ground_truths: Optional list of reference answers for context_recall.

        Returns:
            dict with metric scores.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)
        logger.info(f"Running RAGAS evaluation on {len(questions)} samples")

        results = evaluate(dataset, metrics=self.metrics)
        scores = results.to_pandas().mean().to_dict()
        logger.info(f"RAGAS scores: {scores}")
        return scores

    def evaluate_pipeline(
        self,
        pipeline,
        test_cases: list[dict],
    ) -> dict:
        """
        Run the full RAG pipeline on test_cases and evaluate.

        Each test_case should have:
            - question: str
            - ground_truth: str (optional)
        """
        questions, answers, contexts, ground_truths = [], [], [], []

        for case in test_cases[: settings.RAGAS_SAMPLE_SIZE]:
            q = case["question"]
            result = pipeline.query(q)

            questions.append(q)
            answers.append(result["answer"])
            contexts.append([c["content"] for c in result["chunks"]])
            if "ground_truth" in case:
                ground_truths.append(case["ground_truth"])

        return self.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths or None,
        )

    def evaluate_from_file(self, pipeline, json_path: str) -> dict:
        """Load test cases from a JSON file and evaluate."""
        import json

        with open(json_path) as f:
            test_cases = json.load(f)

        logger.info(f"Loaded {len(test_cases)} test cases from {json_path}")
        return self.evaluate_pipeline(pipeline, test_cases)
