"""
Orchestrates the full topic-bottleneck pipeline end-to-end.
Each step's output is captured in PipelineResult for evaluation.
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.data.cuad_loader import CUADSample
from src.pipeline.base import (
    PipelineResult, TopicExtractor, S0TextGenerator, VocabExcluder,
    ProxyQuestionGenerator, QuestionSelector, CloudAnswerer, Synthesizer,
)


class PipelineRunner:
    def __init__(
        self,
        topic_extractor: TopicExtractor,
        text_generator: S0TextGenerator,
        question_generator: ProxyQuestionGenerator,
        question_selector: QuestionSelector,
        cloud_answerer: CloudAnswerer,
        synthesizer: Synthesizer,
        vocab_excluder: VocabExcluder | None = None,
        num_proxy_questions: int = 10,
        num_selected_questions: int = 3,
    ):
        self._topic_extractor = topic_extractor
        self._text_generator = text_generator
        self._vocab_excluder = vocab_excluder
        self._question_generator = question_generator
        self._question_selector = question_selector
        self._cloud_answerer = cloud_answerer
        self._synthesizer = synthesizer
        self._num_proxy = num_proxy_questions
        self._num_selected = num_selected_questions

    def run(self, sample: CUADSample) -> PipelineResult:
        result = PipelineResult(sample_id=sample.id)
        timings: dict[str, float] = {}

        try:
            # Step 1: Topic extraction
            t0 = time.monotonic()
            result.topics = self._topic_extractor.extract(sample.question, sample.context)
            timings["topic_extraction"] = time.monotonic() - t0

            # Step 2: S0 text generation
            t0 = time.monotonic()
            result.s0_text = self._text_generator.generate(result.topics)
            timings["text_generation"] = time.monotonic() - t0

            # Step 2.5: Vocab exclusion (optional)
            working_text = result.s0_text
            if self._vocab_excluder is not None:
                t0 = time.monotonic()
                result.s0_text_excluded = self._vocab_excluder.exclude(
                    result.s0_text, result.topics
                )
                timings["vocab_exclusion"] = time.monotonic() - t0
                working_text = result.s0_text_excluded

            # Step 3: Proxy question generation
            t0 = time.monotonic()
            result.generated_questions = self._question_generator.generate(
                working_text, n=self._num_proxy
            )
            timings["question_generation"] = time.monotonic() - t0

            # Step 4: Question selection (original question stays local)
            t0 = time.monotonic()
            result.selected_questions = self._question_selector.select(
                sample.question,
                result.generated_questions,
                k=self._num_selected,
            )
            timings["question_selection"] = time.monotonic() - t0

            # Step 5: Cloud answering (S0 questions only sent to cloud)
            t0 = time.monotonic()
            result.s0_answers = self._cloud_answerer.answer(result.selected_questions)
            timings["cloud_answering"] = time.monotonic() - t0

            # Step 6: Local synthesis (original S2 question + S0 answers)
            t0 = time.monotonic()
            result.final_answer = self._synthesizer.synthesize(
                sample.question,
                sample.context,
                result.s0_answers,
            )
            timings["synthesis"] = time.monotonic() - t0

        except Exception as e:
            result.error = str(e)

        result.step_timings = timings
        return result

    def run_batch(
        self,
        samples: list[CUADSample],
        verbose: bool = True,
        checkpoint_dir: Path | None = None,
    ) -> list[PipelineResult]:
        results: list[PipelineResult] = []
        for i, sample in enumerate(samples):
            if verbose:
                print(f"[{i+1}/{len(samples)}] {sample.category} — {sample.id}")
            results.append(self.run(sample))

            # Periodic checkpoint: save partial results every 10 samples
            if checkpoint_dir is not None and (i + 1) % 10 == 0:
                self._save_checkpoint(results, checkpoint_dir)

        # Final checkpoint
        if checkpoint_dir is not None and results:
            self._save_checkpoint(results, checkpoint_dir)

        return results

    @staticmethod
    def _save_checkpoint(results: list[PipelineResult], checkpoint_dir: Path) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / "checkpoint.json"
        data = [
            {"sample_id": r.sample_id, "error": r.error, "has_answer": bool(r.final_answer)}
            for r in results
        ]
        with open(path, "w") as f:
            json.dump(data, f)
        logging.getLogger(__name__).debug("Checkpoint saved: %d results to %s", len(results), path)
