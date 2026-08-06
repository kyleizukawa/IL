"""
Base class for long-horizon agentic coding tasks.

Each task is a hand-crafted scenario with:
- A real multi-file codebase (100-400 lines)
- A task that requires sustained multi-step reasoning
- An efficiency-aware grader that measures reasoning quality
- A teacher reasoning trace that demonstrates thorough analysis

Key design principles (mechanize.work style):
1. Hand-crafted: each task is unique, not procedurally generated
2. Long-horizon: reasoning must be sustained over 500-2000 tokens
3. Efficiency-aware: reward = correctness × (0.6 + 0.4 × reasoning_quality)
4. Failure-mode targeted: each task exposes a specific reasoning weakness
5. Rich reward signal: partial credit + reasoning quality + verification bonus

Each task defines:
- task_id: unique identifier
- reasoning_skill: what reasoning capability this tests
- failure_mode: what specific weakness this exposes in small models
- token_budget: target reasoning length (shapes efficiency scoring)
- expected_concepts: concepts the reasoning should cover (shapes coverage scoring)
"""
import random
import textwrap
from abc import abstractmethod
from typing import Any

from ..graders import (
    extract_answer as grader_extract_answer,
    extract_reasoning as grader_extract_reasoning,
    parse_code_blocks, apply_code_changes, run_tests, run_code,
    compute_test_score, CodeExecutor, code_similarity,
)
from .efficiency import (
    score_reasoning_quality, compute_final_score, shape_rl_reward,
    ReasoningBreakdown,
)


# ── Registry ──
class _TaskRegistry:
    """Registry for long-horizon tasks."""
    def __init__(self):
        self._registry: dict[str, type] = {}

    def register(self, cls):
        self._registry[cls.task_id] = cls
        return cls

    def values(self):
        return self._registry.values()

    def keys(self):
        return self._registry.keys()

    def __getitem__(self, key):
        return self._registry[key]

    def __contains__(self, key):
        return key in self._registry

    def __len__(self):
        return len(self._registry)

    def __call__(self, cls):
        """Allow use as a decorator: @register_long_horizon."""
        return self.register(cls)


register_long_horizon = _TaskRegistry()


class LongHorizonEnv:
    """Base class for hand-crafted long-horizon agentic coding tasks.

    Unlike the procedural AgenticEnv, each subclass is a single unique task.
    There is no procedural generation — the codebase, task, solution, and
    reasoning are all hand-crafted.

    However, tasks can have optional parameterization (e.g., different input
    data) to generate multiple episodes for RL training.

    Subclasses must define:
        task_id: unique identifier (e.g., 'cascading_bug_chain')
        reasoning_skill: what reasoning capability this tests
        failure_mode: what weakness this exposes in small models
        token_budget: target reasoning token count
        expected_concepts: concepts the reasoning should cover

    And implement:
        gen_codebase(): return {filename: content}
        gen_task(codebase): return task description string
        gen_solution(codebase): return {filename: content} (correct solution)
        gen_reasoning(codebase, solution): return teacher reasoning trace
        grade_correctness(codebase, response): return (score, breakdown)
    """

    task_id: str = ""
    reasoning_skill: str = ""
    failure_mode: str = ""
    token_budget: int = 600
    expected_concepts: list[str] = []

    def __init__(self):
        """Initialize with optional parameters for episode variation."""
        pass

    @abstractmethod
    def gen_codebase(self) -> dict[str, str]:
        """Generate the codebase as {filename: content}.

        This is hand-crafted but can have minor variation (e.g., different
        input data) for multiple episodes.
        """

    @abstractmethod
    def gen_task(self, codebase: dict[str, str]) -> str:
        """Generate the task description shown to the model."""

    @abstractmethod
    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        """Generate the correct solution as {filename: content}."""

    @abstractmethod
    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        """Generate teacher reasoning trace for SFT.

        CRITICAL: This must demonstrate thorough, multi-step reasoning:
        - Read the relevant code line by line
        - Trace execution paths across modules
        - Identify the specific issue with evidence
        - Explain the fix with reasoning
        - VERIFY the fix by tracing through test cases
        - Do NOT jump to conclusions or pattern-match

        This is the core "kill laziness" mechanism.
        """

    @abstractmethod
    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        """Grade the correctness of the model's response.

        Returns (score, breakdown) where score is in [0, 1].
        This measures ONLY correctness — reasoning quality is scored separately.
        """

    # ── Full grading with efficiency-aware scoring ──

    def grade(self, codebase: dict[str, str],
              response: str) -> tuple[float, dict]:
        """Full efficiency-aware grading.

        Returns (final_score, breakdown) where:
            final_score = correctness * (0.6 + 0.4 * reasoning_quality)

        The breakdown includes both correctness and reasoning quality details.
        """
        # Grade correctness
        correctness, correct_breakdown = self.grade_correctness(codebase, response)

        # Score reasoning quality
        reasoning_quality, reasoning_breakdown = score_reasoning_quality(
            response,
            expected_concepts=self.expected_concepts,
            token_budget=self.token_budget,
            correctness=correctness,
        )

        # Compute final score
        final_score = compute_final_score(correctness, reasoning_quality)

        # Combine breakdowns
        breakdown = {
            "task_id": self.task_id,
            "correctness": correctness,
            "reasoning_quality": reasoning_quality,
            "final_score": final_score,
            "correctness_details": correct_breakdown,
            "reasoning_details": {
                "coverage": reasoning_breakdown.coverage,
                "concepts_found": reasoning_breakdown.concepts_found,
                "concepts_missing": reasoning_breakdown.concepts_missing,
                "token_efficiency": reasoning_breakdown.token_efficiency,
                "tokens_used": reasoning_breakdown.tokens_used,
                "token_budget": reasoning_breakdown.token_budget,
                "verification": reasoning_breakdown.verification,
                "verification_evidence": reasoning_breakdown.verification_evidence,
                "filler_penalty": reasoning_breakdown.filler_penalty,
                "filler_found": reasoning_breakdown.filler_found,
                "reasoning_length": reasoning_breakdown.reasoning_length,
            },
        }

        return final_score, breakdown

    # ── Convenience methods ──

    def generate_instance(self) -> dict:
        """Generate a complete task instance.

        Returns: {
            task_id, reasoning_skill, failure_mode,
            codebase, task, solution, reasoning,
            token_budget, expected_concepts,
        }
        """
        codebase = self.gen_codebase()
        task = self.gen_task(codebase)
        solution = self.gen_solution(codebase)
        reasoning = self.gen_reasoning(codebase, solution)
        return {
            "task_id": self.task_id,
            "reasoning_skill": self.reasoning_skill,
            "failure_mode": self.failure_mode,
            "codebase": codebase,
            "task": task,
            "solution": solution,
            "reasoning": reasoning,
            "token_budget": self.token_budget,
            "expected_concepts": self.expected_concepts,
        }

    def build_sft_example(self) -> dict:
        """Generate a single SFT training example in chat format."""
        inst = self.generate_instance()
        assistant_msg = (
            f"<reasoning>\n{inst['reasoning']}\n</reasoning>\n"
            f"<answer>\n{self._format_solution(inst['solution'])}\n</answer>"
        )
        return {
            "messages": [
                {"role": "user", "content": inst["task"]},
                {"role": "assistant", "content": assistant_msg},
            ],
            "metadata": {
                "task_id": self.task_id,
                "reasoning_skill": self.reasoning_skill,
                "failure_mode": self.failure_mode,
                "token_budget": self.token_budget,
            },
        }

    def _format_solution(self, solution: dict[str, str]) -> str:
        """Format the solution dict as the answer section."""
        parts = []
        for filename, content in solution.items():
            parts.append(f"```python:{filename}\n{content}\n```")
        return "\n\n".join(parts)

    def build_perfect_response(self) -> str:
        """Build a perfect response (for testing the grader)."""
        inst = self.generate_instance()
        reasoning = inst["reasoning"]
        answer = self._format_solution(inst["solution"])
        return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>"
