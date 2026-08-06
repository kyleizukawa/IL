"""
Base interface for agentic coding RL environments.

Each environment is a self-contained software engineering task that:
1. Procedurally generates a mini-codebase with a specific problem
2. Provides a task description for the model
3. Has a deterministic grader with partial credit (informative reward)
4. Generates teacher reasoning traces for SFT

Design principles (mechanize.work style):
- Quality over quantity: each task provides rich, informative reward signals
- Real codebases: multi-file Python projects with realistic structure
- Distractor code: irrelevant code the model must skip (punishes laziness)
- Partial credit: rewards careful partial work, not just all-or-nothing
- Edge cases: tests that punish superficial pattern-matching
- Teacher traces: demonstrate thorough, line-by-line analysis (not jumping to conclusions)
"""
import random
from abc import ABC, abstractmethod
from typing import Any

# ── Registry ──
ENV_REGISTRY: dict[str, type] = {}


def register_env(cls):
    """Decorator to register an environment class."""
    ENV_REGISTRY[cls.name] = cls
    return cls


class AgenticEnv(ABC):
    """Base class for agentic coding RL environments.

    Subclasses must define:
        name: short identifier (e.g. 'bug_localization')
        skill: what agentic skill this teaches
        difficulty_tiers: list of difficulty levels

    And implement all abstract methods.
    """

    name: str = ""
    skill: str = ""
    difficulty_tiers: list[str] = ["easy", "medium", "hard"]

    @abstractmethod
    def gen_params(self, rng: random.Random, difficulty: str = "medium") -> dict:
        """Generate parameters for one task instance."""

    @abstractmethod
    def gen_codebase(self, params: dict, rng: random.Random) -> dict[str, str]:
        """Generate the codebase as {filename: content}.

        The codebase should be a realistic mini-project with:
        - The target code (with the bug/missing feature/etc.)
        - Supporting modules (imports, helpers)
        - Distractor code (irrelevant functions the model must skip)
        - Existing tests (where applicable)
        """

    @abstractmethod
    def gen_task(self, params: dict, codebase: dict[str, str]) -> str:
        """Generate the task description shown to the model.

        Should include:
        - What the model needs to do
        - Relevant context (failing test output, spec, stack trace, etc.)
        - The codebase files (or pointers to them)
        """

    @abstractmethod
    def gen_solution(self, params: dict, codebase: dict[str, str]) -> dict[str, str]:
        """Generate the correct solution as {filename: content}.

        Only includes files that need to be changed/created.
        """

    @abstractmethod
    def gen_reasoning(self, params: dict, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        """Generate teacher reasoning trace for SFT.

        This is CRITICAL for killing laziness. The teacher must:
        - Read the relevant code line by line
        - Trace execution paths
        - Identify the specific issue with evidence
        - Explain the fix with reasoning
        - NOT jump to conclusions or pattern-match

        Format: free text that goes inside <reasoning> tags.
        """

    @abstractmethod
    def grade(self, params: dict, codebase: dict[str, str],
              response: str) -> tuple[float, dict]:
        """Grade the model's response.

        Args:
            params: task parameters
            codebase: the original codebase
            response: the model's full response text

        Returns:
            (score, breakdown) where score is in [0, 1] and
            breakdown is a dict with detailed scoring info.
        """

    # ── Convenience methods ──

    def generate_instance(self, rng: random.Random, difficulty: str = "medium") -> dict:
        """Generate a complete task instance.

        Returns a dict with: params, codebase, task, solution, reasoning.
        """
        params = self.gen_params(rng, difficulty)
        codebase = self.gen_codebase(params, rng)
        task = self.gen_task(params, codebase)
        solution = self.gen_solution(params, codebase)
        reasoning = self.gen_reasoning(params, codebase, solution)
        return {
            "env_name": self.name,
            "skill": self.skill,
            "difficulty": difficulty,
            "params": params,
            "codebase": codebase,
            "task": task,
            "solution": solution,
            "reasoning": reasoning,
        }

    def build_sft_example(self, rng: random.Random, difficulty: str = "medium") -> dict:
        """Generate a single SFT training example in chat format.

        Returns: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
        """
        inst = self.generate_instance(rng, difficulty)
        user_msg = inst["task"]
        assistant_msg = f"<reasoning>\n{inst['reasoning']}\n</reasoning>\n<answer>\n{self._format_solution_answer(inst['solution'])}\n</answer>"
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "metadata": {
                "env_name": self.name,
                "skill": self.skill,
                "difficulty": difficulty,
            },
        }

    def _format_solution_answer(self, solution: dict[str, str]) -> str:
        """Format the solution dict as the answer section."""
        parts = []
        for filename, content in solution.items():
            parts.append(f"```python:{filename}\n{content}\n```")
        return "\n\n".join(parts)
