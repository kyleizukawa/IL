"""
RL Rollout Interface for Long-Horizon Agentic Coding Tasks.

Adapts the 20 hand-crafted tasks to the GRPO RL training loop with
efficiency-aware reward shaping.

Key difference from the procedural rollout:
- Tasks are hand-crafted (not procedurally generated)
- Reward includes reasoning quality (not just correctness)
- final_reward = shape_rl_reward(correctness, reasoning_quality, ...)
- The reasoning quality component provides the RL signal that shapes
  HOW the model reasons, not just WHAT it produces

For GRPO, we collect group_size rollouts per task. Since tasks are
hand-crafted, we cycle through the 20 tasks (one per iteration) and
collect group_size responses per task.
"""
import sys
import os
import time
import random
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from il_agentic.long_horizon import ALL_TASKS, register_long_horizon
from il_agentic.long_horizon.efficiency import (
    score_reasoning_quality, compute_final_score, shape_rl_reward,
    extract_reasoning, extract_answer,
)


class LongHorizonRollout:
    """Represents a single rollout for a long-horizon task.

    Attributes:
        task_id: which task
        reasoning_skill: what reasoning capability
        task_prompt: the user message
        codebase: the codebase
        response: model's response
        correctness: grader correctness score [0, 1]
        reasoning_quality: efficiency-aware reasoning quality [0, 1]
        final_score: correctness * (0.6 + 0.4 * reasoning_quality)
        rl_reward: shaped reward for RL training
        breakdown: detailed scoring breakdown
    """

    def __init__(self, task_id, reasoning_skill, task_prompt, codebase,
                 expected_concepts, token_budget):
        self.task_id = task_id
        self.reasoning_skill = reasoning_skill
        self.task_prompt = task_prompt
        self.codebase = codebase
        self.expected_concepts = expected_concepts
        self.token_budget = token_budget
        self.response = ""
        self.correctness = 0.0
        self.reasoning_quality = 0.0
        self.final_score = 0.0
        self.rl_reward = 0.0
        self.breakdown = {}
        self.tokens = []
        self.action_positions = []
        self.old_logprobs = []
        self.gen_time = 0.0

    def grade(self, task_instance):
        """Grade this rollout using efficiency-aware scoring."""
        # Grade correctness
        self.correctness, correct_breakdown = task_instance.grade_correctness(
            self.codebase, self.response
        )

        # Score reasoning quality
        self.reasoning_quality, reasoning_breakdown = score_reasoning_quality(
            self.response,
            expected_concepts=self.expected_concepts,
            token_budget=self.token_budget,
            correctness=self.correctness,
        )

        # Compute final score
        self.final_score = compute_final_score(self.correctness, self.reasoning_quality)

        # Shape RL reward
        has_reasoning = bool(extract_reasoning(self.response))
        has_answer = bool(extract_answer(self.response))
        self.rl_reward = shape_rl_reward(
            self.correctness, self.reasoning_quality, self.response,
            has_reasoning, has_answer,
        )

        self.breakdown = {
            "correctness": self.correctness,
            "reasoning_quality": self.reasoning_quality,
            "final_score": self.final_score,
            "rl_reward": self.rl_reward,
            "correctness_details": correct_breakdown,
            "reasoning_details": {
                "coverage": reasoning_breakdown.coverage,
                "token_efficiency": reasoning_breakdown.token_efficiency,
                "verification": reasoning_breakdown.verification,
                "filler_penalty": reasoning_breakdown.filler_penalty,
                "tokens_used": reasoning_breakdown.tokens_used,
            },
        }

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "reasoning_skill": self.reasoning_skill,
            "correctness": self.correctness,
            "reasoning_quality": self.reasoning_quality,
            "final_score": self.final_score,
            "rl_reward": self.rl_reward,
            "has_reasoning": bool(extract_reasoning(self.response)),
            "has_answer": bool(extract_answer(self.response)),
            "response_length": len(self.response),
            "gen_time": self.gen_time,
        }


def sample_task(rng: random.Random, task_ids: list[str] | None = None) -> dict:
    """Sample a single task from the 20 hand-crafted tasks.

    Returns: {
        task_instance: LongHorizonEnv instance,
        codebase: {filename: content},
        task_prompt: task description string,
        task_id: task identifier,
        reasoning_skill: what reasoning capability,
        expected_concepts: concepts for reasoning scoring,
        token_budget: token budget for reasoning,
    }
    """
    if task_ids:
        task_classes = [register_long_horizon[tid] for tid in task_ids
                        if tid in register_long_horizon]
    else:
        task_classes = list(ALL_TASKS)

    task_class = rng.choice(task_classes)
    task = task_class()
    codebase = task.gen_codebase()
    task_prompt = task.gen_task(codebase)

    return {
        "task_instance": task,
        "codebase": codebase,
        "task_prompt": task_prompt,
        "task_id": task.task_id,
        "reasoning_skill": task.reasoning_skill,
        "expected_concepts": task.expected_concepts,
        "token_budget": task.token_budget,
    }


def collect_grpo_group(
    model,
    tokenizer,
    rng: random.Random,
    group_size: int = 6,
    thinking_tokens: int = 800,
    prediction_tokens: int = 1200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    task_ids: list[str] | None = None,
    grade_timeout: float = 20.0,
) -> dict:
    """Collect a group of rollouts for GRPO with efficiency-aware scoring.

    Samples one task, generates group_size responses, grades each with
    both correctness AND reasoning quality, returns the group.

    Returns: {
        task: the sampled task dict,
        rollouts: list of LongHorizonRollout,
        mean_reward: float,
        std_reward: float,
        advantages: list[float],
        mean_correctness: float,
        mean_reasoning_quality: float,
    }
    """
    task = sample_task(rng, task_ids)
    task_instance = task["task_instance"]

    rollouts = []
    for i in range(group_size):
        rollout = LongHorizonRollout(
            task_id=task["task_id"],
            reasoning_skill=task["reasoning_skill"],
            task_prompt=task["task_prompt"],
            codebase=task["codebase"],
            expected_concepts=task["expected_concepts"],
            token_budget=task["token_budget"],
        )
        rollouts.append(rollout)

    # Generate responses (MLX path)
    try:
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "user", "content": task["task_prompt"]}]
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        for i, rollout in enumerate(rollouts):
            t0 = time.time()
            sampler = make_sampler(
                temp=temperature + (i * 0.05),
                top_p=top_p,
                seed=rng.randint(0, 2**31 - 1),
            )

            generated_tokens = []
            max_tokens = thinking_tokens + prediction_tokens
            for token_id in generate_step(
                prompt_tokens, model, sampler, max_tokens=max_tokens,
            ):
                generated_tokens.append(token_id)
                if len(generated_tokens) > 20:
                    text = tokenizer.decode(generated_tokens[-30:])
                    if "</answer>" in text:
                        break

            rollout.tokens = generated_tokens
            rollout.response = tokenizer.decode(generated_tokens)
            rollout.gen_time = time.time() - t0

            # Grade with efficiency-aware scoring
            rollout.grade(task_instance)

    except ImportError:
        for rollout in rollouts:
            rollout.response = "<reasoning>\n[MLX not available]\n</reasoning>\n<answer>\n[no code]\n</answer>"
            rollout.correctness = 0.0
            rollout.reasoning_quality = 0.0
            rollout.final_score = 0.0
            rollout.rl_reward = 0.0

    # Compute group statistics
    rewards = [r.rl_reward for r in rollouts]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0.0

    # GRPO advantages
    eps = 1e-8
    advantages = [(r - mean_reward) / (std_reward + eps) for r in rewards]

    # Additional stats
    correctness_scores = [r.correctness for r in rollouts]
    reasoning_scores = [r.reasoning_quality for r in rollouts]
    mean_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
    mean_reasoning = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0

    return {
        "task": task,
        "rollouts": rollouts,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "advantages": advantages,
        "mean_correctness": mean_correctness,
        "mean_reasoning_quality": mean_reasoning,
    }


# ── Curriculum learning ──

# Group tasks by difficulty (based on token budget and complexity)
TASK_DIFFICULTY = {
    # Easier tasks (shorter reasoning, more straightforward)
    "minimal_change_identification": "easy",
    "bottleneck_isolation": "easy",
    "differential_analysis": "easy",
    "reachability_analysis": "easy",
    "recursive_repair": "easy",
    # Medium tasks
    "cascading_bug_chain": "medium",
    "complexity_optimization": "medium",
    "type_flow_inference": "medium",
    "error_propagation_analysis": "medium",
    "coverage_gap_analysis": "medium",
    "backward_compat_evolution": "medium",
    "invariant_preservation": "medium",
    "race_condition_detection": "medium",
    "property_based_tests": "medium",
    "design_pattern_selection": "medium",
    # Hard tasks (longer reasoning, more complex)
    "cross_module_data_flow": "hard",
    "api_contract_compliance": "hard",
    "state_machine_impl": "hard",
    "spec_compliance_audit": "hard",
    "security_audit": "hard",
}

CURRICULUM = {
    "phase1": {"difficulty": "easy", "task_filter": None},
    "phase2": {"difficulty": "medium", "task_filter": None},
    "phase3": {"difficulty": "hard", "task_filter": None},
}


def get_curriculum_tasks(iteration: int, total_iterations: int) -> list[str]:
    """Get the list of task IDs for the current curriculum phase.

    Phase 1 (0-30%): easy tasks
    Phase 2 (30-70%): medium tasks
    Phase 3 (70-100%): hard tasks
    """
    progress = iteration / max(1, total_iterations)
    if progress < 0.3:
        difficulty = "easy"
    elif progress < 0.7:
        difficulty = "medium"
    else:
        difficulty = "hard"

    return [tid for tid, diff in TASK_DIFFICULTY.items() if diff == difficulty]
