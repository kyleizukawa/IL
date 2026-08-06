"""
RL Rollout Interface for Agentic Coding Environments.

Adapts the agentic environments to the GRPO RL training loop.
Unlike the grid-puzzle rollout, agentic tasks are single-turn:
the model sees the task (codebase + instructions), generates a response
(reasoning + answer), and the grader scores it.

The reward is the grader's score [0, 1], which provides partial credit
for partial progress — critical for RL signal quality.

For GRPO, we collect group_size rollouts per task (same task, different
sampling temperatures/seeds), compute advantages relative to the group,
and update the policy.
"""
import sys
import os
import random
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from il_agentic import ALL_ENVIRONMENTS, ENV_REGISTRY
from il_agentic.graders import extract_reasoning, extract_answer


class AgenticRollout:
    """Represents a single rollout for an agentic coding task.

    Attributes:
        env_name: which environment
        difficulty: difficulty tier
        task_prompt: the user message (task description)
        response: the model's response
        reasoning: extracted reasoning text
        answer: extracted answer text
        score: grader score [0, 1]
        breakdown: grader breakdown dict
        tokens: full token sequence (for GRPO)
        action_positions: positions of model-generated tokens
        old_logprobs: logprobs for each action token
    """

    def __init__(self, env_name, difficulty, task_prompt, params, codebase):
        self.env_name = env_name
        self.difficulty = difficulty
        self.task_prompt = task_prompt
        self.params = params
        self.codebase = codebase
        self.response = ""
        self.reasoning = ""
        self.answer = ""
        self.score = 0.0
        self.breakdown = {}
        self.tokens = []
        self.action_positions = []
        self.old_logprobs = []
        self.gen_time = 0.0

    def grade(self, env_instance):
        """Grade this rollout using the environment's grader."""
        self.reasoning = extract_reasoning(self.response)
        self.answer = extract_answer(self.response)
        self.score, self.breakdown = env_instance.grade(
            self.params, self.codebase, self.response
        )

    def to_dict(self):
        return {
            "env_name": self.env_name,
            "difficulty": self.difficulty,
            "score": self.score,
            "breakdown": self.breakdown,
            "has_reasoning": bool(self.reasoning),
            "has_answer": bool(self.answer),
            "response_length": len(self.response),
            "gen_time": self.gen_time,
        }


def sample_task(rng: random.Random, difficulty: str = "medium",
                env_names: list[str] | None = None) -> dict:
    """Sample a single task from the environment suite.

    Returns: {
        env_instance: AgenticEnv instance,
        params: task params,
        codebase: {filename: content},
        task_prompt: task description string,
        difficulty: difficulty tier,
        env_name: environment name,
    }
    """
    if env_names:
        env_classes = [ENV_REGISTRY[name] for name in env_names if name in ENV_REGISTRY]
    else:
        env_classes = ALL_ENVIRONMENTS

    env_class = rng.choice(env_classes)
    env = env_class()
    params = env.gen_params(rng, difficulty)
    codebase = env.gen_codebase(params, rng)
    task_prompt = env.gen_task(params, codebase)

    return {
        "env_instance": env,
        "params": params,
        "codebase": codebase,
        "task_prompt": task_prompt,
        "difficulty": difficulty,
        "env_name": env.name,
    }


def collect_grpo_group(
    model,
    tokenizer,
    rng: random.Random,
    difficulty: str = "medium",
    group_size: int = 8,
    thinking_tokens: int = 512,
    prediction_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    env_names: list[str] | None = None,
    grade_timeout: float = 15.0,
) -> dict:
    """Collect a group of rollouts for GRPO.

    Samples one task, generates group_size responses with different seeds,
    grades each, and returns the group.

    For MLX-based generation (local Mac), uses mlx_lm.generate_step.
    For PyTorch-based generation (Kaggle), the caller should adapt this.

    Returns: {
        task: the sampled task dict,
        rollouts: list of AgenticRollout,
        mean_score: float,
        std_score: float,
        advantages: list[float],  # normalized advantages for GRPO
    }
    """
    task = sample_task(rng, difficulty, env_names)
    env = task["env_instance"]

    rollouts = []
    for i in range(group_size):
        rollout = AgenticRollout(
            env_name=task["env_name"],
            difficulty=task["difficulty"],
            task_prompt=task["task_prompt"],
            params=task["params"],
            codebase=task["codebase"],
        )
        rollouts.append(rollout)

    # Generate responses
    # This is the MLX-specific path. For PyTorch, replace with model.generate()
    try:
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        THINK_OPEN = "<think>"
        THINK_CLOSE = "</think>"

        messages = [{"role": "user", "content": task["task_prompt"]}]

        # Apply chat template
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        for i, rollout in enumerate(rollouts):
            t0 = time.time()
            sampler = make_sampler(
                temp=temperature + (i * 0.05),  # slight temperature variation
                top_p=top_p,
                seed=rng.randint(0, 2**31 - 1),
            )

            # Generate reasoning + answer in one pass
            # The model should produce <reasoning>...</reasoning><answer>...</answer>
            generated_tokens = []
            for token_id in generate_step(
                prompt_tokens, model, sampler,
                max_tokens=thinking_tokens + prediction_tokens,
            ):
                generated_tokens.append(token_id)
                # Check if we've generated the closing </answer> tag
                if len(generated_tokens) > 10:
                    text = tokenizer.decode(generated_tokens[-20:])
                    if "</answer>" in text:
                        break

            rollout.tokens = generated_tokens
            rollout.response = tokenizer.decode(generated_tokens)
            rollout.gen_time = time.time() - t0

            # Grade
            rollout.grade(env)

    except ImportError:
        # MLX not available — this is the PyTorch path stub
        # The actual generation will be handled by the Kaggle notebook
        for rollout in rollouts:
            rollout.response = "<reasoning>\n[MLX not available — generation skipped]\n</reasoning>\n<answer>\n[no code]\n</answer>"
            rollout.score = 0.0
            rollout.breakdown = {"error": "mlx not available"}

    # Compute group statistics
    scores = [r.score for r in rollouts]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 0.0

    # GRPO advantages: (score - mean) / (std + eps)
    eps = 1e-8
    advantages = [(s - mean_score) / (std_score + eps) for s in scores]

    return {
        "task": task,
        "rollouts": rollouts,
        "mean_score": mean_score,
        "std_score": std_score,
        "advantages": advantages,
    }


# ── Curriculum learning ──

CURRICULUM_PHASES = {
    # Phase 1 (0-30%): easy tasks, focus on pattern recognition
    "phase1": {
        "difficulty": "easy",
        "env_weights": None,  # uniform
    },
    # Phase 2 (30-70%): medium tasks, all environments
    "phase2": {
        "difficulty": "medium",
        "env_weights": None,
    },
    # Phase 3 (70-100%): hard tasks + mixed, focus on hard environments
    "phase3": {
        "difficulty": "hard",
        "env_weights": None,
    },
}


def get_curriculum_config(iteration: int, total_iterations: int) -> dict:
    """Get curriculum configuration for a given iteration.

    Phase 1: 0-30% of training — easy difficulty
    Phase 2: 30-70% — medium difficulty
    Phase 3: 70-100% — hard difficulty
    """
    progress = iteration / max(1, total_iterations)
    if progress < 0.3:
        return CURRICULUM_PHASES["phase1"]
    elif progress < 0.7:
        return CURRICULUM_PHASES["phase2"]
    else:
        return CURRICULUM_PHASES["phase3"]


# ── Reward shaping ──

def shape_reward(score: float, breakdown: dict, params: dict) -> float:
    """Shape the raw grader score into an RL reward.

    Enhancements over raw score:
    - Bonus for having both reasoning AND answer (punishes laziness)
    - Penalty for empty/no response
    - Small bonus for changing the correct file (shows codebase understanding)
    - Small bonus for thorough reasoning (longer reasoning = more thorough)
    """
    reward = score  # base reward [0, 1]

    # Penalty for no response
    if score == 0.0 and not breakdown.get("has_reasoning", False):
        reward = -0.1  # slight negative for complete failure

    # Bonus for having reasoning (punishes skipping analysis)
    if breakdown.get("has_reasoning", False):
        reward += 0.05

    # Bonus for changing the target file (shows understanding)
    if breakdown.get("changed_target", False):
        reward += 0.05

    # Bonus for partial credit (shows some understanding)
    if 0 < score < 1.0:
        reward += 0.03  # partial progress bonus

    # Clamp to [-0.2, 1.2]
    reward = max(-0.2, min(1.2, reward))

    return reward
