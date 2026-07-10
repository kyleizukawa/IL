"""
GRPO (Group-Relative Policy Optimization) trainer for Intuition Learning.

Algorithm:
1. Collect G rollouts for the same puzzle (same initial state, different sampling)
2. Compute total reward R_i for each rollout
3. Compute group-relative advantage: A_i = (R_i - mean(R)) / (std(R) + eps)
4. For each rollout:
   a. Forward the full episode through the model (train mode)
   b. Compute new logprobs at action token positions
   c. Compute per-token ratio: ratio_t = exp(new_logprob_t - old_logprob_t)
   d. Compute clipped loss: -clip(ratio_t, 1-eps, 1+eps) * A_i
5. Average over all action tokens, backprop, update LoRA params

The advantage is group-relative (no value network needed), which saves memory
on the 8GB Mac. The per-token ratio with clipping provides stable policy updates.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np

from .rollout import collect_rollout


def compute_action_logprobs(model, tokens, action_positions):
    """Forward the full episode and compute logprobs at action token positions.

    Memory optimization (the original OOM cause): the previous version called
    ``model(input_tokens)`` which applies the LM head to EVERY position,
    materializing a ``[1, seq_len, vocab=151936]`` fp32 logits tensor
    (~900 MB for a 1500-token episode) inside the autograd graph. On an 8 GB
    Mac this is fatal during the backward pass.

    Fix: forward only the transformer *backbone* (``model.model``) to get the
    hidden states ``[1, seq_len-1, hidden=1536]`` (~10 MB), then apply the
    ``lm_head`` *only* at the action-token positions. This shrinks the logits
    tensor from ``[~1500, 151936]`` to ``[~num_action_tokens, 151936]`` — a
    ~5x reduction — and is the single biggest memory win for GRPO on-device.

    Args:
        model: the MLX model (in train mode). Must expose ``.model`` (backbone)
               and ``.lm_head`` (the LM head) — true for mlx_lm Qwen2 models.
        tokens: list of token IDs for the full episode
        action_positions: list of (start, end) ranges for action tokens

    Returns:
        list of mx.arrays, one per action segment, containing logprobs of each token
    """
    tokens_arr = mx.array(tokens)

    # Forward the BACKBONE only (no LM head) -> hidden states [1, seq-1, hidden]
    # logits[i] predicts tokens[i+1], so input is tokens[:-1].
    input_tokens = tokens_arr[:-1][None]  # [1, seq_len-1]
    hidden = model.model(input_tokens)    # [1, seq_len-1, hidden]
    hidden = hidden[0]                     # [seq_len-1, hidden]

    # Extract logprobs at action positions.
    # hidden[i] predicts tokens[i+1], so for an action token at position p
    # the relevant hidden state is hidden[p-1].
    action_logprob_segments = []
    for (start, end) in action_positions:
        lp_start = start - 1
        lp_end = end - 1
        seg_hidden = hidden[lp_start:lp_end]      # [seg_len, hidden]
        segment_tokens = tokens_arr[start:end]    # [seg_len]
        # Apply LM head ONLY to this small segment -> [seg_len, vocab]
        seg_logits = model.lm_head(seg_hidden)
        seg_logprobs = seg_logits - mx.logsumexp(seg_logits, axis=-1, keepdims=True)
        # Gather logprob of the actual token at each position
        segment_lp = mx.take_along_axis(
            seg_logprobs,
            segment_tokens[:, None],
            axis=-1
        ).squeeze(-1)  # [seg_len]
        action_logprob_segments.append(segment_lp)

    return action_logprob_segments


def grpo_loss(model, rollout, advantage, clip_eps=0.2, kl_beta=0.04):
    """Compute GRPO loss for a single rollout.

    Args:
        model: MLX model (train mode)
        rollout: dict from collect_rollout
        advantage: scalar advantage for this rollout (group-relative)
        clip_eps: PPO clip parameter
        kl_beta: KL penalty coefficient. Penalizes the policy for moving away
            from the behavior policy (old_logprobs). This is the standard PPO
            trust-region KL that prevents catastrophic forgetting of the SFT
            solution. Without it, the policy drifts away from previously-
            learned solutions (which is exactly what happened in the v1 run:
            eval_rotate_003 went 0.84→1.00→0.00 by iter 50).

    Returns:
        loss: scalar mx.array (the policy gradient loss + KL penalty)
        n_action_tokens: number of action tokens in this rollout
    """
    tokens = rollout['tokens']
    action_positions = rollout['action_positions']
    old_logprobs = rollout['old_logprobs']  # list of floats

    # Compute new logprobs at action positions
    new_logprob_segments = compute_action_logprobs(model, tokens, action_positions)

    # Flatten new and old logprobs
    new_lp_flat = mx.concatenate(new_logprob_segments)  # [total_action_tokens]
    old_lp_flat = mx.array(old_logprobs)  # [total_action_tokens]

    # Per-token ratio
    ratio = mx.exp(new_lp_flat - old_lp_flat)

    # Clipped ratio
    clipped_ratio = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps)

    # Policy gradient loss: -clip(ratio, ...) * advantage
    # Negative because we minimize loss = maximize reward
    pg_loss = -mx.minimum(ratio * advantage, clipped_ratio * advantage)

    # KL penalty: approximate KL divergence between new and old policy.
    # KL = E[new_logprob - old_logprob] (per-token, the PPO approximation).
    # This keeps the policy from drifting too far from the behavior policy
    # in a single update, preventing catastrophic forgetting of SFT solutions.
    kl = (new_lp_flat - old_lp_flat).mean()

    loss = pg_loss + kl_beta * kl

    n_tokens = len(old_logprobs)
    return loss.mean(), n_tokens


def collect_group_rollouts(model, tokenizer, puzzle, group_size, **rollout_kwargs):
    """Collect G rollouts for the same puzzle with different seeds."""
    rollouts = []
    for g in range(group_size):
        kwargs = dict(rollout_kwargs)
        kwargs['seed'] = kwargs.get('seed', 42) + g * 10000
        rollout = collect_rollout(model, tokenizer, puzzle, **kwargs)
        rollouts.append(rollout)
    return rollouts


def compute_advantages(rollouts, eps=1e-8):
    """Compute group-relative advantages for GRPO.

    A_i = (R_i - mean(R)) / (std(R) + eps)

    If all rewards are the same (std=0), advantages are 0 (no learning signal).
    """
    rewards = np.array([r['reward'] for r in rollouts])
    mean_r = rewards.mean()
    std_r = rewards.std()

    if std_r < eps:
        # No variance in rewards — no learning signal
        return [0.0] * len(rollouts), mean_r, std_r

    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages.tolist(), mean_r, std_r


class GRPOTrainer:
    """GRPO trainer for intuition learning.

    Manages:
    - LoRA model setup
    - Rollout collection
    - Advantage computation
    - Policy gradient updates
    - Checkpointing
    """

    def __init__(
        self,
        model,
        tokenizer,
        adapter_path="il_rl_adapters",
        learning_rate=1e-5,
        clip_eps=0.2,
        group_size=4,
        gamma=0.9,
        n_steps=2,
        thinking_tokens=100,
        prediction_tokens=128,
        temperature=0.8,
        top_p=0.9,
        memory_limit_gb=5.0,
        kl_beta=0.04,
    ):
        """Args:
            model: an MLX model that ALREADY has LoRA layers applied (e.g. via
                ``mlx_lm.tuner.utils.load_adapters``). The trainer does NOT
                re-apply LoRA — doing so on an already-adapted SFT model would
                stack a second set of LoRA layers and discard the SFT weights.
            tokenizer: the matching mlx_lm tokenizer.
            adapter_path: where to save RL-adapted checkpoints.
            memory_limit_gb: cap on Metal unified-memory allocation. On an 8 GB
                Mac we leave headroom for the OS / model weights (~1.9 GB for a
                4-bit 1.5B model). Set to 0 to disable the limit.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_path = adapter_path
        self.clip_eps = clip_eps
        self.group_size = group_size
        self.gamma = gamma
        self.n_steps = n_steps
        self.thinking_tokens = thinking_tokens
        self.prediction_tokens = prediction_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kl_beta = kl_beta

        # Memory budget for the Metal unified-memory allocator. Capping this
        # gives a clean error instead of an OS-level OOM kill, and lets the
        # allocator reuse the wired memory pool more aggressively.
        # (mx.metal.* are deprecated in newer MLX in favour of mx.* — try both.)
        def _set_mem_limit(b):
            if hasattr(mx, "set_memory_limit"):
                mx.set_memory_limit(b)
            elif mx.metal.is_available():
                mx.metal.set_memory_limit(b)

        def _set_cache_limit(b):
            if hasattr(mx, "set_cache_limit"):
                mx.set_cache_limit(b)
            elif mx.metal.is_available():
                mx.metal.set_cache_limit(b)

        if memory_limit_gb > 0:
            _set_mem_limit(int(memory_limit_gb * 1024 ** 3))
            _set_cache_limit(int(1.5 * 1024 ** 3))
            # Increase the wired memory limit so the OS doesn't kill long
            # GPU command buffers (kIOGPUCommandBufferCallbackErrorImpactingInteractivity).
            if hasattr(mx, "set_wired_limit"):
                mx.set_wired_limit(int(memory_limit_gb * 1024 ** 3))
            elif mx.metal.is_available():
                mx.metal.set_wired_limit(int(memory_limit_gb * 1024 ** 3))

        # Optimizer (only updates trainable LoRA params)
        self.optimizer = opt.Adam(learning_rate=learning_rate)

        # Training state
        self.iteration = 0

    def train_step(self, puzzle):
        """One GRPO training step: collect rollouts, compute advantages, update.

        Returns a dict of metrics for logging.
        """
        import time
        t0 = time.time()

        # Reset peak-memory tracking for a per-iteration reading.
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif mx.metal.is_available():
            mx.metal.reset_peak_memory()
        mx.clear_cache()

        # ── Phase 1: Collect G rollouts (no gradients) ──
        self.model.eval()
        rollouts = []
        for g in range(self.group_size):
            mx.random.seed(42 + self.iteration * 1000 + g * 10000)
            rollout = collect_rollout(
                self.model, self.tokenizer, puzzle,
                n_steps=self.n_steps,
                gamma=self.gamma,
                thinking_tokens=self.thinking_tokens,
                prediction_tokens=self.prediction_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=42 + self.iteration * 1000 + g * 10000,
            )
            rollouts.append(rollout)
            # Free the KV cache / intermediate buffers between rollouts.
            mx.clear_cache()

        rollout_time = time.time() - t0

        # ── Phase 2: Compute group-relative advantages ──
        advantages, mean_reward, std_reward = compute_advantages(rollouts)

        # ── Phase 3: GRPO update (with gradients) ──
        self.model.train()
        t1 = time.time()

        loss_sum = 0.0   # track as a python float (avoid building a graph)
        n_updated = 0
        grad_accum = None

        for rollout, advantage in zip(rollouts, advantages):
            if abs(advantage) < 1e-8:
                continue  # skip zero-advantage rollouts (no learning signal)

            # nn.value_and_grad: the closure must access the model via closure.
            def loss_fn():
                loss, _ = grpo_loss(self.model, rollout, advantage,
                                    self.clip_eps, self.kl_beta)
                return loss

            loss_value_and_grad = nn.value_and_grad(self.model, loss_fn)
            loss_val, grad = loss_value_and_grad()

            # Materialize before accumulating so we don't hold two graphs.
            mx.eval(loss_val, grad)
            loss_sum += float(loss_val)
            n_updated += 1

            if grad_accum is None:
                grad_accum = grad
            else:
                grad_accum = tree_map(lambda x, y: x + y, grad_accum, grad)
            mx.eval(grad_accum)

            # Drop the per-rollout graph / activations before the next one.
            mx.clear_cache()

        # Apply gradient update. MLX is lazy — we MUST eval the updated params
        # and optimizer state, otherwise the update is never realized (this was
        # a silent no-op bug in the original code).
        if grad_accum is not None and n_updated > 0:
            grad_accum = tree_map(lambda x: x / n_updated, grad_accum)
            self.optimizer.update(self.model, grad_accum)
            mx.eval(self.model.parameters(), self.optimizer.state)

        mx.clear_cache()
        update_time = time.time() - t1
        total_time = time.time() - t0

        # Compute metrics
        rewards = [r['reward'] for r in rollouts]
        best_acc = max(r['env'].best_accuracy for r in rollouts)
        any_exact = any(r['env'].first_correct_step is not None for r in rollouts)
        avg_tokens = np.mean([len(r['tokens']) for r in rollouts])
        peak_mem = (mx.get_peak_memory() if hasattr(mx, "get_peak_memory")
                    else mx.metal.get_peak_memory()) / 1e9

        metrics = {
            'iteration': self.iteration,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'best_accuracy': best_acc,
            'any_exact': any_exact,
            'advantages': advantages,
            'avg_episode_tokens': avg_tokens,
            'rollout_time': rollout_time,
            'update_time': update_time,
            'total_time': total_time,
            'peak_memory': peak_mem,
            'loss': loss_sum / max(n_updated, 1),
        }

        self.iteration += 1
        return metrics

    def save(self, path=None):
        """Save LoRA adapter weights + a minimal adapter_config.json.

        The config lets the checkpoint be reloaded with
        ``mlx_lm.tuner.utils.load_adapters`` (or ``mlx_lm.load`` + adapter
        path) for downstream benchmarking.
        """
        import os, json
        path = path or self.adapter_path
        os.makedirs(path, exist_ok=True)
        adapter_weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(f"{path}/adapters.safetensors", adapter_weights)
        # Numbered checkpoint for time-travel / best-iteration recovery.
        ckpt = f"{path}/{self.iteration:07d}_adapters.safetensors"
        mx.save_safetensors(ckpt, adapter_weights)
        cfg = {
            "adapter_path": os.path.basename(path),
            "fine_tune_type": "lora",
            "num_layers": 16,
            "lora_parameters": {"rank": 8, "scale": 1.0, "dropout": 0.0},
        }
        with open(f"{path}/adapter_config.json", "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"Saved adapter weights to {path}/adapters.safetensors (iter {self.iteration})")

    def eval(self, puzzles, group_size=1, temperature=0.0):
        """Evaluate the current policy on a FIXED held-out puzzle set.

        No gradients, no parameter updates — pure policy rollouts. This gives
        an apples-to-apples measure of RL gains across iterations (unlike
        train_step, which uses a fresh puzzle each iter so its reward isn't a
        learning curve).

        Args:
            puzzles: list of puzzle dicts (the held-out eval set). Must be the
                SAME set every call for comparability.
            group_size: rollouts per puzzle. 1 with temperature=0 is greedy
                (deterministic, best for tracking a clean learning curve).
                >1 with temperature>0 samples and reports the best-of-group,
                which measures the policy's exploration ceiling.
            temperature: 0.0 = greedy, >0 = sampled.

        Returns:
            dict of aggregate eval metrics:
                mean_reward, mean_best_accuracy, exact_rate (fraction of
                puzzles solved exactly by >=1 rollout), n_puzzles,
                per_puzzle: list of per-puzzle dicts.
        """
        import time
        t0 = time.time()
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        self.model.eval()

        per_puzzle = []
        rewards = []
        best_accs = []
        exacts = []

        for p in puzzles:
            mx.clear_cache()
            if group_size <= 1 and temperature == 0.0:
                # Greedy single rollout.
                rollout = collect_rollout(
                    self.model, self.tokenizer, p,
                    n_steps=self.n_steps,
                    gamma=self.gamma,
                    thinking_tokens=self.thinking_tokens,
                    prediction_tokens=self.prediction_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    seed=0,
                )
                rollouts = [rollout]
            else:
                rollouts = []
                for g in range(group_size):
                    mx.random.seed(999 + g * 10000)
                    r = collect_rollout(
                        self.model, self.tokenizer, p,
                        n_steps=self.n_steps,
                        gamma=self.gamma,
                        thinking_tokens=self.thinking_tokens,
                        prediction_tokens=self.prediction_tokens,
                        temperature=temperature if temperature > 0 else 0.8,
                        top_p=self.top_p,
                        seed=999 + g * 10000,
                    )
                    rollouts.append(r)

            r_best = max(r['reward'] for r in rollouts)
            acc_best = max(r['env'].best_accuracy for r in rollouts)
            exact = any(r['env'].first_correct_step is not None for r in rollouts)
            rewards.append(r_best)
            best_accs.append(acc_best)
            exacts.append(exact)
            per_puzzle.append({
                'id': p.get('id', '?'),
                'type': p.get('type', '?'),
                'best_reward': r_best,
                'best_accuracy': acc_best,
                'exact': exact,
            })

        elapsed = time.time() - t0
        peak_mem = (mx.get_peak_memory() if hasattr(mx, "get_peak_memory")
                    else mx.metal.get_peak_memory()) / 1e9

        return {
            'iteration': self.iteration,
            'n_puzzles': len(puzzles),
            'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
            'mean_best_accuracy': float(np.mean(best_accs)) if best_accs else 0.0,
            'exact_rate': float(np.mean(exacts)) if exacts else 0.0,
            'n_exact': int(sum(exacts)),
            'eval_time': elapsed,
            'peak_memory': peak_mem,
            'per_puzzle': per_puzzle,
        }
