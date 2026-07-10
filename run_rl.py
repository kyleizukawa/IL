#!/usr/bin/env python3
"""GRPO-based Reinforcement Learning for Intuition Learning (IL).

Pipeline stage 3 (after SFT). Loads the SFT-trained LoRA adapters and continues
with group-relative policy optimization on the IL puzzle environment, directly
on-device with mlx_lm.

Memory strategy for an 8 GB Mac:
  * 4-bit base model (~1.9 GB) instead of 8-bit (~3.5 GB) — frees ~1.6 GB.
  * SFT LoRA adapters (rank 8, 16 layers, ~21 MB) loaded on top.
  * GRPO forward computes logits ONLY at action-token positions (see
    il_rl/grpo.py::compute_action_logprobs) — the original OOM cause was
    materializing full-vocab logits over the whole episode.
  * Bounded Metal memory limit + aggressive mx.clear_cache() between rollouts.
  * Short episodes (thinking=100, prediction=128, n_steps=2) by default.

Usage:
    python run_rl.py --smoke                 # 1 iteration sanity check
    python run_rl.py --iters 50              # short run
    python run_rl.py --iters 200 --group-size 4 --save-every 25
    python run_rl.py --no-sft                # RL from scratch (skip SFT adapters)
"""
import sys, os, json, time, random, argparse, traceback

HERE = "/Users/kzrr/ ILresearch "
sys.path.insert(0, HERE)

# Make the il_rl package and the il environments importable.
os.chdir(HERE)

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import load_adapters

from il_rl.grpo import GRPOTrainer
from il_rl.env import sample_puzzle


def count_trainable(model):
    """Total size of trainable (non-frozen) parameters."""
    return sum(p.size for _, p in tree_flatten(model.trainable_parameters())
               if hasattr(p, "size"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "model_mlx_4bit"),
                    help="Base mlx model path (4-bit by default to save memory).")
    ap.add_argument("--sft-adapters", default=os.path.join(HERE, "il_adapters"),
                    help="SFT LoRA adapter dir to warm-start from.")
    ap.add_argument("--no-sft", action="store_true",
                    help="Skip SFT adapters — RL from the base model.")
    ap.add_argument("--adapter-out", default=os.path.join(HERE, "il_rl_adapters"),
                    help="Where to save RL-adapted checkpoints.")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=2)
    ap.add_argument("--thinking-tokens", type=int, default=384)
    ap.add_argument("--prediction-tokens", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--kl-beta", type=float, default=0.04,
                    help="KL penalty coefficient (prevents policy drift from SFT).")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=10,
                    help="Run held-out eval every N iters (0 = disable).")
    ap.add_argument("--eval-size", type=int, default=8,
                    help="Number of held-out puzzles in the eval set.")
    ap.add_argument("--eval-group-size", type=int, default=4,
                    help="Rollouts per eval puzzle. 4 with temp>0 = best-of-4 (stable curve).")
    ap.add_argument("--eval-temperature", type=float, default=0.3,
                    help="Eval sampling temperature. 0.3 with K=4 gives stable best-of-K.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--memory-limit-gb", type=float, default=5.0)
    ap.add_argument("--smoke", action="store_true",
                    help="Single-iteration sanity check (overrides --iters).")
    ap.add_argument("--log-file", default=os.path.join(HERE, "il_rl_train.log"))
    ap.add_argument("--metrics-file", default=os.path.join(HERE, "il_rl_metrics.json"))
    ap.add_argument("--eval-file", default=os.path.join(HERE, "il_rl_eval.json"),
                    help="Where to dump the eval history (learning curve).")
    return ap.parse_args()


def build_puzzle_sampler(seed):
    """Return a callable that yields fresh puzzles, balanced across env types."""
    from il.environments import ENVIRONMENT_TYPES, generate_puzzle
    rng = random.Random(seed)
    types = list(ENVIRONMENT_TYPES)
    rng.shuffle(types)
    idx = [0]

    def next_puzzle():
        et = types[idx[0] % len(types)]
        idx[0] += 1
        p = generate_puzzle(et, rng)
        p['id'] = f"{et['name']}_{rng.randint(0, 99999)}"
        return p

    return next_puzzle


def build_eval_set(seed, n):
    """Build a FIXED held-out eval puzzle set (deterministic across runs).

    Uses a separate seed from training so the eval puzzles never overlap with
    training puzzles. The same set is reused at every eval point so the
    learning curve is apples-to-apples.
    """
    from il.environments import ENVIRONMENT_TYPES, generate_puzzle
    rng = random.Random(seed + 100000)  # disjoint from training seed
    types = list(ENVIRONMENT_TYPES)
    rng.shuffle(types)
    puzzles = []
    for i in range(n):
        et = types[i % len(types)]
        p = generate_puzzle(et, rng)
        p['id'] = f"eval_{et['name']}_{i:03d}"
        puzzles.append(p)
    return puzzles


def main():
    args = parse_args()
    if args.smoke:
        args.iters = 1
        args.group_size = 2
        args.log_every = 1

    log_f = open(args.log_file, "a", buffering=1)  # line-buffered

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")

    log("\n" + "=" * 70)
    log(f"IL-RL GRPO training | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log(f"  base model : {args.model}")
    log(f"  sft adapters: {args.sft_adapters if not args.no_sft else '(none — RL from scratch)'}")
    log(f"  iters={args.iters} group={args.group_size} n_steps={args.n_steps} "
        f"think={args.thinking_tokens} pred={args.prediction_tokens}")
    log(f"  lr={args.lr} temp={args.temperature} top_p={args.top_p} gamma={args.gamma}")
    log(f"  memory_limit={args.memory_limit_gb} GB  save_every={args.save_every}")

    # ── Load base model (4-bit) ──
    t0 = time.time()
    log(f"\nLoading base model from {args.model} ...")
    model, tokenizer = load(args.model)
    log(f"Loaded in {time.time()-t0:.1f}s")

    # ── Warm-start from SFT adapters (apply LoRA + load SFT weights) ──
    # ORDER MATTERS: freeze the base model FIRST, then apply LoRA. The LoRA
    # params created by linear_to_lora_layers are then the only trainable
    # params. Freezing AFTER load_adapters would freeze the LoRA params too
    # (which is exactly the 0-trainable-params bug we hit).
    if not args.no_sft and os.path.isdir(args.sft_adapters):
        t0 = time.time()
        log(f"\nFreezing base model + loading SFT adapters from {args.sft_adapters} ...")
        model.freeze()
        model = load_adapters(model, args.sft_adapters)
        model.train()
        n_train = count_trainable(model)
        log(f"SFT adapters loaded in {time.time()-t0:.1f}s | trainable params: {n_train/1e6:.3f}M")
        if n_train == 0:
            log("ERROR: no trainable parameters after loading SFT adapters. Aborting.")
            sys.exit(1)
    else:
        log("\nNo SFT warm-start — applying fresh LoRA layers on a frozen base.")
        from mlx_lm.tuner.utils import linear_to_lora_layers
        model.freeze()
        linear_to_lora_layers(model, 16, {'rank': 8, 'scale': 1.0, 'dropout': 0.0})
        model.train()
        n_train = count_trainable(model)
        log(f"  trainable params: {n_train/1e6:.3f}M")

    # ── Build trainer ──
    trainer = GRPOTrainer(
        model, tokenizer,
        adapter_path=args.adapter_out,
        learning_rate=args.lr,
        clip_eps=0.2,
        group_size=args.group_size,
        gamma=args.gamma,
        n_steps=args.n_steps,
        thinking_tokens=args.thinking_tokens,
        prediction_tokens=args.prediction_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        memory_limit_gb=args.memory_limit_gb,
        kl_beta=args.kl_beta,
    )

    next_puzzle = build_puzzle_sampler(args.seed)

    # ── Build fixed held-out eval set ──
    eval_puzzles = []
    eval_history = []
    if args.eval_every > 0:
        log(f"\nBuilding held-out eval set: {args.eval_size} puzzles "
            f"(seed {args.seed}+100000, disjoint from training) ...")
        eval_puzzles = build_eval_set(args.seed, args.eval_size)
        log(f"Eval set: {[p['id'] for p in eval_puzzles]}")
        log(f"Eval: every {args.eval_every} iters | group={args.eval_group_size} "
            f"temp={args.eval_temperature}")

    # ── Baseline eval (iter 0, before any RL updates) ──
    if eval_puzzles:
        log("\n--- BASELINE EVAL (pre-RL) ---")
        ev0 = trainer.eval(eval_puzzles,
                           group_size=args.eval_group_size,
                           temperature=args.eval_temperature)
        eval_history.append(ev0)
        with open(args.eval_file, "w") as f:
            json.dump(eval_history, f, indent=2)
        log(f"[eval iter 0] mean_R={ev0['mean_reward']:+.3f} "
            f"mean_best_acc={ev0['mean_best_accuracy']:.2%} "
            f"exact={ev0['n_exact']}/{ev0['n_puzzles']} "
            f"({ev0['exact_rate']:.1%}) "
            f"t={ev0['eval_time']:.0f}s mem={ev0['peak_memory']:.2f}GB")

    # ── Training loop ──
    metrics_history = []
    log("\nStarting GRPO training...\n")

    best_mean_reward = -1e9
    best_eval_reward = eval_history[0]['mean_reward'] if eval_history else -1e9
    overall_t0 = time.time()

    for it in range(args.iters):
        puzzle = next_puzzle()
        try:
            m = trainer.train_step(puzzle)
        except Exception as e:
            log(f"[iter {it}] ERROR on puzzle {puzzle.get('id','?')}: {e}")
            log(traceback.format_exc()[-800:])
            # Force a cache clear and continue — a single bad puzzle shouldn't
            # kill the run.
            mx.clear_cache()
            continue

        metrics_history.append(m)

        if it % args.log_every == 0 or it == args.iters - 1:
            log(
                f"[iter {m['iteration']:4d}] "
                f"puzzle={puzzle.get('id','?'):<22} "
                f"R={m['mean_reward']:+.3f} (std {m['std_reward']:.3f}, "
                f"max {m['max_reward']:+.3f}) "
                f"best_acc={m['best_accuracy']:.2f} exact={m['any_exact']} "
                f"loss={m['loss']:+.4f} "
                f"toks={m['avg_episode_tokens']:.0f} "
                f"mem={m['peak_memory']:.2f}GB "
                f"rollout={m['rollout_time']:.1f}s update={m['update_time']:.1f}s"
            )

        # Periodic held-out eval (the real learning curve).
        if eval_puzzles and args.eval_every > 0 and \
                (it + 1) % args.eval_every == 0:
            ev = trainer.eval(eval_puzzles,
                              group_size=args.eval_group_size,
                              temperature=args.eval_temperature)
            eval_history.append(ev)
            with open(args.eval_file, "w") as f:
                json.dump(eval_history, f, indent=2)
            delta = ev['mean_reward'] - eval_history[0]['mean_reward']
            log(f"  [eval iter {m['iteration']}] mean_R={ev['mean_reward']:+.3f} "
                f"(Δbaseline {delta:+.3f}) "
                f"mean_best_acc={ev['mean_best_accuracy']:.2%} "
                f"exact={ev['n_exact']}/{ev['n_puzzles']} "
                f"t={ev['eval_time']:.0f}s mem={ev['peak_memory']:.2f}GB")
            if ev['mean_reward'] > best_eval_reward:
                best_eval_reward = ev['mean_reward']
                trainer.save(path=args.adapter_out + "_best")
                log(f"  -> NEW BEST eval reward — saved to {os.path.basename(args.adapter_out)}_best")

        # Checkpointing
        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            trainer.save()
            with open(args.metrics_file, "w") as f:
                json.dump(metrics_history, f, indent=2)
            log(f"  -> checkpoint saved (metrics -> {os.path.basename(args.metrics_file)})")

        # Track best for a final note
        if m['mean_reward'] > best_mean_reward:
            best_mean_reward = m['mean_reward']

    elapsed = time.time() - overall_t0
    log("\n" + "=" * 70)
    log(f"RL training complete: {len(metrics_history)} iters in {elapsed:.0f}s "
        f"({elapsed/60:.1f} min)")
    log("=" * 70)
    if metrics_history:
        n = len(metrics_history)
        first_r = metrics_history[0]['mean_reward']
        last_r = metrics_history[-1]['mean_reward']
        exacts = sum(1 for m in metrics_history if m['any_exact'])
        log(f"  train mean R : first {first_r:+.3f} -> last {last_r:+.3f} "
            f"(per-puzzle, NOT a learning curve)")
        log(f"  best train R : {best_mean_reward:+.3f}")
        log(f"  exact hits   : {exacts}/{n} iters had >=1 exact rollout")
        log(f"  peak mem     : {max(m['peak_memory'] for m in metrics_history):.2f} GB")
    if eval_history:
        log("")
        log(f"  HELD-OUT EVAL (the real learning curve, {len(eval_history)} points):")
        e0 = eval_history[0]
        eN = eval_history[-1]
        log(f"    baseline  : mean_R={e0['mean_reward']:+.3f} "
            f"acc={e0['mean_best_accuracy']:.2%} exact={e0['n_exact']}/{e0['n_puzzles']}")
        log(f"    final     : mean_R={eN['mean_reward']:+.3f} "
            f"acc={eN['mean_best_accuracy']:.2%} exact={eN['n_exact']}/{eN['n_puzzles']}")
        log(f"    Δ reward  : {eN['mean_reward'] - e0['mean_reward']:+.3f}")
        log(f"    Δ acc     : {eN['mean_best_accuracy'] - e0['mean_best_accuracy']:+.2%}")
        log(f"    best R    : {best_eval_reward:+.3f}")
        log(f"    eval file : {args.eval_file}")
    log(f"  adapters    : {args.adapter_out}")
    log(f"  metrics     : {args.metrics_file}")
    log_f.close()


if __name__ == "__main__":
    main()
