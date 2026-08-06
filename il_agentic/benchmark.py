"""
Benchmark for Agentic Coding Environments.

Evaluates a model on held-out task instances across all 15 environments.
Designed to measure transfer of agentic coding skills, not memorization.

The benchmark uses:
- Different random seeds than training (held-out instances)
- All difficulty levels
- Multiple instances per environment per difficulty
- Reports per-environment, per-difficulty, and aggregate scores

Usage:
    python -m il_agentic.benchmark --model model_mlx_4bit --max-tokens 2048
    python -m il_agentic.benchmark --smoke  # single-instance sanity check
    python -m il_agentic.benchmark --resume  # skip already-evaluated instances
"""
import json
import os
import sys
import time
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from il_agentic import ALL_ENVIRONMENTS, ENV_REGISTRY
from il_agentic.graders import extract_reasoning, extract_answer


# ── Benchmark configuration ──

# Held-out seeds (different from training seeds)
BENCHMARK_SEED = 99999

# Number of instances per environment per difficulty
INSTANCES_PER_ENV = {
    "easy": 2,
    "medium": 3,
    "hard": 2,
}

# Transfer environments: these are variant tasks that test generalization
# We use the same environments but with different parameters/seeds
TRANSFER_CONFIG = {
    "use_transfer_seeds": True,
    "transfer_seed_offset": 50000,
}


def generate_benchmark_instances(seed: int = BENCHMARK_SEED) -> list[dict]:
    """Generate the full benchmark instance set.

    Returns list of task instances, each with:
        env_name, difficulty, params, codebase, task, solution, reasoning
    """
    rng = random.Random(seed)
    instances = []

    for env_class in ALL_ENVIRONMENTS:
        env = env_class()
        for difficulty in ["easy", "medium", "hard"]:
            n = INSTANCES_PER_ENV.get(difficulty, 2)
            for i in range(n):
                try:
                    inst = env.generate_instance(rng, difficulty)
                    inst["instance_id"] = f"{env.name}_{difficulty}_{i}"
                    instances.append(inst)
                except Exception as e:
                    print(f"  WARN: {env.name} {difficulty} instance {i} failed: {e}",
                          file=sys.stderr)

    return instances


def evaluate_instance(
    model,
    tokenizer,
    instance: dict,
    max_tokens: int = 2048,
    temperature: float = 0.3,  # low temp for deterministic eval
) -> dict:
    """Evaluate a single benchmark instance.

    Returns: {
        instance_id, env_name, difficulty,
        score, breakdown,
        response_length, reasoning_length, answer_length,
        gen_time, response_preview,
    }
    """
    env = ENV_REGISTRY[instance["env_name"]]()
    task_prompt = instance["task"]

    # Generate response
    t0 = time.time()
    try:
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "user", "content": task_prompt}]
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        sampler = make_sampler(temp=temperature, top_p=0.95)
        generated = []
        for token_id in generate_step(
            prompt_tokens, model, sampler, max_tokens=max_tokens
        ):
            generated.append(token_id)
            if len(generated) > 20:
                text = tokenizer.decode(generated[-30:])
                if "</answer>" in text:
                    break

        response = tokenizer.decode(generated)
        gen_time = time.time() - t0

    except ImportError:
        # No MLX — return placeholder
        response = "<reasoning>\n[MLX not available]\n</reasoning>\n<answer>\n[no code]\n</answer>"
        gen_time = 0.0

    # Grade
    score, breakdown = env.grade(
        instance["params"], instance["codebase"], response
    )

    reasoning = extract_reasoning(response)
    answer = extract_answer(response)

    return {
        "instance_id": instance.get("instance_id", ""),
        "env_name": instance["env_name"],
        "difficulty": instance["difficulty"],
        "score": score,
        "breakdown": breakdown,
        "response_length": len(response),
        "reasoning_length": len(reasoning),
        "answer_length": len(answer),
        "has_reasoning": bool(reasoning),
        "has_answer": bool(answer),
        "gen_time": gen_time,
        "response_preview": response[:500],
    }


def run_benchmark(
    model,
    tokenizer,
    output_path: str = "il_agentic_benchmark.json",
    max_tokens: int = 2048,
    smoke: bool = False,
    resume: bool = False,
    env_filter: list[str] | None = None,
) -> dict:
    """Run the full benchmark.

    Args:
        model: MLX model
        tokenizer: MLX tokenizer
        output_path: where to save results
        max_tokens: max generation tokens
        smoke: if True, only run 1 instance per env (sanity check)
        resume: if True, load existing results and skip evaluated instances
        env_filter: if provided, only evaluate these environments

    Returns: aggregated results dict
    """
    # Load existing results if resuming
    existing_results = {}
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            data = json.load(f)
            for r in data.get("instances", []):
                existing_results[r["instance_id"]] = r
        print(f"Loaded {len(existing_results)} existing results for resume", flush=True)

    # Generate benchmark instances
    instances = generate_benchmark_instances()

    if env_filter:
        instances = [i for i in instances if i["env_name"] in env_filter]

    if smoke:
        # One instance per env, easy difficulty
        seen_envs = set()
        smoke_instances = []
        for inst in instances:
            if inst["env_name"] not in seen_envs and inst["difficulty"] == "easy":
                smoke_instances.append(inst)
                seen_envs.add(inst["env_name"])
        instances = smoke_instances

    print(f"\n{'='*60}", flush=True)
    print(f"IL Agentic Coding Benchmark", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Instances: {len(instances)}", flush=True)
    print(f"  Environments: {len(ALL_ENVIRONMENTS)}", flush=True)
    print(f"  Max tokens: {max_tokens}", flush=True)
    print(f"  Smoke mode: {smoke}", flush=True)
    print(f"  Resume: {resume}", flush=True)
    print(f"", flush=True)

    results = []
    t_start = time.time()

    for i, instance in enumerate(instances):
        inst_id = instance.get("instance_id", f"{instance['env_name']}_{i}")

        # Skip if already evaluated
        if inst_id in existing_results:
            results.append(existing_results[inst_id])
            print(f"  [{i+1}/{len(instances)}] {inst_id} — CACHED (score: {existing_results[inst_id]['score']:.2f})", flush=True)
            continue

        print(f"  [{i+1}/{len(instances)}] {inst_id} ({instance['difficulty']})...",
              flush=True, end=" ")

        try:
            result = evaluate_instance(
                model, tokenizer, instance, max_tokens=max_tokens
            )
            results.append(result)
            print(f"score: {result['score']:.2f} ({result['gen_time']:.1f}s)", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            results.append({
                "instance_id": inst_id,
                "env_name": instance["env_name"],
                "difficulty": instance["difficulty"],
                "score": 0.0,
                "breakdown": {"error": str(e)},
                "gen_time": 0.0,
            })

        # Save intermediate results
        if (i + 1) % 5 == 0 or i == len(instances) - 1:
            _save_results(results, output_path, t_start)

    # Final save
    elapsed = time.time() - t_start
    summary = _summarize_results(results)
    summary["elapsed_seconds"] = elapsed
    summary["n_instances"] = len(results)

    output = {
        "summary": summary,
        "instances": results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    _print_summary(summary)
    return output


def _save_results(results: list, path: str, t_start: float):
    """Save intermediate results."""
    summary = _summarize_results(results)
    summary["elapsed_seconds"] = time.time() - t_start
    summary["n_instances"] = len(results)
    output = {"summary": summary, "instances": results}
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)


def _summarize_results(results: list) -> dict:
    """Compute aggregate statistics."""
    if not results:
        return {}

    # Overall
    scores = [r["score"] for r in results]
    overall = {
        "mean_score": sum(scores) / len(scores),
        "median_score": sorted(scores)[len(scores) // 2],
        "max_score": max(scores),
        "min_score": min(scores),
        "n_zero": sum(1 for s in scores if s == 0.0),
        "n_perfect": sum(1 for s in scores if s >= 0.95),
        "n_partial": sum(1 for s in scores if 0 < s < 0.95),
    }

    # Per environment
    by_env = defaultdict(list)
    for r in results:
        by_env[r["env_name"]].append(r["score"])

    per_env = {}
    for env_name, env_scores in sorted(by_env.items()):
        per_env[env_name] = {
            "mean": sum(env_scores) / len(env_scores),
            "n": len(env_scores),
            "perfect": sum(1 for s in env_scores if s >= 0.95),
            "zero": sum(1 for s in env_scores if s == 0.0),
        }

    # Per difficulty
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r["score"])

    per_diff = {}
    for diff, diff_scores in sorted(by_diff.items()):
        per_diff[diff] = {
            "mean": sum(diff_scores) / len(diff_scores),
            "n": len(diff_scores),
        }

    # Reasoning stats
    has_reasoning = sum(1 for r in results if r.get("has_reasoning", False))
    has_answer = sum(1 for r in results if r.get("has_answer", False))
    avg_reasoning_len = sum(r.get("reasoning_length", 0) for r in results) / len(results)
    avg_gen_time = sum(r.get("gen_time", 0) for r in results) / len(results)

    return {
        "overall": overall,
        "per_environment": per_env,
        "per_difficulty": per_diff,
        "reasoning_rate": has_reasoning / len(results),
        "answer_rate": has_answer / len(results),
        "avg_reasoning_length": avg_reasoning_len,
        "avg_gen_time": avg_gen_time,
    }


def _print_summary(summary: dict):
    """Print benchmark summary."""
    print(f"\n{'='*60}", flush=True)
    print(f"Benchmark Summary", flush=True)
    print(f"{'='*60}", flush=True)

    o = summary["overall"]
    print(f"\n  Overall:", flush=True)
    print(f"    Mean score: {o['mean_score']:.3f}", flush=True)
    print(f"    Median:     {o['median_score']:.3f}", flush=True)
    print(f"    Perfect:    {o['n_perfect']}/{o.get('n_instances', '?')}", flush=True)
    print(f"    Partial:    {o['n_partial']}/{o.get('n_instances', '?')}", flush=True)
    print(f"    Zero:       {o['n_zero']}/{o.get('n_instances', '?')}", flush=True)

    print(f"\n  Per environment:", flush=True)
    for env_name, stats in summary["per_environment"].items():
        print(f"    {env_name:30s} mean={stats['mean']:.3f}  "
              f"perfect={stats['perfect']}/{stats['n']}  "
              f"zero={stats['zero']}/{stats['n']}", flush=True)

    print(f"\n  Per difficulty:", flush=True)
    for diff, stats in summary["per_difficulty"].items():
        print(f"    {diff:10s} mean={stats['mean']:.3f}  n={stats['n']}", flush=True)

    print(f"\n  Reasoning rate: {summary['reasoning_rate']:.1%}", flush=True)
    print(f"  Answer rate:    {summary['answer_rate']:.1%}", flush=True)
    print(f"  Avg reasoning length: {summary['avg_reasoning_length']:.0f} chars", flush=True)
    print(f"  Avg gen time:   {summary['avg_gen_time']:.1f}s", flush=True)
    print(f"  Total elapsed:  {summary['elapsed_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="IL Agentic Benchmark")
    parser.add_argument("--model", type=str, default="model_mlx_4bit",
                        help="model path")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output", type=str, default="il_agentic_benchmark.json")
    parser.add_argument("--smoke", action="store_true", help="single-instance sanity check")
    parser.add_argument("--resume", action="store_true", help="skip evaluated instances")
    parser.add_argument("--envs", type=str, nargs="*",
                        help="filter to specific environments")
    args = parser.parse_args()

    # Load model
    try:
        from mlx_lm import load
        model, tokenizer = load(args.model)
    except ImportError:
        print("MLX not available. Running in dry-run mode (no generation).", flush=True)
        model, tokenizer = None, None

    run_benchmark(
        model=model,
        tokenizer=tokenizer,
        output_path=args.output,
        max_tokens=args.max_tokens,
        smoke=args.smoke,
        resume=args.resume,
        env_filter=args.envs,
    )


if __name__ == '__main__':
    main()
