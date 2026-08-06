"""
Benchmark for Long-Horizon Agentic Coding Tasks.

Evaluates a model on all 20 hand-crafted tasks with efficiency-aware scoring.

Reports:
- Per-task: correctness, reasoning_quality, final_score, rl_reward
- Per-difficulty: aggregate scores
- Overall: aggregate + reasoning quality breakdown
- Reasoning analysis: coverage, efficiency, verification, filler rates

Usage:
    python -m il_agentic.long_horizon.benchmark --model model_mlx_4bit
    python -m il_agentic.long_horizon.benchmark --smoke  # single task
"""
import json
import os
import sys
import time
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from il_agentic.long_horizon import ALL_TASKS, register_long_horizon
from il_agentic.long_horizon.efficiency import (
    score_reasoning_quality, compute_final_score, shape_rl_reward,
    extract_reasoning, extract_answer,
)


def evaluate_task(
    model,
    tokenizer,
    task_instance,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> dict:
    """Evaluate a single task with efficiency-aware scoring.

    Returns detailed results including correctness, reasoning quality,
    and final score.
    """
    codebase = task_instance.gen_codebase()
    task_prompt = task_instance.gen_task(codebase)

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
        response = "<reasoning>\n[MLX not available]\n</reasoning>\n<answer>\n[no code]\n</answer>"
        gen_time = 0.0

    # Grade correctness
    correctness, correct_breakdown = task_instance.grade_correctness(
        codebase, response
    )

    # Score reasoning quality
    reasoning_quality, reasoning_breakdown = score_reasoning_quality(
        response,
        expected_concepts=task_instance.expected_concepts,
        token_budget=task_instance.token_budget,
        correctness=correctness,
    )

    # Compute final score
    final_score = compute_final_score(correctness, reasoning_quality)

    # Shape RL reward
    has_reasoning = bool(extract_reasoning(response))
    has_answer = bool(extract_answer(response))
    rl_reward = shape_rl_reward(
        correctness, reasoning_quality, response,
        has_reasoning, has_answer,
    )

    return {
        "task_id": task_instance.task_id,
        "reasoning_skill": task_instance.reasoning_skill,
        "failure_mode": task_instance.failure_mode,
        "correctness": correctness,
        "reasoning_quality": reasoning_quality,
        "final_score": final_score,
        "rl_reward": rl_reward,
        "has_reasoning": has_reasoning,
        "has_answer": has_answer,
        "response_length": len(response),
        "reasoning_length": reasoning_breakdown.reasoning_length,
        "gen_time": gen_time,
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
        },
        "response_preview": response[:500],
    }


def run_benchmark(
    model,
    tokenizer,
    output_path: str = "il_long_horizon_benchmark.json",
    max_tokens: int = 2048,
    smoke: bool = False,
    task_filter: list[str] | None = None,
) -> dict:
    """Run the full benchmark on all 20 tasks.

    Returns aggregated results with efficiency-aware scoring.
    """
    # Select tasks
    if task_filter:
        task_classes = [register_long_horizon[tid] for tid in task_filter
                        if tid in register_long_horizon]
    else:
        task_classes = list(ALL_TASKS)

    if smoke:
        task_classes = task_classes[:1]

    print(f"\n{'='*60}", flush=True)
    print(f"IL Long-Horizon Benchmark", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Tasks: {len(task_classes)}", flush=True)
    print(f"  Max tokens: {max_tokens}", flush=True)
    print(f"  Smoke: {smoke}", flush=True)
    print(f"", flush=True)

    results = []
    t_start = time.time()

    for i, task_class in enumerate(task_classes):
        task = task_class()
        print(f"  [{i+1}/{len(task_classes)}] {task.task_id} "
              f"({task.reasoning_skill[:40]})...", flush=True, end=" ")

        try:
            result = evaluate_task(model, tokenizer, task, max_tokens=max_tokens)
            results.append(result)
            print(f"correctness={result['correctness']:.2f} "
                  f"reasoning={result['reasoning_quality']:.2f} "
                  f"final={result['final_score']:.2f} "
                  f"({result['gen_time']:.1f}s)", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results.append({
                "task_id": task.task_id,
                "reasoning_skill": task.reasoning_skill,
                "correctness": 0.0,
                "reasoning_quality": 0.0,
                "final_score": 0.0,
                "rl_reward": 0.0,
                "error": str(e),
            })

    # Compute summary
    elapsed = time.time() - t_start
    summary = _summarize(results)
    summary["elapsed_seconds"] = elapsed
    summary["n_tasks"] = len(results)

    output = {"summary": summary, "results": results}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    _print_summary(summary)
    return output


def _summarize(results: list) -> dict:
    """Compute aggregate statistics with efficiency-aware metrics."""
    if not results:
        return {}

    correctness_scores = [r["correctness"] for r in results]
    reasoning_scores = [r["reasoning_quality"] for r in results]
    final_scores = [r["final_score"] for r in results]
    rl_rewards = [r["rl_reward"] for r in results]

    overall = {
        "mean_correctness": sum(correctness_scores) / len(correctness_scores),
        "mean_reasoning_quality": sum(reasoning_scores) / len(reasoning_scores),
        "mean_final_score": sum(final_scores) / len(final_scores),
        "mean_rl_reward": sum(rl_rewards) / len(rl_rewards),
        "n_perfect": sum(1 for s in final_scores if s >= 0.95),
        "n_zero": sum(1 for s in final_scores if s == 0.0),
        "n_partial": sum(1 for s in final_scores if 0 < s < 0.95),
    }

    # Per task
    per_task = {}
    for r in results:
        per_task[r["task_id"]] = {
            "correctness": r["correctness"],
            "reasoning_quality": r["reasoning_quality"],
            "final_score": r["final_score"],
            "rl_reward": r["rl_reward"],
        }

    # Reasoning quality breakdown
    coverage_scores = [r.get("reasoning_details", {}).get("coverage", 0) for r in results]
    efficiency_scores = [r.get("reasoning_details", {}).get("token_efficiency", 0) for r in results]
    verification_scores = [r.get("reasoning_details", {}).get("verification", 0) for r in results]
    filler_penalties = [r.get("reasoning_details", {}).get("filler_penalty", 0) for r in results]

    reasoning_breakdown = {
        "mean_coverage": sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0,
        "mean_token_efficiency": sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0,
        "mean_verification": sum(verification_scores) / len(verification_scores) if verification_scores else 0,
        "mean_filler_penalty": sum(filler_penalties) / len(filler_penalties) if filler_penalties else 0,
        "verification_rate": sum(1 for v in verification_scores if v > 0) / len(verification_scores) if verification_scores else 0,
    }

    # Has reasoning/answer rates
    has_reasoning = sum(1 for r in results if r.get("has_reasoning", False))
    has_answer = sum(1 for r in results if r.get("has_answer", False))

    return {
        "overall": overall,
        "per_task": per_task,
        "reasoning_breakdown": reasoning_breakdown,
        "reasoning_rate": has_reasoning / len(results),
        "answer_rate": has_answer / len(results),
        "avg_reasoning_length": sum(r.get("reasoning_length", 0) for r in results) / len(results),
        "avg_gen_time": sum(r.get("gen_time", 0) for r in results) / len(results),
    }


def _print_summary(summary: dict):
    """Print benchmark summary."""
    print(f"\n{'='*60}", flush=True)
    print(f"Benchmark Summary (Efficiency-Aware)", flush=True)
    print(f"{'='*60}", flush=True)

    o = summary["overall"]
    print(f"\n  Overall:", flush=True)
    print(f"    Mean correctness:       {o['mean_correctness']:.3f}", flush=True)
    print(f"    Mean reasoning quality: {o['mean_reasoning_quality']:.3f}", flush=True)
    print(f"    Mean final score:       {o['mean_final_score']:.3f}", flush=True)
    print(f"    Mean RL reward:         {o['mean_rl_reward']:.3f}", flush=True)
    print(f"    Perfect (≥0.95):        {o['n_perfect']}/{o.get('n_tasks', '?')}", flush=True)
    print(f"    Partial:                {o['n_partial']}/{o.get('n_tasks', '?')}", flush=True)
    print(f"    Zero:                   {o['n_zero']}/{o.get('n_tasks', '?')}", flush=True)

    rb = summary["reasoning_breakdown"]
    print(f"\n  Reasoning Quality Breakdown:", flush=True)
    print(f"    Coverage:          {rb['mean_coverage']:.3f}", flush=True)
    print(f"    Token efficiency:  {rb['mean_token_efficiency']:.3f}", flush=True)
    print(f"    Verification:      {rb['mean_verification']:.3f}", flush=True)
    print(f"    Filler penalty:    {rb['mean_filler_penalty']:.3f}", flush=True)
    print(f"    Verification rate: {rb['verification_rate']:.1%}", flush=True)

    print(f"\n  Per task:", flush=True)
    for task_id, stats in sorted(summary["per_task"].items()):
        print(f"    {task_id:<35} C={stats['correctness']:.2f} "
              f"R={stats['reasoning_quality']:.2f} "
              f"F={stats['final_score']:.2f}", flush=True)

    print(f"\n  Reasoning rate: {summary['reasoning_rate']:.1%}", flush=True)
    print(f"  Answer rate:    {summary['answer_rate']:.1%}", flush=True)
    print(f"  Avg reasoning length: {summary['avg_reasoning_length']:.0f} chars", flush=True)
    print(f"  Avg gen time:   {summary['avg_gen_time']:.1f}s", flush=True)
    print(f"  Total elapsed:  {summary['elapsed_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Long-Horizon Benchmark")
    parser.add_argument("--model", type=str, default="model_mlx_4bit")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output", type=str, default="il_long_horizon_benchmark.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tasks", type=str, nargs="*",
                        help="filter to specific task IDs")
    args = parser.parse_args()

    try:
        from mlx_lm import load
        model, tokenizer = load(args.model)
    except ImportError:
        print("MLX not available. Running in dry-run mode.", flush=True)
        model, tokenizer = None, None

    run_benchmark(
        model=model,
        tokenizer=tokenizer,
        output_path=args.output,
        max_tokens=args.max_tokens,
        smoke=args.smoke,
        task_filter=args.tasks,
    )


if __name__ == '__main__':
    main()
