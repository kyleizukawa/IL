"""
SFT Data Generator for Long-Horizon Agentic Coding Tasks.

Generates chat-formatted training examples from all 20 hand-crafted tasks.
Since each task is hand-crafted (not procedurally generated), we generate
multiple examples per task by:
1. Using the base task as-is (the primary example)
2. Creating variations with different framing/emphasis
3. Adding "self-verification" examples where the teacher traces the verification

The teacher reasoning is designed to be THOROUGH and EFFICIENT:
- 400-800 words of step-by-step reasoning
- Traces code line by line
- Identifies issues with evidence
- Verifies the fix
- Does NOT use filler phrases
- Stays within the token budget
"""
import json
import os
import sys
import random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from il_agentic.long_horizon import ALL_TASKS, register_long_horizon


def generate_sft_dataset(
    examples_per_task: int = 5,
    seed: int = 12345,
    val_ratio: float = 0.1,
) -> tuple[list, list]:
    """Generate the SFT dataset across all 20 hand-crafted tasks.

    Args:
        examples_per_task: number of examples per task (with variations)
        seed: random seed
        val_ratio: fraction for validation

    Returns: (train_examples, val_examples)
    """
    rng = random.Random(seed)
    all_examples = []

    for task_class in ALL_TASKS:
        task = task_class()

        for i in range(examples_per_task):
            try:
                # Generate the base SFT example
                example = task.build_sft_example()

                # Add variation metadata
                example["metadata"]["variation"] = i
                example["metadata"]["example_idx"] = i

                # For variations > 0, we could modify the framing
                # but since these are hand-crafted, the base example is the main one
                # We still include it multiple times with slight prompt variations
                # to prevent overfitting on exact wording

                if i > 0:
                    # Add a prefix to the user message for variety
                    prefixes = [
                        "",  # no prefix
                        "Please solve this step by step.\n\n",
                        "Analyze the code carefully before making changes.\n\n",
                        "Trace through the code to identify the issue, then fix it.\n\n",
                        "Read the code thoroughly. Identify the root cause. Then provide the fix.\n\n",
                    ]
                    prefix = prefixes[i % len(prefixes)]
                    if prefix:
                        example["messages"][0]["content"] = prefix + example["messages"][0]["content"]

                all_examples.append(example)

            except Exception as e:
                print(f"  WARN: task {task.task_id} example {i} failed: {e}",
                      file=sys.stderr)
                continue

    # Shuffle and split
    rng.shuffle(all_examples)
    n_val = max(20, int(len(all_examples) * val_ratio))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    return train_examples, val_examples


def save_jsonl(examples: list, path: str) -> int:
    """Save examples as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for ex in examples:
            out = {'messages': ex['messages']}
            f.write(json.dumps(out) + '\n')
    return len(examples)


def save_with_metadata(examples: list, path: str) -> int:
    """Save examples with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    return len(examples)


def print_stats(train: list, val: list):
    """Print dataset statistics."""
    print(f"\n{'='*60}", flush=True)
    print(f"IL Long-Horizon SFT Dataset Statistics", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Train: {len(train)} examples", flush=True)
    print(f"  Valid: {len(val)} examples", flush=True)
    print(f"  Total: {len(train) + len(val)} examples", flush=True)

    # By task
    task_counts = Counter(ex['metadata']['task_id'] for ex in train + val)
    print(f"\n  By task ({len(task_counts)} tasks):", flush=True)
    for task_id, count in sorted(task_counts.items()):
        print(f"    {task_id}: {count}", flush=True)

    # By reasoning skill
    skill_counts = Counter(ex['metadata']['reasoning_skill'] for ex in train + val)
    print(f"\n  By reasoning skill:", flush=True)
    for skill, count in sorted(skill_counts.items()):
        print(f"    {skill}: {count}", flush=True)

    # Token estimate
    total_chars = sum(
        len(ex['messages'][0]['content']) + len(ex['messages'][1]['content'])
        for ex in train
    )
    est_tokens = total_chars // 4
    print(f"\n  Train chars: {total_chars:,}", flush=True)
    print(f"  Est. train tokens: {est_tokens:,}", flush=True)
    if train:
        print(f"  Avg tokens/example: {est_tokens // len(train):,}", flush=True)

    # Reasoning length stats
    reasoning_lengths = []
    for ex in train:
        asst = ex['messages'][1]['content']
        if '<reasoning>' in asst and '</reasoning>' in asst:
            start = asst.index('<reasoning>') + len('<reasoning>')
            end = asst.index('</reasoning>')
            reasoning_lengths.append(end - start)

    if reasoning_lengths:
        avg_r = sum(reasoning_lengths) / len(reasoning_lengths)
        max_r = max(reasoning_lengths)
        min_r = min(reasoning_lengths)
        print(f"\n  Reasoning length (chars):", flush=True)
        print(f"    avg: {avg_r:.0f}", flush=True)
        print(f"    min: {min_r}", flush=True)
        print(f"    max: {max_r}", flush=True)

    # Show a sample
    if train:
        sample = train[0]
        print(f"\n  Sample example:", flush=True)
        print(f"    Task: {sample['metadata']['task_id']}", flush=True)
        print(f"    Skill: {sample['metadata']['reasoning_skill']}", flush=True)
        user_preview = sample['messages'][0]['content'][:200].replace('\n', ' ')
        print(f"    User (first 200 chars): {user_preview}...", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Long-Horizon SFT dataset")
    parser.add_argument("--examples-per-task", type=int, default=5,
                        help="examples per task (default: 5)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=str, default="il_long_horizon_data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    n_tasks = len(ALL_TASKS)
    total_target = n_tasks * args.examples_per_task
    print(f"Generating Long-Horizon SFT dataset...", flush=True)
    print(f"  {n_tasks} tasks × {args.examples_per_task} examples = {total_target} target", flush=True)

    train, val = generate_sft_dataset(
        examples_per_task=args.examples_per_task,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    train_path = os.path.join(args.output_dir, 'train.jsonl')
    val_path = os.path.join(args.output_dir, 'valid.jsonl')
    test_path = os.path.join(args.output_dir, 'test.jsonl')

    n_train = save_jsonl(train, train_path)
    n_val = save_jsonl(val, val_path)
    n_test = save_jsonl(val[:max(20, len(val) // 3)], test_path)

    meta_path = os.path.join(args.output_dir, 'train_meta.jsonl')
    save_with_metadata(train, meta_path)

    print_stats(train, val)

    print(f"\n  Saved:", flush=True)
    print(f"    Train: {n_train} -> {train_path}", flush=True)
    print(f"    Valid: {n_val} -> {val_path}", flush=True)
    print(f"    Test:  {n_test} -> {test_path}", flush=True)
    print(f"    Meta:  {n_train} -> {meta_path}", flush=True)


if __name__ == '__main__':
    main()
