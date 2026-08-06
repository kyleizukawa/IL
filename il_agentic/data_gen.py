"""
SFT Data Generator for Agentic Coding Environments.

Generates chat-formatted training examples from all 15 agentic environments.
Each example includes:
- User message: the task description (codebase + instructions)
- Assistant message: <reasoning>teacher trace</reasoning><answer>solution code</answer>

The teacher reasoning is designed to kill laziness:
- Demonstrates line-by-line code reading
- Traces execution paths
- Identifies specific issues with evidence
- Explains fixes with reasoning
- Does NOT jump to conclusions or pattern-match

Example types per environment:
- 40% thorough_analysis: full step-by-step trace (medium/hard difficulty)
- 30% concise_fix: shorter but still evidence-based (easy/medium difficulty)
- 20% edge_case_focus: focuses on edge cases and failure modes
- 10% rapid_fix: quick identification + fix (easy difficulty only)
"""
import json
import random
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from il_agentic import ALL_ENVIRONMENTS, ENV_REGISTRY


def generate_sft_dataset(
    n_per_env: int = 60,
    seed: int = 12345,
    val_ratio: float = 0.1,
    difficulty_distribution: dict | None = None,
) -> tuple[list, list]:
    """Generate the full SFT dataset across all 15 environments.

    Args:
        n_per_env: examples per environment
        seed: random seed
        val_ratio: fraction of examples for validation
        difficulty_distribution: optional override for difficulty mix

    Returns: (train_examples, val_examples)
    """
    rng = random.Random(seed)

    # Default difficulty distribution: more medium, some easy and hard
    diff_dist = difficulty_distribution or {
        "easy": 0.25,
        "medium": 0.50,
        "hard": 0.25,
    }

    all_examples = []

    for env_class in ALL_ENVIRONMENTS:
        env = env_class()
        for i in range(n_per_env):
            # Pick difficulty
            roll = rng.random()
            if roll < diff_dist["easy"]:
                difficulty = "easy"
            elif roll < diff_dist["easy"] + diff_dist["medium"]:
                difficulty = "medium"
            else:
                difficulty = "hard"

            # Generate the SFT example
            try:
                example = env.build_sft_example(rng, difficulty)
                example["metadata"]["example_idx"] = i
                all_examples.append(example)
            except Exception as e:
                # Skip failed generations but log
                print(f"  WARN: {env.name} example {i} failed: {e}", file=sys.stderr)
                continue

    # Shuffle and split
    rng.shuffle(all_examples)
    n_val = max(15, int(len(all_examples) * val_ratio))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    return train_examples, val_examples


def save_jsonl(examples: list, path: str) -> int:
    """Save examples as JSONL (one JSON object per line)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for ex in examples:
            out = {'messages': ex['messages']}
            f.write(json.dumps(out) + '\n')
    return len(examples)


def save_with_metadata(examples: list, path: str) -> int:
    """Save examples with metadata (for analysis)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    return len(examples)


def print_stats(train: list, val: list):
    """Print dataset statistics."""
    print(f"\n{'='*60}", flush=True)
    print(f"IL Agentic SFT Dataset Statistics", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Train: {len(train)} examples", flush=True)
    print(f"  Valid: {len(val)} examples", flush=True)
    print(f"  Total: {len(train) + len(val)} examples", flush=True)

    # By environment
    env_counts = Counter(ex['metadata']['env_name'] for ex in train + val)
    print(f"\n  By environment ({len(env_counts)} envs):", flush=True)
    for name, count in sorted(env_counts.items()):
        print(f"    {name}: {count}", flush=True)

    # By difficulty
    diff_counts = Counter(ex['metadata']['difficulty'] for ex in train + val)
    print(f"\n  By difficulty:", flush=True)
    for diff, count in sorted(diff_counts.items()):
        print(f"    {diff}: {count}", flush=True)

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

    # Show a sample
    if train:
        sample = train[0]
        print(f"\n  Sample example:", flush=True)
        print(f"    Env: {sample['metadata']['env_name']}", flush=True)
        print(f"    Difficulty: {sample['metadata']['difficulty']}", flush=True)
        user_preview = sample['messages'][0]['content'][:300].replace('\n', ' ')
        asst_preview = sample['messages'][1]['content'][:300].replace('\n', ' ')
        print(f"    User (first 300 chars): {user_preview}...", flush=True)
        print(f"    Assistant (first 300 chars): {asst_preview}...", flush=True)


def main():
    """Generate and save the SFT dataset."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate IL Agentic SFT dataset")
    parser.add_argument("--n-per-env", type=int, default=60,
                        help="examples per environment (default: 60)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=str, default="il_agentic_data",
                        help="output directory")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Generating IL Agentic SFT dataset...", flush=True)
    print(f"  {len(ALL_ENVIRONMENTS)} environments × {args.n_per_env} examples = "
          f"{len(ALL_ENVIRONMENTS) * args.n_per_env} target examples", flush=True)

    train, val = generate_sft_dataset(
        n_per_env=args.n_per_env,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    # Save
    train_path = os.path.join(args.output_dir, 'train.jsonl')
    val_path = os.path.join(args.output_dir, 'valid.jsonl')
    test_path = os.path.join(args.output_dir, 'test.jsonl')

    n_train = save_jsonl(train, train_path)
    n_val = save_jsonl(val, val_path)
    n_test = save_jsonl(val[:max(15, len(val) // 3)], test_path)

    # Also save with metadata for analysis
    meta_train_path = os.path.join(args.output_dir, 'train_meta.jsonl')
    save_with_metadata(train, meta_train_path)

    print_stats(train, val)

    print(f"\n  Saved:", flush=True)
    print(f"    Train: {n_train} -> {train_path}", flush=True)
    print(f"    Valid: {n_val} -> {val_path}", flush=True)
    print(f"    Test:  {n_test} -> {test_path}", flush=True)
    print(f"    Meta:  {n_train} -> {meta_train_path}", flush=True)


if __name__ == '__main__':
    main()
