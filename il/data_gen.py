"""
IL Training Data Generator

Creates chat-formatted training examples from the IL environment suite.
Three example types to build different aspects of intuition:

1. PREDICTION (60%): "Here are examples. Predict the test output."
   - Teaches: rule inference + application
   - Teacher reasoning: SHORT, direct (2-4 sentences) — teaches efficiency, not looping

2. RULE_INFERENCE (25%): "Here are examples. Describe the transformation rule."
   - Teaches: explicit rule articulation
   - Teacher reasoning: the known rule description

3. RAPID_INTUITION (15%): "Here is ONE example. Predict the test output."
   - Teaches: fast hypothesis formation from minimal data
   - Teacher reasoning: very brief (1-2 sentences)

The CRITICAL design choice: teacher reasoning is SHORT and DIRECT.
The baseline model loops for 4096 tokens without converging. IL training
teaches it to form a hypothesis quickly and commit to an answer.
"""
import json
import random
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from environments import (
    ENVIRONMENT_TYPES, generate_puzzle, generate_dataset,
    grid_to_str, grid_dims, grid_copy
)

# ── Prompt builder (same style as benchmark, but for IL environments) ──

def build_prediction_prompt(puzzle, n_examples=None):
    """Build a prediction prompt from puzzle examples."""
    lines = []
    lines.append("You are an abstract reasoning system. You will be given example input-output grid pairs that demonstrate a transformation rule. You must infer the rule from the examples and apply it to the test input.")
    lines.append("")
    lines.append("Grids are 2D arrays of integers 0-9. 0 represents empty/background.")
    lines.append("")
    lines.append("=== EXAMPLES ===")
    lines.append("")
    examples = puzzle['examples'][:n_examples] if n_examples else puzzle['examples']
    for i, ex in enumerate(examples):
        lines.append(f"--- Example {i+1} ---")
        lines.append("Input:")
        lines.append(grid_to_str(ex['input']))
        lines.append("Output:")
        lines.append(grid_to_str(ex['output']))
        lines.append("")
    lines.append("=== TEST ===")
    lines.append("")
    lines.append("Input:")
    lines.append(grid_to_str(puzzle['test_input']))
    lines.append("")
    lines.append("Apply the transformation rule and output ONLY the resulting grid as a 2D array.")
    return '\n'.join(lines)


def build_rule_inference_prompt(puzzle, n_examples=None):
    """Build a rule-description prompt."""
    lines = []
    lines.append("You are an abstract reasoning system. You will be given example input-output grid pairs that demonstrate a transformation rule. Describe the rule in one or two sentences.")
    lines.append("")
    lines.append("Grids are 2D arrays of integers 0-9. 0 represents empty/background.")
    lines.append("")
    lines.append("=== EXAMPLES ===")
    lines.append("")
    examples = puzzle['examples'][:n_examples] if n_examples else puzzle['examples']
    for i, ex in enumerate(examples):
        lines.append(f"--- Example {i+1} ---")
        lines.append("Input:")
        lines.append(grid_to_str(ex['input']))
        lines.append("Output:")
        lines.append(grid_to_str(ex['output']))
        lines.append("")
    lines.append("What transformation rule maps the inputs to the outputs? Describe it concisely.")
    return '\n'.join(lines)


def build_rapid_prompt(puzzle):
    """Build a rapid-intuition prompt with only 1 example."""
    return build_prediction_prompt(puzzle, n_examples=1)


# ── Teacher reasoning generators (SHORT and DIRECT) ──

def make_teacher_reasoning(puzzle, mode='prediction'):
    """Generate short, direct teacher reasoning for a puzzle.

    This is the KEY to IL: we teach the model to reason efficiently,
    not to loop. The reasoning is 2-4 sentences max.
    """
    rule = puzzle['rule']
    test_in = puzzle['test_input']
    test_out = puzzle['test_output']
    h, w = grid_dims(test_in)
    oh, ow = grid_dims(test_out)

    if mode == 'rule_inference':
        # Just state the rule directly
        return f"The rule is: {rule}"

    # For prediction and rapid modes:
    # Brief reasoning + answer
    reasoning = f"Looking at the examples, the rule is: {rule} "
    if (h, w) != (oh, ow):
        reasoning += f"The output grid has dimensions {oh}x{ow}. "
    reasoning += f"Applying this to the test input:"

    answer = grid_to_str(test_out)
    return f"{reasoning}\n\n{answer}"


# ── Dataset generation ──

def generate_training_data(n_per_type=25, seed=12345, val_ratio=0.1):
    """Generate the full IL training dataset in chat format.

    Returns (train_examples, val_examples) where each example is a dict
    with 'messages' key for mlx-lm LoRA training.
    """
    rng = random.Random(seed)
    all_examples = []

    for et in ENVIRONMENT_TYPES:
        for i in range(n_per_type):
            puzzle = generate_puzzle(et, rng)
            puzzle['id'] = f"{et['name']}_{i+1}"

            # Decide example type
            roll = rng.random()
            if roll < 0.60:
                mode = 'prediction'
                prompt = build_prediction_prompt(puzzle)
            elif roll < 0.85:
                mode = 'rule_inference'
                prompt = build_rule_inference_prompt(puzzle)
            else:
                mode = 'rapid'
                prompt = build_rapid_prompt(puzzle)

            teacher = make_teacher_reasoning(puzzle, mode)

            example = {
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': teacher},
                ],
                'metadata': {
                    'env_type': et['name'],
                    'mode': mode,
                    'puzzle_id': puzzle['id'],
                }
            }
            all_examples.append(example)

    # Shuffle and split
    rng.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * val_ratio))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    return train_examples, val_examples


def save_jsonl(examples, path):
    """Save examples as JSONL (one JSON object per line)."""
    with open(path, 'w') as f:
        for ex in examples:
            # Strip metadata for the saved file (mlx-lm only needs messages)
            out = {'messages': ex['messages']}
            f.write(json.dumps(out) + '\n')
    return len(examples)


def main():
    DATA_DIR = "/Users/kzrr/ ILresearch /il_data"

    print("Generating IL training data...", flush=True)
    train, val = generate_training_data(n_per_type=25, seed=12345)

    train_path = os.path.join(DATA_DIR, 'train.jsonl')
    val_path = os.path.join(DATA_DIR, 'valid.jsonl')
    # Also create a small test set for LoRA evaluation
    test_path = os.path.join(DATA_DIR, 'test.jsonl')

    n_train = save_jsonl(train, train_path)
    n_val = save_jsonl(val, val_path)
    # Use first 20% of val as test
    n_test = save_jsonl(val[:max(5, len(val)//3)], test_path)

    print(f"\nIL Training Data Generated:", flush=True)
    print(f"  Train: {n_train} examples -> {train_path}", flush=True)
    print(f"  Valid: {n_val} examples  -> {val_path}", flush=True)
    print(f"  Test:  {n_test} examples  -> {test_path}", flush=True)

    # Stats by mode
    from collections import Counter
    modes = Counter(ex['metadata']['mode'] for ex in train + val)
    types = Counter(ex['metadata']['env_type'] for ex in train + val)
    print(f"\n  By mode: {dict(modes)}", flush=True)
    print(f"  By env type: {len(types)} types, {dict(list(types.items())[:5])}...", flush=True)

    # Show a sample example
    print(f"\n  Sample training example:", flush=True)
    sample = train[0]
    print(f"    User (first 200 chars): {sample['messages'][0]['content'][:200]}...", flush=True)
    print(f"    Assistant (first 200 chars): {sample['messages'][1]['content'][:200]}...", flush=True)

    # Estimate token count
    total_chars = sum(len(ex['messages'][0]['content']) + len(ex['messages'][1]['content']) for ex in train)
    est_tokens = total_chars // 4  # rough estimate
    print(f"\n  Total chars: {total_chars:,} | Est. tokens: {est_tokens:,}", flush=True)
    print(f"  Avg tokens/example: {est_tokens // n_train}", flush=True)


if __name__ == '__main__':
    main()
