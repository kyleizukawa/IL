#!/usr/bin/env python3
"""Generate improved SFT dataset with chain-of-thought reasoning traces.

Key improvements over v1:
- Uses <midt>...</midt> think tags so the model learns to reason then close
- Step-by-step reasoning: analyze examples → identify rule → apply to test
- 1200+ examples (60 per type × 20 types)
- Mix of detailed CoT, brief CoT, rule inference, and rapid intuition
- All reasoning traces teach the model to converge (not loop)
"""
import json, random, sys, os
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'il'))
from environments import (
    ENVIRONMENT_TYPES, generate_puzzle, generate_dataset,
    grid_to_str, grid_dims, grid_copy, empty_grid
)

THINK_OPEN = "midt"
# Build think close tag from char codes to avoid file-write transport stripping
THINK_CLOSE = chr(60) + chr(47) + 'midt' + chr(62)  # </midt>

def build_prediction_prompt(puzzle, n_examples=None):
    lines = [
        "You are an abstract reasoning system. You will be given example "
        "input-output grid pairs that demonstrate a transformation rule. "
        "You must infer the rule from the examples and apply it to the test input.",
        "",
        "Grids are 2D arrays of integers 0-9. 0 represents empty/background.",
        "", "=== EXAMPLES ===", ""
    ]
    examples = puzzle['examples'][:n_examples] if n_examples else puzzle['examples']
    for i, ex in enumerate(examples):
        lines += [f"--- Example {i+1} ---", "Input:", grid_to_str(ex['input']),
                  "Output:", grid_to_str(ex['output']), ""]
    lines += ["=== TEST ===", "", "Input:", grid_to_str(puzzle['test_input']), "",
              "Apply the transformation rule and output ONLY the resulting grid as a 2D array."]
    return '\n'.join(lines)

def build_rule_inference_prompt(puzzle, n_examples=None):
    lines = [
        "You are an abstract reasoning system. You will be given example "
        "input-output grid pairs that demonstrate a transformation rule. "
        "Describe the rule in one or two sentences.",
        "",
        "Grids are 2D arrays of integers 0-9. 0 represents empty/background.",
        "", "=== EXAMPLES ===", ""
    ]
    examples = puzzle['examples'][:n_examples] if n_examples else puzzle['examples']
    for i, ex in enumerate(examples):
        lines += [f"--- Example {i+1} ---", "Input:", grid_to_str(ex['input']),
                  "Output:", grid_to_str(ex['output']), ""]
    lines += ["What transformation rule maps the inputs to the outputs? Describe it concisely."]
    return '\n'.join(lines)

def build_rapid_prompt(puzzle):
    return build_prediction_prompt(puzzle, n_examples=1)

# ── CoT reasoning generators ──

def make_detailed_cot(puzzle):
    """Detailed chain-of-thought: analyze examples, state rule, apply to test."""
    rule = puzzle['rule']
    test_in = puzzle['test_input']
    test_out = puzzle['test_output']
    h, w = grid_dims(test_in)
    oh, ow = grid_dims(test_out)
    examples = puzzle['examples']

    reasoning = f"Let me analyze the examples to find the pattern.\n\n"
    reasoning += f"Looking at Example 1: the input is a {h}x{w} grid and the output is a {grid_dims(examples[0]['output'])[0]}x{grid_dims(examples[0]['output'])[1]} grid. "
    reasoning += f"The transformation appears to be: {rule}\n\n"

    if len(examples) >= 2:
        eh2, ew2 = grid_dims(examples[1]['output'])
        reasoning += f"Example 2 confirms this: same rule applied to a different grid produces a {eh2}x{ew2} output.\n\n"

    reasoning += f"Now applying this rule to the test input ({h}x{w} grid):\n"
    if (h, w) != (oh, ow):
        reasoning += f"The output grid will have dimensions {oh}x{ow}.\n"
    reasoning += f"Applying the transformation to each cell:\n"

    answer = grid_to_str(test_out)
    return f"{reasoning}\n{THINK_CLOSE}\n{answer}"

def make_brief_cot(puzzle):
    """Brief chain-of-thought: state rule + apply, with think tags."""
    rule = puzzle['rule']
    test_out = puzzle['test_output']
    oh, ow = grid_dims(test_out)

    reasoning = f"The pattern is: {rule} "
    if (grid_dims(puzzle['test_input'])[0], grid_dims(puzzle['test_input'])[1]) != (oh, ow):
        reasoning += f"The output is {oh}x{ow}. "
    reasoning += f"Applying to the test input:"

    answer = grid_to_str(test_out)
    return f"{reasoning}\n{THINK_CLOSE}\n{answer}"

def make_rule_inference_response(puzzle):
    """Rule inference: just describe the rule, with think tags."""
    rule = puzzle['rule']
    return f"Analyzing the examples, the transformation rule is clear.\n{THINK_CLOSE}\nThe rule is: {rule}"

def make_rapid_response(puzzle):
    """Rapid intuition: very brief reasoning from 1 example."""
    rule = puzzle['rule']
    test_out = puzzle['test_output']
    oh, ow = grid_dims(test_out)

    reasoning = f"From the single example, the rule is: {rule} Output is {oh}x{ow}."
    answer = grid_to_str(test_out)
    return f"{reasoning}\n{THINK_CLOSE}\n{answer}"

def generate_training_data(n_per_type=60, seed=12345, val_ratio=0.08):
    rng = random.Random(seed)
    all_examples = []

    for et in ENVIRONMENT_TYPES:
        for i in range(n_per_type):
            puzzle = generate_puzzle(et, rng)
            puzzle['id'] = f"{et['name']}_{i+1}"

            roll = rng.random()
            if roll < 0.45:
                mode = 'detailed_cot'
                prompt = build_prediction_prompt(puzzle)
                teacher = make_detailed_cot(puzzle)
            elif roll < 0.70:
                mode = 'brief_cot'
                prompt = build_prediction_prompt(puzzle)
                teacher = make_brief_cot(puzzle)
            elif roll < 0.85:
                mode = 'rule_inference'
                prompt = build_rule_inference_prompt(puzzle)
                teacher = make_rule_inference_response(puzzle)
            else:
                mode = 'rapid'
                prompt = build_rapid_prompt(puzzle)
                teacher = make_rapid_response(puzzle)

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

    rng.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * val_ratio))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]
    # Test set from val
    test_examples = val_examples[:max(10, len(val_examples)//3)]
    return train_examples, val_examples, test_examples

def save_jsonl(examples, path):
    with open(path, 'w') as f:
        for ex in examples:
            out = {'messages': ex['messages']}
            f.write(json.dumps(out) + '\n')
    return len(examples)

def main():
    OUT_DIR = os.path.join(os.path.dirname(__file__), 'il_dataset_v2')
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating improved IL training data (v2)...", flush=True)
    train, val, test = generate_training_data(n_per_type=60, seed=12345)

    n_train = save_jsonl(train, os.path.join(OUT_DIR, 'train.jsonl'))
    n_val = save_jsonl(val, os.path.join(OUT_DIR, 'valid.jsonl'))
    n_test = save_jsonl(test, os.path.join(OUT_DIR, 'test.jsonl'))

    print(f"\nIL Training Data v2 Generated:", flush=True)
    print(f"  Train: {n_train} examples", flush=True)
    print(f"  Valid: {n_val} examples", flush=True)
    print(f"  Test:  {n_test} examples", flush=True)

    from collections import Counter
    modes = Counter(ex['metadata']['mode'] for ex in train + val)
    print(f"\n  By mode: {dict(modes)}", flush=True)

    # Stats
    assistant_lens = [len(ex['messages'][1]['content']) for ex in train]
    print(f"  Assistant content: min={min(assistant_lens)}, max={max(assistant_lens)}, "
          f"avg={sum(assistant_lens)/len(assistant_lens):.0f} chars", flush=True)
    has_think = sum(1 for ex in train if THINK_CLOSE in ex['messages'][1]['content'])
    print(f"  With think tags: {has_think}/{len(train)}", flush=True)

    # Show samples
    for mode_name in ['detailed_cot', 'brief_cot', 'rule_inference', 'rapid']:
        sample = next(ex for ex in train if ex['metadata']['mode'] == mode_name)
        print(f"\n  --- {mode_name} sample ---")
        print(f"    Assistant (first 300 chars): {sample['messages'][1]['content'][:300]}")

    total_chars = sum(len(ex['messages'][0]['content']) + len(ex['messages'][1]['content']) for ex in train)
    est_tokens = total_chars // 4
    print(f"\n  Total chars: {total_chars:,} | Est. tokens: {est_tokens:,}", flush=True)
    print(f"  Avg tokens/example: {est_tokens // n_train}", flush=True)

if __name__ == '__main__':
    main()
