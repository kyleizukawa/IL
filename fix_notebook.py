#!/usr/bin/env python3
"""Fixes the inference, prompt, and thinking-extraction cells in the notebook."""
import json

NB_PATH = "/Users/kzrr/ ILresearch /mini_arc_agi3_benchmark.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

def get_src(cell):
    return ''.join(cell['source'])

def set_src(cell, new_src):
    cell['source'] = [new_src]

# ── Fix 1: Prompt builder — remove the trailing "midt" line ──
# The model should generate its own midt tag via the chat template.
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = get_src(cell)
    if 'def build_prompt' not in src:
        continue

    new_src = '''def build_prompt(puzzle):
    """Build a zero-instruction prompt from puzzle examples.

    The prompt gives NO description of the transformation rule.
    The model must infer the rule purely from example input→output pairs.
    """
    lines = []
    lines.append("You are an abstract reasoning system. You will be given example input-output grid pairs that demonstrate a transformation rule. You must infer the rule from the examples and apply it to the test input.")
    lines.append("")
    lines.append("Grids are 2D arrays of integers 0-9. 0 represents empty/background.")
    lines.append("")
    lines.append("=== EXAMPLES ===")
    lines.append("")
    for i, ex in enumerate(puzzle['examples']):
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
    lines.append("Think step by step about what rule transforms the inputs into the outputs, then apply it to the test input.")
    return '\\n'.join(lines)

# Show an example prompt
sample_prompt = build_prompt(dataset[0])
print(f"Prompt length: {len(sample_prompt)} chars")
print("=" * 60)
print(sample_prompt[:1500])
print("...")'''

    set_src(cell, new_src)
    print(f"  ✓ Fixed prompt builder (cell {i})")

# ── Fix 2: Inference & parsing cell — use chat template, fix generate, fix extract_thinking ──
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = get_src(cell)
    if 'def parse_grid' not in src or 'def run_inference' not in src:
        continue

    new_src = '''def parse_grid(text):
    """Extract a 2D integer grid from model output text.

    Uses bracket-depth matching to find the last complete 2D array,
    then falls back to individual row extraction.
    """
    # Strategy 1: bracket-depth scan for 2D arrays ([[...]])
    last_2d = None
    i = 0
    while i < len(text):
        if text[i] == '[':
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '[':
                    depth += 1
                elif text[j] == ']':
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j+1]
                        # Must be 2D: contains at least one nested [ ]
                        if candidate.count('[') >= 3:
                            last_2d = candidate
                        i = j
                        break
            else:
                break
        i += 1

    if last_2d:
        rows = re.findall(r'\\[\\s*([\\d\\s,]+)\\]', last_2d)
        grid = []
        for row_str in rows:
            nums = re.findall(r'\\d+', row_str)
            if nums:
                grid.append([int(x) for x in nums])
        if grid:
            return grid

    # Strategy 2: collect individual row arrays [d,d,...]
    row_matches = re.findall(r'\\[\\s*\\d+[\\s,\\d]*\\]', text)
    if row_matches:
        grid = []
        for row_str in row_matches:
            nums = re.findall(r'\\d+', row_str)
            if nums:
                grid.append([int(x) for x in nums])
        if grid:
            return grid

    return None


def extract_thinking(text):
    """Extract the reasoning from midt...</think> tags.

    DeepSeek-R1-Distill generates its own midt tag at the start of
    the response, followed by reasoning, then </think>, then the answer.
    """
    # Standard case: model generated midt...reasoning...</think>
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If there's a midt but no closing tag, take everything after midt
    # up to the first grid-looking bracket
    match2 = re.search(r'<think>(.*?)(?:\\[|$)', text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    # If no midt tag at all, there's no reasoning
    return ""


def run_inference(puzzle, max_new_tokens=MAX_NEW_TOKENS):
    """Run model inference on a single puzzle.

    Uses the tokenizer's chat template so the model generates proper
    midt...</think> reasoning blocks before its final answer.
    """
    user_prompt = build_prompt(puzzle)

    # Use the chat template — this is the correct way to prompt
    # DeepSeek-R1-Distill-Qwen models. The template adds the proper
    # special tokens and generation prompt, causing the model to
    # naturally produce <think>reasoning</think>answer.
    messages = [{"role": "user", "content": user_prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy decoding for determinism
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (exclude the prompt)
    input_len = inputs['input_ids'].shape[1]
    full_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Parse the model's output
    thinking = extract_thinking(full_text)
    predicted_grid = parse_grid(full_text)

    return {
        'raw_output': full_text,
        'thinking': thinking,
        'predicted': predicted_grid,
        'thinking_length': len(thinking),
    }


# Quick test on first puzzle
print("Testing inference on first puzzle...")
test_result = run_inference(dataset[0], max_new_tokens=4096)
print(f"Thinking length: {test_result['thinking_length']} chars")
print(f"Predicted grid: {test_result['predicted']}")
print(f"Expected grid dims: {grid_dims(dataset[0]['test_output'])}")
if test_result['predicted']:
    print(f"Predicted grid dims: {grid_dims(test_result['predicted'])}")
print(f"\\n--- Thinking excerpt (first 800 chars) ---")
print(test_result['thinking'][:800])
print("...")
print(f"\\n--- Raw output excerpt (first 400 chars) ---")
print(test_result['raw_output'][:400])'''

    set_src(cell, new_src)
    print(f"  ✓ Fixed inference & parsing (cell {i})")

# ── Fix 3: Update the markdown cell about prompt construction to reflect changes ──
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'markdown':
        continue
    src = get_src(cell)
    if '## 5. Prompt Construction' not in src:
        continue

    new_src = '''## 5. Prompt Construction

The prompt gives the model **no instructions** about the transformation rule.
It only shows example input→output pairs and asks for the test output.

This forces the model to:
1. Infer the rule from examples
2. Reason through the rule step-by-step (using `<think>` blocks — DeepSeek-R1's native reasoning format)
3. Apply the rule to the test input

**Important**: We use the tokenizer's `apply_chat_template` to format the prompt correctly for DeepSeek-R1-Distill-Qwen. This ensures the model generates its native `<think>reasoning</think>answer` structure, activating its full reasoning capabilities. We do **not** pre-fill the `<think>` tag — the model generates it naturally.'''

    set_src(cell, new_src)
    print(f"  ✓ Fixed prompt construction markdown (cell {i})")

# ── Fix 4: Update the config cell to remove TEMPERATURE/TOP_P references that cause confusion ──
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = get_src(cell)
    if 'MODEL_NAME' not in src or 'TEMPERATURE' not in src:
        continue

    new_src = '''import torch
import random
import json
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque, Counter
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Configuration ──
MODEL_NAME           = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_NEW_TOKENS       = 8192      # generous budget for reasoning
NUM_EXAMPLES         = 3         # example pairs per puzzle
NUM_PUZZLES_PER_TYPE = 3         # puzzles per type
# Greedy decoding (do_sample=False) for deterministic, reproducible outputs.
# We do NOT pass temperature/top_p since they are ignored (and cause warnings)
# when do_sample=False.

print("Configuration loaded.")
print(f"Model: {MODEL_NAME}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print(f"Puzzles per type: {NUM_PUZZLES_PER_TYPE}")
print(f"Total puzzle types: 12 → {12 * NUM_PUZZLES_PER_TYPE} puzzles total")
print(f"Decoding: greedy (do_sample=False)")'''

    set_src(cell, new_src)
    print(f"  ✓ Fixed config cell (cell {i})")

# Write the fixed notebook
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\nNotebook fixed and saved: {NB_PATH}")
