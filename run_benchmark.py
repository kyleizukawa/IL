#!/usr/bin/env python3
"""Run the Mini ARC-AGI 3 benchmark locally on Apple Silicon (MPS).

Adapted from mini_arc_agi3_benchmark.ipynb:
- Uses MPS instead of device_map="auto"
- Single-puzzle smoke test first (verifies the parsing bug fix)
- Then full 36-puzzle benchmark if --full is passed
"""
import sys
import json
import os

# ── Extract code cells from the notebook ──
NB_PATH = "/Users/kzrr/ ILresearch /mini_arc_agi3_benchmark.ipynb"
with open(NB_PATH) as f:
    nb = json.load(f)

code_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        code_cells.append(src)

print(f"Loaded {len(code_cells)} code cells from notebook")

# ── Execute cells 0..N (setup, generators, dataset, prompt, parsing) ──
# Skip cell 0 (pip install — already installed)
# Skip cell 1 (the model-loading cell) — we replace it with MPS version
# We exec cells: index 0=pip, 1=imports/config, 2=model load, 3=grid utils,
#                4=puzzle gen 1-6, 5=puzzle gen 7-12, 6=registry/dataset,
#                7=prompt builder, 8=inference/parsing

# Build a single namespace
ns = {'__name__': '__main__'}

# Cell 0: pip install — skip
# Cell 1: imports + config
print("\n=== Executing imports & config ===")
exec(code_cells[1], ns)

# Cell 2: model load — REPLACE with MPS version
print("\n=== Loading model on MPS ===")
model_load_code = '''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from local directory (avoids re-downloading)
MODEL_PATH = "/Users/kzrr/ ILresearch /model"
print(f"Loading model from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,   # float16 works well on MPS; bfloat16 support is limited
)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded on {device}")
print(f"  Parameters: {n_params / 1e9:.2f}B")
print(f"  Dtype: {next(model.parameters()).dtype}")
'''
exec(model_load_code, ns)

# Cell 3: grid utilities
print("\n=== Executing grid utilities ===")
exec(code_cells[3], ns)

# Cell 4: puzzle generators 1-6
print("\n=== Executing puzzle generators 1-6 ===")
exec(code_cells[4], ns)

# Cell 5: puzzle generators 7-12
print("\n=== Executing puzzle generators 7-12 ===")
exec(code_cells[5], ns)

# Cell 6: registry & dataset
print("\n=== Executing registry & dataset ===")
exec(code_cells[6], ns)

# Cell 7: prompt builder
print("\n=== Executing prompt builder ===")
exec(code_cells[7], ns)

# Cell 8: inference & parsing (THE FIXED CELL)
print("\n=== Executing inference & parsing (fixed) ===")
exec(code_cells[8], ns)

# ── Smoke test: ONE puzzle with a small token budget ──
print("\n" + "=" * 70)
print("SMOKE TEST: single puzzle, max_new_tokens=2048")
print("=" * 70)

smoke_code = '''
import time
t0 = time.time()
test_result = run_inference(dataset[0], max_new_tokens=2048)
elapsed = time.time() - t0
print(f"Time: {elapsed:.1f}s")
print(f"Thinking length: {test_result[\'thinking_length\']} chars")
print(f"Predicted grid: {test_result[\'predicted\']}")
print(f"Expected grid dims: {grid_dims(dataset[0][\'test_output\'])}")
if test_result[\'predicted\']:
    print(f"Predicted grid dims: {grid_dims(test_result[\'predicted\'])}")
else:
    print("Predicted grid dims: None (parse_grid returned None)")
print(f"\\n--- Thinking excerpt (first 600 chars) ---")
print(test_result[\'thinking\'][:600])
print("...")
print(f"\\n--- Raw output excerpt (first 600 chars) ---")
print(test_result[\'raw_output\'][:600])
print("...")
print(f"\\n--- Raw output excerpt (LAST 600 chars) ---")
print(test_result[\'raw_output\'][-600:])

# Bug-fix verification
ok_thinking = test_result[\'thinking_length\'] > 0
ok_predicted = test_result[\'predicted\'] is not None
print(f"\\n=== BUG FIX CHECK ===")
print(f"  thinking_length > 0 : {\'PASS\' if ok_thinking else \'FAIL\'}")
print(f"  predicted is not None: {\'PASS\' if ok_predicted else \'FAIL\'}")
'''
exec(smoke_code, ns)

# ── Full benchmark only if --full passed ──
if "--full" in sys.argv:
    print("\n" + "=" * 70)
    print("FULL BENCHMARK: 36 puzzles")
    print("=" * 70)
    exec(code_cells[10], ns)  # evaluation functions
    exec(code_cells[12], ns)  # run benchmark loop
    exec(code_cells[14], ns)  # aggregate results
    exec(code_cells[15], ns)  # plots
    exec(code_cells[16], ns)  # detailed table
    exec(code_cells[17], ns)  # save json
    print("\nFull benchmark complete. Results in benchmark_results.json")
else:
    print("\n(Smoke test only. Pass --full to run all 36 puzzles.)")
