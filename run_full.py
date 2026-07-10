#!/usr/bin/env python3
"""Run the Mini ARC-AGI 3 benchmark with mlx-lm (4-bit quantized) on Apple Silicon.

Why mlx-lm instead of transformers+MPS:
- The 1.78B float16 model (~3.3 GB) OOM-kills on 8 GB RAM with transformers+MPS.
- mlx-lm 4-bit quantization: ~1.9 GB peak memory, ~23 tok/s (3.5x faster than
  transformers+MPS, and fits comfortably in 8 GB).
- The 4-bit quantized model is created once via mlx_lm.convert (see mlx_smoke.py
  or this script's auto-convert) into model_mlx_4bit/.

All puzzle generators, grid utilities, prompt building, and evaluation logic are
reused from the notebook (exec'd in). Only model loading + inference are replaced.

Saves results to benchmark_results.json AFTER EVERY PUZZLE (interrupt-safe).

Usage:
    python run_full.py                         # 36 puzzles, 2048 tokens
    python run_full.py --max-tokens 4096
    python run_full.py --smoke                 # single-puzzle sanity check
    python run_full.py --types gravity_sort    # subset of puzzle types
    python run_full.py --resume                # skip puzzles already in results
"""
import sys, json, os, time, re, argparse

# ── Args ──
ap = argparse.ArgumentParser()
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--types", nargs="*", default=None)
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()

HERE = "/Users/kzrr/ ILresearch "
NB_PATH = os.path.join(HERE, "mini_arc_agi3_benchmark.ipynb")
MODEL_PATH = os.path.join(HERE, "model")
MLX_PATH = os.path.join(HERE, "model_mlx_4bit")
RESULTS_PATH = os.path.join(HERE, "benchmark_results.json")
DATASET_PATH = os.path.join(HERE, "benchmark_dataset.json")
PLOT_PATH = os.path.join(HERE, "benchmark_results.png")

# ── Load notebook code cells ──
with open(NB_PATH) as f:
    nb = json.load(f)
code_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        code_cells.append(''.join(cell['source']))
print(f"Loaded {len(code_cells)} code cells from notebook", flush=True)

# Code-cell index map (verified):
#   1: imports + config | 3: grid utils | 4: gen 1-6 | 5: gen 7-12
#   6: registry & dataset | 7: prompt builder | 8: inference & parsing
#   9: evaluation functions

ns = {'__name__': '__main__'}

# Cell 1: imports + config (imports torch/transformers/numpy/etc., sets SEED, config vars)
print("=== Imports & config ===", flush=True)
exec(code_cells[1], ns)
ns['MAX_NEW_TOKENS'] = args.max_tokens
print(f"MAX_NEW_TOKENS = {ns['MAX_NEW_TOKENS']}", flush=True)

# Cells 3-7: utilities, generators, dataset, prompt builder
print("=== Grid utilities ===", flush=True);  exec(code_cells[3], ns)
print("=== Generators 1-6 ===", flush=True);  exec(code_cells[4], ns)
print("=== Generators 7-12 ===", flush=True); exec(code_cells[5], ns)
print("=== Registry & dataset ===", flush=True); exec(code_cells[6], ns)
print("=== Prompt builder ===", flush=True);  exec(code_cells[7], ns)

# Cell 8: keep ONLY parse_grid (format-agnostic 2D-array extractor).
# We skip extract_thinking/run_inference (transformers-based) and the test block.
print("=== parse_grid (from notebook) ===", flush=True)
cell8 = code_cells[8]
start = cell8.index('def parse_grid')
end = cell8.index('def extract_thinking')
exec(cell8[start:end], ns)
parse_grid = ns['parse_grid']

# Cell 9: evaluation functions (pure python: grids_equal, shape_match, cell_accuracy, evaluate_result)
print("=== Evaluation functions ===", flush=True)
exec(code_cells[9], ns)
evaluate_result = ns['evaluate_result']
grid_dims = ns['grid_dims']

# ── Load model with mlx-lm (4-bit) ──
from mlx_lm import load, generate, convert

if not os.path.isdir(MLX_PATH):
    print(f"\nConverting {MODEL_PATH} -> {MLX_PATH} (4-bit, one-time)...", flush=True)
    t0 = time.time()
    convert(hf_path=MODEL_PATH, mlx_path=MLX_PATH, quantize=True, q_bits=4)
    print(f"Converted in {time.time()-t0:.1f}s", flush=True)

print(f"\nLoading mlx model from {MLX_PATH} ...", flush=True)
t0 = time.time()
mlx_model, mlx_tokenizer = load(MLX_PATH)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

# ── mlx-lm inference (two-stage: bounded reasoning + forced answer) ──
# Replaces the notebook's transformers run_inference.
# NOTE: the closing think tag is built from char codes because the literal
# sequence (less-than slash) gets stripped by the file-write transport.
THINK_OPEN = 'midt'
THINK_CLOSE = chr(60) + chr(47) + 'midt' + chr(62)   # the closing reasoning tag
EOS = '<｜end▁of▁sentence｜>'

# Two-stage generation:
#   Stage 1: let the model reason up to `reason_tokens`. If it emits the closing
#            think tag + an answer + EOS on its own, we're done (single call).
#   Stage 2: if it hits the reasoning cap WITHOUT emitting the closing tag
#            (R1-1.5B often loops on hard ARC-style puzzles and never converges),
#            we FORCE the closing tag by appending it to the reasoning, then
#            generate a short answer. This guarantees every puzzle gets a real
#            answer attempt under bounded compute (standard for benchmarking
#            reasoning models with a token budget).
def run_inference_mlx(puzzle, reason_tokens=None, answer_tokens=512):
    if reason_tokens is None:
        reason_tokens = args.max_tokens
    user_prompt = ns['build_prompt'](puzzle)
    messages = [{"role": "user", "content": user_prompt}]
    # DeepSeek-R1 chat template appends the assistant prefix + THINK_OPEN + newline
    # so the model generates reasoning, then THINK_CLOSE, then the answer.
    chat_text = mlx_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    t0 = time.time()
    forced = False
    # Stage 1: reasoning (+ maybe answer if model converges)
    out1 = generate(mlx_model, mlx_tokenizer, prompt=chat_text,
                    max_tokens=reason_tokens, verbose=False)
    out1 = out1.strip()
    if EOS in out1:
        out1 = out1.replace(EOS, '').strip()
    if THINK_CLOSE in out1:
        full_text = out1
    else:
        # Model did not finish reasoning within budget -> force an answer.
        forced = True
        forced_prompt = chat_text + out1 + THINK_CLOSE + '\n'
        out2 = generate(mlx_model, mlx_tokenizer, prompt=forced_prompt,
                        max_tokens=answer_tokens, verbose=False)
        out2 = out2.strip()
        if EOS in out2:
            out2 = out2.replace(EOS, '').strip()
        full_text = out1 + THINK_CLOSE + '\n' + out2
    elapsed = time.time() - t0
    if THINK_CLOSE in full_text:
        thinking, answer = full_text.split(THINK_CLOSE, 1)
        thinking = thinking.strip()
        answer = answer.strip()
    else:
        thinking = ""
        answer = full_text
    predicted = parse_grid(answer)
    return {
        'raw_output': full_text,
        'thinking': thinking,
        'predicted': predicted,
        'thinking_length': len(thinking),
        'elapsed': elapsed,
        'forced_answer': forced,
    }

# ── Select puzzles ──
dataset = ns['dataset']
if args.types:
    wanted = set(args.types)
    dataset = [p for p in dataset if p['type'] in wanted]
    print(f"\nFiltered to {len(dataset)} puzzles by type: {sorted(wanted)}", flush=True)

# ── Resume ──
existing = {}
if args.resume and os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        for r in json.load(f):
            existing[r['puzzle_id']] = r
    print(f"Resume: {len(existing)} puzzles already have results.", flush=True)

# ── Smoke test ──
if args.smoke and dataset:
    print("\n" + "=" * 70, flush=True)
    print(f"SMOKE TEST: {dataset[0]['id']}, max_tokens={args.max_tokens}", flush=True)
    print("=" * 70, flush=True)
    r = run_inference_mlx(dataset[0])
    ev = evaluate_result(r, dataset[0])
    print(f"Time: {r['elapsed']:.1f}s | thinking: {r['thinking_length']} chars", flush=True)
    print(f"Predicted: {r['predicted']}", flush=True)
    print(f"Expected dims: {grid_dims(dataset[0]['test_output'])} | "
          f"Predicted dims: {grid_dims(r['predicted']) if r['predicted'] else None}", flush=True)
    print(f"exact={ev['exact_match']} shape={ev['shape_match']} cell_acc={ev['cell_accuracy']:.2%}", flush=True)
    print(f"\n--- thinking (first 700 chars) ---\n{r['thinking'][:700]}\n...", flush=True)
    print(f"\n--- answer (first 400 chars) ---\n{r['raw_output'].split(THINK_CLOSE,1)[-1][:400] if THINK_CLOSE in r['raw_output'] else r['raw_output'][-400:]}", flush=True)

# ── Full benchmark loop ──
print("\n" + "=" * 70, flush=True)
print(f"FULL BENCHMARK: {len(dataset)} puzzles | max_tokens={args.max_tokens}", flush=True)
print("=" * 70, flush=True)

results = list(existing.values())
done_ids = set(existing.keys())
overall_t0 = time.time()
for i, puzzle in enumerate(dataset):
    pid = puzzle['id']
    if pid in done_ids:
        print(f"[{i+1}/{len(dataset)}] {pid}: SKIP (done)", flush=True)
        continue
    print(f"\n[{i+1}/{len(dataset)}] {pid}: {puzzle['description']}", flush=True)
    try:
        r = run_inference_mlx(puzzle)
        ev = evaluate_result(r, puzzle)
        status = "EXACT" if ev['exact_match'] else ("SHAPE" if ev['shape_match'] else "MISS")
        forced_tag = " [FORCED]" if r.get('forced_answer') else ""
        print(f"  {status}{forced_tag} | cell_acc={ev['cell_accuracy']:.2%} | think={ev['thinking_length']}c | "
              f"dims pred={ev['predicted_dims']} exp={ev['expected_dims']} | {r['elapsed']:.1f}s", flush=True)
        results.append({
            'puzzle_id': pid,
            'puzzle_type': puzzle['type'],
            'description': puzzle['description'],
            'exact_match': ev['exact_match'],
            'shape_match': ev['shape_match'],
            'cell_accuracy': ev['cell_accuracy'],
            'has_prediction': ev['has_prediction'],
            'thinking_length': ev['thinking_length'],
            'predicted_dims': str(ev['predicted_dims']),
            'expected_dims': str(ev['expected_dims']),
            'elapsed': r['elapsed'],
            'forced_answer': r.get('forced_answer', False),
            'thinking_excerpt': r['thinking'][:300],
            'raw_output_excerpt': r['raw_output'][-400:],
        })
    except Exception as e:
        import traceback
        print(f"  ERROR: {e} | {traceback.format_exc()[-300:]}", flush=True)
        results.append({
            'puzzle_id': pid, 'puzzle_type': puzzle['type'], 'description': puzzle['description'],
            'exact_match': False, 'shape_match': False, 'cell_accuracy': 0.0,
            'has_prediction': False, 'thinking_length': 0,
            'predicted_dims': 'None', 'expected_dims': str(grid_dims(puzzle['test_output'])),
            'elapsed': 0.0, 'thinking_excerpt': f'ERROR: {str(e)[:300]}', 'raw_output_excerpt': '',
        })
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

loop_time = time.time() - overall_t0
print(f"\nLoop complete: {len(results)} results in {loop_time:.0f}s ({loop_time/60:.1f} min)", flush=True)

# ── Aggregate ──
import numpy as np
total = len(results)
exact = sum(1 for r in results if r['exact_match'])
shape = sum(1 for r in results if r['shape_match'])
avg_cell = float(np.mean([r['cell_accuracy'] for r in results])) if results else 0.0
avg_think = float(np.mean([r['thinking_length'] for r in results])) if results else 0.0
total_time = sum(r['elapsed'] for r in results)

print("\n" + "=" * 70, flush=True)
print("OVERALL RESULTS", flush=True)
print("=" * 70, flush=True)
print(f"Total puzzles:     {total}", flush=True)
if total:
    print(f"Exact matches:     {exact}/{total} = {exact/total:.1%}", flush=True)
    print(f"Shape matches:     {shape}/{total} = {shape/total:.1%}", flush=True)
print(f"Avg cell accuracy: {avg_cell:.1%}", flush=True)
print(f"Avg thinking len:  {avg_think:.0f} chars", flush=True)
print(f"Total infer time:  {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

print("\n" + "-" * 80, flush=True)
print(f"{'Puzzle Type':<32} {'Exact':>7} {'Shape':>7} {'CellAcc':>8} {'AvgThink':>9}", flush=True)
print("-" * 80, flush=True)
type_results = {}
for r in results:
    type_results.setdefault(r['puzzle_type'], []).append(r)
for t in sorted(type_results):
    rs = type_results[t]
    em = sum(1 for r in rs if r['exact_match'])
    sm = sum(1 for r in rs if r['shape_match'])
    ca = float(np.mean([r['cell_accuracy'] for r in rs]))
    tl = float(np.mean([r['thinking_length'] for r in rs]))
    print(f"{rs[0]['description'][:30]:<32} {em:>3}/{len(rs):<3} {sm:>3}/{len(rs):<3} {ca:>7.1%} {tl:>8.0f}c", flush=True)
print("-" * 80, flush=True)

# ── Save dataset ──
with open(DATASET_PATH, 'w') as f:
    json.dump([{'id': p['id'], 'type': p['type'], 'description': p['description'],
                'examples': p['examples'], 'test_input': p['test_input'],
                'test_output': p['test_output']} for p in dataset], f, indent=2)
print(f"\nResults: {RESULTS_PATH}", flush=True)
print(f"Dataset: {DATASET_PATH}", flush=True)

# ── Plot ──
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    types_sorted = sorted(type_results, key=lambda t: sum(1 for r in type_results[t] if r['exact_match']), reverse=True)
    em_rates = [sum(1 for r in type_results[t] if r['exact_match'])/len(type_results[t]) for t in types_sorted]
    ca_rates = [float(np.mean([r['cell_accuracy'] for r in type_results[t]])) for t in types_sorted]
    labels = [type_results[t][0]['description'][:20] for t in types_sorted]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].barh(range(len(types_sorted)), em_rates, color='#06d6a0')
    axes[0,0].set_yticks(range(len(types_sorted))); axes[0,0].set_yticklabels(labels, fontsize=8)
    axes[0,0].set_xlabel('Exact Match Rate'); axes[0,0].set_title('Exact Match by Puzzle Type'); axes[0,0].set_xlim(0,1)
    axes[0,1].barh(range(len(types_sorted)), ca_rates, color='#118ab2')
    axes[0,1].set_yticks(range(len(types_sorted))); axes[0,1].set_yticklabels(labels, fontsize=8)
    axes[0,1].set_xlabel('Avg Cell Accuracy'); axes[0,1].set_title('Cell Accuracy by Puzzle Type'); axes[0,1].set_xlim(0,1)
    think_lens = [r['thinking_length'] for r in results]
    colors_think = ['#06d6a0' if r['exact_match'] else '#e63946' for r in results]
    axes[1,0].bar(range(len(think_lens)), think_lens, color=colors_think)
    axes[1,0].set_xlabel('Puzzle Index'); axes[1,0].set_ylabel('Thinking Length (chars)')
    axes[1,0].set_title('Reasoning Length per Puzzle (green=correct, red=wrong)')
    for r in results:
        c = '#06d6a0' if r['exact_match'] else ('#ffd166' if r['shape_match'] else '#e63946')
        axes[1,1].scatter(r['thinking_length'], r['cell_accuracy'], c=c, s=50, alpha=0.7)
    axes[1,1].set_xlabel('Thinking Length (chars)'); axes[1,1].set_ylabel('Cell Accuracy')
    axes[1,1].set_title('Reasoning Length vs Accuracy'); axes[1,1].set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    print(f"Plot: {PLOT_PATH}", flush=True)
except Exception as e:
    print(f"Plot skipped: {e}", flush=True)

print("\nDONE.", flush=True)
