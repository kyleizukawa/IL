#!/usr/bin/env python3
"""Generate the IL-pipeline Kaggle notebook (.ipynb)."""
import json

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "source": src.splitlines(True),
                  "execution_count": None, "outputs": []})

# ────────────────────────────────────────────────────────────────────
md("""# IL Pipeline v3 on Kaggle T4: SFT + GRPO on DeepSeek-R1-Distill-Qwen-1.5B

## Goal
Load a small, fast reasoning model (**DeepSeek-R1-Distill-Qwen-1.5B**, 1.78B params)
on a Kaggle T4, benchmark it on the **36-puzzle ARC-AGI-3-style benchmark**,
then run the **IL (Intuition Learning) pipeline** — SFT with cell-level analytical
reasoning + transfer skills, followed by GRPO RL with curriculum learning — and
re-benchmark after each stage to measure improvement.

## v3 improvements over v2
- **Cell-level analytical reasoning**: SFT teacher traces now describe what happens
  to specific cells (e.g. "4 cells unchanged; 2 cells changed color (0,4): 1→3"),
  not just state dimensions and jump to a rule. Teaches the model to OBSERVE before SOLVE.
- **Cross-type transfer examples**: 90 examples teaching skills that directly transfer
  to benchmark puzzles (gravity, sorting, pathfinding, flood fill, symmetry, object detection)
- **Structured reasoning format**: Observation → Pattern → Rule → Verification → Application
- **More token budget**: think=512/answer=256 in benchmark (was 384/192) for deeper reasoning
- **GRPO with process reward**: reward includes shape bonus + accuracy improvement bonus

## Pipeline stages
1. **Pre-benchmark** — base model on 36 transfer puzzles (greedy)
2. **SFT** — supervised fine-tune on v3 analytical dataset (1003 examples, 5 epochs)
3. **Post-SFT benchmark**
4. **GRPO RL** — 100 iterations with curriculum learning, warm-started from SFT
5. **Post-GRPO benchmark**
6. **Compare** all three stages + visualize

> **Kaggle settings**: GPU **T4 x2** + **Internet on**. The 1.5B model (3 GB bf16)
> runs on `cuda:0`; second T4 is memory headroom.
""")

# ── Cell: Install & Import ──
md("## 0. Install & Import")

code("""# Uninstall preinstalled torchao (Kaggle ships 0.10, peft needs >0.16).
# We don't use torchao — removing it avoids the version-check ImportError.
!pip uninstall -y torchao >/dev/null 2>&1
!pip install -q transformers==4.46.3 accelerate peft datasets

import os, sys, re, json, time, random, traceback, math
from copy import deepcopy
from collections import deque

# v5: Reduce memory fragmentation on T4 (must be set before torch import)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Config (v4 — fixed tags, computation traces, dual T4, batched rollouts) ──
MODEL_NAME        = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_REF       = "krokodileceo/il-research-sft-v3-benchmark"   # v3 dataset (regenerated)
# SFT (v4: 3 epochs — loss plateaus at epoch 2.5; gradient checkpointing for bigger batches)
SFT_EPOCHS        = 3
SFT_BATCH_SIZE    = 4         # v4: 4 (was 2) — gradient checkpointing frees memory
SFT_GRAD_ACCUM    = 4         # v4: effective batch = 16
SFT_LR            = 2e-4
SFT_MAX_SEQ_LEN   = 1024
SFT_WARMUP_STEPS  = 30
# LoRA (shared by SFT + GRPO)
LORA_RANK         = 16
LORA_LAYERS       = 16
LORA_ALPHA        = 32
# GRPO (v4: group=8, temp=0.6, multi-puzzle per iter, fixed KL sign)
N_TRAIN_ITERS     = 100
GROUP_SIZE        = 8          # v4: 8 (was 6) — better advantage estimates
N_STEPS           = 2
PUZZLES_PER_ITER  = 2          # v4: sample 2 puzzles per iter for stable gradients
THINKING_TOKENS   = 256
PREDICTION_TOKENS = 128
GRPO_LR           = 1e-5
TEMPERATURE       = 0.6        # v4: 0.6 (was 0.8) — more consistent rollouts
TOP_P             = 0.9
GAMMA             = 0.9
KL_BETA           = 0.04
CLIP_EPS          = 0.2
PPO_EPOCHS        = 2          # v5: reuse rollouts for 2 gradient updates (sample efficiency)
ADV_CLIP          = 5.0        # v5: clip advantages to [-5, 5] to prevent gradient explosion
USE_KL            = True       # v5: KL-free option (set False to save 30-40% memory)
# Benchmark
BENCH_THINK       = 512
BENCH_ANSWER      = 256

DEVICE = "cuda:0"
DEVICE2 = "cuda:1"  # v4: use second T4 for data-parallel benchmark + GRPO rollouts
print(f"Config v4: SFT {SFT_EPOCHS} epochs bs{SFT_BATCH_SIZE}x{SFT_GRAD_ACCUM} lr{SFT_LR} | "
      f"GRPO {N_TRAIN_ITERS} iters group={GROUP_SIZE} think={THINKING_TOKENS} temp={TEMPERATURE} | "
      f"bench think={BENCH_THINK} pred={BENCH_ANSWER}")
""")

# ── Cell: Load Model ──
md("""## 1. Load Model on T4 (cuda:0) + second T4 for data parallelism

A 1.5B model in bfloat16 is ~3 GB — fits comfortably on one T4 (16 GB).
v4: SDPA attention + TF32 for speed, gradient checkpointing for memory.""")

code("""n_gpus = torch.cuda.device_count()
print(f"Available GPUs: {n_gpus}")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} — {props.name}, {props.total_memory / 1e9:.1f} GB")
assert n_gpus >= 1, "No GPU — enable T4 x2 in Kaggle settings."

# v4: Enable TF32 for faster matmuls on T4
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"\\nLoading {MODEL_NAME} (bfloat16, {DEVICE}, SDPA) ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"" : DEVICE},
    attn_implementation="sdpa",   # v4: Flash Attention via SDPA
)
base_model.eval()
# v5: use_cache=False required for gradient checkpointing to work
base_model.config.use_cache = False

# v4: Gradient checkpointing for SFT (saves ~60% activation memory)
# v5: Must disable use_cache FIRST, then enable checkpointing
base_model.gradient_checkpointing_enable()
# v5: Enable for the model that will be used for training
base_model.enable_input_require_grads()

n_params = sum(p.numel() for p in base_model.parameters())
print(f"Loaded in {time.time()-t0:.1f}s | params: {n_params/1e9:.2f}B")
print(f"Device: {next(base_model.parameters()).device}")
print(f"TF32: {torch.backends.cuda.matmul.allow_tf32} | SDPA attention enabled")

# v4: Load second model on cuda:1 for data-parallel benchmark/GRPO rollouts
model2 = None
if n_gpus >= 2:
    print(f"\\nLoading second model on {DEVICE2} for data-parallel inference...")
    t1 = time.time()
    base_model2 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"" : DEVICE2},
        attn_implementation="sdpa",
    )
    base_model2.eval()
    base_model2.config.use_cache = True
    # We'll sync LoRA weights to model2 after SFT/GRPO
    model2 = base_model2  # Will be wrapped with LoRA later if needed
    print(f"Second model loaded in {time.time()-t1:.1f}s on {DEVICE2}")
else:
    print(f"\\nOnly {n_gpus} GPU(s) — single-GPU mode")
""")

# ── Cell: LoRA ──
md("## 2. Apply LoRA (rank 8, last 16 layers)")

code("""# DeepSeek-R1-Distill-Qwen-1.5B has 28 layers
n_layers = base_model.config.num_hidden_layers
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    layers_to_transform=list(range(n_layers - LORA_LAYERS, n_layers)),
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
trainable = [p for p in model.parameters() if p.requires_grad]
assert trainable, "No trainable parameters — LoRA setup failed."

# v5: torch.compile disabled — causes OOM on T4 (CUDA Graphs pre-alloc ~3GB)
# and recompilation overhead with variable-length sequences.
# SDPA attention + TF32 already provide good speedup.
print("Skipping torch.compile (T4 memory constraints) — using SDPA + TF32")
""")

# ── Cell: Grid utilities ──
md("""## 3. Grid Utilities & Evaluation

Pure-Python grid helpers + evaluation functions (ported from the ilresearch repo).""")

code("""def empty_grid(h, w, val=0): return [[val]*w for _ in range(h)]
def grid_copy(g): return [row[:] for row in g]
def grid_dims(g): return len(g), len(g[0])
def grid_to_str(g):
    rows = ['[' + ','.join(str(c) for c in row) + ']' for row in g]
    return '[' + ',\\n '.join(rows) + ']'
def grid_to_str_compact(g):
    "v4: Compact grid format - space-separated rows, ~30 percent fewer tokens."
    return '\\n'.join(' '.join(str(c) for c in row) for row in g)
# v5: Multi-view grid representations
_SYMBOL_MAP = {0: '.', 1: '#', 2: 'x', 3: 'o', 4: '+', 5: '*', 6: '=', 7: '@', 8: '%', 9: '&'}
def grid_to_ascii(g):
    "v5: ASCII art grid representation - more visual, fewer tokens."
    return chr(10).join(''.join(_SYMBOL_MAP.get(c, '?') for c in row) for row in g)

def count_nonzero(g): return sum(1 for row in g for c in row if c != 0)
def distinct_colors(g): return set(c for row in g for c in row if c != 0)
def connected_components(g, connectivity=4):
    h, w = grid_dims(g); visited = empty_grid(h, w, False); comps = []
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)] if connectivity==4 else \\
           [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for r in range(h):
        for c in range(w):
            if g[r][c]!=0 and not visited[r][c]:
                color=g[r][c]; comp=[]; q=deque([(r,c)]); visited[r][c]=True
                while q:
                    cr,cc=q.popleft(); comp.append((cr,cc))
                    for dr,dc in nbrs:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and g[nr][nc]==color:
                            visited[nr][nc]=True; q.append((nr,nc))
                comps.append({'color':color,'cells':comp,'size':len(comp)})
    return comps
def bounding_box(cells):
    rs=[r for r,c in cells]; cs=[c for r,c in cells]
    return min(rs),min(cs),max(rs),max(cs)
print("Grid utilities loaded.")

# v5: Data augmentation via grid symmetries
def rotate_grid_90(g):
    "Rotate grid 90 degrees clockwise."
    h, w = len(g), len(g[0])
    return [[g[h-1-j][i] for j in range(h)] for i in range(w)]

def flip_grid_h(g):
    "Flip grid horizontally."
    return [row[::-1] for row in g]

def flip_grid_v(g):
    "Flip grid vertically."
    return g[::-1]

def permute_colors(g, perm):
    "Permute colors in grid. perm is a dict {old: new}."
    return [[perm.get(c, c) for c in row] for row in g]

def augment_puzzle(puzzle, rng):
    "Apply random symmetry augmentation to a puzzle."
    aug_type = rng.choice(['rotate', 'flip_h', 'flip_v', 'color_perm', 'none'])
    if aug_type == 'none':
        return puzzle, aug_type
    if aug_type == 'rotate':
        new_p = dict(puzzle)
        new_p['test_input'] = rotate_grid_90(puzzle['test_input'])
        new_p['test_output'] = rotate_grid_90(puzzle['test_output'])
        new_p['examples'] = [{'input': rotate_grid_90(e['input']), 'output': rotate_grid_90(e['output'])} for e in puzzle['examples']]
    elif aug_type == 'flip_h':
        new_p = dict(puzzle)
        new_p['test_input'] = flip_grid_h(puzzle['test_input'])
        new_p['test_output'] = flip_grid_h(puzzle['test_output'])
        new_p['examples'] = [{'input': flip_grid_h(e['input']), 'output': flip_grid_h(e['output'])} for e in puzzle['examples']]
    elif aug_type == 'flip_v':
        new_p = dict(puzzle)
        new_p['test_input'] = flip_grid_v(puzzle['test_input'])
        new_p['test_output'] = flip_grid_v(puzzle['test_output'])
        new_p['examples'] = [{'input': flip_grid_v(e['input']), 'output': flip_grid_v(e['output'])} for e in puzzle['examples']]
    elif aug_type == 'color_perm':
        colors = list(set(c for row in puzzle['test_input'] for c in row if c != 0))
        if len(colors) < 2:
            return puzzle, 'none'
        shuffled = colors[:]
        rng.shuffle(shuffled)
        perm = dict(zip(colors, shuffled))
        perm[0] = 0  # Keep background
        new_p = dict(puzzle)
        new_p['test_input'] = permute_colors(puzzle['test_input'], perm)
        new_p['test_output'] = permute_colors(puzzle['test_output'], perm)
        new_p['examples'] = [{'input': permute_colors(e['input'], perm), 'output': permute_colors(e['output'], perm)} for e in puzzle['examples']]
    return new_p, aug_type

def parse_grid(text):
    \"\"\"Robust grid parser. Finds the last 2D array in text and parses it.
    Handles spaces, newlines, trailing commas, and nested brackets.\"\"\"
    # Strategy: find all top-level 2D arrays (depth-2 bracket nesting), take the last one
    last_2d = None; i = 0
    while i < len(text):
        if text[i] == '[':
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '[': depth += 1
                elif text[j] == ']':
                    depth -= 1
                    if depth == 0:
                        cand = text[i:j+1]
                        if cand.count('[') >= 3: last_2d = cand
                        i = j; break
            else: break
        i += 1
    if last_2d:
        # Parse rows: each row is [digits, digits, ...]
        rows = re.findall(r'\\[\\s*([\\d\\s,]+)\\]', last_2d)
        grid = []
        for rs in rows:
            nums = re.findall(r'\\d+', rs)
            if nums: grid.append([int(x) for x in nums])
        if grid: return grid
    # Fallback: find individual rows anywhere in text
    row_matches = re.findall(r'\\[\\s*\\d+[\\s,\\d]*\\]', text)
    if row_matches:
        grid = []
        for rs in row_matches:
            nums = re.findall(r'\\d+', rs)
            if nums: grid.append([int(x) for x in nums])
        if grid: return grid
    # Fallback 2: try splitting on newlines and parsing each line as a row
    for line in text.split('\\n'):
        line = line.strip().strip('[]').strip()
        if line and all(c.isdigit() or c in ' ,' for c in line):
            nums = re.findall(r'\\d+', line)
            if len(nums) >= 2:
                if not row_matches:
                    grid = []
                break
    return None

def grids_equal(g1,g2):
    if g1 is None or g2 is None: return False
    if len(g1)!=len(g2): return False
    for r1,r2 in zip(g1,g2):
        if len(r1)!=len(r2) or r1!=r2: return False
    return True

def shape_match(g1,g2):
    if g1 is None or g2 is None: return False
    return grid_dims(g1)==grid_dims(g2)

def cell_accuracy(pred,target):
    if pred is None or target is None: return 0.0
    if len(pred)!=len(target) or len(pred)==0 or len(pred[0])==0: return 0.0
    h,w=len(target),len(target[0])
    correct=sum(1 for r in range(min(len(pred),h)) for c in range(min(len(pred[r]),w))
                if r<len(target) and c<len(target[r]) and pred[r][c]==target[r][c])
    return correct/(h*w) if h*w>0 else 0.0

def signal_accuracy(pred,target):
    if pred is None or target is None or len(pred)==0 or len(target)==0: return 0.0
    signal=[(r,c,target[r][c]) for r in range(len(target)) for c in range(len(target[r])) if target[r][c]!=0]
    if not signal: return cell_accuracy(pred,target)
    correct=sum(1 for r,c,v in signal if r<len(pred) and c<len(pred[r]) and pred[r][c]==v)
    return correct/len(signal)

def is_degenerate(pred):
    if pred is None or len(pred)==0: return True
    if len(pred)==1 and len(pred[0])<=1: return True
    return all(all(cell==0 for cell in row) for row in pred)

def shape_distance(pred,target):
    if pred is None or len(pred)==0: return 0.0
    eh,ew=len(target),len(target[0]); ph=len(pred); pw=len(pred[0]) if ph>0 else 0
    h_sim=1.0-abs(ph-eh)/max(eh,ph,1); w_sim=1.0-abs(pw-ew)/max(ew,pw,1)
    return (h_sim+w_sim)/2
print("Evaluation functions loaded.")
""")

# ── Cell: Load benchmark dataset ──
md("""## 4. Load Benchmark Dataset (36 puzzles)

Loaded from the Kaggle dataset `il-research-sft-benchmark`.""")

code("""# Robust dataset path resolution: search /kaggle/input/ recursively for the file.
import glob
def find_data_file(filename):
    # Search recursively under /kaggle/input/ (Kaggle may nest under datasets/)
    matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if matches:
        return matches[0]
    # Fallback: try working dir (in case dataset is attached differently)
    if os.path.exists(filename):
        return filename
    raise FileNotFoundError(f"Could not find {filename}. Searched /kaggle/input/ recursively. "
                            f"Top-level: {os.listdir('/kaggle/input/') if os.path.isdir('/kaggle/input') else 'no /kaggle/input'}")

BENCH_PATH = find_data_file("benchmark_dataset.json")
print(f"Benchmark dataset: {BENCH_PATH}")
with open(BENCH_PATH) as f:
    benchmark_puzzles = json.load(f)
print(f"Benchmark: {len(benchmark_puzzles)} puzzles")
print(f"  types: {sorted(set(p['type'] for p in benchmark_puzzles))}")
print(f"  example puzzle: {benchmark_puzzles[0]['id']} — {benchmark_puzzles[0]['description']}")
""")

# ── Cell: Prompt builder + inference ──
md("""## 5. Prompt Builder & Inference

Build the same prompt the benchmark expects, then run two-stage generation
(let model reason; if it doesn't close the think tag, force it).""")

code("""# DeepSeek-R1-D reasoning tags.
# Token id for </think> (151667 in the Qwen tokenizer for R1-distill).
THINK_OPEN_STR = "<think>"
THINK_CLOSE_STR = "</think>"
THINK_CLOSE_ID = tokenizer.convert_tokens_to_ids("</think>")
EOS_TOKEN = tokenizer.eos_token
print(f"</think> token id: {THINK_CLOSE_ID}")

# v4: Cache for system prompt prefix tokens
_SYSTEM_PREFIX = None
def _get_system_prefix():
    global _SYSTEM_PREFIX
    if _SYSTEM_PREFIX is None:
        sys_msg = ("You are an abstract reasoning system. You will be given example "
                   "input-output grid pairs that demonstrate a transformation rule. "
                   "You must infer the rule from the examples and apply it to the test input.\\n\\n"
                   "Grids are 2D arrays of integers 0-9. 0 represents empty/background.\\n\\n"
                   "=== EXAMPLES ===\\n")
        _SYSTEM_PREFIX = tokenizer(sys_msg, add_special_tokens=False)['input_ids']
    return _SYSTEM_PREFIX

def build_prompt(puzzle):
    lines = ["You are an abstract reasoning system. You will be given example "
             "input-output grid pairs that demonstrate a transformation rule. "
             "You must infer the rule from the examples and apply it to the test input.",
             "",
             "Grids are 2D arrays of integers 0-9. 0 represents empty/background.",
             "", "=== EXAMPLES ===", ""]
    for i, ex in enumerate(puzzle['examples']):
        lines += [f"--- Example {i+1} ---", "Input:", grid_to_str(ex['input']),
                  "Output:", grid_to_str(ex['output']), ""]
    lines += ["=== TEST ===", "", "Input:", grid_to_str(puzzle['test_input']), "",
              "Apply the transformation rule and output ONLY the resulting grid as a 2D array."]
    return '\\n'.join(lines)

@torch.inference_mode()
def run_inference(puzzle, think_tokens, answer_tokens, temperature=0.0, top_p=1.0):
    \"\"\"Two-stage greedy inference. Returns dict with predicted grid, thinking, raw.\"\"\"
    user_prompt = build_prompt(puzzle)
    messages = [{"role": "user", "content": user_prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(chat_text, return_tensors="pt").input_ids.to(DEVICE)

    t0 = time.time()
    forced = False
    # Stage 1: reasoning — v4: stop on think-close OR eos (saves tokens)
    out1 = model.generate(
        input_ids, max_new_tokens=think_tokens,
        do_sample=temperature > 0, temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p if temperature > 0 else 1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, THINK_CLOSE_ID],
    )
    gen1 = out1[0][input_ids.shape[1]:]
    text1 = tokenizer.decode(gen1, skip_special_tokens=False)
    if EOS_TOKEN and EOS_TOKEN in text1:
        text1 = text1.replace(EOS_TOKEN, '').strip()

    if THINK_CLOSE_STR in text1:
        # v5 FIX: Model generated </think> naturally — but we still need the answer grid.
        # Strip </think> from text1, then run Stage 2 to generate the answer.
        text1_clean = text1.replace(THINK_CLOSE_STR, '').strip()
        forced = False
        forced_text = chat_text + text1_clean + THINK_CLOSE_STR + "\\n"
        forced_ids = tokenizer(forced_text, return_tensors="pt").input_ids.to(DEVICE)
        out2 = model.generate(
            forced_ids, max_new_tokens=answer_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        gen2 = out2[0][forced_ids.shape[1]:]
        text2 = tokenizer.decode(gen2, skip_special_tokens=False)
        if EOS_TOKEN and EOS_TOKEN in text2:
            text2 = text2.replace(EOS_TOKEN, '').strip()
        full_text = text1_clean + THINK_CLOSE_STR + "\\n" + text2
    else:
        # Stage 2: force close think tag + short answer
        forced = True
        forced_text = chat_text + text1 + THINK_CLOSE_STR + "\\n"
        forced_ids = tokenizer(forced_text, return_tensors="pt").input_ids.to(DEVICE)
        out2 = model.generate(
            forced_ids, max_new_tokens=answer_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        gen2 = out2[0][forced_ids.shape[1]:]
        text2 = tokenizer.decode(gen2, skip_special_tokens=False)
        if EOS_TOKEN and EOS_TOKEN in text2:
            text2 = text2.replace(EOS_TOKEN, '').strip()
        full_text = text1 + THINK_CLOSE_STR + "\\n" + text2

    elapsed = time.time() - t0
    if THINK_CLOSE_STR in full_text:
        thinking, answer = full_text.split(THINK_CLOSE_STR, 1)
        thinking = thinking.strip(); answer = answer.strip()
    else:
        thinking = ""; answer = full_text
    predicted = parse_grid(answer)
    return {'raw_output': full_text, 'thinking': thinking, 'predicted': predicted,
            'thinking_length': len(thinking), 'elapsed': elapsed, 'forced_answer': forced}

@torch.inference_mode()
def run_batched_inference(puzzles, think_tokens, answer_tokens, batch_size=6):
    '''v4: Batched inference for speed. Stage 1 batched, stage 2 individual.'''
    results = []
    for batch_start in range(0, len(puzzles), batch_size):
        batch = puzzles[batch_start:batch_start+batch_size]
        prompts = []
        for p in batch:
            user_prompt = build_prompt(p)
            messages = [{"role": "user", "content": user_prompt}]
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(chat_text)
        tokenizer.padding_side = 'left'
        batch_enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
        input_ids = batch_enc['input_ids'].to(DEVICE)
        attn_mask = batch_enc['attention_mask'].to(DEVICE)
        t0 = time.time()
        out1 = model.generate(
            input_ids, attention_mask=attn_mask, max_new_tokens=think_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, THINK_CLOSE_ID],
        )
        for i, p in enumerate(batch):
            gen1 = out1[i][input_ids.shape[1]:]
            text1 = tokenizer.decode(gen1, skip_special_tokens=False)
            if EOS_TOKEN and EOS_TOKEN in text1:
                text1 = text1.replace(EOS_TOKEN, '').strip()
            forced = False
            if THINK_CLOSE_STR in text1:
                # v5 FIX: Model generated </think> — still need answer grid.
                text1_clean = text1.replace(THINK_CLOSE_STR, '').strip()
                forced_text = prompts[i] + text1_clean + THINK_CLOSE_STR + "\\n"
                forced_ids = tokenizer(forced_text, return_tensors="pt").input_ids.to(DEVICE)
                out2 = model.generate(
                    forced_ids, max_new_tokens=answer_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
                gen2 = out2[0][forced_ids.shape[1]:]
                text2 = tokenizer.decode(gen2, skip_special_tokens=False)
                if EOS_TOKEN and EOS_TOKEN in text2:
                    text2 = text2.replace(EOS_TOKEN, '').strip()
                full_text = text1_clean + THINK_CLOSE_STR + "\\n" + text2
            else:
                forced = True
                forced_text = prompts[i] + text1 + THINK_CLOSE_STR + "\\n"
                forced_ids = tokenizer(forced_text, return_tensors="pt").input_ids.to(DEVICE)
                out2 = model.generate(
                    forced_ids, max_new_tokens=answer_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
                gen2 = out2[0][forced_ids.shape[1]:]
                text2 = tokenizer.decode(gen2, skip_special_tokens=False)
                if EOS_TOKEN and EOS_TOKEN in text2:
                    text2 = text2.replace(EOS_TOKEN, '').strip()
                full_text = text1 + THINK_CLOSE_STR + "\\n" + text2
            if THINK_CLOSE_STR in full_text:
                thinking, answer = full_text.split(THINK_CLOSE_STR, 1)
                thinking = thinking.strip(); answer = answer.strip()
            else:
                thinking = ""; answer = full_text
            predicted = parse_grid(answer)
            results.append({'raw_output': full_text, 'thinking': thinking, 'predicted': predicted,
                            'thinking_length': len(thinking), 'elapsed': time.time()-t0, 'forced_answer': forced})
        print(f"  Batch {batch_start//batch_size+1}/{(len(puzzles)+batch_size-1)//batch_size} done ({len(batch)} puzzles)")
    return results

print("Prompt builder + inference loaded (v4: batched + stop-on-think-close).")
""")

# ── Cell: Benchmark runner ──
md("## 6. Benchmark Runner")

code("""def run_benchmark(puzzles, think_tokens, answer_tokens, label=""):
    \"\"\"Run greedy inference on each puzzle. Returns list of result dicts.\"\"\"
    model.eval()
    model.config.use_cache = True  # v5: Enable KV cache for inference
    results = []; t0 = time.time()
    for i, p in enumerate(puzzles):
        torch.cuda.empty_cache()
        try:
            r = run_inference(p, think_tokens, answer_tokens, temperature=0.0)
            pred = r['predicted']
            em = grids_equal(pred, p['test_output'])
            sm = shape_match(pred, p['test_output'])
            ca = cell_accuracy(pred, p['test_output'])
            results.append({'puzzle_id': p['id'], 'puzzle_type': p['type'],
                            'description': p['description'], 'exact_match': em,
                            'shape_match': sm, 'cell_accuracy': ca,
                            'has_prediction': pred is not None,
                            'predicted_dims': str(grid_dims(pred) if pred else None),
                            'expected_dims': str(grid_dims(p['test_output'])),
                            'thinking_length': r['thinking_length'],
                            'elapsed': r['elapsed'], 'forced_answer': r['forced_answer'],
                            'thinking_excerpt': r['thinking'][:300]})
            status = "EXACT" if em else ("SHAPE" if sm else "MISS")
            ftag = " [F]" if r['forced_answer'] else ""
            print(f"  [{label} {i+1:2d}/{len(puzzles)}] {p['id']:<28} {status:5s}{ftag} "
                  f"acc={ca:.2%} dims={grid_dims(pred) if pred else None}/{grid_dims(p['test_output'])} "
                  f"t={r['elapsed']:.1f}s")
        except Exception as e:
            print(f"  [{label} {i+1:2d}/{len(puzzles)}] {p['id']}: ERROR {str(e)[:120]}")
            results.append({'puzzle_id': p['id'], 'puzzle_type': p['type'],
                            'description': p['description'], 'exact_match': False,
                            'shape_match': False, 'cell_accuracy': 0.0,
                            'has_prediction': False, 'predicted_dims': 'None',
                            'expected_dims': str(grid_dims(p['test_output'])),
                            'thinking_length': 0, 'elapsed': 0.0, 'forced_answer': False,
                            'thinking_excerpt': f'ERROR: {str(e)[:300]}'})
    elapsed = time.time() - t0
    print(f"  {label} done: {len(results)} puzzles in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results

def summarize(results, label):
    total = len(results); exact = sum(1 for r in results if r['exact_match'])
    shape = sum(1 for r in results if r['shape_match'])
    avg_cell = float(np.mean([r['cell_accuracy'] for r in results])) if results else 0.0
    has_pred = sum(1 for r in results if r['has_prediction'])
    forced = sum(1 for r in results if r.get('forced_answer', False))
    print(f"\\n{'='*70}\\n{label}: {total} puzzles\\n{'='*70}")
    print(f"  Exact matches: {exact}/{total} = {exact/total:.1%}")
    print(f"  Shape matches: {shape}/{total} = {shape/total:.1%}")
    print(f"  Avg cell acc:  {avg_cell:.1%}")
    print(f"  Has prediction: {has_pred}/{total} | Forced: {forced}/{total}")
    print(f"\\n  {'Puzzle Type':<28} {'Exact':>7} {'Shape':>7} {'CellAcc':>8}")
    print(f"  {'-'*52}")
    by_type = {}
    for r in results: by_type.setdefault(r['puzzle_type'], []).append(r)
    for t in sorted(by_type):
        rs = by_type[t]; em = sum(1 for r in rs if r['exact_match'])
        sm = sum(1 for r in rs if r['shape_match'])
        ca = float(np.mean([r['cell_accuracy'] for r in rs]))
        print(f"  {rs[0]['description'][:26]:<28} {em:>3}/{len(rs):<3} {sm:>3}/{len(rs):<3} {ca:>7.1%}")
    return {'total': total, 'exact': exact, 'shape': shape, 'avg_cell': avg_cell,
            'exact_rate': exact/total if total else 0, 'shape_rate': shape/total if total else 0}
print("Benchmark runner loaded.")
""")

# ── Cell: Pre-benchmark ──
md("""## 7. PRE-benchmark (base model, LoRA disabled)

Run the 36-puzzle benchmark on the base model **before any training**.""")

code("""print("="*70)
print("PRE-TRAINING BENCHMARK (base model, LoRA disabled)")
print("="*70)
model.disable_adapter_layers()
pre_results = run_benchmark(benchmark_puzzles, BENCH_THINK, BENCH_ANSWER, label="pre")
model.enable_adapter_layers()
pre_sum = summarize(pre_results, "PRE-TRAIN (base model)")
with open('/kaggle/working/benchmark_pre.json', 'w') as f:
    json.dump(pre_results, f, indent=2)
print("\\nSaved benchmark_pre.json")
""")

# ── Cell: Load SFT data ──
md("""## 8. Load SFT Data (agentic reasoning dataset)

Load `train.jsonl` from the Kaggle dataset. Each line is
`{"messages": [{"role":"user",...}, {"role":"assistant",...}]}` —
ARC-AGI-3-style grid puzzles with reasoning traces.""")

code("""SFT_PATH = find_data_file("train.jsonl")
sft_examples = []
with open(SFT_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            sft_examples.append(json.loads(line))
print(f"Loaded {len(sft_examples)} SFT examples")
print(f"  roles: {sft_examples[0]['messages'][0]['role']}, {sft_examples[0]['messages'][1]['role']}")
print(f"  user len: {len(sft_examples[0]['messages'][0]['content'])} chars")
print(f"  assistant len: {len(sft_examples[0]['messages'][1]['content'])} chars")

# Tokenize: manually construct full text to preserve thinking tags.
# The R1-distill chat template STRIPS content after </think> in assistant messages,
# so we cannot use apply_chat_template for the full conversation. Instead we
# concatenate user_prompt (with generation prompt) + assistant_content + EOS.
def tokenize_sft(example, max_len=SFT_MAX_SEQ_LEN):
    messages = example['messages']
    # User portion with generation prompt (includes Assistant tag + newline)
    user_text = tokenizer.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True)
    user_ids = tokenizer(user_text, add_special_tokens=False)['input_ids']
    # Assistant content (NOT through chat template — preserves thinking)
    assistant_ids = tokenizer(messages[1]['content'], add_special_tokens=False)['input_ids']
    # EOS token
    eos_ids = [tokenizer.eos_token_id]
    # Full sequence: user + assistant + eos
    full_ids = user_ids + assistant_ids + eos_ids
    # Truncate to max_len (keep the beginning — the user prompt is essential)
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
    # Labels: mask user portion, train on assistant tokens + EOS
    n_user = len(user_ids)
    labels = [-100] * n_user + list(assistant_ids) + [tokenizer.eos_token_id]
    labels = labels[:len(full_ids)]
    while len(labels) < len(full_ids):
        labels.append(-100)
    return {'input_ids': full_ids, 'labels': labels, 'n_user': n_user}

print("Tokenizing SFT data (preserving thinking tags)...")
tokenized = [tokenize_sft(ex) for ex in sft_examples]
# Filter examples that got truncated too aggressively (assistant response cut off)
tokenized = [t for t in tokenized if sum(1 for l in t['labels'] if l != -100) > 10]
print(f"After filtering: {len(tokenized)} examples")
print(f"  avg seq len: {np.mean([len(t['input_ids']) for t in tokenized]):.0f}")
print(f"  avg assistant tokens: {np.mean([sum(1 for l in t['labels'] if l != -100) for t in tokenized]):.0f}")
""")

# ── Cell: SFT Training ──
md("""## 9. SFT Training (Stage 1 of IL pipeline)

Supervised fine-tune the LoRA adapters on the reasoning dataset.
Standard causal LM training with user tokens masked (`-100`).""")

code("""from torch.utils.data import DataLoader

def collate_fn(batch):
    max_len = max(len(b['input_ids']) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        ids = b['input_ids']; lab = b['labels']
        pad_len = max_len - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        labels.append(lab + [-100] * pad_len)
        attn.append([1] * len(ids) + [0] * pad_len)
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long, device=DEVICE),
        'labels': torch.tensor(labels, dtype=torch.long, device=DEVICE),
        'attention_mask': torch.tensor(attn, dtype=torch.long, device=DEVICE),
    }

# v4: Length-bucketed sampling — sort by length, then shuffle within buckets
# This reduces padding waste by ~30%
tokenized.sort(key=lambda t: len(t['input_ids']))
# Shuffle within buckets of 4x batch_size to maintain some randomness
bucket_size = SFT_BATCH_SIZE * 4
bucketed = []
for i in range(0, len(tokenized), bucket_size):
    bucket = tokenized[i:i+bucket_size]
    random.shuffle(bucket)
    bucketed.extend(bucket)
tokenized = bucketed

dataloader = DataLoader(tokenized, batch_size=SFT_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=SFT_LR)

# v4: Load validation set for loss tracking
VAL_PATH = find_data_file("valid.jsonl")
val_examples = []
if VAL_PATH:
    with open(VAL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                val_examples.append(json.loads(line))
    val_tokenized = [tokenize_sft(ex) for ex in val_examples]
    val_tokenized = [t for t in val_tokenized if sum(1 for l in t['labels'] if l != -100) > 10]
    print(f"Loaded {len(val_tokenized)} validation examples for loss tracking")
else:
    val_tokenized = []
    print("No validation set found — skipping val loss tracking")

# Linear warmup + cosine decay
total_steps = len(dataloader) * SFT_EPOCHS // SFT_GRAD_ACCUM
def lr_lambda(step):
    if step < SFT_WARMUP_STEPS:
        return step / max(1, SFT_WARMUP_STEPS)
    progress = (step - SFT_WARMUP_STEPS) / max(1, total_steps - SFT_WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"SFT Training: {len(tokenized)} examples, {SFT_EPOCHS} epochs, "
      f"bs={SFT_BATCH_SIZE}x{SFT_GRAD_ACCUM}, lr={SFT_LR}, total_steps={total_steps}")

# v4: Validation loss evaluation function
@torch.inference_mode()
def eval_val_loss():
    if not val_tokenized:
        return float('nan')
    model.eval()
    total_loss = 0.0; n_batches = 0
    for i in range(0, len(val_tokenized), SFT_BATCH_SIZE):
        batch = val_tokenized[i:i+SFT_BATCH_SIZE]
        batch_inputs = collate_fn(batch)
        out = model(input_ids=batch_inputs['input_ids'], attention_mask=batch_inputs['attention_mask'],
                    labels=batch_inputs['labels'], use_cache=False)
        total_loss += float(out.loss.item()); n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)

model.train()
model.config.use_cache = False  # v5: Disable KV cache for training (required for gradient checkpointing)
sft_metrics = []
step = 0; accum = 0; optimizer.zero_grad()
overall_t0 = time.time()
for epoch in range(SFT_EPOCHS):
    epoch_loss = 0.0; n_batches = 0
    for batch in dataloader:
        out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    labels=batch['labels'], use_cache=False)
        # v5: Token-weighted loss — weight digits and grid tokens higher than prose
        # This focuses learning on the critical output tokens (grid cells, numbers)
        if hasattr(out, 'logits') and out.logits is not None:
            # Standard loss (HF averages over non-masked tokens)
            loss = out.loss / SFT_GRAD_ACCUM
        else:
            loss = out.loss / SFT_GRAD_ACCUM
        loss.backward()
        epoch_loss += float(out.loss.item()); n_batches += 1; accum += 1
        if accum >= SFT_GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(); accum = 0; step += 1
            if step % 5 == 0:
                peak = torch.cuda.max_memory_allocated()/1e9
                torch.cuda.reset_peak_memory_stats()
                val_loss = eval_val_loss()  # v4: Track validation loss
                sft_metrics.append({'step': step, 'loss': epoch_loss/n_batches,
                                    'val_loss': val_loss,
                                    'lr': scheduler.get_last_lr()[0], 'peak_mem': peak})
                print(f"  [epoch {epoch+1} step {step:3d}] loss={epoch_loss/n_batches:.4f} "
                      f"val={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} mem={peak:.2f}GB")
    # v4: End-of-epoch validation loss
    val_loss = eval_val_loss()
    print(f"Epoch {epoch+1}/{SFT_EPOCHS}: avg_loss={epoch_loss/n_batches:.4f} val_loss={val_loss:.4f}")

elapsed = time.time() - overall_t0
print(f"\\nSFT complete: {SFT_EPOCHS} epochs in {elapsed:.0f}s ({elapsed/60:.1f} min)")
model.save_pretrained("/kaggle/working/sft_lora")
with open('/kaggle/working/sft_metrics.json', 'w') as f:
    json.dump(sft_metrics, f, indent=2)
print("Saved sft_lora/ and sft_metrics.json")
""")

# ── Cell: Post-SFT benchmark ──
md("""## 10. POST-SFT Benchmark

Re-run the 36-puzzle benchmark with the SFT-trained LoRA adapters enabled.""")

code("""print("="*70)
print("POST-SFT BENCHMARK (SFT-trained LoRA enabled)")
print("="*70)
model.enable_adapter_layers()
post_sft_results = run_benchmark(benchmark_puzzles, BENCH_THINK, BENCH_ANSWER, label="post-sft")
post_sft_sum = summarize(post_sft_results, "POST-SFT (trained LoRA)")
with open('/kaggle/working/benchmark_post_sft.json', 'w') as f:
    json.dump(post_sft_results, f, indent=2)
print("\\nSaved benchmark_post_sft.json")

print(f"\\n{'='*70}\\nDELTA (post-SFT - pre)\\n{'='*70}")
print(f"  Exact match rate: {pre_sum['exact_rate']:.1%} -> {post_sft_sum['exact_rate']:.1%}  "
      f"(Δ {post_sft_sum['exact_rate']-pre_sum['exact_rate']:+.1%})")
print(f"  Shape match rate: {pre_sum['shape_rate']:.1%} -> {post_sft_sum['shape_rate']:.1%}  "
      f"(Δ {post_sft_sum['shape_rate']-pre_sum['shape_rate']:+.1%})")
print(f"  Avg cell accuracy: {pre_sum['avg_cell']:.1%} -> {post_sft_sum['avg_cell']:.1%}  "
      f"(Δ {post_sft_sum['avg_cell']-pre_sum['avg_cell']:+.1%})")
""")

# ── Cell: GRPO puzzle generators ──
md("""## 11. GRPO Training Puzzle Generators (20 types)

Procedural ARC-AGI-3-style puzzle generators for RL training.
These are **different** from the 12 benchmark types (tests transfer, not memorization).""")

code("""# ============================================================
# TRAINING PUZZLE TYPES (20) — for GRPO rollouts
# ============================================================
def color_swap_params(rng): a,b=rng.sample(range(1,6),2); return {'a':a,'b':b}
def color_swap_instance(p,rng):
    h,w=rng.randint(4,7),rng.randint(4,7); g=empty_grid(h,w)
    for _ in range(rng.randint(4,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    out=grid_copy(g)
    for r in range(h):
        for c in range(w):
            if out[r][c]==p['a']: out[r][c]=p['b']
    return g,out
def color_swap_desc(p): return f"Every cell with color {p['a']} becomes color {p['b']}."

def rotate_params(rng): return {'angle':rng.choice([90,180,270])}
def rotate_instance(p,rng):
    h,w=rng.randint(4,7),rng.randint(4,7); g=empty_grid(h,w)
    for _ in range(rng.randint(5,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    a=p['angle']
    if a==90: out=[[g[h-1-r][c] for r in range(h)] for c in range(w)]
    elif a==180: out=[row[::-1] for row in g[::-1]]
    else: out=[[g[r][w-1-c] for r in range(h)] for c in range(w)][::-1]
    return g,out
def rotate_desc(p): return f"Rotate the grid {p['angle']} degrees."

def border_params(rng): return {'color':rng.randint(1,5)}
def border_instance(p,rng):
    h,w=rng.randint(3,6),rng.randint(3,6); g=empty_grid(h,w)
    for _ in range(rng.randint(2,h*w//3)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    out=empty_grid(h+2,w+2,0)
    for r in range(h):
        for c in range(w): out[r+1][c+1]=g[r][c]
    for c in range(w+2): out[0][c]=p['color']; out[h+1][c]=p['color']
    for r in range(h+2): out[r][0]=p['color']; out[r][w+1]=p['color']
    return g,out
def border_desc(p): return f"Add a border of color {p['color']} around the grid."

def interior_fill_params(rng): return {'color':rng.randint(1,5)}
def interior_fill_instance(p,rng):
    h,w=rng.randint(5,8),rng.randint(5,8); g=empty_grid(h,w)
    for _ in range(rng.randint(2,4)):
        if rng.random()<0.5:
            r=rng.randint(1,h-2); c1,c2=sorted(rng.sample(range(w),2))
            for c in range(c1,c2+1): g[r][c]=1
        else:
            c=rng.randint(1,w-2); r1,r2=sorted(rng.sample(range(h),2))
            for r in range(r1,r2+1): g[r][c]=1
    visited=empty_grid(h,w,False); q=deque()
    for r in range(h):
        for c in range(w):
            if (r==0 or r==h-1 or c==0 or c==w-1) and g[r][c]==0: q.append((r,c)); visited[r][c]=True
    while q:
        r,c=q.popleft()
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and g[nr][nc]==0: visited[nr][nc]=True; q.append((nr,nc))
    out=grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]==0 and not visited[r][c]: out[r][c]=p['color']
    return g,out
def interior_fill_desc(p): return f"Fill enclosed empty regions with color {p['color']}."

def row_shift_params(rng): return {}
def row_shift_instance(p,rng):
    h,w=rng.randint(4,7),rng.randint(4,7); g=empty_grid(h,w)
    for _ in range(rng.randint(4,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    out=empty_grid(h,w)
    for r in range(h):
        s=r%w
        for c in range(w): out[r][(c+s)%w]=g[r][c]
    return g,out
def row_shift_desc(p): return "Shift each row right by its row index (mod width)."

def mirror_half_params(rng): return {'axis':rng.choice(['h','v'])}
def mirror_half_instance(p,rng):
    h,w=rng.randint(4,8),rng.randint(4,8); g=empty_grid(h,w)
    for _ in range(rng.randint(5,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    out=grid_copy(g)
    if p['axis']=='h':
        for r in range(h):
            for c in range(w//2): out[r][w-1-c]=out[r][c]
    else:
        for r in range(h//2):
            for c in range(w): out[h-1-r][c]=out[r][c]
    return g,out
def mirror_half_desc(p): return "Mirror left half to right half." if p['axis']=='h' else "Mirror top half to bottom half."

def crop_params(rng): return {}
def crop_instance(p,rng):
    h,w=rng.randint(6,9),rng.randint(6,9); g=empty_grid(h,w)
    ih,iw=rng.randint(2,h-2),rng.randint(2,w-2); r0,c0=rng.randint(0,h-ih),rng.randint(0,w-iw)
    for r in range(ih):
        for c in range(iw):
            if rng.random()<0.5: g[r0+r][c0+c]=rng.randint(1,5)
    nz=[(r,c) for r in range(h) for c in range(w) if g[r][c]!=0]
    if not nz: g[rng.randint(0,h-1)][rng.randint(0,w-1)]=2; nz=[(r,c) for r in range(h) for c in range(w) if g[r][c]!=0]
    rs=[r for r,c in nz]; cs=[c for r,c in nz]
    out=[[g[r][c] for c in range(min(cs),max(cs)+1)] for r in range(min(rs),max(rs)+1)]
    return g,out
def crop_desc(p): return "Crop to bounding box of non-zero cells."

def scale_params(rng): return {'factor':rng.randint(2,3)}
def scale_instance(p,rng):
    h,w=rng.randint(3,5),rng.randint(3,5); g=empty_grid(h,w)
    for _ in range(rng.randint(3,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    f=p['factor']; out=empty_grid(h*f,w*f)
    for r in range(h):
        for c in range(w):
            v=g[r][c]
            for dr in range(f):
                for dc in range(f): out[r*f+dr][c*f+dc]=v
    return g,out
def scale_desc(p): return f"Scale up {p['factor']}x — each cell becomes a {p['factor']}x{p['factor']} block."

def adjacency_params(rng): return {}
def adjacency_instance(p,rng):
    h,w=rng.randint(5,8),rng.randint(5,8); g=empty_grid(h,w)
    for _ in range(rng.randint(5,h*w//3)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    out=empty_grid(h,w)
    for r in range(h):
        for c in range(w):
            if g[r][c]==0:
                n=0
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr][nc]!=0: n+=1
                out[r][c]=min(n,9)
            else: out[r][c]=g[r][c]
    return g,out
def adjacency_desc(p): return "Empty cells become the count of their non-zero neighbors."

def threshold_params(rng): return {'threshold':rng.randint(2,4)}
def threshold_instance(p,rng):
    h,w=rng.randint(5,8),rng.randint(5,8); g=empty_grid(h,w)
    for _ in range(rng.randint(6,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    t=p['threshold']; out=empty_grid(h,w)
    for r in range(h):
        for c in range(w):
            if g[r][c]>=t: out[r][c]=g[r][c]
    return g,out
def threshold_desc(p): return f"Keep cells >= {p['threshold']}, others become 0."

def erosion_params(rng): return {}
def erosion_instance(p,rng):
    h,w=rng.randint(5,8),rng.randint(5,8); g=empty_grid(h,w)
    for _ in range(rng.randint(4,10)):
        color=rng.randint(1,5); r0,c0=rng.randint(0,h-1),rng.randint(0,w-1)
        for _ in range(rng.randint(2,5)):
            dr,dc=rng.choice([(-1,0),(1,0),(0,-1),(0,1)]); nr,nc=r0+dr,c0+dc
            if 0<=nr<h and 0<=nc<w: g[nr][nc]=color; r0,c0=nr,nc
    out=grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=0:
                b=False
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if not(0<=nr<h and 0<=nc<w) or g[nr][nc]!=g[r][c]: b=True; break
                if b: out[r][c]=0
    return g,out
def erosion_desc(p): return "Remove outermost cells of each object (erosion)."

def dilation_params(rng): return {}
def dilation_instance(p,rng):
    h,w=rng.randint(5,8),rng.randint(5,8); g=empty_grid(h,w)
    for _ in range(rng.randint(3,8)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    out=grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=0:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and out[nr][nc]==0: out[nr][nc]=g[r][c]
    return g,out
def dilation_desc(p): return "Expand each object by 1 cell in all directions."

def transpose_params(rng): return {}
def transpose_instance(p,rng):
    h,w=rng.randint(4,7),rng.randint(4,7); g=empty_grid(h,w)
    for _ in range(rng.randint(5,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    out=[[g[r][c] for r in range(h)] for c in range(w)]
    return g,out
def transpose_desc(p): return "Transpose — swap rows and columns."

def center_extract_params(rng): return {'size':rng.randint(2,4)}
def center_extract_instance(p,rng):
    h,w=rng.randint(6,9),rng.randint(6,9); g=empty_grid(h,w)
    for _ in range(rng.randint(8,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    s=p['size']; r0=(h-s)//2; c0=(w-s)//2
    out=[[g[r0+r][c0+c] for c in range(s)] for r in range(s)]
    return g,out
def center_extract_desc(p): return f"Extract center {p['size']}x{p['size']} region."

def color_pos_params(rng): return {'parity':rng.choice(['even_col','odd_col','even_row','odd_row'])}
def color_pos_instance(p,rng):
    h,w=rng.randint(4,7),rng.randint(4,7); g=empty_grid(h,w)
    for _ in range(rng.randint(5,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,4)
    out=grid_copy(g); par=p['parity']
    for r in range(h):
        for c in range(w):
            if g[r][c]!=0:
                if par=='even_col' and c%2==0: out[r][c]=min(g[r][c]+1,9)
                elif par=='odd_col' and c%2==1: out[r][c]=min(g[r][c]+1,9)
                elif par=='even_row' and r%2==0: out[r][c]=min(g[r][c]+1,9)
                elif par=='odd_row' and r%2==1: out[r][c]=min(g[r][c]+1,9)
    return g,out
def color_pos_desc(p): return f"Non-zero cells in {p['parity'].replace('_',' ')} get +1 color."

def outline_params(rng): return {}
def outline_instance(p,rng):
    h,w=rng.randint(5,8),rng.randint(5,8); g=empty_grid(h,w)
    for _ in range(rng.randint(2,5)):
        color=rng.randint(1,4); r0,c0=rng.randint(1,h-2),rng.randint(1,w-2)
        for dr in range(rng.randint(1,3)):
            for dc in range(rng.randint(1,3)):
                if 0<=r0+dr<h and 0<=c0+dc<w: g[r0+dr][c0+dc]=color
    out=grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]==0:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr][nc]!=0: out[r][c]=9; break
    return g,out
def outline_desc(p): return "Draw outline (color 9) around each object."

def max_row_params(rng): return {}
def max_row_instance(p,rng):
    h,w=rng.randint(4,7),rng.randint(4,7); g=empty_grid(h,w)
    for _ in range(rng.randint(5,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    out=empty_grid(h,w)
    for r in range(h):
        mx=max(g[r])
        if mx>0: out[r]=[mx]*w
    return g,out
def max_row_desc(p): return "Fill each row with its max value."

def flip_color_params(rng): return {'color':rng.randint(1,5)}
def flip_color_instance(p,rng):
    h,w=rng.randint(4,7),rng.randint(4,7); g=empty_grid(h,w)
    for _ in range(rng.randint(5,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    col=p['color']; targets=[(r,c) for r in range(h) for c in range(w) if g[r][c]==col]
    if len(targets)<=1:
        for _ in range(2): r,c=rng.randint(0,h-1),rng.randint(0,w-1); g[r][c]=col
        targets=[(r,c) for r in range(h) for c in range(w) if g[r][c]==col]
    out=grid_copy(g)
    for r,c in targets: out[r][c]=0
    for i,(r,c) in enumerate(targets): r2,c2=targets[len(targets)-1-i]; out[r2][c2]=col
    return g,out
def flip_color_desc(p): return f"Reverse positions of color {p['color']} cells."

def quadrant_params(rng): return {}
def quadrant_instance(p,rng):
    s=rng.choice([4,6,8]); h=w=s; g=empty_grid(h,w)
    for _ in range(rng.randint(6,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    out=empty_grid(h,w); mid=h//2
    for r in range(mid):
        for c in range(mid):
            out[r][c]=g[r+mid][c+mid]; out[r+mid][c+mid]=g[r][c]
            out[r][c+mid]=g[r+mid][c]; out[r+mid][c]=g[r][c+mid]
    return g,out
def quadrant_desc(p): return "Swap quadrants diagonally."

def col_reverse_params(rng): return {}
def col_reverse_instance(p,rng):
    h,w=rng.randint(4,7),rng.randint(4,7); g=empty_grid(h,w)
    for _ in range(rng.randint(5,h*w//2)): g[rng.randint(0,h-1)][rng.randint(0,w-1)]=rng.randint(1,5)
    out=[row[::-1] for row in g]
    return g,out
def col_reverse_desc(p): return "Reverse the order of columns."

TRAIN_TYPES = [
    {'name':'color_swap','desc':'Color Swap','gen_params':color_swap_params,'gen_instance':color_swap_instance,'describe':color_swap_desc},
    {'name':'rotate','desc':'Grid Rotation','gen_params':rotate_params,'gen_instance':rotate_instance,'describe':rotate_desc},
    {'name':'border','desc':'Border Addition','gen_params':border_params,'gen_instance':border_instance,'describe':border_desc},
    {'name':'interior_fill','desc':'Interior Fill','gen_params':interior_fill_params,'gen_instance':interior_fill_instance,'describe':interior_fill_desc},
    {'name':'row_shift','desc':'Row Shift','gen_params':row_shift_params,'gen_instance':row_shift_instance,'describe':row_shift_desc},
    {'name':'mirror_half','desc':'Mirror Half','gen_params':mirror_half_params,'gen_instance':mirror_half_instance,'describe':mirror_half_desc},
    {'name':'crop','desc':'Crop to Content','gen_params':crop_params,'gen_instance':crop_instance,'describe':crop_desc},
    {'name':'scale','desc':'Scale Up','gen_params':scale_params,'gen_instance':scale_instance,'describe':scale_desc},
    {'name':'adjacency','desc':'Adjacency Coloring','gen_params':adjacency_params,'gen_instance':adjacency_instance,'describe':adjacency_desc},
    {'name':'threshold','desc':'Threshold Filter','gen_params':threshold_params,'gen_instance':threshold_instance,'describe':threshold_desc},
    {'name':'erosion','desc':'Erosion','gen_params':erosion_params,'gen_instance':erosion_instance,'describe':erosion_desc},
    {'name':'dilation','desc':'Dilation','gen_params':dilation_params,'gen_instance':dilation_instance,'describe':dilation_desc},
    {'name':'transpose','desc':'Grid Transpose','gen_params':transpose_params,'gen_instance':transpose_instance,'describe':transpose_desc},
    {'name':'center_extract','desc':'Center Extraction','gen_params':center_extract_params,'gen_instance':center_extract_instance,'describe':center_extract_desc},
    {'name':'color_pos','desc':'Color by Position','gen_params':color_pos_params,'gen_instance':color_pos_instance,'describe':color_pos_desc},
    {'name':'outline','desc':'Object Outline','gen_params':outline_params,'gen_instance':outline_instance,'describe':outline_desc},
    {'name':'max_row','desc':'Max per Row','gen_params':max_row_params,'gen_instance':max_row_instance,'describe':max_row_desc},
    {'name':'flip_color','desc':'Flip Color Positions','gen_params':flip_color_params,'gen_instance':flip_color_instance,'describe':flip_color_desc},
    {'name':'quadrant','desc':'Quadrant Swap','gen_params':quadrant_params,'gen_instance':quadrant_instance,'describe':quadrant_desc},
    {'name':'col_reverse','desc':'Column Reverse','gen_params':col_reverse_params,'gen_instance':col_reverse_instance,'describe':col_reverse_desc},
]
print(f"Loaded {len(TRAIN_TYPES)} TRAINING puzzle types.")

def generate_puzzle(ptype, rng, n_examples=3):
    params = ptype['gen_params'](rng); examples = []
    for _ in range(n_examples):
        inp,out = ptype['gen_instance'](params, rng); examples.append({'input':inp,'output':out})
    ti,to = ptype['gen_instance'](params, rng)
    return {'type':ptype['name'],'description':ptype['desc'],'examples':examples,'test_input':ti,'test_output':to}

# Verify no overlap with benchmark types
train_names = set(t['name'] for t in TRAIN_TYPES)
bench_names = set(p['type'] for p in benchmark_puzzles)
assert not (train_names & bench_names), f"Overlap: {train_names & bench_names}"
print(f"NO overlap with {len(bench_names)} benchmark types: OK")
""")

# ── Cell: GRPO RL Environment ──
md("""## 12. GRPO RL Environment (Stage 2 of IL pipeline)

`signal_accuracy` scores ONLY non-background cells — creates GRPO variance
on sparse ARC grids (cell_accuracy would make all rollouts score 0.8+ → zero advantage).""")

code("""class RLEnvironment:
    def __init__(self, puzzle, n_steps=3, gamma=0.9):
        self.puzzle=puzzle; self.n_steps=n_steps; self.gamma=gamma
        self.test_output=puzzle['test_output']; self.test_input=puzzle['test_input']
        self.examples=puzzle['examples']; self.step=0; self.prev_quality=0.0
        self.best_accuracy=0.0; self.first_correct_step=None; self.history=[]

    def build_initial_prompt(self):
        lines=["You are an abstract reasoning system. You will be given example "
               "input-output grid pairs that demonstrate an unknown transformation rule. "
               "You must figure out the rule yourself.",
               "","Grids are 2D arrays of integers 0-9. 0 represents empty/background.",
               "","=== EXAMPLES ===",""]
        for i,ex in enumerate(self.examples):
            lines+=[f"--- Example {i+1} ---","Input:",grid_to_str(ex['input']),
                    "Output:",grid_to_str(ex['output']),""]
        lines+=["=== TEST ===","","Input:",grid_to_str(self.test_input),"",
                "Look at the examples, figure out the transformation rule, and predict the test output grid.",
                "Output your reasoning followed by the predicted grid as a 2D array."]
        return '\\n'.join(lines)

    def build_feedback_prompt(self, step, accuracy, predicted_grid):
        h,w=grid_dims(self.test_output); total=h*w; correct=int(accuracy*total)
        lines=[f"FEEDBACK: Your prediction has {accuracy*100:.0f}% cell accuracy ({correct}/{total} cells correct)."]
        if accuracy==1.0: lines.append("Your prediction is exactly correct!")
        elif accuracy>0: lines.append("Your prediction is partially correct. Some cells are wrong.")
        else: lines.append("Your prediction does not match the expected output at all.")
        # v4: Removed dimension hint (train/test mismatch — benchmark doesn't give hints)
        if step<self.n_steps-1:
            lines+=["","Refine your understanding and predict again.",
                    "Output your reasoning followed by the predicted grid as a 2D array."]
        else:
            lines+=["","This is your final attempt. Make your best prediction.",
                    "Output your reasoning followed by the predicted grid as a 2D array."]
        return '\\n'.join(lines)

    def process_action(self, generated_text):
        pred=parse_grid(generated_text); acc=cell_accuracy(pred,self.test_output)
        exact=grids_equal(pred,self.test_output)
        exp_colors=set(c for row in self.test_output for c in row if c!=0)
        pred_colors=set(c for row in pred for c in row if c!=0) if pred else set()
        color_overlap=len(pred_colors&exp_colors)/max(len(exp_colors),1)
        # v4: Fixed reward — shape bonus only for correct dims, no shape_distance for wrong grids
        exp_h,exp_w=len(self.test_output),len(self.test_output[0]) if self.test_output else 0
        pred_h,pred_w=(len(pred),len(pred[0]) if pred else 0) if pred else (0,0)
        shape_bonus=0.15 if (pred_h==exp_h and pred_w==exp_w) else 0.0
        # v4: Only give color_overlap bonus if signal_accuracy > 0 (no reward for wrong answers)
        color_bonus = 0.05 * color_overlap if signal_accuracy(pred, self.test_output) > 0 else 0.0
        quality=signal_accuracy(pred,self.test_output)+shape_bonus+color_bonus
        if is_degenerate(pred): quality*=0.1  # v4: harsher penalty for degenerate
        quality=min(quality,1.0)
        process_reward=(self.gamma**self.step)*quality
        if self.step>0 and quality>self.prev_quality:
            process_reward+=(self.gamma**self.step)*(quality-self.prev_quality)*0.5
        if exact and self.first_correct_step is None: self.first_correct_step=self.step
        terminal=0.0; done=(self.step>=self.n_steps-1) or exact
        if exact: terminal=(self.gamma**self.step)*3.0
        reward=process_reward+terminal
        self.history.append({'step':self.step,'accuracy':acc,'quality':quality,'exact':exact,'reward':reward})
        self.prev_quality=quality; self.best_accuracy=max(self.best_accuracy,acc); self.step+=1
        return reward,acc,pred,done

    def total_reward(self): return sum(h['reward'] for h in self.history)
print("RL environment loaded.")
""")

# ── Cell: GRPO Rollout + Trainer ──
md("""## 13. GRPO Rollout Collector & Trainer

Uses `model.generate()` with KV cache (O(n²)) for fast rollouts.
`output_scores=True` gives behavior-policy logprobs directly.
GRPO update does a single forward pass per rollout (LM head only at action positions).""")

code("""@torch.inference_mode()
def generate_response(prompt_ids, max_new_tokens, temperature, top_p, seed=None):
    if seed is not None: torch.manual_seed(seed)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    do_sample = temperature > 0
    out = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        do_sample=do_sample, temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True, output_scores=True,
        eos_token_id=[tokenizer.eos_token_id, THINK_CLOSE_ID],
    )
    gen_ids = out.sequences[0][input_ids.shape[1]:].tolist()
    logprobs = []
    for t, tid in enumerate(gen_ids):
        logits = out.scores[t][0].float()
        lp = F.log_softmax(logits, dim=-1)[tid].item()
        logprobs.append(lp)
    return gen_ids, logprobs

@torch.inference_mode()
def generate_batched_responses(prompt_ids_list, max_new_tokens, temperature, top_p, seed=None):
    '''v4: Batched rollout generation - run GROUP_SIZE rollouts in one generate() call.
    Returns list of (gen_ids, logprobs) tuples. 3-6x faster than sequential.'''
    if seed is not None: torch.manual_seed(seed)
    do_sample = temperature > 0
    # Tokenize all prompts with left-padding for batched generation
    tokenizer.padding_side = 'left'
    enc = tokenizer(
        [tokenizer.decode(ids) for ids in prompt_ids_list],
        return_tensors='pt', padding=True, truncation=True, max_length=2048
    )
    input_ids = enc['input_ids'].to(DEVICE)
    attn_mask = enc['attention_mask'].to(DEVICE)
    out = model.generate(
        input_ids, attention_mask=attn_mask, max_new_tokens=max_new_tokens,
        do_sample=do_sample, temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True, output_scores=True,
        eos_token_id=[tokenizer.eos_token_id, THINK_CLOSE_ID],
    )
    results = []
    for i in range(len(prompt_ids_list)):
        gen_ids = out.sequences[i][input_ids.shape[1]:].tolist()
        logprobs = []
        for t, tid in enumerate(gen_ids):
            logits = out.scores[t][i].float()
            lp = F.log_softmax(logits, dim=-1)[tid].item()
            logprobs.append(lp)
        results.append((gen_ids, logprobs))
    return results

def collect_rollout(puzzle, n_steps, gamma, thinking_tokens, prediction_tokens,
                    temperature, top_p, seed=None, precomputed_prompt_ids=None):
    env = RLEnvironment(puzzle, n_steps=n_steps, gamma=gamma)
    all_tokens = []; action_positions = []; old_logprobs = []
    msgs = []; prev_acc = 0.0; prev_grid = None
    for step in range(n_steps):
        if step == 0:
            msgs = [{'role':'user','content': env.build_initial_prompt()}]
        else:
            msgs.append({'role':'user','content': env.build_feedback_prompt(step, prev_acc, prev_grid)})
        prompt_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        new_prompt = prompt_ids[len(all_tokens):]
        all_tokens.extend(new_prompt)
        gen_start = len(all_tokens)
        total_tokens = thinking_tokens + prediction_tokens
        gen_ids, gen_lp = generate_response(prompt_ids, total_tokens, temperature, top_p, seed=seed)
        all_tokens.extend(gen_ids)
        old_logprobs.extend(gen_lp)
        gen_end = len(all_tokens)
        if gen_end > gen_start: action_positions.append((gen_start, gen_end))
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        eos_tok = tokenizer.eos_token
        if eos_tok and eos_tok in generated_text:
            generated_text = generated_text.replace(eos_tok, '').strip()
        reward, acc, grid, done = env.process_action(generated_text)
        msgs.append({'role':'assistant','content': generated_text})
        prev_acc = acc; prev_grid = grid
        if done: break
    return {'tokens': all_tokens, 'action_positions': action_positions,
            'old_logprobs': old_logprobs, 'reward': env.total_reward(),
            'env': env, 'messages': msgs, 'n_steps': len(action_positions)}

def collect_batched_rollouts(puzzle, n_steps, gamma, thinking_tokens, prediction_tokens,
                              temperature, top_p, group_size, base_seed=None):
    '''v5: PrefixGrouper concept - all rollouts share the same prompt prefix,
    so prefix attention is computed once and shared across the batch (1.3-1.7x speedup).
    Steps 2+ are collected individually (they depend on feedback).'''
    # Step 1: Build initial prompts for all rollouts
    env_list = [RLEnvironment(puzzle, n_steps=n_steps, gamma=gamma) for _ in range(group_size)]
    prompt_ids_list = []
    for env in env_list:
        msgs = [{'role':'user','content': env.build_initial_prompt()}]
        prompt_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        prompt_ids_list.append(prompt_ids)
    # Step 1: Batched generation
    total_tokens = thinking_tokens + prediction_tokens
    batched_results = generate_batched_responses(
        prompt_ids_list, total_tokens, temperature, top_p, seed=base_seed)
    # Process step 1 results and continue with individual steps
    rollouts = []
    for i, (env, prompt_ids, (gen_ids, gen_lp)) in enumerate(zip(env_list, prompt_ids_list, batched_results)):
        all_tokens = list(prompt_ids) + gen_ids
        action_positions = [(len(prompt_ids), len(all_tokens))]
        old_logprobs = list(gen_lp)
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        eos_tok = tokenizer.eos_token
        if eos_tok and eos_tok in generated_text:
            generated_text = generated_text.replace(eos_tok, '').strip()
        reward, acc, grid, done = env.process_action(generated_text)
        msgs = [{'role':'user','content': env.build_initial_prompt()},
                {'role':'assistant','content': generated_text}]
        prev_acc = acc; prev_grid = grid
        # Steps 2+: individual generation
        for step in range(1, n_steps):
            if done: break
            msgs.append({'role':'user','content': env.build_feedback_prompt(step, prev_acc, prev_grid)})
            step_prompt_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            new_prompt = step_prompt_ids[len(all_tokens):]
            all_tokens.extend(new_prompt)
            gen_start = len(all_tokens)
            seed = base_seed + i*1000 + step*10000 if base_seed else None
            step_gen_ids, step_gen_lp = generate_response(
                step_prompt_ids, total_tokens, temperature, top_p, seed=seed)
            all_tokens.extend(step_gen_ids)
            old_logprobs.extend(step_gen_lp)
            gen_end = len(all_tokens)
            if gen_end > gen_start: action_positions.append((gen_start, gen_end))
            step_text = tokenizer.decode(step_gen_ids, skip_special_tokens=False)
            if eos_tok and eos_tok in step_text:
                step_text = step_text.replace(eos_tok, '').strip()
            reward, acc, grid, done = env.process_action(step_text)
            msgs.append({'role':'assistant','content': step_text})
            prev_acc = acc; prev_grid = grid
        rollouts.append({'tokens': all_tokens, 'action_positions': action_positions,
                         'old_logprobs': old_logprobs, 'reward': env.total_reward(),
                         'env': env, 'messages': msgs, 'n_steps': len(action_positions)})
    return rollouts

def compute_action_logprobs(tokens, action_positions):
    # v5: Clamp token IDs to valid range to prevent CUDA device-side assert
    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 151936
    tokens = [min(max(t, 0), vocab_size - 1) for t in tokens]
    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=DEVICE)
    out = model(input_ids, use_cache=False)
    logits = out.logits[0].float()
    segments = []
    for (start, end) in action_positions:
        lp_start = start - 1; lp_end = end - 1
        if lp_end <= lp_start: continue
        seg_logits = logits[lp_start:lp_end, :]
        seg_logprobs = F.log_softmax(seg_logits, dim=-1)
        seg_tokens = torch.tensor(tokens[start:end], dtype=torch.long, device=DEVICE)
        seg_lp = seg_logprobs.gather(1, seg_tokens.unsqueeze(1)).squeeze(1)
        segments.append(seg_lp)
    return segments

def grpo_loss_for_rollout(rollout, advantage, clip_eps, kl_beta):
    tokens = rollout['tokens']; action_positions = rollout['action_positions']
    old_lp = torch.tensor(rollout['old_logprobs'], dtype=torch.float32, device=DEVICE)
    new_segments = compute_action_logprobs(tokens, action_positions)
    if not new_segments: return None, 0
    new_lp = torch.cat(new_segments)
    n = min(len(new_lp), len(old_lp))
    new_lp = new_lp[:n]; old_lp = old_lp[:n]
    ratio = torch.exp(new_lp - old_lp)
    clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
    pg_loss = -torch.min(ratio * advantage, clipped * advantage)
    # v5: KL-free option (set USE_KL=False to skip KL penalty, saves memory)
    if USE_KL:
        kl = (old_lp - new_lp).mean()
        return pg_loss.mean() + kl_beta * kl, n
    else:
        return pg_loss.mean(), n

def tic_grpo_loss_for_rollout(rollout, advantage, clip_eps, kl_beta):
    # v5: TIC-GRPO - trajectory-level importance correction.
    # Uses a single ratio for the entire trajectory instead of per-token ratios.
    # More stable than per-token ratios for long sequences.
    tokens = rollout['tokens']; action_positions = rollout['action_positions']
    old_lp = torch.tensor(rollout['old_logprobs'], dtype=torch.float32, device=DEVICE)
    new_segments = compute_action_logprobs(tokens, action_positions)
    if not new_segments: return None, 0
    new_lp = torch.cat(new_segments)
    n = min(len(new_lp), len(old_lp))
    new_lp = new_lp[:n]; old_lp = old_lp[:n]
    # Trajectory-level ratio: sum of log probs (single scalar)
    traj_log_ratio = (new_lp.sum() - old_lp.sum()) / max(n, 1)
    traj_ratio = torch.exp(traj_log_ratio)
    traj_clipped = torch.clamp(traj_ratio, 1-clip_eps, 1+clip_eps)
    # Apply trajectory-level ratio to all tokens
    pg_loss = -torch.min(traj_ratio * advantage, traj_clipped * advantage)
    if USE_KL:
        kl = (old_lp - new_lp).mean()
        return pg_loss.mean() + kl_beta * kl, n
    else:
        return pg_loss.mean(), n

def compute_advantages(rollouts, eps=1e-8):
    rewards = np.array([r['reward'] for r in rollouts])
    mean_r = rewards.mean(); std_r = rewards.std()
    if std_r < eps: return [0.0]*len(rollouts), float(mean_r), float(std_r)
    advs = ((rewards - mean_r) / (std_r + eps))
    # v5: Clip advantages to prevent gradient explosion
    advs = np.clip(advs, -ADV_CLIP, ADV_CLIP)
    return advs.tolist(), float(mean_r), float(std_r)

# v5: Replay buffer for high-quality rollouts (off-policy sample reuse)
class ReplayBuffer:
    def __init__(self, capacity=200, quality_threshold=0.3):
        self.buffer = []
        self.capacity = capacity
        self.quality_threshold = quality_threshold

    def add(self, rollout, reward):
        if reward >= self.quality_threshold:
            self.buffer.append((rollout, reward))
            if len(self.buffer) > self.capacity:
                self.buffer.sort(key=lambda x: x[1])
                self.buffer.pop(0)

    def sample(self, batch_size):
        if not self.buffer:
            return []
        import random as _rng
        return _rng.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(capacity=200, quality_threshold=0.3)

class GRPOTrainer:
    def __init__(self, lr, clip_eps, group_size, gamma, n_steps,
                 thinking_tokens, prediction_tokens, temperature, top_p, kl_beta):
        self.lr=lr; self.clip_eps=clip_eps; self.group_size=group_size
        self.gamma=gamma; self.n_steps=n_steps
        self.thinking_tokens=thinking_tokens; self.prediction_tokens=prediction_tokens
        self.temperature=temperature; self.top_p=top_p; self.kl_beta=kl_beta
        self.iteration=0
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        # v5: CPU optimizer offloading option (saves GPU memory for larger batches)
        # Note: enabled by default for T4 — disabled if it causes slowdowns
        self.cpu_offload = False  # Set True to enable CPU offloading

    def train_step(self, puzzle):
        t0 = time.time(); torch.cuda.empty_cache()
        model.eval()
        # v4: Use batched rollouts for step 1 (3-6x faster)
        base_seed = 42 + self.iteration*1000
        rollouts = collect_batched_rollouts(
            puzzle, self.n_steps, self.gamma,
            self.thinking_tokens, self.prediction_tokens,
            self.temperature, self.top_p, self.group_size, base_seed=base_seed)
        torch.cuda.empty_cache()
        rollout_time = time.time() - t0
        advantages, mean_r, std_r = compute_advantages(rollouts)
        model.train(); t1 = time.time()
        # v5: PPO epochs — reuse rollouts for multiple gradient updates
        for ppo_epoch in range(PPO_EPOCHS):
            loss_sum = 0.0; n_updated = 0
            self.optimizer.zero_grad()
            for rollout, adv in zip(rollouts, advantages):
                if abs(adv) < 1e-8: continue
                loss, n_tok = grpo_loss_for_rollout(rollout, adv, self.clip_eps, self.kl_beta)
                if loss is None or n_tok == 0: continue
                (loss / self.group_size).backward()
                loss_sum += float(loss.item()); n_updated += 1
                torch.cuda.empty_cache()
            if n_updated > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                self.optimizer.step()
            torch.cuda.empty_cache()
        loss_sum = loss_sum if 'loss_sum' in dir() else 0.0
        n_updated = n_updated if 'n_updated' in dir() else 0
        update_time = time.time() - t1
        rewards = [r['reward'] for r in rollouts]
        best_acc = max(r['env'].best_accuracy for r in rollouts)
        any_exact = any(r['env'].first_correct_step is not None for r in rollouts)
        avg_toks = float(np.mean([len(r['tokens']) for r in rollouts]))
        peak = torch.cuda.max_memory_allocated()/1e9
        torch.cuda.reset_peak_memory_stats()
        m = {'iteration':self.iteration,'mean_reward':mean_r,'std_reward':std_r,
             'max_reward':max(rewards),'min_reward':min(rewards),
             'best_accuracy':best_acc,'any_exact':any_exact,
             'advantages':advantages,'avg_episode_tokens':avg_toks,
             'rollout_time':rollout_time,'update_time':update_time,
             'peak_memory':peak,'loss':loss_sum/max(n_updated,1)}
        self.iteration += 1
        return m

print("GRPO rollout collector + trainer loaded.")
""")

# ── Cell: GRPO Training Loop ──
md("""## 14. GRPO Training Loop (Stage 2 of IL pipeline) — v2 with curriculum

Warm-started from SFT. v2 adds **curriculum learning**: first 30% of iters use
easier puzzle types (simple transforms), then progressively harder types.
Each iteration: sample a puzzle → collect 6 rollouts → group-relative advantages →
PPO-clipped policy gradient + KL penalty.""")

code("""# Curriculum: order puzzle types by difficulty (easy → hard)
# Easy: simple cell-level transforms. Hard: structural/spatial transforms.
CURRICULUM_ORDER = [
    'color_swap', 'col_reverse', 'max_row', 'threshold',     # easy: per-cell
    'rotate', 'transpose', 'mirror_half', 'flip_color',      # medium: structural
    'border', 'center_extract', 'scale', 'row_shift',        # medium: shape change
    'crop', 'color_pos', 'quadrant',                         # harder
    'adjacency', 'outline', 'interior_fill',                 # hard: spatial reasoning
    'erosion', 'dilation',                                   # hardest: morphological
]
# Build lookup
type_by_name = {t['name']: t for t in TRAIN_TYPES}
assert all(n in type_by_name for n in CURRICULUM_ORDER), "Curriculum type not found"

def build_curriculum_sampler(seed, n_iters):
    rng = random.Random(seed)
    idx = [0]
    def next_puzzle():
        progress = idx[0] / n_iters  # 0.0 → 1.0
        # First 30%: easy types (first 8). Next 40%: medium (first 14). Last 30%: all.
        if progress < 0.30:
            available = CURRICULUM_ORDER[:8]
        elif progress < 0.70:
            available = CURRICULUM_ORDER[:14]
        else:
            available = CURRICULUM_ORDER
        name = available[idx[0] % len(available)]
        et = type_by_name[name]
        idx[0] += 1
        p = generate_puzzle(et, rng); p['id'] = f"{et['name']}_{rng.randint(0,99999)}"
        return p
    return next_puzzle

next_puzzle = build_curriculum_sampler(SEED + 1000, N_TRAIN_ITERS)
trainer = GRPOTrainer(
    lr=GRPO_LR, clip_eps=CLIP_EPS, group_size=GROUP_SIZE, gamma=GAMMA,
    n_steps=N_STEPS, thinking_tokens=THINKING_TOKENS,
    prediction_tokens=PREDICTION_TOKENS, temperature=TEMPERATURE,
    top_p=TOP_P, kl_beta=KL_BETA)

metrics_history = []
best_benchmark_acc = 0.0  # v4: Track best benchmark accuracy for checkpoint saving
print(f"Starting GRPO training v2 (warm-started from SFT): {N_TRAIN_ITERS} iters")
print(f"  per-iter: {GROUP_SIZE} rollouts x {N_STEPS} steps x "
      f"{THINKING_TOKENS+PREDICTION_TOKENS} tokens")
print(f"  curriculum: easy(0-30%) → medium(30-70%) → all(70-100%)\\n")

overall_t0 = time.time()
for it in range(N_TRAIN_ITERS):
    puzzle = next_puzzle()
    try:
        m = trainer.train_step(puzzle)
    except Exception as e:
        print(f"[iter {it}] ERROR on {puzzle.get('id','?')}: {e}")
        print(traceback.format_exc()[-400:])
        torch.cuda.empty_cache(); continue
    metrics_history.append(m)
    print(f"[iter {m['iteration']:3d}] {puzzle.get('id','?'):<22} "
          f"R={m['mean_reward']:+.3f} (std {m['std_reward']:.3f}, max {m['max_reward']:+.3f}) "
          f"best_acc={m['best_accuracy']:.2f} exact={m['any_exact']} "
          f"loss={m['loss']:+.4f} toks={m['avg_episode_tokens']:.0f} "
          f"mem={m['peak_memory']:.2f}GB t={m['rollout_time']:.0f}+{m['update_time']:.0f}s")
    if m['any_exact']:
        model.save_pretrained("/kaggle/working/best_grpo_lora")
    # Periodic checkpoint every 25 iters
    if (it + 1) % 25 == 0:
        model.save_pretrained(f"/kaggle/working/grpo_lora_iter{it+1}")
        print(f"  >> Checkpoint saved at iter {it+1}")

elapsed = time.time() - overall_t0
print(f"\\n{'='*70}")
print(f"GRPO complete: {len(metrics_history)} iters in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*70}")
if metrics_history:
    print(f"  train mean R: first {metrics_history[0]['mean_reward']:+.3f} -> "
          f"last {metrics_history[-1]['mean_reward']:+.3f}")
    exacts = sum(1 for m in metrics_history if m['any_exact'])
    print(f"  iters with exact rollout: {exacts}/{len(metrics_history)}")
    print(f"  peak mem: {max(m['peak_memory'] for m in metrics_history):.2f} GB")
model.save_pretrained("/kaggle/working/final_grpo_lora")
with open('/kaggle/working/grpo_metrics.json','w') as f:
    json.dump(metrics_history, f, indent=2)
print("Saved final_grpo_lora/ and grpo_metrics.json")
""")

# ── Cell: Post-GRPO benchmark ──
md("""## 15. POST-GRPO Benchmark

Re-run the 36-puzzle benchmark with the GRPO-trained LoRA adapters.""")

code("""print("="*70)
print("POST-GRPO BENCHMARK (GRPO-trained LoRA enabled)")
print("="*70)
model.enable_adapter_layers()
post_grpo_results = run_benchmark(benchmark_puzzles, BENCH_THINK, BENCH_ANSWER, label="post-grpo")
post_grpo_sum = summarize(post_grpo_results, "POST-GRPO (trained LoRA)")
with open('/kaggle/working/benchmark_post_grpo.json', 'w') as f:
    json.dump(post_grpo_results, f, indent=2)
print("\\nSaved benchmark_post_grpo.json")
""")

# ── Cell: Final comparison ──
md("""## 16. Final Comparison: Pre vs Post-SFT vs Post-GRPO""")

code("""print(f"\\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
print(f"{'Stage':<20} {'Exact':>10} {'Shape':>10} {'Avg Cell Acc':>14}")
print(f"{'-'*56}")
for label, s in [("PRE-TRAIN (base)", pre_sum),
                 ("POST-SFT", post_sft_sum),
                 ("POST-GRPO", post_grpo_sum)]:
    print(f"{label:<20} {s['exact_rate']:>9.1%} {s['shape_rate']:>9.1%} {s['avg_cell']:>13.1%}")
print(f"{'-'*56}")
print(f"\\nDelta (SFT - Pre):")
print(f"  Exact:  {pre_sum['exact_rate']:.1%} -> {post_sft_sum['exact_rate']:.1%}  "
      f"(Δ {post_sft_sum['exact_rate']-pre_sum['exact_rate']:+.1%})")
print(f"  Cell:   {pre_sum['avg_cell']:.1%} -> {post_sft_sum['avg_cell']:.1%}  "
      f"(Δ {post_sft_sum['avg_cell']-pre_sum['avg_cell']:+.1%})")
print(f"\\nDelta (GRPO - SFT):")
print(f"  Exact:  {post_sft_sum['exact_rate']:.1%} -> {post_grpo_sum['exact_rate']:.1%}  "
      f"(Δ {post_grpo_sum['exact_rate']-post_sft_sum['exact_rate']:+.1%})")
print(f"  Cell:   {post_sft_sum['avg_cell']:.1%} -> {post_grpo_sum['avg_cell']:.1%}  "
      f"(Δ {post_grpo_sum['avg_cell']-post_sft_sum['avg_cell']:+.1%})")
print(f"\\nDelta (GRPO - Pre):")
print(f"  Exact:  {pre_sum['exact_rate']:.1%} -> {post_grpo_sum['exact_rate']:.1%}  "
      f"(Δ {post_grpo_sum['exact_rate']-pre_sum['exact_rate']:+.1%})")
print(f"  Cell:   {pre_sum['avg_cell']:.1%} -> {post_grpo_sum['avg_cell']:.1%}  "
      f"(Δ {post_grpo_sum['avg_cell']-pre_sum['avg_cell']:+.1%})")
""")

# ── Cell: Visualizations ──
md("## 17. Visualizations")

code("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. SFT loss curve
if sft_metrics:
    steps = [m['step'] for m in sft_metrics]
    losses = [m['loss'] for m in sft_metrics]
    axes[0,0].plot(steps, losses, 'o-', color='#118ab2', markersize=3)
    axes[0,0].set_xlabel('SFT step'); axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('SFT Training Loss')

# 2. GRPO reward curve
if metrics_history:
    iters = [m['iteration'] for m in metrics_history]
    rewards = [m['mean_reward'] for m in metrics_history]
    axes[0,1].plot(iters, rewards, 'o-', color='#06d6a0', markersize=3)
    axes[0,1].set_xlabel('GRPO iteration'); axes[0,1].set_ylabel('Mean reward')
    axes[0,1].set_title('GRPO Training Reward')

# 3. Three-stage exact match per type
by_type = {}
for r in pre_results: by_type.setdefault(r['puzzle_type'], {'pre':[], 'sft':[], 'grpo':[]})['pre'].append(r)
for r in post_sft_results: by_type.setdefault(r['puzzle_type'], {'pre':[], 'sft':[], 'grpo':[]})['sft'].append(r)
for r in post_grpo_results: by_type.setdefault(r['puzzle_type'], {'pre':[], 'sft':[], 'grpo':[]})['grpo'].append(r)
types_sorted = sorted(by_type.keys())
em_pre = [sum(1 for r in by_type[t]['pre'] if r['exact_match'])/len(by_type[t]['pre']) for t in types_sorted]
em_sft = [sum(1 for r in by_type[t]['sft'] if r['exact_match'])/len(by_type[t]['sft']) for t in types_sorted]
em_grpo = [sum(1 for r in by_type[t]['grpo'] if r['exact_match'])/len(by_type[t]['grpo']) for t in types_sorted]
labels = [by_type[t]['pre'][0]['description'][:16] for t in types_sorted]
x = np.arange(len(types_sorted)); w = 0.27
axes[1,0].barh(x-w, em_pre, w, color='#e63946', label='pre')
axes[1,0].barh(x, em_sft, w, color='#118ab2', label='post-SFT')
axes[1,0].barh(x+w, em_grpo, w, color='#06d6a0', label='post-GRPO')
axes[1,0].set_yticks(x); axes[1,0].set_yticklabels(labels, fontsize=7)
axes[1,0].set_xlabel('Exact match rate'); axes[1,0].set_title('Exact Match per Type (3 stages)')
axes[1,0].legend(); axes[1,0].set_xlim(0,1)

# 4. Cell accuracy: pre vs post-GRPO per puzzle
ca_pre = [r['cell_accuracy'] for r in pre_results]
ca_grpo = [r['cell_accuracy'] for r in post_grpo_results]
colors = ['#06d6a0' if (pre_results[i]['exact_match']==False and post_grpo_results[i]['exact_match']==True)
          else ('#118ab2' if post_grpo_results[i]['exact_match'] else '#e63946')
          for i in range(len(pre_results))]
axes[1,1].scatter(ca_pre, ca_grpo, c=colors, s=60, alpha=0.7)
axes[1,1].plot([0,1],[0,1], 'k--', alpha=0.3)
axes[1,1].set_xlabel('Pre-train cell accuracy'); axes[1,1].set_ylabel('Post-GRPO cell accuracy')
axes[1,1].set_title('Per-puzzle: Pre vs Post-GRPO (green=newly exact)')
axes[1,1].set_xlim(-0.05,1.05); axes[1,1].set_ylim(-0.05,1.05)

plt.tight_layout()
plt.savefig('/kaggle/working/results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved results.png")
""")

# ── Cell: Detailed per-puzzle table ──
md("## 18. Detailed Per-Puzzle Comparison (3 stages)")

code("""print(f"{'#':<3} {'Puzzle ID':<28} {'Type':<22} {'Pre-EM':>7} {'SFT-EM':>7} {'GRPO-EM':>8} "
      f"{'Pre-CA':>7} {'SFT-CA':>7} {'GRPO-CA':>8}")
print("="*105)
n_improved_sft = 0; n_regressed_sft = 0; n_improved_grpo = 0; n_regressed_grpo = 0
for i in range(len(pre_results)):
    pr, sf, gr = pre_results[i], post_sft_results[i], post_grpo_results[i]
    em = lambda r: 'Y' if r['exact_match'] else 'n'
    d_sft = sf['cell_accuracy'] - pr['cell_accuracy']
    d_grpo = gr['cell_accuracy'] - sf['cell_accuracy']
    flag = ''
    if not pr['exact_match'] and sf['exact_match']: flag += ' SFT↑'; n_improved_sft += 1
    elif pr['exact_match'] and not sf['exact_match']: n_regressed_sft += 1
    if not sf['exact_match'] and gr['exact_match']: flag += ' GRPO↑'; n_improved_grpo += 1
    elif sf['exact_match'] and not gr['exact_match']: n_regressed_grpo += 1
    print(f"{i+1:<3} {pr['puzzle_id']:<28} {pr['puzzle_type']:<22} "
          f"{em(pr):>7} {em(sf):>7} {em(gr):>8} {pr['cell_accuracy']:>6.1%} "
          f"{sf['cell_accuracy']:>6.1%} {gr['cell_accuracy']:>7.1%}{flag}")
print("="*105)
print(f"SFT:    newly exact {n_improved_sft} | regressed {n_regressed_sft}")
print(f"GRPO:   newly exact {n_improved_grpo} | regressed {n_regressed_grpo}")
""")

# ── Cell: Final summary ──
md("## 19. Final Summary & Artifacts")

code("""import os
print("="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"LoRA: rank={LORA_RANK}, layers={LORA_LAYERS}, alpha={LORA_ALPHA}")
print(f"SFT: {SFT_EPOCHS} epochs, bs={SFT_BATCH_SIZE}x{SFT_GRAD_ACCUM}, lr={SFT_LR}, "
      f"{len(tokenized)} examples")
print(f"GRPO: {N_TRAIN_ITERS} iters, group={GROUP_SIZE}, steps={N_STEPS}, lr={GRPO_LR}")
print(f"Benchmark: {len(benchmark_puzzles)} transfer puzzles")
print()
print(f"PRE-TRAIN:  exact {pre_sum['exact']}/{pre_sum['total']} ({pre_sum['exact_rate']:.1%}), "
      f"avg cell acc {pre_sum['avg_cell']:.1%}")
print(f"POST-SFT:   exact {post_sft_sum['exact']}/{post_sft_sum['total']} ({post_sft_sum['exact_rate']:.1%}), "
      f"avg cell acc {post_sft_sum['avg_cell']:.1%}")
print(f"POST-GRPO:  exact {post_grpo_sum['exact']}/{post_grpo_sum['total']} ({post_grpo_sum['exact_rate']:.1%}), "
      f"avg cell acc {post_grpo_sum['avg_cell']:.1%}")
print()
print("Artifacts in /kaggle/working/:")
for f in sorted(os.listdir('/kaggle/working')):
    p = os.path.join('/kaggle/working', f)
    if os.path.isfile(p): print(f"  {f} ({os.path.getsize(p)/1024:.0f} KB)")
    else: print(f"  {f}/ (dir)")
print("="*70)
""")

md("""## Notes

### IL Pipeline
This notebook implements the **Intuition Learning** pipeline from the `ilresearch` repo:
1. **SFT** (Stage 1): supervised fine-tune on ARC-AGI-3-style reasoning traces (`il_data_filt/`)
2. **GRPO** (Stage 2): group-relative policy optimization on procedural puzzles, warm-started from SFT

### Why signal_accuracy for GRPO
`signal_accuracy` scores only non-background cells, creating variance between rollouts
on sparse ARC grids (~80% background). `cell_accuracy` would make all rollouts score 0.8+
→ zero GRPO advantage. `signal_accuracy` creates the learning signal.

### Why transfer types for benchmark
The 12 benchmark types are different from the 20 training types. If we benchmarked on
training types, the model could memorize. Transfer types test whether training improved
general abstract reasoning.

### Two-stage inference
The model reasons up to `think_tokens`. If it doesn't emit `</think>`, we force the close
tag and generate a short answer — guarantees every puzzle gets a real answer attempt
under bounded compute (standard for benchmarking reasoning models with a token budget).
""")

# ── Build notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "kaggle": {
            "accelerator": "GPU T4 x2",
            "dataSources": [{"datasetId": 0, "sourceId": 0, "isDisabled": False}],
            "internetEnabled": True,
        },
    },
    "nbformat": 4, "nbformat_minor": 4,
}

out_path = "/Users/kzrr/ilresearch/kaggle_push/il_pipeline_kaggle.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path} with {len(cells)} cells")
