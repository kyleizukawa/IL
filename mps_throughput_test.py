#!/usr/bin/env python3
"""Measure MPS generation throughput for the 1.5B model so we can size the benchmark."""
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/Users/kzrr/ ILresearch /model"
print(f"Loading {MODEL_PATH} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16).to("mps")
model.eval()
print(f"Loaded. dtype={next(model.parameters()).dtype}", flush=True)

# Short prompt (mimics a benchmark puzzle prompt ~1.2k chars)
prompt = "You are an abstract reasoning system. " * 60
ids = tok(prompt, return_tensors="pt").to("mps")
print(f"Prompt tokens: {ids['input_ids'].shape[1]}", flush=True)

# Warmup (first MPS kernel compile is slow)
print("Warmup (64 tokens)...", flush=True)
t0 = time.time()
with torch.no_grad():
    _ = model.generate(**ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
print(f"  warmup: {time.time()-t0:.1f}s", flush=True)

# Real measurement
for budget in [512, 2048]:
    print(f"\nGenerating {budget} tokens...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=budget, do_sample=False, pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    n_new = out.shape[1] - ids['input_ids'].shape[1]
    print(f"  generated {n_new} tokens in {dt:.1f}s → {n_new/dt:.1f} tok/s", flush=True)
    print(f"  projected: 36 puzzles x 8192 tokens = {36*8192/(n_new/dt)/60:.0f} min", flush=True)
    print(f"  projected: 36 puzzles x {budget} tokens = {36*budget/(n_new/dt)/60:.0f} min", flush=True)
    print(f"  projected: 12 puzzles x {budget} tokens = {12*budget/(n_new/dt)/60:.0f} min", flush=True)
