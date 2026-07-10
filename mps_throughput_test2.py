#!/usr/bin/env python3
"""Quick MPS throughput re-test on AC power (256 tokens)."""
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/Users/kzrr/ ILresearch /model"
print(f"Loading {MODEL_PATH} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16).to("mps")
model.eval()
print(f"Loaded.", flush=True)

prompt = "You are an abstract reasoning system. " * 60
ids = tok(prompt, return_tensors="pt").to("mps")
print(f"Prompt tokens: {ids['input_ids'].shape[1]}", flush=True)

# Warmup
print("Warmup (32 tokens)...", flush=True)
t0 = time.time()
with torch.no_grad():
    _ = model.generate(**ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
print(f"  warmup: {time.time()-t0:.1f}s", flush=True)

# Measure 256 tokens
print("Generating 256 tokens...", flush=True)
t0 = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
dt = time.time() - t0
n_new = out.shape[1] - ids['input_ids'].shape[1]
tps = n_new/dt
print(f"  generated {n_new} tokens in {dt:.1f}s -> {tps:.2f} tok/s", flush=True)
print(f"  projected 36 puzzles x 2048 tokens = {36*2048/tps/60:.1f} min", flush=True)
print(f"  projected 36 puzzles x 512 tokens  = {36*512/tps/60:.1f} min", flush=True)
