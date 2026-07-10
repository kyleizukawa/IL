#!/usr/bin/env python3
"""Convert the local DeepSeek-R1-Distill-Qwen-1.5B to 4-bit MLX format,
then load + generate a few tokens. Measures memory and throughput."""
import time, os, resource
from mlx_lm import load, generate, convert

MODEL_PATH = "/Users/kzrr/ ILresearch /model"
MLX_PATH = "/Users/kzrr/ ILresearch /model_mlx_4bit"

if not os.path.exists(MLX_PATH):
    print(f"Converting {MODEL_PATH} -> {MLX_PATH} (4-bit)...", flush=True)
    t0 = time.time()
    convert(hf_path=MODEL_PATH, mlx_path=MLX_PATH, quantize=True, q_bits=4)
    print(f"Converted in {time.time()-t0:.1f}s", flush=True)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    print(f"ru_maxrss after convert: {ru.ru_maxrss/1e6:.2f} GB", flush=True)
else:
    print(f"{MLX_PATH} already exists, skipping conversion.", flush=True)

print(f"\nLoading {MLX_PATH} ...", flush=True)
t0 = time.time()
model, tokenizer = load(MLX_PATH)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)
ru = resource.getrusage(resource.RUSAGE_SELF)
print(f"ru_maxrss after load: {ru.ru_maxrss/1e6:.2f} GB", flush=True)

# Chat-formatted prompt (DeepSeek-R1 style)
messages = [{"role": "user", "content": "What is 7 + 5? Think step by step, then give the answer."}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(f"\nPrompt (first 300 chars):\n{prompt[:300]}\n...", flush=True)

print("\nGenerating (max_tokens=256)...", flush=True)
t0 = time.time()
out = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=True)
dt = time.time() - t0
print(f"\nGenerated in {dt:.1f}s -> {256/dt:.1f} tok/s", flush=True)
print(f"Output (first 600 chars):\n{out[:600]}", flush=True)
ru = resource.getrusage(resource.RUSAGE_SELF)
print(f"\nru_maxrss after gen: {ru.ru_maxrss/1e6:.2f} GB", flush=True)
