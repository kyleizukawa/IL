# IL-research — Intuition Learning pipeline

RL-like "intuition learning" pipeline tested on **DeepSeek-R1-Distill-Qwen-1.5B**
via mlx_lm on an 8 GB Apple Silicon Mac.

## Layout
- `il/`            — puzzle environments (20 ARC-AGI3-style generators) + SFT data gen
- `il_data/`       — raw SFT jsonl (train/valid/test)
- `il_data_filt/`  — SFT data filtered to <=768 tokens (used for the SFT run)
- `il_adapters/`   — SFT-trained LoRA adapters (rank 8, 16 layers, ~21 MB)
- `il_rl/`         — GRPO RL trainer (`env.py`, `rollout.py`, `grpo.py`)
- `il_rl_adapters/`— RL-trained LoRA checkpoints (warm-started from SFT)
- `model/`         — original fp16 DeepSeek-R1-Distill-Qwen-1.5B
- `model_mlx_4bit/`— 4-bit MLX model (used for RL; ~1.9 GB)
- `model_mlx_8bit/`— 8-bit MLX model (used for SFT; ~3.5 GB)
- `run_full.py`    — baseline benchmark (no training) over 36 puzzles
- `run_rl.py`      — GRPO RL training runner (stage 3, after SFT)
- `mini_arc_agi3_benchmark.ipynb` — original notebook (puzzle generators + eval)

## Commands

### Baseline benchmark (inference only, 4-bit model)
    python run_full.py --max-tokens 2048
    python run_full.py --smoke            # single-puzzle sanity check
    python run_full.py --resume           # skip puzzles already in results

### SFT (supervised fine-tuning) — already done, adapters in il_adapters/
    python -m mlx_lm.lora --config il_adapters/adapter_config.json
    # (config: model_mlx_8bit, il_data_filt, max_seq_length 768, rank 8,
    #  num_layers 16, lr 1e-4, iters 800, save_every 50)

### RL (GRPO) — the stage we were running when we OOM'd (now fixed)
    python run_rl.py --smoke                       # 1-iter sanity check
    python run_rl.py --iters 50 --group-size 4     # short run
    python run_rl.py --iters 200 --group-size 4 --save-every 25
    python run_rl.py --no-sft                      # RL from base (skip SFT warm-start)

Key flags: `--thinking-tokens`, `--prediction-tokens`, `--n-steps`,
`--lr`, `--temperature`, `--memory-limit-gb` (default 5.5).

## Memory notes (8 GB Mac)
- The original OOM was in `il_rl/grpo.py::compute_action_logprobs`: it called
  `model(input_tokens)` which materializes full-vocab logits
  `[1, ~1500, 151936]` (~900 MB fp32) over the whole episode inside the autograd
  graph. Fix: forward the backbone (`model.model`) only, then apply `model.lm_head`
  ONLY at action-token positions (~5x logits reduction).
- RL uses the 4-bit base model (~1.9 GB) instead of 8-bit (~3.5 GB) to free headroom.
- `mx.set_memory_limit` + `mx.clear_cache()` between rollouts keep peak ~5.3-5.7 GB.
- Verified: 5-iter run, peak 5.71 GB, no OOM, no leak across iterations.

## Bugs fixed during this session
1. OOM in `compute_action_logprobs` (full-vocab logits over full episode) — see above.
2. Double-LoRA: `GRPOTrainer.__init__` re-applied `linear_to_lora_layers` on top of
   an already-LoRA-adapted SFT model, discarding SFT weights. Trainer now accepts a
   pre-adapted model; the runner freezes the base THEN calls `load_adapters`.
3. Silent no-op update: `optimizer.update(...)` was never `mx.eval`'d, so (MLX being
   lazy) the params never actually changed. Now `mx.eval(model.parameters(), optimizer.state)`.
4. Wrong think-close tag in `rollout.py`: `</>` -> `</midt>` (token 151649).
5. `freeze()` ordering: must freeze base BEFORE applying LoRA, else LoRA params are
   frozen too (0 trainable params).
