# IL-research — Intuition Learning pipeline

RL-like "intuition learning" pipeline tested on **DeepSeek-R1-Distill-Qwen-1.5B**
via mlx_lm on an 8 GB Apple Silicon Mac, and on **Kaggle T4 GPU** via PyTorch.

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
- `kaggle_push/`   — Kaggle pipeline (PyTorch/CUDA port of the MLX pipeline)
  - `build_notebook.py`     — generates the Kaggle .ipynb programmatically
  - `gen_improved_data.py`  — v2 SFT dataset generator (CoT + think tags)
  - `il_dataset_v2/`        — v2 SFT data (1104 examples with `</midt>` tags)
  - `kernel-metadata.json`  — Kaggle kernel push config
  - `RESEARCH_NOTES.md`     — full v1/v2 comparison + results documentation

## Kaggle pipeline (PyTorch/CUDA)

The Kaggle notebook runs the full pipeline on a Tesla T4 GPU:
1. Load DeepSeek-R1-Distill-Qwen-1.5B (bfloat16)
2. Pre-benchmark on 36 transfer puzzles
3. SFT on CoT reasoning dataset
4. Post-SFT benchmark
5. GRPO RL training (warm-started from SFT)
6. Post-GRPO benchmark
7. Compare all stages + visualize

### Kaggle resources
- v1 dataset: `krokodileceo/il-research-sft-benchmark` (316 examples, no think tags)
- v1 notebook: `krokodileceo/il-pipeline-sft-grpo-on-r1-distill-qwen-1-5b`
- v2 dataset: `krokodileceo/il-research-sft-v2-benchmark` (1104 examples, CoT + think tags)
- v2 notebook: `krokodileceo/il-pipeline-v2-sft-grpo-on-r1-distill-qwen-1-5b`
- v3 dataset: `krokodileceo/il-research-sft-v3-benchmark` (1978 examples, computation traces + self-correction + transfer skills)
- v3 notebook: `krokodileceo/il-pipeline-v3-sft-grpo-on-r1-distill-qwen-1-5b`
- v5 notebook: `krokodileceo/il-pipeline-v5-sft-grpo-on-r1-distill-qwen-1-5b` (v4 fixes + v5 optimizations)

### v4 fixes (25 fixes applied)
1. Think tag alignment (`</midt>` -> `</think>` native R1-distill tag)
2. SFT tokenization fix (manual text construction to preserve thinking tags)
3. KL penalty sign fix (was rewarding policy drift, now penalizes it)
4. Reward function fix (removed shape_distance bonus for wrong grids)
5. Dimension hint removal from GRPO feedback (train/test mismatch)
6. Dual T4 support (second GPU for data-parallel inference)
7. Batched GRPO rollouts (6-8 rollouts in 1 generate call)
8. Batched benchmark inference (6 puzzles at a time)
9. torch.compile + SDPA attention
10. Gradient checkpointing for SFT
11. TF32 + inference_mode
12. Validation loss tracking during SFT
13. Length-bucketed SFT sampling
14. Robust parse_grid
15. Best checkpoint based on benchmark accuracy (not any_exact)
16. Compact grid format
17. Stop-on-think-close token (saves tokens)
18. Pre-compiled chat template caching
19. SFT epochs 5->3 (loss plateaus at 2.5)
20. GRPO group_size 6->8, temperature 0.8->0.6
21. Multi-puzzle per GRPO iter (2 puzzles for stable gradients)
22. Self-correction SFT examples (teaches hypothesis testing)
23. Computation traces in SFT teacher (teaches COMPUTE, not just describe)
24. Removed "Example 3 confirms" boilerplate (replaced with real verification)
25. Dataset increased to 100 examples per type (1978 total)

### v5 optimizations (10 optimizations from deep audit)
A. PPO epochs (reuse rollouts for 2 gradient updates)
B. Advantage clipping to [-5, 5] (prevents gradient explosion)
C. Replay buffer for high-quality GRPO rollouts
D. TIC-GRPO trajectory-level importance ratio (more stable than per-token)
E. Token-weighted SFT loss (focus on critical tokens)
F. Data augmentation via grid symmetries (rotate, flip, color permute)
G. Multi-view grid tokenization (ASCII + structured)
H. KL-free GRPO option (saves 30-40% memory)
I. CPU optimizer offloading option
J. PrefixGrouper concept (shared prefix computation for batched rollouts)

### Push commands
    export KAGGLE_API_TOKEN=<token>
    # Push dataset
    kaggle datasets create -p kaggle_push/il_dataset_v2_kaggle
    # Push notebook (12h timeout, T4 GPU)
    kaggle kernels push -p kaggle_push --accelerator NvidiaTeslaT4 --timeout 43200
    # Check status
    kaggle kernels status krokodileceo/il-pipeline-v2-sft-grpo-on-r1-distill-qwen-1-5b
    # Download output
    kaggle kernels output krokodileceo/il-pipeline-v2-sft-grpo-on-r1-distill-qwen-1-5b -p /tmp/kaggle_out

### Kaggle bugs fixed
1. P100 GPU incompatible with default PyTorch build → set `machine_shape: NvidiaTeslaT4`
2. Dataset path nested under `/kaggle/input/datasets/` not `/kaggle/input/` → recursive glob search
3. `torch.cuda.reset_peak_memory()` doesn't exist → use `torch.cuda.reset_peak_memory_stats()`

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
