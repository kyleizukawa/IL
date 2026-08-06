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

## il_agentic/ — Agentic Coding RL Environments (mechanize.work-style)

15 RL environments for agentic coding, codebase reasoning, and killing laziness
in small models. Built in the style of [mechanize.work](https://mechanize.work):
each environment is a self-contained software engineering task with a
deterministic grader that provides informative reward signals for RL.

### Layout
- `il_agentic/base.py`           — AgenticEnv base class + registry
- `il_agentic/graders.py`        — CodeExecutor sandbox, test runner, response parsing
- `il_agentic/data_gen.py`       — SFT data generator (all 15 envs)
- `il_agentic/rollout.py`        — GRPO rollout interface for agentic tasks
- `il_agentic/benchmark.py`      — held-out benchmark (105 instances, 7 per env)
- `il_agentic/test_environments.py` — test suite (62 tests, all passing)
- `il_agentic/build_kaggle_notebook.py` — generates Kaggle .ipynb for T4 training
- `il_agentic/environments/`     — 15 environment modules (~13,800 lines total)
- `il_agentic_data/`             — SFT dataset v1 (900 examples, ~1.4M tokens)
- `il_agentic_data_v2/`          — SFT dataset v2 (1500 examples, ~2.3M tokens)

### 15 Environments

| # | Name | Skill | Domains |
|---|------|-------|---------|
| 1 | bug_localization | Finding/fixing bugs by tracing code | string_utils, math_utils, list_utils, data_processor |
| 2 | feature_impl | Implementing features from specs | todo_manager, shopping_cart, temp_converter, text_formatter |
| 3 | refactor_preserve | Refactoring while keeping tests green | data_validator, calculator, report_generator |
| 4 | test_writing | Writing tests that catch mutants | password_validator, date_parser, url_parser, poker_evaluator |
| 5 | api_client | Implementing API clients from specs | user_api, weather_api, payment_api, file_storage_api |
| 6 | perf_optimize | Optimizing slow code | duplicate_finder, string_builder, sum_calculator, frequency_counter |
| 7 | codebase_nav | Navigating multi-file codebases | web_app, cli_tool, data_pipeline |
| 8 | type_annotate | Adding type annotations | data_processing, graph_algorithm, text_processing, config_parser |
| 9 | doc_gen | Writing docstrings from code | math_library, string_utils, data_structures, file_io |
| 10 | config_fix | Fixing broken configs | database, web_server, logging, feature_flags |
| 11 | error_handling | Adding error handling | calculator, list_processor, string_parser, data_aggregator |
| 12 | data_transform | Data transformation pipelines | csv_filter_map, json_extract, log_parse, sales_agg |
| 13 | algorithm_impl | Implementing algorithms from specs | lru_cache, union_find, sliding_window, interval_scheduler |
| 14 | code_review | Identifying issues in code diffs | bug_introduction, security_issue, perf_problem, style_issues |
| 15 | stacktrace_debug | Debugging from stack traces | keyerror, indexerror, typeerror_none, recursionerror |

### Environment interface
Each environment implements:
- `gen_params(rng, difficulty)` — generate task parameters
- `gen_codebase(params, rng)` — generate {filename: content} mini-codebase
- `gen_task(params, codebase)` — generate task description for the model
- `gen_solution(params, codebase)` — generate correct solution
- `gen_reasoning(params, codebase, solution)` — teacher reasoning trace (kills laziness)
- `grade(params, codebase, response)` — returns (score, breakdown) with partial credit

Model response format: `<reasoning>...analysis...</reasoning><answer>...code...</answer>`

### Design principles (mechanize.work style)
1. **Quality over quantity** — each env has rich, informative reward signals (not just pass/fail)
2. **Real codebases** — multi-file Python projects with realistic structure
3. **Distractor code** — irrelevant functions the model must skip (punishes laziness)
4. **Partial credit** — rewards careful partial work, not just all-or-nothing
5. **Edge cases** — tests that punish superficial pattern-matching
6. **Teacher traces** — demonstrate thorough, line-by-line analysis (not jumping to conclusions)
7. **Difficulty scaling** — easy/medium/hard with more files, distractors, and subtler bugs

### Grader infrastructure
- `CodeExecutor`: sandboxed subprocess execution with timeout (no network, restricted env)
- `run_tests`: runs test functions against a codebase, returns per-test pass/fail
- `parse_code_blocks`: extracts ```python:filename``` blocks from model response
- `apply_code_changes`: merges model's code changes into the codebase
- Partial credit: code similarity bonus if right file changed but tests fail

### Commands
    # Run test suite (62 tests)
    python3 -m il_agentic.test_environments
    python3 -m il_agentic.test_environments --env bug_localization  # test one

    # Generate SFT dataset
    python3 -m il_agentic.data_gen --n-per-env 60 --output-dir il_agentic_data
    python3 -m il_agentic.data_gen --n-per-env 100 --output-dir il_agentic_data_v2

    # Run benchmark (requires MLX model)
    python3 -m il_agentic.benchmark --model model_mlx_4bit --max-tokens 2048
    python3 -m il_agentic.benchmark --smoke  # sanity check

    # Build Kaggle notebook
    python3 -m il_agentic.build_kaggle_notebook --output il_agentic_kaggle.ipynb

### SFT dataset stats
- v1: 900 examples (60 per env), ~1.4M tokens, avg 1,731 tokens/example
- v2: 1500 examples (100 per env), ~2.3M tokens, avg 1,725 tokens/example
- Difficulty mix: 25% easy, 50% medium, 25% hard
- All examples have <reasoning> and <answer> tags
- Teacher reasoning demonstrates line-by-line analysis (not jumping to conclusions)

### Benchmark
- 105 held-out instances (7 per env, different seeds from training)
- 30 easy + 45 medium + 30 hard
- Reports per-environment, per-difficulty, and aggregate scores
- Measures: score, reasoning rate, answer rate, reasoning length, gen time

## il_agentic/long_horizon/ — 20 Hand-Crafted Long-Horizon Tasks (mechanize.work-style)

20 hand-crafted, mechanize.work-style RL environments for building strong
long-horizon agentic coding reasoning in small models. Each task is a single,
unique, hand-crafted scenario (NOT procedurally generated) with a real
multi-file codebase (80-400 lines) and efficiency-aware reward shaping.

### Key innovation: Efficiency-aware reward shaping

    final_score = correctness * (0.6 + 0.4 * reasoning_quality)

    reasoning_quality = coverage * 0.40    # did model reason about right concepts?
                         + efficiency * 0.30  # within token budget?
                         + verification * 0.20  # did model check its work?
                         + (1 - filler) * 0.10  # no generic boilerplate

This means:
- Wrong answers always get 0 (no credit for efficient wrong answers)
- Right answers with lazy reasoning get 0.6 × correctness
- Right answers with thorough, verified reasoning get up to 1.0 × correctness
- The 0.4 spread is the RL signal that shapes HOW the model reasons

### Layout
- `long_horizon/base.py`        — LongHorizonEnv base class + registry
- `long_horizon/efficiency.py`  — reasoning quality scorer (4 dimensions)
- `long_horizon/data_gen.py`    — SFT data generator for all 20 tasks
- `long_horizon/rollout.py`     — GRPO rollout with efficiency-aware rewards
- `long_horizon/benchmark.py`   — benchmark with reasoning quality breakdown
- `long_horizon/test_tasks.py`  — test suite (22 tests, all passing)
- `long_horizon/tasks/`         — 20 hand-crafted task modules (~9,500 lines)
- `il_long_horizon_data/`       — SFT dataset v1 (100 examples, ~137K tokens)
- `il_long_horizon_data_v2/`    — SFT dataset v2 (200 examples, ~302K tokens)

### 20 Hand-Crafted Tasks

| # | Task ID | Reasoning Skill | Failure Mode | Budget |
|---|---------|----------------|--------------|--------|
| 1 | cascading_bug_chain | Multi-step cascading reasoning | Fixes first bug, misses cascade | 800 |
| 2 | cross_module_data_flow | Sustained reasoning over 5+ modules | Loses track across modules | 900 |
| 3 | invariant_preservation | Mathematical reasoning about invariants | Refactors without understanding invariant | 700 |
| 4 | complexity_optimization | Algorithmic complexity reasoning | Optimizes without understanding why | 700 |
| 5 | api_contract_compliance | Constraint satisfaction (10+ constraints) | Satisfies some, misses others | 800 |
| 6 | race_condition_detection | Concurrency reasoning about interleavings | Can't reason about non-deterministic order | 700 |
| 7 | recursive_repair | Recursion reasoning — tracing calls | Can't trace recursive paths | 800 |
| 8 | type_flow_inference | Type inference across function boundaries | Annotates in isolation, not flow | 700 |
| 9 | property_based_tests | Abstract reasoning about properties | Writes example-based, not property-based | 700 |
| 10 | design_pattern_selection | Pattern recognition + application | Doesn't recognize when pattern applies | 800 |
| 11 | error_propagation_analysis | Error handling — tracing error paths | Catches errors at wrong level | 700 |
| 12 | state_machine_impl | State reasoning — transitions and guards | Misses guards, allows illegal states | 800 |
| 13 | reachability_analysis | Code path reasoning — unreachable code | Can't reason about reachable branches | 600 |
| 14 | bottleneck_isolation | Performance reasoning — finding bottleneck | Optimizes everything, not the bottleneck | 600 |
| 15 | backward_compat_evolution | API design — evolving without breaking | Changes API without considering callers | 700 |
| 16 | differential_analysis | Comparative reasoning — critical difference | Can't identify which difference matters | 600 |
| 17 | spec_compliance_audit | Spec reading + finding all deviations | Finds 1-2 deviations, misses rest | 800 |
| 18 | minimal_change_identification | Precision reasoning — minimal fix | Over-engineers fixes | 500 |
| 19 | coverage_gap_analysis | Coverage reasoning — untested paths | Tests already-tested paths | 700 |
| 20 | security_audit | Security reasoning — tracing input flows | Misses subtle vulnerabilities | 800 |

### Design principles (mechanize.work style)
1. **Hand-crafted** — each task is unique, not procedurally generated
2. **Long-horizon** — reasoning must be sustained over 500-2000 tokens
3. **Efficiency-aware** — reward measures reasoning quality, not just correctness
4. **Failure-mode targeted** — each task exposes a specific reasoning weakness
5. **Rich reward signal** — 4-dimensional reasoning quality + correctness
6. **Teacher traces** — 400-800 words of step-by-step analysis with verification

### Reasoning quality dimensions
1. **Coverage (40%)** — did the model reason about the right concepts?
2. **Token efficiency (30%)** — did the model reason within budget? (only for correct answers)
3. **Verification (20%)** — did the model check/verify its work?
4. **No-filler (10%)** — did the model avoid generic boilerplate?

### Commands
    # Run test suite (22 tests)
    python3 -m il_agentic.long_horizon.test_tasks
    python3 -m il_agentic.long_horizon.test_tasks --task cascading_bug_chain

    # Generate SFT dataset
    python3 -m il_agentic.long_horizon.data_gen --examples-per-task 5
    python3 -m il_agentic.long_horizon.data_gen --examples-per-task 10 --output-dir il_long_horizon_data_v2

    # Run benchmark (requires MLX model)
    python3 -m il_agentic.long_horizon.benchmark --model model_mlx_4bit
    python3 -m il_agentic.long_horizon.benchmark --smoke

### SFT dataset stats
- v1: 100 examples (5 per task), ~137K tokens, avg 1,706 tokens/example
- v2: 200 examples (10 per task), ~302K tokens, avg 1,679 tokens/example
- Reasoning length: avg 3,931 chars (min 2,306, max 6,124)
- All examples have <reasoning> and <answer> tags
- Teacher reasoning demonstrates thorough, verified analysis

### Curriculum learning (RL)
- Phase 1 (0-30% of training): 5 easy tasks (shorter reasoning)
- Phase 2 (30-70%): 10 medium tasks
- Phase 3 (70-100%): 5 hard tasks (longest reasoning, most complex)

