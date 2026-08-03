# IL Pipeline Research Notes — v1 & v2

## Overview

The IL (Intuition Learning) pipeline trains a small reasoning model to solve
ARC-AGI-3-style grid transformation puzzles. The pipeline has three stages:
**pre-benchmark** → **SFT** → **GRPO RL** → **post-benchmark** after each stage.

Model: **DeepSeek-R1-Distill-Qwen-1.5B** (1.78B params, bfloat16)
Hardware: **Kaggle Tesla T4 x2** (16 GB VRAM each, model on cuda:0)

---

## v1 Run (completed 2026-08-01)

### Config
- SFT: 316 examples, 3 epochs, bs=2x8, lr=1e-4, max_seq=768, LoRA rank=8
- GRPO: 40 iters, group=4, think=128, pred=96, lr=5e-6
- Benchmark: think=256, answer=128

### SFT Dataset (v1)
- 316 training examples from 20 environment types (~16 per type)
- No think tags (`</midt>`) — assistant response was just "Looking at the examples, the rule is: X. Applying this to the test input:\n\n[[grid]]"
- Very short reasoning: avg 196 chars, max 375 chars
- Mix: 60% prediction, 25% rule inference, 15% rapid intuition

### Results

| Stage | Exact Match | Shape Match | Avg Cell Accuracy |
|-------|------------|------------|-------------------|
| PRE-TRAIN (base) | 0.0% | 5.6% (2/36) | 6.0% |
| POST-SFT | 0.0% | 8.3% (3/36) | 7.8% |
| POST-GRPO | 0.0% | 5.6% (2/36) | 10.2% |

### Deltas
- SFT vs Pre: Cell accuracy +1.8%, shape match +2.8%
- GRPO vs SFT: Cell accuracy +2.4%, shape match -2.8%
- Total (GRPO vs Pre): Cell accuracy +4.2%

### Timing
- Pre-benchmark: 10.5 min (36 puzzles × ~17s each)
- SFT: 28.2 min (3 epochs, 59 steps)
- Post-SFT benchmark: 12.6 min
- GRPO: 73.9 min (40 iters, ~110s per iter)
- Post-GRPO benchmark: 12.6 min
- **Total: ~2h 18min**

### GRPO Details
- 2/40 iterations had exact rollouts (interior_fill, erosion)
- Mean reward: 0.363 → 0.079 (decreased — model explored harder puzzles)
- Peak memory: 10.93 GB
- SFT loss: 1.37 → 0.87 (only 11 logging steps)

### v1 Issues Identified
1. **No think tags in SFT data** — model never learned to close reasoning with `</midt>`, causing all benchmark answers to be "forced" (model didn't converge within budget)
2. **Too few examples** (316) — insufficient for learning 20 different transformation types
3. **Very short reasoning** (~196 chars) — didn't teach step-by-step pattern analysis
4. **SFT undertrained** — only 59 steps, loss only dropped from 1.37 to 0.87
5. **GRPO too short** — 40 iters with group=4 is too small for meaningful policy improvement
6. **Small token budgets** — think=128/pred=96 in GRPO, think=256/answer=128 in benchmark too restrictive for hard puzzles
7. **No curriculum** — GRPO sampled puzzle types randomly, wasting early iterations on hard types

---

## v2 Run (started 2026-08-01)

### Improvements

#### 1. Better SFT Dataset (v2)
- **1104 training examples** (3.5x more than v1's 316)
- 60 examples per type × 20 types
- **All examples have `</midt>` think tags** — reasoning then close tag then answer
- **Step-by-step CoT reasoning traces**: analyze examples → identify rule → apply to test
- 4 example modes:
  - 45% detailed_cot: full step-by-step analysis (~500 chars)
  - 25% brief_cot: concise rule + application (~200 chars)
  - 20% rule_inference: describe the rule only (~150 chars)
  - 15% rapid: 1-example fast intuition (~180 chars)
- Avg assistant content: 366 chars (v1 was 196)
- Est. ~398K tokens total, ~360 tokens/example

#### 2. Better SFT Training
- **5 epochs** (was 3) — more passes over the data
- **LR 2e-4** (was 1e-4) — higher LR for LoRA adapters converges faster
- **LoRA rank 16** (was 8) — double the capacity for learning reasoning patterns
- **LoRA alpha 32** (was 16) — alpha = 2× rank for proper scaling
- **Warmup 30 steps** (was 20)
- Expected: ~345 steps (1104 examples / 16 effective batch × 5 epochs)

#### 3. Better GRPO
- **100 iterations** (was 40) — 2.5x more training
- **Group size 6** (was 4) — better advantage estimates
- **Think=192, pred=128** (was 128/96) — more reasoning budget
- **LR 1e-5** (was 5e-6) — faster policy updates
- **Curriculum learning**: 
  - First 30% (iters 0-29): easy types (color_swap, col_reverse, max_row, threshold, rotate, transpose, mirror_half, flip_color)
  - Middle 40% (iters 30-69): + medium types (border, center_extract, scale, row_shift, crop, color_pos)
  - Last 30% (iters 70-99): all 20 types
- **Shape match bonus** in reward: +0.15 quality when predicted grid has correct dimensions
- **Periodic checkpoints** every 25 iterations

#### 4. Better Benchmark
- **Think=384** (was 256) — 50% more reasoning budget
- **Answer=192** (was 128) — 50% more generation budget

### Expected Timing (estimate)
- Pre-benchmark: ~15 min (36 puzzles × ~25s with bigger budgets)
- SFT: ~45 min (5 epochs, ~345 steps, 1104 examples)
- Post-SFT benchmark: ~15 min
- GRPO: ~200 min (100 iters × ~120s per iter with group=6)
- Post-GRPO benchmark: ~15 min
- **Total: ~5 hours** (well within 12h Kaggle limit)

### Kaggle Resources
- Notebook: `krokodileceo/il-pipeline-v2-sft-grpo-on-r1-distill-qwen-1-5b`
- Dataset: `krokodileceo/il-research-sft-v2-benchmark`
- Timeout: 43200s (12 hours max)
- Accelerator: NvidiaTeslaT4

### v2 Results (completed 2026-08-02)

#### Final Comparison

| Stage | Exact Match | Shape Match | Avg Cell Accuracy |
|-------|------------|------------|-------------------|
| PRE-TRAIN (base) | 0.0% (0/36) | 5.6% (2/36) | 4.9% |
| POST-SFT | 0.0% (0/36) | 36.1% (13/36) | 30.9% |
| POST-GRPO | 0.0% (0/36) | **55.6% (20/36)** | **39.5%** |

#### Deltas
- **SFT vs Pre**: Cell accuracy **+26.0%** (4.9% → 30.9%), shape match **+30.6%** (5.6% → 36.1%)
- **GRPO vs SFT**: Cell accuracy **+8.5%** (30.9% → 39.5%), shape match **+19.4%** (36.1% → 55.6%)
- **Total (GRPO vs Pre)**: Cell accuracy **+34.5%** (4.9% → 39.5%), shape match **+50.0%** (5.6% → 55.6%)

#### v1 vs v2 Comparison

| Metric | v1 Total Δ | v2 Total Δ | Improvement |
|--------|-----------|-----------|-------------|
| Cell accuracy | +4.2% | **+34.5%** | **8.2x better** |
| Shape match | +0.0% | **+50.0%** | **+50.0%** |
| Exact match | +0.0% | +0.0% | same (0/36) |

#### Per-Puzzle-Type Breakdown (v2)

| Puzzle Type | Pre (Shape/Cell) | Post-SFT (Shape/Cell) | Post-GRPO (Shape/Cell) |
|-------------|-----------------|----------------------|----------------------|
| Symmetry Completion | 2/3, 44.0% | 3/3, 62.0% | 3/3, 67.7% |
| Maze Path (BFS) | 0/3, 0.0% | 3/3, 65.0% | 3/3, 57.7% |
| Row Extremes Marking | 0/3, 0.0% | 2/3, 50.4% | 3/3, 47.4% |
| Region Flood Coloring | 0/3, 7.9% | 0/3, 16.0% | 3/3, 46.1% |
| Color Chain Transform | 0/3, 0.0% | 1/3, 40.3% | 2/3, 66.1% |
| Tile Pattern Extrapolation | 0/3, 0.0% | 2/3, 53.7% | 2/3, 44.4% |
| Gravity + Column Sort | 0/3, 0.0% | 2/3, 51.3% | 2/3, 46.3% |
| Conditional Transform | 0/3, 0.0% | 0/3, 32.5% | 1/3, 58.7% |
| Diagonal Fill | 0/3, 0.0% | 0/3, 0.0% | 1/3, 23.8% |
| Largest Object Replication | 0/3, 3.3% | 0/3, 0.0% | 0/3, 15.2% |
| Block Expansion | 0/3, 0.0% | 0/3, 0.0% | 0/3, 0.0% |
| Shape Sort & Arrange | 0/3, 3.8% | 0/3, 0.0% | 0/3, 0.0% |

#### SFT Training Details
- 58 logging steps (every 5 steps), 293 total steps
- Loss: 1.7138 → 0.0579 (96.6% reduction — v1 only dropped 1.37 → 0.87, 36% reduction)
- Peak memory: 10.95 GB
- Training time: 141.6 min (2.4 hours)

#### GRPO Training Details
- 100 iterations, 29/100 had exact rollouts (v1: 2/40)
- Mean reward: 2.321 → 1.280 (decreased as curriculum introduced harder types)
- Peak memory: 12.25 GB (v1: 10.93 GB — larger due to group=6)
- Training time: 289.8 min (4.8 hours)
- Checkpoints saved at iters 25, 50, 75, 100
- Exact rollouts by curriculum phase:
  - Easy phase (iters 0-29): frequent exacts (color_swap, max_row, flip_color, threshold)
  - Medium phase (iters 30-69): moderate exacts (border, center_extract, rotate)
  - Hard phase (iters 70-99): fewer exacts (erosion, dilation — hardest types)

#### Timing (v2)
- Pre-benchmark: 15.0 min (36 puzzles × ~25s with think=384)
- SFT: 141.6 min (5 epochs, 293 steps, 1104 examples)
- Post-SFT benchmark: 14.2 min
- GRPO: 289.8 min (100 iters × ~174s per iter with group=6)
- Post-GRPO benchmark: 13.7 min
- **Total: ~7h 55min** (well within 12h Kaggle limit)

#### Key Observations
1. **SFT was the biggest win**: Cell accuracy jumped from 4.9% to 30.9% (+26.0%) just from SFT.
   The CoT reasoning traces with `</midt>` tags taught the model to reason about patterns
   and produce properly-shaped grids. v1's SFT only gained +1.8% because it lacked think tags.
2. **GRPO further improved shape matching**: From 36.1% to 55.6% (+19.4%). The shape match
   bonus in the reward function helped the model learn correct output dimensions.
3. **No exact matches on transfer puzzles**: The 12 benchmark puzzle types are intentionally
   different from the 20 training types. Getting 0/36 exact is expected — the model is
   transferring intuition, not memorizing. But 20/36 shape matches means the model correctly
   identifies the output structure for over half the puzzles.
4. **SFT loss converged dramatically**: 1.71 → 0.058 (96.6% reduction) vs v1's 1.37 → 0.87
   (36% reduction). The higher LR (2e-4), more data (1104 vs 316), and more epochs (5 vs 3)
   all contributed.
5. **GRPO curriculum worked well**: 29/100 exact rollouts (vs v1's 2/40). The easy-phase
   puzzles (color_swap, max_row, flip_color) were solved frequently, building confidence
   before harder types.
6. **Hardest puzzle types remain unsolved**: Block Expansion (0% across all stages) and
   Shape Sort & Arrange (0% across all stages) are the most structurally complex — they
   require understanding object composition and sorting, which the 1.5B model struggles with.

---

## v2 Reasoning Output Analysis

### The critical finding: the model is NOT actually analyzing environments

Examining the `thinking_excerpt` field from benchmark results across all stages:

**v1 (all stages):**
```
"Okay, so I've got this problem where I need to figure out the transformation 
rule from some given examples and then apply it to a test input. Let me try to 
break this down step by step. First, I'll look at the examples..."
```
→ **Generic boilerplate**. The model says "let me look at the examples" but never
actually describes what it sees. All 36/36 were forced. Avg think length: ~700 chars.

**v2 post-SFT:**
```
"Looking at Example 1: the input is a 7x4 grid and the output is a 7x4 grid. 
The transformation appears to be: The top half is mirrored to the bottom half."
```
→ **Template mimicry**. The model learned the SFT teacher's format ("Looking at
Example 1: the input is a XxY grid...") but not the act of analysis. It states
dimensions and jumps to a rule — never traces what happens to specific cells.

**v2 post-GRPO:**
```
"Looking at Example 1: the input is a 7x9 grid and the output is a 7x9 grid. 
The transformation seems to be: Every cell with color 1 is changed to color 4."
```
→ **Even shorter**. GRPO reduced thinking length (913→651 chars avg) because it
rewards outcome, not process. The model jumps to guesses faster — often wrong
guesses. On maze_path_1, it guessed "color swap" (a training type) instead of
recognizing pathfinding.

### Three core problems identified
1. **Template mimicry, not reasoning**: Model learned to fill in "Looking at Example 1:
   input is XxY, output is XxY, rule is Z" — but never actually observes cell content.
2. **No cell-level analysis**: Model never traces what happens to specific cells.
   Real reasoning: "Cell (0,3)=2 in input, cell (3,3)=2 in output — it shifted down 3 rows."
3. **Training-type pattern matching**: On benchmark puzzles, model maps to nearest
   training type instead of analyzing the actual transformation.

### What real intuition would look like
```
Let me examine the examples carefully.
Example 1: Input has non-zero cells at (0,3), (1,1), (2,5).
Output has non-zero cells at (3,3), (4,1), (5,5).
The cells shifted down by 3 rows. Colors are preserved.
Pattern: cells fall to the bottom of their column (gravity sort).
Applying to test: each column's non-zero cells stack at the bottom.
```

---

## v3 Run (started 2026-08-02)

### Design Rationale

v3 addresses the three core problems identified in v2 reasoning analysis:

#### 1. Cell-level analytical reasoning (fixes "template mimicry")
- SFT teacher traces now include **cell-level observations**: "4 cells unchanged;
  2 cells changed color (e.g. (0,4): 1→3); 18 cells appeared (e.g. (0,0): 0→3)"
- Structured format: **Observation → Pattern → Rule → Verification → Application**
- The model is taught to *describe what it sees* before stating a rule
- 470 analytical_cot examples (47% of dataset) with full cell-level analysis

#### 2. Cross-type transfer examples (fixes "training-type pattern matching")
- 90 examples teaching 6 transferable skills that directly map to benchmark puzzles:
  - **gravity** (cells fall to bottom) → transfers to gravity_sort benchmark
  - **sorting** (reorder by value) → transfers to gravity_sort, shape_sort
  - **pathfinding** (trace path through maze) → transfers to maze_path
  - **flood_fill** (fill connected regions) → transfers to region_coloring
  - **symmetry** (mirror/reflect) → transfers to symmetry_completion
  - **object_detection** (find largest object) → transfers to largest_replication
- These are NOT the same as benchmark puzzles — they teach the underlying *skill*
  with different surface patterns, so the model learns the skill not the pattern

#### 3. Larger token budgets (fixes "forced answers")
- Benchmark: think=512/answer=256 (was 384/192) — allows full analytical reasoning
- GRPO: think=256/pred=128 (was 192/128) — more reasoning during RL
- SFT max_seq_len=1024 (was 768) — accommodates longer analytical traces

### v3 Dataset
- 1003 training examples (20 env types × 50 + 6 transfer skills × 15)
- Avg assistant content: 524 chars (v2 was 366, v1 was 196)
- All examples have `</midt>` think tags
- 4 modes: 47% analytical_cot, 27% concise_analysis, 15% rule_inference, 20% rapid
- Est. ~393K tokens, ~391 tokens/example

### v3 Config
- SFT: 5 epochs, lr=2e-4, max_seq=1024, LoRA rank=16
- GRPO: 100 iters, group=6, think=256/pred=128, curriculum learning
- Benchmark: think=512, answer=256
- Timeout: 43200s (12 hours)

### Kaggle Resources
- Dataset: `krokodileceo/il-research-sft-v3-benchmark`
- Notebook: `krokodileceo/il-pipeline-v3-sft-grpo-on-r1-distill-qwen-1-5b`

### v3 Results
*(to be filled in after run completes)*

---

## Environment Types (20 total)

The IL environment suite has 20 grid transformation types, intentionally different
from the 12 benchmark puzzle types to test transfer of intuition:

| # | Name | Description | Difficulty |
|---|------|-------------|-----------|
| 1 | color_swap | All cells of color A → color B | Easy |
| 2 | rotate | Rotate grid 90/180/270° | Medium |
| 3 | border | Add colored border around grid | Medium |
| 4 | interior_fill | Fill enclosed empty regions | Hard |
| 5 | row_shift | Shift each row by its index | Medium |
| 6 | mirror_half | Mirror left/top half to right/bottom | Medium |
| 7 | crop | Crop to bounding box of non-zero | Hard |
| 8 | scale | Each cell → NxN block | Medium |
| 9 | adjacency | Color by neighbor count | Hard |
| 10 | threshold | Keep cells ≥ threshold | Easy |
| 11 | erosion | Remove outermost cells of objects | Hardest |
| 12 | dilation | Expand objects by 1 cell | Hard |
| 13 | transpose | Swap rows and columns | Medium |
| 14 | center_extract | Extract center NxN region | Medium |
| 15 | color_pos | Cells in even/odd positions +1 | Hard |
| 16 | outline | Draw outline around objects | Hard |
| 17 | max_row | Fill row with max value | Easy |
| 18 | flip_color | Reverse positions of one color | Medium |
| 19 | quadrant | Swap four quadrants diagonally | Hard |
| 20 | col_reverse | Reverse column order | Easy |

## Benchmark Puzzles (12 types, 36 total)

The benchmark uses 12 *different* puzzle types (3 each) to test transfer:
- Gravity + Column Sort
- Maze Path (BFS)
- Symmetry Completion
- Region Flood Coloring
- Largest Object Replication
- Tile Pattern Extrapolation
- Color Chain
- Block Expansion
- Conditional Transform
- Diagonal Fill
- Row Extremes Marking
- Shape Sort & Arrange

---

## Files

### v1
- `build_notebook.py` — notebook generator (v1 config preserved in git history)
- `il_dataset/` — v1 SFT dataset (316 examples)
- Kaggle: `krokodileceo/il-research-sft-benchmark` (v1 dataset)
- Kaggle: `krokodileceo/il-pipeline-sft-grpo-on-r1-distill-qwen-1-5b` (v1 notebook)

### v2
- `build_notebook.py` — notebook generator (v2 config, current)
- `gen_improved_data.py` — v2 dataset generator
- `il_dataset_v2/` — v2 SFT dataset (1104 examples, local copy)
- `il_dataset_v2_kaggle/` — v2 dataset upload staging
- Kaggle: `krokodileceo/il-research-sft-v2-benchmark` (v2 dataset)
- Kaggle: `krokodileceo/il-pipeline-v2-sft-grpo-on-r1-distill-qwen-1-5b` (v2 notebook)
