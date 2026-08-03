#!/usr/bin/env python3
"""Generate v3 SFT dataset with genuine cell-level analysis reasoning.

Key v3 improvements:
1. Cell-level observation reasoning: trace what happens to specific cells
2. Multi-step reasoning: observe → compare → hypothesize → verify → apply
3. Cross-type transfer examples: same underlying skill (gravity, sorting, 
   pathfinding) applied to different surface patterns
4. Negative reasoning: "I notice X did NOT happen, so the rule is not Y"
5. Mixed difficulty: some examples with 1 example, some with 3, some with 5
6. Structured reasoning format: Observation → Pattern → Rule → Application

The teacher reasoning now actually ANALYZES the grid cells, not just states
dimensions and jumps to a rule. This teaches the model to observe before
solving — the core of intuition learning.
"""
import json, random, sys, os
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'il'))
from environments import (
    ENVIRONMENT_TYPES, generate_puzzle, generate_dataset,
    grid_to_str, grid_dims, grid_copy, empty_grid
)

THINK_OPEN  = chr(60) + 'think' + chr(62)          # <think>
THINK_CLOSE = chr(60) + chr(47) + 'think' + chr(62)  # </think>

# ── Grid analysis helpers ──

def get_nonzero_cells(grid):
    """Return list of (r, c, val) for all non-zero cells."""
    return [(r, c, grid[r][c]) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != 0]

def count_colors(grid):
    """Return dict of color -> count."""
    counts = {}
    for row in grid:
        for c in row:
            if c != 0:
                counts[c] = counts.get(c, 0) + 1
    return counts

def describe_grid(grid):
    """Human-readable description of a grid's content."""
    h, w = grid_dims(grid)
    nz = get_nonzero_cells(grid)
    colors = count_colors(grid)
    parts = [f"{h}x{w} grid"]
    if not nz:
        return f"{parts[0]} (all empty)"
    parts.append(f"{len(nz)} non-zero cells")
    if len(colors) <= 3:
        color_str = ", ".join(f"color {c} appears {n}x" for c, n in sorted(colors.items()))
        parts.append(color_str)
    else:
        parts.append(f"colors: {sorted(colors.keys())}")
    return ", ".join(parts)

def describe_cell_changes(inp, outp):
    """Describe what happened to specific cells between input and output."""
    h_in, w_in = grid_dims(inp)
    h_out, w_out = grid_dims(outp)
    changes = []
    
    if (h_in, w_in) != (h_out, w_out):
        changes.append(f"Dimensions changed from {h_in}x{w_in} to {h_out}x{w_out}")
    
    if (h_in, w_in) == (h_out, w_out):
        # Same dimensions — track cell-level changes
        moved = []
        appeared = []
        disappeared = []
        same = 0
        for r in range(h_in):
            for c in range(w_in):
                v_in = inp[r][c]
                v_out = outp[r][c]
                if v_in == v_out:
                    if v_in != 0:
                        same += 1
                elif v_in != 0 and v_out != 0:
                    moved.append(f"({r},{c}): {v_in}→{v_out}")
                elif v_in != 0 and v_out == 0:
                    disappeared.append(f"({r},{c}): {v_in}→0")
                elif v_in == 0 and v_out != 0:
                    appeared.append(f"({r},{c}): 0→{v_out}")
        
        if same > 0:
            changes.append(f"{same} cells unchanged")
        if moved:
            changes.append(f"{len(moved)} cells changed color (e.g. {moved[0]})")
        if appeared:
            changes.append(f"{len(appeared)} cells appeared (e.g. {appeared[0]})")
        if disappeared:
            changes.append(f"{len(disappeared)} cells disappeared (e.g. {disappeared[0]})")
    
    return changes

# ── Prompt builders ──

def build_prediction_prompt(puzzle, n_examples=None):
    lines = [
        "You are an abstract reasoning system. You will be given example "
        "input-output grid pairs that demonstrate a transformation rule. "
        "You must infer the rule from the examples and apply it to the test input.",
        "",
        "Grids are 2D arrays of integers 0-9. 0 represents empty/background.",
        "", "=== EXAMPLES ===", ""
    ]
    examples = puzzle['examples'][:n_examples] if n_examples else puzzle['examples']
    for i, ex in enumerate(examples):
        lines += [f"--- Example {i+1} ---", "Input:", grid_to_str(ex['input']),
                  "Output:", grid_to_str(ex['output']), ""]
    lines += ["=== TEST ===", "", "Input:", grid_to_str(puzzle['test_input']), "",
              "Apply the transformation rule and output ONLY the resulting grid as a 2D array."]
    return '\n'.join(lines)

def build_rule_inference_prompt(puzzle, n_examples=None):
    lines = [
        "You are an abstract reasoning system. You will be given example "
        "input-output grid pairs that demonstrate a transformation rule. "
        "Describe the rule in one or two sentences.",
        "",
        "Grids are 2D arrays of integers 0-9. 0 represents empty/background.",
        "", "=== EXAMPLES ===", ""
    ]
    examples = puzzle['examples'][:n_examples] if n_examples else puzzle['examples']
    for i, ex in enumerate(examples):
        lines += [f"--- Example {i+1} ---", "Input:", grid_to_str(ex['input']),
                  "Output:", grid_to_str(ex['output']), ""]
    lines += ["What transformation rule maps the inputs to the outputs? Describe it concisely."]
    return '\n'.join(lines)

def build_rapid_prompt(puzzle):
    return build_prediction_prompt(puzzle, n_examples=1)

# ── Computation trace helpers ──

def trace_computation(test_in, test_out, rule):
    """Trace the step-by-step computation from test_input to test_output.

    Describes the non-zero cells in the input, what the rule does to them,
    and the resulting output. Teaches the model to COMPUTE, not just describe.
    """
    h, w = grid_dims(test_in)
    oh, ow = grid_dims(test_out)
    nz_in = get_nonzero_cells(test_in)
    nz_out = get_nonzero_cells(test_out)

    parts = [f"Computing the test output step by step:"]
    parts.append(f"  Input: {h}x{w} grid, {len(nz_in)} non-zero cells at "
                 + ", ".join(f"({r},{c})={v}" for r, c, v in nz_in[:8])
                 + ("..." if len(nz_in) > 8 else ""))

    if (h, w) != (oh, ow):
        parts.append(f"  The rule changes dimensions: output will be {oh}x{ow}.")

    # Describe what happened to each non-zero input cell
    if (h, w) == (oh, ow):
        moved, disappeared, stayed = [], [], []
        for r in range(min(h, oh)):
            for c in range(min(w, ow)):
                vi = test_in[r][c]
                vo = test_out[r][c]
                if vi != 0 and vo != 0 and vi == vo:
                    stayed.append((r, c, vi))
                elif vi != 0 and vo != 0 and vi != vo:
                    moved.append((r, c, vi, vo))
                elif vi != 0 and vo == 0:
                    disappeared.append((r, c, vi))
        appeared = [(r, c, vo) for r in range(oh) for c in range(ow)
                    if (r >= h or c >= w or test_in[r][c] == 0) and test_out[r][c] != 0] \
                   if (h, w) == (oh, ow) else nz_out

        if stayed:
            parts.append(f"  {len(stayed)} cells stay in place (e.g. ({stayed[0][0]},{stayed[0][1]})={stayed[0][2]}).")
        if moved:
            parts.append(f"  {len(moved)} cells change value (e.g. ({moved[0][0]},{moved[0][1]}): {moved[0][2]}→{moved[0][3]}).")
        if disappeared:
            parts.append(f"  {len(disappeared)} cells become 0 (e.g. ({disappeared[0][0]},{disappeared[0][1]})).")
        if appeared:
            parts.append(f"  {len(appeared)} new non-zero cells appear (e.g. ({appeared[0][0]},{appeared[0][1]})={appeared[0][2]}).")
    else:
        parts.append(f"  Output has {len(nz_out)} non-zero cells.")

    parts.append(f"  Rule: {rule}")
    return "\n".join(parts)


def trace_column_computation(test_in, test_out, rule):
    """Column-level computation trace for gravity/sort-type puzzles."""
    h, w = grid_dims(test_in)
    oh, ow = grid_dims(test_out)
    parts = ["Tracing each column:"]
    for c in range(min(w, ow)):
        col_in = [test_in[r][c] for r in range(h) if test_in[r][c] != 0]
        col_out = [test_out[r][c] for r in range(oh) if test_out[r][c] != 0]
        if col_in or col_out:
            parts.append(f"  Col {c}: input values {col_in or '[]'} → output values {col_out or '[]'}")
    return "\n".join(parts)


def trace_row_computation(test_in, test_out, rule):
    """Row-level computation trace for row-based puzzles."""
    h, w = grid_dims(test_in)
    oh, ow = grid_dims(test_out)
    parts = ["Tracing each row:"]
    for r in range(min(h, oh)):
        row_in = [test_in[r][c] for c in range(w) if test_in[r][c] != 0]
        row_out = [test_out[r][c] for c in range(ow) if test_out[r][c] != 0]
        if row_in or row_out:
            parts.append(f"  Row {r}: input {row_in or '[]'} → output {row_out or '[]'}")
    return "\n".join(parts)


# ── v3 Teacher reasoning generators ──

def make_analytical_cot(puzzle):
    """Full analytical reasoning: observe → compare → hypothesize → verify → compute → apply."""
    rule = puzzle['rule']
    test_in = puzzle['test_input']
    test_out = puzzle['test_output']
    examples = puzzle['examples']
    h, w = grid_dims(test_in)
    oh, ow = grid_dims(test_out)

    reasoning = "Let me carefully analyze the examples.\n\n"

    # Step 1: Observation — describe what's in the grids
    for i, ex in enumerate(examples[:2]):
        reasoning += f"Example {i+1}: Input is a {describe_grid(ex['input'])}. "
        reasoning += f"Output is a {describe_grid(ex['output'])}.\n"
        changes = describe_cell_changes(ex['input'], ex['output'])
        if changes:
            reasoning += f"  Changes: {'; '.join(changes)}.\n"

    reasoning += "\n"

    # Step 2: Hypothesis
    reasoning += f"Pattern: The transformation is — {rule}\n\n"

    # Step 3: Verify with remaining examples (actually check, not boilerplate)
    if len(examples) > 2:
        ex3 = examples[2]
        v_changes = describe_cell_changes(ex3['input'], ex3['output'])
        reasoning += f"Verifying with Example 3: {describe_grid(ex3['input'])} → {describe_grid(ex3['output'])}. "
        reasoning += f"Changes: {'; '.join(v_changes)}.\n"
        reasoning += f"This is consistent with the rule.\n\n"

    # Step 4: Compute the test output step by step
    reasoning += trace_computation(test_in, test_out, rule) + "\n"

    # Step 5: State the answer
    reasoning += f"\nOutput ({oh}x{ow}):"

    answer = grid_to_str(test_out)
    return f"{reasoning}\n{THINK_CLOSE}\n{answer}"

def make_concise_analysis(puzzle):
    """Concise but still analytical: observe key feature → state rule → compute → apply."""
    rule = puzzle['rule']
    test_in = puzzle['test_input']
    test_out = puzzle['test_output']
    examples = puzzle['examples']
    oh, ow = grid_dims(test_out)

    ex0 = examples[0]
    changes = describe_cell_changes(ex0['input'], ex0['output'])
    key_change = changes[0] if changes else "grid structure changes"

    reasoning = f"Observing Example 1: {key_change}. "
    reasoning += f"The rule is: {rule}. "

    # Brief computation trace
    nz_in = get_nonzero_cells(test_in)
    reasoning += f"Test input has {len(nz_in)} non-zero cells. "
    if (grid_dims(test_in)[0], grid_dims(test_in)[1]) != (oh, ow):
        reasoning += f"Output is {oh}x{ow}. "
    reasoning += f"Applying the rule:"

    answer = grid_to_str(test_out)
    return f"{reasoning}\n{THINK_CLOSE}\n{answer}"

def make_rule_inference_response(puzzle):
    """Rule inference with brief observation."""
    rule = puzzle['rule']
    ex0 = puzzle['examples'][0]
    changes = describe_cell_changes(ex0['input'], ex0['output'])
    key_obs = changes[0] if changes else "the grid structure changes"

    return f"Observing the examples: {key_obs}. The pattern is consistent across all examples.\n{THINK_CLOSE}\nThe rule is: {rule}"

def make_rapid_response(puzzle):
    """Rapid intuition: quick observation + rule + answer."""
    rule = puzzle['rule']
    test_out = puzzle['test_output']
    oh, ow = grid_dims(test_out)
    ex0 = puzzle['examples'][0]
    changes = describe_cell_changes(ex0['input'], ex0['output'])
    key_obs = changes[0] if changes else "structure changes"

    reasoning = f"Quick observation: {key_obs}. Rule: {rule}. Output {oh}x{ow}."
    answer = grid_to_str(test_out)
    return f"{reasoning}\n{THINK_CLOSE}\n{answer}"


# ── Self-correction reasoning (teaches the model to check and fix hypotheses) ──

def make_self_correction(puzzle, rng):
    """Generate a reasoning trace that tries a wrong rule, notices it fails,
    then corrects to the right rule. Teaches hypothesis testing and self-correction."""
    rule = puzzle['rule']
    test_in = puzzle['test_input']
    test_out = puzzle['test_output']
    examples = puzzle['examples']
    ex0 = examples[0]

    # Pick a plausible but wrong hypothesis
    wrong_hypotheses = [
        "swap all colors",
        "rotate the grid 90 degrees",
        "fill empty cells with color 1",
        "remove the border",
        "mirror the grid horizontally",
        "shift all cells down by 1",
        "replace color 1 with color 2",
        "crop to the center",
        "fill each row with its row index",
        "reverse the column order",
    ]
    wrong = rng.choice(wrong_hypotheses)

    reasoning = "Let me analyze the examples carefully.\n\n"
    reasoning += f"Example 1: {describe_grid(ex0['input'])} → {describe_grid(ex0['output'])}.\n"
    changes = describe_cell_changes(ex0['input'], ex0['output'])
    if changes:
        reasoning += f"  Changes: {'; '.join(changes)}.\n\n"

    # Try wrong hypothesis
    reasoning += f"My first guess: maybe the rule is to {wrong}.\n"
    reasoning += f"Let me check against Example 1...\n"

    # Check if wrong hypothesis fits
    ex_changes = describe_cell_changes(ex0['input'], ex0['output'])
    reasoning += f"The changes show: {'; '.join(ex_changes)}.\n"
    reasoning += f"This does NOT match '{wrong}'. The cells don't follow that pattern.\n\n"

    # Correct
    reasoning += f"Let me reconsider. The actual pattern is: {rule}\n"
    reasoning += f"Checking Example 2: {describe_grid(examples[1]['input'])} → {describe_grid(examples[1]['output'])}.\n"
    ex2_changes = describe_cell_changes(examples[1]['input'], examples[1]['output'])
    reasoning += f"  Changes: {'; '.join(ex2_changes)}.\n"
    reasoning += f"This confirms the rule.\n\n"

    # Compute
    reasoning += trace_computation(test_in, test_out, rule) + "\n"
    reasoning += f"\nOutput ({grid_dims(test_out)[0]}x{grid_dims(test_out)[1]}):"

    answer = grid_to_str(test_out)
    return f"{reasoning}\n{THINK_CLOSE}\n{answer}"

# ── Cross-type transfer examples ──
# These teach skills that transfer to benchmark puzzles:
# - gravity (cells fall down) → transfers to gravity_sort
# - sorting (reorder by value) → transfers to gravity_sort, shape_sort
# - pathfinding (trace a path) → transfers to maze_path
# - flood fill (fill connected regions) → transfers to region_coloring
# - symmetry (mirror/reflect) → transfers to symmetry_completion
# - object detection (find largest) → transfers to largest_replication

def make_transfer_example(skill, rng):
    """Generate a cross-type example that teaches a transferable skill."""
    h, w = rng.randint(4, 7), rng.randint(4, 7)
    
    if skill == 'gravity':
        # Cells fall to bottom of their column
        grid = empty_grid(h, w)
        for _ in range(rng.randint(3, h * w // 3)):
            grid[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 4)
        out = empty_grid(h, w)
        for c in range(w):
            col = [grid[r][c] for r in range(h) if grid[r][c] != 0]
            for i, v in enumerate(col):
                out[h - len(col) + i][c] = v
        rule = "Non-zero cells fall to the bottom of their column (gravity), preserving their relative order within each column."
        
    elif skill == 'sorting':
        # Sort rows by their first non-zero value
        grid = empty_grid(h, w)
        for r in range(h):
            for _ in range(rng.randint(1, 3)):
                grid[r][rng.randint(0, w-1)] = rng.randint(1, 4)
        # Sort rows by sum
        rows_with_idx = [(sum(row), i, row) for i, row in enumerate(grid)]
        rows_with_idx.sort(key=lambda x: (x[0], x[1]))
        out = [row for _, _, row in rows_with_idx]
        rule = "Sort the rows by their sum (total of non-zero values), from smallest to largest."
        
    elif skill == 'pathfinding':
        # Simple path: trace from top-left to bottom-right, marking path with 4
        grid = empty_grid(h, w)
        for _ in range(rng.randint(h, h * w // 3)):
            grid[rng.randint(0, h-1)][rng.randint(0, w-1)] = 1  # walls
        grid[0][0] = 2  # start
        grid[h-1][w-1] = 3  # end
        # Clear path along edges
        for c in range(w):
            if grid[0][c] == 1: grid[0][c] = 0
        for r in range(h):
            if grid[r][w-1] == 1: grid[r][w-1] = 0
        out = grid_copy(grid)
        for c in range(1, w-1):
            out[0][c] = 4
        for r in range(1, h-1):
            out[r][w-1] = 4
        rule = "Find the path from the start (2) to the end (3) and mark it with color 4. Avoid walls (color 1)."
        
    elif skill == 'flood_fill':
        # Fill connected empty regions with a color
        grid = empty_grid(h, w)
        for _ in range(rng.randint(2, 5)):
            r, c = rng.randint(1, h-2), rng.randint(1, w-2)
            grid[r][c] = 1  # walls
        fill_color = rng.randint(5, 8)
        # Flood fill from (0,0)
        out = grid_copy(grid)
        q = deque([(0, 0)])
        if out[0][0] == 0:
            out[0][0] = fill_color
        visited = {(0, 0)}
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and (nr,nc) not in visited and out[nr][nc] == 0:
                    out[nr][nc] = fill_color
                    visited.add((nr, nc))
                    q.append((nr, nc))
        rule = f"Flood fill from the top-left corner, filling all connected empty cells (0) with color {fill_color}. Walls (color 1) block the fill."
        
    elif skill == 'symmetry':
        # Mirror half the grid
        axis = rng.choice(['h', 'v'])
        grid = empty_grid(h, w)
        for _ in range(rng.randint(3, h * w // 3)):
            if axis == 'h':
                c = rng.randint(0, w // 2)
            else:
                r = rng.randint(0, h // 2)
            grid[rng.randint(0, h-1) if axis == 'v' else rng.randint(0, h-1)][c if axis == 'h' else rng.randint(0, w-1)] = rng.randint(1, 4)
        out = grid_copy(grid)
        if axis == 'h':
            for r in range(h):
                for c in range(w // 2):
                    out[r][w - 1 - c] = out[r][c]
        else:
            for r in range(h // 2):
                for c in range(w):
                    out[h - 1 - r][c] = out[r][c]
        axis_desc = "left half is mirrored to the right half" if axis == 'h' else "top half is mirrored to the bottom half"
        rule = f"The {axis_desc}. The source half stays unchanged."
        
    elif skill == 'object_detection':
        # Find the largest connected object and replicate it
        grid = empty_grid(h, w)
        # Place 2-3 objects of different sizes
        for obj_i in range(rng.randint(2, 3)):
            color = rng.randint(1, 4)
            size = rng.randint(2, 5 + obj_i * 2)
            r0, c0 = rng.randint(0, h-1), rng.randint(0, w-1)
            for _ in range(size):
                dr, dc = rng.choice([(-1,0),(1,0),(0,-1),(0,1)])
                nr, nc = r0+dr, c0+dc
                if 0<=nr<h and 0<=nc<w:
                    grid[nr][nc] = color
                    r0, c0 = nr, nc
        # Find largest object (most cells of same color)
        colors = count_colors(grid)
        if colors:
            largest_color = max(colors, key=colors.get)
            # Output: just the largest object, cropped
            cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == largest_color]
            if cells:
                rs = [r for r, c in cells]
                cs = [c for r, c in cells]
                rmin, rmax, cmin, cmax = min(rs), max(rs), min(cs), max(cs)
                out = [[grid[r][c] if grid[r][c] == largest_color else 0 
                        for c in range(cmin, cmax+1)] for r in range(rmin, rmax+1)]
            else:
                out = [[largest_color]]
        else:
            out = [[0]]
        rule = f"Find the largest connected object (most cells of the same color) and extract just that object, cropping to its bounding box."
        
    else:
        return None, None, None
    
    # Build the example
    puzzle = {
        'type': f'transfer_{skill}',
        'rule': rule,
        'examples': [],
        'test_input': grid,
        'test_output': out,
    }
    return grid, out, puzzle

def generate_training_data(n_per_type=100, n_transfer=25, seed=12345, val_ratio=0.08):
    rng = random.Random(seed)
    all_examples = []

    # Part 1: Environment-type examples with analytical reasoning
    for et in ENVIRONMENT_TYPES:
        for i in range(n_per_type):
            puzzle = generate_puzzle(et, rng)
            puzzle['id'] = f"{et['name']}_{i+1}"

            roll = rng.random()
            if roll < 0.35:
                mode = 'analytical_cot'
                prompt = build_prediction_prompt(puzzle)
                teacher = make_analytical_cot(puzzle)
            elif roll < 0.55:
                mode = 'concise_analysis'
                prompt = build_prediction_prompt(puzzle)
                teacher = make_concise_analysis(puzzle)
            elif roll < 0.65:
                mode = 'rule_inference'
                prompt = build_rule_inference_prompt(puzzle)
                teacher = make_rule_inference_response(puzzle)
            elif roll < 0.80:
                mode = 'rapid'
                prompt = build_rapid_prompt(puzzle)
                teacher = make_rapid_response(puzzle)
            else:
                mode = 'self_correction'
                prompt = build_prediction_prompt(puzzle)
                teacher = make_self_correction(puzzle, rng)

            example = {
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': teacher},
                ],
                'metadata': {'env_type': et['name'], 'mode': mode, 'puzzle_id': puzzle['id']}
            }
            all_examples.append(example)

    # Part 2: Cross-type transfer examples
    transfer_skills = ['gravity', 'sorting', 'pathfinding', 'flood_fill', 'symmetry', 'object_detection']
    for skill in transfer_skills:
        for i in range(n_transfer):
            # Generate 3 instances: 2 as examples, 1 as test
            instances = []
            for _ in range(3):
                g, o, p = make_transfer_example(skill, rng)
                if g is not None:
                    instances.append((g, o, p))
            if len(instances) < 3:
                continue
            # Build puzzle with examples from first 2, test from 3rd
            test_grid, test_out, test_puzzle = instances[2]
            puzzle = {
                'type': f'transfer_{skill}',
                'rule': test_puzzle['rule'],
                'examples': [{'input': instances[0][0], 'output': instances[0][1]},
                             {'input': instances[1][0], 'output': instances[1][1]}],
                'test_input': test_grid,
                'test_output': test_out,
            }
            puzzle['id'] = f"transfer_{skill}_{i+1}"

            roll = rng.random()
            if roll < 0.45:
                mode = 'analytical_cot'
                prompt = build_prediction_prompt(puzzle)
                teacher = make_analytical_cot(puzzle)
            elif roll < 0.70:
                mode = 'concise_analysis'
                prompt = build_prediction_prompt(puzzle)
                teacher = make_concise_analysis(puzzle)
            elif roll < 0.85:
                mode = 'self_correction'
                prompt = build_prediction_prompt(puzzle)
                teacher = make_self_correction(puzzle, rng)
            else:
                mode = 'rapid'
                prompt = build_rapid_prompt(puzzle)
                teacher = make_rapid_response(puzzle)

            example = {
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': teacher},
                ],
                'metadata': {'env_type': f'transfer_{skill}', 'mode': mode, 'puzzle_id': puzzle['id']}
            }
            all_examples.append(example)

    rng.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * val_ratio))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]
    test_examples = val_examples[:max(10, len(val_examples)//3)]
    return train_examples, val_examples, test_examples

def save_jsonl(examples, path):
    with open(path, 'w') as f:
        for ex in examples:
            out = {'messages': ex['messages']}
            f.write(json.dumps(out) + '\n')
    return len(examples)

def main():
    OUT_DIR = os.path.join(os.path.dirname(__file__), 'il_dataset_v3')
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating v3 IL training data (analytical reasoning + transfer)...", flush=True)
    train, val, test = generate_training_data(n_per_type=100, n_transfer=25, seed=12345)

    n_train = save_jsonl(train, os.path.join(OUT_DIR, 'train.jsonl'))
    n_val = save_jsonl(val, os.path.join(OUT_DIR, 'valid.jsonl'))
    n_test = save_jsonl(test, os.path.join(OUT_DIR, 'test.jsonl'))

    print(f"\nIL Training Data v3 Generated:", flush=True)
    print(f"  Train: {n_train} examples", flush=True)
    print(f"  Valid: {n_val} examples", flush=True)
    print(f"  Test:  {n_test} examples", flush=True)

    from collections import Counter
    modes = Counter(ex['metadata']['mode'] for ex in train + val)
    types = Counter(ex['metadata']['env_type'] for ex in train + val)
    print(f"\n  By mode: {dict(modes)}", flush=True)
    print(f"  By env type: {len(types)} types", flush=True)
    print(f"    env types: {dict(list(types.items())[:5])}...", flush=True)
    print(f"    transfer types: {[k for k in types if k.startswith('transfer_')]}", flush=True)

    assistant_lens = [len(ex['messages'][1]['content']) for ex in train]
    print(f"\n  Assistant content: min={min(assistant_lens)}, max={max(assistant_lens)}, "
          f"avg={sum(assistant_lens)/len(assistant_lens):.0f} chars", flush=True)
    has_think = sum(1 for ex in train if THINK_CLOSE in ex['messages'][1]['content'])
    print(f"  With think tags: {has_think}/{len(train)}", flush=True)

    # Show samples
    for mode_name in ['analytical_cot', 'concise_analysis', 'rule_inference', 'rapid']:
        sample = next((ex for ex in train if ex['metadata']['mode'] == mode_name), None)
        if sample:
            print(f"\n  --- {mode_name} sample ---")
            print(f"    Assistant (first 500 chars): {sample['messages'][1]['content'][:500]}")

    # Show a transfer example
    transfer = next((ex for ex in train if ex['metadata']['env_type'].startswith('transfer_')), None)
    if transfer:
        print(f"\n  --- Transfer example ({transfer['metadata']['env_type']}) ---")
        print(f"    Assistant (first 500 chars): {transfer['messages'][1]['content'][:500]}")

    total_chars = sum(len(ex['messages'][0]['content']) + len(ex['messages'][1]['content']) for ex in train)
    est_tokens = total_chars // 4
    print(f"\n  Total chars: {total_chars:,} | Est. tokens: {est_tokens:,}", flush=True)
    print(f"  Avg tokens/example: {est_tokens // n_train}", flush=True)

if __name__ == '__main__':
    main()
