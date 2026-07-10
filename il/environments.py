"""
IL Environment Suite — 20 diverse grid-world transformation types.

Each environment type provides:
  - gen_params(rng):  generate rule parameters (shared across instances)
  - gen_instance(params, rng):  generate one (input, output) grid pair
  - describe_rule(params):  human-readable rule description (teacher reasoning)
  - rule_name:  short name for logging

These are INTENTIONALLY DIFFERENT from the 12 benchmark puzzle types,
to test transfer of intuition rather than memorization.

Grid conventions: 2D arrays of integers 0-9. 0 = empty/background.
"""
import random
from copy import deepcopy

# ── Grid helpers (self-contained, no external deps) ──

def empty_grid(h, w, val=0):
    return [[val] * w for _ in range(h)]

def grid_copy(g):
    return [row[:] for row in g]

def grid_dims(g):
    return len(g), len(g[0])

def grid_to_str(g):
    rows = ['[' + ','.join(str(c) for c in row) + ']' for row in g]
    return '[' + ',\n '.join(rows) + ']'

def count_nonzero(g):
    return sum(1 for row in g for c in row if c != 0)

def distinct_colors(g):
    return set(c for row in g for c in row if c != 0)

# ============================================================
# ENV 1: Color Swap — all cells of color A become color B
# ============================================================
def color_swap_params(rng):
    a, b = rng.sample(range(1, 6), 2)
    return {'a': a, 'b': b}

def color_swap_instance(params, rng):
    h, w = rng.randint(4, 7), rng.randint(4, 7)
    g = empty_grid(h, w)
    for _ in range(rng.randint(4, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    out = grid_copy(g)
    for r in range(h):
        for c in range(w):
            if out[r][c] == params['a']:
                out[r][c] = params['b']
    return g, out

def color_swap_desc(params):
    return f"Every cell with color {params['a']} is changed to color {params['b']}. All other cells stay the same."

# ============================================================
# ENV 2: Grid Rotation — rotate 90/180/270 degrees
# ============================================================
def rotate_params(rng):
    return {'angle': rng.choice([90, 180, 270])}

def rotate_instance(params, rng):
    h, w = rng.randint(4, 7), rng.randint(4, 7)
    g = empty_grid(h, w)
    for _ in range(rng.randint(5, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    a = params['angle']
    if a == 90:
        out = [[g[h-1-r][c] for r in range(h)] for c in range(w)]
    elif a == 180:
        out = [row[::-1] for row in g[::-1]]
    else:  # 270
        out = [[g[r][w-1-c] for r in range(h)] for c in range(w-1, -1, -1)]
        out = [[g[r][w-1-c] for r in range(h)] for c in range(w)]
        out = out[::-1]
    return g, out

def rotate_desc(params):
    return f"Rotate the entire grid {params['angle']} degrees."

# ============================================================
# ENV 3: Border Addition — add a colored border around the grid
# ============================================================
def border_params(rng):
    return {'color': rng.randint(1, 5)}

def border_instance(params, rng):
    h, w = rng.randint(3, 6), rng.randint(3, 6)
    g = empty_grid(h, w)
    for _ in range(rng.randint(2, h * w // 3)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    col = params['color']
    out = empty_grid(h + 2, w + 2, 0)
    for r in range(h):
        for c in range(w):
            out[r+1][c+1] = g[r][c]
    for c in range(w + 2):
        out[0][c] = col
        out[h+1][c] = col
    for r in range(h + 2):
        out[r][0] = col
        out[r][w+1] = col
    return g, out

def border_desc(params):
    return f"Add a border of color {params['color']} around the entire grid."

# ============================================================
# ENV 4: Interior Fill — fill enclosed empty regions with a color
# ============================================================
def interior_fill_params(rng):
    return {'color': rng.randint(1, 5)}

def interior_fill_instance(params, rng):
    h, w = rng.randint(5, 8), rng.randint(5, 8)
    g = empty_grid(h, w)
    # Draw some walls (color 1)
    for _ in range(rng.randint(2, 4)):
        if rng.random() < 0.5:
            r = rng.randint(1, h-2)
            c1, c2 = sorted(rng.sample(range(w), 2))
            for c in range(c1, c2+1):
                g[r][c] = 1
        else:
            c = rng.randint(1, w-2)
            r1, r2 = sorted(rng.sample(range(h), 2))
            for r in range(r1, r2+1):
                g[r][c] = 1
    # Flood fill from edges to find exterior empty cells
    from collections import deque
    visited = empty_grid(h, w, False)
    q = deque()
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h-1 or c == 0 or c == w-1) and g[r][c] == 0:
                q.append((r, c))
                visited[r][c] = True
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and g[nr][nc]==0:
                visited[nr][nc] = True
                q.append((nr, nc))
    out = grid_copy(g)
    col = params['color']
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0 and not visited[r][c]:
                out[r][c] = col
    return g, out

def interior_fill_desc(params):
    return f"Fill all enclosed empty regions (surrounded by walls) with color {params['color']}. Exterior empty cells remain 0."

# ============================================================
# ENV 5: Row Shift — shift each row right by its row index (mod width)
# ============================================================
def row_shift_params(rng):
    return {}

def row_shift_instance(params, rng):
    h, w = rng.randint(4, 7), rng.randint(4, 7)
    g = empty_grid(h, w)
    for _ in range(rng.randint(4, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    out = empty_grid(h, w)
    for r in range(h):
        shift = r % w
        for c in range(w):
            out[r][(c + shift) % w] = g[r][c]
    return g, out

def row_shift_desc(params):
    return "Shift each row to the right by its row index (row 0 shifts 0, row 1 shifts 1, etc.), wrapping around."

# ============================================================
# ENV 6: Mirror Half — mirror the left half to the right half
# ============================================================
def mirror_half_params(rng):
    return {'axis': rng.choice(['h', 'v'])}

def mirror_half_instance(params, rng):
    h, w = rng.randint(4, 8), rng.randint(4, 8)
    g = empty_grid(h, w)
    for _ in range(rng.randint(5, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    out = grid_copy(g)
    if params['axis'] == 'h':
        for r in range(h):
            for c in range(w // 2):
                out[r][w - 1 - c] = out[r][c]
    else:
        for r in range(h // 2):
            for c in range(w):
                out[h - 1 - r][c] = out[r][c]
    return g, out

def mirror_half_desc(params):
    axis = "left half is mirrored to the right half" if params['axis'] == 'h' else "top half is mirrored to the bottom half"
    return f"The {axis}. The source half stays unchanged."

# ============================================================
# ENV 7: Crop to Content — remove empty border rows/columns
# ============================================================
def crop_params(rng):
    return {}

def crop_instance(params, rng):
    h, w = rng.randint(6, 9), rng.randint(6, 9)
    g = empty_grid(h, w)
    # Place content in inner area
    inner_h, inner_w = rng.randint(2, h-2), rng.randint(2, w-2)
    r0, c0 = rng.randint(0, h-inner_h), rng.randint(0, w-inner_w)
    for r in range(inner_h):
        for c in range(inner_w):
            if rng.random() < 0.5:
                g[r0+r][c0+c] = rng.randint(1, 5)
    # Find bounding box of non-zero
    nz = [(r, c) for r in range(h) for c in range(w) if g[r][c] != 0]
    if not nz:
        g[rng.randint(0,h-1)][rng.randint(0,w-1)] = 2
        nz = [(r, c) for r in range(h) for c in range(w) if g[r][c] != 0]
    rs = [r for r, c in nz]
    cs = [c for r, c in nz]
    rmin, rmax, cmin, cmax = min(rs), max(rs), min(cs), max(cs)
    out = [[g[r][c] for c in range(cmin, cmax+1)] for r in range(rmin, rmax+1)]
    return g, out

def crop_desc(params):
    return "Crop the grid to the bounding box of all non-zero cells, removing empty border rows and columns."

# ============================================================
# ENV 8: Scale Up — each cell becomes a NxN block
# ============================================================
def scale_params(rng):
    return {'factor': rng.randint(2, 3)}

def scale_instance(params, rng):
    h, w = rng.randint(3, 5), rng.randint(3, 5)
    g = empty_grid(h, w)
    for _ in range(rng.randint(3, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    f = params['factor']
    out = empty_grid(h * f, w * f)
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            for dr in range(f):
                for dc in range(f):
                    out[r*f+dr][c*f+dc] = v
    return g, out

def scale_desc(params):
    return f"Scale the grid up by {params['factor']}x — each cell becomes a {params['factor']}x{params['factor']} block of the same color."

# ============================================================
# ENV 9: Adjacency Coloring — color cells by number of non-zero neighbors
# ============================================================
def adjacency_params(rng):
    return {}

def adjacency_instance(params, rng):
    h, w = rng.randint(5, 8), rng.randint(5, 8)
    g = empty_grid(h, w)
    for _ in range(rng.randint(5, h * w // 3)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    out = empty_grid(h, w)
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0:
                n = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr][nc] != 0:
                        n += 1
                out[r][c] = min(n, 9)
            else:
                out[r][c] = g[r][c]
    return g, out

def adjacency_desc(params):
    return "For each empty cell, count its non-zero orthogonal neighbors and set the cell to that count. Non-empty cells stay unchanged."

# ============================================================
# ENV 10: Threshold Filter — keep only cells >= threshold, rest become 0
# ============================================================
def threshold_params(rng):
    return {'threshold': rng.randint(2, 4)}

def threshold_instance(params, rng):
    h, w = rng.randint(5, 8), rng.randint(5, 8)
    g = empty_grid(h, w)
    for _ in range(rng.randint(6, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    t = params['threshold']
    out = empty_grid(h, w)
    for r in range(h):
        for c in range(w):
            if g[r][c] >= t:
                out[r][c] = g[r][c]
    return g, out

def threshold_desc(params):
    return f"Keep only cells with value >= {params['threshold']}. All other cells become 0."

# ============================================================
# ENV 11: Erosion — remove the outermost cells of each object
# ============================================================
def erosion_params(rng):
    return {}

def erosion_instance(params, rng):
    from collections import deque
    h, w = rng.randint(5, 8), rng.randint(5, 8)
    g = empty_grid(h, w)
    for _ in range(rng.randint(4, 10)):
        color = rng.randint(1, 5)
        r0, c0 = rng.randint(0, h-1), rng.randint(0, w-1)
        size = rng.randint(2, 5)
        for _ in range(size):
            dr, dc = rng.choice([(-1,0),(1,0),(0,-1),(0,1)])
            nr, nc = r0+dr, c0+dc
            if 0<=nr<h and 0<=nc<w:
                g[nr][nc] = color
                r0, c0 = nr, nc
    out = grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                # Check if this is a border cell of its object
                is_border = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if not (0<=nr<h and 0<=nc<w) or g[nr][nc] != g[r][c]:
                        is_border = True
                        break
                if is_border:
                    out[r][c] = 0
    return g, out

def erosion_desc(params):
    return "Remove the outermost cells of each connected object (erosion). Interior cells remain."

# ============================================================
# ENV 12: Dilation — expand each object by 1 cell in all directions
# ============================================================
def dilation_params(rng):
    return {}

def dilation_instance(params, rng):
    h, w = rng.randint(5, 8), rng.randint(5, 8)
    g = empty_grid(h, w)
    for _ in range(rng.randint(3, 8)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    out = grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and out[nr][nc] == 0:
                        out[nr][nc] = g[r][c]
    return g, out

def dilation_desc(params):
    return "Expand each object by one cell in all 4 directions (dilation). New cells take the color of the expanding object."

# ============================================================
# ENV 13: Grid Transpose — swap rows and columns
# ============================================================
def transpose_params(rng):
    return {}

def transpose_instance(params, rng):
    h, w = rng.randint(4, 7), rng.randint(4, 7)
    g = empty_grid(h, w)
    for _ in range(rng.randint(5, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    out = [[g[r][c] for r in range(h)] for c in range(w)]
    return g, out

def transpose_desc(params):
    return "Transpose the grid — swap rows and columns. Element at (r,c) moves to (c,r)."

# ============================================================
# ENV 14: Center Extraction — extract the center NxN region
# ============================================================
def center_extract_params(rng):
    return {'size': rng.randint(2, 4)}

def center_extract_instance(params, rng):
    h, w = rng.randint(6, 9), rng.randint(6, 9)
    g = empty_grid(h, w)
    for _ in range(rng.randint(8, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    s = params['size']
    r0 = (h - s) // 2
    c0 = (w - s) // 2
    out = [[g[r0+r][c0+c] for c in range(s)] for r in range(s)]
    return g, out

def center_extract_desc(params):
    return f"Extract the center {params['size']}x{params['size']} region of the grid."

# ============================================================
# ENV 15: Color Replace by Position — cells in even columns get +1 color
# ============================================================
def color_pos_params(rng):
    return {'parity': rng.choice(['even_col', 'odd_col', 'even_row', 'odd_row'])}

def color_pos_instance(params, rng):
    h, w = rng.randint(4, 7), rng.randint(4, 7)
    g = empty_grid(h, w)
    for _ in range(rng.randint(5, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 4)
    out = grid_copy(g)
    p = params['parity']
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                if p == 'even_col' and c % 2 == 0:
                    out[r][c] = min(g[r][c] + 1, 9)
                elif p == 'odd_col' and c % 2 == 1:
                    out[r][c] = min(g[r][c] + 1, 9)
                elif p == 'even_row' and r % 2 == 0:
                    out[r][c] = min(g[r][c] + 1, 9)
                elif p == 'odd_row' and r % 2 == 1:
                    out[r][c] = min(g[r][c] + 1, 9)
    return g, out

def color_pos_desc(params):
    p = params['parity'].replace('_', ' ')
    return f"Non-zero cells in {p} positions get their color increased by 1 (max 9). Other cells stay the same."

# ============================================================
# ENV 16: Object Outline — draw outline around each object
# ============================================================
def outline_params(rng):
    return {}

def outline_instance(params, rng):
    h, w = rng.randint(5, 8), rng.randint(5, 8)
    g = empty_grid(h, w)
    for _ in range(rng.randint(2, 5)):
        color = rng.randint(1, 4)
        r0, c0 = rng.randint(1, h-2), rng.randint(1, w-2)
        for dr in range(rng.randint(1, 3)):
            for dc in range(rng.randint(1, 3)):
                if 0<=r0+dr<h and 0<=c0+dc<w:
                    g[r0+dr][c0+dc] = color
    out = grid_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0:
                # Check if adjacent to any non-zero
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr][nc] != 0:
                        out[r][c] = 9
                        break
    return g, out

def outline_desc(params):
    return "Draw an outline (color 9) around each object — fill empty cells that are adjacent to any non-zero cell with 9."

# ============================================================
# ENV 17: Max per Row — replace each row with its max value
# ============================================================
def max_row_params(rng):
    return {}

def max_row_instance(params, rng):
    h, w = rng.randint(4, 7), rng.randint(4, 7)
    g = empty_grid(h, w)
    for _ in range(rng.randint(5, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    out = empty_grid(h, w)
    for r in range(h):
        mx = max(g[r])
        if mx > 0:
            out[r] = [mx] * w
    return g, out

def max_row_desc(params):
    return "For each row, find the maximum value and fill the entire row with that value. Empty rows (all zeros) stay all zeros."

# ============================================================
# ENV 18: Flip Specific Color — reverse the order of cells of one color
# ============================================================
def flip_color_params(rng):
    return {'color': rng.randint(1, 5)}

def flip_color_instance(params, rng):
    h, w = rng.randint(4, 7), rng.randint(4, 7)
    g = empty_grid(h, w)
    for _ in range(rng.randint(5, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    col = params['color']
    # Find positions of the target color in row-major order
    positions = [(r, c) for r in range(h) for c in range(w) if g[r][c] == col]
    if len(positions) <= 1:
        # Ensure at least 2 cells of the target color
        for _ in range(2):
            r, c = rng.randint(0, h-1), rng.randint(0, w-1)
            g[r][c] = col
        positions = [(r, c) for r in range(h) for c in range(w) if g[r][c] == col]
    out = grid_copy(g)
    # Reverse: first position gets last position's value, etc.
    # Since all are the same color, this is a no-op visually.
    # Instead: move target color cells to reversed positions
    n = len(positions)
    for i, (r, c) in enumerate(positions):
        # Swap with the position at index n-1-i
        j = n - 1 - i
        r2, c2 = positions[j]
        out[r][c] = g[r2][c2]
        out[r2][c2] = g[r][c]
    # Actually, since all target cells are the same color, we need a different approach.
    # Let's make it: cells of the target color are moved to the opposite side of the grid
    out = grid_copy(g)
    target_cells = [(r, c) for r in range(h) for c in range(w) if g[r][c] == col]
    # Clear target cells
    for r, c in target_cells:
        out[r][c] = 0
    # Place them in reversed row-major order
    for i, (r, c) in enumerate(target_cells):
        r2, c2 = target_cells[len(target_cells) - 1 - i]
        out[r2][c2] = col
    return g, out

def flip_color_desc(params):
    return f"Take all cells of color {params['color']} and reverse their positions in row-major order."

# ============================================================
# ENV 19: Quadrant Swap — swap top-left with bottom-right, top-right with bottom-left
# ============================================================
def quadrant_params(rng):
    return {}

def quadrant_instance(params, rng):
    s = rng.choice([4, 6, 8])
    h = w = s
    g = empty_grid(h, w)
    for _ in range(rng.randint(6, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    out = empty_grid(h, w)
    mid = h // 2
    for r in range(mid):
        for c in range(mid):
            out[r][c] = g[r+mid][c+mid]
            out[r+mid][c+mid] = g[r][c]
            out[r][c+mid] = g[r+mid][c]
            out[r+mid][c] = g[r][c+mid]
    return g, out

def quadrant_desc(params):
    return "Swap the four quadrants diagonally: top-left swaps with bottom-right, top-right swaps with bottom-left."

# ============================================================
# ENV 20: Column Reverse — reverse the order of columns
# ============================================================
def col_reverse_params(rng):
    return {}

def col_reverse_instance(params, rng):
    h, w = rng.randint(4, 7), rng.randint(4, 7)
    g = empty_grid(h, w)
    for _ in range(rng.randint(5, h * w // 2)):
        g[rng.randint(0, h-1)][rng.randint(0, w-1)] = rng.randint(1, 5)
    out = [row[::-1] for row in g]
    return g, out

def col_reverse_desc(params):
    return "Reverse the order of columns — the leftmost column becomes the rightmost and vice versa."

# ============================================================
# REGISTRY
# ============================================================
ENVIRONMENT_TYPES = [
    {'name': 'color_swap',       'desc': 'Color Swap',           'gen_params': color_swap_params,       'gen_instance': color_swap_instance,       'describe': color_swap_desc},
    {'name': 'rotate',           'desc': 'Grid Rotation',        'gen_params': rotate_params,           'gen_instance': rotate_instance,           'describe': rotate_desc},
    {'name': 'border',           'desc': 'Border Addition',      'gen_params': border_params,           'gen_instance': border_instance,           'describe': border_desc},
    {'name': 'interior_fill',    'desc': 'Interior Fill',        'gen_params': interior_fill_params,    'gen_instance': interior_fill_instance,    'describe': interior_fill_desc},
    {'name': 'row_shift',        'desc': 'Row Shift',            'gen_params': row_shift_params,        'gen_instance': row_shift_instance,        'describe': row_shift_desc},
    {'name': 'mirror_half',      'desc': 'Mirror Half',          'gen_params': mirror_half_params,      'gen_instance': mirror_half_instance,      'describe': mirror_half_desc},
    {'name': 'crop',             'desc': 'Crop to Content',      'gen_params': crop_params,             'gen_instance': crop_instance,             'describe': crop_desc},
    {'name': 'scale',            'desc': 'Scale Up',             'gen_params': scale_params,            'gen_instance': scale_instance,            'describe': scale_desc},
    {'name': 'adjacency',        'desc': 'Adjacency Coloring',   'gen_params': adjacency_params,        'gen_instance': adjacency_instance,        'describe': adjacency_desc},
    {'name': 'threshold',        'desc': 'Threshold Filter',     'gen_params': threshold_params,        'gen_instance': threshold_instance,        'describe': threshold_desc},
    {'name': 'erosion',          'desc': 'Erosion',              'gen_params': erosion_params,          'gen_instance': erosion_instance,          'describe': erosion_desc},
    {'name': 'dilation',         'desc': 'Dilation',             'gen_params': dilation_params,         'gen_instance': dilation_instance,         'describe': dilation_desc},
    {'name': 'transpose',        'desc': 'Grid Transpose',       'gen_params': transpose_params,        'gen_instance': transpose_instance,        'describe': transpose_desc},
    {'name': 'center_extract',   'desc': 'Center Extraction',    'gen_params': center_extract_params,   'gen_instance': center_extract_instance,   'describe': center_extract_desc},
    {'name': 'color_pos',        'desc': 'Color by Position',    'gen_params': color_pos_params,        'gen_instance': color_pos_instance,        'describe': color_pos_desc},
    {'name': 'outline',          'desc': 'Object Outline',       'gen_params': outline_params,          'gen_instance': outline_instance,          'describe': outline_desc},
    {'name': 'max_row',          'desc': 'Max per Row',          'gen_params': max_row_params,          'gen_instance': max_row_instance,          'describe': max_row_desc},
    {'name': 'flip_color',       'desc': 'Flip Color Positions', 'gen_params': flip_color_params,       'gen_instance': flip_color_instance,       'describe': flip_color_desc},
    {'name': 'quadrant',         'desc': 'Quadrant Swap',        'gen_params': quadrant_params,         'gen_instance': quadrant_instance,         'describe': quadrant_desc},
    {'name': 'col_reverse',      'desc': 'Column Reverse',       'gen_params': col_reverse_params,      'gen_instance': col_reverse_instance,      'describe': col_reverse_desc},
]

# ============================================================
# Puzzle generation (same interface as benchmark)
# ============================================================
def generate_puzzle(env_type, rng, n_examples=3):
    params = env_type['gen_params'](rng)
    examples = []
    for _ in range(n_examples):
        inp, out = env_type['gen_instance'](params, rng)
        examples.append({'input': inp, 'output': out})
    test_inp, test_out = env_type['gen_instance'](params, rng)
    return {
        'type': env_type['name'],
        'description': env_type['desc'],
        'rule': env_type['describe'](params),
        'params': str(params),
        'examples': examples,
        'test_input': test_inp,
        'test_output': test_out,
    }

def generate_dataset(n_per_type=25, seed=12345):
    rng = random.Random(seed)
    dataset = []
    for et in ENVIRONMENT_TYPES:
        for i in range(n_per_type):
            p = generate_puzzle(et, rng)
            p['id'] = f"{et['name']}_{i+1}"
            dataset.append(p)
    return dataset


if __name__ == '__main__':
    ds = generate_dataset(n_per_type=3, seed=42)
    print(f"Generated {len(ds)} puzzles from {len(ENVIRONMENT_TYPES)} environment types")
    for p in ds[:5]:
        h, w = grid_dims(p['test_input'])
        oh, ow = grid_dims(p['test_output'])
        print(f"  {p['id']}: {p['description']} | {h}x{w} -> {oh}x{ow} | rule: {p['rule'][:60]}")
    # Verify all puzzles have valid output
    bad = 0
    for p in ds:
        if not p['test_output'] or len(p['test_output']) == 0:
            bad += 1
    print(f"\nInvalid puzzles: {bad}/{len(ds)}")
