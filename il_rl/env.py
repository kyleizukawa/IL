"""
RL Environment for Intuition Learning.

Wraps a puzzle from the IL environment suite into a multi-step episode:
  Step 0: model sees examples, emits hypothesis + prediction (pure intuition)
  Step 1: model gets feedback (cell accuracy), refines hypothesis + prediction
  Step 2: final attempt after second round of feedback

The agent is told NOTHING about the rule. It must explore its own understanding
path. Rewards are:
  - Per-step: predictive improvement (cell_acc_t - cell_acc_{t-1})
  - Terminal: +1.0 if exact match, discounted by gamma^step
  - Regression penalty: negative reward if accuracy drops

This rewards the QUALITY of the understanding path, not mere step count.
"""
import re
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'il'))
from environments import (
    ENVIRONMENT_TYPES, generate_puzzle, grid_to_str, grid_dims, grid_copy
)


def parse_grid(text):
    """Extract a 2D integer grid from model output text.

    Uses bracket-depth matching to find the last complete 2D array,
    then falls back to individual row extraction.
    """
    last_2d = None
    i = 0
    while i < len(text):
        if text[i] == '[':
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '[':
                    depth += 1
                elif text[j] == ']':
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j+1]
                        if candidate.count('[') >= 3:
                            last_2d = candidate
                        i = j
                        break
            else:
                break
        i += 1

    if last_2d:
        rows = re.findall(r'\[\s*([\d\s,]+)\]', last_2d)
        grid = []
        for row_str in rows:
            nums = re.findall(r'\d+', row_str)
            if nums:
                grid.append([int(x) for x in nums])
        if grid:
            return grid

    row_matches = re.findall(r'\[\s*\d+[\s,\d]*\]', text)
    if row_matches:
        grid = []
        for row_str in row_matches:
            nums = re.findall(r'\d+', row_str)
            if nums:
                grid.append([int(x) for x in nums])
        if grid:
            return grid

    return None


def grids_equal(g1, g2):
    if g1 is None or g2 is None:
        return False
    if len(g1) != len(g2):
        return False
    for r1, r2 in zip(g1, g2):
        if len(r1) != len(r2):
            return False
        if r1 != r2:
            return False
    return True


def cell_accuracy(pred, target):
    if pred is None or target is None:
        return 0.0
    if len(pred) != len(target):
        return 0.0
    if len(pred) == 0 or len(pred[0]) == 0:
        return 0.0
    h, w = len(target), len(target[0])
    correct = 0
    total = h * w
    for r in range(min(len(pred), h)):
        for c in range(min(len(pred[r]), w)):
            if r < len(target) and c < len(target[r]) and pred[r][c] == target[r][c]:
                correct += 1
    return correct / total if total > 0 else 0.0


def overlap_accuracy(pred, target):
    """Cell accuracy on the overlapping region, regardless of shape mismatch.

    This gives a continuous signal even when the predicted grid has wrong
    dimensions. Different rollouts will produce different overlap accuracies,
    providing a learning signal for GRPO.
    """
    if pred is None or target is None:
        return 0.0
    if len(pred) == 0 or len(target) == 0:
        return 0.0
    overlap_h = min(len(pred), len(target))
    if overlap_h == 0:
        return 0.0
    total = 0
    correct = 0
    for r in range(overlap_h):
        pred_w = len(pred[r]) if r < len(pred) else 0
        tgt_w = len(target[r]) if r < len(target) else 0
        overlap_w = min(pred_w, tgt_w)
        for c in range(overlap_w):
            total += 1
            if pred[r][c] == target[r][c]:
                correct += 1
    return correct / total if total > 0 else 0.0


def signal_accuracy(pred, target):
    """Accuracy on NON-BACKGROUND (non-zero) cells of the target only.

    This is the key fix for the GRPO signal weakness. The old ``overlap_accuracy``
    counts ALL matching cells including background (0s). On sparse ARC grids
    (often 80-90% background), predicting all-zeros gets 80-90% overlap accuracy.
    This makes all rollouts look the same → near-zero GRPO advantages → no
    learning signal.

    ``signal_accuracy`` only scores the cells that actually matter — the colored
    cells that encode the transformation output. A prediction that gets the
    background right but the signal wrong scores 0%. This creates real variance
    between rollouts: one that finds 3/10 signal cells scores 30%, one that
    finds 7/10 scores 70%. That variance is what GRPO needs.
    """
    if pred is None or target is None:
        return 0.0
    if len(pred) == 0 or len(target) == 0:
        return 0.0
    signal_cells = []
    for r in range(len(target)):
        for c in range(len(target[r])):
            if target[r][c] != 0:
                signal_cells.append((r, c, target[r][c]))
    if not signal_cells:
        # No signal cells in target — fall back to full cell accuracy
        return cell_accuracy(pred, target)
    correct = 0
    for r, c, val in signal_cells:
        if r < len(pred) and c < len(pred[r]) and pred[r][c] == val:
            correct += 1
    return correct / len(signal_cells)


def is_degenerate(pred):
    """Check if a prediction is degenerate (lazy/gaming the reward).

    Degenerate predictions: None, empty, single-cell, or all-zeros.
    These get high overlap_accuracy on sparse grids but carry zero understanding.
    """
    if pred is None or len(pred) == 0:
        return True
    if len(pred) == 1 and len(pred[0]) <= 1:
        return True
    return all(all(cell == 0 for cell in row) for row in pred)


def shape_distance(pred, target):
    """Normalized shape similarity (0=wrong, 1=perfect shape match).

    Rewards grids that are closer to the correct dimensions.
    """
    if pred is None or len(pred) == 0:
        return 0.0
    exp_h, exp_w = len(target), len(target[0])
    pred_h = len(pred)
    pred_w = len(pred[0]) if pred_h > 0 else 0
    h_sim = 1.0 - abs(pred_h - exp_h) / max(exp_h, pred_h, 1)
    w_sim = 1.0 - abs(pred_w - exp_w) / max(exp_w, pred_w, 1)
    return (h_sim + w_sim) / 2


class RLEnvironment:
    """Manages a single multi-step episode for one puzzle.

    The agent is told nothing about the rule. It explores by emitting
    hypotheses and predictions, receiving feedback after each attempt.
    """

    def __init__(self, puzzle, n_steps=3, gamma=0.9):
        self.puzzle = puzzle
        self.n_steps = n_steps
        self.gamma = gamma
        self.test_output = puzzle['test_output']
        self.test_input = puzzle['test_input']
        self.examples = puzzle['examples']
        self.step = 0
        self.prev_accuracy = 0.0
        self.prev_quality = 0.0
        self.best_accuracy = 0.0
        self.best_quality = 0.0
        self.first_correct_step = None
        self.history = []

    def build_initial_prompt(self):
        """Build the first prompt: examples + instructions. No rule info."""
        lines = []
        lines.append("You are an abstract reasoning system. You will be given example input-output grid pairs that demonstrate an unknown transformation rule. You must figure out the rule yourself.")
        lines.append("")
        lines.append("Grids are 2D arrays of integers 0-9. 0 represents empty/background.")
        lines.append("")
        lines.append("=== EXAMPLES ===")
        lines.append("")
        for i, ex in enumerate(self.examples):
            lines.append(f"--- Example {i+1} ---")
            lines.append("Input:")
            lines.append(grid_to_str(ex['input']))
            lines.append("Output:")
            lines.append(grid_to_str(ex['output']))
            lines.append("")
        lines.append("=== TEST ===")
        lines.append("")
        lines.append("Input:")
        lines.append(grid_to_str(self.test_input))
        lines.append("")
        lines.append("Look at the examples, figure out the transformation rule, and predict the test output grid.")
        lines.append("Output your reasoning followed by the predicted grid as a 2D array.")
        return '\n'.join(lines)

    def build_feedback_prompt(self, step, accuracy, predicted_grid):
        """Build feedback for steps 1+. No rule info, only performance signal.

        Format aligned with SFT — simple, direct, no rigid HYPOTHESIS/PREDICTION
        template that the model didn't learn during SFT.
        """
        h, w = grid_dims(self.test_output)
        total_cells = h * w
        correct_cells = int(accuracy * total_cells)

        lines = []
        lines.append(f"FEEDBACK: Your prediction has {accuracy*100:.0f}% cell accuracy ({correct_cells}/{total_cells} cells correct).")

        if accuracy == 1.0:
            lines.append("Your prediction is exactly correct!")
        elif accuracy > 0:
            lines.append("Your prediction is partially correct. Some cells are wrong.")
        else:
            lines.append("Your prediction does not match the expected output at all.")

        # Dimension hint if wrong shape
        if predicted_grid is not None:
            pred_h, pred_w = len(predicted_grid), len(predicted_grid[0]) if predicted_grid else 0
            exp_h, exp_w = grid_dims(self.test_output)
            if (pred_h, pred_w) != (exp_h, exp_w):
                lines.append(f"Hint: the output grid has dimensions {exp_h}x{exp_w}, but your prediction was {pred_h}x{pred_w}.")

        if step < self.n_steps - 1:
            lines.append("")
            lines.append("Refine your understanding and predict again.")
            lines.append("Output your reasoning followed by the predicted grid as a 2D array.")
        else:
            lines.append("")
            lines.append("This is your final attempt. Make your best prediction.")
            lines.append("Output your reasoning followed by the predicted grid as a 2D array.")

        return '\n'.join(lines)

    def process_action(self, generated_text):
        """Process the model's generated text for this step.

        Returns (reward, accuracy, predicted_grid, done).

        REWARD REDESIGN (v2):
        The v1 reward used overlap_accuracy (counts ALL cells including
        background 0s). On sparse ARC grids (~80% background), all rollouts
        got 0.7-0.9 overlap → near-zero GRPO advantages → no learning.
        It also had a perverse incentive: doing badly on step 0 to get a
        big "improvement" on step 1.

        v2 fixes:
        - signal_accuracy: scores ONLY non-background cells (the transformation
          output). This creates real variance between rollouts.
        - Big terminal bonus (3.0 * gamma^step) for exact match — creates the
          spike that GRPO needs. One rollout solving exactly gets 3x reward,
          making the advantage large.
        - Degenerate prediction penalty: all-zeros/single-cell predictions
          get quality * 0.3 (no more gaming with lazy outputs).
        - Positive-improvement-only bonus: only rewards GETTING BETTER, not
          the artificial "improvement" from being deliberately bad on step 0.
        """
        predicted_grid = parse_grid(generated_text)
        accuracy = cell_accuracy(predicted_grid, self.test_output)
        exact = grids_equal(predicted_grid, self.test_output)

        # Shape match check
        exp_h, exp_w = grid_dims(self.test_output)
        shape_match = False
        if predicted_grid is not None and len(predicted_grid) > 0:
            pred_h = len(predicted_grid)
            pred_w = len(predicted_grid[0]) if pred_h > 0 else 0
            shape_match = (pred_h == exp_h and pred_w == exp_w)

        # Color overlap: fraction of expected colors present in prediction
        exp_colors = set(c for row in self.test_output for c in row if c != 0)
        if predicted_grid:
            pred_colors = set(c for row in predicted_grid for c in row if c != 0)
            color_overlap = len(pred_colors & exp_colors) / max(len(exp_colors), 1)
        else:
            color_overlap = 0.0

        # Composite quality — signal_accuracy is the dominant component.
        # This is what creates variance between rollouts for GRPO.
        quality = signal_accuracy(predicted_grid, self.test_output)
        quality += 0.15 * shape_distance(predicted_grid, self.test_output)
        quality += 0.05 * color_overlap

        # Degenerate prediction penalty — no reward for lazy outputs
        if is_degenerate(predicted_grid):
            quality *= 0.3

        quality = min(quality, 1.0)

        # Process reward: absolute quality at each step (weighted by gamma^step)
        # This rewards being good at EVERY step, not just improving.
        process_reward = (self.gamma ** self.step) * quality

        # IL improvement bonus: reward POSITIVE improvement only.
        # This is the IL signal — the agent's ability to refine its understanding
        # from feedback. Only triggered when the agent actually gets better,
        # eliminating the "do badly on step 0" perverse incentive.
        if self.step > 0 and quality > self.prev_quality:
            improvement = quality - self.prev_quality
            process_reward += (self.gamma ** self.step) * improvement * 0.5

        # Track first correct step
        if exact and self.first_correct_step is None:
            self.first_correct_step = self.step

        # Terminal bonus: BIG spike for exact match (3.0 * gamma^step).
        # This is the key variance creator for GRPO. When one rollout in the
        # group solves exactly and others don't, the advantage is large.
        # v1 had gamma^step (~0.9) — too small relative to the dense quality
        # (~0.8) to create meaningful variance. v2 has 3.0 * gamma^step.
        terminal_reward = 0.0
        done = (self.step >= self.n_steps - 1) or exact
        if exact:
            terminal_reward = (self.gamma ** self.step) * 3.0

        reward = process_reward + terminal_reward

        self.history.append({
            'step': self.step,
            'accuracy': accuracy,
            'quality': quality,
            'exact': exact,
            'shape_match': shape_match,
            'color_overlap': color_overlap,
            'predicted_grid': predicted_grid,
            'process_reward': process_reward,
            'terminal_reward': terminal_reward,
            'reward': reward,
        })

        self.prev_accuracy = accuracy
        self.prev_quality = quality
        self.best_accuracy = max(self.best_accuracy, accuracy)
        self.best_quality = max(getattr(self, 'best_quality', 0), quality)
        self.step += 1

        return reward, accuracy, predicted_grid, done

    def total_reward(self):
        """Total discounted reward for the entire episode."""
        return sum(h['reward'] for h in self.history)

    def summary(self):
        """Human-readable episode summary for logging."""
        accs = [f"{h['accuracy']:.2f}" for h in self.history]
        exact = any(h['exact'] for h in self.history)
        return (f"steps={len(self.history)} accs=[{','.join(accs)}] "
                f"best={self.best_accuracy:.2f} exact={exact} "
                f"total_reward={self.total_reward():.3f}")


def sample_puzzle(rng=None):
    """Sample a random puzzle from the IL environment suite."""
    if rng is None:
        rng = random.Random()
    et = rng.choice(ENVIRONMENT_TYPES)
    puzzle = generate_puzzle(et, rng)
    puzzle['id'] = f"{et['name']}_{rng.randint(0, 99999)}"
    return puzzle
