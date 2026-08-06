"""
Efficiency-aware reasoning quality scorer.

This is the core innovation for long-horizon reasoning tasks.
Instead of just grading correctness, we measure the QUALITY of the
reasoning process itself — this provides a much richer RL signal.

Four dimensions of reasoning quality:
1. Coverage: Did the model reason about the right concepts?
2. Efficiency: Did the model reason within a token budget?
3. Verification: Did the model check/verify its answer?
4. No-filler: Did the model avoid generic filler phrases?

Final reward shaping:
    final_score = correctness * (0.6 + 0.4 * reasoning_quality)

This ensures:
- Wrong answers get 0 regardless of reasoning quality
- Right answers with excellent reasoning get up to 1.0
- Right answers with poor reasoning get as low as 0.6
- The 0.4 spread is the RL signal that shapes reasoning behavior
"""
import re
from dataclasses import dataclass, field
from typing import Any


# ── Verification keywords — model checking its work ──
VERIFICATION_KEYWORDS = [
    "verify", "check", "let me check", "confirm", "trace through",
    "let me trace", "test this", "if i run", "let me verify",
    "to confirm", "double-check", "let me walk through", "mentally",
    "let me verify by", "tracing the execution", "walk through",
    "let me simulate", "step through", "let me step through",
]

# ── Filler phrases — generic boilerplate that wastes tokens ──
FILLER_PHRASES = [
    "let me think about this",
    "this is an interesting problem",
    "i need to analyze",
    "let me consider",
    "this is a complex",
    "i should first understand",
    "let me start by understanding",
    "this requires careful",
    "i'll need to think",
    "let me approach this",
    "this seems like",
    "i need to figure out",
    "let me break this down",
    "first, let me understand",
    "i should approach",
]

# ── Expected concepts per task (populated by each task) ──
EXPECTED_CONCEPTS: dict[str, list[str]] = {}


@dataclass
class ReasoningBreakdown:
    """Detailed breakdown of reasoning quality scoring."""
    coverage: float = 0.0          # fraction of expected concepts mentioned
    concepts_found: list[str] = field(default_factory=list)
    concepts_missing: list[str] = field(default_factory=list)
    token_efficiency: float = 0.0  # reward for reasoning within budget
    tokens_used: int = 0
    token_budget: int = 0
    verification: float = 0.0      # did the model verify its answer?
    verification_evidence: list[str] = field(default_factory=list)
    filler_penalty: float = 0.0    # penalty for generic filler
    filler_found: list[str] = field(default_factory=list)
    reasoning_quality: float = 0.0 # combined score [0, 1]
    reasoning_length: int = 0


def extract_reasoning(response: str) -> str:
    """Extract content between <reasoning> and </reasoning> tags."""
    match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_answer(response: str) -> str:
    """Extract content between <answer> and </answer> tags."""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: everything after </reasoning>
    match = re.search(r'</reasoning>\s*(.*)', response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def score_reasoning_quality(
    response: str,
    expected_concepts: list[str],
    token_budget: int = 600,
    correctness: float = 0.0,
) -> tuple[float, ReasoningBreakdown]:
    """Score the quality of reasoning in a model response.

    Args:
        response: the full model response
        expected_concepts: concepts the reasoning should cover
        token_budget: target token count for reasoning (rough = chars/4)
        correctness: the correctness score [0, 1] from the task grader

    Returns:
        (reasoning_quality, breakdown) where reasoning_quality is [0, 1]

    Scoring dimensions:
        coverage (40%): fraction of expected concepts mentioned in reasoning
        efficiency (30%): reward for reasoning within budget (only if correct)
        verification (20%): did the model verify/check its work?
        no_filler (10%): penalty for generic filler phrases
    """
    reasoning = extract_reasoning(response)
    reasoning_lower = reasoning.lower()
    breakdown = ReasoningBreakdown(
        token_budget=token_budget,
        reasoning_length=len(reasoning),
    )

    # ── 1. Coverage: did the model reason about the right things? ──
    breakdown.concepts_found = [
        c for c in expected_concepts if c.lower() in reasoning_lower
    ]
    breakdown.concepts_missing = [
        c for c in expected_concepts if c.lower() not in reasoning_lower
    ]
    breakdown.coverage = (
        len(breakdown.concepts_found) / len(expected_concepts)
        if expected_concepts else 1.0
    )

    # ── 2. Token efficiency: did the model reason within budget? ──
    breakdown.tokens_used = len(reasoning) // 4  # rough token estimate
    if correctness >= 0.5:
        # Only reward efficiency for mostly-correct answers
        if breakdown.tokens_used <= token_budget:
            # Within budget — full efficiency score
            # Bonus for being concise but thorough
            ratio = breakdown.tokens_used / max(1, token_budget)
            # Sweet spot: 50-90% of budget (thorough but not verbose)
            if 0.5 <= ratio <= 0.9:
                breakdown.token_efficiency = 1.0
            elif ratio < 0.5:
                # Too short — might be lazy
                breakdown.token_efficiency = 0.5 + ratio
            else:
                # Slightly over budget
                breakdown.token_efficiency = max(0.7, 1.0 - (ratio - 0.9) * 0.5)
        else:
            # Over budget — penalize verbosity
            overage = (breakdown.tokens_used - token_budget) / token_budget
            breakdown.token_efficiency = max(0.2, 1.0 - overage * 0.6)
    else:
        # Wrong answer — no efficiency credit
        breakdown.token_efficiency = 0.0

    # ── 3. Verification: did the model check its work? ──
    breakdown.verification_evidence = [
        kw for kw in VERIFICATION_KEYWORDS if kw in reasoning_lower
    ]
    # Score: 0.5 for any verification, up to 1.0 for multiple checks
    n_checks = len(breakdown.verification_evidence)
    breakdown.verification = min(1.0, 0.3 + n_checks * 0.15) if n_checks > 0 else 0.0

    # ── 4. No-filler: penalize generic boilerplate ──
    breakdown.filler_found = [
        phrase for phrase in FILLER_PHRASES if phrase in reasoning_lower
    ]
    n_filler = len(breakdown.filler_found)
    breakdown.filler_penalty = min(1.0, n_filler * 0.15)

    # ── Combined reasoning quality ──
    breakdown.reasoning_quality = (
        breakdown.coverage * 0.40
        + breakdown.token_efficiency * 0.30
        + breakdown.verification * 0.20
        + (1.0 - breakdown.filler_penalty) * 0.10
    )

    # Clamp to [0, 1]
    breakdown.reasoning_quality = max(0.0, min(1.0, breakdown.reasoning_quality))

    return breakdown.reasoning_quality, breakdown


def compute_final_score(correctness: float, reasoning_quality: float) -> float:
    """Compute the final efficiency-aware score.

    final_score = correctness * (0.6 + 0.4 * reasoning_quality)

    This means:
    - correctness=0 → final=0 (wrong answers never get credit)
    - correctness=1, reasoning_quality=0 → final=0.6 (right but lazy)
    - correctness=1, reasoning_quality=1 → final=1.0 (right and thorough)
    - correctness=0.5, reasoning_quality=0.5 → final=0.4 (partial both)
    """
    return correctness * (0.6 + 0.4 * reasoning_quality)


def shape_rl_reward(
    correctness: float,
    reasoning_quality: float,
    response: str,
    has_reasoning: bool = True,
    has_answer: bool = True,
) -> float:
    """Shape the final score into an RL reward with additional bonuses/penalties.

    Enhancements over raw final_score:
    - Bonus for having both reasoning AND answer (punishes skipping analysis)
    - Penalty for empty/no response
    - Bonus for verification (checking your work)
    - Small bonus for changing the correct file (shows codebase understanding)
    """
    base = compute_final_score(correctness, reasoning_quality)

    # Penalty for no reasoning (skipping analysis entirely)
    if not has_reasoning:
        base *= 0.3  # heavy penalty for no reasoning

    # Penalty for no answer
    if not has_answer:
        base *= 0.5

    # Bonus for high reasoning quality (thorough analysis)
    if reasoning_quality > 0.8 and correctness > 0.8:
        base += 0.05  # small bonus for excellent work

    # Clamp
    return max(0.0, min(1.1, base))
