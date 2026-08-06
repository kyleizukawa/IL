"""
Minimal Change Identification — Precision reasoning task.

A calculator module has a bug: the percentage function uses integer division
(`//`) instead of float division (`/`). The correct fix is changing a single
character. The model must identify the root cause and make the MINIMAL fix,
not over-engineer the solution.

Grader: 0.7 * correctness + 0.3 * minimality, where minimality rewards
changing only 1 line.

Failure mode: small models over-engineer fixes, changing many lines when
a single character would suffice.
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    parse_code_blocks, apply_code_changes, run_tests,
    compute_test_score, code_similarity,
)


@register_long_horizon
class MinimalChangeIdentification(LongHorizonEnv):
    task_id = "minimal_change_identification"
    reasoning_skill = "Precision reasoning — finding the minimal correct fix"
    failure_mode = "Small models over-engineer fixes or change too much"
    token_budget = 500
    expected_concepts = ["minimal", "fix", "precise", "root cause", "over-engineer", "single line", "verify"]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        calculator = textwrap.dedent("""\
            \"\"\"Calculator module with percentage and statistics functions.\"\"\"


            def evaluate(a, b):
                \"\"\"Calculate the percentage of a relative to b.

                Returns (a / b) * 100 as a float.
                For example, evaluate(1, 4) should return 25.0.
                \"\"\"
                if b == 0:
                    raise ValueError("Cannot divide by zero")
                return a // b * 100


            def average(numbers):
                \"\"\"Calculate the average of a list of numbers.\"\"\"
                if not numbers:
                    return 0.0
                return sum(numbers) / len(numbers)


            def median(numbers):
                \"\"\"Calculate the median of a list of numbers.\"\"\"
                if not numbers:
                    return 0.0
                sorted_nums = sorted(numbers)
                n = len(sorted_nums)
                mid = n // 2
                if n % 2 == 0:
                    return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2.0
                else:
                    return float(sorted_nums[mid])


            def percentage_change(old, new):
                \"\"\"Calculate the percentage change from old to new.\"\"\"
                if old == 0:
                    return 0.0
                return ((new - old) / old) * 100.0


            def ratio(a, b):
                \"\"\"Calculate the ratio of a to b as a float.\"\"\"
                if b == 0:
                    raise ValueError("Cannot divide by zero")
                return a / b


            def format_result(value, decimals=2):
                \"\"\"Format a numeric result to specified decimal places.\"\"\"
                return f"{value:.{decimals}f}"
            """)

        test_file = textwrap.dedent("""\
            \"\"\"Tests for calculator.evaluate — verifies correct float results.\"\"\"
            from calculator import evaluate


            def test_simple_percentage():
                result = evaluate(1, 4)
                assert result == 25.0, f"Expected 25.0, got {result}"


            def test_half():
                result = evaluate(1, 2)
                assert result == 50.0, f"Expected 50.0, got {result}"


            def test_zero_numerator():
                result = evaluate(0, 5)
                assert result == 0.0, f"Expected 0.0, got {result}"


            def test_small_fraction():
                result = evaluate(1, 3)
                assert abs(result - 33.333333333333336) < 0.0001, f"Expected ~33.33, got {result}"


            def test_large_numbers():
                result = evaluate(750, 3000)
                assert result == 25.0, f"Expected 25.0, got {result}"


            def test_division_by_zero():
                try:
                    evaluate(1, 0)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    pass
            """)

        return {
            "calculator.py": calculator,
            "test_calculator.py": test_file,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent("""\
            The `evaluate` function in `calculator.py` has a bug. The tests in
            `test_calculator.py` are failing.

            Your task:
            1. Identify the root cause of the bug.
            2. Make the MINIMAL fix — change as few lines as possible.
            3. Do NOT rewrite the function or add unnecessary code.

            The goal is precision: the best fix changes exactly ONE line.

            Provide your fixed `calculator.py` in a code block.
            """)

    # ── Solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        fixed = codebase["calculator.py"].replace(
            "return a // b * 100",
            "return a / b * 100",
        )
        return {"calculator.py": fixed}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent("""\
            Let me trace through the evaluate function to find the root cause of
            the bug. The function is supposed to calculate (a / b) * 100 as a float.

            Looking at the code:
                def evaluate(a, b):
                    if b == 0:
                        raise ValueError("Cannot divide by zero")
                    return a // b * 100

            The docstring says it should return (a / b) * 100 as a float, and
            gives the example: evaluate(1, 4) should return 25.0.

            Let me trace evaluate(1, 4):
            - b = 4, not zero, so we proceed.
            - a // b = 1 // 4 = 0 (integer division! This truncates to 0)
            - 0 * 100 = 0
            - Returns 0, but expected 25.0.

            The root cause is clear: the function uses `//` (integer division)
            instead of `/` (float division). In Python, `1 // 4` gives 0 because
            integer division truncates toward negative infinity, while `1 / 4`
            gives 0.25, and 0.25 * 100 = 25.0.

            Let me verify with another test case: evaluate(1, 2).
            - With `//`: 1 // 2 = 0, 0 * 100 = 0. Wrong (expected 50.0).
            - With `/`: 1 / 2 = 0.5, 0.5 * 100 = 50.0. Correct.

            And evaluate(1, 3):
            - With `//`: 1 // 3 = 0, 0 * 100 = 0. Wrong (expected ~33.33).
            - With `/`: 1 / 3 = 0.333..., 0.333... * 100 = 33.333... Correct.

            The fix is minimal: change `//` to `/` on the return line. This is a
            single character change (well, two characters: `//` -> `/`).

            I should NOT over-engineer this. I don't need to:
            - Add float() conversion (unnecessary, / already returns float)
            - Change the function structure
            - Add error handling (already present)
            - Modify other functions (average, median, etc. are correct)

            Let me confirm the other functions are correct:
            - average(): uses sum() / len() — correct float division.
            - median(): uses / 2.0 — correct.
            - percentage_change(): uses / old — correct.
            - ratio(): uses a / b — correct.

            Only evaluate() has the bug. The minimal fix is changing line:
                return a // b * 100
            to:
                return a / b * 100

            This is the single line that needs to change. Let me verify this fix
            passes all 6 tests:
            - test_simple_percentage: 1/4*100 = 25.0 ✓
            - test_half: 1/2*100 = 50.0 ✓
            - test_zero_numerator: 0/5*100 = 0.0 ✓
            - test_small_fraction: 1/3*100 = 33.333... ✓
            - test_large_numbers: 750/3000*100 = 25.0 ✓
            - test_division_by_zero: still raises ValueError ✓

            The fix is precise and minimal — exactly one line changed, one
            operator modified. No over-engineering needed.
            """)

    # ── Grader ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        changes = parse_code_blocks(response)
        if not changes:
            return 0.0, {"reason": "no code blocks found in response"}

        new_codebase = apply_code_changes(codebase, changes)

        if "calculator.py" not in new_codebase:
            return 0.0, {"reason": "calculator.py not in response"}

        # Run tests
        test_code = codebase["test_calculator.py"]
        results = run_tests(new_codebase, test_code, timeout=10.0)
        test_score, test_breakdown = compute_test_score(results)

        # Compute minimality: how many lines differ from original
        original = codebase["calculator.py"].split('\n')
        modified = new_codebase["calculator.py"].split('\n')

        changed_lines = 0
        max_len = max(len(original), len(modified))
        for i in range(max_len):
            orig_line = original[i].strip() if i < len(original) else ""
            mod_line = modified[i].strip() if i < len(modified) else ""
            if orig_line != mod_line:
                changed_lines += 1

        if changed_lines <= 1:
            minimality = 1.0
        elif changed_lines <= 3:
            minimality = 0.5
        else:
            minimality = 0.2

        score = 0.7 * test_score + 0.3 * minimality

        return score, {
            "test_score": test_score,
            "test_breakdown": test_breakdown,
            "minimality": minimality,
            "changed_lines": changed_lines,
            "combined_score": score,
            "results": results.get("results", []),
        }
