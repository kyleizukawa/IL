"""
Task: reachability_analysis

Reasoning skill: Code path reasoning — identifying unreachable code.

The model is given a `validator.py` module with three pieces of dead code:
  1. A branch with a contradictory condition (`x > 100 and x < 50`).
  2. A function `legacy_validate` that is never called.
  3. A code path after `return True` that can never execute.

This is a Q&A + code task: the model must explain which code is unreachable
(text) and remove it (code).

Failure mode: small models cannot reason about which branches are reachable
and either leave dead code or accidentally remove live code.
"""
import re
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer, extract_reasoning, parse_code_blocks,
    apply_code_changes, run_tests, compute_test_score,
)


@register_long_horizon
class ReachabilityAnalysis(LongHorizonEnv):
    task_id = "reachability_analysis"
    reasoning_skill = "Code path reasoning — identifying unreachable code"
    failure_mode = (
        "Small models cannot reason about which branches are reachable and "
        "either leave dead code in place or accidentally remove live code."
    )
    token_budget = 600
    expected_concepts = [
        "reachable", "unreachable", "dead code", "branch",
        "condition", "flow", "verify",
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        validator = textwrap.dedent('''
            """Input validation module.

            Contains several pieces of dead code that should be identified and
            removed without changing the observable behavior of validate_input.
            """

            def validate_input(x):
                """Validate a numeric input.

                Returns:
                    - "negative" if x < 0
                    - "small"    if 0 <= x < 50
                    - "large"    if x >= 100
                    - "medium"   otherwise (50 <= x < 100)

                Raises:
                    TypeError if x is not an int or float.
                """
                if not isinstance(x, (int, float)):
                    raise TypeError("x must be numeric")

                if x < 0:
                    return "negative"

                # --- DEAD CODE #1: contradictory condition ---
                # x > 100 AND x < 50 can never be True simultaneously.
                if x > 100 and x < 50:
                    return "impossible"

                if x < 50:
                    return "small"

                if x >= 100:
                    return "large"

                return True

                # --- DEAD CODE #2: unreachable after return ---
                print("this line is never reached")
                return "medium"


            def legacy_validate(x):
                """Old validation routine kept for reference.

                --- DEAD CODE #3: never called from anywhere ---
                """
                if x < 0:
                    return False
                return True


            def validate_batch(items):
                """Validate a list of inputs, returning a list of results."""
                results = []
                for item in items:
                    results.append(validate_input(item))
                return results
        ''').strip()

        tests = textwrap.dedent('''
            import validator


            # ── Behavior tests (must still pass after dead-code removal) ──

            def test_negative():
                assert validator.validate_input(-5) == "negative"

            def test_small():
                assert validator.validate_input(25) == "small"

            def test_large():
                assert validator.validate_input(150) == "large"

            # ── Structural tests (dead code must be gone) ──

            def test_no_legacy_validate():
                # legacy_validate should be removed entirely.
                assert not hasattr(validator, "legacy_validate"), \\
                    "legacy_validate should be removed"

            def test_no_contradictory_branch():
                import inspect
                src = inspect.getsource(validator.validate_input)
                # The contradictory condition "x > 100 and x < 50" must be gone.
                assert "x > 100 and x < 50" not in src, \\
                    "contradictory branch must be removed"

            def test_no_unreachable_after_return():
                import inspect
                src = inspect.getsource(validator.validate_input)
                # Nothing should appear after the final return except
                # possibly whitespace.  Check that "print" and the
                # unreachable "medium" return are gone.
                assert "this line is never reached" not in src, \\
                    "unreachable code after return must be removed"
                # The function should end with a return statement, not
                # have code after it.
                lines = [l for l in src.split("\\n") if l.strip()]
                assert lines[-1].strip().startswith("return"), \\
                    "function should end with a return"
        ''').strip()

        return {
            "validator.py": validator,
            "test_reachability.py": tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''
            You are given `validator.py`, which contains three pieces of
            unreachable (dead) code:

            1. A branch with a contradictory condition that can never be True.
            2. A function that is never called from anywhere in the module.
            3. A code path after a `return` statement that can never execute.

            Your task has two parts:

            (A) In your reasoning, identify each piece of dead code and explain
                WHY it is unreachable — trace the condition or control flow
                that makes it dead.

            (B) Remove all three pieces of dead code from `validator.py`
                WITHOUT changing the observable behavior of `validate_input`
                and `validate_batch`.

            The 6 tests in `test_reachability.py` must pass: 3 behavior tests
            confirm the functions still work, and 3 structural tests confirm
            the dead code is gone.

            Return your solution as a code block tagged with the filename:

            ```python:validator.py
            ...
            ```
        ''').strip()

    # ── Correct solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        fixed = textwrap.dedent('''
            """Input validation module — dead code removed."""

            def validate_input(x):
                """Validate a numeric input.

                Returns:
                    - "negative" if x < 0
                    - "small"    if 0 <= x < 50
                    - "large"    if x >= 100
                    - "medium"   otherwise (50 <= x < 100)

                Raises:
                    TypeError if x is not an int or float.
                """
                if not isinstance(x, (int, float)):
                    raise TypeError("x must be numeric")

                if x < 0:
                    return "negative"

                if x < 50:
                    return "small"

                if x >= 100:
                    return "large"

                return "medium"


            def validate_batch(items):
                """Validate a list of inputs, returning a list of results."""
                results = []
                for item in items:
                    results.append(validate_input(item))
                return results
        ''').strip()

        return {"validator.py": fixed}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''
            Let me read validator.py and reason about the control flow of
            each function to find all unreachable code.

            Step 1 — Analyze validate_input branch by branch:
            - `if not isinstance(x, (int, float)): raise TypeError` — guard,
              reachable for non-numeric input.
            - `if x < 0: return "negative"` — reachable for negatives.
            - `if x > 100 and x < 50: return "impossible"` — DEAD CODE #1.
              This condition is contradictory: a number cannot be both
              greater than 100 and less than 50 simultaneously.  The branch
              is unreachable for any value of x.  I can verify by checking
              the condition: x > 100 requires x >= 101, x < 50 requires
              x <= 49; no real number satisfies both.  Remove it.
            - `if x < 50: return "small"` — reachable for 0 <= x < 50.
            - `if x >= 100: return "large"` — reachable for x >= 100.
            - `return True` — reachable for 50 <= x < 100.  But wait — the
              docstring says this case should return "medium", not True.
              The `return True` is followed by `print(...)` and
              `return "medium"`, which are DEAD CODE #2 — unreachable after
              the return.  The correct fix is to change `return True` to
              `return "medium"` and remove the dead lines after it.  This
              preserves the documented behavior (the "medium" case) while
              removing the unreachable code.

            Step 2 — Analyze legacy_validate:
            It is defined but never called from validate_input, validate_batch,
            or anywhere else in the module.  I searched the entire file for
            "legacy_validate" and the only occurrence is the def line.  No
            internal caller, and the module does not export it in __all__.
            DEAD CODE #3 — remove the entire function.

            Step 3 — Analyze validate_batch:
            It calls validate_input in a loop.  No dead code here.  Keep it
            unchanged.

            Step 4 — Apply the removals:
            - Remove the contradictory `if x > 100 and x < 50` branch.
            - Change `return True` to `return "medium"` and delete the
              `print(...)` and `return "medium"` lines after it.
            - Delete the `legacy_validate` function entirely.

            Step 5 — Verify behavior is preserved:
            - validate_input(-5): x < 0 -> "negative". OK.
            - validate_input(25): not < 0, not contradictory (removed), x < 50
              -> "small". OK.
            - validate_input(75): not < 0, not < 50, not >= 100 -> "medium".
              This now returns "medium" instead of True, which matches the
              docstring. OK.
            - validate_input(150): x >= 100 -> "large". OK.
            - validate_batch([1, 2]) -> ["small", "small"]. OK.

            Step 6 — Verify structural tests:
            - legacy_validate is gone -> hasattr returns False. OK.
            - "x > 100 and x < 50" not in source. OK.
            - No "this line is never reached" in source; function ends with
              `return "medium"`. OK.

            Let me double-check I did not remove any reachable code: the
            contradictory branch was truly impossible, the code after return
            was unreachable by definition, and legacy_validate had zero
            callers.  All removals are safe.  The only behavior change is
            fixing the "medium" case from `return True` to `return "medium"`,
            which is a bug fix required by the dead-code removal (the real
            "medium" return was unreachable after `return True`).
        ''').strip()

    # ── Grader ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        answer = extract_answer(response)
        changes = parse_code_blocks(answer)
        if not changes:
            reasoning = extract_reasoning(response)
            changes = parse_code_blocks(reasoning)
        if not changes:
            return 0.0, {"reason": "no code blocks found in response"}

        new_codebase = apply_code_changes(codebase, changes)
        test_code = codebase.get("test_reachability.py", "")
        results = run_tests(new_codebase, test_code, timeout=15.0)
        test_score, test_breakdown = compute_test_score(results)

        # Split: 3 behavior + 3 structural -> 0.5 each
        test_results = results.get("results", [])
        behavior_names = {
            "test_negative", "test_small", "test_large",
        }
        structural_names = {
            "test_no_legacy_validate",
            "test_no_contradictory_branch",
            "test_no_unreachable_after_return",
        }
        behavior_pass = sum(
            1 for r in test_results
            if r["name"] in behavior_names and r["status"] == "pass"
        )
        structural_pass = sum(
            1 for r in test_results
            if r["name"] in structural_names and r["status"] == "pass"
        )
        behavior_score = behavior_pass / 3.0 if behavior_names else 0.0
        dead_code_score = structural_pass / 3.0 if structural_names else 0.0
        score = 0.5 * behavior_score + 0.5 * dead_code_score

        breakdown = {
            "total": results.get("total", 0),
            "passed": results.get("passed", 0),
            "behavior_score": behavior_score,
            "dead_code_score": dead_code_score,
            "score": score,
            "results": test_results,
            "method": "behavior + structural split",
        }
        return score, breakdown
