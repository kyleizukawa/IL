"""
Coverage Gap Analysis — Coverage reasoning task.

A password checker has 6 code paths. The existing test file only covers 3
of them. The model must identify the 3 untested paths and write tests for
each gap.

Grader: run model's new tests against the correct implementation and score
based on fraction of previously-untested paths now covered.

Failure mode: small models write more tests for already-tested paths instead
of identifying the actual gaps.
"""
import re
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    parse_code_blocks, apply_code_changes, run_tests,
    compute_test_score,
)


@register_long_horizon
class CoverageGapAnalysis(LongHorizonEnv):
    task_id = "coverage_gap_analysis"
    reasoning_skill = "Coverage reasoning — identifying untested code paths"
    failure_mode = "Small models write more tests for already-tested paths"
    token_budget = 700
    expected_concepts = ["coverage", "untested", "path", "branch", "edge case", "missing", "gap", "verify"]

    # The 6 paths and which are tested
    _ALL_PATHS = ["normal", "empty", "too_short", "no_uppercase", "no_digit", "no_special"]
    _TESTED_PATHS = ["normal", "empty", "too_short"]
    _UNTESTED_PATHS = ["no_uppercase", "no_digit", "no_special"]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        password_checker = textwrap.dedent("""\
            \"\"\"Password strength checker module.\"\"\"


            def check_password(password):
                \"\"\"Check if a password meets security requirements.

                Requirements:
                - At least 8 characters long
                - Contains at least one uppercase letter
                - Contains at least one digit
                - Contains at least one special character (!@#$%^&*)

                Returns:
                    tuple: (is_valid: bool, message: str)

                Paths:
                1. Empty password -> (False, "Password cannot be empty")
                2. Too short -> (False, "Password must be at least 8 characters")
                3. No uppercase -> (False, "Password must contain an uppercase letter")
                4. No digit -> (False, "Password must contain a digit")
                5. No special char -> (False, "Password must contain a special character")
                6. Valid password -> (True, "Password is valid")
                \"\"\"
                if not password:
                    return (False, "Password cannot be empty")

                if len(password) < 8:
                    return (False, "Password must be at least 8 characters")

                has_upper = any(c.isupper() for c in password)
                if not has_upper:
                    return (False, "Password must contain an uppercase letter")

                has_digit = any(c.isdigit() for c in password)
                if not has_digit:
                    return (False, "Password must contain a digit")

                special_chars = "!@#$%^&*"
                has_special = any(c in special_chars for c in password)
                if not has_special:
                    return (False, "Password must contain a special character")

                return (True, "Password is valid")
            """)

        existing_tests = textwrap.dedent("""\
            \"\"\"Existing tests for check_password — only covers 3 of 6 paths.\"\"\"
            from password_checker import check_password


            def test_valid_password():
                \"\"\"Path 6: normal valid password.\"\"\"
                valid, msg = check_password("Secure1!")
                assert valid is True
                assert msg == "Password is valid"


            def test_empty_password():
                \"\"\"Path 1: empty password.\"\"\"
                valid, msg = check_password("")
                assert valid is False
                assert msg == "Password cannot be empty"


            def test_too_short():
                \"\"\"Path 2: password too short.\"\"\"
                valid, msg = check_password("Ab1!")
                assert valid is False
                assert msg == "Password must be at least 8 characters"
            """)

        return {
            "password_checker.py": password_checker,
            "test_password.py": existing_tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent("""\
            The `password_checker.py` module has a `check_password` function with
            6 code paths (documented in the docstring).

            The existing `test_password.py` only covers 3 of the 6 paths.

            Your task:
            1. Read the implementation and identify ALL 6 code paths.
            2. Read the existing tests and determine which paths are already covered.
            3. Identify the 3 UNTESTED paths (the coverage gaps).
            4. Write a new test file `test_gaps.py` with one test for each untested path.

            Each test should:
            - Call check_password with an input that triggers the untested path
            - Assert the correct return value (False) and error message

            Do NOT write tests for paths that are already covered.
            Provide your `test_gaps.py` in a code block.
            """)

    # ── Solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        test_gaps = textwrap.dedent("""\
            \"\"\"Tests for the 3 untested paths in check_password.\"\"\"
            from password_checker import check_password


            def test_no_uppercase():
                \"\"\"Path 3: password with no uppercase letter.\"\"\"
                valid, msg = check_password("secure1!")
                assert valid is False
                assert msg == "Password must contain an uppercase letter"


            def test_no_digit():
                \"\"\"Path 4: password with no digit.\"\"\"
                valid, msg = check_password("Securepass!")
                assert valid is False
                assert msg == "Password must contain a digit"


            def test_no_special_char():
                \"\"\"Path 5: password with no special character.\"\"\"
                valid, msg = check_password("Secure123")
                assert valid is False
                assert msg == "Password must contain a special character"
            """)
        return {"test_gaps.py": test_gaps}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent("""\
            I need to analyze the password checker for coverage gaps. Let me first
            map all code paths in check_password, then check which are tested.

            Reading the implementation of check_password:
            1. Empty password: `if not password` -> returns (False, "Password cannot be empty")
            2. Too short: `if len(password) < 8` -> returns (False, "Password must be at least 8 characters")
            3. No uppercase: `if not has_upper` -> returns (False, "Password must contain an uppercase letter")
            4. No digit: `if not has_digit` -> returns (False, "Password must contain a digit")
            5. No special char: `if not has_special` -> returns (False, "Password must contain a special character")
            6. Valid: falls through all checks -> returns (True, "Password is valid")

            So there are 6 distinct branch paths, each with an early return (except
            path 6). Each branch represents a different edge case in password validation.

            Now let me read the existing tests in test_password.py:
            - test_valid_password: calls check_password("Secure1!") -> triggers path 6 (valid).
              This covers path 6 (normal/valid).
            - test_empty_password: calls check_password("") -> triggers path 1 (empty).
              This covers path 1 (empty).
            - test_too_short: calls check_password("Ab1!") -> triggers path 2 (too short).
              This covers path 2 (too short).

            Coverage map:
            - Path 1 (empty): TESTED ✓
            - Path 2 (too short): TESTED ✓
            - Path 3 (no uppercase): UNTESTED ✗ — gap
            - Path 4 (no digit): UNTESTED ✗ — gap
            - Path 5 (no special char): UNTESTED ✗ — gap
            - Path 6 (valid): TESTED ✓

            The 3 coverage gaps are paths 3, 4, and 5. These missing tests represent
            untested branches that could hide bugs. I need to write tests for
            each of these untested paths.

            For path 3 (no uppercase): I need a password that is:
            - Not empty (passes check 1)
            - At least 8 chars (passes check 2)
            - Has NO uppercase letter (triggers check 3)
            Example: "secure1!" — 8 chars, lowercase, has digit, has special.
            Expected: (False, "Password must contain an uppercase letter")

            For path 4 (no digit): I need a password that is:
            - Not empty, at least 8 chars, has uppercase
            - Has NO digit (triggers check 4)
            Example: "Securepass!" — 11 chars, has uppercase 'S', no digit, has special.
            Expected: (False, "Password must contain a digit")

            For path 5 (no special char): I need a password that is:
            - Not empty, at least 8 chars, has uppercase, has digit
            - Has NO special character (triggers check 5)
            Example: "Secure123" — 9 chars, has uppercase 'S', has digits '123', no special.
            Expected: (False, "Password must contain a special character")

            Let me verify each test input actually triggers the intended path:

            "secure1!": not empty ✓, len=8 ✓, no uppercase (all lowercase) ✓ -> path 3.
            "Securepass!": not empty ✓, len=11 ✓, has 'S' ✓, no digit ✓ -> path 4.
            "Secure123": not empty ✓, len=9 ✓, has 'S' ✓, has '1' ✓, no special ✓ -> path 5.

            All three test inputs correctly trigger their intended untested paths.
            I should NOT write tests for paths 1, 2, or 6 — those are already covered
            in test_password.py. Writing redundant tests for covered paths would show
            a lack of coverage analysis.

            Let me verify the tests pass against the correct implementation by tracing
            each through check_password:
            - "secure1!" -> not empty, len=8>=8, no upper -> (False, "uppercase letter") ✓
            - "Securepass!" -> not empty, len=11>=8, has 'S', no digit -> (False, "digit") ✓
            - "Secure123" -> not empty, len=9>=8, has 'S', has '1', no special -> (False, "special") ✓

            All tests should pass. The 3 coverage gaps are now filled.
            """)

    # ── Grader ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        changes = parse_code_blocks(response)
        if not changes:
            return 0.0, {"reason": "no code blocks found in response"}

        # The model should provide test_gaps.py
        test_gaps_code = None
        for filename, content in changes.items():
            if "test_gaps" in filename or "test" in filename:
                test_gaps_code = content
                break

        if test_gaps_code is None:
            # Try any file that looks like tests
            for filename, content in changes.items():
                if "def test_" in content:
                    test_gaps_code = content
                    break

        if test_gaps_code is None:
            return 0.0, {"reason": "no test file found in response"}

        # Run the model's tests against the correct implementation
        results = run_tests(codebase, test_gaps_code, timeout=10.0)

        # Check which untested paths are now covered by examining test names
        # and test results
        test_results = results.get("results", [])
        test_names = [r.get("name", "") for r in test_results]
        test_text = " ".join(test_names).lower()

        paths_covered = set()
        for path in self._UNTESTED_PATHS:
            path_keywords = {
                "no_uppercase": ["uppercase", "upper", "no_upper"],
                "no_digit": ["digit", "no_digit", "number"],
                "no_special": ["special", "no_special", "symbol"],
            }
            if any(kw in test_text for kw in path_keywords[path]):
                # Also check the test passed
                for r in test_results:
                    if r.get("status") == "pass" and any(
                        kw in r.get("name", "").lower()
                        for kw in path_keywords[path]
                    ):
                        paths_covered.add(path)

        # If keyword matching didn't find enough, check by running tests
        # and seeing how many pass
        n_passed = results.get("passed", 0)
        n_untested = len(self._UNTESTED_PATHS)

        # Score based on fraction of untested paths now covered
        if len(paths_covered) >= n_untested:
            coverage_score = 1.0
        else:
            # Fallback: use fraction of passing tests (expect 3 tests)
            coverage_score = min(1.0, n_passed / n_untested)

        # Also check they didn't just duplicate existing tests
        # by looking for tests that test already-covered paths
        existing_keywords = {
            "empty": ["empty"],
            "too_short": ["short", "too_short"],
            "normal": ["valid", "normal", "good"],
        }
        redundant = 0
        for path, keywords in existing_keywords.items():
            for r in test_results:
                if r.get("status") == "pass" and any(
                    kw in r.get("name", "").lower() for kw in keywords
                ):
                    redundant += 1
                    break

        # Penalize redundancy slightly
        if redundant > 0 and len(paths_covered) < n_untested:
            coverage_score *= 0.8

        return coverage_score, {
            "paths_covered": list(paths_covered),
            "paths_still_missing": [p for p in self._UNTESTED_PATHS if p not in paths_covered],
            "tests_passed": n_passed,
            "tests_total": results.get("total", 0),
            "redundant_tests": redundant,
            "score": coverage_score,
            "results": test_results,
        }
