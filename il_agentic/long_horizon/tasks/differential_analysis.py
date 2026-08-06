"""
Differential Analysis — Comparative reasoning task.

Two nearly-identical quicksort implementations differ in only 2-3 lines.
One is correct, one has a subtle bug: the partition uses `<=` instead of
`<`, causing infinite recursion on arrays with many duplicate elements.

The model must compare the two files, identify the CRITICAL difference
(not just any difference), and fix the buggy version.

Failure mode: small models identify surface differences but cannot trace
which one actually causes the bug, leading to incorrect or unnecessary fixes.
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    parse_code_blocks, apply_code_changes, run_tests,
    compute_test_score, code_similarity,
)


@register_long_horizon
class DifferentialAnalysis(LongHorizonEnv):
    task_id = "differential_analysis"
    reasoning_skill = "Comparative reasoning — finding the critical difference between similar code"
    failure_mode = "Small models can't identify which difference matters among many"
    token_budget = 600
    expected_concepts = ["compare", "difference", "correct", "buggy", "critical", "subtle", "trace", "verify"]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        sort_correct = textwrap.dedent("""\
            \"\"\"Quicksort implementation — correct version.\"\"\"


            def partition(arr, low, high):
                \"\"\"Partition the array around a pivot.

                Elements smaller than the pivot go to the left,
                elements greater than or equal to the pivot go to the right.
                \"\"\"
                pivot = arr[high]
                i = low - 1
                for j in range(low, high):
                    if arr[j] < pivot:
                        i += 1
                        arr[i], arr[j] = arr[j], arr[i]
                arr[i + 1], arr[high] = arr[high], arr[i + 1]
                return i + 1


            def quicksort(arr, low, high):
                \"\"\"Sort arr[low:high+1] in place using quicksort.\"\"\"
                if low < high:
                    pi = partition(arr, low, high)
                    quicksort(arr, low, pi - 1)
                    quicksort(arr, pi + 1, high)


            def sort_list(data):
                \"\"\"Return a sorted copy of the input list.\"\"\"
                result = list(data)
                if len(result) <= 1:
                    return result
                quicksort(result, 0, len(result) - 1)
                return result
            """)

        sort_buggy = textwrap.dedent("""\
            \"\"\"Quicksort implementation — buggy version.\"\"\"


            def partition(arr, low, high):
                \"\"\"Partition the array around a pivot.

                Elements smaller than the pivot go to the left,
                elements greater than or equal to the pivot go to the right.
                \"\"\"
                pivot = arr[high]
                i = low - 1
                for j in range(low, high):
                    if arr[j] <= pivot:
                        i += 1
                        arr[i], arr[j] = arr[j], arr[i]
                arr[i + 1], arr[high] = arr[high], arr[i + 1]
                return i + 1


            def quicksort(arr, low, high):
                \"\"\"Sort arr[low:high+1] in place using quicksort.\"\"\"
                if low < high:
                    pi = partition(arr, low, high)
                    quicksort(arr, low, pi - 1)
                    quicksort(arr, pi + 1, high)


            def sort_list(data):
                \"\"\"Return a sorted copy of the input list.\"\"\"
                result = list(data)
                if len(result) <= 1:
                    return result
                quicksort(result, 0, len(result) - 1)
                return result
            """)

        test_file = textwrap.dedent("""\
            \"\"\"Tests for sort_list — includes duplicate-heavy cases.\"\"\"
            from sort_buggy import sort_list


            def test_basic_sort():
                assert sort_list([3, 1, 2]) == [1, 2, 3]


            def test_empty_list():
                assert sort_list([]) == []


            def test_single_element():
                assert sort_list([42]) == [42]


            def test_already_sorted():
                assert sort_list([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


            def test_all_duplicates():
                result = sort_list([5, 5, 5, 5, 5, 5, 5, 5])
                assert result == [5, 5, 5, 5, 5, 5, 5, 5]


            def test_many_duplicates():
                result = sort_list([3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2])
                assert result == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
            """)

        return {
            "sort_correct.py": sort_correct,
            "sort_buggy.py": sort_buggy,
            "test_sort.py": test_file,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent(f"""\
            You are given two quicksort implementations: `sort_correct.py` and `sort_buggy.py`.
            They are nearly identical but one has a subtle bug.

            The buggy version fails on certain inputs (the tests in `test_sort.py` will
            reveal which ones). Your task:

            1. Compare the two files line by line to find ALL differences.
            2. Trace each difference to determine which is the CRITICAL one that causes the bug.
            3. Fix `sort_buggy.py` by changing only the critical line(s).

            Do NOT change `sort_correct.py` — it is reference code.
            Do NOT rewrite `sort_buggy.py` from scratch — make the minimal fix.

            Provide your fixed `sort_buggy.py` in a code block.
            """)

    # ── Solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        fixed = codebase["sort_buggy.py"].replace(
            "if arr[j] <= pivot:",
            "if arr[j] < pivot:",
        )
        return {"sort_buggy.py": fixed}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent("""\
            I need to compare the correct and buggy quicksort implementations to find
            the critical difference. Let me trace through both files line by line.

            Comparing the two files, they are ~90% identical. Both define partition,
            quicksort, and sort_list with the same structure. Let me find the
            differences:

            sort_correct.py partition:
                if arr[j] < pivot:

            sort_buggy.py partition:
                if arr[j] <= pivot:

            This is the only difference: `<` vs `<=` in the partition comparison.
            The quicksort function, sort_list, pivot assignment, and swap logic are
            all identical. So there is exactly one line that differs.

            Now I need to verify this is the critical difference. Let me trace the
            buggy version with `<=` on an array with many duplicates.

            Consider arr = [5,5,5,5,5,5,5,5], low=0, high=7.
            - pivot = arr[7] = 5
            - Every element arr[j] = 5, and 5 <= 5 is True, so i increments for
              every j: i goes from -1 to 6.
            - Return i+1 = 7 (equals high)

            quicksort recurses: left = quicksort(arr, 0, 6), right = quicksort(arr, 8, 7)
            [skipped]. The left partition is [0:6] — only 1 element smaller than the
            original. This pattern repeats: each recursive call shrinks the range by
            just 1. With `<=`, all elements equal to the pivot go left, making the
            left partition nearly the full array. The right partition is always empty.

            With the correct `<` comparison: arr[j] < pivot is False for all elements
            (5 < 5 is False), so no swaps occur. i stays at -1. Return 0. The left
            partition is empty, right is [1:7]. This is also O(n) depth for
            all-duplicates, but the key difference is that `<` produces correct
            output while `<=` can cause issues with specific duplicate patterns.

            The subtle bug: with `<=`, when the array has many duplicates of the
            pivot value, the partition function returns high (the last index),
            causing the left subarray to be [low, high-1] and the right to be empty.
            This leads to O(n) recursion depth. For large arrays with many
            duplicates, this exceeds Python's recursion limit, causing a
            RecursionError. The tests with all-duplicates and many-duplicates
            arrays are designed to expose this.

            This is the critical difference. The fix is to change `<=` to `<` in
            sort_buggy.py's partition function — making it match the correct version.

            Let me verify the fix: changing `if arr[j] <= pivot:` to
            `if arr[j] < pivot:` makes the buggy version identical to the correct
            version, which passes all 6 tests including test_all_duplicates and
            test_many_duplicates. The fix is correct and minimal.
            """)

    # ── Grader ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        changes = parse_code_blocks(response)
        if not changes:
            return 0.0, {"reason": "no code blocks found in response"}

        # Check that the model actually changed the target file
        if "sort_buggy.py" not in changes:
            return 0.0, {"reason": "sort_buggy.py not modified — must fix the buggy file",
                        "files_changed": list(changes.keys())}

        # Check that the code was actually modified (not just re-submitted unchanged)
        if changes["sort_buggy.py"].strip() == codebase.get("sort_buggy.py", "").strip():
            return 0.0, {"reason": "sort_buggy.py unchanged — no fix applied"}

        new_codebase = apply_code_changes(codebase, changes)

        if "sort_buggy.py" not in new_codebase:
            return 0.0, {"reason": "sort_buggy.py not in response"}

        test_code = codebase["test_sort.py"]
        results = run_tests(new_codebase, test_code, timeout=15.0)
        test_score, test_breakdown = compute_test_score(results)

        # Check similarity to the original buggy code — penalize complete rewrites
        orig_sim = code_similarity(
            new_codebase.get("sort_buggy.py", ""),
            codebase["sort_buggy.py"],
        )
        # If the solution is too different from the original (complete rewrite),
        # apply a penalty. The task requires a minimal fix, not a rewrite.
        if orig_sim < 0.5:
            # Heavy penalty for rewriting from scratch
            score = test_score * 0.3
        elif orig_sim < 0.8:
            # Moderate penalty for significant changes
            score = test_score * 0.7
        else:
            score = test_score

        sol_sim = code_similarity(
            new_codebase.get("sort_buggy.py", ""),
            self.gen_solution(codebase)["sort_buggy.py"],
        )

        return score, {
            "test_score": test_score,
            "original_similarity": orig_sim,
            "solution_similarity": sol_sim,
            "test_breakdown": test_breakdown,
            "results": results.get("results", []),
        }
