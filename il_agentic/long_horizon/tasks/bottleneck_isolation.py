"""
Task: bottleneck_isolation

Reasoning skill: Performance reasoning — identifying the specific bottleneck.

The model is given a `report_generator.py` module with three functions:
  - `load_data`   — O(n), fast
  - `process_items` — O(n^2) due to `list.index()` inside a loop (THE BOTTLENECK)
  - `format_output` — O(n), fast

This is a Q&A + code task: the model must identify the bottleneck in its
reasoning (text) and optimize only that function (code).

Failure mode: small models "optimize" everything instead of finding the real
bottleneck, wasting effort and often introducing bugs.
"""
import re
import time
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer, extract_reasoning, parse_code_blocks,
    apply_code_changes, run_tests, compute_test_score, run_code,
)


@register_long_horizon
class BottleneckIsolation(LongHorizonEnv):
    task_id = "bottleneck_isolation"
    reasoning_skill = "Performance reasoning — identifying the specific bottleneck"
    failure_mode = (
        "Small models 'optimize' everything instead of finding the real "
        "bottleneck, wasting effort and often introducing bugs."
    )
    token_budget = 600
    expected_concepts = [
        "bottleneck", "performance", "profile", "O(n)",
        "nested loop", "hash", "lookup", "verify",
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        report_gen = textwrap.dedent('''
            """Report generation module.

            One function has a performance bottleneck.  Identify it, explain
            why it is slow, and optimize ONLY that function.
            """

            def load_data(rows):
                """Load raw rows into a list of dicts.  O(n) — already fast."""
                result = []
                for row in rows:
                    entry = {
                        "id": row[0],
                        "name": row[1],
                        "value": row[2],
                    }
                    result.append(entry)
                return result


            def process_items(items, lookups):
                """For each item, find its matching lookup by id and merge.

                This function is the bottleneck.  It uses list.index() inside
                a loop, making it O(n * m) where n = len(items) and
                m = len(lookups).
                """
                output = []
                lookup_ids = [l["id"] for l in lookups]
                for item in items:
                    # BUG: list.index() is O(m), called n times -> O(n*m)
                    idx = lookup_ids.index(item["id"])
                    merged = dict(item)
                    merged["lookup_value"] = lookups[idx]["value"]
                    output.append(merged)
                return output


            def format_output(items):
                """Format items as a text report.  O(n) — already fast."""
                lines = []
                for item in items:
                    lines.append(f'{item["id"]}: {item["name"]} = {item["value"]}')
                return "\\n".join(lines)


            def generate_report(rows, lookups):
                """End-to-end: load, process, format."""
                items = load_data(rows)
                processed = process_items(items, lookups)
                return format_output(processed)
        ''').strip()

        tests = textwrap.dedent('''
            import time
            import report_generator as rg


            # ── Correctness tests ──

            def test_load_data():
                rows = [(1, "a", 10), (2, "b", 20)]
                data = rg.load_data(rows)
                assert len(data) == 2
                assert data[0] == {"id": 1, "name": "a", "value": 10}

            def test_process_items_basic():
                items = [{"id": 1, "name": "a", "value": 10},
                         {"id": 2, "name": "b", "value": 20}]
                lookups = [{"id": 1, "value": "x"},
                           {"id": 2, "value": "y"}]
                out = rg.process_items(items, lookups)
                assert out[0]["lookup_value"] == "x"
                assert out[1]["lookup_value"] == "y"

            def test_process_items_unordered():
                items = [{"id": 3, "name": "c", "value": 30},
                         {"id": 1, "name": "a", "value": 10}]
                lookups = [{"id": 1, "value": "x"},
                           {"id": 2, "value": "y"},
                           {"id": 3, "value": "z"}]
                out = rg.process_items(items, lookups)
                assert out[0]["lookup_value"] == "z"
                assert out[1]["lookup_value"] == "x"

            def test_format_output():
                items = [{"id": 1, "name": "a", "value": 10}]
                text = rg.format_output(items)
                assert "1: a = 10" in text

            def test_generate_report_end_to_end():
                rows = [(1, "a", 10), (2, "b", 20)]
                lookups = [{"id": 1, "value": "x"},
                           {"id": 2, "value": "y"}]
                report = rg.generate_report(rows, lookups)
                assert "1: a = 10" in report
                assert "2: b = 20" in report

            # ── Performance test ──

            def test_process_items_performance():
                n = 5000
                items = [{"id": i, "name": f"item{i}", "value": i} for i in range(n)]
                lookups = [{"id": i, "value": f"v{i}"} for i in range(n)]
                # Shuffle lookups so index() can't get lucky.
                import random
                random.seed(42)
                random.shuffle(lookups)
                start = time.time()
                out = rg.process_items(items, lookups)
                elapsed = time.time() - start
                assert len(out) == n
                assert elapsed < 1.0, f"too slow: {elapsed:.2f}s"
        ''').strip()

        return {
            "report_generator.py": report_gen,
            "test_bottleneck.py": tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''
            You are given `report_generator.py` with three functions:
            `load_data`, `process_items`, and `format_output`.

            One of these functions is the performance bottleneck.  Your task:

            (A) In your reasoning, identify WHICH function is the bottleneck
                and explain WHY — analyze the time complexity of each function
                and point to the specific operation that causes the slowdown.

            (B) Optimize ONLY the bottleneck function.  Do not change the other
                two functions.  The optimized version must produce identical
                output and run in under 1 second on 5000 items.

            The 5 correctness tests + 1 performance test in
            `test_bottleneck.py` must all pass.

            Return your solution as a code block tagged with the filename:

            ```python:report_generator.py
            ...
            ```
        ''').strip()

    # ── Correct solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        fixed = textwrap.dedent('''
            """Report generation module — bottleneck optimized."""

            def load_data(rows):
                """Load raw rows into a list of dicts.  O(n) — already fast."""
                result = []
                for row in rows:
                    entry = {
                        "id": row[0],
                        "name": row[1],
                        "value": row[2],
                    }
                    result.append(entry)
                return result


            def process_items(items, lookups):
                """For each item, find its matching lookup by id and merge.

                Optimized: build a hash map (dict) from lookup id -> lookup
                so each item is an O(1) lookup instead of O(m) list.index().
                Overall O(n + m) instead of O(n * m).
                """
                lookup_map = {l["id"]: l for l in lookups}
                output = []
                for item in items:
                    matched = lookup_map[item["id"]]
                    merged = dict(item)
                    merged["lookup_value"] = matched["value"]
                    output.append(merged)
                return output


            def format_output(items):
                """Format items as a text report.  O(n) — already fast."""
                lines = []
                for item in items:
                    lines.append(f'{item["id"]}: {item["name"]} = {item["value"]}')
                return "\\n".join(lines)


            def generate_report(rows, lookups):
                """End-to-end: load, process, format."""
                items = load_data(rows)
                processed = process_items(items, lookups)
                return format_output(processed)
        ''').strip()

        return {"report_generator.py": fixed}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''
            Let me analyze the time complexity of each function in
            report_generator.py to find the bottleneck.

            Step 1 — Profile each function's complexity:
            load_data: single loop over `rows`, building a dict per row and
            appending to a list.  Each iteration is O(1) work.  Total: O(n)
            where n is the number of rows.  This is already optimal — you
            cannot load n items faster than O(n).  Not the bottleneck.

            Step 2 — process_items:
            It builds `lookup_ids = [l["id"] for l in lookups]` — O(m).
            Then for each of the n items, it calls `lookup_ids.index(item["id"])`.
            `list.index()` is a linear scan — O(m) per call.  Called n times,
            this is O(n * m).  When n and m are both 5000, that is 25 million
            comparisons.  This is the bottleneck: a nested loop in disguise.
            The `list.index()` call hides the inner loop, but the complexity
            is quadratic.

            Step 3 — format_output:
            Single loop over items, building strings.  O(n).  Already optimal.
            Not the bottleneck.

            Step 4 — Decide the fix:
            The bottleneck is process_items.  The fix is to replace the O(m)
            linear scan with an O(1) hash lookup.  Build a dict mapping
            lookup id -> lookup dict: `lookup_map = {l["id"]: l for l in
            lookups}`.  This is O(m) to build.  Then for each item,
            `lookup_map[item["id"]]` is O(1) average.  Total: O(n + m) instead
            of O(n * m).  For n = m = 5000, that is ~10000 operations instead
            of ~25 million.

            Step 5 — Verify correctness:
            - test_process_items_basic: items with ids 1,2 and lookups with
              ids 1,2.  lookup_map = {1: ..., 2: ...}.  item id 1 -> "x",
              item id 2 -> "y".  Matches original output. OK.
            - test_process_items_unordered: lookups shuffled.  lookup_map
              still maps id -> lookup correctly regardless of order.  item
              id 3 -> "z", item id 1 -> "x". OK.
            - test_generate_report_end_to_end: load_data + process_items +
              format_output.  Output unchanged. OK.

            Step 6 — Verify performance:
            n = 5000, m = 5000, lookups shuffled.  Building lookup_map is
            O(5000).  Processing 5000 items with O(1) lookups is O(5000).
            Total ~10000 operations — well under 1 second.  The original
            O(n*m) = 25M operations would take several seconds.  Let me
            confirm the performance test asserts elapsed < 1.0 — yes.

            To confirm: I only changed process_items.  load_data and
            format_output are untouched.  The bottleneck was the hidden
            nested loop from list.index() inside a loop; the fix uses a hash
            map for O(1) lookup.  I verified both correctness and performance.
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
        test_code = codebase.get("test_bottleneck.py", "")
        results = run_tests(new_codebase, test_code, timeout=30.0)
        test_results = results.get("results", [])

        # Split tests
        correctness_names = {
            "test_load_data", "test_process_items_basic",
            "test_process_items_unordered", "test_format_output",
            "test_generate_report_end_to_end",
        }
        perf_name = "test_process_items_performance"

        correctness_pass = sum(
            1 for r in test_results
            if r["name"] in correctness_names and r["status"] == "pass"
        )
        perf_pass = sum(
            1 for r in test_results
            if r["name"] == perf_name and r["status"] == "pass"
        )
        correctness_score = correctness_pass / 5.0
        performance_score = 1.0 if perf_pass else 0.0

        # Check if reasoning identifies the right function
        reasoning = extract_reasoning(response)
        reasoning_lower = reasoning.lower()
        bottleneck_identified = (
            "process_items" in reasoning_lower
            and ("bottleneck" in reasoning_lower or "o(n" in reasoning_lower
                 or "o(n*m)" in reasoning_lower or "o(n²)" in reasoning_lower
                 or "nested" in reasoning_lower or "list.index" in reasoning_lower
                 or "index" in reasoning_lower)
        )
        bottleneck_score = 1.0 if bottleneck_identified else 0.0

        score = (
            0.4 * correctness_score
            + 0.3 * performance_score
            + 0.3 * bottleneck_score
        )

        breakdown = {
            "total": results.get("total", 0),
            "passed": results.get("passed", 0),
            "correctness_score": correctness_score,
            "performance_score": performance_score,
            "bottleneck_identified": bottleneck_identified,
            "bottleneck_score": bottleneck_score,
            "score": score,
            "results": test_results,
            "method": "0.4*correctness + 0.3*perf + 0.3*bottleneck_id",
        }
        return score, breakdown
