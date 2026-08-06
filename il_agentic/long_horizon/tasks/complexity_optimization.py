"""
Task: complexity_optimization

Reasoning skill: Algorithmic complexity reasoning.

A `find_matches` function matches items from two lists by a key function.
The current implementation is O(n*m) — a nested loop comparing each pair.
The optimal implementation is O(n+m) using a dict lookup.

The code also has a distractor: an unrelated O(n log n) sort that looks slow
but is NOT the bottleneck.  Small models may "optimize" the sort instead of
the actual bottleneck.

Tests: 6 correctness tests + 1 performance test (must complete within a time
limit on 10000 items).

Score = 0.5 * correctness + 0.5 * speedup_score, where speedup_score is 1.0
if the performance test passes and 0.0 if it times out.

Failure mode: small models "optimize" without understanding why the code is
slow, targeting the wrong operation (the sort distractor).
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer, extract_reasoning, parse_code_blocks,
    apply_code_changes, run_tests, compute_test_score,
)


@register_long_horizon
class ComplexityOptimization(LongHorizonEnv):
    task_id = "complexity_optimization"
    reasoning_skill = "Algorithmic complexity reasoning"
    failure_mode = (
        "Small models optimize without understanding why code is slow, "
        "targeting a distractor operation instead of the actual bottleneck."
    )
    token_budget = 700
    expected_concepts = [
        "complexity", "O(n)", "O(n^2)", "nested loop",
        "bottleneck", "hash map", "lookup", "verify",
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        matcher = textwrap.dedent('''
            """Matcher — finds matching items between two lists by a key.

            The current implementation is correct but slow.  Your task is
            to optimize it without changing its behavior.
            """

            def find_matches(list_a, list_b, key_func):
                """Find items in list_a that have a matching key in list_b.

                Returns a list of (item_a, item_b) pairs where
                key_func(item_a) == key_func(item_b).

                Current implementation: O(n*m) nested loop.
                This is the bottleneck for large inputs.
                """
                # Distractor: this sort looks expensive but is O(n log n),
                # which is NOT the bottleneck.  The nested loop below is.
                sorted_b = sorted(list_b, key=key_func)

                matches = []
                for item_a in list_a:
                    key_a = key_func(item_a)
                    for item_b in sorted_b:
                        if key_func(item_b) == key_a:
                            matches.append((item_a, item_b))
                            break  # only first match per item_a
                return matches


            def find_matches_by_id(items, catalog):
                """Convenience wrapper: match by 'id' field."""
                return find_matches(items, catalog, lambda x: x["id"])
        ''').strip()

        tests = textwrap.dedent('''
            import time
            from matcher import find_matches, find_matches_by_id


            # ── Correctness tests ──

            def test_basic_match():
                a = [{"id": 1}, {"id": 2}, {"id": 3}]
                b = [{"id": 2, "name": "b"}, {"id": 3, "name": "c"}]
                result = find_matches(a, b, lambda x: x["id"])
                assert len(result) == 2
                assert result[0] == ({"id": 2}, {"id": 2, "name": "b"})
                assert result[1] == ({"id": 3}, {"id": 3, "name": "c"})

            def test_no_matches():
                a = [{"id": 1}, {"id": 2}]
                b = [{"id": 3}, {"id": 4}]
                result = find_matches(a, b, lambda x: x["id"])
                assert len(result) == 0

            def test_first_match_only():
                a = [{"id": 1}]
                b = [{"id": 1, "v": 1}, {"id": 1, "v": 2}]
                result = find_matches(a, b, lambda x: x["id"])
                assert len(result) == 1
                assert result[0][1]["v"] == 1

            def test_empty_lists():
                result = find_matches([], [], lambda x: x)
                assert result == []

            def test_string_keys():
                a = ["apple", "banana", "cherry"]
                b = ["banana", "date", "cherry"]
                result = find_matches(a, b, lambda x: x)
                assert len(result) == 2
                assert result[0] == ("banana", "banana")
                assert result[1] == ("cherry", "cherry")

            def test_wrapper_by_id():
                items = [{"id": 10}, {"id": 20}]
                catalog = [{"id": 10, "price": 5}, {"id": 20, "price": 10}]
                result = find_matches_by_id(items, catalog)
                assert len(result) == 2
                assert result[0][1]["price"] == 5

            def test_partial_overlap():
                a = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
                b = [{"id": 2, "x": 1}, {"id": 4, "x": 2}]
                result = find_matches(a, b, lambda x: x["id"])
                assert len(result) == 2
                assert result[0][0]["id"] == 2
                assert result[1][0]["id"] == 4

            def test_duplicate_keys_in_a():
                a = [{"id": 1}, {"id": 1}, {"id": 2}]
                b = [{"id": 1, "val": "first"}]
                result = find_matches(a, b, lambda x: x["id"])
                assert len(result) == 2
                assert result[0][1]["val"] == "first"
                assert result[1][1]["val"] == "first"

            # ── Performance test ──

            def test_performance_large_input():
                n = 10000
                a = [{"id": i} for i in range(n)]
                b = [{"id": i, "data": i * 2} for i in range(n)]
                start = time.time()
                result = find_matches(a, b, lambda x: x["id"])
                elapsed = time.time() - start
                assert len(result) == n, f"expected {n} matches, got {len(result)}"
                assert elapsed < 2.0, f"too slow: {elapsed:.2f}s (must be < 2.0s)"
        ''').strip()

        return {
            "matcher.py": matcher,
            "test_complexity.py": tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''
            You are given a `find_matches` function in `matcher.py` that
            finds matching items between two lists by a key function.

            The function is correct but slow — it uses a nested loop that
            is O(n*m) in time complexity.  Your task is to optimize it to
            O(n+m) using a hash map (dict) for lookups, WITHOUT changing
            its behavior.

            Note: the code also has a `sorted()` call that may look like a
            performance issue, but it is O(n log n) and is NOT the
            bottleneck.  Focus on the actual bottleneck.

            All 9 tests in `test_complexity.py` must pass -- 8 correctness
            tests and 1 performance test.  The performance test runs on
            10000 items and must complete in under 2 seconds.

            Score = 0.5 * correctness + 0.5 * speedup_score, where
            speedup_score is 1.0 if the performance test passes.

            Return your solution as a code block tagged with the filename:

            ```python:matcher.py
            ...
            ```
        ''').strip()

    # ── Correct solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        solution = textwrap.dedent('''
            """Matcher — finds matching items between two lists by a key.

            Optimized: uses a dict (hash map) for O(n+m) lookups instead
            of the O(n*m) nested loop.
            """

            def find_matches(list_a, list_b, key_func):
                """Find items in list_a that have a matching key in list_b.

                Returns a list of (item_a, item_b) pairs where
                key_func(item_a) == key_func(item_b).

                Optimized: builds a dict from list_b keys for O(1) lookup,
                giving O(n+m) total complexity.
                """
                # Build a hash map from list_b: key -> first matching item.
                # This is O(m) time and O(m) space.
                key_to_item_b = {}
                for item_b in list_b:
                    key_b = key_func(item_b)
                    if key_b not in key_to_item_b:
                        key_to_item_b[key_b] = item_b

                # Look up each item_a's key in the hash map: O(n) time.
                matches = []
                for item_a in list_a:
                    key_a = key_func(item_a)
                    if key_a in key_to_item_b:
                        matches.append((item_a, key_to_item_b[key_a]))
                return matches


            def find_matches_by_id(items, catalog):
                """Convenience wrapper: match by 'id' field."""
                return find_matches(items, catalog, lambda x: x["id"])
        ''').strip()

        return {"matcher.py": solution}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''
            Let me analyze the algorithmic complexity of the current code
            to identify the true bottleneck before optimizing.

            Step 1 — Analyze the current implementation:
            I read `find_matches` and see two operations:
            1. `sorted_b = sorted(list_b, key=key_func)` — this is O(m log m)
               where m = len(list_b).  It sorts list_b by the key function.
            2. The nested loop: for each item_a in list_a (n iterations),
               for each item_b in sorted_b (up to m iterations), compare
               keys.  This is O(n*m) in the worst case.

            The total complexity is O(m log m + n*m).  The dominant term
            is O(n*m) — the nested loop.  The sort is O(m log m) which is
            strictly smaller than O(n*m) when n and m are both large.

            Step 2 — Identify the bottleneck:
            The nested loop is the bottleneck.  For n = m = 10000, the
            nested loop does up to 100,000,000 iterations (100 million),
            while the sort does about 10000 * 14 = 140,000 operations.
            The sort is roughly 700x faster than the nested loop.  The
            sort is a distractor — it looks like it could be slow, but it
            is not the bottleneck.  Optimizing the sort would not help
            meaningfully.

            Step 3 — Plan the optimization:
            I will replace the nested loop with a hash map (dict) lookup.
            The idea: build a dict mapping each key from list_b to the
            first item with that key.  Then for each item_a, look up its
            key in the dict — O(1) per lookup.

            Building the dict: O(m) time (iterate list_b once, compute
            key, store in dict).
            Looking up all items_a: O(n) time (iterate list_a once, compute
            key, dict lookup).
            Total: O(n + m) — linear instead of quadratic.

            The dict uses O(m) extra space, which is acceptable.

            Step 4 — Preserve the "first match only" behavior:
            The original code breaks after finding the first match for
            each item_a.  I must preserve this.  In the dict, I store only
            the first item_b for each key: `if key_b not in
            key_to_item_b: key_to_item_b[key_b] = item_b`.  This ensures
            the dict maps each key to the first item_b with that key,
            matching the original behavior.

            Step 5 — Remove the distractor sort:
            The sorted_b variable is no longer used in the optimized
            version.  I remove the sort entirely.  This also removes the
            O(m log m) term, but that was never the bottleneck.

            Step 6 — Verify correctness by tracing through tests:
            - test_basic_match: a=[{1},{2},{3}], b=[{2},{3}].  Dict:
              {2: {2,"b"}, 3: {3,"c"}}.  Lookups: 1->miss, 2->hit, 3->hit.
              Result: [({2},{2,"b"}), ({3},{3,"c"})].  2 matches. OK.
            - test_no_matches: a=[{1},{2}], b=[{3},{4}].  Dict: {3:..,4:..}.
              Lookups: 1->miss, 2->miss.  Result: []. OK.
            - test_first_match_only: a=[{1}], b=[{1,v:1},{1,v:2}].  Dict:
              {1: {1,v:1}} (first only).  Lookup: 1->hit.  Result:
              [({1},{1,v:1})].  v=1. OK.
            - test_empty_lists: a=[], b=[].  Dict: {}.  No lookups.
              Result: []. OK.
            - test_string_keys: a=["apple","banana","cherry"],
              b=["banana","date","cherry"].  Dict: {"banana":"banana",
              "date":"date", "cherry":"cherry"}.  Lookups: "apple"->miss,
              "banana"->hit, "cherry"->hit.  Result: [("banana",
              "banana"), ("cherry","cherry")].  2 matches. OK.
            - test_wrapper_by_id: items=[{10},{20}],
              catalog=[{10,price:5},{20,price:10}].  Dict: {10:{10,5},
              20:{20,10}}.  Lookups: 10->hit, 20->hit.  Result: 2 matches,
              first price=5. OK.
            - test_partial_overlap: a=[{1},{2},{3},{4}],
              b=[{2,x:1},{4,x:2}].  Dict: {2:{2,x:1}, 4:{4,x:2}}.
              Lookups: 1->miss, 2->hit, 3->miss, 4->hit.  Result: 2
              matches, ids 2 and 4. OK.
            - test_duplicate_keys_in_a: a=[{1},{1},{2}],
              b=[{1,val:"first"}].  Dict: {1:{1,val:"first"}}.  Lookups:
              1->hit, 1->hit, 2->miss.  Result: 2 matches, both with
              val="first".  The dict stores the first item_b per key, and
              both item_a entries with id=1 match it. OK.

            Step 7 — Verify performance:
            With n = m = 10000:
            - Building dict: 10000 iterations, each O(1) -> 10000 ops.
            - Lookups: 10000 iterations, each O(1) -> 10000 ops.
            - Total: ~20000 ops, vs 100,000,000 for the nested loop.
            This is a 5000x speedup.  The performance test requires < 2.0s.
            At ~20000 operations, this completes in milliseconds. OK.

            Step 8 — Analyze space complexity tradeoff:
            The original code uses O(m) space for sorted_b (a copy of
            list_b) plus O(min(n,m)) for the matches list.  The optimized
            code uses O(m) space for the dict plus O(min(n,m)) for matches.
            The space complexity is the same — O(m) — but the dict provides
            O(1) lookup vs the list's O(m) scan.  This is a classic
            time-space tradeoff where we use a hash map to convert a
            linear search into a constant-time lookup.  The extra space
            is justified by the massive time improvement (5000x for
            n=m=10000).

            Step 9 — Consider edge cases in the hash map:
            - Unhashable keys: if key_func returns an unhashable type
              (e.g., a list), the dict insertion will raise TypeError.
              The original code would also fail (comparing unhashable
              types with ==).  So the behavior is equivalent.
            - None keys: if key_func returns None, the dict handles it
              fine (None is a valid dict key).  The original code also
              handles None via == comparison.  Equivalent.
            - Duplicate keys in list_b: the dict stores only the first
              item_b per key, matching the original break-on-first-match
              behavior.  Verified in test_duplicate_keys_in_a.

            To confirm: the bottleneck was the O(n*m) nested loop, not the
            O(n log n) sort distractor.  By replacing the nested loop with
            a hash map lookup, I reduced the complexity from O(n*m) to
            O(n+m) while preserving the first-match-only behavior.  I have
            verified all 8 correctness tests and the performance test pass
            by tracing through the algorithm and analyzing the operation
            count.
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
        test_code = codebase.get("test_complexity.py", "")
        results = run_tests(new_codebase, test_code, timeout=15.0)

        # Separate correctness and performance results.
        result_list = results.get("results", [])
        correctness_tests = [r for r in result_list
                             if not r.get("name", "").startswith("test_performance")]
        perf_tests = [r for r in result_list
                      if r.get("name", "").startswith("test_performance")]

        n_correct = sum(1 for r in correctness_tests if r.get("status") == "pass")
        n_correct_total = len(correctness_tests)
        correctness = n_correct / n_correct_total if n_correct_total > 0 else 0.0

        # Performance test: pass if status is "pass", fail otherwise.
        speedup_score = 1.0 if any(r.get("status") == "pass" for r in perf_tests) else 0.0

        # If timed out, the performance test won't be in results at all.
        if results.get("timed_out"):
            speedup_score = 0.0

        score = 0.5 * correctness + 0.5 * speedup_score

        breakdown = {
            "total": results.get("total", 0),
            "passed": results.get("passed", 0),
            "correctness": correctness,
            "speedup_score": speedup_score,
            "score": score,
            "results": result_list,
            "method": "weighted_correctness_performance",
        }
        return score, breakdown
