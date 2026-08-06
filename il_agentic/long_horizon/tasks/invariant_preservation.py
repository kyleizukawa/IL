"""
Task: invariant_preservation

Reasoning skill: Mathematical reasoning about code correctness.

A `SortedDict` class maintains keys in sorted order after every operation.
The code is messy (duplicated sorting logic, long methods) and needs
refactoring.  The model must refactor WHILE preserving the sorted invariant.

The invariant: after every operation (insert, delete, update), the internal
keys list remains sorted in ascending order.  The class exposes keys() which
returns the sorted list, and the model must ensure this is always true.

Tests: 10 tests -- 5 behavior tests + 5 invariant tests (check sorted order
after each operation).  If any invariant test fails, the score is capped at
0.3 (refactoring broke correctness).

Failure mode: small models refactor without understanding the invariant,
breaking the sorted-order property.
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer, extract_reasoning, parse_code_blocks,
    apply_code_changes, run_tests, compute_test_score,
)


@register_long_horizon
class InvariantPreservation(LongHorizonEnv):
    task_id = "invariant_preservation"
    reasoning_skill = "Mathematical reasoning about code correctness"
    failure_mode = (
        "Small models refactor without understanding the invariant, "
        "breaking the sorted-order property that the class must maintain."
    )
    token_budget = 700
    expected_concepts = [
        "invariant", "sorted", "monotonic", "preserve",
        "refactor", "verify", "property", "correctness",
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        sorted_dict = textwrap.dedent('''
            """SortedDict — a dictionary that maintains keys in sorted order.

            The invariant: after every operation (insert, delete, update),
            the internal _keys list remains sorted in ascending order.
            keys() always returns the sorted list of keys.

            This code is messy and needs refactoring, but the sorted
            invariant MUST be preserved.  Do not break correctness while
            cleaning up.
            """

            class SortedDict:
                def __init__(self):
                    self._keys = []
                    self._values = {}

                def insert(self, key, value):
                    """Insert a key-value pair, maintaining sorted order."""
                    if key in self._values:
                        raise KeyError(f"key {key} already exists")
                    self._values[key] = value
                    self._keys.append(key)
                    # Duplicated sorting logic — should be centralized.
                    self._keys.sort()

                def delete(self, key):
                    """Delete a key, maintaining sorted order."""
                    if key not in self._values:
                        raise KeyError(f"key {key} not found")
                    del self._values[key]
                    self._keys.remove(key)
                    # Redundant sort after remove — remove preserves order.
                    self._keys.sort()

                def update(self, key, value):
                    """Update an existing key's value (no key change)."""
                    if key not in self._values:
                        raise KeyError(f"key {key} not found")
                    self._values[key] = value
                    # Unnecessary sort — value update does not change key order.
                    self._keys.sort()

                def get(self, key):
                    """Get the value for a key."""
                    return self._values.get(key)

                def keys(self):
                    """Return keys in sorted order."""
                    return list(self._keys)

                def values(self):
                    """Return values in key-sorted order."""
                    return [self._values[k] for k in self._keys]

                def items(self):
                    """Return (key, value) pairs in key-sorted order."""
                    return [(k, self._values[k]) for k in self._keys]

                def __len__(self):
                    return len(self._keys)

                def __contains__(self, key):
                    return key in self._values

                def _is_sorted(self):
                    """Check if _keys is sorted (for invariant testing)."""
                    return all(self._keys[i] <= self._keys[i+1]
                               for i in range(len(self._keys)-1))
        ''').strip()

        tests = textwrap.dedent('''
            from sorted_dict import SortedDict


            # ── Behavior tests ──

            def test_insert_and_get():
                d = SortedDict()
                d.insert(3, "three")
                d.insert(1, "one")
                d.insert(2, "two")
                assert d.get(1) == "one"
                assert d.get(2) == "two"
                assert d.get(3) == "three"

            def test_keys_returned_sorted():
                d = SortedDict()
                d.insert(5, "a")
                d.insert(2, "b")
                d.insert(8, "c")
                d.insert(1, "d")
                assert d.keys() == [1, 2, 5, 8]

            def test_delete_removes_key():
                d = SortedDict()
                d.insert(1, "a")
                d.insert(2, "b")
                d.insert(3, "c")
                d.delete(2)
                assert d.keys() == [1, 3]
                assert d.get(2) is None

            def test_update_changes_value():
                d = SortedDict()
                d.insert(1, "a")
                d.update(1, "b")
                assert d.get(1) == "b"

            def test_items_in_sorted_order():
                d = SortedDict()
                d.insert(3, "c")
                d.insert(1, "a")
                d.insert(2, "b")
                assert d.items() == [(1, "a"), (2, "b"), (3, "c")]

            # ── Invariant tests (sorted order after each operation) ──

            def test_invariant_after_insert():
                d = SortedDict()
                d.insert(5, "a")
                d.insert(2, "b")
                d.insert(8, "c")
                d.insert(1, "d")
                d.insert(3, "e")
                assert d._is_sorted(), "keys not sorted after inserts"
                assert d.keys() == sorted(d.keys())

            def test_invariant_after_delete():
                d = SortedDict()
                for k in [5, 2, 8, 1, 3, 7]:
                    d.insert(k, str(k))
                d.delete(2)
                d.delete(8)
                d.delete(1)
                assert d._is_sorted(), "keys not sorted after deletes"
                assert d.keys() == sorted(d.keys())

            def test_invariant_after_update():
                d = SortedDict()
                for k in [5, 2, 8, 1, 3]:
                    d.insert(k, str(k))
                d.update(5, "new")
                d.update(1, "new")
                d.update(8, "new")
                assert d._is_sorted(), "keys not sorted after updates"
                assert d.keys() == sorted(d.keys())

            def test_invariant_mixed_operations():
                d = SortedDict()
                d.insert(3, "a")
                d.insert(1, "b")
                d.delete(3)
                d.insert(5, "c")
                d.insert(2, "d")
                d.update(1, "e")
                d.delete(5)
                d.insert(4, "f")
                assert d._is_sorted(), "keys not sorted after mixed ops"
                assert d.keys() == sorted(d.keys())

            def test_invariant_with_negative_and_float_keys():
                d = SortedDict()
                d.insert(-5, "a")
                d.insert(3.5, "b")
                d.insert(0, "c")
                d.insert(-2.5, "d")
                d.insert(10, "e")
                d.delete(0)
                assert d._is_sorted(), "keys not sorted with mixed key types"
                assert d.keys() == sorted(d.keys())
        ''').strip()

        return {
            "sorted_dict.py": sorted_dict,
            "test_invariant.py": tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''
            You are given a `SortedDict` class in `sorted_dict.py` that
            maintains keys in sorted order after every operation.

            The code is messy: it has duplicated sorting logic (calls
            `self._keys.sort()` redundantly in delete and update where it
            is not needed), long methods, and no centralized invariant
            enforcement.

            Your task: refactor the code to be cleaner WHILE preserving the
            sorted invariant.  The invariant is:

                After every operation (insert, delete, update), the
                internal _keys list must remain sorted in ascending order.

            You may:
            - Centralize the sorting logic into a helper method.
            - Remove redundant sort calls.
            - Simplify method implementations.
            - Add a `_re_sort` or `_maintain_sorted` helper.

            You must NOT:
            - Break the sorted invariant.
            - Change the public API (method signatures, return values).
            - Remove the _is_sorted method (used by invariant tests).

            All 10 tests in `test_invariant.py` must pass — 5 behavior
            tests and 5 invariant tests.  If any invariant test fails,
            your score will be capped at 0.3.

            Return your solution as a code block tagged with the filename:

            ```python:sorted_dict.py
            ...
            ```
        ''').strip()

    # ── Correct solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        solution = textwrap.dedent('''
            """SortedDict — a dictionary that maintains keys in sorted order.

            Refactored: sorting is centralized in _maintain_sorted, and
            redundant sort calls are removed from delete and update.
            The sorted invariant is preserved after every operation.
            """

            class SortedDict:
                def __init__(self):
                    self._keys = []
                    self._values = {}

                def _maintain_sorted(self):
                    """Centralized helper to enforce the sorted invariant."""
                    self._keys.sort()

                def insert(self, key, value):
                    """Insert a key-value pair, maintaining sorted order."""
                    if key in self._values:
                        raise KeyError(f"key {key} already exists")
                    self._values[key] = value
                    self._keys.append(key)
                    self._maintain_sorted()

                def delete(self, key):
                    """Delete a key.  Order is preserved by list.remove."""
                    if key not in self._values:
                        raise KeyError(f"key {key} not found")
                    del self._values[key]
                    self._keys.remove(key)
                    # No sort needed: removing an element from a sorted
                    # list leaves the remaining elements in sorted order.

                def update(self, key, value):
                    """Update an existing key's value (no key change)."""
                    if key not in self._values:
                        raise KeyError(f"key {key} not found")
                    self._values[key] = value
                    # No sort needed: updating a value does not change key
                    # order, so the invariant is already preserved.

                def get(self, key):
                    """Get the value for a key."""
                    return self._values.get(key)

                def keys(self):
                    """Return keys in sorted order."""
                    return list(self._keys)

                def values(self):
                    """Return values in key-sorted order."""
                    return [self._values[k] for k in self._keys]

                def items(self):
                    """Return (key, value) pairs in key-sorted order."""
                    return [(k, self._values[k]) for k in self._keys]

                def __len__(self):
                    return len(self._keys)

                def __contains__(self, key):
                    return key in self._values

                def _is_sorted(self):
                    """Check if _keys is sorted (for invariant testing)."""
                    return all(self._keys[i] <= self._keys[i+1]
                               for i in range(len(self._keys)-1))
        ''').strip()

        return {"sorted_dict.py": solution}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''
            Let me reason about the invariant and how each method maintains
            it, then plan a refactoring that preserves correctness.

            Step 1 — Identify the invariant:
            The core property is that _keys remains sorted in ascending
            order after every operation.  This is a monotonic property:
            once sorted, the list stays sorted unless an operation breaks
            the order.  I need to verify that each method either preserves
            the sorted order or explicitly re-sorts.

            Step 2 — Trace each method to understand how the invariant is
            maintained:

            - insert: appends a key to _keys, then calls sort().  The
              append breaks sorted order (the new key may be smaller than
              existing ones), but sort() restores it.  The sort is
              necessary here.  Correct.

            - delete: removes a key from _keys via list.remove(), then
              calls sort().  But list.remove() removes one element from a
              sorted list — the remaining elements are still in sorted
              order.  The sort() call is redundant.  It does not break
              correctness, but it is wasted work.

            - update: changes a value in _values, then calls sort().  But
              update does not touch _keys at all — the key order is
              unchanged.  The sort() call is completely unnecessary.  It
              does not break correctness, but it is wasted work.

            Step 3 — Plan the refactoring:
            I will centralize the sorting logic into a _maintain_sorted
            helper method.  This makes the invariant enforcement explicit
            and easy to find.  Then I will:
            - Keep the sort call in insert (it is needed).
            - Remove the redundant sort from delete (list.remove preserves
              order on a sorted list).
            - Remove the unnecessary sort from update (no key change).
            The public API stays the same.  The _is_sorted method stays
            for invariant testing.

            Step 4 — Verify the refactoring preserves the invariant:
            Let me trace through each operation with the refactored code:

            - insert(5, "a") on empty dict: _keys = [5], sorted. OK.
            - insert(2, "b"): _keys = [5, 2] -> _maintain_sorted() ->
              [2, 5]. Sorted. OK.
            - insert(8, "c"): _keys = [2, 5, 8] -> sort -> [2, 5, 8].
              Sorted. OK.
            - insert(1, "d"): _keys = [2, 5, 8, 1] -> sort -> [1, 2, 5, 8].
              Sorted. OK.

            - delete(2): _keys = [1, 2, 5, 8] -> remove(2) -> [1, 5, 8].
              Still sorted (removing from sorted list preserves order).
              No sort needed.  _is_sorted() -> True. OK.

            - update(1, "new"): _keys unchanged = [1, 5, 8].  Still
              sorted.  No sort needed.  _is_sorted() -> True. OK.

            Step 5 — Verify against the invariant tests:
            - test_invariant_after_insert: inserts 5,2,8,1,3 in random
              order.  After each insert, _maintain_sorted() runs.  Final
              _keys = [1,2,3,5,8].  _is_sorted() -> True. OK.
            - test_invariant_after_delete: inserts 5,2,8,1,3,7 then
              deletes 2,8,1.  After deletes: _keys = [3,5,7] (remove
              preserves order).  _is_sorted() -> True. OK.
            - test_invariant_after_update: inserts then updates values.
              _keys never changes during update.  _is_sorted() -> True.
              OK.
            - test_invariant_mixed_operations: mix of insert, delete,
              update.  Each insert sorts, each delete preserves order,
              each update does not touch keys.  Final _keys sorted. OK.
            - test_invariant_with_negative_and_float_keys: inserts
              -5, 3.5, 0, -2.5, 10 then deletes 0.  Python sorts mixed
              numeric types correctly.  _keys after inserts = [-5, -2.5,
              0, 3.5, 10].  After delete(0) = [-5, -2.5, 3.5, 10].
              _is_sorted() -> True. OK.

            Step 6 — Verify behavior tests:
            - test_insert_and_get: values stored correctly. OK.
            - test_keys_returned_sorted: keys() returns sorted list. OK.
            - test_delete_removes_key: key gone after delete. OK.
            - test_update_changes_value: value updated. OK.
            - test_items_in_sorted_order: items in key order. OK.

            To confirm: I identified the sorted invariant as a monotonic
            property that is preserved by delete (remove from sorted list
            stays sorted) and update (no key change), and only needs active
            enforcement in insert.  The refactoring centralizes the sort
            in _maintain_sorted, removes redundant calls, and preserves
            both the invariant and the public API.  I have verified all 10
            tests pass by tracing through each operation and checking the
            sorted property at every step.
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

        # Check that the model actually changed the target file
        target_file = "sorted_dict.py"
        if target_file not in changes:
            return 0.0, {"reason": f"target file {target_file} not modified",
                        "files_changed": list(changes.keys())}

        # Check that the code was actually modified (not just re-submitted unchanged)
        if changes[target_file].strip() == codebase.get(target_file, "").strip():
            return 0.0, {"reason": "code unchanged — no refactoring performed",
                        "files_changed": list(changes.keys())}

        test_code = codebase.get("test_invariant.py", "")
        results = run_tests(new_codebase, test_code, timeout=15.0)
        score, breakdown = compute_test_score(results)

        # Cap score at 0.3 if any invariant test fails.
        invariant_failed = any(
            r.get("status") != "pass" and "invariant" in r.get("name", "")
            for r in results.get("results", [])
        )
        if invariant_failed and score > 0.3:
            score = 0.3
            breakdown["capped"] = True
            breakdown["cap_reason"] = "invariant test failed, score capped at 0.3"

        breakdown["results"] = results.get("results", [])
        breakdown["method"] = "run_tests"
        return score, breakdown
