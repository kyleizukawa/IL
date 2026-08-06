"""
Task: backward_compat_evolution

Reasoning skill: API design reasoning — evolving APIs without breaking callers.

The model is given a `config_manager.py` module with a `ConfigManager` class
whose `get(key)` returns a string and `set(key, value)` takes a string.  The
new requirement is to support typed values (int, float, bool, list) while
keeping the old string-based methods working unchanged for backward
compatibility.

The model must add `get_typed(key, type_hint)` and `set_typed(key, value,
type_hint)` without breaking existing callers.

Failure mode: small models change the existing API without considering
existing callers, breaking backward compatibility.
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer, extract_reasoning, parse_code_blocks,
    apply_code_changes, run_tests, compute_test_score,
)


@register_long_horizon
class BackwardCompatEvolution(LongHorizonEnv):
    task_id = "backward_compat_evolution"
    reasoning_skill = "API design reasoning — evolving APIs without breaking callers"
    failure_mode = (
        "Small models change the existing API without considering existing "
        "callers, breaking backward compatibility."
    )
    token_budget = 700
    expected_concepts = [
        "backward", "compatible", "deprecate", "default",
        "caller", "break", "migrate", "verify",
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        config_manager = textwrap.dedent('''
            """Configuration manager — string-only key/value store.

            Existing callers rely on:
              - get(key) -> str
              - set(key, value) where value is a str

            New requirement: support typed values (int, float, bool, list)
            WITHOUT breaking the existing string API.
            """

            class ConfigManager:
                def __init__(self):
                    self._data: dict[str, str] = {}

                def get(self, key: str) -> str:
                    """Get a config value as a string.

                    Returns "" if the key does not exist.
                    """
                    return self._data.get(key, "")

                def set(self, key: str, value: str) -> None:
                    """Set a config value (must be a string)."""
                    if not isinstance(value, str):
                        raise TypeError("value must be a string")
                    self._data[key] = value

                def keys(self) -> list[str]:
                    """Return all config keys."""
                    return list(self._data.keys())

                def delete(self, key: str) -> None:
                    """Delete a config key."""
                    self._data.pop(key, None)
        ''').strip()

        tests = textwrap.dedent('''
            from config_manager import ConfigManager


            # ── Old API tests (must still pass unchanged) ──

            def test_old_get_returns_string():
                cm = ConfigManager()
                cm.set("name", "alice")
                val = cm.get("name")
                assert val == "alice"
                assert isinstance(val, str)

            def test_old_get_missing_returns_empty_string():
                cm = ConfigManager()
                assert cm.get("nope") == ""
                assert isinstance(cm.get("nope"), str)

            def test_old_set_string():
                cm = ConfigManager()
                cm.set("greeting", "hello")
                assert cm.get("greeting") == "hello"

            def test_old_set_non_string_raises():
                cm = ConfigManager()
                try:
                    cm.set("count", 42)
                    assert False, "should raise TypeError"
                except TypeError:
                    pass

            def test_old_keys_and_delete():
                cm = ConfigManager()
                cm.set("a", "1")
                cm.set("b", "2")
                assert set(cm.keys()) == {"a", "b"}
                cm.delete("a")
                assert "a" not in cm.keys()

            # ── New API tests ──

            def test_new_set_typed_int():
                cm = ConfigManager()
                cm.set_typed("count", 42, int)
                assert cm.get_typed("count", int) == 42
                assert isinstance(cm.get_typed("count", int), int)

            def test_new_set_typed_float():
                cm = ConfigManager()
                cm.set_typed("rate", 3.14, float)
                assert cm.get_typed("rate", float) == 3.14
                assert isinstance(cm.get_typed("rate", float), float)

            def test_new_set_typed_bool():
                cm = ConfigManager()
                cm.set_typed("enabled", True, bool)
                assert cm.get_typed("enabled", bool) is True

            def test_new_set_typed_list():
                cm = ConfigManager()
                cm.set_typed("tags", ["a", "b", "c"], list)
                assert cm.get_typed("tags", list) == ["a", "b", "c"]

            def test_new_get_typed_missing_returns_default():
                cm = ConfigManager()
                assert cm.get_typed("nope", int) == 0
                assert cm.get_typed("nope", list) == []
        ''').strip()

        return {
            "config_manager.py": config_manager,
            "test_backward_compat.py": tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''
            You are given `config_manager.py` with a `ConfigManager` class.

            Current API (existing callers depend on these):
            - `get(key)` returns a string ("" if missing)
            - `set(key, value)` takes a string, raises TypeError for non-strings

            New requirement: support typed values (int, float, bool, list).

            Your task:
            - Add `get_typed(key, type_hint)` that returns the value converted
              to `type_hint`, with a sensible default when the key is missing
              (0 for int, 0.0 for float, False for bool, [] for list).
            - Add `set_typed(key, value, type_hint)` that stores the value and
              its type.
            - The OLD `get`/`set` methods must still work EXACTLY as before.
              Do not change their signatures or behavior.

            All 10 tests in `test_backward_compat.py` must pass: 5 old API
            tests + 5 new API tests.  If any old API test fails, your score is
            capped at 0.3.

            Return your solution as a code block tagged with the filename:

            ```python:config_manager.py
            ...
            ```
        ''').strip()

    # ── Correct solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        evolved = textwrap.dedent('''
            """Configuration manager — string + typed key/value store.

            Backward compatible: get/set still work with strings.
            New: get_typed/set_typed support int, float, bool, list.
            """

            _TYPE_DEFAULTS = {
                int: 0,
                float: 0.0,
                bool: False,
                list: [],
                str: "",
            }


            class ConfigManager:
                def __init__(self):
                    self._data: dict[str, str] = {}
                    self._typed: dict[str, tuple] = {}  # key -> (value, type)

                # ── Old string API (unchanged) ──

                def get(self, key: str) -> str:
                    """Get a config value as a string.

                    Returns "" if the key does not exist.
                    If the key was set via set_typed, return its string
                    representation so old callers still get a str.
                    """
                    if key in self._typed:
                        return str(self._typed[key][0])
                    return self._data.get(key, "")

                def set(self, key: str, value: str) -> None:
                    """Set a config value (must be a string)."""
                    if not isinstance(value, str):
                        raise TypeError("value must be a string")
                    self._data[key] = value
                    # Clear any typed override so get() returns the new string.
                    self._typed.pop(key, None)

                def keys(self) -> list[str]:
                    """Return all config keys."""
                    return list(set(self._data.keys()) | set(self._typed.keys()))

                def delete(self, key: str) -> None:
                    """Delete a config key."""
                    self._data.pop(key, None)
                    self._typed.pop(key, None)

                # ── New typed API ──

                def get_typed(self, key: str, type_hint: type):
                    """Get a config value as the given type.

                    Returns the type's default when the key is missing.
                    """
                    if key in self._typed:
                        val, stored_type = self._typed[key]
                        if stored_type is type_hint:
                            return val
                        # Try converting if a different type was stored.
                        return self._convert(val, type_hint)
                    if key in self._data:
                        return self._convert(self._data[key], type_hint)
                    return _TYPE_DEFAULTS.get(type_hint, None)

                def set_typed(self, key: str, value, type_hint: type) -> None:
                    """Set a typed config value."""
                    if not isinstance(value, type_hint):
                        raise TypeError(
                            f"value must be of type {type_hint.__name__}"
                        )
                    self._typed[key] = (value, type_hint)
                    # Also keep a string version for old get() callers.
                    self._data[key] = str(value)

                @staticmethod
                def _convert(raw, type_hint):
                    """Convert a raw value/string to the target type."""
                    if type_hint is bool:
                        if isinstance(raw, str):
                            return raw.lower() in ("true", "1", "yes")
                        return bool(raw)
                    if type_hint is list:
                        if isinstance(raw, list):
                            return list(raw)
                        if isinstance(raw, str):
                            import json
                            try:
                                parsed = json.loads(raw)
                                return parsed if isinstance(parsed, list) else [raw]
                            except (json.JSONDecodeError, ValueError):
                                return [raw] if raw else []
                        return [raw]
                    if type_hint is int:
                        return int(raw)
                    if type_hint is float:
                        return float(raw)
                    return type_hint(raw)
        ''').strip()

        return {"config_manager.py": evolved}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''
            Let me reason about how to evolve this API without breaking
            existing callers.

            Step 1 — Understand the current contract:
            - `get(key)` returns a str, default "" for missing keys.
            - `set(key, value)` requires value to be a str, raises TypeError
              otherwise.
            - `keys()` and `delete(key)` are auxiliary.

            Existing callers assume get always returns a string and set only
            accepts strings.  Any change that makes get return a non-string,
            or makes set accept non-strings, would break these callers.  I
            must NOT change the signatures or return types of get/set.

            Step 2 — Design the new typed API:
            I will add two new methods:
            - `get_typed(key, type_hint)`: returns the value as type_hint.
              For missing keys, return a sensible default per type
              (0 for int, 0.0 for float, False for bool, [] for list).
            - `set_typed(key, value, type_hint)`: stores the value with its
              type metadata.

            These are additive — they do not touch get/set, so old callers
            are unaffected.  This is the backward-compatible approach: new
            methods alongside old ones, not replacing them.  The old methods
            could be marked as deprecated in documentation, but they must
            keep working so callers can migrate gradually.

            Step 3 — Storage design:
            I need to store typed values separately from string values so
            get() can still return a string.  I will keep:
            - `_data: dict[str, str]` for string values (old API).
            - `_typed: dict[str, tuple(value, type)]` for typed values.
            When set_typed is called, I store the typed value in _typed AND
            a string version in _data, so get() returns str(value) for
            backward compatibility.  When set is called, I store in _data and
            clear any _typed override so get() returns the new string.

            Step 4 — get_typed conversion:
            If a key was set via set_typed, return the stored value if the
            type matches.  If a key was set via set (string only), convert
            the string to the requested type.  For missing keys, return the
            type default.  I need a _convert helper:
            - bool: "true"/"1"/"yes" -> True, else False.
            - list: try json.loads, fall back to [raw] or [].
            - int/float: int(raw)/float(raw).
            This lets a caller set("count", "42") then get_typed("count", int)
            -> 42, which is a nice migration path.

            Step 5 — Verify old API tests:
            - test_old_get_returns_string: set("name","alice"), get("name")
              -> "alice" (str).  _typed is empty, so get returns _data value.
              OK.
            - test_old_get_missing_returns_empty_string: get("nope") -> "".
              OK.
            - test_old_set_string: set("greeting","hello"), get -> "hello".
              OK.
            - test_old_set_non_string_raises: set("count", 42) -> TypeError.
              I kept the isinstance check. OK.
            - test_old_keys_and_delete: keys() now unions _data and _typed
              keys.  After set("a","1") and set("b","2"), keys = {"a","b"}.
              delete("a") removes from both. OK.

            Step 6 — Verify new API tests:
            - test_new_set_typed_int: set_typed("count",42,int), get_typed
              -> 42 (int). OK.
            - test_new_set_typed_float: set_typed("rate",3.14,float) -> 3.14.
              OK.
            - test_new_set_typed_bool: set_typed("enabled",True,bool) -> True.
              OK.
            - test_new_set_typed_list: set_typed("tags",["a","b","c"],list)
              -> ["a","b","c"]. OK.
            - test_new_get_typed_missing_returns_default: get_typed("nope",int)
              -> 0, get_typed("nope",list) -> []. OK.

            Step 7 — Check backward compat edge case:
            If a caller does set_typed("count", 42, int) then get("count"),
              get returns str(42) = "42" — a string, as old callers expect.
            If a caller does set("count", "42") then get_typed("count", int),
              _convert("42", int) -> 42.  This migration path works: old
              callers can migrate to typed access without changing their
              set() calls first.

            To confirm: the old get/set API is completely unchanged in
            signature and behavior.  The new get_typed/set_typed are purely
            additive.  I verified all 10 tests pass — 5 old + 5 new.  No
            existing caller breaks.
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
        test_code = codebase.get("test_backward_compat.py", "")
        results = run_tests(new_codebase, test_code, timeout=15.0)
        test_results = results.get("results", [])

        old_api_names = {
            "test_old_get_returns_string",
            "test_old_get_missing_returns_empty_string",
            "test_old_set_string",
            "test_old_set_non_string_raises",
            "test_old_keys_and_delete",
        }
        new_api_names = {
            "test_new_set_typed_int",
            "test_new_set_typed_float",
            "test_new_set_typed_bool",
            "test_new_set_typed_list",
            "test_new_get_typed_missing_returns_default",
        }

        old_pass = sum(
            1 for r in test_results
            if r["name"] in old_api_names and r["status"] == "pass"
        )
        new_pass = sum(
            1 for r in test_results
            if r["name"] in new_api_names and r["status"] == "pass"
        )
        old_score = old_pass / 5.0
        new_score = new_pass / 5.0
        raw_score = (old_score + new_score) / 2.0

        # Cap at 0.3 if any old API test fails
        if old_score < 1.0:
            score = min(0.3, raw_score)
        else:
            score = raw_score

        breakdown = {
            "total": results.get("total", 0),
            "passed": results.get("passed", 0),
            "old_api_score": old_score,
            "new_api_score": new_score,
            "old_api_capped": old_score < 1.0,
            "score": score,
            "results": test_results,
            "method": "avg(old,new), capped at 0.3 if old < 1.0",
        }
        return score, breakdown
