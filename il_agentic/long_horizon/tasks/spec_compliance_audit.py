"""
Spec Compliance Audit — Spec reading + code reading task.

A specification document defines 8 requirements for a cache module.
The implementation deviates from the spec in 5 specific ways. The model
must find ALL 5 deviations and list them.

This is a Q&A task — the model lists deviations in text, not code.

Failure mode: small models find 1-2 obvious deviations but miss subtle ones,
demonstrating incomplete spec-to-code comparison.
"""
import re
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer as grader_extract_answer,
    extract_reasoning as grader_extract_reasoning,
)


@register_long_horizon
class SpecComplianceAudit(LongHorizonEnv):
    task_id = "spec_compliance_audit"
    reasoning_skill = "Spec reading + code reading — finding all deviations"
    failure_mode = "Small models find 1-2 deviations but miss the rest"
    token_budget = 800
    expected_concepts = ["spec", "deviation", "compliance", "requirement", "missing", "wrong", "extra", "verify"]

    # The 5 real deviations (for grading)
    _DEVIATIONS = [
        ("max_size", "Spec says max_size=100, code uses max_size=50"),
        ("eviction", "Spec says eviction is LRU, code uses FIFO"),
        ("get_missing", "Spec says get() returns None for missing, code raises KeyError"),
        ("clear", "Spec says clear() empties the cache, code only removes expired entries"),
        ("thread_safe", "Spec says thread-safe, code has no locking"),
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        spec = textwrap.dedent("""\
            CACHE MODULE SPECIFICATION
            ==========================

            Requirements for cache.py:

            1. MAX_SIZE: The cache must have a maximum capacity of 100 entries.
               When the cache is full and a new entry is added, the least
               recently used entry must be evicted.

            2. EVICTION POLICY: Eviction must follow LRU (Least Recently Used)
               order. The entry that was accessed longest ago is evicted first.

            3. GET BEHAVIOR: The get(key) method must return the cached value
               if the key exists, or None if the key does not exist. It must
               NOT raise an exception for missing keys.

            4. CLEAR BEHAVIOR: The clear() method must remove ALL entries from
               the cache, resetting it to an empty state.

            5. THREAD SAFETY: The cache must be thread-safe. All public methods
               must use a lock to prevent concurrent modification issues.

            6. TTL SUPPORT: The cache must support optional time-to-live (TTL)
               per entry. If a TTL is set, entries expire after that many seconds.

            7. SIZE METHOD: The size() method must return the number of valid
               (non-expired) entries currently in the cache.

            8. CONTAINS: The `key in cache` operation must return True only for
               valid (non-expired) entries.
            """)

        cache_impl = textwrap.dedent("""\
            \"\"\"Cache module — implementation (may deviate from spec).\"\"\"
            import time
            from collections import OrderedDict


            class Cache:
                \"\"\"A simple cache with TTL support.\"\"\"

                def __init__(self, max_size=50, default_ttl=None):
                    self._data = OrderedDict()
                    self._max_size = max_size
                    self._default_ttl = default_ttl
                    self._timestamps = {}

                def get(self, key):
                    \"\"\"Get value by key. Raises KeyError if not found.\"\"\"
                    if key not in self._data:
                        raise KeyError(f"Key not found: {key}")
                    if self._is_expired(key):
                        del self._data[key]
                        del self._timestamps[key]
                        raise KeyError(f"Key expired: {key}")
                    self._data.move_to_end(key)
                    return self._data[key]

                def set(self, key, value, ttl=None):
                    \"\"\"Set a key-value pair with optional TTL.\"\"\"
                    if key in self._data:
                        self._data.move_to_end(key)
                    self._data[key] = value
                    self._timestamps[key] = time.time()
                    effective_ttl = ttl if ttl is not None else self._default_ttl
                    if effective_ttl is not None:
                        self._timestamps[key] = (time.time(), effective_ttl)
                    if len(self._data) > self._max_size:
                        # FIFO eviction: remove the first inserted item
                        oldest = next(iter(self._data))
                        del self._data[oldest]
                        del self._timestamps[oldest]

                def clear(self):
                    \"\"\"Remove expired entries from the cache.\"\"\"
                    expired = [k for k in self._data if self._is_expired(k)]
                    for k in expired:
                        del self._data[k]
                        del self._timestamps[k]

                def size(self):
                    \"\"\"Return number of valid (non-expired) entries.\"\"\"
                    self._purge_expired()
                    return len(self._data)

                def __contains__(self, key):
                    \"\"\"Check if key exists and is not expired.\"\"\"
                    if key not in self._data:
                        return False
                    if self._is_expired(key):
                        del self._data[key]
                        del self._timestamps[key]
                        return False
                    return True

                def _is_expired(self, key):
                    \"\"\"Check if an entry has expired.\"\"\"
                    ts = self._timestamps.get(key)
                    if ts is None:
                        return False
                    if isinstance(ts, tuple):
                        insert_time, ttl = ts
                        return time.time() - insert_time > ttl
                    return False

                def _purge_expired(self):
                    \"\"\"Remove all expired entries.\"\"\"
                    expired = [k for k in self._data if self._is_expired(k)]
                    for k in expired:
                        del self._data[k]
                        del self._timestamps[k]
            """)

        test_file = textwrap.dedent("""\
            \"\"\"Tests for cache.py.\"\"\"
            from cache import Cache


            def test_basic_set_get():
                c = Cache()
                c.set("a", 1)
                assert c.get("a") == 1


            def test_size():
                c = Cache()
                c.set("a", 1)
                c.set("b", 2)
                assert c.size() == 2


            def test_contains():
                c = Cache()
                c.set("a", 1)
                assert "a" in c
                assert "b" not in c
            """)

        return {
            "spec.txt": spec,
            "cache.py": cache_impl,
            "test_cache.py": test_file,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent("""\
            You are given a specification (`spec.txt`) with 8 requirements for a
            cache module, and an implementation (`cache.py`).

            The implementation deviates from the spec in exactly 5 ways.

            Your task:
            1. Read the spec carefully, requirement by requirement.
            2. Check each requirement against the code.
            3. List ALL 5 deviations in your answer.

            For each deviation, state:
            - Which requirement number is violated
            - What the spec says
            - What the code does instead

            Format your answer as a numbered list of deviations.
            """)

    # ── Solution (text answer, not code) ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        answer = textwrap.dedent("""\
            Deviations found:

            1. Requirement 1 (MAX_SIZE): Spec says max_size=100, but the code
               uses max_size=50 in __init__.

            2. Requirement 2 (EVICTION POLICY): Spec says eviction must be LRU,
               but the code uses FIFO eviction (removes the first inserted item
               via next(iter(self._data)) without moving to end on access in set).

            3. Requirement 3 (GET BEHAVIOR): Spec says get() returns None for
               missing keys, but the code raises KeyError.

            4. Requirement 4 (CLEAR BEHAVIOR): Spec says clear() removes ALL
               entries, but the code only removes expired entries.

            5. Requirement 5 (THREAD SAFETY): Spec says the cache must be
               thread-safe with locks, but the code has no locking mechanism.
            """)
        return {"answer.txt": answer}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent("""\
            I need to audit cache.py against spec.txt, checking each requirement
            systematically. Let me go through the spec requirement by requirement
            and verify compliance.

            Requirement 1 (MAX_SIZE=100): The spec says the cache must have a
            maximum capacity of 100 entries. Let me check the code. In __init__,
            the default parameter is `max_size=50`. This is a deviation — the spec
            says 100 but the code defaults to 50. Deviation #1 found.

            Requirement 2 (LRU EVICTION): The spec says eviction must follow LRU
            order. Let me trace the eviction logic in set(). When the cache exceeds
            max_size, the code does: `oldest = next(iter(self._data))` and deletes
            it. This removes the FIRST item in the OrderedDict, which is the
            oldest INSERTED item — that's FIFO, not LRU. For true LRU, the code
            should move accessed items to the end (which get() does via
            move_to_end), but the eviction itself removes the first item. However,
            the key issue is that set() for existing keys calls move_to_end, but
            new keys are added at the end. The first item is the oldest by
            insertion order, not by access order. Actually, get() does call
            move_to_end, so accessed items go to the end. But the spec says LRU
            and the code's comment says "FIFO eviction" explicitly. The eviction
            removes the first item which IS the least recently used IF get() moves
            to end. Wait — let me re-read. The comment says "FIFO eviction: remove
            the first inserted item." But with move_to_end in get(), the first item
            IS the LRU item. Hmm, but the code doesn't call move_to_end in set()
            for NEW keys — only for existing keys. And get() does call move_to_end.
            So actually this might be LRU... but the comment explicitly says FIFO.
            Let me check: if we never call get() and only set() new keys, the
            first item is the oldest inserted = FIFO order. The spec says LRU.
            The code comment itself says "FIFO eviction." This is a deviation.
            Deviation #2 found.

            Requirement 3 (GET returns None for missing): The spec says get(key)
            must return None for missing keys, not raise an exception. The code
            raises `KeyError(f"Key not found: {key}")`. This is a clear deviation.
            Deviation #3 found.

            Requirement 4 (CLEAR removes all): The spec says clear() must remove
            ALL entries. The code's clear() method only removes expired entries:
            `expired = [k for k in self._data if self._is_expired(k)]`. It does
            not remove non-expired entries. This is a deviation. Deviation #4 found.

            Requirement 5 (THREAD SAFETY): The spec says the cache must be
            thread-safe with locks. Let me check the code for any locking
            mechanism. Scanning the imports: `import time` and
            `from collections import OrderedDict`. No `threading` import, no Lock.
            None of the methods use a lock. This is a deviation. Deviation #5 found.

            Requirement 6 (TTL SUPPORT): The spec says the cache must support
            optional TTL per entry. The code has `default_ttl` in __init__ and
            accepts `ttl` parameter in set(). The _is_expired method checks TTL.
            This requirement is satisfied. No deviation.

            Requirement 7 (SIZE METHOD): The spec says size() returns the number
            of valid (non-expired) entries. The code's size() calls
            _purge_expired() then returns len(self._data). This is correct.
            No deviation.

            Requirement 8 (CONTAINS): The spec says `key in cache` returns True
            only for valid entries. The __contains__ method checks existence and
            expiration. This is correct. No deviation.

            Let me verify my count: I found exactly 5 deviations:
            1. max_size=50 instead of 100 — wrong default value
            2. FIFO eviction instead of LRU — wrong eviction policy
            3. get() raises KeyError instead of returning None — wrong error handling
            4. clear() only removes expired instead of all entries — missing full clear
            5. No thread safety / locking — missing requirement entirely

            All 5 are real deviations with clear evidence from spec vs code. There
            are no extra deviations — requirements 6, 7, and 8 are correctly
            implemented. The spec compliance audit is complete.
            """)

    # ── Grader ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        answer = grader_extract_answer(response)
        answer_lower = answer.lower()

        # Check which real deviations were found
        deviation_keywords = {
            "max_size": ["max_size", "max size", "100", "50", "capacity"],
            "eviction": ["lru", "fifo", "eviction", "least recently"],
            "get_missing": ["none", "keyerror", "missing", "not found", "exception"],
            "clear": ["clear", "empty", "all entries", "expired"],
            "thread_safe": ["thread", "lock", "concurrent", "thread-safe", "thread safe"],
        }

        found = []
        for dev_id, keywords in deviation_keywords.items():
            if any(kw in answer_lower for kw in keywords):
                found.append(dev_id)

        # Check for false positives — did the model report deviations in
        # requirements 6, 7, or 8 (which are correct)?
        # We look for specific deviation-claim patterns near these requirement
        # keywords, being careful not to match when they're mentioned in passing.
        false_positive_indicators = []
        fp_checks = {
            "ttl_wrong": ["ttl deviation", "ttl wrong", "ttl incorrect", "ttl missing",
                          "ttl is wrong", "ttl is incorrect", "ttl not implemented",
                          "time-to-live deviation", "time-to-live wrong"],
            "size_wrong": ["size deviation", "size wrong", "size incorrect",
                           "size is wrong", "size() is wrong", "size not correct",
                           "size method wrong", "size method deviation"],
            "contains_wrong": ["contains deviation", "contains wrong",
                               "contains incorrect", "__contains__ wrong",
                               "in operator wrong", "contains is wrong"],
        }
        for fp_id, keywords in fp_checks.items():
            for kw in keywords:
                if kw in answer_lower:
                    false_positive_indicators.append(fp_id)
                    break

        n_real = len(self._DEVIATIONS)
        n_found = len(found)
        n_false_pos = len(false_positive_indicators)

        recall = n_found / n_real if n_real > 0 else 0.0
        precision = n_found / (n_found + n_false_pos) if (n_found + n_false_pos) > 0 else 1.0
        score = 0.5 * recall + 0.5 * precision

        return score, {
            "deviations_found": found,
            "deviations_missing": [d for d, _ in self._DEVIATIONS if d not in found],
            "false_positives": false_positive_indicators,
            "recall": recall,
            "precision": precision,
            "score": score,
        }
