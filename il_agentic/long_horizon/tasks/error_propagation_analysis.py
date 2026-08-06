"""
Task: error_propagation_analysis

Reasoning skill: Error handling reasoning — tracing error paths through call chains.

A 3-module call chain (api_handler -> service -> repository) has broken error
propagation: the repository raises KeyError for missing records, the service
catches it and re-raises as ValueError (losing the "not found" semantic), and
the api_handler catches ValueError and returns a 500 for everything (including
not-found cases that should be 404).

The model must introduce a custom NotFoundError at the repository level, let it
propagate through the service, and catch it in the api_handler to return 404 —
while real errors still produce 500.

Failure mode: small models catch errors at the wrong level (too early or too
late), collapsing distinct error conditions into a single handler.
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer, extract_reasoning, parse_code_blocks,
    apply_code_changes, run_tests, compute_test_score,
)


@register_long_horizon
class ErrorPropagationAnalysis(LongHorizonEnv):
    task_id = "error_propagation_analysis"
    reasoning_skill = "Error handling reasoning — tracing error paths through call chains"
    failure_mode = (
        "Small models catch errors at the wrong level (too early or too late), "
        "collapsing distinct error conditions into a single handler."
    )
    token_budget = 700
    expected_concepts = [
        "error", "propagate", "exception", "call chain",
        "catch", "raise", "level", "verify",
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        repository = textwrap.dedent('''
            """Data access layer — talks to the in-memory store."""

            _store: dict[str, dict] = {}


            def seed(records: dict[str, dict]):
                """Populate the store with initial records."""
                _store.clear()
                _store.update(records)


            def get_record(record_id: str) -> dict:
                """Fetch a record by id.

                Raises KeyError when the record does not exist.
                """
                if record_id not in _store:
                    raise KeyError(record_id)
                return dict(_store[record_id])


            def save_record(record_id: str, data: dict) -> None:
                """Insert or update a record."""
                if not isinstance(data, dict):
                    raise TypeError("data must be a dict")
                _store[record_id] = dict(data)
        ''').strip()

        service = textwrap.dedent('''
            """Business logic layer — sits between the API handler and repository."""

            from repository import get_record, save_record


            def fetch_record(record_id: str) -> dict:
                """Return a record after applying business rules.

                Catches KeyError from the repository and re-raises as ValueError
                so callers get a uniform exception type.
                """
                try:
                    record = get_record(record_id)
                except KeyError:
                    # BUG: converting KeyError to ValueError loses the
                    # "not found" semantic — callers cannot distinguish a
                    # missing record from a genuinely bad value.
                    raise ValueError(f"record {record_id} not found")
                if "archived" in record and record["archived"]:
                    raise ValueError(f"record {record_id} is archived")
                return record


            def update_record(record_id: str, patch: dict) -> dict:
                """Apply a partial update and return the new record."""
                try:
                    existing = get_record(record_id)
                except KeyError:
                    raise ValueError(f"record {record_id} not found")
                existing.update(patch)
                save_record(record_id, existing)
                return existing
        ''').strip()

        api_handler = textwrap.dedent('''
            """HTTP-ish handler layer — translates service calls into status codes."""

            import service


            def handle_get(record_id: str) -> dict:
                """Handle a GET request for a record.

                Returns {"status": int, "body": dict}.
                """
                try:
                    record = service.fetch_record(record_id)
                except ValueError as exc:
                    # BUG: every ValueError becomes a 500, even when the
                    # record simply does not exist (should be 404).
                    return {"status": 500, "body": {"error": str(exc)}}
                return {"status": 200, "body": record}


            def handle_put(record_id: str, patch: dict) -> dict:
                """Handle a PUT request to update a record."""
                try:
                    record = service.update_record(record_id, patch)
                except ValueError as exc:
                    return {"status": 500, "body": {"error": str(exc)}}
                return {"status": 200, "body": record}
        ''').strip()

        tests = textwrap.dedent('''
            from repository import seed
            from api_handler import handle_get, handle_put


            def _setup():
                seed({"1": {"name": "alice", "archived": False},
                      "2": {"name": "bob", "archived": True}})


            # ── Correct behavior tests ──

            def test_get_existing():
                _setup()
                resp = handle_get("1")
                assert resp["status"] == 200
                assert resp["body"]["name"] == "alice"

            def test_put_existing():
                _setup()
                resp = handle_put("1", {"name": "ALICE"})
                assert resp["status"] == 200
                assert resp["body"]["name"] == "ALICE"

            def test_get_missing_returns_404():
                _setup()
                resp = handle_get("999")
                assert resp["status"] == 404

            def test_put_missing_returns_404():
                _setup()
                resp = handle_put("999", {"name": "x"})
                assert resp["status"] == 404

            # ── Error handling tests ──

            def test_get_archived_returns_404():
                _setup()
                resp = handle_get("2")
                assert resp["status"] == 404

            def test_get_missing_body_has_error():
                _setup()
                resp = handle_get("999")
                assert "error" in resp["body"]

            def test_real_error_returns_500():
                # Corrupt the store so repository raises something unexpected.
                _setup()
                import repository
                repository._store.clear()
                repository._store["x"] = "not a dict"  # triggers TypeError downstream
                resp = handle_get("x")
                assert resp["status"] == 500

            def test_handler_does_not_swallow_type_errors():
                _setup()
                import api_handler, service
                original = service.fetch_record
                def boom(record_id):
                    raise TypeError("boom")
                service.fetch_record = boom
                try:
                    resp = api_handler.handle_get("1")
                    assert resp["status"] == 500
                finally:
                    service.fetch_record = original
        ''').strip()

        return {
            "repository.py": repository,
            "service.py": service,
            "api_handler.py": api_handler,
            "test_error_propagation.py": tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent(f'''
            You are given a 3-module call chain:

              api_handler.py  ->  service.py  ->  repository.py

            The error handling is broken: the repository raises `KeyError` for
            missing records, the service catches it and re-raises as `ValueError`
            (losing the "not found" meaning), and the api_handler catches
            `ValueError` and returns HTTP 500 for everything — including
            not-found cases that should be 404.

            Fix the error handling at the RIGHT level of the call chain:

            1. `repository.py` should raise a custom `NotFoundError` exception
               when a record is missing.
            2. `service.py` should let `NotFoundError` propagate (or re-raise it)
               so the handler can distinguish "not found" from real errors.
            3. `api_handler.py` should catch `NotFoundError` and return 404, and
               catch other exceptions and return 500.

            All 8 tests in `test_error_propagation.py` must pass.  The four
            correct-behavior tests verify normal flow; the four error-handling
            tests verify that missing records yield 404 while real errors yield
            500.

            Return your solution as code blocks tagged with the filename, e.g.

            ```python:repository.py
            ...
            ```
        ''').strip()

    # ── Correct solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        repository = textwrap.dedent('''
            """Data access layer — talks to the in-memory store."""

            class NotFoundError(Exception):
                """Raised when a record id does not exist in the store."""
                pass


            _store: dict[str, dict] = {}


            def seed(records: dict[str, dict]):
                """Populate the store with initial records."""
                _store.clear()
                _store.update(records)


            def get_record(record_id: str) -> dict:
                """Fetch a record by id.

                Raises NotFoundError when the record does not exist.
                """
                if record_id not in _store:
                    raise NotFoundError(record_id)
                return dict(_store[record_id])


            def save_record(record_id: str, data: dict) -> None:
                """Insert or update a record."""
                if not isinstance(data, dict):
                    raise TypeError("data must be a dict")
                _store[record_id] = dict(data)
        ''').strip()

        service = textwrap.dedent('''
            """Business logic layer — sits between the API handler and repository."""

            from repository import get_record, save_record, NotFoundError


            def fetch_record(record_id: str) -> dict:
                """Return a record after applying business rules.

                NotFoundError from the repository is allowed to propagate so the
                handler can map it to 404.  Archived records are also treated as
                not-found for the caller's purposes.
                """
                record = get_record(record_id)
                if record.get("archived"):
                    raise NotFoundError(f"record {record_id} is archived")
                return record


            def update_record(record_id: str, patch: dict) -> dict:
                """Apply a partial update and return the new record."""
                existing = get_record(record_id)  # raises NotFoundError
                existing.update(patch)
                save_record(record_id, existing)
                return existing
        ''').strip()

        api_handler = textwrap.dedent('''
            """HTTP-ish handler layer — translates service calls into status codes."""

            import service
            from repository import NotFoundError


            def handle_get(record_id: str) -> dict:
                """Handle a GET request for a record."""
                try:
                    record = service.fetch_record(record_id)
                except NotFoundError as exc:
                    return {"status": 404, "body": {"error": str(exc)}}
                except Exception as exc:
                    return {"status": 500, "body": {"error": str(exc)}}
                return {"status": 200, "body": record}


            def handle_put(record_id: str, patch: dict) -> dict:
                """Handle a PUT request to update a record."""
                try:
                    record = service.update_record(record_id, patch)
                except NotFoundError as exc:
                    return {"status": 404, "body": {"error": str(exc)}}
                except Exception as exc:
                    return {"status": 500, "body": {"error": str(exc)}}
                return {"status": 200, "body": record}
        ''').strip()

        return {
            "repository.py": repository,
            "service.py": service,
            "api_handler.py": api_handler,
        }

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''
            Let me trace the error propagation through the call chain to
            understand why the current code is broken and where the fix belongs.

            Step 1 — Read repository.py:
            `get_record` raises `KeyError(record_id)` when the id is not in
            `_store`.  KeyError is a generic built-in; it does not carry the
            semantic "this record was not found" — it just means "this dict
            key is missing".  That is an error at the right level (the data
            layer knows the record is missing) but the wrong exception type
            for a domain concept.

            Step 2 — Read service.py:
            `fetch_record` does `try: get_record(...) except KeyError: raise
            ValueError(...)`.  This is the core mistake: it catches the error
            at the wrong level and converts it to ValueError.  ValueError is
            even more generic — it means "bad value".  By converting KeyError
            to ValueError, the service loses the "not found" information.
            The handler downstream cannot tell the difference between "record
            missing" and "record archived" (also ValueError) and "some other
            bad value".  The exception should propagate, not be collapsed.

            Step 3 — Read api_handler.py:
            `handle_get` catches `ValueError` and returns 500 for everything.
            Because the service collapsed not-found and real errors into the
            same ValueError, the handler has no way to return 404 for missing
            records.  It is catching at the right level (the boundary) but
            with the wrong exception type and wrong status mapping.

            Step 4 — Decide the fix at each level:
            - repository.py: define a custom `NotFoundError` and raise it
              instead of KeyError.  This preserves the domain semantic at the
              source of the error.
            - service.py: let NotFoundError propagate unchanged.  The service
              should not catch and re-raise as a generic type.  For archived
              records, raise NotFoundError too (they are effectively not
              available to callers).
            - api_handler.py: catch NotFoundError first -> 404, then catch
              Exception -> 500.  This maps the domain error to the correct
              HTTP status and still handles real errors as 500.

            Step 5 — Verify each error path by tracing through the tests:
            - test_get_existing: id "1" exists, not archived -> 200. OK.
            - test_put_existing: id "1" exists -> update -> 200. OK.
            - test_get_missing_returns_404: id "999" -> repository raises
              NotFoundError -> service propagates -> handler catches
              NotFoundError -> 404. OK.
            - test_put_missing_returns_404: same path through update_record ->
              get_record raises NotFoundError -> handler 404. OK.
            - test_get_archived_returns_404: id "2" archived -> service raises
              NotFoundError -> handler 404. OK.
            - test_get_missing_body_has_error: 404 body still contains
              "error" key. OK.
            - test_real_error_returns_500: store corrupted with a non-dict ->
              repository.get_record returns the string -> service tries
              record.get("archived") on a str -> AttributeError -> handler
              catches Exception -> 500. OK.
            - test_handler_does_not_swallow_type_errors: service.fetch_record
              monkeypatched to raise TypeError -> handler catches Exception
              (not NotFoundError) -> 500. OK.

            Let me double-check the exception hierarchy: NotFoundError extends
            Exception, not ValueError, so the handler's `except NotFoundError`
            clause runs before `except Exception`.  Order matters — the more
            specific catch must come first.  I have verified the ordering is
            correct in both handle_get and handle_put.

            To confirm: the key insight is that error information must propagate
            through the call chain without being collapsed.  Catching too early
            (in the service) and converting to a generic type breaks the
            handler's ability to distinguish error kinds.  The fix is to raise
            a specific exception at the source, let it propagate, and catch it
            at the boundary where the status mapping happens.
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
        test_code = codebase.get("test_error_propagation.py", "")
        results = run_tests(new_codebase, test_code, timeout=15.0)
        score, breakdown = compute_test_score(results)
        breakdown["results"] = results.get("results", [])
        breakdown["method"] = "run_tests"
        return score, breakdown
