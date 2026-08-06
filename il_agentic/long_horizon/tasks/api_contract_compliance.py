"""
Task: api_contract_compliance

Reasoning skill: Constraint satisfaction reasoning (10+ constraints).

A `UserRepository` class skeleton needs to be implemented to satisfy a
10-constraint API contract.  Each constraint is an independent requirement
that must be individually satisfied, but some constraints interact (e.g.,
"update must reject unknown user IDs" and "update must not allow changing
the user ID" must both hold simultaneously).

The 10 constraints:
1. add(user) must reject users with empty name
2. add(user) must reject duplicate user IDs
3. get(user_id) must return None for missing users (not raise)
4. update(user_id, changes) must reject unknown user IDs
5. update must not allow changing the user ID
6. delete(user_id) must be idempotent (deleting non-existent returns False)
7. list_all() must return a copy (not the internal list)
8. find_by_email(email) must be case-insensitive
9. All methods must accept keyword arguments
10. add must return the added user object

Tests: 12 tests (one per constraint + 2 extra edge cases).

Failure mode: small models satisfy some constraints but miss others,
especially the subtle ones (case-insensitivity, immutability of list_all,
idempotent delete).
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer, extract_reasoning, parse_code_blocks,
    apply_code_changes, run_tests, compute_test_score,
)


@register_long_horizon
class APIContractCompliance(LongHorizonEnv):
    task_id = "api_contract_compliance"
    reasoning_skill = "Constraint satisfaction reasoning (10+ constraints)"
    failure_mode = (
        "Small models satisfy some constraints but miss others, especially "
        "subtle ones like case-insensitivity, list immutability, and "
        "idempotent delete."
    )
    token_budget = 800
    expected_concepts = [
        "contract", "constraint", "validate", "edge case",
        "return type", "error", "immutable", "verify",
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        contract = textwrap.dedent('''
            # UserRepository API Contract

            ## Constraints

            1. add(user) must reject users with empty name (raise ValueError)
            2. add(user) must reject duplicate user IDs (raise ValueError)
            3. get(user_id) must return None for missing users (not raise)
            4. update(user_id, changes) must reject unknown user IDs
               (raise KeyError)
            5. update must not allow changing the user ID (if "id" in
               changes, raise ValueError)
            6. delete(user_id) must be idempotent: deleting a non-existent
               user returns False (no error)
            7. list_all() must return a copy of the internal list, not the
               internal list itself (mutations to the returned list must
               not affect the repository)
            8. find_by_email(email) must be case-insensitive
            9. All methods must accept keyword arguments (e.g.,
               get(user_id="x"), add(user={"id": "1", ...}))
            10. add(user) must return the added user object

            ## User object format
            A user is a dict with at least: "id" (str), "name" (str),
            "email" (str).
        ''').strip()

        skeleton = textwrap.dedent('''
            """User repository — implements the API contract.

            Fill in each method to satisfy all 10 constraints in the
            contract.  Each constraint is tested individually.
            """

            class UserRepository:
                def __init__(self):
                    self._users = {}  # user_id -> user dict
                    self._order = []  # insertion order for list_all

                def add(self, user):
                    """Add a user to the repository.

                    Constraints:
                    - Reject empty name (ValueError)
                    - Reject duplicate user IDs (ValueError)
                    - Return the added user object
                    """
                    # TODO: implement
                    raise NotImplementedError

                def get(self, user_id):
                    """Get a user by ID.

                    Constraint: return None for missing users (not raise).
                    """
                    # TODO: implement
                    raise NotImplementedError

                def update(self, user_id, changes):
                    """Update a user with partial changes.

                    Constraints:
                    - Reject unknown user IDs (KeyError)
                    - Do not allow changing the user ID (ValueError)
                    """
                    # TODO: implement
                    raise NotImplementedError

                def delete(self, user_id):
                    """Delete a user by ID.

                    Constraint: idempotent — return False for non-existent
                    users, True for successful deletion.
                    """
                    # TODO: implement
                    raise NotImplementedError

                def list_all(self):
                    """Return all users.

                    Constraint: return a copy of the internal list, not
                    the internal list itself.
                    """
                    # TODO: implement
                    raise NotImplementedError

                def find_by_email(self, email):
                    """Find a user by email (case-insensitive).

                    Constraint: matching must be case-insensitive.
                    Returns the user dict or None.
                    """
                    # TODO: implement
                    raise NotImplementedError
        ''').strip()

        tests = textwrap.dedent('''
            from user_repository import UserRepository


            def _make_user(uid="1", name="alice", email="alice@example.com"):
                return {"id": uid, "name": name, "email": email}


            # ── Constraint tests ──

            def test_add_rejects_empty_name():
                repo = UserRepository()
                user = _make_user(name="")
                try:
                    repo.add(user)
                    assert False, "should have raised ValueError"
                except ValueError:
                    pass

            def test_add_rejects_duplicate_id():
                repo = UserRepository()
                repo.add(_make_user(uid="1"))
                try:
                    repo.add(_make_user(uid="1", name="bob"))
                    assert False, "should have raised ValueError"
                except ValueError:
                    pass

            def test_get_returns_none_for_missing():
                repo = UserRepository()
                assert repo.get("nonexistent") is None

            def test_update_rejects_unknown_id():
                repo = UserRepository()
                try:
                    repo.update("nonexistent", {"name": "new"})
                    assert False, "should have raised KeyError"
                except KeyError:
                    pass

            def test_update_rejects_id_change():
                repo = UserRepository()
                repo.add(_make_user(uid="1"))
                try:
                    repo.update("1", {"id": "2"})
                    assert False, "should have raised ValueError"
                except ValueError:
                    pass

            def test_delete_is_idempotent():
                repo = UserRepository()
                # Deleting non-existent user returns False, no error.
                result = repo.delete("nonexistent")
                assert result is False

            def test_list_all_returns_copy():
                repo = UserRepository()
                repo.add(_make_user(uid="1"))
                repo.add(_make_user(uid="2"))
                users = repo.list_all()
                users.append({"id": "999", "name": "hack", "email": "h@e.com"})
                # Internal state should not be affected.
                assert len(repo.list_all()) == 2, "list_all did not return a copy"

            def test_find_by_email_case_insensitive():
                repo = UserRepository()
                repo.add(_make_user(uid="1", email="Alice@Example.COM"))
                result = repo.find_by_email("alice@example.com")
                assert result is not None
                assert result["id"] == "1"

            def test_methods_accept_keyword_args():
                repo = UserRepository()
                repo.add(user=_make_user(uid="1"))
                u = repo.get(user_id="1")
                assert u is not None
                assert u["id"] == "1"
                repo.update(user_id="1", changes={"name": "bob"})
                assert repo.get("1")["name"] == "bob"
                ok = repo.delete(user_id="1")
                assert ok is True

            def test_add_returns_user():
                repo = UserRepository()
                user = _make_user(uid="1")
                result = repo.add(user)
                assert result is user, "add must return the added user object"

            # ── Extra edge case tests ──

            def test_delete_existing_returns_true():
                repo = UserRepository()
                repo.add(_make_user(uid="1"))
                result = repo.delete("1")
                assert result is True
                assert repo.get("1") is None

            def test_find_by_email_missing_returns_none():
                repo = UserRepository()
                assert repo.find_by_email("nobody@example.com") is None
        ''').strip()

        return {
            "contract.md": contract,
            "user_repository.py": skeleton,
            "test_contract.py": tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''
            You are given an API contract (`contract.md`) with 10
            constraints for a `UserRepository` class, and a skeleton
            implementation in `user_repository.py`.

            Implement every method to satisfy ALL 10 constraints:

            1. add(user) rejects empty name (ValueError)
            2. add(user) rejects duplicate user IDs (ValueError)
            3. get(user_id) returns None for missing users
            4. update(user_id, changes) rejects unknown IDs (KeyError)
            5. update rejects changing the user ID (ValueError)
            6. delete(user_id) is idempotent (returns False for missing)
            7. list_all() returns a copy (not the internal list)
            8. find_by_email(email) is case-insensitive
            9. All methods accept keyword arguments
            10. add(user) returns the added user object

            Pay attention to interactions between constraints: e.g.,
            constraint 4 and 5 both apply to update — an unknown ID raises
            KeyError, but a known ID with an "id" change raises ValueError.

            All 12 tests in `test_contract.py` must pass.

            Return your solution as a code block tagged with the filename:

            ```python:user_repository.py
            ...
            ```
        ''').strip()

    # ── Correct solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        solution = textwrap.dedent('''
            """User repository — implements the 10-constraint API contract."""

            class UserRepository:
                def __init__(self):
                    self._users = {}  # user_id -> user dict
                    self._order = []  # insertion order for list_all

                def add(self, user):
                    """Add a user to the repository.

                    Constraints: reject empty name, reject duplicate IDs,
                    return the added user object.
                    """
                    if not user.get("name"):
                        raise ValueError("user name must not be empty")
                    uid = user["id"]
                    if uid in self._users:
                        raise ValueError(f"duplicate user id: {uid}")
                    stored = dict(user)
                    self._users[uid] = stored
                    self._order.append(uid)
                    return user

                def get(self, user_id):
                    """Get a user by ID.  Returns None for missing users."""
                    user = self._users.get(user_id)
                    return dict(user) if user is not None else None

                def update(self, user_id, changes):
                    """Update a user with partial changes.

                    Constraints: reject unknown IDs (KeyError), reject
                    changing the user ID (ValueError).
                    """
                    if user_id not in self._users:
                        raise KeyError(f"unknown user id: {user_id}")
                    if "id" in changes:
                        raise ValueError("cannot change user id")
                    user = self._users[user_id]
                    user.update(changes)

                def delete(self, user_id):
                    """Delete a user by ID.  Idempotent: returns False for
                    non-existent users, True for successful deletion."""
                    if user_id not in self._users:
                        return False
                    del self._users[user_id]
                    self._order.remove(user_id)
                    return True

                def list_all(self):
                    """Return all users as a copy of the internal list."""
                    return [dict(self._users[uid]) for uid in self._order]

                def find_by_email(self, email):
                    """Find a user by email (case-insensitive).

                    Returns the user dict or None.
                    """
                    email_lower = email.lower()
                    for uid in self._order:
                        user = self._users[uid]
                        if user.get("email", "").lower() == email_lower:
                            return dict(user)
                    return None
        ''').strip()

        return {"user_repository.py": solution}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''
            Let me read the contract constraint by constraint and implement
            each one, then verify against its test and check for
            interactions between constraints.

            Step 1 — Constraint 1: add rejects empty name.
            I check `if not user.get("name")` and raise ValueError.  This
            catches both missing name and empty string name.  The test
            `test_add_rejects_empty_name` passes a user with name="" and
            expects ValueError.  Let me verify: `not "".get` — wait, the
            user is a dict, so `user.get("name")` returns "" which is
            falsy, so `not ""` is True, and we raise.  Correct.

            Step 2 — Constraint 2: add rejects duplicate user IDs.
            I check `if uid in self._users` and raise ValueError.  The test
            adds a user with uid="1", then tries to add another with
            uid="1".  The second add sees "1" in _users and raises.
            Correct.

            Step 3 — Constraint 10: add returns the added user object.
            I return `user` (the original argument) at the end of add.
            The test `test_add_returns_user` checks `result is user`.
            Correct — I return the same object, not a copy.

            Step 4 — Constraint 3: get returns None for missing users.
            I use `self._users.get(user_id)` which returns None if the key
            is not found.  No exception raised.  The test
            `test_get_returns_none_for_missing` checks `repo.get("nonexistent")
            is None`.  Correct.

            Step 5 — Constraint 4: update rejects unknown user IDs.
            I check `if user_id not in self._users` and raise KeyError.
            The test `test_update_rejects_unknown_id` expects KeyError.
            Correct.

            Step 6 — Constraint 5: update rejects changing the user ID.
            I check `if "id" in changes` and raise ValueError.  This must
            be checked AFTER the unknown ID check — if the user does not
            exist, we raise KeyError first (constraint 4 takes priority).
            The test `test_update_rejects_id_change` adds a user with
            uid="1", then calls update("1", {"id": "2"}) and expects
            ValueError.  Let me verify: user "1" exists (passes constraint
            4 check), then "id" is in changes -> raise ValueError.
            Correct.

            Interaction check: if someone calls update("nonexistent",
            {"id": "2"}), which error fires?  The unknown ID check comes
            first, so KeyError is raised.  This is the correct priority —
            we validate the user exists before checking what changes are
            allowed.

            Step 7 — Constraint 6: delete is idempotent.
            I check `if user_id not in self._users: return False`.  No
            exception raised.  The test `test_delete_is_idempotent` calls
            delete("nonexistent") and expects False.  Correct.  For
            existing users, I delete and return True.

            Step 8 — Constraint 7: list_all returns a copy.
            I return `[dict(self._users[uid]) for uid in self._order]` —
            a new list of new dicts.  The test appends to the returned list
            and checks the internal state is unchanged.  Since the returned
            list is a new list object, appending to it does not affect
            _order or _users.  Correct.  The internal list is _order (a
            list of IDs), and I never return it directly.

            Step 9 — Constraint 8: find_by_email is case-insensitive.
            I lower-case both the search email and the stored email before
            comparing: `user.get("email", "").lower() == email.lower()`.
            The test stores "Alice@Example.COM" and searches for
            "alice@example.com".  Both lower-case to "alice@example.com".
            Correct.

            Step 10 — Constraint 9: all methods accept keyword arguments.
            In Python, any method that takes positional args also accepts
            them as keyword args by default.  The test calls
            `repo.add(user=...)`, `repo.get(user_id="1")`,
            `repo.update(user_id="1", changes=...)`,
            `repo.delete(user_id="1")`.  My method signatures are
            `add(self, user)`, `get(self, user_id)`, `update(self,
            user_id, changes)`, `delete(self, user_id)` — all parameters
            can be passed as keywords.  Correct.

            Step 11 — Verify extra edge cases:
            - test_delete_existing_returns_true: add user "1", delete "1"
              -> returns True, get("1") is None.  My delete removes from
              _users and _order, returns True.  Correct.
            - test_find_by_email_missing_returns_none: no users, search
              returns None.  The loop finds nothing, returns None.
              Correct.

            Step 12 — Verify data integrity:
            I store `dict(user)` (a copy) in _users to avoid external
            mutations affecting the repository.  get() returns a copy too,
            so callers cannot mutate the internal state through the
            returned object.  list_all() returns copies of each user dict.
            This is a defensive measure that supports the immutability
            constraint (#7).

            To confirm: I have implemented all 10 constraints, verified
            each against its test, checked the interaction between
            constraints 4 and 5 (unknown ID check before ID change check),
            and verified the 2 extra edge case tests.  The key subtleties
            are: case-insensitive email matching (lower-case both sides),
            list_all returning a copy (new list of new dicts), idempotent
            delete (return False, not raise), and add returning the
            original user object (not a copy).
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
        test_code = codebase.get("test_contract.py", "")
        results = run_tests(new_codebase, test_code, timeout=15.0)
        score, breakdown = compute_test_score(results)
        breakdown["results"] = results.get("results", [])
        breakdown["method"] = "run_tests"
        return score, breakdown
