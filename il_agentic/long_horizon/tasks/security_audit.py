"""
Security Audit — Security reasoning task.

A user service module has 4 security vulnerabilities:
1. SQL injection: builds query with f-string instead of parameterized query
2. Path traversal: uses user input directly in file path without sanitization
3. Command injection: passes user input to subprocess with shell=True
4. Information leak: returns full exception messages to the caller

The model must identify all 4 vulnerabilities and fix them. This is a
Q&A + code task — identify vulnerabilities in text, then fix in code.

Grader: 0.5 * behavior_tests + 0.5 * security_tests

Failure mode: small models miss subtle vulnerabilities beyond the obvious
SQL injection, and fail to trace input flows to dangerous sinks.
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    parse_code_blocks, apply_code_changes, run_tests,
    compute_test_score,
)


@register_long_horizon
class SecurityAudit(LongHorizonEnv):
    task_id = "security_audit"
    reasoning_skill = "Security reasoning — tracing input flows for vulnerabilities"
    failure_mode = "Small models miss subtle vulnerabilities (not just SQL injection)"
    token_budget = 800
    expected_concepts = ["security", "vulnerability", "injection", "input", "sanitize", "escape", "flow", "verify"]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        user_service = textwrap.dedent("""\
            \"\"\"User service module — handles user data operations.

            WARNING: This module contains security vulnerabilities.
            \"\"\"
            import os
            import subprocess


            class UserService:
                \"\"\"Service for managing user data.\"\"\"

                def __init__(self, db_connection, data_dir="/app/data"):
                    self.db = db_connection
                    self.data_dir = data_dir

                def get_user(self, user_id):
                    \"\"\"Get a user by ID from the database.\"\"\"
                    query = f"SELECT * FROM users WHERE id = '{user_id}'"
                    try:
                        result = self.db.execute(query)
                        return result
                    except Exception as e:
                        return {"error": str(e)}

                def save_profile(self, user_id, profile_data):
                    \"\"\"Save user profile data to a file.\"\"\"
                    filepath = os.path.join(self.data_dir, f"{user_id}.json")
                    try:
                        with open(filepath, 'w') as f:
                            f.write(profile_data)
                        return {"status": "saved"}
                    except Exception as e:
                        return {"error": str(e)}

                def run_report(self, report_name):
                    \"\"\"Run a system report by name.\"\"\"
                    cmd = f"generate_report --name {report_name}"
                    try:
                        result = subprocess.run(
                            cmd, shell=True, capture_output=True, text=True
                        )
                        return {"output": result.stdout}
                    except Exception as e:
                        return {"error": str(e)}

                def delete_user(self, user_id):
                    \"\"\"Delete a user from the database.\"\"\"
                    query = f"DELETE FROM users WHERE id = '{user_id}'"
                    try:
                        self.db.execute(query)
                        return {"status": "deleted"}
                    except Exception as e:
                        return {"error": str(e)}
            """)

        test_file = textwrap.dedent("""\
            \"\"\"Tests for user_service — behavior + security tests.\"\"\"
            import os
            import tempfile
            from user_service import UserService


            class MockDB:
                def __init__(self):
                    self.queries = []
                    self.results = {}
                def execute(self, query, params=None):
                    self.queries.append((query, params))
                    if "SELECT" in query:
                        return self.results.get("select", {"id": 1, "name": "test"})
                    return None


            # ── Behavior tests ──

            def test_get_user_behavior():
                db = MockDB()
                svc = UserService(db)
                result = svc.get_user(42)
                assert result is not None


            def test_save_profile_behavior():
                db = MockDB()
                with tempfile.TemporaryDirectory() as tmpdir:
                    svc = UserService(db, data_dir=tmpdir)
                    result = svc.save_profile("user123", '{"name": "test"}')
                    assert result["status"] == "saved"


            def test_delete_user_behavior():
                db = MockDB()
                svc = UserService(db)
                result = svc.delete_user(42)
                assert result["status"] == "deleted"


            def test_run_report_behavior():
                db = MockDB()
                svc = UserService(db)
                result = svc.run_report("monthly_summary")
                assert "output" in result or "error" in result


            # ── Security tests ──

            def test_sql_injection_blocked():
                \"\"\"SQL injection in get_user should be blocked.\"\"\"
                db = MockDB()
                svc = UserService(db)
                # Classic SQL injection attempt
                malicious_id = "1' OR '1'='1"
                svc.get_user(malicious_id)
                # Check that the query used parameterization, not f-string
                query, params = db.queries[-1]
                assert params is not None, "Query should use parameterized inputs"
                assert malicious_id not in query, "User input should not be in query string"


            def test_path_traversal_blocked():
                \"\"\"Path traversal in save_profile should be blocked.\"\"\"
                db = MockDB()
                with tempfile.TemporaryDirectory() as tmpdir:
                    svc = UserService(db, data_dir=tmpdir)
                    malicious_id = "../../etc/passwd"
                    result = svc.save_profile(malicious_id, "data")
                    # The file should NOT be created outside data_dir
                    assert not os.path.exists("/etc/passwd.json")
                    # Should either reject or sanitize the path
                    assert result.get("status") != "saved" or os.path.join(
                        tmpdir, "../../etc/passwd.json"
                    ) != os.path.normpath(os.path.join(tmpdir, "../../etc/passwd.json"))


            def test_command_injection_blocked():
                \"\"\"Command injection in run_report should be blocked.\"\"\"
                db = MockDB()
                svc = UserService(db)
                malicious_name = "test; rm -rf /"
                result = svc.run_report(malicious_name)
                # Should not execute the injected command
                # If shell=False is used, the injection is treated as a single arg
                assert "output" in result or "error" in result


            def test_info_leak_blocked():
                \"\"\"Exception details should not be leaked to caller.\"\"\"
                db = MockDB()
                db.results = {}
                db.execute = lambda q, p=None: (_ for _ in ()).throw(
                    Exception("Internal DB connection string: server=db.internal:5432; password=secret123")
                )
                svc = UserService(db)
                result = svc.get_user(1)
                error_msg = result.get("error", "")
                assert "password" not in error_msg.lower(), "Error message should not contain internal details"
                assert "secret" not in error_msg.lower(), "Error message should not contain secrets"
            """)

        return {
            "user_service.py": user_service,
            "test_security.py": test_file,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent("""\
            The `user_service.py` module contains 4 security vulnerabilities.
            Your task has two parts:

            PART 1 — Identify all 4 vulnerabilities:
            In your reasoning, trace each user input flow through the code to
            identify where user input reaches a dangerous sink. For each
            vulnerability, explain:
            - Which function is affected
            - What the vulnerability is (type)
            - How user input flows to the dangerous sink
            - What the fix should be

            PART 2 — Fix all 4 vulnerabilities:
            Provide a fixed `user_service.py` that:
            - Uses parameterized queries for database operations
            - Sanitizes file paths to prevent path traversal
            - Uses shell=False for subprocess calls
            - Returns generic error messages (no internal details)

            The tests in `test_security.py` include both behavior tests and
            security tests. Your fix must pass ALL of them.

            Provide your fixed `user_service.py` in a code block.
            """)

    # ── Solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        fixed = textwrap.dedent("""\
            \"\"\"User service module — handles user data operations (fixed).\"\"\"
            import os
            import subprocess
            import re


            class UserService:
                \"\"\"Service for managing user data.\"\"\"

                def __init__(self, db_connection, data_dir="/app/data"):
                    self.db = db_connection
                    self.data_dir = data_dir

                def get_user(self, user_id):
                    \"\"\"Get a user by ID from the database.\"\"\"
                    query = "SELECT * FROM users WHERE id = %s"
                    try:
                        result = self.db.execute(query, (user_id,))
                        return result
                    except Exception:
                        return {"error": "Failed to retrieve user"}

                def save_profile(self, user_id, profile_data):
                    \"\"\"Save user profile data to a file.\"\"\"
                    # Sanitize user_id to prevent path traversal
                    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', str(user_id))
                    if not safe_id:
                        return {"error": "Invalid user ID"}
                    filepath = os.path.join(self.data_dir, f"{safe_id}.json")
                    # Verify the resolved path is within data_dir
                    if not os.path.abspath(filepath).startswith(
                        os.path.abspath(self.data_dir)
                    ):
                        return {"error": "Invalid file path"}
                    try:
                        with open(filepath, 'w') as f:
                            f.write(profile_data)
                        return {"status": "saved"}
                    except Exception:
                        return {"error": "Failed to save profile"}

                def run_report(self, report_name):
                    \"\"\"Run a system report by name.\"\"\"
                    # Use shell=False and pass args as a list to prevent injection
                    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', str(report_name))
                    if not safe_name:
                        return {"error": "Invalid report name"}
                    cmd = ["generate_report", "--name", safe_name]
                    try:
                        result = subprocess.run(
                            cmd, shell=False, capture_output=True, text=True
                        )
                        return {"output": result.stdout}
                    except Exception:
                        return {"error": "Failed to run report"}

                def delete_user(self, user_id):
                    \"\"\"Delete a user from the database.\"\"\"
                    query = "DELETE FROM users WHERE id = %s"
                    try:
                        self.db.execute(query, (user_id,))
                        return {"status": "deleted"}
                    except Exception:
                        return {"error": "Failed to delete user"}
            """)
        return {"user_service.py": fixed}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent("""\
            I need to trace all user input flows through user_service.py to identify
            security vulnerabilities. Let me analyze each function systematically.

            Function 1: get_user(user_id)
            The user_id parameter comes from the caller (potentially untrusted).
            Let me trace the flow:
            - user_id is interpolated into an f-string: f"SELECT * FROM users WHERE id = '{user_id}'"
            - This query string is passed directly to self.db.execute(query)
            The user input flows directly into the SQL query string without
            parameterization. This is a SQL injection vulnerability. An attacker
            could pass: 1' OR '1'='1 — which would make the query:
            SELECT * FROM users WHERE id = '1' OR '1'='1' — returning all users.
            Fix: use parameterized queries — "SELECT * FROM users WHERE id = %s"
            with (user_id,) as a parameter tuple.

            Function 2: save_profile(user_id, profile_data)
            The user_id parameter flows into:
            - filepath = os.path.join(self.data_dir, f"{user_id}.json")
            - This filepath is used in open(filepath, 'w')
            The user input flows directly into a file path without sanitization.
            This is a path traversal vulnerability. An attacker could pass:
            "../../etc/passwd" — which would create a file outside data_dir.
            Fix: sanitize user_id to remove path separators, and verify the
            resolved path stays within data_dir using os.path.abspath.

            Function 3: run_report(report_name)
            The report_name parameter flows into:
            - cmd = f"generate_report --name {report_name}"
            - subprocess.run(cmd, shell=True, ...)
            The user input flows into a shell command with shell=True. This is a
            command injection vulnerability. An attacker could pass:
            "test; rm -rf /" — which would execute the destructive command.
            Fix: use shell=False and pass arguments as a list:
            ["generate_report", "--name", safe_name]. Also sanitize and escape
            the input to remove shell metacharacters.

            Function 4: delete_user(user_id)
            Same pattern as get_user — user_id is interpolated into an f-string
            SQL query. This is also a SQL injection vulnerability.
            Fix: same as get_user — use parameterized queries.

            Function 5 (cross-cutting): error handling
            All functions catch exceptions and return str(e) to the caller:
            - return {"error": str(e)}
            This leaks internal exception details (including stack traces, database
            connection strings, file paths, etc.) to the caller. This is an
            information leak vulnerability.
            Fix: return generic error messages like "Failed to retrieve user"
            instead of str(e).

            Summary of 4 vulnerabilities:
            1. SQL injection in get_user and delete_user (f-string query)
            2. Path traversal in save_profile (unsanitized file path)
            3. Command injection in run_report (shell=True with user input)
            4. Information leak in all functions (str(e) returned to caller)

            Now let me implement the fixes:
            - get_user: "SELECT * FROM users WHERE id = %s" with (user_id,)
            - delete_user: "DELETE FROM users WHERE id = %s" with (user_id,)
            - save_profile: sanitize user_id with regex, verify path within data_dir
            - run_report: use shell=False, pass args as list, sanitize report_name
            - All functions: return generic error messages instead of str(e)

            Let me verify the fixes pass all tests:

            Behavior tests:
            - test_get_user_behavior: get_user(42) with parameterized query — MockDB
              needs to accept params. The query "SELECT * FROM users WHERE id = %s"
              with params (42,) should work. ✓
            - test_save_profile_behavior: save_profile("user123", ...) — "user123"
              passes sanitization (alphanumeric), file created in tmpdir. ✓
            - test_delete_user_behavior: delete_user(42) with parameterized query. ✓
            - test_run_report_behavior: run_report("monthly_summary") — sanitized,
              shell=False. The command may fail (generate_report not installed)
              but returns error dict. ✓

            Security tests:
            - test_sql_injection_blocked: "1' OR '1'='1" is passed as a parameter,
              not in the query string. params is not None, and the malicious string
              is not in the query. ✓
            - test_path_traversal_blocked: "../../etc/passwd" is sanitized to
              "etcpasswd" (path separators removed), and the path check prevents
              traversal. ✓
            - test_command_injection_blocked: "test; rm -rf /" is sanitized to
              "testrm-rf-" and passed as a single arg with shell=False. ✓
            - test_info_leak_blocked: exceptions return generic messages, not str(e).
              No "password" or "secret" in the error. ✓

            All 8 tests should pass with the fixes applied. Let me verify the MockDB
            works with parameterized queries — the test's MockDB.execute accepts
            (query, params) so it should work.
            """)

    # ── Grader ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        changes = parse_code_blocks(response)
        if not changes:
            return 0.0, {"reason": "no code blocks found in response"}

        # Check that the model actually changed the target file
        if "user_service.py" not in changes:
            return 0.0, {"reason": "user_service.py not modified — must fix vulnerabilities",
                        "files_changed": list(changes.keys())}

        # Check that the code was actually modified (not just re-submitted unchanged)
        if changes["user_service.py"].strip() == codebase.get("user_service.py", "").strip():
            return 0.0, {"reason": "user_service.py unchanged — no security fixes applied"}

        new_codebase = apply_code_changes(codebase, changes)

        if "user_service.py" not in new_codebase:
            return 0.0, {"reason": "user_service.py not in response"}

        # Run all tests (behavior + security)
        test_code = codebase["test_security.py"]
        results = run_tests(new_codebase, test_code, timeout=15.0)
        test_score, test_breakdown = compute_test_score(results)

        # Separate behavior and security tests
        test_results = results.get("results", [])
        behavior_tests = [r for r in test_results if "behavior" in r.get("name", "")]
        security_tests = [r for r in test_results if "blocked" in r.get("name", "")]

        behavior_passed = sum(1 for r in behavior_tests if r.get("status") == "pass")
        security_passed = sum(1 for r in security_tests if r.get("status") == "pass")

        behavior_total = len(behavior_tests) if behavior_tests else 4
        security_total = len(security_tests) if security_tests else 4

        behavior_score = behavior_passed / behavior_total if behavior_total > 0 else 0.0
        security_score = security_passed / security_total if security_total > 0 else 0.0

        # If we couldn't separate tests, use overall score
        if not behavior_tests and not security_tests:
            behavior_score = test_score
            security_score = test_score

        combined = 0.5 * behavior_score + 0.5 * security_score

        return combined, {
            "behavior_score": behavior_score,
            "security_score": security_score,
            "behavior_passed": behavior_passed,
            "behavior_total": behavior_total,
            "security_passed": security_passed,
            "security_total": security_total,
            "combined_score": combined,
            "test_breakdown": test_breakdown,
            "results": test_results,
        }
