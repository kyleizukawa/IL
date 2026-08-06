"""
Environment 5: API Client Implementation

Skill: Implementing an API client from a specification.

The model is given an API spec (endpoints, request/response formats) and
a mock server. It must implement a client class with correct method calls,
response parsing, and error handling.

Difficulty scaling:
- easy: 2 endpoints, simple flat responses
- medium: 4 endpoints, nested responses, query parameters
- hard: 6 endpoints, pagination, error codes, nested data
"""
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, extract_reasoning, parse_code_blocks, apply_code_changes,
    compute_test_score, CodeExecutor,
)


def _run_tests(codebase, test_code, timeout=15.0):
    """Run test functions against a codebase using CodeExecutor directly.

    Works around a textwrap.dedent common-prefix issue in graders.run_tests
    by constructing the wrapper at column 0 (no indentation needed).
    """
    wrapper = (
        "import sys, json, traceback, io\n"
        "\n"
        "results = []\n"
        "total = 0\n"
        "passed = 0\n"
        "failed = 0\n"
        "errors = 0\n"
        "\n"
        + test_code
        + "\n\n"
        "import inspect\n"
        "test_funcs = [(name, obj) for name, obj in list(globals().items())\n"
        "              if name.startswith('test_') and callable(obj)]\n"
        "\n"
        "for name, func in test_funcs:\n"
        "    total += 1\n"
        "    try:\n"
        "        func()\n"
        "        passed += 1\n"
        "        results.append({'name': name, 'status': 'pass'})\n"
        "    except AssertionError as e:\n"
        "        failed += 1\n"
        "        results.append({'name': name, 'status': 'fail', 'error': str(e)})\n"
        "    except Exception as e:\n"
        "        errors += 1\n"
        "        results.append({'name': name, 'status': 'error', 'error': traceback.format_exc()})\n"
        "\n"
        "output = {\n"
        "    'total': total, 'passed': passed, 'failed': failed, 'errors': errors,\n"
        "    'results': results,\n"
        "}\n"
        "print(json.dumps(output))\n"
    )
    with CodeExecutor(timeout=timeout) as executor:
        executor.write_codebase(codebase)
        result = executor.run(wrapper)

    if result['timed_out'] or result['returncode'] != 0:
        return {
            'total': 0, 'passed': 0, 'failed': 0, 'errors': 0,
            'results': [], 'stdout': result.get('stdout', ''),
            'stderr': result.get('stderr', ''),
            'error': result.get('error', 'Unknown error'),
            'timed_out': result.get('timed_out', False),
        }

    import json as _json
    stdout = result['stdout']
    for line in stdout.strip().split('\n'):
        line = line.strip()
        if line.startswith('{') and 'total' in line:
            try:
                return _json.loads(line)
            except _json.JSONDecodeError:
                pass
    return {
        'total': 0, 'passed': 0, 'failed': 0, 'errors': 0,
        'results': [], 'stdout': stdout, 'stderr': result['stderr'],
        'error': 'No JSON output found',
    }


# ── Domain definitions ──
# Each domain has:
#   spec: the API specification text shown to the model
#   mock_server: Python class simulating the API
#   client_template: the skeleton the model must complete
#   client_solution: the correct client implementation
#   tests: test cases that run the client against the mock server

DOMAINS = {
    # ── Domain 1: User REST API ──
    "user_api": {
        "spec": textwrap.dedent('''
            API Specification: User Management REST API
            Base URL: http://localhost:8000

            Endpoints:
            1. GET /users
               - Query params: limit (int, optional, default=10)
               - Response 200: {"users": [{"id": int, "name": str, "email": str}, ...]}
               - Response 400: {"error": "invalid limit"}

            2. GET /users/{user_id}
               - Path param: user_id (int)
               - Response 200: {"id": int, "name": str, "email": str}
               - Response 404: {"error": "user not found"}

            3. POST /users
               - Body: {"name": str, "email": str}
               - Response 201: {"id": int, "name": str, "email": str}
               - Response 400: {"error": "invalid input"}

            4. DELETE /users/{user_id}
               - Path param: user_id (int)
               - Response 200: {"deleted": true, "id": int}
               - Response 404: {"error": "user not found"}
        ''').strip(),
        "mock_server": textwrap.dedent('''
            class MockServer:
                """Mock server simulating the User Management REST API."""
                def __init__(self):
                    self._users = {}
                    self._next_id = 1

                def handle(self, method, path, query=None, body=None):
                    query = query or {}
                    body = body or {}
                    if method == "GET" and path == "/users":
                        try:
                            limit = int(query.get("limit", 10))
                        except (ValueError, TypeError):
                            return 400, {"error": "invalid limit"}
                        users = list(self._users.values())[:limit]
                        return 200, {"users": users}
                    if method == "GET" and path.startswith("/users/"):
                        try:
                            uid = int(path.split("/")[-1])
                        except ValueError:
                            return 404, {"error": "user not found"}
                        if uid not in self._users:
                            return 404, {"error": "user not found"}
                        return 200, self._users[uid]
                    if method == "POST" and path == "/users":
                        if "name" not in body or "email" not in body:
                            return 400, {"error": "invalid input"}
                        uid = self._next_id
                        self._next_id += 1
                        user = {"id": uid, "name": body["name"], "email": body["email"]}
                        self._users[uid] = user
                        return 201, user
                    if method == "DELETE" and path.startswith("/users/"):
                        try:
                            uid = int(path.split("/")[-1])
                        except ValueError:
                            return 404, {"error": "user not found"}
                        if uid not in self._users:
                            return 404, {"error": "user not found"}
                        del self._users[uid]
                        return 200, {"deleted": True, "id": uid}
                    return 404, {"error": "not found"}
        ''').strip(),
        "client_template": textwrap.dedent('''
            class UserAPIClient:
                """Client for the User Management REST API."""
                def __init__(self, server):
                    self.server = server

                def list_users(self, limit=10):
                    """GET /users - list users with optional limit."""
                    # TODO: implement
                    pass

                def get_user(self, user_id):
                    """GET /users/{user_id} - get a single user."""
                    # TODO: implement
                    pass

                def create_user(self, name, email):
                    """POST /users - create a new user."""
                    # TODO: implement
                    pass

                def delete_user(self, user_id):
                    """DELETE /users/{user_id} - delete a user."""
                    # TODO: implement
                    pass
        ''').strip(),
        "client_solution": textwrap.dedent('''
            class UserAPIClient:
                """Client for the User Management REST API."""
                def __init__(self, server):
                    self.server = server

                def list_users(self, limit=10):
                    """GET /users - list users with optional limit."""
                    status, data = self.server.handle("GET", "/users", query={"limit": limit})
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data["users"]

                def get_user(self, user_id):
                    """GET /users/{user_id} - get a single user."""
                    status, data = self.server.handle("GET", f"/users/{user_id}")
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def create_user(self, name, email):
                    """POST /users - create a new user."""
                    status, data = self.server.handle("POST", "/users", body={"name": name, "email": email})
                    if status != 201:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def delete_user(self, user_id):
                    """DELETE /users/{user_id} - delete a user."""
                    status, data = self.server.handle("DELETE", f"/users/{user_id}")
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data
        ''').strip(),
        "tests": textwrap.dedent('''
            from mock_server import MockServer
            from api_client import UserAPIClient

            def test_list_users_empty():
                server = MockServer()
                client = UserAPIClient(server)
                users = client.list_users()
                assert users == [], f"expected empty list, got {users}"

            def test_create_and_get_user():
                server = MockServer()
                client = UserAPIClient(server)
                created = client.create_user("Alice", "alice@example.com")
                assert created["name"] == "Alice"
                assert created["email"] == "alice@example.com"
                assert "id" in created
                fetched = client.get_user(created["id"])
                assert fetched["name"] == "Alice"
                assert fetched["email"] == "alice@example.com"

            def test_list_users_with_limit():
                server = MockServer()
                client = UserAPIClient(server)
                for i in range(5):
                    client.create_user(f"User{i}", f"user{i}@example.com")
                users = client.list_users(limit=3)
                assert len(users) == 3, f"expected 3 users, got {len(users)}"

            def test_delete_user():
                server = MockServer()
                client = UserAPIClient(server)
                created = client.create_user("Bob", "bob@example.com")
                result = client.delete_user(created["id"])
                assert result["deleted"] == True
                assert result["id"] == created["id"]

            def test_get_nonexistent_user():
                server = MockServer()
                client = UserAPIClient(server)
                try:
                    client.get_user(999)
                    assert False, "should have raised an exception"
                except Exception as e:
                    assert "not found" in str(e).lower()

            def test_delete_nonexistent_user():
                server = MockServer()
                client = UserAPIClient(server)
                try:
                    client.delete_user(999)
                    assert False, "should have raised an exception"
                except Exception as e:
                    assert "not found" in str(e).lower()
        ''').strip(),
    },

    # ── Domain 2: Weather API ──
    "weather_api": {
        "spec": textwrap.dedent('''
            API Specification: Weather Service API
            Base URL: http://localhost:8000

            Endpoints:
            1. GET /weather/current
               - Query params: city (str, required), units (str, optional: "metric"|"imperial", default="metric")
               - Response 200: {"city": str, "temperature": float, "units": str, "conditions": str, "humidity": int}
               - Response 400: {"error": "city is required"}

            2. GET /weather/forecast
               - Query params: city (str, required), days (int, optional, default=5)
               - Response 200: {"city": str, "forecast": [{"day": int, "temperature": float, "conditions": str}, ...]}
               - Response 400: {"error": "city is required"}

            3. GET /weather/alerts
               - Query params: city (str, required)
               - Response 200: {"city": str, "alerts": [{"type": str, "severity": str, "message": str}, ...]}
               - Response 404: {"error": "no alerts for city"}
        ''').strip(),
        "mock_server": textwrap.dedent('''
            class MockServer:
                """Mock server simulating the Weather Service API."""
                def handle(self, method, path, query=None, body=None):
                    query = query or {}
                    if method == "GET" and path == "/weather/current":
                        if "city" not in query:
                            return 400, {"error": "city is required"}
                        units = query.get("units", "metric")
                        return 200, {"city": query["city"], "temperature": 22.5, "units": units, "conditions": "sunny", "humidity": 65}
                    if method == "GET" and path == "/weather/forecast":
                        if "city" not in query:
                            return 400, {"error": "city is required"}
                        days = int(query.get("days", 5))
                        forecast = [{"day": i, "temperature": 20.0 + i, "conditions": "partly cloudy"} for i in range(days)]
                        return 200, {"city": query["city"], "forecast": forecast}
                    if method == "GET" and path == "/weather/alerts":
                        if "city" not in query:
                            return 400, {"error": "city is required"}
                        return 404, {"error": "no alerts for city"}
                    return 404, {"error": "not found"}
        ''').strip(),
        "client_template": textwrap.dedent('''
            class WeatherClient:
                """Client for the Weather Service API."""
                def __init__(self, server):
                    self.server = server

                def get_current(self, city, units="metric"):
                    """GET /weather/current - get current weather."""
                    # TODO: implement
                    pass

                def get_forecast(self, city, days=5):
                    """GET /weather/forecast - get weather forecast."""
                    # TODO: implement
                    pass

                def get_alerts(self, city):
                    """GET /weather/alerts - get weather alerts. Returns None if no alerts."""
                    # TODO: implement
                    pass
        ''').strip(),
        "client_solution": textwrap.dedent('''
            class WeatherClient:
                """Client for the Weather Service API."""
                def __init__(self, server):
                    self.server = server

                def get_current(self, city, units="metric"):
                    """GET /weather/current - get current weather."""
                    status, data = self.server.handle("GET", "/weather/current", query={"city": city, "units": units})
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def get_forecast(self, city, days=5):
                    """GET /weather/forecast - get weather forecast."""
                    status, data = self.server.handle("GET", "/weather/forecast", query={"city": city, "days": days})
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def get_alerts(self, city):
                    """GET /weather/alerts - get weather alerts. Returns None if no alerts."""
                    status, data = self.server.handle("GET", "/weather/alerts", query={"city": city})
                    if status == 404:
                        return None
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data
        ''').strip(),
        "tests": textwrap.dedent('''
            from mock_server import MockServer
            from api_client import WeatherClient

            def test_get_current():
                server = MockServer()
                client = WeatherClient(server)
                result = client.get_current("London")
                assert result["city"] == "London"
                assert result["temperature"] == 22.5
                assert result["conditions"] == "sunny"
                assert result["units"] == "metric"

            def test_get_current_imperial():
                server = MockServer()
                client = WeatherClient(server)
                result = client.get_current("Tokyo", units="imperial")
                assert result["units"] == "imperial"
                assert result["city"] == "Tokyo"

            def test_get_forecast():
                server = MockServer()
                client = WeatherClient(server)
                result = client.get_forecast("Paris", days=3)
                assert result["city"] == "Paris"
                assert len(result["forecast"]) == 3
                assert result["forecast"][0]["day"] == 0
                assert result["forecast"][2]["day"] == 2

            def test_get_forecast_default_days():
                server = MockServer()
                client = WeatherClient(server)
                result = client.get_forecast("Berlin")
                assert len(result["forecast"]) == 5

            def test_get_alerts_none():
                server = MockServer()
                client = WeatherClient(server)
                result = client.get_alerts("Moscow")
                assert result is None

            def test_get_current_missing_city():
                server = MockServer()
                client = WeatherClient(server)
                try:
                    client.get_current("")
                    assert False, "should have raised"
                except Exception:
                    pass
        ''').strip(),
    },

    # ── Domain 3: Payment API ──
    "payment_api": {
        "spec": textwrap.dedent('''
            API Specification: Payment Processing API
            Base URL: http://localhost:8000

            Endpoints:
            1. POST /payments
               - Body: {"amount": float, "currency": str, "description": str}
               - Response 201: {"id": str, "amount": float, "currency": str, "description": str, "status": "pending"}
               - Response 400: {"error": "invalid amount"} or {"error": "currency required"}

            2. GET /payments/{payment_id}
               - Path param: payment_id (str)
               - Response 200: {"id": str, "amount": float, "currency": str, "description": str, "status": str}
               - Response 404: {"error": "payment not found"}

            3. POST /payments/{payment_id}/refund
               - Path param: payment_id (str)
               - Body: {"amount": float, optional}
               - Response 200: {"id": str, "payment_id": str, "refund_amount": float, "status": "refunded"}
               - Response 404: {"error": "payment not found"}

            4. GET /payments
               - Query params: status (str, optional), limit (int, optional, default=20)
               - Response 200: {"payments": [...], "total": int, "limit": int}
               - Response 400: {"error": "invalid limit"}
        ''').strip(),
        "mock_server": textwrap.dedent('''
            class MockServer:
                """Mock server simulating the Payment Processing API."""
                def __init__(self):
                    self._payments = {}
                    self._next_id = 1

                def handle(self, method, path, query=None, body=None):
                    query = query or {}
                    body = body or {}
                    if method == "POST" and path == "/payments":
                        if "amount" not in body or body["amount"] <= 0:
                            return 400, {"error": "invalid amount"}
                        if "currency" not in body:
                            return 400, {"error": "currency required"}
                        pid = f"pay_{self._next_id}"
                        self._next_id += 1
                        payment = {"id": pid, "amount": body["amount"], "currency": body["currency"],
                                   "description": body.get("description", ""), "status": "pending"}
                        self._payments[pid] = payment
                        return 201, payment
                    if method == "GET" and path == "/payments":
                        try:
                            limit = int(query.get("limit", 20))
                        except (ValueError, TypeError):
                            return 400, {"error": "invalid limit"}
                        payments = list(self._payments.values())
                        if "status" in query:
                            payments = [p for p in payments if p["status"] == query["status"]]
                        total = len(payments)
                        payments = payments[:limit]
                        return 200, {"payments": payments, "total": total, "limit": limit}
                    if method == "GET" and path.startswith("/payments/") and not path.endswith("/refund"):
                        pid = path.split("/")[-1]
                        if pid not in self._payments:
                            return 404, {"error": "payment not found"}
                        return 200, self._payments[pid]
                    if method == "POST" and path.endswith("/refund"):
                        pid = path.split("/")[2]
                        if pid not in self._payments:
                            return 404, {"error": "payment not found"}
                        refund_amount = body.get("amount", self._payments[pid]["amount"])
                        self._payments[pid]["status"] = "refunded"
                        return 200, {"id": f"ref_{pid}", "payment_id": pid, "refund_amount": refund_amount, "status": "refunded"}
                    return 404, {"error": "not found"}
        ''').strip(),
        "client_template": textwrap.dedent('''
            class PaymentClient:
                """Client for the Payment Processing API."""
                def __init__(self, server):
                    self.server = server

                def create_payment(self, amount, currency, description=""):
                    """POST /payments - create a payment."""
                    # TODO: implement
                    pass

                def get_payment(self, payment_id):
                    """GET /payments/{payment_id} - get payment details."""
                    # TODO: implement
                    pass

                def refund_payment(self, payment_id, amount=None):
                    """POST /payments/{payment_id}/refund - refund a payment."""
                    # TODO: implement
                    pass

                def list_payments(self, status=None, limit=20):
                    """GET /payments - list payments with optional status filter."""
                    # TODO: implement
                    pass
        ''').strip(),
        "client_solution": textwrap.dedent('''
            class PaymentClient:
                """Client for the Payment Processing API."""
                def __init__(self, server):
                    self.server = server

                def create_payment(self, amount, currency, description=""):
                    """POST /payments - create a payment."""
                    status, data = self.server.handle("POST", "/payments",
                                                       body={"amount": amount, "currency": currency, "description": description})
                    if status != 201:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def get_payment(self, payment_id):
                    """GET /payments/{payment_id} - get payment details."""
                    status, data = self.server.handle("GET", f"/payments/{payment_id}")
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def refund_payment(self, payment_id, amount=None):
                    """POST /payments/{payment_id}/refund - refund a payment."""
                    body = {}
                    if amount is not None:
                        body["amount"] = amount
                    status, data = self.server.handle("POST", f"/payments/{payment_id}/refund", body=body)
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def list_payments(self, status=None, limit=20):
                    """GET /payments - list payments with optional status filter."""
                    query = {"limit": limit}
                    if status is not None:
                        query["status"] = status
                    status_code, data = self.server.handle("GET", "/payments", query=query)
                    if status_code != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data
        ''').strip(),
        "tests": textwrap.dedent('''
            from mock_server import MockServer
            from api_client import PaymentClient

            def test_create_payment():
                server = MockServer()
                client = PaymentClient(server)
                result = client.create_payment(100.0, "USD", "Test payment")
                assert result["amount"] == 100.0
                assert result["currency"] == "USD"
                assert result["status"] == "pending"
                assert result["id"].startswith("pay_")

            def test_get_payment():
                server = MockServer()
                client = PaymentClient(server)
                created = client.create_payment(50.0, "EUR")
                fetched = client.get_payment(created["id"])
                assert fetched["amount"] == 50.0
                assert fetched["currency"] == "EUR"

            def test_refund_payment():
                server = MockServer()
                client = PaymentClient(server)
                created = client.create_payment(75.0, "GBP")
                refund = client.refund_payment(created["id"])
                assert refund["status"] == "refunded"
                assert refund["payment_id"] == created["id"]
                assert refund["refund_amount"] == 75.0

            def test_refund_partial():
                server = MockServer()
                client = PaymentClient(server)
                created = client.create_payment(100.0, "USD")
                refund = client.refund_payment(created["id"], amount=30.0)
                assert refund["refund_amount"] == 30.0

            def test_list_payments():
                server = MockServer()
                client = PaymentClient(server)
                client.create_payment(10.0, "USD")
                client.create_payment(20.0, "USD")
                result = client.list_payments()
                assert result["total"] == 2
                assert len(result["payments"]) == 2

            def test_list_payments_by_status():
                server = MockServer()
                client = PaymentClient(server)
                p1 = client.create_payment(10.0, "USD")
                client.create_payment(20.0, "USD")
                client.refund_payment(p1["id"])
                result = client.list_payments(status="refunded")
                assert result["total"] == 1
                assert result["payments"][0]["status"] == "refunded"

            def test_get_nonexistent_payment():
                server = MockServer()
                client = PaymentClient(server)
                try:
                    client.get_payment("pay_999")
                    assert False, "should have raised"
                except Exception as e:
                    assert "not found" in str(e).lower()
        ''').strip(),
    },

    # ── Domain 4: File Storage API ──
    "file_storage_api": {
        "spec": textwrap.dedent('''
            API Specification: File Storage API
            Base URL: http://localhost:8000

            Endpoints:
            1. POST /files
               - Body: {"filename": str, "content": str}
               - Response 201: {"id": str, "filename": str, "size": int, "created": true}
               - Response 400: {"error": "filename required"}

            2. GET /files/{file_id}
               - Path param: file_id (str)
               - Response 200: {"id": str, "filename": str, "content": str, "size": int}
               - Response 404: {"error": "file not found"}

            3. GET /files
               - Query params: page (int, optional, default=1), per_page (int, optional, default=10)
               - Response 200: {"files": [...], "page": int, "per_page": int, "total": int, "total_pages": int}
               - Response 400: {"error": "invalid page"}

            4. PUT /files/{file_id}
               - Path param: file_id (str)
               - Body: {"content": str}
               - Response 200: {"id": str, "filename": str, "size": int, "updated": true}
               - Response 404: {"error": "file not found"}

            5. DELETE /files/{file_id}
               - Path param: file_id (str)
               - Response 200: {"deleted": true, "id": str}
               - Response 404: {"error": "file not found"}

            6. GET /files/search
               - Query params: q (str, required)
               - Response 200: {"results": [{"id": str, "filename": str, "size": int}, ...], "count": int}
               - Response 400: {"error": "query required"}
        ''').strip(),
        "mock_server": textwrap.dedent('''
            class MockServer:
                """Mock server simulating the File Storage API."""
                def __init__(self):
                    self._files = {}
                    self._next_id = 1

                def handle(self, method, path, query=None, body=None):
                    query = query or {}
                    body = body or {}
                    if method == "POST" and path == "/files":
                        if "filename" not in body:
                            return 400, {"error": "filename required"}
                        fid = f"file_{self._next_id}"
                        self._next_id += 1
                        content = body.get("content", "")
                        f = {"id": fid, "filename": body["filename"], "content": content, "size": len(content)}
                        self._files[fid] = f
                        return 201, {"id": fid, "filename": body["filename"], "size": len(content), "created": True}
                    if method == "GET" and path == "/files/search":
                        if "q" not in query:
                            return 400, {"error": "query required"}
                        results = [{"id": fid, "filename": f["filename"], "size": f["size"]}
                                   for fid, f in self._files.items() if query["q"].lower() in f["filename"].lower()]
                        return 200, {"results": results, "count": len(results)}
                    if method == "GET" and path == "/files":
                        try:
                            page = int(query.get("page", 1))
                            per_page = int(query.get("per_page", 10))
                        except (ValueError, TypeError):
                            return 400, {"error": "invalid page"}
                        if page < 1:
                            return 400, {"error": "invalid page"}
                        all_files = list(self._files.values())
                        total = len(all_files)
                        total_pages = (total + per_page - 1) // per_page if per_page > 0 else 1
                        start = (page - 1) * per_page
                        end = start + per_page
                        return 200, {"files": all_files[start:end], "page": page, "per_page": per_page,
                                     "total": total, "total_pages": max(1, total_pages)}
                    if method == "GET" and path.startswith("/files/"):
                        fid = path.split("/")[-1]
                        if fid not in self._files:
                            return 404, {"error": "file not found"}
                        return 200, self._files[fid]
                    if method == "PUT" and path.startswith("/files/"):
                        fid = path.split("/")[-1]
                        if fid not in self._files:
                            return 404, {"error": "file not found"}
                        content = body.get("content", "")
                        self._files[fid]["content"] = content
                        self._files[fid]["size"] = len(content)
                        return 200, {"id": fid, "filename": self._files[fid]["filename"], "size": len(content), "updated": True}
                    if method == "DELETE" and path.startswith("/files/"):
                        fid = path.split("/")[-1]
                        if fid not in self._files:
                            return 404, {"error": "file not found"}
                        del self._files[fid]
                        return 200, {"deleted": True, "id": fid}
                    return 404, {"error": "not found"}
        ''').strip(),
        "client_template": textwrap.dedent('''
            class FileStorageClient:
                """Client for the File Storage API."""
                def __init__(self, server):
                    self.server = server

                def upload_file(self, filename, content=""):
                    """POST /files - upload a new file."""
                    # TODO: implement
                    pass

                def get_file(self, file_id):
                    """GET /files/{file_id} - get file content."""
                    # TODO: implement
                    pass

                def list_files(self, page=1, per_page=10):
                    """GET /files - list files with pagination."""
                    # TODO: implement
                    pass

                def update_file(self, file_id, content):
                    """PUT /files/{file_id} - update file content."""
                    # TODO: implement
                    pass

                def delete_file(self, file_id):
                    """DELETE /files/{file_id} - delete a file."""
                    # TODO: implement
                    pass

                def search_files(self, query):
                    """GET /files/search - search files by name."""
                    # TODO: implement
                    pass
        ''').strip(),
        "client_solution": textwrap.dedent('''
            class FileStorageClient:
                """Client for the File Storage API."""
                def __init__(self, server):
                    self.server = server

                def upload_file(self, filename, content=""):
                    """POST /files - upload a new file."""
                    status, data = self.server.handle("POST", "/files", body={"filename": filename, "content": content})
                    if status != 201:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def get_file(self, file_id):
                    """GET /files/{file_id} - get file content."""
                    status, data = self.server.handle("GET", f"/files/{file_id}")
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def list_files(self, page=1, per_page=10):
                    """GET /files - list files with pagination."""
                    status, data = self.server.handle("GET", "/files", query={"page": page, "per_page": per_page})
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def update_file(self, file_id, content):
                    """PUT /files/{file_id} - update file content."""
                    status, data = self.server.handle("PUT", f"/files/{file_id}", body={"content": content})
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def delete_file(self, file_id):
                    """DELETE /files/{file_id} - delete a file."""
                    status, data = self.server.handle("DELETE", f"/files/{file_id}")
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data

                def search_files(self, query):
                    """GET /files/search - search files by name."""
                    status, data = self.server.handle("GET", "/files/search", query={"q": query})
                    if status != 200:
                        raise Exception(data.get("error", "request failed"))
                    return data
        ''').strip(),
        "tests": textwrap.dedent('''
            from mock_server import MockServer
            from api_client import FileStorageClient

            def test_upload_and_get():
                server = MockServer()
                client = FileStorageClient(server)
                result = client.upload_file("test.txt", "hello world")
                assert result["filename"] == "test.txt"
                assert result["size"] == 11
                assert result["created"] == True
                fetched = client.get_file(result["id"])
                assert fetched["content"] == "hello world"
                assert fetched["filename"] == "test.txt"

            def test_list_files_pagination():
                server = MockServer()
                client = FileStorageClient(server)
                for i in range(15):
                    client.upload_file(f"file_{i}.txt", f"content {i}")
                result = client.list_files(page=1, per_page=10)
                assert result["total"] == 15
                assert len(result["files"]) == 10
                assert result["page"] == 1
                result2 = client.list_files(page=2, per_page=10)
                assert len(result2["files"]) == 5

            def test_update_file():
                server = MockServer()
                client = FileStorageClient(server)
                created = client.upload_file("doc.txt", "original")
                updated = client.update_file(created["id"], "updated content")
                assert updated["updated"] == True
                assert updated["size"] == 15
                fetched = client.get_file(created["id"])
                assert fetched["content"] == "updated content"

            def test_delete_file():
                server = MockServer()
                client = FileStorageClient(server)
                created = client.upload_file("temp.txt", "temp")
                result = client.delete_file(created["id"])
                assert result["deleted"] == True
                try:
                    client.get_file(created["id"])
                    assert False, "should have raised"
                except Exception as e:
                    assert "not found" in str(e).lower()

            def test_search_files():
                server = MockServer()
                client = FileStorageClient(server)
                client.upload_file("report.pdf", "data")
                client.upload_file("notes.txt", "data")
                client.upload_file("report_final.pdf", "data")
                result = client.search_files("report")
                assert result["count"] == 2

            def test_get_nonexistent():
                server = MockServer()
                client = FileStorageClient(server)
                try:
                    client.get_file("file_999")
                    assert False, "should have raised"
                except Exception as e:
                    assert "not found" in str(e).lower()
        ''').strip(),
    },
}


# ── Distractor code (irrelevant modules to test if model can focus) ──

DISTRACTORS = [
    textwrap.dedent('''
        def format_currency(amount, currency="USD"):
            """Format a currency amount (not relevant to the task)."""
            symbols = {"USD": "$", "EUR": "\\u20ac", "GBP": "\\u00a3"}
            symbol = symbols.get(currency, "")
            return f"{symbol}{amount:,.2f}"

        def parse_amount(text):
            """Parse a currency string (not relevant to the task)."""
            import re
            match = re.match(r'[\\$\\u20ac\\u00a3]?([\\d,]+\\.?\\d*)', text.strip())
            if not match:
                return 0.0
            return float(match.group(1).replace(",", ""))
    ''').strip(),
    textwrap.dedent('''
        class RateLimiter:
            """Simple rate limiter (not relevant to the task)."""
            def __init__(self, max_calls, window_seconds=60):
                self.max_calls = max_calls
                self.window = window_seconds
                self._calls = []

            def allow(self):
                import time
                now = time.time()
                self._calls = [t for t in self._calls if now - t < self.window]
                if len(self._calls) >= self.max_calls:
                    return False
                self._calls.append(now)
                return True
    ''').strip(),
    textwrap.dedent('''
        def retry(func, max_attempts=3, delay=1.0):
            """Retry a function with backoff (not relevant to the task)."""
            import time
            for attempt in range(max_attempts):
                try:
                    return func()
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
    ''').strip(),
]


@register_env
class APIClientEnv(AgenticEnv):
    name = "api_client"
    skill = "Implementing an API client from a specification"
    difficulty_tiers = ["easy", "medium", "hard"]

    def gen_params(self, rng, difficulty="medium"):
        domain_name = rng.choice(list(DOMAINS.keys()))
        domain = DOMAINS[domain_name]
        n_endpoints = len(domain["tests"].split("def test_")) - 1
        n_distractors = {"easy": 0, "medium": 1, "hard": 2}[difficulty]
        distractors = rng.sample(DISTRACTORS, n_distractors) if n_distractors else []
        return {
            "domain": domain_name,
            "difficulty": difficulty,
            "n_endpoints": n_endpoints,
            "n_distractors": n_distractors,
            "distractor_indices": [DISTRACTORS.index(d) for d in distractors] if distractors else [],
            "seed": rng.randint(0, 999999),
        }

    def gen_codebase(self, params, rng):
        domain = DOMAINS[params["domain"]]
        codebase = {
            "mock_server.py": domain["mock_server"],
            "api_client.py": domain["client_template"],
        }
        for idx in params.get("distractor_indices", []):
            codebase[f"util_{idx}.py"] = DISTRACTORS[idx]
        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        lines = []
        lines.append("You are a software engineer implementing an API client.")
        lines.append("")
        lines.append("You are given an API specification and a mock server. Your task is to:")
        lines.append("1. Read the API specification carefully")
        lines.append("2. Study the mock server to understand the request/response format")
        lines.append("3. Implement the client class in `api_client.py` with all methods")
        lines.append("4. Ensure proper error handling (raise exceptions on non-2xx responses)")
        lines.append("")
        lines.append("=== API SPECIFICATION ===")
        lines.append(domain["spec"])
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")
        lines.append("=== TEST CASES (your client must pass these) ===")
        lines.append("```python")
        lines.append(domain["tests"])
        lines.append("```")
        lines.append("")
        lines.append("Implement the client methods in `api_client.py`. The mock server's `handle` method")
        lines.append("takes (method, path, query=None, body=None) and returns (status_code, response_data).")
        lines.append("")
        lines.append("Provide your solution in the following format:")
        lines.append("<reasoning>")
        lines.append("...analyze the spec, explain each endpoint, describe your implementation strategy...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append("```python:api_client.py")
        lines.append("# the complete client implementation")
        lines.append("```")
        lines.append("</answer>")
        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        return {"api_client.py": domain["client_solution"]}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        spec_lines = domain["spec"].strip().split("\n")
        client_code = domain["client_solution"]

        reasoning_lines = [
            "Let me carefully analyze the API specification and implement the client.",
            "",
            "First, I'll read the API specification to understand each endpoint:",
            "",
        ]

        current_endpoint = None
        for line in spec_lines:
            if line.strip().startswith("Endpoints:") or line.strip().startswith("API Specification:"):
                reasoning_lines.append(f"  {line.strip()}")
            elif line.strip() and line.strip()[0].isdigit() and "." in line.strip()[:3]:
                current_endpoint = line.strip()
                reasoning_lines.append(f"\nEndpoint: {line.strip()}")
            elif line.strip().startswith("- ") or line.strip().startswith("Response"):
                reasoning_lines.append(f"  {line.strip()}")

        reasoning_lines.extend([
            "",
            "Now let me study the mock server to understand the interface:",
            "The mock server's `handle` method takes (method, path, query=None, body=None)",
            "and returns (status_code, response_data) as a tuple.",
            "",
            "Key observations from the mock server:",
            "- I need to construct the correct path for each endpoint",
            "- Query parameters are passed as a dict to the `query` argument",
            "- Request body is passed as a dict to the `body` argument",
            "- I must check the status code and raise an exception on errors",
            "- The response data is a dict that I should return or extract fields from",
            "",
            "Now let me implement each method in the client class:",
            "",
        ])

        # Analyze each method in the solution
        for method_line in client_code.split("\n"):
            stripped = method_line.strip()
            if stripped.startswith("def ") and "def __init__" not in stripped:
                method_name = stripped.split("(")[0].replace("def ", "")
                reasoning_lines.append(f"Method `{method_name}`:")
                if "GET" in stripped or "get" in method_name.lower():
                    reasoning_lines.append("  - This is a GET request, so I pass query parameters")
                elif "POST" in stripped or "POST" in method_line or "create" in method_name.lower() or "upload" in method_name.lower():
                    reasoning_lines.append("  - This is a POST request, so I pass a body dict")
                elif "PUT" in stripped or "update" in method_name.lower():
                    reasoning_lines.append("  - This is a PUT request, so I pass a body dict")
                elif "DELETE" in stripped or "delete" in method_name.lower():
                    reasoning_lines.append("  - This is a DELETE request")
                reasoning_lines.append("  - I check the status code and raise Exception with the error message on failure")
                reasoning_lines.append("  - I return the response data on success")
                reasoning_lines.append("")

        reasoning_lines.extend([
            "Let me also verify the error handling:",
            "- For 4xx responses, the server returns an 'error' field in the response data",
            "- I extract this error message and raise it as an exception",
            "- For the weather alerts endpoint, 404 means 'no alerts' which should return None, not raise",
            "- For pagination endpoints, I need to pass page and per_page as query parameters",
            "",
            "I also need to make sure path parameters are correctly interpolated into the URL path.",
            "For example, /users/{user_id} becomes f\"/users/{user_id}\" with the actual value.",
            "",
            "The implementation is complete. Let me verify each method handles all the response codes",
            "specified in the API spec, and that the return values match what the tests expect.",
        ])

        return "\n".join(reasoning_lines)

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]

        answer = extract_answer(response)
        code_changes = parse_code_blocks(answer)

        if not code_changes:
            return 0.0, {
                "reason": "no code blocks found in answer",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        modified_codebase = apply_code_changes(codebase, code_changes)

        results = _run_tests(modified_codebase, domain["tests"], timeout=15.0)
        score, breakdown = compute_test_score(results)

        breakdown["domain"] = params["domain"]
        breakdown["difficulty"] = params["difficulty"]
        breakdown["has_reasoning"] = bool(extract_reasoning(response))
        breakdown["files_changed"] = list(code_changes.keys())
        breakdown["changed_target"] = "api_client.py" in code_changes

        # Partial credit for changing the right file even if tests fail
        if score == 0.0 and breakdown["changed_target"]:
            from ..graders import code_similarity
            sim = code_similarity(
                code_changes.get("api_client.py", ""),
                domain["client_solution"],
            )
            if sim > 0.5:
                score = 0.2 * sim
                breakdown["partial_credit"] = f"client is {sim:.0%} similar to correct, awarded partial credit"

        return score, breakdown
