"""
Environment 12: Data Transformation Pipeline

Skill: Implementing data transformation pipelines from specifications.

The model is given input data + a transformation spec and must implement
a function that transforms the data. The codebase contains a stub function
and supporting modules. The model must read the spec, understand the
expected transformation, and implement the function correctly.

Domains:
- CSV row filtering + mapping
- JSON nested key extraction
- Log parsing into structured records
- Sales data aggregation

Difficulty scaling:
- easy: single transformation, clear spec
- medium: chained transformations (2-3 steps)
- hard: conditional transformations with multiple branches
"""
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, extract_reasoning, parse_code_blocks, apply_code_changes,
    run_tests, compute_test_score, code_similarity,
)


# ── Domain definitions ──
# Each domain has variants at different difficulty levels.
# Each variant has: spec, stub_code, solution_code, test_code, input_data

DOMAINS = {

    # ── Domain 1: CSV row filtering + mapping ──
    "csv_filter_map": {
        "module": "csv_transform",
        "easy": {
            "spec": (
                "Implement `transform_rows(rows)` that takes a list of CSV rows "
                "(each row is a list of strings) and returns a list of rows where "
                "the first column equals 'active'. Keep all columns unchanged."
            ),
            "stub": textwrap.dedent('''
                def transform_rows(rows):
                    """Filter rows where the first column is 'active'."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def transform_rows(rows):
                    """Filter rows where the first column is 'active'."""
                    return [row for row in rows if len(row) > 0 and row[0] == 'active']
            ''').strip(),
            "test": textwrap.dedent('''
                from csv_transform import transform_rows
                def test_basic_filter():
                    rows = [["active", "Alice", "30"], ["inactive", "Bob", "25"], ["active", "Carol", "40"]]
                    result = transform_rows(rows)
                    assert result == [["active", "Alice", "30"], ["active", "Carol", "40"]]
                def test_all_active():
                    rows = [["active", "A"], ["active", "B"]]
                    assert transform_rows(rows) == [["active", "A"], ["active", "B"]]
                def test_none_active():
                    rows = [["inactive", "X"], ["pending", "Y"]]
                    assert transform_rows(rows) == []
                def test_empty():
                    assert transform_rows([]) == []
                def test_empty_row():
                    assert transform_rows([[]]) == []
            ''').strip(),
        },
        "medium": {
            "spec": (
                "Implement `transform_rows(rows)` that takes a list of CSV rows "
                "(each row is a list of strings with columns: status, name, age, score) "
                "and returns a list of [name, score] for rows where status is 'active' "
                "AND age (as int) is >= 18. Convert score to float."
            ),
            "stub": textwrap.dedent('''
                def transform_rows(rows):
                    """Filter active adults, return [name, score_as_float]."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def transform_rows(rows):
                    """Filter active adults, return [name, score_as_float]."""
                    result = []
                    for row in rows:
                        if len(row) < 4:
                            continue
                        status, name, age, score = row[0], row[1], row[2], row[3]
                        if status == 'active' and int(age) >= 18:
                            result.append([name, float(score)])
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from csv_transform import transform_rows
                def test_basic():
                    rows = [["active", "Alice", "30", "95.5"], ["inactive", "Bob", "25", "80.0"], ["active", "Carol", "40", "88.3"]]
                    assert transform_rows(rows) == [["Alice", 95.5], ["Carol", 88.3]]
                def test_minor_excluded():
                    rows = [["active", "Teen", "16", "90.0"], ["active", "Adult", "21", "85.0"]]
                    assert transform_rows(rows) == [["Adult", 85.0]]
                def test_empty():
                    assert transform_rows([]) == []
                def test_all_filtered():
                    rows = [["inactive", "X", "30", "50.0"], ["active", "Y", "15", "60.0"]]
                    assert transform_rows(rows) == []
                def test_score_conversion():
                    rows = [["active", "Z", "20", "100"]]
                    assert transform_rows(rows) == [["Z", 100.0]]
            ''').strip(),
        },
        "hard": {
            "spec": (
                "Implement `transform_rows(rows)` that takes a list of CSV rows "
                "(columns: status, name, age, score, category) and returns a list of "
                "[name, grade] where: status is 'active' OR category is 'premium', "
                "age >= 18, and grade is 'A' if score >= 90, 'B' if score >= 80, "
                "'C' if score >= 70, 'D' otherwise. Skip rows with missing columns."
            ),
            "stub": textwrap.dedent('''
                def transform_rows(rows):
                    """Complex filter + grade assignment."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def transform_rows(rows):
                    """Complex filter + grade assignment."""
                    result = []
                    for row in rows:
                        if len(row) < 5:
                            continue
                        status, name, age, score, category = row[0], row[1], row[2], row[3], row[4]
                        if not (status == 'active' or category == 'premium'):
                            continue
                        if int(age) < 18:
                            continue
                        score_val = float(score)
                        if score_val >= 90:
                            grade = 'A'
                        elif score_val >= 80:
                            grade = 'B'
                        elif score_val >= 70:
                            grade = 'C'
                        else:
                            grade = 'D'
                        result.append([name, grade])
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from csv_transform import transform_rows
                def test_active_high_score():
                    rows = [["active", "Alice", "30", "95", "basic"]]
                    assert transform_rows(rows) == [["Alice", "A"]]
                def test_premium_low_age():
                    rows = [["inactive", "Bob", "15", "90", "premium"]]
                    assert transform_rows(rows) == []
                def test_premium_adult():
                    rows = [["inactive", "Carol", "25", "85", "premium"]]
                    assert transform_rows(rows) == [["Carol", "B"]]
                def test_grade_boundaries():
                    rows = [
                        ["active", "A", "20", "90", "x"],
                        ["active", "B", "20", "89", "x"],
                        ["active", "C", "20", "80", "x"],
                        ["active", "D", "20", "79", "x"],
                        ["active", "E", "20", "70", "x"],
                        ["active", "F", "20", "69", "x"],
                    ]
                    assert transform_rows(rows) == [["A","A"],["B","B"],["C","B"],["D","C"],["E","C"],["F","D"]]
                def test_missing_columns():
                    rows = [["active", "X", "20"], ["active", "Y", "20", "85", "basic"]]
                    assert transform_rows(rows) == [["Y", "B"]]
                def test_empty():
                    assert transform_rows([]) == []
            ''').strip(),
        },
    },

    # ── Domain 2: JSON nested key extraction ──
    "json_extract": {
        "module": "json_transform",
        "easy": {
            "spec": (
                "Implement `extract_values(data, key)` that takes a list of dicts "
                "and a key string, and returns a list of values for that key from "
                "each dict. If a dict doesn't have the key, skip it."
            ),
            "stub": textwrap.dedent('''
                def extract_values(data, key):
                    """Extract values for a key from a list of dicts."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def extract_values(data, key):
                    """Extract values for a key from a list of dicts."""
                    return [d[key] for d in data if key in d]
            ''').strip(),
            "test": textwrap.dedent('''
                from json_transform import extract_values
                def test_basic():
                    data = [{"name": "Alice"}, {"name": "Bob"}, {"age": 30}]
                    assert extract_values(data, "name") == ["Alice", "Bob"]
                def test_all_have_key():
                    data = [{"x": 1}, {"x": 2}, {"x": 3}]
                    assert extract_values(data, "x") == [1, 2, 3]
                def test_none_have_key():
                    data = [{"a": 1}, {"b": 2}]
                    assert extract_values(data, "z") == []
                def test_empty():
                    assert extract_values([], "key") == []
            ''').strip(),
        },
        "medium": {
            "spec": (
                "Implement `extract_nested(data, path)` that takes a list of dicts "
                "and a dot-separated path string (e.g. 'user.address.city'). "
                "Returns a list of values found at that path in each dict. "
                "If any intermediate key is missing, skip that dict."
            ),
            "stub": textwrap.dedent('''
                def extract_nested(data, path):
                    """Extract values at a dot-separated path from list of dicts."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def extract_nested(data, path):
                    """Extract values at a dot-separated path from list of dicts."""
                    keys = path.split('.')
                    result = []
                    for item in data:
                        current = item
                        found = True
                        for key in keys:
                            if isinstance(current, dict) and key in current:
                                current = current[key]
                            else:
                                found = False
                                break
                        if found:
                            result.append(current)
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from json_transform import extract_nested
                def test_single_level():
                    data = [{"name": "A"}, {"name": "B"}]
                    assert extract_nested(data, "name") == ["A", "B"]
                def test_two_levels():
                    data = [{"user": {"city": "NYC"}}, {"user": {"city": "LA"}}]
                    assert extract_nested(data, "user.city") == ["NYC", "LA"]
                def test_three_levels():
                    data = [{"a": {"b": {"c": 1}}}, {"a": {"b": {"c": 2}}}]
                    assert extract_nested(data, "a.b.c") == [1, 2]
                def test_missing_intermediate():
                    data = [{"a": {"b": 1}}, {"a": {}}, {"x": 1}]
                    assert extract_nested(data, "a.b") == [1]
                def test_empty():
                    assert extract_nested([], "a.b") == []
            ''').strip(),
        },
        "hard": {
            "spec": (
                "Implement `extract_nested(data, path, default=None)` that takes a "
                "list of dicts, a dot-separated path, and a default value. Returns a "
                "list of values at that path. If a key is missing, use the default. "
                "If an intermediate value is a list, iterate into each element. "
                "If the final key maps to a list, flatten all results into one list."
            ),
            "stub": textwrap.dedent('''
                def extract_nested(data, path, default=None):
                    """Extract values at path with defaults and list flattening."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def extract_nested(data, path, default=None):
                    """Extract values at path with defaults and list flattening."""
                    keys = path.split('.')
                    result = []
                    for item in data:
                        current = item
                        for key in keys:
                            if isinstance(current, dict) and key in current:
                                current = current[key]
                            else:
                                current = default
                                break
                        if isinstance(current, list):
                            result.extend(current)
                        else:
                            result.append(current)
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from json_transform import extract_nested
                def test_basic():
                    data = [{"a": {"b": 1}}, {"a": {"b": 2}}]
                    assert extract_nested(data, "a.b") == [1, 2]
                def test_default():
                    data = [{"a": {"b": 1}}, {"a": {}}]
                    assert extract_nested(data, "a.b", default=0) == [1, 0]
                def test_list_flatten():
                    data = [{"tags": ["x", "y"]}, {"tags": ["z"]}]
                    assert extract_nested(data, "tags") == ["x", "y", "z"]
                def test_missing_with_default():
                    data = [{"x": 1}, {"y": 2}]
                    assert extract_nested(data, "z", default="N/A") == ["N/A", "N/A"]
                def test_empty():
                    assert extract_nested([], "a.b") == []
                def test_mixed_list_and_scalar():
                    data = [{"v": [1, 2]}, {"v": 3}]
                    assert extract_nested(data, "v") == [1, 2, 3]
            ''').strip(),
        },
    },

    # ── Domain 3: Log parsing into structured records ──
    "log_parse": {
        "module": "log_transform",
        "easy": {
            "spec": (
                "Implement `parse_logs(lines)` that takes a list of log strings "
                "in format 'TIMESTAMP LEVEL message' and returns a list of dicts "
                "with keys 'timestamp', 'level', 'message'. Skip lines that don't "
                "match the format."
            ),
            "stub": textwrap.dedent('''
                def parse_logs(lines):
                    """Parse log lines into structured records."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def parse_logs(lines):
                    """Parse log lines into structured records."""
                    result = []
                    for line in lines:
                        parts = line.split(' ', 2)
                        if len(parts) < 3:
                            continue
                        result.append({
                            'timestamp': parts[0],
                            'level': parts[1],
                            'message': parts[2],
                        })
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from log_transform import parse_logs
                def test_basic():
                    lines = ["2024-01-01 INFO Server started", "2024-01-02 ERROR Crashed"]
                    result = parse_logs(lines)
                    assert result == [
                        {"timestamp": "2024-01-01", "level": "INFO", "message": "Server started"},
                        {"timestamp": "2024-01-02", "level": "ERROR", "message": "Crashed"},
                    ]
                def test_skip_invalid():
                    lines = ["invalid line", "2024-01-01 INFO OK"]
                    result = parse_logs(lines)
                    assert result == [{"timestamp": "2024-01-01", "level": "INFO", "message": "OK"}]
                def test_empty():
                    assert parse_logs([]) == []
                def test_message_with_spaces():
                    lines = ["2024-01-01 WARN disk almost full"]
                    result = parse_logs(lines)
                    assert result[0]["message"] == "disk almost full"
            ''').strip(),
        },
        "medium": {
            "spec": (
                "Implement `parse_logs(lines)` that parses log lines in format "
                "'[TIMESTAMP] LEVEL [COMPONENT] message' and returns dicts with keys "
                "'timestamp', 'level', 'component', 'message'. Lines without brackets "
                "should have component set to 'unknown'. Skip completely malformed lines."
            ),
            "stub": textwrap.dedent('''
                def parse_logs(lines):
                    """Parse structured log lines with component field."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                import re
                def parse_logs(lines):
                    """Parse structured log lines with component field."""
                    result = []
                    for line in lines:
                        m = re.match(r'\\[(\\S+)\\]\\s+(\\w+)\\s+(?:\\[(\\S+)\\]\\s+)?(.*)', line)
                        if not m:
                            continue
                        result.append({
                            'timestamp': m.group(1),
                            'level': m.group(2),
                            'component': m.group(3) or 'unknown',
                            'message': m.group(4),
                        })
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from log_transform import parse_logs
                def test_with_component():
                    lines = ["[2024-01-01] ERROR [DB] connection failed"]
                    result = parse_logs(lines)
                    assert result == [{"timestamp": "2024-01-01", "level": "ERROR", "component": "DB", "message": "connection failed"}]
                def test_without_component():
                    lines = ["[2024-01-01] INFO server started"]
                    result = parse_logs(lines)
                    assert result == [{"timestamp": "2024-01-01", "level": "INFO", "component": "unknown", "message": "server started"}]
                def test_mixed():
                    lines = ["[T1] INFO [A] ok", "[T2] WARN no component", "garbage"]
                    result = parse_logs(lines)
                    assert len(result) == 2
                    assert result[0]["component"] == "A"
                    assert result[1]["component"] == "unknown"
                def test_empty():
                    assert parse_logs([]) == []
            ''').strip(),
        },
        "hard": {
            "spec": (
                "Implement `parse_logs(lines)` that parses log lines and returns "
                "structured records. Format: '[TIMESTAMP] LEVEL [COMPONENT] message'. "
                "Additionally: extract key=value pairs from the message into a 'fields' "
                "dict. If level is 'ERROR' or 'FATAL', add 'severity': 'high'. If level "
                "is 'WARN', add 'severity': 'medium'. Otherwise 'severity': 'low'. "
                "Skip malformed lines."
            ),
            "stub": textwrap.dedent('''
                def parse_logs(lines):
                    """Parse logs with field extraction and severity."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                import re
                def parse_logs(lines):
                    """Parse logs with field extraction and severity."""
                    severity_map = {'ERROR': 'high', 'FATAL': 'high', 'WARN': 'medium'}
                    result = []
                    for line in lines:
                        m = re.match(r'\\[(\\S+)\\]\\s+(\\w+)\\s+(?:\\[(\\S+)\\]\\s+)?(.*)', line)
                        if not m:
                            continue
                        timestamp, level, component, message = m.group(1), m.group(2), m.group(3) or 'unknown', m.group(4)
                        fields = {}
                        for fm in re.finditer(r'(\\w+)=(\\S+)', message):
                            fields[fm.group(1)] = fm.group(2)
                        result.append({
                            'timestamp': timestamp,
                            'level': level,
                            'component': component,
                            'message': message,
                            'fields': fields,
                            'severity': severity_map.get(level, 'low'),
                        })
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from log_transform import parse_logs
                def test_error_severity():
                    lines = ["[T1] ERROR [DB] connection failed code=500"]
                    r = parse_logs(lines)[0]
                    assert r["severity"] == "high"
                    assert r["fields"] == {"code": "500"}
                def test_warn_severity():
                    lines = ["[T1] WARN disk usage=90%"]
                    r = parse_logs(lines)[0]
                    assert r["severity"] == "medium"
                    assert r["fields"]["usage"] == "90%"
                def test_info_severity():
                    lines = ["[T1] INFO server started port=8080"]
                    r = parse_logs(lines)[0]
                    assert r["severity"] == "low"
                    assert r["fields"]["port"] == "8080"
                def test_no_fields():
                    lines = ["[T1] INFO [API] simple message"]
                    r = parse_logs(lines)[0]
                    assert r["fields"] == {}
                def test_skip_malformed():
                    lines = ["garbage", "[T1] ERROR [DB] fail code=42"]
                    result = parse_logs(lines)
                    assert len(result) == 1
                def test_empty():
                    assert parse_logs([]) == []
            ''').strip(),
        },
    },

    # ── Domain 4: Sales data aggregation ──
    "sales_agg": {
        "module": "sales_transform",
        "easy": {
            "spec": (
                "Implement `aggregate_sales(records)` that takes a list of dicts "
                "with keys 'product' and 'amount' (float) and returns a dict mapping "
                "product name to total amount."
            ),
            "stub": textwrap.dedent('''
                def aggregate_sales(records):
                    """Aggregate sales by product."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def aggregate_sales(records):
                    """Aggregate sales by product."""
                    totals = {}
                    for r in records:
                        product = r['product']
                        totals[product] = totals.get(product, 0) + r['amount']
                    return totals
            ''').strip(),
            "test": textwrap.dedent('''
                from sales_transform import aggregate_sales
                def test_basic():
                    records = [{"product": "A", "amount": 10.0}, {"product": "A", "amount": 5.0}, {"product": "B", "amount": 20.0}]
                    assert aggregate_sales(records) == {"A": 15.0, "B": 20.0}
                def test_single():
                    assert aggregate_sales([{"product": "X", "amount": 100.0}]) == {"X": 100.0}
                def test_empty():
                    assert aggregate_sales([]) == {}
                def test_same_product():
                    records = [{"product": "Z", "amount": 1.0}, {"product": "Z", "amount": 2.0}, {"product": "Z", "amount": 3.0}]
                    assert aggregate_sales(records) == {"Z": 6.0}
            ''').strip(),
        },
        "medium": {
            "spec": (
                "Implement `aggregate_sales(records)` that takes a list of dicts "
                "with keys 'product', 'amount', 'region' and returns a dict mapping "
                "region -> product -> total amount. Missing regions or products should "
                "not appear in the output."
            ),
            "stub": textwrap.dedent('''
                def aggregate_sales(records):
                    """Aggregate sales by region then product."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def aggregate_sales(records):
                    """Aggregate sales by region then product."""
                    result = {}
                    for r in records:
                        region = r['region']
                        product = r['product']
                        if region not in result:
                            result[region] = {}
                        result[region][product] = result[region].get(product, 0) + r['amount']
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from sales_transform import aggregate_sales
                def test_basic():
                    records = [
                        {"product": "A", "amount": 10.0, "region": "US"},
                        {"product": "A", "amount": 5.0, "region": "US"},
                        {"product": "B", "amount": 20.0, "region": "EU"},
                    ]
                    assert aggregate_sales(records) == {"US": {"A": 15.0}, "EU": {"B": 20.0}}
                def test_multi_region():
                    records = [
                        {"product": "X", "amount": 1.0, "region": "US"},
                        {"product": "X", "amount": 2.0, "region": "EU"},
                    ]
                    assert aggregate_sales(records) == {"US": {"X": 1.0}, "EU": {"X": 2.0}}
                def test_empty():
                    assert aggregate_sales([]) == {}
                def test_multi_product_region():
                    records = [
                        {"product": "A", "amount": 10.0, "region": "US"},
                        {"product": "B", "amount": 20.0, "region": "US"},
                    ]
                    assert aggregate_sales(records) == {"US": {"A": 10.0, "B": 20.0}}
            ''').strip(),
        },
        "hard": {
            "spec": (
                "Implement `aggregate_sales(records)` that takes a list of dicts "
                "with keys 'product', 'amount', 'region', 'quarter' and returns a dict "
                "mapping region -> quarter -> {total, count, avg, products}. "
                "'total' is sum of amounts, 'count' is number of records, 'avg' is "
                "total/count (0 if count is 0), 'products' is a dict of product -> total. "
                "Skip records with missing keys."
            ),
            "stub": textwrap.dedent('''
                def aggregate_sales(records):
                    """Aggregate sales by region, quarter with stats."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def aggregate_sales(records):
                    """Aggregate sales by region, quarter with stats."""
                    result = {}
                    for r in records:
                        for key in ('product', 'amount', 'region', 'quarter'):
                            if key not in r:
                                continue
                        if not all(k in r for k in ('product', 'amount', 'region', 'quarter')):
                            continue
                        region = r['region']
                        quarter = r['quarter']
                        if region not in result:
                            result[region] = {}
                        if quarter not in result[region]:
                            result[region][quarter] = {'total': 0, 'count': 0, 'products': {}}
                        stats = result[region][quarter]
                        stats['total'] += r['amount']
                        stats['count'] += 1
                        stats['products'][r['product']] = stats['products'].get(r['product'], 0) + r['amount']
                    for region in result:
                        for quarter in result[region]:
                            stats = result[region][quarter]
                            stats['avg'] = stats['total'] / stats['count'] if stats['count'] > 0 else 0
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from sales_transform import aggregate_sales
                def test_basic():
                    records = [{"product": "A", "amount": 10.0, "region": "US", "quarter": "Q1"}]
                    r = aggregate_sales(records)["US"]["Q1"]
                    assert r["total"] == 10.0
                    assert r["count"] == 1
                    assert r["avg"] == 10.0
                    assert r["products"] == {"A": 10.0}
                def test_multi():
                    records = [
                        {"product": "A", "amount": 10.0, "region": "US", "quarter": "Q1"},
                        {"product": "B", "amount": 20.0, "region": "US", "quarter": "Q1"},
                        {"product": "A", "amount": 30.0, "region": "EU", "quarter": "Q2"},
                    ]
                    us = aggregate_sales(records)["US"]["Q1"]
                    assert us["total"] == 30.0
                    assert us["count"] == 2
                    assert us["avg"] == 15.0
                    assert us["products"] == {"A": 10.0, "B": 20.0}
                def test_skip_missing():
                    records = [{"product": "A", "amount": 10.0, "region": "US"}, {"product": "B", "amount": 5.0, "region": "US", "quarter": "Q1"}]
                    r = aggregate_sales(records)
                    assert r == {"US": {"Q1": {"total": 5.0, "count": 1, "avg": 5.0, "products": {"B": 5.0}}}}
                def test_empty():
                    assert aggregate_sales([]) == {}
            ''').strip(),
        },
    },
}


# ── Distractor code (irrelevant functions to test if model can focus) ─

DISTRACTORS = [
    textwrap.dedent('''
        def format_currency(amount, symbol="$"):
            """Format a number as currency (not relevant to the task)."""
            return f"{symbol}{amount:,.2f}"

        def validate_email(email):
            """Simple email validation (not relevant to the task)."""
            return "@" in email and "." in email.split("@")[-1]
    ''').strip(),
    textwrap.dedent('''
        def slugify(text):
            """Convert text to URL slug (not relevant to the task)."""
            import re
            text = text.lower().strip()
            text = re.sub(r'[^a-z0-9\\s-]', '', text)
            return re.sub(r'[-\\s]+', '-', text)

        def truncate_text(text, max_len, suffix="..."):
            """Truncate text (not relevant to the task)."""
            if len(text) <= max_len:
                return text
            return text[:max_len - len(suffix)] + suffix
    ''').strip(),
    textwrap.dedent('''
        def deep_merge(a, b):
            """Deep merge two dicts (not relevant to the task)."""
            result = dict(a)
            for key, val in b.items():
                if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                    result[key] = deep_merge(result[key], val)
                else:
                    result[key] = val
            return result

        def flatten_dict(d, prefix=""):
            """Flatten nested dict (not relevant to the task)."""
            items = {}
            for key, val in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(val, dict):
                    items.update(flatten_dict(val, full_key))
                else:
                    items[full_key] = val
            return items
    ''').strip(),
]


@register_env
class DataTransformEnv(AgenticEnv):
    name = "data_transform"
    skill = "Implementing data transformation pipelines from specifications"
    difficulty_tiers = ["easy", "medium", "hard"]

    def gen_params(self, rng, difficulty="medium"):
        domain_name = rng.choice(list(DOMAINS.keys()))
        n_distractors = {"easy": 0, "medium": 1, "hard": 2}[difficulty]
        distractors = rng.sample(DISTRACTORS, n_distractors) if n_distractors else []
        return {
            "domain": domain_name,
            "difficulty": difficulty,
            "n_distractors": n_distractors,
            "distractor_indices": [DISTRACTORS.index(d) for d in distractors] if distractors else [],
            "seed": rng.randint(0, 999999),
        }

    def gen_codebase(self, params, rng):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        module = domain["module"]

        codebase = {f"{module}.py": variant["stub"]}

        for idx in params.get("distractor_indices", []):
            codebase[f"helper_{idx}.py"] = DISTRACTORS[idx]

        codebase["test_spec.py"] = variant["test"]
        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        module = domain["module"]

        lines = []
        lines.append("You are a software engineer implementing a data transformation pipeline.")
        lines.append("")
        lines.append("Your task is to implement the function described in the specification below.")
        lines.append("The function stub is already in the codebase. You need to fill in the implementation.")
        lines.append("")
        lines.append("=== SPECIFICATION ===")
        lines.append(variant["spec"])
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")
        lines.append("=== TEST CASES ===")
        lines.append("The test file `test_spec.py` contains test cases your implementation must pass.")
        lines.append("")
        lines.append("Provide your implementation in the following format:")
        lines.append("<reasoning>")
        lines.append("...analyze the spec, break it into steps, handle edge cases...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append(f"```python:{module}.py")
        lines.append("# the complete implementation")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        module = domain["module"]
        return {f"{module}.py": variant["solution"]}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        module = domain["module"]

        reasoning = textwrap.dedent(f"""
            Let me carefully analyze the specification and implement the transformation.

            SPECIFICATION ANALYSIS:
            {variant['spec']}

            Let me break this down into steps:

            1. First, I need to understand the input format. The function receives data
               in a specific structure — I need to parse each element correctly.

            2. Next, I need to understand what transformation is required:
               - What filtering conditions apply?
               - What mapping/projection is needed on the output?
               - Are there type conversions involved?

            3. I need to handle edge cases:
               - Empty input lists
               - Missing keys or columns
               - Malformed entries that should be skipped
               - Type conversion errors

            Let me look at the test cases to verify my understanding:
            The tests check basic functionality, edge cases (empty input), and
            boundary conditions. This tells me I need to be careful about:
            - Not crashing on empty input
            - Correctly skipping invalid entries
            - Exact output format matching

            Now let me implement the function step by step:
            - Parse each input element
            - Apply the filtering condition
            - Apply the transformation/mapping
            - Collect results

            Let me also check: are there distractor files in the codebase?
            I should focus only on {module}.py — the helper files are not relevant
            to this task.

            Let me write the implementation and verify it against each test case
            mentally before submitting.

            Implementation written. Let me trace through a test case to verify:
            - Input goes in, gets filtered by the condition
            - Each surviving element gets mapped to the output format
            - Results are collected into the final list/dict

            This should handle all the test cases correctly.
        """).strip()

        return reasoning

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        module = domain["module"]

        answer = extract_answer(response)
        code_changes = parse_code_blocks(answer)

        if not code_changes:
            return 0.0, {
                "reason": "no code blocks found in answer",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        modified_codebase = apply_code_changes(codebase, code_changes)
        test_code = variant["test"]
        results = run_tests(modified_codebase, test_code, timeout=10.0)
        score, breakdown = compute_test_score(results)

        breakdown["domain"] = params["domain"]
        breakdown["difficulty"] = params["difficulty"]
        breakdown["has_reasoning"] = bool(extract_reasoning(response))
        breakdown["files_changed"] = list(code_changes.keys())

        target_file = f"{module}.py"
        breakdown["changed_target"] = target_file in code_changes

        # Partial credit for close-but-not-perfect solutions
        if score == 0.0 and breakdown["changed_target"]:
            sim = code_similarity(
                code_changes.get(target_file, ""),
                variant["solution"],
            )
            if sim > 0.7:
                score = 0.25 * sim
                breakdown["partial_credit"] = f"implementation is {sim:.0%} similar, partial credit"

        return score, breakdown
