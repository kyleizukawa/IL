"""
Environment 15: Stack Trace Debugging

Skill: Debugging crashes from a stack trace.

The model is given a codebase + a stack trace from a crash and must fix
the issue. The stack trace points to the crash location, but the root cause
may be in a different file. The model must trace the stack to find the
root cause and write a fix.

Domains:
- KeyError on missing dict key
- IndexError on empty list
- TypeError on None
- AttributeError on wrong type
- RecursionError from infinite recursion

Difficulty scaling:
- easy: single file, obvious crash location
- medium: 2 files, need to trace the stack
- hard: 3+ files with distractors, subtle crash root cause

Grading:
- 1.0 if no crash AND correct result
- 0.5 if no crash but wrong result
- 0.0 if still crashes
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
# Each variant has: buggy_codebase (dict of files), stack_trace, fixed_code,
#   test_code, crash_description

DOMAINS = {

    # ── Domain 1: KeyError on missing dict key ──
    "keyerror": {
        "easy": {
            "files": {
                "config_loader.py": textwrap.dedent('''
                    def load_config(config_dict, key):
                        """Load a value from config by key."""
                        return config_dict[key]

                    def get_all_values(config_dict, keys):
                        """Get multiple values from config."""
                        return [load_config(config_dict, k) for k in keys]
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 5, in <module>
                    result = get_all_values(config, ["host", "port", "debug"])
                  File "config_loader.py", line 6, in get_all_values
                    return [load_config(config_dict, k) for k in keys]
                  File "config_loader.py", line 3, in load_config
                    return config_dict[key]
                             ~~~~~~~~~~~^^^^^^^
                KeyError: 'debug'
            ''').strip(),
            "fixed": {
                "config_loader.py": textwrap.dedent('''
                    def load_config(config_dict, key, default=None):
                        """Load a value from config by key."""
                        return config_dict.get(key, default)

                    def get_all_values(config_dict, keys):
                        """Get multiple values from config."""
                        return [load_config(config_dict, k) for k in keys]
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from config_loader import get_all_values, load_config
                def test_missing_key():
                    config = {"host": "localhost", "port": 8080}
                    result = get_all_values(config, ["host", "port", "debug"])
                    assert result == ["localhost", 8080, None]
                def test_all_keys_present():
                    config = {"a": 1, "b": 2}
                    assert get_all_values(config, ["a", "b"]) == [1, 2]
                def test_empty():
                    assert get_all_values({}, ["x"]) == [None]
                def test_load_with_default():
                    assert load_config({}, "missing", "default") == "default"
                def test_load_existing():
                    assert load_config({"x": 42}, "x") == 42
            ''').strip(),
            "crash_desc": "KeyError: 'debug' — the config dict doesn't have a 'debug' key",
        },
        "medium": {
            "files": {
                "config_loader.py": textwrap.dedent('''
                    def load_config(config_dict, key):
                        """Load a value from config by key."""
                        return config_dict[key]

                    def get_all_values(config_dict, keys):
                        """Get multiple values from config."""
                        return [load_config(config_dict, k) for k in keys]
                ''').strip(),
                "app.py": textwrap.dedent('''
                    from config_loader import get_all_values

                    REQUIRED_KEYS = ["host", "port", "database", "username", "password"]

                    def initialize_app(config):
                        """Initialize app with required config values."""
                        values = get_all_values(config, REQUIRED_KEYS)
                        return dict(zip(REQUIRED_KEYS, values))

                    def connect_database(app_config):
                        """Connect to database using app config."""
                        conn_str = f"postgresql://{app_config['username']}:{app_config['password']}@{app_config['host']}:{app_config['port']}/{app_config['database']}"
                        return conn_str
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 3, in <module>
                    app = initialize_app(config)
                  File "app.py", line 7, in initialize_app
                    values = get_all_values(config, REQUIRED_KEYS)
                  File "config_loader.py", line 6, in get_all_values
                    return [load_config(config_dict, k) for k in keys]
                  File "config_loader.py", line 3, in load_config
                    return config_dict[key]
                             ~~~~~~~~~~~^^^^^^^
                KeyError: 'password'
            ''').strip(),
            "fixed": {
                "config_loader.py": textwrap.dedent('''
                    def load_config(config_dict, key, default=None):
                        """Load a value from config by key."""
                        return config_dict.get(key, default)

                    def get_all_values(config_dict, keys):
                        """Get multiple values from config."""
                        return [load_config(config_dict, k) for k in keys]
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from app import initialize_app, connect_database
                def test_missing_key():
                    config = {"host": "localhost", "port": 5432, "database": "mydb", "username": "admin"}
                    app = initialize_app(config)
                    assert app["password"] is None
                def test_all_keys():
                    config = {"host": "localhost", "port": 5432, "database": "mydb", "username": "admin", "password": "secret"}
                    app = initialize_app(config)
                    conn = connect_database(app)
                    assert "admin:secret" in conn
                def test_partial_config():
                    config = {"host": "localhost"}
                    app = initialize_app(config)
                    assert app["host"] == "localhost"
                    assert app["port"] is None
                def test_empty_config():
                    app = initialize_app({})
                    assert app["host"] is None
            ''').strip(),
            "crash_desc": "KeyError: 'password' — the config dict is missing the 'password' key",
        },
        "hard": {
            "files": {
                "config_loader.py": textwrap.dedent('''
                    def load_config(config_dict, key):
                        """Load a value from config by key."""
                        return config_dict[key]

                    def get_all_values(config_dict, keys):
                        """Get multiple values from config."""
                        return [load_config(config_dict, k) for k in keys]
                ''').strip(),
                "app.py": textwrap.dedent('''
                    from config_loader import get_all_values

                    REQUIRED_KEYS = ["host", "port", "database", "username", "password"]

                    def initialize_app(config):
                        """Initialize app with required config values."""
                        values = get_all_values(config, REQUIRED_KEYS)
                        return dict(zip(REQUIRED_KEYS, values))

                    def connect_database(app_config):
                        """Connect to database using app config."""
                        conn_str = f"postgresql://{app_config['username']}:{app_config['password']}@{app_config['host']}:{app_config['port']}/{app_config['database']}"
                        return conn_str
                ''').strip(),
                "runner.py": textwrap.dedent('''
                    from app import initialize_app, connect_database

                    def run_migrations(config):
                        """Run database migrations."""
                        app = initialize_app(config)
                        conn = connect_database(app)
                        return f"Migrations run on {conn}"

                    def validate_config(config):
                        """Validate that config has required keys."""
                        required = ["host", "port", "database"]
                        for key in required:
                            if key not in config:
                                raise ValueError(f"Missing required key: {key}")
                        return True
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 8, in <module>
                    result = run_migrations(config)
                  File "runner.py", line 5, in run_migrations
                    app = initialize_app(config)
                  File "app.py", line 7, in initialize_app
                    values = get_all_values(config, REQUIRED_KEYS)
                  File "config_loader.py", line 6, in get_all_values
                    return [load_config(config_dict, k) for k in keys]
                  File "config_loader.py", line 3, in load_config
                    return config_dict[key]
                             ~~~~~~~~~~~^^^^^^^
                KeyError: 'password'
            ''').strip(),
            "fixed": {
                "config_loader.py": textwrap.dedent('''
                    def load_config(config_dict, key, default=None):
                        """Load a value from config by key."""
                        return config_dict.get(key, default)

                    def get_all_values(config_dict, keys):
                        """Get multiple values from config."""
                        return [load_config(config_dict, k) for k in keys]
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from runner import run_migrations, validate_config
                from app import initialize_app
                def test_missing_password():
                    config = {"host": "localhost", "port": 5432, "database": "mydb", "username": "admin"}
                    result = run_migrations(config)
                    assert "Migrations run on" in result
                def test_full_config():
                    config = {"host": "localhost", "port": 5432, "database": "mydb", "username": "admin", "password": "secret"}
                    result = run_migrations(config)
                    assert "admin:secret" in result
                def test_validate_passes():
                    config = {"host": "localhost", "port": 5432, "database": "mydb"}
                    assert validate_config(config) == True
                def test_validate_fails():
                    config = {"host": "localhost"}
                    try:
                        validate_config(config)
                        assert False, "Should have raised ValueError"
                    except ValueError:
                        assert True
                def test_partial_init():
                    config = {"host": "localhost", "port": 8080}
                    app = initialize_app(config)
                    assert app["host"] == "localhost"
                    assert app["database"] is None
            ''').strip(),
            "crash_desc": "KeyError: 'password' — the config dict is missing the 'password' key",
        },
    },

    # ── Domain 2: IndexError on empty list ──
    "indexerror": {
        "easy": {
            "files": {
                "list_processor.py": textwrap.dedent('''
                    def get_first(items):
                        """Get the first item from a list."""
                        return items[0]

                    def process_list(items):
                        """Process a list, returning the first item squared."""
                        first = get_first(items)
                        return first ** 2
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 2, in <module>
                    result = process_list([])
                  File "list_processor.py", line 6, in process_list
                    first = get_first(items)
                  File "list_processor.py", line 3, in get_first
                    return items[0]
                           ~~~~~~^^^
                IndexError: list index out of range
            ''').strip(),
            "fixed": {
                "list_processor.py": textwrap.dedent('''
                    def get_first(items, default=None):
                        """Get the first item from a list."""
                        if not items:
                            return default
                        return items[0]

                    def process_list(items):
                        """Process a list, returning the first item squared."""
                        first = get_first(items)
                        if first is None:
                            return None
                        return first ** 2
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from list_processor import get_first, process_list
                def test_empty_list():
                    assert process_list([]) is None
                def test_non_empty():
                    assert process_list([5]) == 25
                def test_get_first_empty():
                    assert get_first([]) is None
                def test_get_first_default():
                    assert get_first([], "default") == "default"
                def test_get_first_non_empty():
                    assert get_first([1, 2, 3]) == 1
            ''').strip(),
            "crash_desc": "IndexError: list index out of range — accessing items[0] on an empty list",
        },
        "medium": {
            "files": {
                "list_processor.py": textwrap.dedent('''
                    def get_first(items):
                        """Get the first item from a list."""
                        return items[0]

                    def get_last(items):
                        """Get the last item from a list."""
                        return items[-1]

                    def process_list(items):
                        """Process a list, returning first and last as tuple."""
                        first = get_first(items)
                        last = get_last(items)
                        return (first, last)
                ''').strip(),
                "analyzer.py": textwrap.dedent('''
                    from list_processor import process_list

                    def analyze_data(data_points):
                        """Analyze data points, returning range."""
                        if not data_points:
                            return {"range": 0, "first": None, "last": None}
                        first, last = process_list(data_points)
                        return {"range": last - first, "first": first, "last": last}

                    def summarize(data_points):
                        """Summarize data points."""
                        analysis = analyze_data(data_points)
                        return f"Range: {analysis['range']}, First: {analysis['first']}, Last: {analysis['last']}"
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 3, in <module>
                    summary = summarize([])
                  File "analyzer.py", line 11, in summarize
                    analysis = analyze_data(data_points)
                  File "analyzer.py", line 6, in analyze_data
                    first, last = process_list(data_points)
                  File "list_processor.py", line 11, in process_list
                    first = get_first(items)
                  File "list_processor.py", line 3, in get_first
                    return items[0]
                           ~~~~~~^^^
                IndexError: list index out of range
            ''').strip(),
            "fixed": {
                "list_processor.py": textwrap.dedent('''
                    def get_first(items, default=None):
                        """Get the first item from a list."""
                        if not items:
                            return default
                        return items[0]

                    def get_last(items, default=None):
                        """Get the last item from a list."""
                        if not items:
                            return default
                        return items[-1]

                    def process_list(items):
                        """Process a list, returning first and last as tuple."""
                        first = get_first(items)
                        last = get_last(items)
                        return (first, last)
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from analyzer import analyze_data, summarize
                from list_processor import process_list, get_first, get_last
                def test_empty_list():
                    result = analyze_data([])
                    assert result["range"] == 0
                    assert result["first"] is None
                    assert result["last"] is None
                def test_non_empty():
                    result = analyze_data([1, 5, 10])
                    assert result["range"] == 9
                    assert result["first"] == 1
                    assert result["last"] == 10
                def test_single_item():
                    result = analyze_data([7])
                    assert result["range"] == 0
                    assert result["first"] == 7
                    assert result["last"] == 7
                def test_summarize():
                    s = summarize([1, 2, 3])
                    assert "Range: 2" in s
                def test_process_empty():
                    assert process_list([]) == (None, None)
            ''').strip(),
            "crash_desc": "IndexError: list index out of range — analyze_data checks for empty but process_list still crashes",
        },
        "hard": {
            "files": {
                "list_processor.py": textwrap.dedent('''
                    def get_first(items):
                        """Get the first item from a list."""
                        return items[0]

                    def get_last(items):
                        """Get the last item from a list."""
                        return items[-1]

                    def process_list(items):
                        """Process a list, returning first and last as tuple."""
                        first = get_first(items)
                        last = get_last(items)
                        return (first, last)
                ''').strip(),
                "analyzer.py": textwrap.dedent('''
                    from list_processor import process_list

                    def analyze_data(data_points):
                        """Analyze data points, returning range."""
                        if not data_points:
                            return {"range": 0, "first": None, "last": None}
                        first, last = process_list(data_points)
                        return {"range": last - first, "first": first, "last": last}

                    def summarize(data_points):
                        """Summarize data points."""
                        analysis = analyze_data(data_points)
                        return f"Range: {analysis['range']}, First: {analysis['first']}, Last: {analysis['last']}"
                ''').strip(),
                "pipeline.py": textwrap.dedent('''
                    from analyzer import analyze_data, summarize

                    def run_pipeline(raw_data):
                        """Run the full data pipeline."""
                        cleaned = clean_data(raw_data)
                        if not cleaned:
                            return {"status": "empty", "summary": "No data", "analysis": None}
                        analysis = analyze_data(cleaned)
                        summary = summarize(cleaned)
                        return {"status": "ok", "summary": summary, "analysis": analysis}

                    def clean_data(raw_data):
                        """Clean raw data by removing None values."""
                        return [x for x in raw_data if x is not None]

                    def batch_process(datasets):
                        """Process multiple datasets."""
                        results = []
                        for ds in datasets:
                            try:
                                result = run_pipeline(ds)
                                results.append(result)
                            except Exception as e:
                                results.append({"status": "error", "error": str(e)})
                        return results
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 5, in <module>
                    results = batch_process([[1, 2, 3], [], [None, None], [4, 5]])
                  File "pipeline.py", line 21, in batch_process
                    result = run_pipeline(ds)
                  File "pipeline.py", line 6, in run_pipeline
                    analysis = analyze_data(cleaned)
                  File "analyzer.py", line 6, in analyze_data
                    first, last = process_list(data_points)
                  File "list_processor.py", line 11, in process_list
                    first = get_first(items)
                  File "list_processor.py", line 3, in get_first
                    return items[0]
                           ~~~~~~^^^
                IndexError: list index out of range
            ''').strip(),
            "fixed": {
                "list_processor.py": textwrap.dedent('''
                    def get_first(items, default=None):
                        """Get the first item from a list."""
                        if not items:
                            return default
                        return items[0]

                    def get_last(items, default=None):
                        """Get the last item from a list."""
                        if not items:
                            return default
                        return items[-1]

                    def process_list(items):
                        """Process a list, returning first and last as tuple."""
                        first = get_first(items)
                        last = get_last(items)
                        return (first, last)
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from pipeline import run_pipeline, batch_process, clean_data
                from analyzer import analyze_data
                def test_empty_dataset():
                    result = run_pipeline([])
                    assert result["status"] == "empty"
                def test_all_none():
                    result = run_pipeline([None, None])
                    assert result["status"] == "empty"
                def test_normal():
                    result = run_pipeline([1, 2, 3])
                    assert result["status"] == "ok"
                    assert result["analysis"]["range"] == 2
                def test_batch():
                    results = batch_process([[1, 2, 3], [], [None, None], [4, 5]])
                    assert len(results) == 4
                    assert results[0]["status"] == "ok"
                    assert results[1]["status"] == "empty"
                    assert results[2]["status"] == "empty"
                    assert results[3]["status"] == "ok"
                def test_clean_data():
                    assert clean_data([None, 1, None, 2]) == [1, 2]
                    assert clean_data([]) == []
                    assert clean_data([None, None]) == []
            ''').strip(),
            "crash_desc": "IndexError: list index out of range — clean_data removes None values resulting in empty list, but analyze_data/process_list don't handle it",
        },
    },

    # ── Domain 3: TypeError on None ──
    "typeerror_none": {
        "easy": {
            "files": {
                "math_ops.py": textwrap.dedent('''
                    def safe_divide(a, b):
                        """Divide a by b safely."""
                        return a / b

                    def compute_ratio(numerator, denominator):
                        """Compute the ratio of two numbers."""
                        return safe_divide(numerator, denominator)
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 2, in <module>
                    result = compute_ratio(10, None)
                  File "math_ops.py", line 6, in compute_ratio
                    return safe_divide(numerator, denominator)
                  File "math_ops.py", line 3, in safe_divide
                    return a / b
                           ~~^~~
                TypeError: unsupported operand type(s) for /: 'int' and 'NoneType'
            ''').strip(),
            "fixed": {
                "math_ops.py": textwrap.dedent('''
                    def safe_divide(a, b):
                        """Divide a by b safely."""
                        if a is None or b is None:
                            return None
                        if b == 0:
                            return None
                        return a / b

                    def compute_ratio(numerator, denominator):
                        """Compute the ratio of two numbers."""
                        return safe_divide(numerator, denominator)
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from math_ops import safe_divide, compute_ratio
                def test_none_denominator():
                    assert compute_ratio(10, None) is None
                def test_none_numerator():
                    assert compute_ratio(None, 5) is None
                def test_zero_division():
                    assert compute_ratio(10, 0) is None
                def test_normal():
                    assert compute_ratio(10, 2) == 5.0
                def test_safe_divide_direct():
                    assert safe_divide(6, 3) == 2.0
                    assert safe_divide(1, 0) is None
                    assert safe_divide(None, 1) is None
            ''').strip(),
            "crash_desc": "TypeError: unsupported operand type(s) for /: 'int' and 'NoneType' — dividing by None",
        },
        "medium": {
            "files": {
                "math_ops.py": textwrap.dedent('''
                    def safe_divide(a, b):
                        """Divide a by b safely."""
                        return a / b

                    def compute_ratio(numerator, denominator):
                        """Compute the ratio of two numbers."""
                        return safe_divide(numerator, denominator)
                ''').strip(),
                "stats.py": textwrap.dedent('''
                    from math_ops import compute_ratio

                    def calculate_stats(values):
                        """Calculate statistics for a list of values."""
                        if not values:
                            return {"mean": None, "ratio": None}
                        mean = sum(values) / len(values)
                        first = values[0]
                        ratio = compute_ratio(first, mean)
                        return {"mean": mean, "ratio": ratio}

                    def process_metrics(metrics_dict):
                        """Process a dictionary of metrics."""
                        results = {}
                        for key, values in metrics_dict.items():
                            results[key] = calculate_stats(values)
                        return results
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 4, in <module>
                    results = process_metrics({"temp": [20, None, 22], "humid": [50, 60]})
                  File "stats.py", line 12, in process_metrics
                    results[key] = calculate_stats(values)
                  File "stats.py", line 7, in calculate_stats
                    ratio = compute_ratio(first, mean)
                  File "math_ops.py", line 6, in compute_ratio
                    return safe_divide(numerator, denominator)
                  File "math_ops.py", line 3, in safe_divide
                    return a / b
                           ~~^~~
                TypeError: unsupported operand type(s) for /: 'int' and 'float'
            ''').strip(),
            "fixed": {
                "math_ops.py": textwrap.dedent('''
                    def safe_divide(a, b):
                        """Divide a by b safely."""
                        if a is None or b is None:
                            return None
                        if b == 0:
                            return None
                        return a / b

                    def compute_ratio(numerator, denominator):
                        """Compute the ratio of two numbers."""
                        return safe_divide(numerator, denominator)
                ''').strip(),
                "stats.py": textwrap.dedent('''
                    from math_ops import compute_ratio

                    def calculate_stats(values):
                        """Calculate statistics for a list of values."""
                        clean_values = [v for v in values if v is not None]
                        if not clean_values:
                            return {"mean": None, "ratio": None}
                        mean = sum(clean_values) / len(clean_values)
                        first = clean_values[0]
                        ratio = compute_ratio(first, mean)
                        return {"mean": mean, "ratio": ratio}

                    def process_metrics(metrics_dict):
                        """Process a dictionary of metrics."""
                        results = {}
                        for key, values in metrics_dict.items():
                            results[key] = calculate_stats(values)
                        return results
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from stats import calculate_stats, process_metrics
                def test_with_none_values():
                    result = calculate_stats([20, None, 22])
                    assert result["mean"] is not None
                    assert result["ratio"] is not None
                def test_all_none():
                    result = calculate_stats([None, None])
                    assert result["mean"] is None
                    assert result["ratio"] is None
                def test_normal():
                    result = calculate_stats([10, 20, 30])
                    assert result["mean"] == 20.0
                    assert result["ratio"] == 0.5
                def test_process_metrics():
                    results = process_metrics({"a": [10, 20], "b": [None, None]})
                    assert results["a"]["mean"] == 15.0
                    assert results["b"]["mean"] is None
                def test_empty():
                    result = calculate_stats([])
                    assert result["mean"] is None
            ''').strip(),
            "crash_desc": "TypeError: unsupported operand type(s) for /: 'int' and 'float' — None values in the list cause sum() to fail",
        },
        "hard": {
            "files": {
                "math_ops.py": textwrap.dedent('''
                    def safe_divide(a, b):
                        """Divide a by b safely."""
                        return a / b

                    def compute_ratio(numerator, denominator):
                        """Compute the ratio of two numbers."""
                        return safe_divide(numerator, denominator)
                ''').strip(),
                "stats.py": textwrap.dedent('''
                    from math_ops import compute_ratio

                    def calculate_stats(values):
                        """Calculate statistics for a list of values."""
                        if not values:
                            return {"mean": None, "ratio": None}
                        mean = sum(values) / len(values)
                        first = values[0]
                        ratio = compute_ratio(first, mean)
                        return {"mean": mean, "ratio": ratio}

                    def process_metrics(metrics_dict):
                        """Process a dictionary of metrics."""
                        results = {}
                        for key, values in metrics_dict.items():
                            results[key] = calculate_stats(values)
                        return results
                ''').strip(),
                "report.py": textwrap.dedent('''
                    from stats import process_metrics
                    import json

                    def generate_report(sensor_data):
                        """Generate a JSON report from sensor data."""
                        metrics = process_metrics(sensor_data)
                        report = {
                            "summary": {},
                            "alerts": [],
                            "raw": metrics,
                        }
                        for key, stats in metrics.items():
                            if stats["mean"] is not None:
                                report["summary"][key] = f"mean={stats['mean']:.2f}"
                                if stats["ratio"] is not None and stats["ratio"] < 0.5:
                                    report["alerts"].append(f"{key} ratio low: {stats['ratio']:.2f}")
                        return json.dumps(report, indent=2)

                    def validate_sensor_data(sensor_data):
                        """Validate sensor data structure."""
                        if not isinstance(sensor_data, dict):
                            return False
                        for key, values in sensor_data.items():
                            if not isinstance(values, list):
                                return False
                        return True
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 6, in <module>
                    report = generate_report({"temp": [20, None, 22], "humid": [50, 60], "press": [None, None]})
                  File "report.py", line 7, in generate_report
                    metrics = process_metrics(sensor_data)
                  File "stats.py", line 12, in process_metrics
                    results[key] = calculate_stats(values)
                  File "stats.py", line 6, in calculate_stats
                    mean = sum(values) / len(values)
                  File "stats.py", line 6, in calculate_stats
                    mean = sum(values) / len(values)
                TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
            ''').strip(),
            "fixed": {
                "math_ops.py": textwrap.dedent('''
                    def safe_divide(a, b):
                        """Divide a by b safely."""
                        if a is None or b is None:
                            return None
                        if b == 0:
                            return None
                        return a / b

                    def compute_ratio(numerator, denominator):
                        """Compute the ratio of two numbers."""
                        return safe_divide(numerator, denominator)
                ''').strip(),
                "stats.py": textwrap.dedent('''
                    from math_ops import compute_ratio

                    def calculate_stats(values):
                        """Calculate statistics for a list of values."""
                        clean_values = [v for v in values if v is not None]
                        if not clean_values:
                            return {"mean": None, "ratio": None}
                        mean = sum(clean_values) / len(clean_values)
                        first = clean_values[0]
                        ratio = compute_ratio(first, mean)
                        return {"mean": mean, "ratio": ratio}

                    def process_metrics(metrics_dict):
                        """Process a dictionary of metrics."""
                        results = {}
                        for key, values in metrics_dict.items():
                            results[key] = calculate_stats(values)
                        return results
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from report import generate_report, validate_sensor_data
                from stats import calculate_stats
                import json
                def test_with_none():
                    report = json.loads(generate_report({"temp": [20, None, 22], "humid": [50, 60]}))
                    assert "temp" in report["summary"]
                    assert "humid" in report["summary"]
                def test_all_none():
                    report = json.loads(generate_report({"press": [None, None]}))
                    assert "press" not in report["summary"]
                def test_normal():
                    report = json.loads(generate_report({"a": [10, 20, 30]}))
                    assert "a" in report["summary"]
                def test_validate():
                    assert validate_sensor_data({"a": [1, 2]}) == True
                    assert validate_sensor_data({"a": "not a list"}) == False
                    assert validate_sensor_data("not a dict") == False
                def test_stats_none():
                    result = calculate_stats([None, None])
                    assert result["mean"] is None
                    assert result["ratio"] is None
            ''').strip(),
            "crash_desc": "TypeError: unsupported operand type(s) for +: 'int' and 'NoneType' — sum() fails on list containing None values",
        },
    },

    # ── Domain 4: RecursionError from infinite recursion ──
    "recursionerror": {
        "easy": {
            "files": {
                "tree_traversal.py": textwrap.dedent('''
                    def sum_tree(node):
                        """Sum all values in a tree. Node is dict with 'value' and 'children'."""
                        total = node['value']
                        for child in node.get('children', []):
                            total += sum_tree(child)
                        return total
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 8, in <module>
                    result = sum_tree(cyclic_tree)
                  File "tree_traversal.py", line 5, in sum_tree
                    total += sum_tree(child)
                  File "tree_traversal.py", line 5, in sum_tree
                    total += sum_tree(child)
                  File "tree_traversal.py", line 5, in sum_tree
                    total += sum_tree(child)
                  ...
                RecursionError: maximum recursion depth exceeded
            ''').strip(),
            "fixed": {
                "tree_traversal.py": textwrap.dedent('''
                    def sum_tree(node, _visited=None):
                        """Sum all values in a tree. Node is dict with 'value' and 'children'."""
                        if _visited is None:
                            _visited = set()
                        node_id = id(node)
                        if node_id in _visited:
                            return 0
                        _visited.add(node_id)
                        total = node['value']
                        for child in node.get('children', []):
                            total += sum_tree(child, _visited)
                        return total
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from tree_traversal import sum_tree
                def test_simple_tree():
                    tree = {"value": 1, "children": [{"value": 2, "children": []}, {"value": 3, "children": []}]}
                    assert sum_tree(tree) == 6
                def test_single_node():
                    assert sum_tree({"value": 42, "children": []}) == 42
                def test_no_children_key():
                    assert sum_tree({"value": 10}) == 10
                def test_cyclic():
                    a = {"value": 1, "children": []}
                    b = {"value": 2, "children": [a]}
                    a["children"].append(b)
                    assert sum_tree(a) == 3
                def test_deep_tree():
                    node = {"value": 0, "children": []}
                    current = node
                    for i in range(100):
                        child = {"value": i + 1, "children": []}
                        current["children"] = [child]
                        current = child
                    assert sum_tree(node) == sum(range(1, 101))
            ''').strip(),
            "crash_desc": "RecursionError: maximum recursion depth exceeded — cyclic reference in tree causes infinite recursion",
        },
        "medium": {
            "files": {
                "tree_traversal.py": textwrap.dedent('''
                    def sum_tree(node):
                        """Sum all values in a tree. Node is dict with 'value' and 'children'."""
                        total = node['value']
                        for child in node.get('children', []):
                            total += sum_tree(child)
                        return total

                    def count_nodes(node):
                        """Count nodes in a tree."""
                        count = 1
                        for child in node.get('children', []):
                            count += count_nodes(child)
                        return count
                ''').strip(),
                "tree_ops.py": textwrap.dedent('''
                    from tree_traversal import sum_tree, count_nodes

                    def tree_stats(root):
                        """Get statistics about a tree."""
                        total = sum_tree(root)
                        count = count_nodes(root)
                        if count == 0:
                            return {"total": 0, "count": 0, "average": 0}
                        return {"total": total, "count": count, "average": total / count}

                    def find_max_value(node):
                        """Find the maximum value in a tree."""
                        max_val = node['value']
                        for child in node.get('children', []):
                            child_max = find_max_value(child)
                            if child_max > max_val:
                                max_val = child_max
                        return max_val
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 3, in <module>
                    stats = tree_stats(cyclic_tree)
                  File "tree_ops.py", line 6, in tree_stats
                    total = sum_tree(root)
                  File "tree_traversal.py", line 5, in sum_tree
                    total += sum_tree(child)
                  File "tree_traversal.py", line 5, in sum_tree
                    total += sum_tree(child)
                  ...
                RecursionError: maximum recursion depth exceeded
            ''').strip(),
            "fixed": {
                "tree_traversal.py": textwrap.dedent('''
                    def sum_tree(node, _visited=None):
                        """Sum all values in a tree. Node is dict with 'value' and 'children'."""
                        if _visited is None:
                            _visited = set()
                        node_id = id(node)
                        if node_id in _visited:
                            return 0
                        _visited.add(node_id)
                        total = node['value']
                        for child in node.get('children', []):
                            total += sum_tree(child, _visited)
                        return total

                    def count_nodes(node, _visited=None):
                        """Count nodes in a tree."""
                        if _visited is None:
                            _visited = set()
                        node_id = id(node)
                        if node_id in _visited:
                            return 0
                        _visited.add(node_id)
                        count = 1
                        for child in node.get('children', []):
                            count += count_nodes(child, _visited)
                        return count
                ''').strip(),
                "tree_ops.py": textwrap.dedent('''
                    from tree_traversal import sum_tree, count_nodes

                    def tree_stats(root):
                        """Get statistics about a tree."""
                        total = sum_tree(root)
                        count = count_nodes(root)
                        if count == 0:
                            return {"total": 0, "count": 0, "average": 0}
                        return {"total": total, "count": count, "average": total / count}

                    def find_max_value(node, _visited=None):
                        """Find the maximum value in a tree."""
                        if _visited is None:
                            _visited = set()
                        node_id = id(node)
                        if node_id in _visited:
                            return float('-inf')
                        _visited.add(node_id)
                        max_val = node['value']
                        for child in node.get('children', []):
                            child_max = find_max_value(child, _visited)
                            if child_max > max_val:
                                max_val = child_max
                        return max_val
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from tree_ops import tree_stats, find_max_value
                from tree_traversal import sum_tree, count_nodes
                def test_simple_stats():
                    tree = {"value": 10, "children": [{"value": 20, "children": []}, {"value": 30, "children": []}]}
                    stats = tree_stats(tree)
                    assert stats["total"] == 60
                    assert stats["count"] == 3
                    assert stats["average"] == 20.0
                def test_cyclic_stats():
                    a = {"value": 1, "children": []}
                    b = {"value": 2, "children": [a]}
                    a["children"].append(b)
                    stats = tree_stats(a)
                    assert stats["total"] == 3
                    assert stats["count"] == 2
                def test_find_max():
                    tree = {"value": 5, "children": [{"value": 10, "children": []}, {"value": 3, "children": []}]}
                    assert find_max_value(tree) == 10
                def test_find_max_cyclic():
                    a = {"value": 1, "children": []}
                    b = {"value": 5, "children": [a]}
                    a["children"].append(b)
                    assert find_max_value(a) == 5
                def test_single_node():
                    assert sum_tree({"value": 42, "children": []}) == 42
                    assert count_nodes({"value": 42, "children": []}) == 1
            ''').strip(),
            "crash_desc": "RecursionError: maximum recursion depth exceeded — cyclic reference causes infinite recursion in both sum_tree and count_nodes",
        },
        "hard": {
            "files": {
                "tree_traversal.py": textwrap.dedent('''
                    def sum_tree(node):
                        """Sum all values in a tree. Node is dict with 'value' and 'children'."""
                        total = node['value']
                        for child in node.get('children', []):
                            total += sum_tree(child)
                        return total

                    def count_nodes(node):
                        """Count nodes in a tree."""
                        count = 1
                        for child in node.get('children', []):
                            count += count_nodes(child)
                        return count

                    def tree_depth(node):
                        """Calculate the depth of a tree."""
                        if not node.get('children'):
                            return 1
                        return 1 + max(tree_depth(child) for child in node['children'])
                ''').strip(),
                "tree_ops.py": textwrap.dedent('''
                    from tree_traversal import sum_tree, count_nodes, tree_depth

                    def tree_stats(root):
                        """Get statistics about a tree."""
                        total = sum_tree(root)
                        count = count_nodes(root)
                        depth = tree_depth(root)
                        if count == 0:
                            return {"total": 0, "count": 0, "average": 0, "depth": 0}
                        return {"total": total, "count": count, "average": total / count, "depth": depth}

                    def find_max_value(node):
                        """Find the maximum value in a tree."""
                        max_val = node['value']
                        for child in node.get('children', []):
                            child_max = find_max_value(child)
                            if child_max > max_val:
                                max_val = child_max
                        return max_val
                ''').strip(),
                "tree_report.py": textwrap.dedent('''
                    from tree_ops import tree_stats, find_max_value
                    import json

                    def generate_tree_report(root, name="tree"):
                        """Generate a JSON report for a tree structure."""
                        stats = tree_stats(root)
                        max_val = find_max_value(root)
                        report = {
                            "name": name,
                            "stats": stats,
                            "max_value": max_val,
                            "health": "good" if stats["count"] > 0 else "empty",
                        }
                        return json.dumps(report, indent=2)

                    def compare_trees(tree_a, tree_b):
                        """Compare two trees and return differences."""
                        stats_a = tree_stats(tree_a)
                        stats_b = tree_stats(tree_b)
                        diffs = {}
                        for key in stats_a:
                            if stats_a[key] != stats_b[key]:
                                diffs[key] = {"a": stats_a[key], "b": stats_b[key]}
                        return diffs
                ''').strip(),
            },
            "stack_trace": textwrap.dedent('''
                Traceback (most recent call last):
                  File "main.py", line 5, in <module>
                    report = generate_tree_report(cyclic_tree, "sensor_tree")
                  File "tree_report.py", line 6, in generate_tree_report
                    stats = tree_stats(root)
                  File "tree_ops.py", line 7, in tree_stats
                    total = sum_tree(root)
                  File "tree_traversal.py", line 5, in sum_tree
                    total += sum_tree(child)
                  File "tree_traversal.py", line 5, in sum_tree
                    total += sum_tree(child)
                  ...
                RecursionError: maximum recursion depth exceeded
            ''').strip(),
            "fixed": {
                "tree_traversal.py": textwrap.dedent('''
                    def sum_tree(node, _visited=None):
                        """Sum all values in a tree. Node is dict with 'value' and 'children'."""
                        if _visited is None:
                            _visited = set()
                        node_id = id(node)
                        if node_id in _visited:
                            return 0
                        _visited.add(node_id)
                        total = node['value']
                        for child in node.get('children', []):
                            total += sum_tree(child, _visited)
                        return total

                    def count_nodes(node, _visited=None):
                        """Count nodes in a tree."""
                        if _visited is None:
                            _visited = set()
                        node_id = id(node)
                        if node_id in _visited:
                            return 0
                        _visited.add(node_id)
                        count = 1
                        for child in node.get('children', []):
                            count += count_nodes(child, _visited)
                        return count

                    def tree_depth(node, _visited=None):
                        """Calculate the depth of a tree."""
                        if _visited is None:
                            _visited = set()
                        node_id = id(node)
                        if node_id in _visited:
                            return 0
                        _visited.add(node_id)
                        if not node.get('children'):
                            return 1
                        return 1 + max(tree_depth(child, _visited) for child in node['children'])
                ''').strip(),
                "tree_ops.py": textwrap.dedent('''
                    from tree_traversal import sum_tree, count_nodes, tree_depth

                    def tree_stats(root):
                        """Get statistics about a tree."""
                        total = sum_tree(root)
                        count = count_nodes(root)
                        depth = tree_depth(root)
                        if count == 0:
                            return {"total": 0, "count": 0, "average": 0, "depth": 0}
                        return {"total": total, "count": count, "average": total / count, "depth": depth}

                    def find_max_value(node, _visited=None):
                        """Find the maximum value in a tree."""
                        if _visited is None:
                            _visited = set()
                        node_id = id(node)
                        if node_id in _visited:
                            return float('-inf')
                        _visited.add(node_id)
                        max_val = node['value']
                        for child in node.get('children', []):
                            child_max = find_max_value(child, _visited)
                            if child_max > max_val:
                                max_val = child_max
                        return max_val
                ''').strip(),
            },
            "test": textwrap.dedent('''
                from tree_report import generate_tree_report, compare_trees
                from tree_ops import tree_stats, find_max_value
                from tree_traversal import sum_tree, count_nodes, tree_depth
                import json
                def test_cyclic_report():
                    a = {"value": 1, "children": []}
                    b = {"value": 5, "children": [a]}
                    a["children"].append(b)
                    report = json.loads(generate_tree_report(a, "test"))
                    assert report["stats"]["total"] == 6
                    assert report["stats"]["count"] == 2
                    assert report["max_value"] == 5
                def test_normal_report():
                    tree = {"value": 10, "children": [{"value": 20, "children": []}]}
                    report = json.loads(generate_tree_report(tree))
                    assert report["stats"]["total"] == 30
                    assert report["max_value"] == 20
                def test_compare():
                    a = {"value": 1, "children": [{"value": 2, "children": []}]}
                    b = {"value": 1, "children": [{"value": 3, "children": []}]}
                    diffs = compare_trees(a, b)
                    assert "total" in diffs
                def test_depth_cyclic():
                    a = {"value": 1, "children": []}
                    b = {"value": 2, "children": [a]}
                    a["children"].append(b)
                    assert tree_depth(a) == 2
                def test_depth_normal():
                    tree = {"value": 1, "children": [{"value": 2, "children": [{"value": 3, "children": []}]}]}
                    assert tree_depth(tree) == 3
            ''').strip(),
            "crash_desc": "RecursionError: maximum recursion depth exceeded — cyclic reference in tree causes infinite recursion across multiple tree functions",
        },
    },
}


# ── Distractor code ──

DISTRACTORS = [
    textwrap.dedent('''
        def format_tree(node, indent=0):
            """Format tree as text (not relevant to the crash)."""
            lines = ["  " * indent + str(node.get('value', '?'))]
            for child in node.get('children', []):
                lines.extend(format_tree(child, indent + 1))
            return lines

        def serialize_tree(node):
            """Serialize tree to flat list (not relevant to the crash)."""
            result = [node.get('value')]
            for child in node.get('children', []):
                result.extend(serialize_tree(child))
            return result
    ''').strip(),
    textwrap.dedent('''
        class TreeBuilder:
            """Build trees from flat data (not relevant to the crash)."""
            def __init__(self):
                self.nodes = {}
            def add_node(self, id, value, parent_id=None):
                self.nodes[id] = {"value": value, "children": []}
                if parent_id and parent_id in self.nodes:
                    self.nodes[parent_id]["children"].append(self.nodes[id])
            def get_root(self, root_id):
                return self.nodes.get(root_id)
    ''').strip(),
]


@register_env
class StacktraceDebugEnv(AgenticEnv):
    name = "stacktrace_debug"
    skill = "Debugging crashes from a stack trace"
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

        codebase = dict(variant["files"])

        for idx in params.get("distractor_indices", []):
            codebase[f"helper_{idx}.py"] = DISTRACTORS[idx]

        codebase["test_crash.py"] = variant["test"]
        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]

        lines = []
        lines.append("You are a software engineer debugging a crash in an unfamiliar codebase.")
        lines.append("")
        lines.append("A crash has occurred. The stack trace is shown below. Your task is to:")
        lines.append("1. Read the stack trace to identify the crash location")
        lines.append("2. Trace back through the call stack to find the root cause")
        lines.append("3. Fix the issue so the code no longer crashes")
        lines.append("4. The fix should handle the edge case gracefully (not just catch the exception)")
        lines.append("")
        lines.append("=== STACK TRACE ===")
        lines.append("```")
        lines.append(variant["stack_trace"])
        lines.append("```")
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            if filename == "test_crash.py":
                continue
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")
        lines.append("=== TEST FILE (test_crash.py) ===")
        lines.append("The test file verifies that the previously-crashing code path now works correctly.")
        lines.append("```python")
        lines.append(variant["test"])
        lines.append("```")
        lines.append("")
        lines.append("Provide your fix in the following format:")
        lines.append("<reasoning>")
        lines.append("...read the stack trace, identify the crash location, trace to root cause, explain the fix...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append("```python:filename.py")
        lines.append("# the corrected code for each file that needs to be changed")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        return dict(variant["fixed"])

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]

        reasoning = textwrap.dedent(f"""
            Let me carefully analyze this stack trace to find and fix the crash.

            === STACK TRACE ANALYSIS ===
            {variant['stack_trace']}

            The crash is: {variant['crash_desc']}

            Let me trace through the stack frame by frame, from the bottom up:

            1. The innermost frame shows where the crash actually occurs.
               I need to read this file and understand what operation fails.

            2. The next frame up shows who called the crashing function.
               I need to understand what arguments were passed.

            3. Continuing up the stack, I trace the call chain back to the
               root cause — where the problematic data originates.

            === ROOT CAUSE ANALYSIS ===
            Looking at the code files, I need to find where the problematic
            data or condition originates. The crash location may not be the
            root cause — the root cause is where the bad data or state is
            introduced.

            Let me read each file in the call chain:
            - I'll check what data flows between functions
            - I'll look for missing null/empty checks
            - I'll look for edge cases that aren't handled

            === FIX ===
            The fix should:
            1. Handle the edge case at the appropriate level (usually the root cause,
               not just the crash location)
            2. Return a sensible default or handle gracefully
            3. Not break existing functionality for normal inputs

            Let me also check: are there distractor files? I should focus only
            on the files in the stack trace and the files they import.

            I'll now write the corrected code for the affected file(s).
            After the fix, I'll mentally trace through the test cases to verify:
            - The previously-crashing input should now work
            - Normal inputs should still produce correct results
            - Edge cases (empty, None, etc.) should be handled
        """).strip()

        return reasoning

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]

        answer = extract_answer(response)
        code_changes = parse_code_blocks(answer)

        if not code_changes:
            return 0.0, {
                "reason": "no code blocks found in answer",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        # Apply changes to codebase
        modified_codebase = apply_code_changes(codebase, code_changes)

        # Run the tests
        test_code = variant["test"]
        results = run_tests(modified_codebase, test_code, timeout=10.0)

        total = results.get('total', 0)
        passed = results.get('passed', 0)
        errors = results.get('errors', 0)
        has_reasoning = bool(extract_reasoning(response))

        # Check if the code still crashes (errors indicate crashes/exceptions)
        still_crashes = errors > 0

        if total == 0:
            # No tests ran — likely import error or syntax error
            return 0.0, {
                "reason": "no tests ran — likely syntax or import error",
                "has_reasoning": has_reasoning,
                "files_changed": list(code_changes.keys()),
                "stderr": results.get('stderr', '')[:500],
            }

        if still_crashes and passed == 0:
            # Still crashes on all tests
            return 0.0, {
                "reason": "code still crashes",
                "has_reasoning": has_reasoning,
                "total": total,
                "passed": passed,
                "errors": errors,
                "files_changed": list(code_changes.keys()),
            }

        if passed == total and not still_crashes:
            # All tests pass, no crashes
            return 1.0, {
                "reason": "all tests pass, no crashes",
                "total": total,
                "passed": passed,
                "errors": errors,
                "has_reasoning": has_reasoning,
                "files_changed": list(code_changes.keys()),
            }

        # Partial: some tests pass but not all, or some errors
        if not still_crashes:
            # No crashes but wrong results on some tests
            score = 0.5 * (passed / total) if total > 0 else 0.0
            return score, {
                "reason": "no crash but some tests fail (wrong results)",
                "total": total,
                "passed": passed,
                "errors": errors,
                "score": score,
                "has_reasoning": has_reasoning,
                "files_changed": list(code_changes.keys()),
            }
        else:
            # Mixed: some pass, some crash
            score = 0.5 * (passed / total) if total > 0 else 0.0
            return score, {
                "reason": "mixed: some tests pass, some still crash",
                "total": total,
                "passed": passed,
                "errors": errors,
                "score": score,
                "has_reasoning": has_reasoning,
                "files_changed": list(code_changes.keys()),
            }
