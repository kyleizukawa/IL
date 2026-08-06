"""
Environment 3: Refactor While Preserving Behavior

Skill: Refactoring code to improve quality while keeping all tests passing.

The model is given working but messy code (long functions, duplicated logic,
poor naming) and must refactor it to improve structural quality while
ensuring all existing tests still pass.

Grading:
- 50% of score: test pass rate (all original tests must still pass)
- 50% of score: structural improvement (function count increased,
  max function length decreased, duplication reduced)

Difficulty scaling:
- easy: single file, one obvious refactoring target
- medium: 2 files, multiple refactoring opportunities
- hard: 3 files with distractors, complex refactoring requiring cross-file understanding
"""
import ast
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, parse_code_blocks, apply_code_changes,
    run_tests, compute_test_score, extract_reasoning,
)


# ── Domain definitions ──
# Each domain has:
#   - "messy": the code to refactor
#   - "tests": tests that verify behavior (must still pass after refactor)
#   - "module": the main module name
#   - "structural_goals": description of what structural improvements to make
#   - "min_functions": minimum function count expected after refactor
#   - "max_func_lines": maximum lines per function expected after refactor

DOMAINS = {
    "data_validator": {
        "module": "validator",
        "messy": textwrap.dedent('''
            def validate_user(data):
                """Validate a user dict. Returns list of error messages (empty = valid)."""
                errors = []
                if "name" not in data:
                    errors.append("name is required")
                else:
                    if not isinstance(data["name"], str):
                        errors.append("name must be a string")
                    else:
                        if len(data["name"]) == 0:
                            errors.append("name cannot be empty")
                        else:
                            if len(data["name"]) > 100:
                                errors.append("name too long")
                if "age" not in data:
                    errors.append("age is required")
                else:
                    if not isinstance(data["age"], int):
                        errors.append("age must be an integer")
                    else:
                        if data["age"] < 0:
                            errors.append("age cannot be negative")
                        else:
                            if data["age"] > 150:
                                errors.append("age too high")
                if "email" not in data:
                    errors.append("email is required")
                else:
                    if not isinstance(data["email"], str):
                        errors.append("email must be a string")
                    else:
                        if "@" not in data["email"]:
                            errors.append("email must contain @")
                        else:
                            if "." not in data["email"].split("@")[-1]:
                                errors.append("email must have valid domain")
                return errors

            def validate_product(data):
                """Validate a product dict. Returns list of error messages."""
                errors = []
                if "name" not in data:
                    errors.append("name is required")
                else:
                    if not isinstance(data["name"], str):
                        errors.append("name must be a string")
                    else:
                        if len(data["name"]) == 0:
                            errors.append("name cannot be empty")
                        else:
                            if len(data["name"]) > 200:
                                errors.append("name too long")
                if "price" not in data:
                    errors.append("price is required")
                else:
                    if not isinstance(data["price"], (int, float)):
                        errors.append("price must be a number")
                    else:
                        if data["price"] < 0:
                            errors.append("price cannot be negative")
                if "sku" not in data:
                    errors.append("sku is required")
                else:
                    if not isinstance(data["sku"], str):
                        errors.append("sku must be a string")
                    else:
                        if len(data["sku"]) == 0:
                            errors.append("sku cannot be empty")
                return errors
        ''').strip(),
        "tests": textwrap.dedent('''
            from validator import validate_user, validate_product
            def test_valid_user():
                errors = validate_user({"name": "Alice", "age": 30, "email": "alice@test.com"})
                assert errors == []
            def test_user_missing_name():
                errors = validate_user({"age": 30, "email": "alice@test.com"})
                assert "name is required" in errors
            def test_user_empty_name():
                errors = validate_user({"name": "", "age": 30, "email": "a@b.com"})
                assert "name cannot be empty" in errors
            def test_user_name_too_long():
                errors = validate_user({"name": "x" * 101, "age": 30, "email": "a@b.com"})
                assert "name too long" in errors
            def test_user_invalid_age():
                errors = validate_user({"name": "A", "age": "old", "email": "a@b.com"})
                assert "age must be an integer" in errors
            def test_user_negative_age():
                errors = validate_user({"name": "A", "age": -5, "email": "a@b.com"})
                assert "age cannot be negative" in errors
            def test_user_age_too_high():
                errors = validate_user({"name": "A", "age": 200, "email": "a@b.com"})
                assert "age too high" in errors
            def test_user_bad_email():
                errors = validate_user({"name": "A", "age": 30, "email": "noatsign"})
                assert "email must contain @" in errors
            def test_user_bad_domain():
                errors = validate_user({"name": "A", "age": 30, "email": "a@nodot"})
                assert "email must have valid domain" in errors
            def test_valid_product():
                errors = validate_product({"name": "Widget", "price": 9.99, "sku": "W123"})
                assert errors == []
            def test_product_missing_price():
                errors = validate_product({"name": "W", "sku": "S"})
                assert "price is required" in errors
            def test_product_negative_price():
                errors = validate_product({"name": "W", "price": -1, "sku": "S"})
                assert "price cannot be negative" in errors
            def test_product_missing_sku():
                errors = validate_product({"name": "W", "price": 1})
                assert "sku is required" in errors
        ''').strip(),
        "structural_goals": (
            "1. Extract a helper function for string field validation "
            "(required, type check, min/max length) to eliminate duplication "
            "between validate_user and validate_product.\n"
            "2. Extract a helper for required field checking.\n"
            "3. Reduce the deeply nested if/else chains to flat validation checks."
        ),
        "min_functions": 4,
        "max_func_lines": 15,
    },
    "calculator": {
        "module": "calculator",
        "messy": textwrap.dedent('''
            def calculate(operation, a, b):
                """Perform a calculation based on operation string."""
                if operation == "add":
                    result = a + b
                    return result
                elif operation == "subtract":
                    result = a - b
                    return result
                elif operation == "multiply":
                    result = a * b
                    return result
                elif operation == "divide":
                    if b == 0:
                        raise ValueError("Cannot divide by zero")
                    result = a / b
                    return result
                elif operation == "power":
                    result = 1
                    for i in range(int(b)):
                        result = result * a
                    return result
                elif operation == "mod":
                    if b == 0:
                        raise ValueError("Cannot mod by zero")
                    result = a % b
                    return result
                elif operation == "floor_div":
                    if b == 0:
                        raise ValueError("Cannot divide by zero")
                    result = a // b
                    return result
                elif operation == "abs_diff":
                    result = abs(a - b)
                    return result
                elif operation == "max":
                    result = a if a > b else b
                    return result
                elif operation == "min":
                    result = a if a < b else b
                    return result
                else:
                    raise ValueError(f"Unknown operation: {operation}")

            def calculate_batch(operations):
                """Perform multiple calculations. Each item is (op, a, b)."""
                results = []
                for item in operations:
                    op = item[0]
                    a = item[1]
                    b = item[2]
                    if op == "add":
                        result = a + b
                        results.append(result)
                    elif op == "subtract":
                        result = a - b
                        results.append(result)
                    elif op == "multiply":
                        result = a * b
                        results.append(result)
                    elif op == "divide":
                        if b == 0:
                            raise ValueError("Cannot divide by zero")
                        result = a / b
                        results.append(result)
                    elif op == "power":
                        result = 1
                        for i in range(int(b)):
                            result = result * a
                        results.append(result)
                    elif op == "mod":
                        if b == 0:
                            raise ValueError("Cannot mod by zero")
                        result = a % b
                        results.append(result)
                    elif op == "floor_div":
                        if b == 0:
                            raise ValueError("Cannot divide by zero")
                        result = a // b
                        results.append(result)
                    elif op == "abs_diff":
                        result = abs(a - b)
                        results.append(result)
                    elif op == "max":
                        result = a if a > b else b
                        results.append(result)
                    elif op == "min":
                        result = a if a < b else b
                        results.append(result)
                    else:
                        raise ValueError(f"Unknown operation: {op}")
                return results
        ''').strip(),
        "tests": textwrap.dedent('''
            from calculator import calculate, calculate_batch
            def test_add():
                assert calculate("add", 2, 3) == 5
            def test_subtract():
                assert calculate("subtract", 10, 4) == 6
            def test_multiply():
                assert calculate("multiply", 3, 4) == 12
            def test_divide():
                assert calculate("divide", 10, 2) == 5
            def test_divide_by_zero():
                try:
                    calculate("divide", 1, 0)
                    assert False, "Should raise"
                except ValueError:
                    pass
            def test_power():
                assert calculate("power", 2, 3) == 8
            def test_mod():
                assert calculate("mod", 10, 3) == 1
            def test_mod_by_zero():
                try:
                    calculate("mod", 1, 0)
                    assert False, "Should raise"
                except ValueError:
                    pass
            def test_floor_div():
                assert calculate("floor_div", 10, 3) == 3
            def test_abs_diff():
                assert calculate("abs_diff", 5, 8) == 3
            def test_max():
                assert calculate("max", 3, 7) == 7
            def test_min():
                assert calculate("min", 3, 7) == 3
            def test_unknown_op():
                try:
                    calculate("unknown", 1, 2)
                    assert False, "Should raise"
                except ValueError:
                    pass
            def test_batch():
                results = calculate_batch([("add", 1, 2), ("multiply", 3, 4)])
                assert results == [3, 12]
            def test_batch_with_error():
                try:
                    calculate_batch([("add", 1, 2), ("divide", 1, 0)])
                    assert False, "Should raise"
                except ValueError:
                    pass
        ''').strip(),
        "structural_goals": (
            "1. Extract each operation into its own helper function or use a dispatch "
            "dictionary mapping operation names to functions.\n"
            "2. Eliminate the duplicated logic between calculate and calculate_batch - "
            "calculate_batch should call calculate for each item.\n"
            "3. Remove the unnecessary 'result = ...; return result' pattern - just "
            "return directly.\n"
            "4. Consolidate the divide-by-zero checks."
        ),
        "min_functions": 4,
        "max_func_lines": 15,
    },
    "report_generator": {
        "module": "report",
        "messy": textwrap.dedent('''
            def generate_report(data, format_type):
                """Generate a report from data in the specified format."""
                if format_type == "text":
                    lines = []
                    lines.append("REPORT")
                    lines.append("======")
                    lines.append("")
                    if "title" in data:
                        lines.append("Title: " + data["title"])
                    if "author" in data:
                        lines.append("Author: " + data["author"])
                    lines.append("")
                    if "items" in data:
                        lines.append("Items:")
                        for item in data["items"]:
                            lines.append("  - " + item)
                    lines.append("")
                    if "summary" in data:
                        lines.append("Summary: " + data["summary"])
                    return "\\n".join(lines)
                elif format_type == "html":
                    lines = []
                    lines.append("<html>")
                    lines.append("<head><title>Report</title></head>")
                    lines.append("<body>")
                    if "title" in data:
                        lines.append("<h1>" + data["title"] + "</h1>")
                    if "author" in data:
                        lines.append("<p>Author: " + data["author"] + "</p>")
                    if "items" in data:
                        lines.append("<ul>")
                        for item in data["items"]:
                            lines.append("<li>" + item + "</li>")
                        lines.append("</ul>")
                    if "summary" in data:
                        lines.append("<p>" + data["summary"] + "</p>")
                    lines.append("</body>")
                    lines.append("</html>")
                    return "\\n".join(lines)
                elif format_type == "markdown":
                    lines = []
                    if "title" in data:
                        lines.append("# " + data["title"])
                        lines.append("")
                    if "author" in data:
                        lines.append("**Author:** " + data["author"])
                        lines.append("")
                    if "items" in data:
                        lines.append("## Items")
                        for item in data["items"]:
                            lines.append("- " + item)
                        lines.append("")
                    if "summary" in data:
                        lines.append("## Summary")
                        lines.append(data["summary"])
                    return "\\n".join(lines)
                else:
                    raise ValueError(f"Unknown format: {format_type}")

            def generate_summary(data):
                """Generate a one-line summary string."""
                parts = []
                if "title" in data:
                    parts.append(data["title"])
                if "author" in data:
                    parts.append("by " + data["author"])
                if "items" in data:
                    parts.append(str(len(data["items"])) + " items")
                if "summary" in data:
                    parts.append(data["summary"])
                return " | ".join(parts)
        ''').strip(),
        "tests": textwrap.dedent('''
            from report import generate_report, generate_summary
            def test_text_report():
                data = {"title": "Test", "author": "Bob", "items": ["a", "b"], "summary": "Done"}
                report = generate_report(data, "text")
                assert "REPORT" in report
                assert "Title: Test" in report
                assert "Author: Bob" in report
                assert "  - a" in report
                assert "  - b" in report
                assert "Summary: Done" in report
            def test_text_report_minimal():
                report = generate_report({}, "text")
                assert "REPORT" in report
            def test_html_report():
                data = {"title": "T", "items": ["x"]}
                report = generate_report(data, "html")
                assert "<html>" in report
                assert "<h1>T</h1>" in report
                assert "<li>x</li>" in report
                assert "</html>" in report
            def test_html_report_full():
                data = {"title": "T", "author": "A", "items": ["x"], "summary": "S"}
                report = generate_report(data, "html")
                assert "<p>Author: A</p>" in report
                assert "<p>S</p>" in report
            def test_markdown_report():
                data = {"title": "T", "author": "A", "items": ["x"], "summary": "S"}
                report = generate_report(data, "markdown")
                assert "# T" in report
                assert "**Author:** A" in report
                assert "- x" in report
                assert "## Summary" in report
            def test_markdown_minimal():
                report = generate_report({}, "markdown")
                assert report == ""
            def test_unknown_format():
                try:
                    generate_report({}, "xml")
                    assert False, "Should raise"
                except ValueError:
                    pass
            def test_summary():
                data = {"title": "T", "author": "A", "items": ["x", "y"], "summary": "S"}
                s = generate_summary(data)
                assert "T" in s
                assert "by A" in s
                assert "2 items" in s
                assert "S" in s
            def test_summary_empty():
                assert generate_summary({}) == ""
        ''').strip(),
        "structural_goals": (
            "1. Extract helper functions for each format (text, html, markdown) to "
            "reduce the giant generate_report function.\n"
            "2. Extract a helper for rendering items in each format to eliminate "
            "the repeated item-looping logic.\n"
            "3. Use a dispatch dictionary to map format names to formatter functions.\n"
            "4. Reduce the max function length - no function should be longer than 15 lines."
        ),
        "min_functions": 5,
        "max_func_lines": 15,
    },
}


# ── Distractor code ──

DISTRACTORS = [
    textwrap.dedent('''
        def format_table(rows):
            """Format rows as a text table (not relevant to the task)."""
            if not rows:
                return ""
            widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
            lines = []
            for row in rows:
                lines.append(" | ".join(str(val).ljust(w) for val, w in zip(row, widths)))
            return "\\n".join(lines)
    ''').strip(),
    textwrap.dedent('''
        def colorize(text, color):
            """Add ANSI color codes (not relevant to the task)."""
            colors = {"red": 31, "green": 32, "yellow": 33, "blue": 34}
            code = colors.get(color, 0)
            return f"\\033[{code}m{text}\\033[0m" if code else text
    ''').strip(),
    textwrap.dedent('''
        def memoize(func):
            """Memoize a function (not relevant to the task)."""
            cache = {}
            def wrapper(*args):
                if args not in cache:
                    cache[args] = func(*args)
                return cache[args]
            return wrapper
    ''').strip(),
]


def _count_functions(code: str) -> int:
    """Count top-level and nested function definitions in code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            count += 1
    return count


def _max_function_lines(code: str) -> int:
    """Return the maximum number of lines in any single function."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 999
    max_lines = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, 'end_lineno') and node.end_lineno:
                lines = node.end_lineno - node.lineno + 1
                max_lines = max(max_lines, lines)
    return max_lines


def _count_duplication(code: str) -> int:
    """Estimate code duplication by counting repeated 3-line sequences."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 999
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    if len(lines) < 6:
        return 0
    sequences = []
    for i in range(len(lines) - 2):
        seq = tuple(lines[i:i+3])
        sequences.append(seq)
    from collections import Counter
    counts = Counter(sequences)
    duplication = sum(c - 1 for c in counts.values() if c > 1)
    return duplication


@register_env
class RefactorPreserveEnv(AgenticEnv):
    name = "refactor_preserve"
    skill = "Refactoring code to improve quality while keeping all tests passing"
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
        module = domain["module"]
        codebase = {f"{module}.py": domain["messy"]}

        for idx in params.get("distractor_indices", []):
            distractor = DISTRACTORS[idx]
            codebase[f"helper_{idx}.py"] = distractor

        codebase["test_behavior.py"] = domain["tests"]
        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]

        # Compute original metrics for reference
        orig_funcs = _count_functions(domain["messy"])
        orig_max_lines = _max_function_lines(domain["messy"])
        orig_duplication = _count_duplication(domain["messy"])

        lines = []
        lines.append("You are a software engineer refactoring an existing codebase.")
        lines.append("")
        lines.append("The code works correctly (tests in `test_behavior.py` pass), but it has")
        lines.append("poor code quality: long functions, duplicated logic, and deep nesting.")
        lines.append("Your task is to:")
        lines.append("1. Refactor the code to improve quality while preserving ALL behavior")
        lines.append("2. All existing tests MUST still pass after refactoring")
        lines.append("3. Improve the code structure as described in the goals below")
        lines.append("")
        lines.append("=== STRUCTURAL GOALS ===")
        lines.append(domain["structural_goals"])
        lines.append("")
        lines.append(f"Current metrics: {orig_funcs} functions, max function length "
                     f"{orig_max_lines} lines, duplication score {orig_duplication}")
        lines.append(f"Target: at least {domain['min_functions']} functions, "
                     f"max function length <= {domain['max_func_lines']} lines")
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")

        lines.append("Provide your refactored code in the following format:")
        lines.append("<reasoning>")
        lines.append("...identify what's messy, explain your refactoring strategy,")
        lines.append("describe each extraction/improvement you'll make...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append(f"```python:{module}.py")
        lines.append("# the refactored code")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]
        # The solution is the messy code refactored - for grading we don't
        # provide a fixed solution, we grade based on tests + structure
        return {f"{module}.py": domain["messy"]}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]
        orig_funcs = _count_functions(domain["messy"])
        orig_max_lines = _max_function_lines(domain["messy"])
        orig_duplication = _count_duplication(domain["messy"])

        reasoning = textwrap.dedent(f"""
            Let me carefully analyze the code to understand what needs refactoring.

            First, I'll read {module}.py line by line to identify the code quality issues:

            Current metrics show {orig_funcs} functions, max function length {orig_max_lines} lines,
            and duplication score {orig_duplication}. These are the problems I need to fix.

            Looking at the code, I can identify these specific issues:
            - The functions are very long with deeply nested if/else chains
            - There is significant code duplication between functions
            - The logic could be extracted into smaller, focused helper functions

            Let me trace through the tests to understand the exact behavior I must preserve:
            - test_behavior.py contains the tests that define the expected behavior
            - Each test calls specific functions with specific inputs and checks outputs
            - I must ensure all of these still pass after refactoring

            My refactoring strategy:
            1. First, I'll identify the duplicated patterns. Looking at the code carefully,
            I can see repeated validation/checking logic that can be extracted into helpers.
            2. I'll extract helper functions that encapsulate the repeated logic, making
            sure to preserve the exact error messages and behavior.
            3. I'll flatten the deeply nested if/else chains into sequential checks with
            early returns or continue patterns.
            4. I'll use a dispatch pattern where appropriate to replace long if/elif chains.

            Let me be very careful about preserving behavior:
            - Error messages must match exactly (tests check for specific strings)
            - Return types must be the same (lists, strings, etc.)
            - Exception types must be the same (ValueError)
            - Edge cases (empty inputs, missing fields) must be handled the same way

            Now let me write the refactored code. I'll extract helper functions one at a time,
            verifying that each extraction preserves the behavior tested by the tests.

            After refactoring, I'll mentally trace through each test to make sure it still passes:
            - Each test's expected output must match what my refactored code produces
            - The error messages in exceptions must be identical
            - The structure of returned data must be the same

            The refactored code should have more functions, shorter max function length,
            and less duplication while passing all the same tests.
        """).strip()

        return reasoning

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]

        answer = extract_answer(response)
        code_changes = parse_code_blocks(answer)

        if not code_changes:
            return 0.0, {
                "reason": "no code blocks found in answer",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        modified_codebase = apply_code_changes(codebase, code_changes)

        # Run behavior tests (must all pass)
        test_results = run_tests(modified_codebase, domain["tests"], timeout=10.0)
        test_score, test_breakdown = compute_test_score(test_results)

        # Compute structural metrics
        refactored_code = code_changes.get(f"{module}.py", "")
        orig_funcs = _count_functions(domain["messy"])
        orig_max_lines = _max_function_lines(domain["messy"])
        orig_duplication = _count_duplication(domain["messy"])

        new_funcs = _count_functions(refactored_code)
        new_max_lines = _max_function_lines(refactored_code)
        new_duplication = _count_duplication(refactored_code)

        # Structural score components
        # 1. Function count improvement (more functions = better decomposition)
        func_target = domain["min_functions"]
        func_score = min(1.0, new_funcs / func_target) if func_target > 0 else 0.0

        # 2. Max function length reduction (shorter = better)
        max_len_target = domain["max_func_lines"]
        if new_max_lines <= max_len_target:
            len_score = 1.0
        elif new_max_lines < orig_max_lines:
            # Partial credit for improvement
            len_score = max(0.0, (orig_max_lines - new_max_lines) / (orig_max_lines - max_len_target))
        else:
            len_score = 0.0

        # 3. Duplication reduction
        if orig_duplication > 0:
            dup_score = max(0.0, 1.0 - (new_duplication / orig_duplication))
        else:
            dup_score = 1.0  # No duplication to begin with

        structural_score = (func_score * 0.4 + len_score * 0.4 + dup_score * 0.2)

        # If tests don't all pass, structural improvement doesn't count
        if test_score < 1.0:
            score = test_score * 0.5
        else:
            score = 0.5 + 0.5 * structural_score

        breakdown = {
            "test_results": test_breakdown,
            "test_score": test_score,
            "structural_score": structural_score,
            "metrics": {
                "orig_functions": orig_funcs,
                "new_functions": new_funcs,
                "orig_max_lines": orig_max_lines,
                "new_max_lines": new_max_lines,
                "orig_duplication": orig_duplication,
                "new_duplication": new_duplication,
                "func_score": func_score,
                "len_score": len_score,
                "dup_score": dup_score,
            },
            "has_reasoning": bool(extract_reasoning(response)),
            "files_changed": list(code_changes.keys()),
            "changed_target": f"{module}.py" in code_changes,
            "score": score,
        }

        return score, breakdown
