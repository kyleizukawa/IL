"""
Environment 11: Error Handling Addition

Skill: Adding proper error handling to code that crashes on edge cases.

The model is given code that works on normal inputs but crashes on edge cases
(empty input, None, wrong type, division by zero, index out of range, missing
keys) and must add error handling. The grader runs edge case tests against the
model's code. Score = fraction of tests that don't crash AND return correct
results.

Difficulty scaling:
- easy: 1 edge case, obvious crash (e.g., division by zero)
- medium: 2-3 edge cases across different functions
- hard: 4+ edge cases with subtle failure modes (missing keys, type errors,
  None propagation, empty collections)
"""
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, parse_code_blocks, apply_code_changes,
    extract_reasoning, run_tests, compute_test_score,
)


# ── Domain templates ──
# Each domain has: crash-prone code, correct code with error handling, edge case tests

DOMAINS = {
    "calculator": {
        "crash_prone": textwrap.dedent('''
            def divide(a, b):
                return a / b

            def mean(numbers):
                return sum(numbers) / len(numbers)

            def median(numbers):
                sorted_nums = sorted(numbers)
                n = len(sorted_nums)
                mid = n // 2
                if n % 2 == 0:
                    return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
                return sorted_nums[mid]

            def percentage(part, total):
                return (part / total) * 100

            def safe_mod(a, b):
                return a % b
        ''').strip(),
        "correct": textwrap.dedent('''
            def divide(a, b):
                if b == 0:
                    raise ValueError("Cannot divide by zero")
                return a / b

            def mean(numbers):
                if not numbers:
                    raise ValueError("Cannot compute mean of empty list")
                return sum(numbers) / len(numbers)

            def median(numbers):
                if not numbers:
                    raise ValueError("Cannot compute median of empty list")
                sorted_nums = sorted(numbers)
                n = len(sorted_nums)
                mid = n // 2
                if n % 2 == 0:
                    return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
                return sorted_nums[mid]

            def percentage(part, total):
                if total == 0:
                    raise ValueError("Total cannot be zero")
                return (part / total) * 100

            def safe_mod(a, b):
                if b == 0:
                    raise ValueError("Cannot compute modulo with zero divisor")
                return a % b
        ''').strip(),
        "tests": textwrap.dedent('''
            from calculator import divide, mean, median, percentage, safe_mod
            import traceback

            def test_divide_normal():
                assert divide(10, 2) == 5.0

            def test_divide_by_zero():
                try:
                    divide(10, 0)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except ZeroDivisionError:
                    assert False, "Should raise ValueError, not ZeroDivisionError"

            def test_divide_floats():
                assert abs(divide(7.5, 2.5) - 3.0) < 1e-9

            def test_mean_normal():
                assert mean([1, 2, 3, 4, 5]) == 3.0

            def test_mean_empty():
                try:
                    mean([])
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except ZeroDivisionError:
                    assert False, "Should raise ValueError, not ZeroDivisionError"

            def test_mean_single():
                assert mean([42]) == 42.0

            def test_median_normal():
                assert median([3, 1, 2]) == 2

            def test_median_even():
                assert median([1, 2, 3, 4]) == 2.5

            def test_median_empty():
                try:
                    median([])
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except IndexError:
                    assert False, "Should raise ValueError, not IndexError"

            def test_percentage_normal():
                assert percentage(25, 100) == 25.0

            def test_percentage_zero_total():
                try:
                    percentage(10, 0)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except ZeroDivisionError:
                    assert False, "Should raise ValueError, not ZeroDivisionError"

            def test_safe_mod_normal():
                assert safe_mod(10, 3) == 1

            def test_safe_mod_zero():
                try:
                    safe_mod(10, 0)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except ZeroDivisionError:
                    assert False, "Should raise ValueError, not ZeroDivisionError"
        ''').strip(),
        "edge_cases": [
            "divide by zero → should raise ValueError, not ZeroDivisionError",
            "mean of empty list → should raise ValueError, not ZeroDivisionError",
            "median of empty list → should raise ValueError, not IndexError",
            "percentage with zero total → should raise ValueError, not ZeroDivisionError",
            "safe_mod with zero divisor → should raise ValueError, not ZeroDivisionError",
        ],
    },
    "list_processor": {
        "crash_prone": textwrap.dedent('''
            def first_element(lst):
                return lst[0]

            def last_element(lst):
                return lst[-1]

            def second_largest(numbers):
                unique = list(set(numbers))
                unique.sort()
                return unique[-2]

            def pairwise_sum(a, b):
                return [a[i] + b[i] for i in range(len(a))]

            def nested_get(data, keys):
                result = data
                for key in keys:
                    result = result[key]
                return result
        ''').strip(),
        "correct": textwrap.dedent('''
            def first_element(lst):
                if not lst:
                    raise ValueError("List is empty, cannot get first element")
                return lst[0]

            def last_element(lst):
                if not lst:
                    raise ValueError("List is empty, cannot get last element")
                return lst[-1]

            def second_largest(numbers):
                if len(numbers) < 2:
                    raise ValueError("Need at least 2 elements for second largest")
                unique = list(set(numbers))
                if len(unique) < 2:
                    raise ValueError("Need at least 2 unique elements for second largest")
                unique.sort()
                return unique[-2]

            def pairwise_sum(a, b):
                if len(a) != len(b):
                    raise ValueError("Lists must have equal length")
                if not a:
                    return []
                return [a[i] + b[i] for i in range(len(a))]

            def nested_get(data, keys):
                if not keys:
                    raise ValueError("Keys list cannot be empty")
                result = data
                for key in keys:
                    if not isinstance(result, dict) or key not in result:
                        raise KeyError(f"Key '{key}' not found in nested structure")
                    result = result[key]
                return result
        ''').strip(),
        "tests": textwrap.dedent('''
            from list_processor import first_element, last_element, second_largest, pairwise_sum, nested_get

            def test_first_normal():
                assert first_element([1, 2, 3]) == 1

            def test_first_empty():
                try:
                    first_element([])
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except IndexError:
                    assert False, "Should raise ValueError, not IndexError"

            def test_last_normal():
                assert last_element([1, 2, 3]) == 3

            def test_last_empty():
                try:
                    last_element([])
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except IndexError:
                    assert False, "Should raise ValueError, not IndexError"

            def test_second_largest_normal():
                assert second_largest([3, 1, 4, 1, 5]) == 4

            def test_second_largest_too_few():
                try:
                    second_largest([5])
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except IndexError:
                    assert False, "Should raise ValueError, not IndexError"

            def test_second_largest_all_same():
                try:
                    second_largest([3, 3, 3])
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except IndexError:
                    assert False, "Should raise ValueError, not IndexError"

            def test_pairwise_normal():
                assert pairwise_sum([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

            def test_pairwise_unequal():
                try:
                    pairwise_sum([1, 2], [3, 4, 5])
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except IndexError:
                    assert False, "Should raise ValueError, not IndexError"

            def test_pairwise_empty():
                assert pairwise_sum([], []) == []

            def test_nested_get_normal():
                data = {"a": {"b": {"c": 42}}}
                assert nested_get(data, ["a", "b", "c"]) == 42

            def test_nested_get_missing_key():
                try:
                    nested_get({"a": {}}, ["a", "b"])
                    assert False, "Should have raised KeyError"
                except KeyError:
                    assert True
                except TypeError:
                    assert False, "Should raise KeyError, not TypeError"

            def test_nested_get_empty_keys():
                try:
                    nested_get({"a": 1}, [])
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
        ''').strip(),
        "edge_cases": [
            "first_element on empty list → should raise ValueError, not IndexError",
            "last_element on empty list → should raise ValueError, not IndexError",
            "second_largest with <2 elements → should raise ValueError, not IndexError",
            "second_largest with all same values → should raise ValueError, not IndexError",
            "pairwise_sum with unequal lengths → should raise ValueError, not IndexError",
            "nested_get with missing key → should raise KeyError, not TypeError",
            "nested_get with empty keys → should raise ValueError",
        ],
    },
    "string_parser": {
        "crash_prone": textwrap.dedent('''
            def parse_int(s):
                return int(s)

            def split_and_get(text, delimiter, index):
                parts = text.split(delimiter)
                return parts[index]

            def extract_between(text, start, end):
                start_idx = text.index(start)
                end_idx = text.index(end, start_idx + len(start))
                return text[start_idx + len(start):end_idx]

            def parse_key_value(text):
                key, value = text.split("=")
                return key.strip(), value.strip()

            def nth_word(text, n):
                words = text.split()
                return words[n]
        ''').strip(),
        "correct": textwrap.dedent('''
            def parse_int(s):
                if s is None:
                    raise ValueError("Input cannot be None")
                try:
                    return int(s)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot parse '{s}' as integer: {e}")

            def split_and_get(text, delimiter, index):
                if text is None:
                    raise ValueError("Text cannot be None")
                parts = text.split(delimiter)
                if index < 0 or index >= len(parts):
                    raise IndexError(f"Index {index} out of range for {len(parts)} parts")
                return parts[index]

            def extract_between(text, start, end):
                if text is None:
                    raise ValueError("Text cannot be None")
                try:
                    start_idx = text.index(start)
                except ValueError:
                    raise ValueError(f"Start marker '{start}' not found in text")
                try:
                    end_idx = text.index(end, start_idx + len(start))
                except ValueError:
                    raise ValueError(f"End marker '{end}' not found after start marker")
                return text[start_idx + len(start):end_idx]

            def parse_key_value(text):
                if text is None:
                    raise ValueError("Text cannot be None")
                if "=" not in text:
                    raise ValueError("Missing '=' delimiter in key-value pair")
                parts = text.split("=", 1)
                if len(parts) != 2:
                    raise ValueError("Invalid key-value format")
                return parts[0].strip(), parts[1].strip()

            def nth_word(text, n):
                if text is None:
                    raise ValueError("Text cannot be None")
                words = text.split()
                if n < 0 or n >= len(words):
                    raise IndexError(f"Word index {n} out of range for {len(words)} words")
                return words[n]
        ''').strip(),
        "tests": textwrap.dedent('''
            from string_parser import parse_int, split_and_get, extract_between, parse_key_value, nth_word

            def test_parse_int_normal():
                assert parse_int("42") == 42

            def test_parse_int_none():
                try:
                    parse_int(None)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except TypeError:
                    assert False, "Should raise ValueError, not TypeError"

            def test_parse_int_invalid():
                try:
                    parse_int("abc")
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True

            def test_split_and_get_normal():
                assert split_and_get("a,b,c", ",", 1) == "b"

            def test_split_and_get_out_of_range():
                try:
                    split_and_get("a,b", ",", 5)
                    assert False, "Should have raised IndexError"
                except IndexError:
                    assert True

            def test_split_and_get_none():
                try:
                    split_and_get(None, ",", 0)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except AttributeError:
                    assert False, "Should raise ValueError, not AttributeError"

            def test_extract_between_normal():
                assert extract_between("hello [world] end", "[", "]") == "world"

            def test_extract_between_no_start():
                try:
                    extract_between("hello world", "[", "]")
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True

            def test_extract_between_no_end():
                try:
                    extract_between("hello [world end", "[", "]")
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True

            def test_parse_key_value_normal():
                assert parse_key_value("name = Alice") == ("name", "Alice")

            def test_parse_key_value_no_delimiter():
                try:
                    parse_key_value("just a key")
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True

            def test_parse_key_value_none():
                try:
                    parse_key_value(None)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except AttributeError:
                    assert False, "Should raise ValueError, not AttributeError"

            def test_nth_word_normal():
                assert nth_word("the quick brown fox", 2) == "brown"

            def test_nth_word_out_of_range():
                try:
                    nth_word("hello world", 5)
                    assert False, "Should have raised IndexError"
                except IndexError:
                    assert True

            def test_nth_word_none():
                try:
                    nth_word(None, 0)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True
                except AttributeError:
                    assert False, "Should raise ValueError, not AttributeError"
        ''').strip(),
        "edge_cases": [
            "parse_int(None) → should raise ValueError, not TypeError",
            "parse_int('abc') → should raise ValueError with clear message",
            "split_and_get with out-of-range index → should raise IndexError",
            "split_and_get(None, ...) → should raise ValueError, not AttributeError",
            "extract_between with missing start marker → should raise ValueError, not ValueError (raw)",
            "extract_between with missing end marker → should raise ValueError",
            "parse_key_value without '=' → should raise ValueError, not ValueError (unpack)",
            "parse_key_value(None) → should raise ValueError, not AttributeError",
            "nth_word with out-of-range index → should raise IndexError",
            "nth_word(None, ...) → should raise ValueError, not AttributeError",
        ],
    },
    "data_aggregator": {
        "crash_prone": textwrap.dedent('''
            from collections import defaultdict

            def sum_by_key(records, key):
                total = 0
                for record in records:
                    total += record[key]
                return total

            def group_and_count(records, key):
                counts = defaultdict(int)
                for record in records:
                    counts[record[key]] += 1
                return dict(counts)

            def max_by_key(records, key):
                return max(records, key=lambda r: r[key])

            def aggregate_field(records, field, operation):
                values = [r[field] for r in records]
                return operation(values)

            def merge_records(base, override):
                result = {}
                for k in base:
                    result[k] = override[k] if k in override else base[k]
                return result
        ''').strip(),
        "correct": textwrap.dedent('''
            from collections import defaultdict

            def sum_by_key(records, key):
                if not records:
                    raise ValueError("Cannot sum over empty records list")
                total = 0
                for record in records:
                    if key not in record:
                        raise KeyError(f"Key '{key}' not found in record")
                    total += record[key]
                return total

            def group_and_count(records, key):
                if not records:
                    return {}
                counts = defaultdict(int)
                for record in records:
                    if key not in record:
                        raise KeyError(f"Key '{key}' not found in record")
                    counts[record[key]] += 1
                return dict(counts)

            def max_by_key(records, key):
                if not records:
                    raise ValueError("Cannot find max in empty records list")
                for record in records:
                    if key not in record:
                        raise KeyError(f"Key '{key}' not found in record")
                return max(records, key=lambda r: r[key])

            def aggregate_field(records, field, operation):
                if not records:
                    raise ValueError("Cannot aggregate empty records list")
                values = []
                for r in records:
                    if field not in r:
                        raise KeyError(f"Field '{field}' not found in record")
                    values.append(r[field])
                return operation(values)

            def merge_records(base, override):
                if not isinstance(base, dict) or not isinstance(override, dict):
                    raise TypeError("Both base and override must be dictionaries")
                result = {}
                for k in base:
                    if k in override:
                        result[k] = override[k]
                    else:
                        result[k] = base[k]
                return result
        ''').strip(),
        "tests": textwrap.dedent('''
            from data_aggregator import sum_by_key, group_and_count, max_by_key, aggregate_field, merge_records

            def test_sum_by_key_normal():
                records = [{"amount": 10}, {"amount": 20}, {"amount": 30}]
                assert sum_by_key(records, "amount") == 60

            def test_sum_by_key_empty():
                try:
                    sum_by_key([], "amount")
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True

            def test_sum_by_key_missing_key():
                try:
                    sum_by_key([{"amount": 10}, {"price": 20}], "amount")
                    assert False, "Should have raised KeyError"
                except KeyError:
                    assert True

            def test_group_and_count_normal():
                records = [{"type": "A"}, {"type": "B"}, {"type": "A"}]
                assert group_and_count(records, "type") == {"A": 2, "B": 1}

            def test_group_and_count_empty():
                assert group_and_count([], "type") == {}

            def test_group_and_count_missing_key():
                try:
                    group_and_count([{"type": "A"}, {"category": "B"}], "type")
                    assert False, "Should have raised KeyError"
                except KeyError:
                    assert True

            def test_max_by_key_normal():
                records = [{"score": 10}, {"score": 30}, {"score": 20}]
                assert max_by_key(records, "score") == {"score": 30}

            def test_max_by_key_empty():
                try:
                    max_by_key([], "score")
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True

            def test_max_by_key_missing_key():
                try:
                    max_by_key([{"score": 10}, {"rating": 20}], "score")
                    assert False, "Should have raised KeyError"
                except KeyError:
                    assert True

            def test_aggregate_field_normal():
                records = [{"val": 1}, {"val": 2}, {"val": 3}]
                assert aggregate_field(records, "val", sum) == 6

            def test_aggregate_field_empty():
                try:
                    aggregate_field([], "val", sum)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    assert True

            def test_aggregate_field_missing_field():
                try:
                    aggregate_field([{"val": 1}, {"score": 2}], "val", sum)
                    assert False, "Should have raised KeyError"
                except KeyError:
                    assert True

            def test_merge_records_normal():
                base = {"a": 1, "b": 2, "c": 3}
                override = {"b": 20, "d": 4}
                result = merge_records(base, override)
                assert result == {"a": 1, "b": 20, "c": 3}

            def test_merge_records_non_dict():
                try:
                    merge_records("not a dict", {"a": 1})
                    assert False, "Should have raised TypeError"
                except TypeError:
                    assert True
                except Exception as e:
                    assert False, f"Should raise TypeError, not {type(e).__name__}"
        ''').strip(),
        "edge_cases": [
            "sum_by_key on empty list → should raise ValueError",
            "sum_by_key with missing key in record → should raise KeyError",
            "group_and_count with missing key → should raise KeyError",
            "max_by_key on empty list → should raise ValueError",
            "max_by_key with missing key → should raise KeyError",
            "aggregate_field on empty list → should raise ValueError",
            "aggregate_field with missing field → should raise KeyError",
            "merge_records with non-dict input → should raise TypeError",
        ],
    },
}


# ── Distractor code ──

DISTRACTORS = [
    textwrap.dedent('''
        def format_bytes(n):
            units = [("B", 1), ("KB", 1024), ("MB", 1048576), ("GB", 1073741824)]
            for unit, size in reversed(units):
                if n >= size:
                    return f"{n / size:.1f}{unit}"
            return f"{n}B"
    ''').strip(),
    textwrap.dedent('''
        def slugify(text):
            import re
            text = text.lower().strip()
            text = re.sub(r"[^a-z0-9\\s-]", "", text)
            text = re.sub(r"[\\s-]+", "-", text)
            return text.strip("-")
    ''').strip(),
    textwrap.dedent('''
        def clamp(value, low, high):
            return max(low, min(value, high))
    ''').strip(),
]


@register_env
class ErrorHandlingEnv(AgenticEnv):
    name = "error_handling"
    skill = "Adding proper error handling to code that crashes on edge cases"
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
        main_module = params["domain"]
        codebase = {f"{main_module}.py": domain["crash_prone"]}

        # Add test file
        codebase["test_edge_cases.py"] = domain["tests"]

        for idx in params.get("distractor_indices", []):
            codebase[f"helper_{idx}.py"] = DISTRACTORS[idx]

        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]

        lines = []
        lines.append("You are a software engineer adding error handling to crash-prone code.")
        lines.append("")
        lines.append("The code in the main module works on normal inputs but crashes on edge cases")
        lines.append("(empty inputs, None, division by zero, index out of range, missing keys, etc.).")
        lines.append("")
        lines.append("Your task is to:")
        lines.append("1. Read each function and identify what edge cases would cause it to crash")
        lines.append("2. Add proper error handling (raise ValueError, KeyError, IndexError, TypeError with clear messages)")
        lines.append("3. The error handling should produce the CORRECT exception type as expected by the tests")
        lines.append("4. Do NOT change the normal (happy path) behavior of any function")
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")

        lines.append("The tests in `test_edge_cases.py` will check that your error handling produces")
        lines.append("the correct exception types (not just any crash). Read the tests carefully!")
        lines.append("")
        lines.append("Provide your fixed code in the following format:")
        lines.append("<reasoning>")
        lines.append("...trace each function with edge case inputs, identify crash points, explain the fix...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append(f"```python:{main_module}.py")
        lines.append("# the code with error handling added")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]
        return {f"{main_module}.py": domain["correct"]}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]
        edge_cases = domain["edge_cases"]

        lines = []
        lines.append(f"I need to add error handling to {main_module}.py. Let me trace each function")
        lines.append("with edge case inputs to identify where it would crash.")
        lines.append("")
        lines.append("First, let me read the test file to understand what exception types are expected:")
        lines.append("- The tests check for specific exception types (ValueError, KeyError, IndexError, TypeError)")
        lines.append("- The tests verify that the WRONG exception type is not raised (e.g., ZeroDivisionError instead of ValueError)")
        lines.append("")

        for i, ec in enumerate(edge_cases, 1):
            lines.append(f"Edge case {i}: {ec}")
            lines.append(f"  - I need to trace the code with this input to see where it crashes.")
            lines.append(f"  - Then add a check before the crash point that raises the correct exception type.")
            lines.append("")

        lines.append("Let me now trace through each function:")
        lines.append("")
        lines.append("For each function, I'll:")
        lines.append("1. Identify the crash point (which line raises the wrong exception)")
        lines.append("2. Add a guard clause before that line")
        lines.append("3. Raise the correct exception type with a descriptive message")
        lines.append("4. Make sure the normal path still works unchanged")
        lines.append("")
        lines.append(f"I've identified {len(edge_cases)} edge cases to handle. Let me write the corrected version")

        return "\n".join(lines)

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]

        answer = extract_answer(response)
        code_changes = parse_code_blocks(answer)

        if not code_changes:
            return 0.0, {
                "reason": "no code blocks found in answer",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        target_file = f"{main_module}.py"
        if target_file not in code_changes:
            for fname in code_changes:
                if main_module in fname:
                    target_file = fname
                    break

        if target_file not in code_changes:
            return 0.0, {
                "reason": "target file not found in response",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        # Apply changes and run tests
        modified = apply_code_changes(codebase, code_changes)
        test_code = domain["tests"]
        results = run_tests(modified, test_code, timeout=15.0)
        score, breakdown = compute_test_score(results)

        breakdown["has_reasoning"] = bool(extract_reasoning(response))
        breakdown["files_changed"] = list(code_changes.keys())
        breakdown["changed_target"] = target_file in code_changes
        breakdown["edge_case_count"] = len(domain["edge_cases"])

        return score, breakdown
