"""
Environment 1: Bug Localization & Fix

Skill: Finding and fixing bugs in unfamiliar code.

The model is given a multi-file Python codebase with an injected bug
and a failing test. It must trace the code, find the bug, and write a fix.

Bug types: off-by-one, wrong operator, inverted condition, missing edge case,
wrong variable, type coercion error, incorrect default, missing return.

Difficulty scaling:
- easy: single file, obvious bug, clear test
- medium: 2-3 files, subtle bug, need to trace imports
- hard: 3-4 files with distractors, subtle bug requiring cross-file tracing
"""
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, extract_reasoning, parse_code_blocks, apply_code_changes,
    run_tests, compute_test_score, CodeExecutor
)


# ── Code templates for different domains ──

DOMAINS = {
    "string_utils": {
        "correct": textwrap.dedent('''
            def reverse_words(text):
                """Reverse the order of words in a string."""
                words = text.split()
                return ' '.join(reversed(words))

            def count_vowels(s):
                """Count vowels in a string."""
                return sum(1 for c in s.lower() if c in 'aeiou')

            def capitalize_first(text):
                """Capitalize only the first letter, keep rest as-is."""
                if not text:
                    return text
                return text[0].upper() + text[1:]

            def truncate(text, max_len, suffix="..."):
                """Truncate text to max_len characters, adding suffix if truncated."""
                if len(text) <= max_len:
                    return text
                return text[:max_len - len(suffix)] + suffix

            def camel_to_snake(name):
                """Convert CamelCase to snake_case."""
                result = []
                for i, c in enumerate(name):
                    if c.isupper() and i > 0:
                        result.append('_')
                    result.append(c.lower())
                return ''.join(result)

            def strip_non_alpha(s):
                """Remove all non-alphabetic characters."""
                return ''.join(c for c in s if c.isalpha())
        ''').strip(),
        "bugs": [
            {
                "name": "off_by_one_in_truncate",
                "desc": "truncate uses max_len instead of max_len - len(suffix)",
                "inject": lambda code: code.replace(
                    "text[:max_len - len(suffix)] + suffix",
                    "text[:max_len] + suffix"
                ),
                "fix_hint": "The truncation length doesn't account for the suffix length",
                "test": textwrap.dedent('''
                    from string_utils import truncate
                    def test_truncate_short():
                        assert truncate("hello", 10) == "hello"
                    def test_truncate_exact():
                        assert truncate("hello world!", 12) == "hello world!"
                    def test_truncate_long():
                        result = truncate("this is a very long string", 15)
                        assert len(result) == 15
                        assert result.endswith("...")
                    def test_truncate_custom_suffix():
                        result = truncate("abcdefghij", 7, suffix="!!")
                        assert len(result) == 7
                        assert result.endswith("!!")
                ''').strip(),
            },
            {
                "name": "wrong_vowel_count",
                "desc": "count_vowels includes 'y'",
                "inject": lambda code: code.replace(
                    "if c in 'aeiou'",
                    "if c in 'aeiouy'"
                ),
                "fix_hint": "The vowel check includes an incorrect character",
                "test": textwrap.dedent('''
                    from string_utils import count_vowels
                    def test_count_basic():
                        assert count_vowels("hello") == 2
                    def test_count_empty():
                        assert count_vowels("") == 0
                    def test_count_all_vowels():
                        assert count_vowels("aeiou") == 5
                    def test_count_with_y():
                        assert count_vowels("rhythm") == 0
                    def test_count_mixed_case():
                        assert count_vowels("HELLO World") == 3
                ''').strip(),
            },
            {
                "name": "capitalize_empty_crash",
                "desc": "capitalize_first crashes on empty string",
                "inject": lambda code: code.replace(
                    "if not text:\n    return text\n    return text[0].upper()",
                    "return text[0].upper()"
                ).replace(
                    "if not text:\n                        return text\n                    return text[0].upper()",
                    "return text[0].upper()"
                ),
                "fix_hint": "The function doesn't handle empty strings",
                "test": textwrap.dedent('''
                    from string_utils import capitalize_first
                    def test_capitalize_basic():
                        assert capitalize_first("hello") == "Hello"
                    def test_capitalize_already_upper():
                        assert capitalize_first("Hello") == "Hello"
                    def test_capitalize_empty():
                        assert capitalize_first("") == ""
                    def test_capitalize_single_char():
                        assert capitalize_first("a") == "A"
                    def test_capitalize_preserves_rest():
                        assert capitalize_first("hELLO") == "HELLO"
                ''').strip(),
            },
        ],
    },
    "math_utils": {
        "correct": textwrap.dedent('''
            def factorial(n):
                """Compute n! iteratively."""
                if n < 0:
                    raise ValueError("n must be non-negative")
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return result

            def is_prime(n):
                """Check if n is prime."""
                if n < 2:
                    return False
                if n == 2:
                    return True
                if n % 2 == 0:
                    return False
                for i in range(3, int(n**0.5) + 1, 2):
                    if n % i == 0:
                        return False
                return True

            def gcd(a, b):
                """Compute greatest common divisor."""
                while b:
                    a, b = b, a % b
                return abs(a)

            def lcm(a, b):
                """Compute least common multiple."""
                if a == 0 or b == 0:
                    return 0
                return abs(a * b) // gcd(a, b)

            def fibonacci(n):
                """Return the n-th Fibonacci number (0-indexed)."""
                if n < 0:
                    raise ValueError("n must be non-negative")
                if n <= 1:
                    return n
                a, b = 0, 1
                for _ in range(2, n + 1):
                    a, b = b, a + b
                return b

            def power(base, exp):
                """Compute base^exp for non-negative integer exp."""
                if exp < 0:
                    raise ValueError("exp must be non-negative")
                result = 1
                for _ in range(exp):
                    result *= base
                return result
        ''').strip(),
        "bugs": [
            {
                "name": "fibonacci_off_by_one",
                "desc": "fibonacci returns a instead of b",
                "inject": lambda code: code.replace(
                    "return b\n\n            def power",
                    "return a\n\n            def power"
                ),
                "fix_hint": "The function returns the wrong variable after the loop",
                "test": textwrap.dedent('''
                    from math_utils import fibonacci
                    def test_fib_0():
                        assert fibonacci(0) == 0
                    def test_fib_1():
                        assert fibonacci(1) == 1
                    def test_fib_2():
                        assert fibonacci(2) == 1
                    def test_fib_5():
                        assert fibonacci(5) == 5
                    def test_fib_10():
                        assert fibonacci(10) == 55
                    def test_fib_7():
                        assert fibonacci(7) == 13
                ''').strip(),
            },
            {
                "name": "is_prime_includes_1",
                "desc": "is_prime returns True for n=1",
                "inject": lambda code: code.replace(
                    "if n < 2:\n                return False",
                    "if n < 1:\n                return False"
                ).replace(
                    "if n < 2:\n                    return False",
                    "if n < 1:\n                    return False"
                ),
                "fix_hint": "The lower bound for primality check is wrong",
                "test": textwrap.dedent('''
                    from math_utils import is_prime
                    def test_prime_2():
                        assert is_prime(2) == True
                    def test_prime_1():
                        assert is_prime(1) == False
                    def test_prime_0():
                        assert is_prime(0) == False
                    def test_prime_negative():
                        assert is_prime(-5) == False
                    def test_prime_17():
                        assert is_prime(17) == True
                    def test_prime_18():
                        assert is_prime(18) == False
                ''').strip(),
            },
            {
                "name": "gcd_missing_abs",
                "desc": "gcd doesn't handle negative inputs",
                "inject": lambda code: code.replace(
                    "return abs(a)",
                    "return a"
                ),
                "fix_hint": "The function doesn't handle negative numbers correctly",
                "test": textwrap.dedent('''
                    from math_utils import gcd
                    def test_gcd_basic():
                        assert gcd(12, 8) == 4
                    def test_gcd_negative():
                        assert gcd(-12, 8) == 4
                    def test_gcd_both_negative():
                        assert gcd(-12, -8) == 4
                    def test_gcd_zero():
                        assert gcd(0, 5) == 5
                    def test_gcd_same():
                        assert gcd(7, 7) == 7
                ''').strip(),
            },
        ],
    },
    "list_utils": {
        "correct": textwrap.dedent('''
            def flatten(nested):
                """Flatten a nested list one level deep."""
                result = []
                for item in nested:
                    if isinstance(item, list):
                        result.extend(item)
                    else:
                        result.append(item)
                return result

            def chunk(lst, size):
                """Split a list into chunks of given size."""
                if size <= 0:
                    raise ValueError("size must be positive")
                return [lst[i:i + size] for i in range(0, len(lst), size)]

            def unique(lst):
                """Remove duplicates while preserving order."""
                seen = set()
                result = []
                for item in lst:
                    if item not in seen:
                        seen.add(item)
                        result.append(item)
                return result

            def partition(lst, predicate):
                """Split list into (matching, non-matching) based on predicate."""
                matching = [x for x in lst if predicate(x)]
                non_matching = [x for x in lst if not predicate(x)]
                return matching, non_matching

            def interleave(a, b):
                """Interleave two lists. Excess elements appended at end."""
                result = []
                min_len = min(len(a), len(b))
                for i in range(min_len):
                    result.append(a[i])
                    result.append(b[i])
                result.extend(a[min_len:])
                result.extend(b[min_len:])
                return result

            def rotate(lst, n):
                """Rotate list by n positions (positive = right)."""
                if not lst:
                    return []
                n = n % len(lst)
                return lst[-n:] + lst[:-n]
        ''').strip(),
        "bugs": [
            {
                "name": "chunk_wrong_step",
                "desc": "chunk uses size+1 instead of size as step",
                "inject": lambda code: code.replace(
                    "range(0, len(lst), size)",
                    "range(0, len(lst), size + 1)"
                ),
                "fix_hint": "The step size in the range is incorrect",
                "test": textwrap.dedent('''
                    from list_utils import chunk
                    def test_chunk_basic():
                        assert chunk([1,2,3,4,5,6], 2) == [[1,2],[3,4],[5,6]]
                    def test_chunk_uneven():
                        assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]
                    def test_chunk_single():
                        assert chunk([1,2,3], 1) == [[1],[2],[3]]
                    def test_chunk_full():
                        assert chunk([1,2,3], 3) == [[1,2,3]]
                    def test_chunk_large():
                        assert chunk([1,2,3], 5) == [[1,2,3]]
                ''').strip(),
            },
            {
                "name": "rotate_wrong_direction",
                "desc": "rotate goes left instead of right",
                "inject": lambda code: code.replace(
                    "lst[-n:] + lst[:-n]",
                    "lst[n:] + lst[:n]"
                ),
                "fix_hint": "The rotation direction is inverted",
                "test": textwrap.dedent('''
                    from list_utils import rotate
                    def test_rotate_right_1():
                        assert rotate([1,2,3,4,5], 1) == [5,1,2,3,4]
                    def test_rotate_right_2():
                        assert rotate([1,2,3,4,5], 2) == [4,5,1,2,3]
                    def test_rotate_zero():
                        assert rotate([1,2,3], 0) == [1,2,3]
                    def test_rotate_full():
                        assert rotate([1,2,3], 3) == [1,2,3]
                    def test_rotate_empty():
                        assert rotate([], 5) == []
                ''').strip(),
            },
            {
                "name": "unique_mutates_seen_wrong",
                "desc": "unique checks result instead of seen for duplicates",
                "inject": lambda code: code.replace(
                    "if item not in seen:",
                    "if item not in result:"
                ),
                "fix_hint": "The duplicate check uses the wrong collection (O(n) list instead of O(1) set, but also changes behavior with unhashable items)",
                "test": textwrap.dedent('''
                    from list_utils import unique
                    def test_unique_basic():
                        assert unique([1,2,2,3,3,3]) == [1,2,3]
                    def test_unique_empty():
                        assert unique([]) == []
                    def test_unique_all_same():
                        assert unique([5,5,5,5]) == [5]
                    def test_unique_no_dups():
                        assert unique([1,2,3]) == [1,2,3]
                    def test_unique_preserves_order():
                        assert unique([3,1,2,1,3]) == [3,1,2]
                ''').strip(),
            },
        ],
    },
    "data_processor": {
        "correct": textwrap.dedent('''
            from collections import defaultdict

            def group_by(items, key_func):
                """Group items by a key function."""
                groups = defaultdict(list)
                for item in items:
                    groups[key_func(item)].append(item)
                return dict(groups)

            def filter_and_sort(items, predicate, key_func=None, reverse=False):
                """Filter items, then sort by key."""
                filtered = [x for x in items if predicate(x)]
                if key_func:
                    filtered.sort(key=key_func, reverse=reverse)
                else:
                    filtered.sort(reverse=reverse)
                return filtered

            def aggregate(items, key_func, agg_func):
                """Group by key, then aggregate values."""
                groups = defaultdict(list)
                for item in items:
                    groups[key_func(item)].append(item)
                return {k: agg_func(v) for k, v in groups.items()}

            def merge_sorted(a, b):
                """Merge two sorted lists into one sorted list."""
                result = []
                i, j = 0, 0
                while i < len(a) and j < len(b):
                    if a[i] <= b[j]:
                        result.append(a[i])
                        i += 1
                    else:
                        result.append(b[j])
                        j += 1
                result.extend(a[i:])
                result.extend(b[j:])
                return result

            def count_by(items, key_func):
                """Count items per key."""
                counts = defaultdict(int)
                for item in items:
                    counts[key_func(item)] += 1
                return dict(counts)
        ''').strip(),
        "bugs": [
            {
                "name": "merge_sorted_wrong_comparison",
                "desc": "merge_sorted uses < instead of <=",
                "inject": lambda code: code.replace(
                    "if a[i] <= b[j]:",
                    "if a[i] < b[j]:"
                ),
                "fix_hint": "The comparison operator in the merge loop is wrong, causing instability with equal elements",
                "test": textwrap.dedent('''
                    from data_processor import merge_sorted
                    def test_merge_basic():
                        assert merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]
                    def test_merge_duplicates():
                        assert merge_sorted([1,2,2], [2,3]) == [1,2,2,2,3]
                    def test_merge_empty_a():
                        assert merge_sorted([], [1,2]) == [1,2]
                    def test_merge_empty_b():
                        assert merge_sorted([1,2], []) == [1,2]
                    def test_merge_both_empty():
                        assert merge_sorted([], []) == []
                ''').strip(),
            },
            {
                "name": "filter_and_sort_missing_filtered",
                "desc": "filter_and_sort sorts original list instead of filtered",
                "inject": lambda code: code.replace(
                    "filtered.sort(key=key_func, reverse=reverse)\n                else:\n                    filtered.sort(reverse=reverse)",
                    "items.sort(key=key_func, reverse=reverse)\n                else:\n                    items.sort(reverse=reverse)"
                ),
                "fix_hint": "The sort operates on the wrong list",
                "test": textwrap.dedent('''
                    from data_processor import filter_and_sort
                    def test_filter_sort_basic():
                        result = filter_and_sort([3,1,4,1,5], lambda x: x > 2)
                        assert result == [3,4,5]
                    def test_filter_sort_with_key():
                        result = filter_and_sort([3,1,4,1,5], lambda x: x > 1, key_func=lambda x: -x)
                        assert result == [5,4,3,1]
                    def test_filter_sort_empty():
                        assert filter_and_sort([], lambda x: True) == []
                    def test_filter_sort_none_pass():
                        assert filter_and_sort([1,2,3], lambda x: x > 10) == []
                ''').strip(),
            },
        ],
    },
}


# ── Distractor code (irrelevant functions to test if model can focus) ──

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

        def parse_csv(text):
            """Parse CSV text into rows (not relevant to the task)."""
            lines = text.strip().split("\\n")
            return [line.split(",") for line in lines]
    ''').strip(),
    textwrap.dedent('''
        def colorize(text, color):
            """Add ANSI color codes (not relevant to the task)."""
            colors = {"red": 31, "green": 32, "yellow": 33, "blue": 34}
            code = colors.get(color, 0)
            return f"\\033[{code}m{text}\\033[0m" if code else text

        def pad_left(text, width, char=" "):
            """Pad text on the left (not relevant to the task)."""
            return char * max(0, width - len(text)) + text
    ''').strip(),
    textwrap.dedent('''
        def debounce(func, delay):
            """Debounce a function call (not relevant to the task)."""
            import time
            last_call = [0]
            def wrapper(*args, **kwargs):
                now = time.time()
                if now - last_call[0] >= delay:
                    last_call[0] = now
                    return func(*args, **kwargs)
            return wrapper

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


@register_env
class BugLocalizationEnv(AgenticEnv):
    name = "bug_localization"
    skill = "Finding and fixing bugs in unfamiliar code by tracing execution paths"
    difficulty_tiers = ["easy", "medium", "hard"]

    def gen_params(self, rng, difficulty="medium"):
        domain_name = rng.choice(list(DOMAINS.keys()))
        domain = DOMAINS[domain_name]
        bug = rng.choice(domain["bugs"])
        n_distractors = {"easy": 0, "medium": 1, "hard": 2}[difficulty]
        distractors = rng.sample(DISTRACTORS, n_distractors) if n_distractors else []
        return {
            "domain": domain_name,
            "bug_name": bug["name"],
            "difficulty": difficulty,
            "n_distractors": n_distractors,
            "distractor_indices": [DISTRACTORS.index(d) for d in distractors] if distractors else [],
            "seed": rng.randint(0, 999999),
        }

    def gen_codebase(self, params, rng):
        domain = DOMAINS[params["domain"]]
        # Find bug by name (bugs is a list of dicts)
        bug = next(b for b in domain["bugs"] if b["name"] == params["bug_name"])

        # Inject the bug
        buggy_code = bug["inject"](domain["correct"])

        # Build codebase
        main_module = params["domain"]
        codebase = {f"{main_module}.py": buggy_code}

        # Add distractor modules
        for idx in params.get("distractor_indices", []):
            distractor = DISTRACTORS[idx]
            distractor_name = f"helper_{idx}.py"
            codebase[distractor_name] = distractor

        # Add a test file that shows the failing test
        codebase["test_failing.py"] = bug["test"]

        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        bug = next(b for b in domain["bugs"] if b["name"] == params["bug_name"])
        main_module = params["domain"]

        lines = []
        lines.append("You are a software engineer debugging an unfamiliar codebase.")
        lines.append("")
        lines.append(f"The test suite in `test_failing.py` is failing. Your task is to:")
        lines.append("1. Read the code carefully, tracing the execution path from the failing test")
        lines.append("2. Identify the specific bug (which line, what's wrong)")
        lines.append("3. Write the corrected version of the file")
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")

        lines.append("=== FAILING TEST OUTPUT ===")
        lines.append("```")
        lines.append(f"Running test_failing.py...")
        lines.append(f"FAILED: One or more tests in test_failing.py are failing.")
        lines.append(f"Hint: {bug['fix_hint']}")
        lines.append("```")
        lines.append("")
        lines.append("Provide your fix in the following format:")
        lines.append("<reasoning>")
        lines.append("...trace the code, identify the bug, explain the fix...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append(f"```python:{main_module}.py")
        lines.append("# the corrected code")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]
        # The solution is the correct (un-bugged) code
        return {f"{main_module}.py": domain["correct"]}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        bug = next(b for b in domain["bugs"] if b["name"] == params["bug_name"])
        main_module = params["domain"]

        reasoning = textwrap.dedent(f"""
            Let me trace through the failing tests to find the bug.

            First, I'll look at the test file to understand what's expected:
            The tests in test_failing.py test functions from {main_module}.py.

            Now let me read {main_module}.py carefully, focusing on the functions being tested.

            Looking at the test cases, they test specific functions. Let me trace each one:

            The hint says: "{bug['fix_hint']}"

            Let me examine the code line by line:
            {bug['desc']}

            I can see the issue now. {bug['fix_hint']}.
            The bug is: {bug['desc']}.

            The fix is to correct this: I need to change the buggy code back to the correct
            implementation. Let me write the corrected version of {main_module}.py.

            I'll also verify: the other functions in the file are not affected by this bug,
            so I only need to fix the specific function that has the issue.
        """).strip()

        return reasoning

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]
        bug = next(b for b in domain["bugs"] if b["name"] == params["bug_name"])

        # Parse the model's response
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
        test_code = bug["test"]
        results = run_tests(modified_codebase, test_code, timeout=10.0)
        score, breakdown = compute_test_score(results)

        # Add extra info
        breakdown["bug_name"] = bug["name"]
        breakdown["bug_desc"] = bug["desc"]
        breakdown["has_reasoning"] = bool(extract_reasoning(response))
        breakdown["files_changed"] = list(code_changes.keys())

        # Check if the model actually changed the target file
        target_file = f"{main_module}.py"
        breakdown["changed_target"] = target_file in code_changes

        # Bonus: if the model identified the right file but tests still fail,
        # give partial credit for correct identification
        if score == 0.0 and breakdown["changed_target"]:
            # Check if the fix is close to correct
            from ..graders import code_similarity
            sim = code_similarity(
                code_changes.get(target_file, ""),
                domain["correct"]
            )
            if sim > 0.8:
                score = 0.3 * sim
                breakdown["partial_credit"] = f"fix is {sim:.0%} similar to correct, awarded partial credit"

        return score, breakdown
