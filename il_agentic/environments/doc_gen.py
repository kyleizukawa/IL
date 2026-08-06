"""
Environment 9: Documentation Generation

Skill: Writing accurate docstrings from code.

The model is given code with no docstrings and must write docstrings for all
functions and classes. The grader uses AST parsing to check that every
function/class/method has a docstring, and that each docstring contains
expected keywords (parameter names, return type descriptions, key behavior words).

Score = 0.5 * coverage (fraction of functions with docstrings)
      + 0.5 * keyword_match (fraction of expected keywords found)

Difficulty scaling:
- easy: simple functions, straightforward behavior
- medium: functions with edge cases, multiple parameters, conditional returns
- hard: classes with methods, complex behavior, interactions between methods
"""
import ast
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, parse_code_blocks, apply_code_changes,
    extract_reasoning,
)


# ── Domain templates ──
# Each domain has: undocumented code, documented solution, expected keywords per function

DOMAINS = {
    "math_library": {
        "undocumented": textwrap.dedent('''
            def factorial(n):
                if n < 0:
                    raise ValueError("n must be non-negative")
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return result

            def is_prime(n):
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
                while b:
                    a, b = b, a % b
                return abs(a)

            def fibonacci(n):
                if n < 0:
                    raise ValueError("n must be non-negative")
                if n <= 1:
                    return n
                a, b = 0, 1
                for _ in range(2, n + 1):
                    a, b = b, a + b
                return b

            def binomial(n, k):
                if k < 0 or k > n:
                    return 0
                if k == 0 or k == n:
                    return 1
                k = min(k, n - k)
                result = 1
                for i in range(k):
                    result = result * (n - i) // (i + 1)
                return result
        ''').strip(),
        "documented": textwrap.dedent('''
            def factorial(n):
                """Compute the factorial of a non-negative integer n.

                Args:
                    n: A non-negative integer.

                Returns:
                    The factorial of n (n!).

                Raises:
                    ValueError: If n is negative.
                """
                if n < 0:
                    raise ValueError("n must be non-negative")
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return result

            def is_prime(n):
                """Check if an integer n is a prime number.

                Args:
                    n: An integer to test for primality.

                Returns:
                    True if n is prime, False otherwise.
                """
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
                """Compute the greatest common divisor of two integers.

                Args:
                    a: First integer.
                    b: Second integer.

                Returns:
                    The greatest common divisor of a and b.
                """
                while b:
                    a, b = b, a % b
                return abs(a)

            def fibonacci(n):
                """Return the n-th Fibonacci number (0-indexed).

                Args:
                    n: A non-negative integer index.

                Returns:
                    The n-th Fibonacci number.

                Raises:
                    ValueError: If n is negative.
                """
                if n < 0:
                    raise ValueError("n must be non-negative")
                if n <= 1:
                    return n
                a, b = 0, 1
                for _ in range(2, n + 1):
                    a, b = b, a + b
                return b

            def binomial(n, k):
                """Compute the binomial coefficient C(n, k).

                Args:
                    n: Total number of items.
                    k: Number of items to choose.

                Returns:
                    The binomial coefficient C(n, k), or 0 if k is out of range.
                """
                if k < 0 or k > n:
                    return 0
                if k == 0 or k == n:
                    return 1
                k = min(k, n - k)
                result = 1
                for i in range(k):
                    result = result * (n - i) // (i + 1)
                return result
        ''').strip(),
        "keywords": {
            "factorial": ["n", "factorial", "non-negative", "ValueError"],
            "is_prime": ["n", "prime", "True", "False"],
            "gcd": ["a", "b", "greatest", "divisor"],
            "fibonacci": ["n", "Fibonacci", "non-negative", "ValueError"],
            "binomial": ["n", "k", "binomial", "coefficient"],
        },
    },
    "string_utils": {
        "undocumented": textwrap.dedent('''
            def reverse_words(text):
                words = text.split()
                return ' '.join(reversed(words))

            def count_vowels(s):
                return sum(1 for c in s.lower() if c in 'aeiou')

            def truncate(text, max_len, suffix="..."):
                if len(text) <= max_len:
                    return text
                return text[:max_len - len(suffix)] + suffix

            def camel_to_snake(name):
                result = []
                for i, c in enumerate(name):
                    if c.isupper() and i > 0:
                        result.append('_')
                    result.append(c.lower())
                return ''.join(result)

            def levenshtein(a, b):
                m, n = len(a), len(b)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(n + 1):
                    dp[0][j] = j
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if a[i-1] == b[j-1]:
                            dp[i][j] = dp[i-1][j-1]
                        else:
                            dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                return dp[m][n]
        ''').strip(),
        "documented": textwrap.dedent('''
            def reverse_words(text):
                """Reverse the order of words in a string.

                Args:
                    text: The input string.

                Returns:
                    A string with words in reversed order.
                """
                words = text.split()
                return ' '.join(reversed(words))

            def count_vowels(s):
                """Count the number of vowels in a string.

                Args:
                    s: The input string.

                Returns:
                    The count of vowels (a, e, i, o, u), case-insensitive.
                """
                return sum(1 for c in s.lower() if c in 'aeiou')

            def truncate(text, max_len, suffix="..."):
                """Truncate text to a maximum length, appending a suffix if truncated.

                Args:
                    text: The input string to truncate.
                    max_len: Maximum length of the result string.
                    suffix: String to append if truncation occurs.

                Returns:
                    The truncated string, with suffix if truncation happened.
                """
                if len(text) <= max_len:
                    return text
                return text[:max_len - len(suffix)] + suffix

            def camel_to_snake(name):
                """Convert a CamelCase string to snake_case.

                Args:
                    name: A CamelCase string.

                Returns:
                    The snake_case version of the input string.
                """
                result = []
                for i, c in enumerate(name):
                    if c.isupper() and i > 0:
                        result.append('_')
                    result.append(c.lower())
                return ''.join(result)

            def levenshtein(a, b):
                """Compute the Levenshtein edit distance between two strings.

                Args:
                    a: First string.
                    b: Second string.

                Returns:
                    The minimum number of edits (insert, delete, substitute) to transform a into b.
                """
                m, n = len(a), len(b)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(n + 1):
                    dp[0][j] = j
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if a[i-1] == b[j-1]:
                            dp[i][j] = dp[i-1][j-1]
                        else:
                            dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                return dp[m][n]
        ''').strip(),
        "keywords": {
            "reverse_words": ["text", "reverse", "words", "string"],
            "count_vowels": ["s", "vowels", "count"],
            "truncate": ["text", "max_len", "suffix", "truncate"],
            "camel_to_snake": ["name", "CamelCase", "snake_case"],
            "levenshtein": ["a", "b", "edit", "distance"],
        },
    },
    "data_structures": {
        "undocumented": textwrap.dedent('''
            class Stack:
                def __init__(self):
                    self._items = []

                def push(self, item):
                    self._items.append(item)

                def pop(self):
                    if not self._items:
                        raise IndexError("pop from empty stack")
                    return self._items.pop()

                def peek(self):
                    if not self._items:
                        raise IndexError("peek from empty stack")
                    return self._items[-1]

                def is_empty(self):
                    return len(self._items) == 0

                def size(self):
                    return len(self._items)

            class Queue:
                def __init__(self):
                    self._items = []

                def enqueue(self, item):
                    self._items.append(item)

                def dequeue(self):
                    if not self._items:
                        raise IndexError("dequeue from empty queue")
                    return self._items.pop(0)

                def is_empty(self):
                    return len(self._items) == 0

                def size(self):
                    return len(self._items)
        ''').strip(),
        "documented": textwrap.dedent('''
            class Stack:
                """A LIFO (Last-In-First-Out) stack data structure.

                Supports push, pop, peek, and size operations.
                """

                def __init__(self):
                    """Initialize an empty stack."""
                    self._items = []

                def push(self, item):
                    """Push an item onto the top of the stack.

                    Args:
                        item: The item to push onto the stack.
                    """
                    self._items.append(item)

                def pop(self):
                    """Remove and return the top item from the stack.

                    Returns:
                        The item at the top of the stack.

                    Raises:
                        IndexError: If the stack is empty.
                    """
                    if not self._items:
                        raise IndexError("pop from empty stack")
                    return self._items.pop()

                def peek(self):
                    """Return the top item without removing it.

                    Returns:
                        The item at the top of the stack.

                    Raises:
                        IndexError: If the stack is empty.
                    """
                    if not self._items:
                        raise IndexError("peek from empty stack")
                    return self._items[-1]

                def is_empty(self):
                    """Check if the stack is empty.

                    Returns:
                        True if the stack has no items, False otherwise.
                    """
                    return len(self._items) == 0

                def size(self):
                    """Return the number of items in the stack.

                    Returns:
                        The number of items currently in the stack.
                    """
                    return len(self._items)

            class Queue:
                """A FIFO (First-In-First-Out) queue data structure.

                Supports enqueue, dequeue, and size operations.
                """

                def __init__(self):
                    """Initialize an empty queue."""
                    self._items = []

                def enqueue(self, item):
                    """Add an item to the end of the queue.

                    Args:
                        item: The item to add to the queue.
                    """
                    self._items.append(item)

                def dequeue(self):
                    """Remove and return the item at the front of the queue.

                    Returns:
                        The item at the front of the queue.

                    Raises:
                        IndexError: If the queue is empty.
                    """
                    if not self._items:
                        raise IndexError("dequeue from empty queue")
                    return self._items.pop(0)

                def is_empty(self):
                    """Check if the queue is empty.

                    Returns:
                        True if the queue has no items, False otherwise.
                    """
                    return len(self._items) == 0

                def size(self):
                    """Return the number of items in the queue.

                    Returns:
                        The number of items currently in the queue.
                    """
                    return len(self._items)
        ''').strip(),
        "keywords": {
            "Stack": ["LIFO", "stack", "push", "pop"],
            "Stack.push": ["item", "push", "stack"],
            "Stack.pop": ["pop", "top", "IndexError", "empty"],
            "Stack.peek": ["peek", "top", "IndexError", "empty"],
            "Stack.is_empty": ["empty", "True", "False"],
            "Stack.size": ["size", "number", "items"],
            "Queue": ["FIFO", "queue", "enqueue", "dequeue"],
            "Queue.enqueue": ["item", "enqueue", "queue"],
            "Queue.dequeue": ["dequeue", "front", "IndexError", "empty"],
            "Queue.is_empty": ["empty", "True", "False"],
            "Queue.size": ["size", "number", "items"],
        },
    },
    "file_io": {
        "undocumented": textwrap.dedent('''
            import json
            import os

            def read_json(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)

            def write_json(filepath, data, indent=2):
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=indent)

            def read_lines(filepath):
                with open(filepath, 'r') as f:
                    return [line.rstrip('\\n') for line in f]

            def append_line(filepath, line):
                with open(filepath, 'a') as f:
                    f.write(line + '\\n')

            def safe_remove(filepath):
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return True
                return False

            def get_file_size(filepath):
                if not os.path.exists(filepath):
                    return None
                return os.path.getsize(filepath)
        ''').strip(),
        "documented": textwrap.dedent('''
            import json
            import os

            def read_json(filepath):
                """Read and parse a JSON file.

                Args:
                    filepath: Path to the JSON file to read.

                Returns:
                    The parsed JSON content as a Python object.
                """
                with open(filepath, 'r') as f:
                    return json.load(f)

            def write_json(filepath, data, indent=2):
                """Write data to a file as JSON.

                Args:
                    filepath: Path to the output file.
                    data: Python object to serialize as JSON.
                    indent: Number of spaces for indentation.
                """
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=indent)

            def read_lines(filepath):
                """Read a text file and return lines without trailing newlines.

                Args:
                    filepath: Path to the text file to read.

                Returns:
                    A list of strings, one per line, without trailing newlines.
                """
                with open(filepath, 'r') as f:
                    return [line.rstrip('\\n') for line in f]

            def append_line(filepath, line):
                """Append a single line to a file.

                Args:
                    filepath: Path to the file to append to.
                    line: The line of text to append.
                """
                with open(filepath, 'a') as f:
                    f.write(line + '\\n')

            def safe_remove(filepath):
                """Remove a file if it exists, without raising an error.

                Args:
                    filepath: Path to the file to remove.

                Returns:
                    True if the file was removed, False if it did not exist.
                """
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return True
                return False

            def get_file_size(filepath):
                """Get the size of a file in bytes.

                Args:
                    filepath: Path to the file.

                Returns:
                    The file size in bytes, or None if the file does not exist.
                """
                if not os.path.exists(filepath):
                    return None
                return os.path.getsize(filepath)
        ''').strip(),
        "keywords": {
            "read_json": ["filepath", "JSON", "parse", "read"],
            "write_json": ["filepath", "data", "JSON", "indent", "write"],
            "read_lines": ["filepath", "lines", "newlines", "read"],
            "append_line": ["filepath", "line", "append"],
            "safe_remove": ["filepath", "remove", "True", "False", "exists"],
            "get_file_size": ["filepath", "size", "bytes", "None"],
        },
    },
}


# ── Distractor code ──

DISTRACTORS = [
    textwrap.dedent('''
        def format_table(rows):
            def _width(col):
                return max(len(str(row[col])) for row in rows) if rows else 0
            if not rows:
                return ""
            cols = range(len(rows[0]))
            widths = [_width(c) for c in cols]
            lines = []
            for row in rows:
                lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))
            return "\\n".join(lines)
    ''').strip(),
    textwrap.dedent('''
        def memoize(func):
            cache = {}
            def wrapper(*args):
                if args not in cache:
                    cache[args] = func(*args)
                return cache[args]
            return wrapper
    ''').strip(),
    textwrap.dedent('''
        def colorize(text, color):
            colors = {"red": 31, "green": 32, "yellow": 33, "blue": 34}
            code = colors.get(color, 0)
            return f"\\033[{code}m{text}\\033[0m" if code else text
    ''').strip(),
]


def _check_docstrings(code: str, expected_keywords: dict) -> dict:
    """Use AST to check docstring presence and keyword matching.

    Returns coverage score and keyword match score.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"coverage": 0.0, "keyword_match": 0.0, "details": {}, "error": "syntax_error"}

    found_docstrings = {}
    total_funcs = 0
    funcs_with_docstring = 0

    def _check_node(node, prefix=""):
        nonlocal total_funcs, funcs_with_docstring
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}" if prefix else child.name
                total_funcs += 1

                # Check for docstring (first statement is a string expression)
                has_doc = False
                docstring_text = ""
                if child.body and isinstance(child.body[0], ast.Expr):
                    if isinstance(child.body[0].value, ast.Constant) and isinstance(child.body[0].value.value, str):
                        has_doc = True
                        docstring_text = child.body[0].value.value
                    elif isinstance(child.body[0].value, ast.Str):
                        has_doc = True
                        docstring_text = child.body[0].value.s

                if has_doc:
                    funcs_with_docstring += 1

                # Check keywords
                kw_score = 0.0
                if name in expected_keywords:
                    expected_kws = expected_keywords[name]
                    if has_doc:
                        doc_lower = docstring_text.lower()
                        matched = sum(1 for kw in expected_kws if kw.lower() in doc_lower)
                        kw_score = matched / len(expected_kws) if expected_kws else 0.0
                    found_docstrings[name] = {
                        "has_docstring": has_doc,
                        "keyword_score": kw_score,
                        "docstring_length": len(docstring_text),
                    }
                else:
                    found_docstrings[name] = {
                        "has_docstring": has_doc,
                        "keyword_score": 0.0,
                        "docstring_length": len(docstring_text) if has_doc else 0,
                    }

                # Recurse into class methods
                if isinstance(child, ast.ClassDef):
                    _check_node(child, prefix=f"{child.name}.")

    _check_node(tree)

    coverage = funcs_with_docstring / total_funcs if total_funcs > 0 else 0.0

    # Keyword match: only count functions that have expected keywords
    keyword_scores = []
    for name, info in found_docstrings.items():
        if name in expected_keywords:
            keyword_scores.append(info["keyword_score"])
    keyword_match = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0

    return {
        "coverage": coverage,
        "keyword_match": keyword_match,
        "details": found_docstrings,
        "total_funcs": total_funcs,
        "funcs_with_docstring": funcs_with_docstring,
    }


@register_env
class DocGenEnv(AgenticEnv):
    name = "doc_gen"
    skill = "Writing accurate docstrings from code"
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
        codebase = {f"{main_module}.py": domain["undocumented"]}

        for idx in params.get("distractor_indices", []):
            codebase[f"helper_{idx}.py"] = DISTRACTORS[idx]

        return codebase

    def gen_task(self, params, codebase):
        main_module = params["domain"]
        lines = []
        lines.append("You are a software engineer writing documentation for existing code.")
        lines.append("")
        lines.append("Your task is to:")
        lines.append("1. Read each function and class carefully")
        lines.append("2. Write a docstring for EVERY function, method, and class")
        lines.append("3. Each docstring should describe: what it does, its parameters, its return value, and any exceptions raised")
        lines.append("4. Use clear, accurate language — do not make up behavior that isn't in the code")
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")

        lines.append("Provide your documented code in the following format:")
        lines.append("<reasoning>")
        lines.append("...read each function, describe what it does, list parameters, describe return value...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append(f"```python:{main_module}.py")
        lines.append("# the documented code")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]
        return {f"{main_module}.py": domain["documented"]}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]
        keywords = domain["keywords"]
        func_names = list(keywords.keys())

        lines = []
        lines.append(f"I need to write docstrings for all functions and classes in {main_module}.py.")
        lines.append("Let me read each function carefully to understand what it does before writing documentation.")
        lines.append("")

        for fname in func_names:
            lines.append(f"Analyzing `{fname}`:")
            lines.append(f"  - Reading the function body to understand its behavior.")
            lines.append(f"  - Identifying parameters and their roles.")
            lines.append(f"  - Determining the return value and its type.")
            lines.append(f"  - Noting any edge cases or exceptions raised.")
            kw_list = keywords[fname]
            lines.append(f"  - Key concepts to document: {', '.join(kw_list)}")
            lines.append("")

        lines.append("Now I'll write the documented version of the file, adding a docstring to each function/class.")
        lines.append("I'll make sure each docstring accurately describes the code's actual behavior,")
        lines.append("not what I might assume from the function name alone.")

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
        submitted = code_changes.get(target_file, "")
        if not submitted:
            for fname, content in code_changes.items():
                if main_module in fname or fname.endswith(".py"):
                    submitted = content
                    target_file = fname
                    break

        if not submitted:
            return 0.0, {
                "reason": "target file not found in response",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        # Check syntax
        try:
            ast.parse(submitted)
        except SyntaxError as e:
            return 0.0, {
                "reason": f"syntax error in submitted code: {e}",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        # Check docstrings
        result = _check_docstrings(submitted, domain["keywords"])

        coverage = result["coverage"]
        keyword_match = result["keyword_match"]
        score = 0.5 * coverage + 0.5 * keyword_match

        breakdown = {
            "coverage": coverage,
            "keyword_match": keyword_match,
            "total_funcs": result.get("total_funcs", 0),
            "funcs_with_docstring": result.get("funcs_with_docstring", 0),
            "details": result.get("details", {}),
            "has_reasoning": bool(extract_reasoning(response)),
            "files_changed": list(code_changes.keys()),
            "changed_target": target_file in code_changes,
            "final_score": score,
        }

        return score, breakdown
