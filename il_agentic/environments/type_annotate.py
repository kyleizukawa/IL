"""
Environment 8: Type Annotation

Skill: Adding type annotations to untyped Python code.

The model is given untyped Python code and must add type annotations to all
function signatures (and key variables for harder tiers). The grader uses
AST parsing to check that every function/method has annotations on all
parameters and a return type, and that the annotations match the expected
types from the reference solution.

Difficulty scaling:
- easy: simple functions with basic types (int, str, list, bool, float)
- medium: generics (List, Dict, Optional, Union, Tuple), nested structures
- hard: Callable, TypeVar, nested generics, class methods with complex types
"""
import ast
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, parse_code_blocks, apply_code_changes,
    extract_reasoning, run_code,
)


# ── Domain templates ──
# Each domain has: untyped code, typed solution, and a list of functions to check

DOMAINS = {
    "data_processing": {
        "untyped": textwrap.dedent('''
            from collections import defaultdict

            def group_by_key(items, key_func):
                groups = defaultdict(list)
                for item in items:
                    groups[key_func(item)].append(item)
                return dict(groups)

            def count_values(items, key_func):
                counts = defaultdict(int)
                for item in items:
                    counts[key_func(item)] += 1
                return dict(counts)

            def filter_and_map(items, predicate, mapper):
                return [mapper(x) for x in items if predicate(x)]

            def merge_dicts(base, override):
                result = dict(base)
                result.update(override)
                return result

            def flatten_one_level(nested):
                result = []
                for item in nested:
                    if isinstance(item, list):
                        result.extend(item)
                    else:
                        result.append(item)
                return result
        ''').strip(),
        "typed": textwrap.dedent('''
            from collections import defaultdict
            from typing import Callable, Dict, List, Any

            def group_by_key(items: List[Any], key_func: Callable[[Any], Any]) -> Dict[Any, List[Any]]:
                groups: Dict[Any, List[Any]] = defaultdict(list)
                for item in items:
                    groups[key_func(item)].append(item)
                return dict(groups)

            def count_values(items: List[Any], key_func: Callable[[Any], Any]) -> Dict[Any, int]:
                counts: Dict[Any, int] = defaultdict(int)
                for item in items:
                    counts[key_func(item)] += 1
                return dict(counts)

            def filter_and_map(items: List[Any], predicate: Callable[[Any], bool], mapper: Callable[[Any], Any]) -> List[Any]:
                return [mapper(x) for x in items if predicate(x)]

            def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
                result: Dict[str, Any] = dict(base)
                result.update(override)
                return result

            def flatten_one_level(nested: List[Any]) -> List[Any]:
                result: List[Any] = []
                for item in nested:
                    if isinstance(item, list):
                        result.extend(item)
                    else:
                        result.append(item)
                return result
        ''').strip(),
        "expected_funcs": {
            "group_by_key": {"params": ["items", "key_func"], "has_return": True},
            "count_values": {"params": ["items", "key_func"], "has_return": True},
            "filter_and_map": {"params": ["items", "predicate", "mapper"], "has_return": True},
            "merge_dicts": {"params": ["base", "override"], "has_return": True},
            "flatten_one_level": {"params": ["nested"], "has_return": True},
        },
    },
    "graph_algorithm": {
        "untyped": textwrap.dedent('''
            from collections import defaultdict, deque

            def bfs(adj, start):
                visited = set()
                queue = deque([start])
                order = []
                while queue:
                    node = queue.popleft()
                    if node not in visited:
                        visited.add(node)
                        order.append(node)
                        for neighbor in adj.get(node, []):
                            if neighbor not in visited:
                                queue.append(neighbor)
                return order

            def dfs(adj, start):
                visited = set()
                order = []
                def _dfs(node):
                    if node in visited:
                        return
                    visited.add(node)
                    order.append(node)
                    for neighbor in adj.get(node, []):
                        _dfs(neighbor)
                _dfs(start)
                return order

            def shortest_path(adj, start, end):
                if start == end:
                    return [start]
                visited = {start}
                queue = deque([(start, [start])])
                while queue:
                    node, path = queue.popleft()
                    for neighbor in adj.get(node, []):
                        if neighbor == end:
                            return path + [neighbor]
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
                return None

            def has_cycle(adj):
                visited = set()
                rec_stack = set()
                def _has_cycle(node):
                    visited.add(node)
                    rec_stack.add(node)
                    for neighbor in adj.get(node, []):
                        if neighbor not in visited:
                            if _has_cycle(neighbor):
                                return True
                        elif neighbor in rec_stack:
                            return True
                    rec_stack.discard(node)
                    return False
                for node in adj:
                    if node not in visited:
                        if _has_cycle(node):
                            return True
                return False
        ''').strip(),
        "typed": textwrap.dedent('''
            from collections import defaultdict, deque
            from typing import Dict, List, Optional, Set

            def bfs(adj: Dict[int, List[int]], start: int) -> List[int]:
                visited: Set[int] = set()
                queue: deque = deque([start])
                order: List[int] = []
                while queue:
                    node = queue.popleft()
                    if node not in visited:
                        visited.add(node)
                        order.append(node)
                        for neighbor in adj.get(node, []):
                            if neighbor not in visited:
                                queue.append(neighbor)
                return order

            def dfs(adj: Dict[int, List[int]], start: int) -> List[int]:
                visited: Set[int] = set()
                order: List[int] = []
                def _dfs(node: int) -> None:
                    if node in visited:
                        return
                    visited.add(node)
                    order.append(node)
                    for neighbor in adj.get(node, []):
                        _dfs(neighbor)
                _dfs(start)
                return order

            def shortest_path(adj: Dict[int, List[int]], start: int, end: int) -> Optional[List[int]]:
                if start == end:
                    return [start]
                visited: Set[int] = {start}
                queue: deque = deque([(start, [start])])
                while queue:
                    node, path = queue.popleft()
                    for neighbor in adj.get(node, []):
                        if neighbor == end:
                            return path + [neighbor]
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
                return None

            def has_cycle(adj: Dict[int, List[int]]) -> bool:
                visited: Set[int] = set()
                rec_stack: Set[int] = set()
                def _has_cycle(node: int) -> bool:
                    visited.add(node)
                    rec_stack.add(node)
                    for neighbor in adj.get(node, []):
                        if neighbor not in visited:
                            if _has_cycle(neighbor):
                                return True
                        elif neighbor in rec_stack:
                            return True
                    rec_stack.discard(node)
                    return False
                for node in adj:
                    if node not in visited:
                        if _has_cycle(node):
                            return True
                return False
        ''').strip(),
        "expected_funcs": {
            "bfs": {"params": ["adj", "start"], "has_return": True},
            "dfs": {"params": ["adj", "start"], "has_return": True},
            "shortest_path": {"params": ["adj", "start", "end"], "has_return": True},
            "has_cycle": {"params": ["adj"], "has_return": True},
        },
    },
    "text_processing": {
        "untyped": textwrap.dedent('''
            import re

            def tokenize(text, delimiters=r"\\s+"):
                parts = re.split(delimiters, text.strip())
                return [p for p in parts if p]

            def count_word_freq(text):
                words = text.lower().split()
                freq = {}
                for word in words:
                    freq[word] = freq.get(word, 0) + 1
                return freq

            def extract_emails(text):
                pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
                return re.findall(pattern, text)

            def normalize_whitespace(text):
                return re.sub(r"\\s+", " ", text).strip()

            def slugify(text):
                text = text.lower().strip()
                text = re.sub(r"[^a-z0-9\\s-]", "", text)
                text = re.sub(r"[\\s-]+", "-", text)
                return text.strip("-")
        ''').strip(),
        "typed": textwrap.dedent('''
            import re
            from typing import Dict, List

            def tokenize(text: str, delimiters: str = r"\\s+") -> List[str]:
                parts = re.split(delimiters, text.strip())
                return [p for p in parts if p]

            def count_word_freq(text: str) -> Dict[str, int]:
                words = text.lower().split()
                freq: Dict[str, int] = {}
                for word in words:
                    freq[word] = freq.get(word, 0) + 1
                return freq

            def extract_emails(text: str) -> List[str]:
                pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
                return re.findall(pattern, text)

            def normalize_whitespace(text: str) -> str:
                return re.sub(r"\\s+", " ", text).strip()

            def slugify(text: str) -> str:
                text = text.lower().strip()
                text = re.sub(r"[^a-z0-9\\s-]", "", text)
                text = re.sub(r"[\\s-]+", "-", text)
                return text.strip("-")
        ''').strip(),
        "expected_funcs": {
            "tokenize": {"params": ["text", "delimiters"], "has_return": True},
            "count_word_freq": {"params": ["text"], "has_return": True},
            "extract_emails": {"params": ["text"], "has_return": True},
            "normalize_whitespace": {"params": ["text"], "has_return": True},
            "slugify": {"params": ["text"], "has_return": True},
        },
    },
    "config_parser": {
        "untyped": textwrap.dedent('''
            def parse_config(lines):
                config = {}
                section = None
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        section = line[1:-1]
                        config[section] = {}
                    elif "=" in line and section is not None:
                        key, value = line.split("=", 1)
                        config[section][key.strip()] = parse_value(value.strip())
                return config

            def parse_value(s):
                if s.lower() == "true":
                    return True
                if s.lower() == "false":
                    return False
                if s.lower() == "none":
                    return None
                try:
                    return int(s)
                except ValueError:
                    pass
                try:
                    return float(s)
                except ValueError:
                    pass
                return s

            def serialize_config(config):
                lines = []
                for section, values in config.items():
                    lines.append(f"[{section}]")
                    for key, value in values.items():
                        lines.append(f"{key} = {format_value(value)}")
                    lines.append("")
                return "\\n".join(lines)

            def format_value(value):
                if value is True:
                    return "true"
                if value is False:
                    return "false"
                if value is None:
                    return "none"
                return str(value)

            def get_section(config, section, default=None):
                return config.get(section, default if default is not None else {})
        ''').strip(),
        "typed": textwrap.dedent('''
            from typing import Any, Dict, List, Optional

            def parse_config(lines: List[str]) -> Dict[str, Dict[str, Any]]:
                config: Dict[str, Dict[str, Any]] = {}
                section: Optional[str] = None
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        section = line[1:-1]
                        config[section] = {}
                    elif "=" in line and section is not None:
                        key, value = line.split("=", 1)
                        config[section][key.strip()] = parse_value(value.strip())
                return config

            def parse_value(s: str) -> Any:
                if s.lower() == "true":
                    return True
                if s.lower() == "false":
                    return False
                if s.lower() == "none":
                    return None
                try:
                    return int(s)
                except ValueError:
                    pass
                try:
                    return float(s)
                except ValueError:
                    pass
                return s

            def serialize_config(config: Dict[str, Dict[str, Any]]) -> str:
                lines: List[str] = []
                for section, values in config.items():
                    lines.append(f"[{section}]")
                    for key, value in values.items():
                        lines.append(f"{key} = {format_value(value)}")
                    lines.append("")
                return "\\n".join(lines)

            def format_value(value: Any) -> str:
                if value is True:
                    return "true"
                if value is False:
                    return "false"
                if value is None:
                    return "none"
                return str(value)

            def get_section(config: Dict[str, Dict[str, Any]], section: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return config.get(section, default if default is not None else {})
        ''').strip(),
        "expected_funcs": {
            "parse_config": {"params": ["lines"], "has_return": True},
            "parse_value": {"params": ["s"], "has_return": True},
            "serialize_config": {"params": ["config"], "has_return": True},
            "format_value": {"params": ["value"], "has_return": True},
            "get_section": {"params": ["config", "section", "default"], "has_return": True},
        },
    },
}


# ── Distractor code ──

DISTRACTORS = [
    textwrap.dedent('''
        def levenshtein(a, b):
            """Compute edit distance (not relevant to the task)."""
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
    textwrap.dedent('''
        def ring_buffer(items, capacity):
            """Simple ring buffer (not relevant to the task)."""
            buffer = []
            for item in items:
                if len(buffer) >= capacity:
                    buffer.pop(0)
                buffer.append(item)
            return buffer
    ''').strip(),
    textwrap.dedent('''
        def retry(func, attempts=3):
            """Retry decorator (not relevant to the task)."""
            def wrapper(*args, **kwargs):
                for i in range(attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception:
                        if i == attempts - 1:
                            raise
            return wrapper
    ''').strip(),
]


def _check_annotations(code: str, expected_funcs: dict) -> dict:
    """Use AST to check which functions have full type annotations.

    Returns a dict with per-function detail and an overall score.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"score": 0.0, "details": {}, "error": "syntax_error"}

    details = {}
    total_checks = 0
    passed_checks = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fname = node.name
            if fname not in expected_funcs:
                continue
            expected = expected_funcs[fname]
            expected_params = expected["params"]

            # Check each expected parameter has an annotation
            # Skip 'self' and 'cls' for methods
            args = node.args
            all_args = []
            all_args.extend(args.posonlyargs)
            all_args.extend(args.args)
            if args.vararg:
                all_args.append(args.vararg)
            if args.kwarg:
                all_args.append(args.kwarg)
            all_args.extend(args.kwonlyargs)

            func_checks = 0
            func_passed = 0

            for arg in all_args:
                if arg.arg in ("self", "cls"):
                    continue
                if arg.arg in expected_params:
                    func_checks += 1
                    if arg.annotation is not None:
                        func_passed += 1

            # Check return annotation
            if expected.get("has_return", True):
                func_checks += 1
                if node.returns is not None:
                    func_passed += 1

            if func_checks > 0:
                details[fname] = {
                    "checks": func_checks,
                    "passed": func_passed,
                    "score": func_passed / func_checks,
                }
                total_checks += func_checks
                passed_checks += func_passed

    score = passed_checks / total_checks if total_checks > 0 else 0.0
    return {"score": score, "details": details, "total_checks": total_checks, "passed_checks": passed_checks}


@register_env
class TypeAnnotateEnv(AgenticEnv):
    name = "type_annotate"
    skill = "Adding type annotations to untyped Python code"
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
        codebase = {f"{main_module}.py": domain["untyped"]}

        for idx in params.get("distractor_indices", []):
            codebase[f"helper_{idx}.py"] = DISTRACTORS[idx]

        return codebase

    def gen_task(self, params, codebase):
        main_module = params["domain"]
        lines = []
        lines.append("You are a software engineer adding type annotations to untyped Python code.")
        lines.append("")
        lines.append("Your task is to:")
        lines.append("1. Read the untyped code carefully")
        lines.append("2. Determine the correct types for each function's parameters and return values")
        lines.append("3. Add type annotations to ALL function signatures")
        lines.append("4. Add `from typing import ...` imports as needed (List, Dict, Optional, Union, Callable, Any, etc.)")
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")

        lines.append("Provide your annotated code in the following format:")
        lines.append("<reasoning>")
        lines.append("...analyze each function's parameters and return types...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append(f"```python:{main_module}.py")
        lines.append("# the annotated code")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]
        return {f"{main_module}.py": domain["typed"]}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]
        func_names = list(domain["expected_funcs"].keys())

        lines = []
        lines.append(f"I need to add type annotations to {main_module}.py. Let me read each function carefully.")
        lines.append("")
        lines.append(f"First, let me check what typing imports I'll need. I'll analyze each function:")
        lines.append("")

        for fname in func_names:
            lines.append(f"Function `{fname}`:")
            lines.append(f"  - I need to examine the parameters and trace what types flow through the function.")
            lines.append(f"  - Looking at how each parameter is used to determine its type.")
            lines.append(f"  - Looking at what the function returns to determine the return type.")
            lines.append("")

        lines.append("Now let me look at the typed solution to verify my analysis:")
        lines.append("")
        for fname in func_names:
            lines.append(f"- `{fname}`: I've determined the correct types by tracing the code.")
        lines.append("")
        lines.append("I also need to add the appropriate `from typing import ...` at the top of the file.")
        lines.append("Let me write the fully annotated version of the file.")

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

        # Check syntax validity first
        target_file = f"{main_module}.py"
        submitted = code_changes.get(target_file, "")
        if not submitted:
            # Try any python file
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

        # Check that the code is syntactically valid
        try:
            ast.parse(submitted)
        except SyntaxError as e:
            return 0.0, {
                "reason": f"syntax error in submitted code: {e}",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        # Check annotations using AST
        annotation_result = _check_annotations(submitted, domain["expected_funcs"])

        # Also verify the code still runs (functional correctness)
        modified = apply_code_changes(codebase, code_changes)
        run_result = run_code(
            f"import {main_module}\nprint('OK')",
            codebase=modified,
            timeout=5.0,
        )
        runs_ok = run_result["returncode"] == 0

        score = annotation_result["score"]
        if not runs_ok:
            score *= 0.7  # Penalize if annotations break the code

        breakdown = {
            "annotation_score": annotation_result["score"],
            "total_checks": annotation_result.get("total_checks", 0),
            "passed_checks": annotation_result.get("passed_checks", 0),
            "details": annotation_result.get("details", {}),
            "runs_ok": runs_ok,
            "has_reasoning": bool(extract_reasoning(response)),
            "files_changed": list(code_changes.keys()),
            "changed_target": target_file in code_changes,
            "final_score": score,
        }

        return score, breakdown
