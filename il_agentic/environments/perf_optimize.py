"""
Environment 6: Performance Optimization

Skill: Optimizing slow code while preserving correctness.

The model is given correct but slow code (O(n^2) where O(n) is possible,
unnecessary recomputation, redundant loops) and must optimize it.

Grader: 0.5 * correctness + 0.5 * speedup_score
  where speedup_score = min(1.0, actual_speedup / target_speedup)

Difficulty scaling:
- easy: obvious optimization (concat -> join)
- medium: need to use a dict/set for O(1) lookup
- hard: need algorithmic insight (rethink the approach)
"""
import random
import textwrap
import time
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
#   slow_code: correct but O(n^2) or worse implementation
#   fast_code: the optimized O(n) or O(n log n) solution
#   tests: correctness tests (must all pass)
#   perf_test: code that measures execution time on large input
#   target_speedup: the minimum speedup factor for full credit
#   bottleneck_desc: description of the performance issue
#   optimization_desc: description of the optimization strategy

DOMAINS = {
    # ── Domain 1: Duplicate finder using nested loops ──
    "duplicate_finder": {
        "slow_code": textwrap.dedent('''
            def find_duplicates(items):
                """Find all items that appear more than once. Returns a list of duplicates."""
                duplicates = []
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        if items[i] == items[j] and items[i] not in duplicates:
                            duplicates.append(items[i])
                            break
                return duplicates
        ''').strip(),
        "fast_code": textwrap.dedent('''
            def find_duplicates(items):
                """Find all items that appear more than once. Returns a list of duplicates."""
                seen = set()
                dupes = set()
                duplicates = []
                for item in items:
                    if item in seen:
                        if item not in dupes:
                            dupes.add(item)
                            duplicates.append(item)
                    else:
                        seen.add(item)
                return duplicates
        ''').strip(),
        "tests": textwrap.dedent('''
            from optimizer import find_duplicates
            def test_no_duplicates():
                assert find_duplicates([1, 2, 3, 4, 5]) == []
            def test_simple_duplicates():
                assert find_duplicates([1, 2, 2, 3, 3, 3]) == [2, 3]
            def test_all_same():
                assert find_duplicates([5, 5, 5, 5]) == [5]
            def test_empty():
                assert find_duplicates([]) == []
            def test_single():
                assert find_duplicates([1]) == []
            def test_strings():
                assert find_duplicates(["a", "b", "a", "c", "b"]) == ["a", "b"]
            def test_preserves_first_occurrence_order():
                result = find_duplicates([3, 1, 2, 1, 3, 2])
                assert set(result) == {1, 2, 3}
        ''').strip(),
        "perf_test": textwrap.dedent('''
            from optimizer import find_duplicates
            import time
            data = list(range(5000)) * 3
            start = time.time()
            result = find_duplicates(data)
            elapsed = time.time() - start
            assert len(result) == 5000
            print(f"TIME:{elapsed:.6f}")
        ''').strip(),
        "target_speedup": 10.0,
        "bottleneck_desc": "The nested for-loop creates O(n^2) comparisons, and the 'not in duplicates' check on a list adds another O(n) per item.",
        "optimization_desc": "Use a set for O(1) membership testing. Track seen items and duplicates separately, checking duplicates as a set too.",
    },

    # ── Domain 2: String builder using concatenation ──
    "string_builder": {
        "slow_code": textwrap.dedent('''
            def build_csv(headers, rows):
                """Build a CSV string from headers and rows."""
                result = ""
                for h in headers:
                    result = result + h + ","
                result = result + "\\n"
                for row in rows:
                    for val in row:
                        result = result + str(val) + ","
                    result = result + "\\n"
                return result
        ''').strip(),
        "fast_code": textwrap.dedent('''
            def build_csv(headers, rows):
                """Build a CSV string from headers and rows."""
                parts = []
                for h in headers:
                    parts.append(h)
                    parts.append(",")
                parts.append("\\n")
                for row in rows:
                    for val in row:
                        parts.append(str(val))
                        parts.append(",")
                    parts.append("\\n")
                return "".join(parts)
        ''').strip(),
        "tests": textwrap.dedent('''
            from optimizer import build_csv
            def test_basic_csv():
                result = build_csv(["name", "age"], [["Alice", 30], ["Bob", 25]])
                lines = result.strip().split("\\n")
                assert "name,age," in lines[0]
                assert "Alice,30," in lines[1]
                assert "Bob,25," in lines[2]
            def test_single_row():
                result = build_csv(["x"], [["1"]])
                assert "x," in result
                assert "1," in result
            def test_empty_rows():
                result = build_csv(["a", "b"], [])
                assert result == "a,b,\\n"
            def test_empty_headers():
                result = build_csv([], [])
                assert result == "\\n"
            def test_large_values():
                result = build_csv(["col"], [["x" * 100]])
                assert "x" * 100 in result
        ''').strip(),
        "perf_test": textwrap.dedent('''
            from optimizer import build_csv
            import time
            headers = [f"col_{i}" for i in range(50)]
            rows = [[f"val_{j}_{k}" for j in range(50)] for k in range(500)]
            start = time.time()
            result = build_csv(headers, rows)
            elapsed = time.time() - start
            assert len(result) > 0
            print(f"TIME:{elapsed:.6f}")
        ''').strip(),
        "target_speedup": 10.0,
        "bottleneck_desc": "String concatenation with += creates a new string each time, copying all previous content. This is O(n^2) for n total characters.",
        "optimization_desc": "Use a list to accumulate parts and join them at the end with ''.join(parts), which is O(n) total.",
    },

    # ── Domain 3: Sum calculator with redundant iteration ──
    "sum_calculator": {
        "slow_code": textwrap.dedent('''
            def running_sums(data, queries):
                """For each query (start, end), return sum(data[start:end]).
                data is a list of numbers. queries is a list of (start, end) tuples."""
                results = []
                for start, end in queries:
                    total = 0
                    for i in range(start, end):
                        total += data[i]
                    results.append(total)
                return results
        ''').strip(),
        "fast_code": textwrap.dedent('''
            def running_sums(data, queries):
                """For each query (start, end), return sum(data[start:end]).
                data is a list of numbers. queries is a list of (start, end) tuples."""
                prefix = [0]
                for val in data:
                    prefix.append(prefix[-1] + val)
                results = []
                for start, end in queries:
                    results.append(prefix[end] - prefix[start])
                return results
        ''').strip(),
        "tests": textwrap.dedent('''
            from optimizer import running_sums
            def test_basic():
                data = [1, 2, 3, 4, 5]
                queries = [(0, 3), (1, 4), (2, 5)]
                result = running_sums(data, queries)
                assert result == [6, 9, 12]
            def test_single_element():
                assert running_sums([10, 20, 30], [(1, 2)]) == [20]
            def test_full_range():
                assert running_sums([1, 2, 3], [(0, 3)]) == [6]
            def test_empty_queries():
                assert running_sums([1, 2, 3], []) == []
            def test_overlapping():
                data = [5, 10, 15, 20, 25]
                queries = [(0, 2), (1, 3), (0, 5), (2, 4)]
                result = running_sums(data, queries)
                assert result == [15, 25, 75, 35]
            def test_negative_numbers():
                data = [-1, -2, 3, -4, 5]
                queries = [(0, 3), (2, 5)]
                assert running_sums(data, queries) == [0, 4]
        ''').strip(),
        "perf_test": textwrap.dedent('''
            from optimizer import running_sums
            import time
            data = list(range(50000))
            queries = [(i, i + 500) for i in range(0, 45000, 50)]
            start = time.time()
            result = running_sums(data, queries)
            elapsed = time.time() - start
            assert len(result) == 900
            print(f"TIME:{elapsed:.6f}")
        ''').strip(),
        "target_speedup": 5.0,
        "bottleneck_desc": "Each query re-iterates over the range [start, end) to compute the sum. With many overlapping queries, this is O(q * n) where q is the number of queries.",
        "optimization_desc": "Precompute a prefix sum array where prefix[i] = sum(data[0:i]). Then each query is O(1): sum(data[start:end]) = prefix[end] - prefix[start].",
    },

    # ── Domain 4: Frequency counter using sort ──
    "frequency_counter": {
        "slow_code": textwrap.dedent('''
            def top_k_frequent(items, k):
                """Return the k most frequent items, sorted by frequency (descending).
                Ties broken by first occurrence order."""
                counted = []
                frequencies = []
                for item in items:
                    if item not in counted:
                        counted.append(item)
                        count = 0
                        for x in items:
                            if x == item:
                                count += 1
                        first_idx = items.index(item)
                        frequencies.append((item, count, first_idx))
                frequencies.sort(key=lambda x: (-x[1], x[2]))
                return [item for item, _, _ in frequencies[:k]]
        ''').strip(),
        "fast_code": textwrap.dedent('''
            def top_k_frequent(items, k):
                """Return the k most frequent items, sorted by frequency (descending).
                Ties broken by first occurrence order."""
                from collections import Counter
                counts = Counter(items)
                first_occurrence = {}
                for i, item in enumerate(items):
                    if item not in first_occurrence:
                        first_occurrence[item] = i
                sorted_items = sorted(counts.keys(),
                                      key=lambda x: (-counts[x], first_occurrence[x]))
                return sorted_items[:k]
        ''').strip(),
        "tests": textwrap.dedent('''
            from optimizer import top_k_frequent
            def test_basic():
                items = [1, 1, 1, 2, 2, 3]
                assert top_k_frequent(items, 2) == [1, 2]
            def test_all_unique():
                assert top_k_frequent([1, 2, 3, 4], 2) == [1, 2]
            def test_ties():
                items = [1, 2, 1, 2, 3]
                result = top_k_frequent(items, 2)
                assert set(result) == {1, 2}
            def test_k_larger_than_unique():
                assert top_k_frequent([1, 1, 2], 5) == [1, 2]
            def test_single_item():
                assert top_k_frequent([5, 5, 5], 1) == [5]
            def test_empty():
                assert top_k_frequent([], 3) == []
            def test_strings():
                items = ["apple", "banana", "apple", "cherry", "banana", "apple"]
                assert top_k_frequent(items, 2) == ["apple", "banana"]
        ''').strip(),
        "perf_test": textwrap.dedent('''
            from optimizer import top_k_frequent
            import time
            data = [i % 100 for i in range(10000)]
            start = time.time()
            result = top_k_frequent(data, 10)
            elapsed = time.time() - start
            assert len(result) == 10
            print(f"TIME:{elapsed:.6f}")
        ''').strip(),
        "target_speedup": 5.0,
        "bottleneck_desc": "For each unique item, the code scans the entire list to count occurrences (O(n * u) where u is unique items). The 'item not in counted' check on a list is also O(u) per item.",
        "optimization_desc": "Use collections.Counter for O(n) frequency counting in a single pass, and a dict for O(1) first occurrence tracking. Then sort only the unique items by frequency.",
    },
}


# ── Distractor code ──

DISTRACTORS = [
    textwrap.dedent('''
        def merge_dicts(*dicts):
            """Merge multiple dicts (not relevant to the task)."""
            result = {}
            for d in dicts:
                result.update(d)
            return result

        def invert_dict(d):
            """Invert a dict's keys and values (not relevant to the task)."""
            return {v: k for k, v in d.items()}
    ''').strip(),
    textwrap.dedent('''
        class CircularBuffer:
            """A fixed-size circular buffer (not relevant to the task)."""
            def __init__(self, size):
                self.size = size
                self.buffer = [None] * size
                self.head = 0
                self.count = 0

            def push(self, item):
                self.buffer[self.head] = item
                self.head = (self.head + 1) % self.size
                self.count = min(self.count + 1, self.size)

            def to_list(self):
                start = (self.head - self.count) % self.size
                return [self.buffer[(start + i) % self.size] for i in range(self.count)]
    ''').strip(),
    textwrap.dedent('''
        def levenshtein(s1, s2):
            """Compute edit distance (not relevant to the task)."""
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            prev = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                curr = [i + 1]
                for j, c2 in enumerate(s2):
                    ins = prev[j + 1] + 1
                    dele = curr[j] + 1
                    sub = prev[j] + (c1 != c2)
                    curr.append(min(ins, dele, sub))
                prev = curr
            return prev[-1]
    ''').strip(),
]


@register_env
class PerfOptimizeEnv(AgenticEnv):
    name = "perf_optimize"
    skill = "Optimizing slow code while preserving correctness"
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
        codebase = {"optimizer.py": domain["slow_code"]}
        for idx in params.get("distractor_indices", []):
            codebase[f"helper_{idx}.py"] = DISTRACTORS[idx]
        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        lines = []
        lines.append("You are a software engineer optimizing slow code.")
        lines.append("")
        lines.append("The code in `optimizer.py` is correct but has a performance problem.")
        lines.append("Your task is to:")
        lines.append("1. Read the code carefully and identify the performance bottleneck")
        lines.append("2. Understand what the function does (correctness must be preserved)")
        lines.append("3. Write an optimized version that produces the same results but runs faster")
        lines.append("4. The optimized code must pass all correctness tests AND run significantly faster")
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")
        lines.append("=== CORRECTNESS TESTS (must all pass) ===")
        lines.append("```python")
        lines.append(domain["tests"])
        lines.append("```")
        lines.append("")
        lines.append(f"Target speedup: at least {domain['target_speedup']:.0f}x faster on large inputs.")
        lines.append("")
        lines.append("Provide your optimized solution in the following format:")
        lines.append("<reasoning>")
        lines.append("...identify the bottleneck, explain the optimization strategy, write the optimized code...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append("```python:optimizer.py")
        lines.append("# the optimized code")
        lines.append("```")
        lines.append("</answer>")
        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        return {"optimizer.py": domain["fast_code"]}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        slow_code = domain["slow_code"]
        fast_code = domain["fast_code"]

        reasoning = textwrap.dedent(f"""
            Let me analyze the slow code to identify the performance bottleneck.

            First, I'll read the code in `optimizer.py` carefully:

            ```python
            {slow_code}
            ```

            Let me trace through the execution to understand the time complexity:

            {domain['bottleneck_desc']}

            The current approach has poor time complexity. Let me think about what the function
            actually needs to compute, and whether there's a more efficient way.

            The key insight is:
            {domain['optimization_desc']}

            Let me now write the optimized version. I need to make sure:
            1. The output is identical to the original for all inputs
            2. The time complexity is improved
            3. Edge cases (empty input, single element, etc.) are handled correctly

            Here's my optimized implementation:

            ```python
            {fast_code}
            ```

            Let me verify this preserves correctness:
            - The function signature is the same
            - The return type is the same
            - The logic produces the same results, just more efficiently

            Let me also check the edge cases:
            - Empty input: handled the same way as the original
            - Single element: no special case needed, the optimized logic handles it
            - Large input: the optimized approach avoids the O(n^2) bottleneck

            The optimization should achieve at least {domain['target_speedup']:.0f}x speedup
            on large inputs while maintaining full correctness.
        """).strip()

        return reasoning

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

        # 1. Run correctness tests
        results = _run_tests(modified_codebase, domain["tests"], timeout=15.0)
        correctness_score, test_breakdown = compute_test_score(results)

        # 2. Run performance test
        speedup_score = 0.0
        slow_time = None
        fast_time = None
        actual_speedup = 0.0

        # First, time the slow code
        with CodeExecutor(timeout=30.0) as executor:
            executor.write_codebase(codebase)
            slow_result = executor.run(domain["perf_test"])
            if slow_result["returncode"] == 0:
                for line in slow_result["stdout"].split("\n"):
                    if line.startswith("TIME:"):
                        try:
                            slow_time = float(line.split(":")[1])
                        except ValueError:
                            pass

        # Then, time the fast code
        if slow_time is not None and slow_time > 0:
            with CodeExecutor(timeout=30.0) as executor:
                executor.write_codebase(modified_codebase)
                fast_result = executor.run(domain["perf_test"])
                if fast_result["returncode"] == 0:
                    for line in fast_result["stdout"].split("\n"):
                        if line.startswith("TIME:"):
                            try:
                                fast_time = float(line.split(":")[1])
                            except ValueError:
                                pass

        if slow_time is not None and fast_time is not None and fast_time > 0:
            actual_speedup = slow_time / fast_time
            speedup_score = min(1.0, actual_speedup / domain["target_speedup"])

        # Final score: 0.5 * correctness + 0.5 * speedup
        final_score = 0.5 * correctness_score + 0.5 * speedup_score

        breakdown = {
            "domain": params["domain"],
            "difficulty": params["difficulty"],
            "correctness_score": correctness_score,
            "speedup_score": speedup_score,
            "actual_speedup": round(actual_speedup, 2),
            "target_speedup": domain["target_speedup"],
            "slow_time": slow_time,
            "fast_time": fast_time,
            "tests_passed": test_breakdown.get("passed", 0),
            "tests_total": test_breakdown.get("total", 0),
            "has_reasoning": bool(extract_reasoning(response)),
            "files_changed": list(code_changes.keys()),
            "changed_target": "optimizer.py" in code_changes,
            "final_score": final_score,
        }

        # If correctness is 0, no partial credit for speed alone
        if correctness_score == 0.0:
            final_score = 0.0
            breakdown["reason"] = "correctness tests failed, no credit for speed alone"

        return final_score, breakdown
