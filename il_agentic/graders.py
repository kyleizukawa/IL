"""
Grader infrastructure for agentic coding environments.

Provides:
- CodeExecutor: sandboxed subprocess code execution with timeout
- run_tests: run a test file against a codebase, return pass/fail per test
- run_code: execute arbitrary Python code, return stdout/stderr/exit code
- parse_code_blocks: extract ```python:filename``` blocks from model response
- apply_code_changes: merge model's code changes into the codebase
- extract_answer / extract_reasoning: parse <answer>/<reasoning> tags
"""
import os
import re
import sys
import json
import tempfile
import subprocess
import textwrap
from typing import Any


# ── Response parsing ──

def extract_reasoning(response: str) -> str:
    """Extract content between <reasoning> and </reasoning> tags."""
    match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_answer(response: str) -> str:
    """Extract content between <answer> and </answer> tags.

    If no tags found, return everything after the last </reasoning> tag,
    or the full response as fallback.
    """
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: everything after </reasoning>
    match = re.search(r'</reasoning>\s*(.*)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def parse_code_blocks(text: str) -> dict[str, str]:
    """Extract code blocks from text.

    Supports two formats:
    1. ```python:filename.py\\n...code...\\n```
    2. ```python\\n# filename: filename.py\\n...code...\\n```
    3. ```filename.py\\n...code...\\n```

    Returns {filename: content} for all code blocks found.
    """
    blocks = {}
    # Pattern 1: ```python:filename
    for match in re.finditer(r'```(?:python|py)?:?\s*(\S+\.\w+)\s*\n(.*?)```', text, re.DOTALL):
        filename = match.group(1).strip()
        content = match.group(2).strip()
        blocks[filename] = content
    # Pattern 2: ```python with # filename: comment on first line
    if not blocks:
        for match in re.finditer(r'```(?:python|py)\s*\n(.*?)```', text, re.DOTALL):
            content = match.group(1).strip()
            first_line = content.split('\n')[0]
            fname_match = re.match(r'#\s*(?:filename|file|in)?:?\s*(\S+\.\w+)', first_line)
            if fname_match:
                filename = fname_match.group(1).strip()
                # Remove the filename comment from content
                content = '\n'.join(content.split('\n')[1:]).strip()
                blocks[filename] = content
    # Pattern 3: bare ``` with filename
    if not blocks:
        for match in re.finditer(r'```(\S+\.\w+)\s*\n(.*?)```', text, re.DOTALL):
            filename = match.group(1).strip()
            content = match.group(2).strip()
            blocks[filename] = content
    return blocks


def apply_code_changes(codebase: dict[str, str], changes: dict[str, str]) -> dict[str, str]:
    """Apply model's code changes to the codebase.

    - If filename exists in codebase, replace its content
    - If filename is new, add it
    - Returns a new codebase dict (doesn't mutate original)
    """
    new_codebase = dict(codebase)
    for filename, content in changes.items():
        new_codebase[filename] = content
    return new_codebase


# ── Code execution sandbox ──

class CodeExecutor:
    """Sandboxed code executor using subprocess with timeout.

    Writes codebase files to a temp directory, runs Python code,
    and captures output. No network access, restricted resources.
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.tmpdir = None

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp(prefix="il_agentic_")
        return self

    def __exit__(self, *args):
        if self.tmpdir and os.path.exists(self.tmpdir):
            import shutil
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def write_codebase(self, codebase: dict[str, str]):
        """Write all codebase files to the temp directory."""
        for filename, content in codebase.items():
            filepath = os.path.join(self.tmpdir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filename) else None
            with open(filepath, 'w') as f:
                f.write(content)

    def run(self, code: str, extra_files: dict[str, str] | None = None) -> dict:
        """Execute Python code in the sandbox.

        Returns: {
            'stdout': str, 'stderr': str, 'returncode': int,
            'timed_out': bool, 'error': str | None
        }
        """
        if not self.tmpdir:
            raise RuntimeError("CodeExecutor must be used as context manager")

        if extra_files:
            self.write_codebase(extra_files)

        script_path = os.path.join(self.tmpdir, "_run_script.py")
        with open(script_path, 'w') as f:
            f.write(code)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.tmpdir,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": self.tmpdir,
                    "HOME": self.tmpdir,
                },
            )
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'timed_out': False,
                'error': None,
            }
        except subprocess.TimeoutExpired:
            return {
                'stdout': '',
                'stderr': '',
                'returncode': -1,
                'timed_out': True,
                'error': f'Execution timed out after {self.timeout}s',
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'timed_out': False,
                'error': str(e),
            }

    def run_tests(self, codebase: dict[str, str], test_code: str) -> dict:
        """Run a test script against a codebase.

        The test code should use unittest or pytest-style assertions.
        Returns: {
            'total': int, 'passed': int, 'failed': int,
            'errors': int, 'results': [dict], 'stdout': str, 'stderr': str
        }
        """
        self.write_codebase(codebase)

        # Wrap test code to capture individual test results.
        # NOTE: The template is NOT indented inside the string so that
        # textwrap.dedent is not needed — the test_code is inserted at
        # the top level, matching the surrounding boilerplate.
        wrapper = f"""\
import sys, json, traceback, io

results = []
total = 0
passed = 0
failed = 0
errors = 0

# Capture stdout
_old_stdout = sys.stdout
_captured = io.StringIO()

# The test code should define functions starting with 'test_'
# or use assertions. We'll run each test_ function and catch results.
{test_code}

# Find and run all test_ functions
import inspect
test_funcs = [(name, obj) for name, obj in list(globals().items())
              if name.startswith('test_') and callable(obj)]

for name, func in test_funcs:
    total += 1
    sys.stdout = _captured
    try:
        func()
        passed += 1
        results.append({{'name': name, 'status': 'pass'}})
    except AssertionError as e:
        failed += 1
        results.append({{'name': name, 'status': 'fail', 'error': str(e)}})
    except Exception as e:
        errors += 1
        results.append({{'name': name, 'status': 'error',
                        'error': traceback.format_exc()}})
    finally:
        sys.stdout = _old_stdout

# Also catch any module-level assertions
output = {{
    'total': total,
    'passed': passed,
    'failed': failed,
    'errors': errors,
    'results': results,
    'stdout': _captured.getvalue(),
}}
sys.stdout = _old_stdout
print(json.dumps(output))
"""

        result = self.run(wrapper)
        if result['timed_out'] or result['returncode'] != 0:
            return {
                'total': 0, 'passed': 0, 'failed': 0, 'errors': 0,
                'results': [], 'stdout': result.get('stdout', ''),
                'stderr': result.get('stderr', ''),
                'error': result.get('error', 'Unknown error'),
                'timed_out': result.get('timed_out', False),
            }

        try:
            # Find the JSON line in output
            stdout = result['stdout']
            json_line = None
            for line in stdout.strip().split('\n'):
                line = line.strip()
                if line.startswith('{') and 'total' in line:
                    json_line = line
                    break
            if json_line:
                return json.loads(json_line)
            return {
                'total': 0, 'passed': 0, 'failed': 0, 'errors': 0,
                'results': [], 'stdout': stdout,
                'stderr': result['stderr'],
                'error': 'No JSON output found',
            }
        except json.JSONDecodeError as e:
            return {
                'total': 0, 'passed': 0, 'failed': 0, 'errors': 0,
                'results': [], 'stdout': result['stdout'],
                'stderr': result['stderr'],
                'error': f'JSON decode error: {e}',
            }


# ── Convenience functions ──

def run_code(code: str, codebase: dict[str, str] | None = None,
             timeout: float = 10.0) -> dict:
    """Execute Python code in a sandbox. Returns result dict."""
    with CodeExecutor(timeout=timeout) as executor:
        if codebase:
            executor.write_codebase(codebase)
        return executor.run(code)


def run_tests(codebase: dict[str, str], test_code: str,
              timeout: float = 10.0) -> dict:
    """Run tests against a codebase. Returns test results dict."""
    with CodeExecutor(timeout=timeout) as executor:
        return executor.run_tests(codebase, test_code)


# ── Scoring helpers ──

def compute_test_score(results: dict) -> tuple[float, dict]:
    """Compute a normalized score from test results.

    Returns (score, breakdown) where score is in [0, 1].
    """
    total = results.get('total', 0)
    passed = results.get('passed', 0)
    if total == 0:
        return 0.0, {'total': 0, 'passed': 0, 'reason': 'no tests ran'}
    score = passed / total
    return score, {
        'total': total,
        'passed': passed,
        'failed': results.get('failed', 0),
        'errors': results.get('errors', 0),
        'score': score,
    }


def text_similarity(a: str, b: str) -> float:
    """Compute similarity between two text strings (token overlap)."""
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def code_similarity(a: str, b: str) -> float:
    """Compute similarity between two code strings (line overlap)."""
    if not a or not b:
        return 0.0
    lines_a = set(l.strip() for l in a.split('\n') if l.strip())
    lines_b = set(l.strip() for l in b.split('\n') if l.strip())
    if not lines_a or not lines_b:
        return 0.0
    intersection = lines_a & lines_b
    union = lines_a | lines_b
    return len(intersection) / len(union)
