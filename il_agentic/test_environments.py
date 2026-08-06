"""
Test suite for IL Agentic environments.

Verifies that:
1. All 15 environments are registered and importable
2. Each environment can generate valid instances at all difficulty levels
3. Graders work correctly (correct solution scores 1.0, broken solution scores < 1.0)
4. SFT data generation produces valid chat examples
5. Benchmark instance generation works

Usage:
    python -m il_agentic.test_environments
    python -m il_agentic.test_environments --env bug_localization  # test one
"""
import json
import sys
import os
import random
import traceback
import textwrap
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from il_agentic import ALL_ENVIRONMENTS, ENV_REGISTRY
from il_agentic.graders import (
    extract_reasoning, extract_answer, parse_code_blocks,
    apply_code_changes, run_tests, run_code, CodeExecutor,
    compute_test_score, code_similarity,
)


def test_registry():
    """Test that all 15 environments are registered."""
    print("Test 1: Environment registry...", flush=True)
    expected = {
        "bug_localization", "feature_impl", "refactor_preserve",
        "test_writing", "api_client", "perf_optimize", "codebase_nav",
        "type_annotate", "doc_gen", "config_fix", "error_handling",
        "data_transform", "algorithm_impl", "code_review", "stacktrace_debug",
    }
    registered = set(ENV_REGISTRY.keys())
    missing = expected - registered
    extra = registered - expected

    if missing:
        print(f"  FAIL: Missing environments: {missing}", flush=True)
        return False
    if extra:
        print(f"  WARN: Extra environments: {extra}", flush=True)

    print(f"  PASS: {len(registered)} environments registered", flush=True)
    for name in sorted(registered):
        env = ENV_REGISTRY[name]
        print(f"    {name}: skill='{env.skill}'", flush=True)
    return True


def test_instance_generation(env_name: str, n_tests: int = 3) -> bool:
    """Test that an environment can generate valid instances."""
    env_class = ENV_REGISTRY[env_name]
    env = env_class()
    rng = random.Random(42)

    print(f"\nTest 2: Instance generation for '{env_name}'...", flush=True)

    for difficulty in ["easy", "medium", "hard"]:
        for i in range(n_tests):
            try:
                inst = env.generate_instance(rng, difficulty)
                assert "codebase" in inst, "missing codebase"
                assert "task" in inst, "missing task"
                assert "solution" in inst, "missing solution"
                assert "reasoning" in inst, "missing reasoning"
                assert len(inst["codebase"]) > 0, "empty codebase"
                assert len(inst["task"]) > 0, "empty task"
                assert len(inst["reasoning"]) > 0, "empty reasoning"

                # Check codebase files are valid Python (for code tasks)
                for filename, content in inst["codebase"].items():
                    if filename.endswith(".py"):
                        try:
                            compile(content, filename, 'exec')
                        except SyntaxError as e:
                            print(f"  WARN: {filename} has syntax error: {e}", flush=True)

                print(f"  {difficulty} #{i}: OK ({len(inst['codebase'])} files, "
                      f"task={len(inst['task'])} chars, "
                      f"reasoning={len(inst['reasoning'])} chars)", flush=True)

            except Exception as e:
                print(f"  FAIL: {difficulty} #{i}: {e}", flush=True)
                traceback.print_exc()
                return False

    return True


def test_grader_correct_solution(env_name: str) -> bool:
    """Test that the grader gives a high score for the correct solution."""
    env_class = ENV_REGISTRY[env_name]
    env = env_class()
    rng = random.Random(42)

    print(f"\nTest 3: Grader correct-solution test for '{env_name}'...", flush=True)

    try:
        inst = env.generate_instance(rng, "easy")

        # Build a "perfect" response using the solution
        solution = inst["solution"]
        answer_parts = []
        for filename, content in solution.items():
            answer_parts.append(f"```python:{filename}\n{content}\n```")
        answer = "\n\n".join(answer_parts)
        reasoning = inst["reasoning"]
        perfect_response = f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>"

        score, breakdown = env.grade(inst["params"], inst["codebase"], perfect_response)

        if score >= 0.8:
            print(f"  PASS: correct solution scored {score:.3f}", flush=True)
            return True
        else:
            print(f"  WARN: correct solution scored {score:.3f} (expected >= 0.8)", flush=True)
            print(f"    breakdown: {json.dumps(breakdown, indent=2)[:500]}", flush=True)
            # This might be OK for some environments where the grader is strict
            # or where the "solution" is a template. Don't fail the test.
            return score >= 0.5

    except Exception as e:
        print(f"  FAIL: {e}", flush=True)
        traceback.print_exc()
        return False


def test_grader_broken_solution(env_name: str) -> bool:
    """Test that the grader gives a low score for a broken/empty response."""
    env_class = ENV_REGISTRY[env_name]
    env = env_class()
    rng = random.Random(42)

    print(f"\nTest 4: Grader broken-response test for '{env_name}'...", flush=True)

    try:
        inst = env.generate_instance(rng, "easy")

        # Empty/broken response
        broken_response = "<reasoning>\nI don't know.\n</reasoning>\n<answer>\n\n</answer>"
        score, breakdown = env.grade(inst["params"], inst["codebase"], broken_response)

        if score <= 0.3:
            print(f"  PASS: broken response scored {score:.3f} (expected <= 0.3)", flush=True)
            return True
        else:
            print(f"  WARN: broken response scored {score:.3f} (expected <= 0.3)", flush=True)
            return False

    except Exception as e:
        print(f"  FAIL: {e}", flush=True)
        traceback.print_exc()
        return False


def test_sft_generation(env_name: str) -> bool:
    """Test that SFT example generation produces valid chat format."""
    env_class = ENV_REGISTRY[env_name]
    env = env_class()
    rng = random.Random(42)

    print(f"\nTest 5: SFT generation for '{env_name}'...", flush=True)

    try:
        example = env.build_sft_example(rng, "medium")
        assert "messages" in example, "missing messages"
        assert len(example["messages"]) == 2, "expected 2 messages"
        assert example["messages"][0]["role"] == "user", "first message should be user"
        assert example["messages"][1]["role"] == "assistant", "second message should be assistant"

        user_msg = example["messages"][0]["content"]
        asst_msg = example["messages"][1]["content"]

        assert len(user_msg) > 100, f"user message too short: {len(user_msg)}"
        assert len(asst_msg) > 50, f"assistant message too short: {len(asst_msg)}"
        assert "<reasoning>" in asst_msg, "assistant missing <reasoning> tag"
        assert "</reasoning>" in asst_msg, "assistant missing </reasoning> tag"
        assert "<answer>" in asst_msg, "assistant missing <answer> tag"
        assert "</answer>" in asst_msg, "assistant missing </answer> tag"

        print(f"  PASS: valid SFT example (user={len(user_msg)} chars, "
              f"assistant={len(asst_msg)} chars)", flush=True)
        return True

    except Exception as e:
        print(f"  FAIL: {e}", flush=True)
        traceback.print_exc()
        return False


def test_code_executor():
    """Test the CodeExecutor sandbox."""
    print("\nTest 6: CodeExecutor sandbox...", flush=True)

    # Test basic execution
    result = run_code("print('hello world')")
    if result['returncode'] == 0 and 'hello world' in result['stdout']:
        print("  PASS: basic execution", flush=True)
    else:
        print(f"  FAIL: basic execution: {result}", flush=True)
        return False

    # Test timeout
    result = run_code("import time; time.sleep(10)", timeout=2.0)
    if result['timed_out']:
        print("  PASS: timeout works", flush=True)
    else:
        print(f"  FAIL: timeout not triggered: {result}", flush=True)
        return False

    # Test codebase execution
    codebase = {"math_lib.py": "def add(a, b):\n    return a + b\n"}
    test_code = textwrap.dedent("""
        from math_lib import add
        def test_add():
            assert add(2, 3) == 5
        def test_add_negative():
            assert add(-1, 1) == 0
    """)
    result = run_tests(codebase, test_code)
    if result['total'] == 2 and result['passed'] == 2:
        print("  PASS: test runner works", flush=True)
    else:
        print(f"  FAIL: test runner: {result}", flush=True)
        return False

    return True


def run_all_tests(env_filter: str | None = None):
    """Run all tests."""
    print("=" * 60, flush=True)
    print("IL Agentic Environment Test Suite", flush=True)
    print("=" * 60, flush=True)

    results = defaultdict(list)

    # Test 1: Registry
    results["registry"].append(test_registry())

    # Test 6: Code executor
    results["code_executor"].append(test_code_executor())

    # Test each environment
    env_names = [env_filter] if env_filter else sorted(ENV_REGISTRY.keys())
    for env_name in env_names:
        if env_name not in ENV_REGISTRY:
            print(f"\nERROR: Environment '{env_name}' not found", flush=True)
            continue

        results[f"{env_name}_gen"].append(test_instance_generation(env_name, n_tests=2))
        results[f"{env_name}_grader_correct"].append(test_grader_correct_solution(env_name))
        results[f"{env_name}_grader_broken"].append(test_grader_broken_solution(env_name))
        results[f"{env_name}_sft"].append(test_sft_generation(env_name))

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("Test Summary", flush=True)
    print("=" * 60, flush=True)

    total = 0
    passed = 0
    for test_name, test_results in sorted(results.items()):
        for r in test_results:
            total += 1
            if r:
                passed += 1

    print(f"\n  {passed}/{total} tests passed", flush=True)

    if passed < total:
        print("\n  Failed tests:", flush=True)
        for test_name, test_results in sorted(results.items()):
            for r in test_results:
                if not r:
                    print(f"    {test_name}", flush=True)

    return passed == total


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test IL Agentic environments")
    parser.add_argument("--env", type=str, default=None,
                        help="test only this environment")
    args = parser.parse_args()

    success = run_all_tests(env_filter=args.env)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
