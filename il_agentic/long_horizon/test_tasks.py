"""
Test suite for Long-Horizon Agentic Coding Tasks.

Tests all 20 hand-crafted tasks for:
1. Registration and metadata
2. Codebase generation
3. Task description generation
4. Solution correctness (perfect response gets high score)
5. Bad response penalization
6. SFT example generation
7. Reasoning quality scoring
8. Efficiency-aware reward shaping
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from il_agentic.long_horizon import ALL_TASKS, register_long_horizon
from il_agentic.long_horizon.efficiency import (
    score_reasoning_quality, compute_final_score, shape_rl_reward,
    extract_reasoning, extract_answer, ReasoningBreakdown,
)
from il_agentic.long_horizon.base import LongHorizonEnv


class TestRegistry(unittest.TestCase):
    """Test that all 20 tasks are registered."""

    def test_all_20_registered(self):
        self.assertEqual(len(ALL_TASKS), 20,
                         f"Expected 20 tasks, got {len(ALL_TASKS)}")

    def test_unique_task_ids(self):
        ids = [t().task_id for t in ALL_TASKS]
        self.assertEqual(len(ids), len(set(ids)),
                         f"Duplicate task IDs: {ids}")

    def test_all_have_metadata(self):
        for task_class in ALL_TASKS:
            task = task_class()
            self.assertTrue(task.task_id, f"Missing task_id")
            self.assertTrue(task.reasoning_skill, f"Missing reasoning_skill for {task.task_id}")
            self.assertTrue(task.failure_mode, f"Missing failure_mode for {task.task_id}")
            self.assertGreater(task.token_budget, 0, f"Missing token_budget for {task.task_id}")
            self.assertIsInstance(task.expected_concepts, list, f"expected_concepts not list for {task.task_id}")

    def test_all_have_concepts(self):
        for task_class in ALL_TASKS:
            task = task_class()
            self.assertGreaterEqual(len(task.expected_concepts), 5,
                f"{task.task_id} has only {len(task.expected_concepts)} concepts (need >=5)")


class TestCodebaseGeneration(unittest.TestCase):
    """Test that all tasks generate valid codebases."""

    def test_all_generate_codebase(self):
        for task_class in ALL_TASKS:
            task = task_class()
            with self.subTest(task=task.task_id):
                codebase = task.gen_codebase()
                self.assertIsInstance(codebase, dict, f"{task.task_id}: codebase not dict")
                self.assertGreater(len(codebase), 0, f"{task.task_id}: empty codebase")
                for filename, content in codebase.items():
                    self.assertIsInstance(filename, str)
                    self.assertIsInstance(content, str)
                    self.assertGreater(len(content), 0, f"{task.task_id}: empty file {filename}")

    def test_all_generate_task(self):
        for task_class in ALL_TASKS:
            task = task_class()
            with self.subTest(task=task.task_id):
                codebase = task.gen_codebase()
                task_desc = task.gen_task(codebase)
                self.assertIsInstance(task_desc, str)
                self.assertGreater(len(task_desc), 100,
                    f"{task.task_id}: task description too short ({len(task_desc)} chars)")

    def test_all_generate_solution(self):
        for task_class in ALL_TASKS:
            task = task_class()
            with self.subTest(task=task.task_id):
                codebase = task.gen_codebase()
                solution = task.gen_solution(codebase)
                self.assertIsInstance(solution, dict)
                self.assertGreater(len(solution), 0, f"{task.task_id}: empty solution")

    def test_all_generate_reasoning(self):
        for task_class in ALL_TASKS:
            task = task_class()
            with self.subTest(task=task.task_id):
                codebase = task.gen_codebase()
                solution = task.gen_solution(codebase)
                reasoning = task.gen_reasoning(codebase, solution)
                self.assertIsInstance(reasoning, str)
                self.assertGreater(len(reasoning), 200,
                    f"{task.task_id}: reasoning too short ({len(reasoning)} chars)")


class TestPerfectResponse(unittest.TestCase):
    """Test that perfect responses get high scores."""

    def test_perfect_response_high_score(self):
        for task_class in ALL_TASKS:
            task = task_class()
            with self.subTest(task=task.task_id):
                perfect = task.build_perfect_response()
                codebase = task.gen_codebase()
                score, breakdown = task.grade(codebase, perfect)

                correctness = breakdown.get('correctness', 0)
                final = breakdown.get('final_score', 0)

                self.assertGreater(correctness, 0.5,
                    f"{task.task_id}: perfect response correctness={correctness:.2f} (expected >0.5)")
                self.assertGreater(final, 0.5,
                    f"{task.task_id}: perfect response final={final:.2f} (expected >0.5)")


class TestBadResponse(unittest.TestCase):
    """Test that bad responses get low scores."""

    BAD_RESPONSE = (
        "<reasoning>\nI think we need to change something.\n"
        "</reasoning>\n<answer>\n```python:main.py\npass\n```\n</answer>"
    )

    def test_bad_response_low_score(self):
        for task_class in ALL_TASKS:
            task = task_class()
            with self.subTest(task=task.task_id):
                codebase = task.gen_codebase()
                score, breakdown = task.grade(codebase, self.BAD_RESPONSE)
                final = breakdown.get('final_score', 0)
                self.assertLess(final, 0.5,
                    f"{task.task_id}: bad response final={final:.2f} (expected <0.5)")


class TestSFTExample(unittest.TestCase):
    """Test SFT example generation."""

    def test_all_generate_sft_example(self):
        for task_class in ALL_TASKS:
            task = task_class()
            with self.subTest(task=task.task_id):
                example = task.build_sft_example()
                self.assertIn('messages', example)
                self.assertEqual(len(example['messages']), 2)
                self.assertEqual(example['messages'][0]['role'], 'user')
                self.assertEqual(example['messages'][1]['role'], 'assistant')
                self.assertIn('metadata', example)
                self.assertEqual(example['metadata']['task_id'], task.task_id)

    def test_sft_example_has_reasoning_and_answer(self):
        for task_class in ALL_TASKS:
            task = task_class()
            with self.subTest(task=task.task_id):
                example = task.build_sft_example()
                assistant_msg = example['messages'][1]['content']
                self.assertIn('<reasoning>', assistant_msg)
                self.assertIn('</reasoning>', assistant_msg)
                self.assertIn('<answer>', assistant_msg)
                self.assertIn('</answer>', assistant_msg)


class TestEfficiencyScoring(unittest.TestCase):
    """Test the efficiency-aware reasoning quality scorer."""

    def test_extract_reasoning(self):
        response = "<reasoning>\nThis is my reasoning.\n</reasoning>\n<answer>\nCode here.\n</answer>"
        reasoning = extract_reasoning(response)
        self.assertIn("This is my reasoning", reasoning)

    def test_extract_answer(self):
        response = "<reasoning>\nThis is my reasoning.\n</reasoning>\n<answer>\nCode here.\n</answer>"
        answer = extract_answer(response)
        self.assertIn("Code here", answer)

    def test_compute_final_score(self):
        # Wrong answer -> 0 regardless of reasoning
        self.assertEqual(compute_final_score(0.0, 1.0), 0.0)
        # Right answer, no reasoning -> 0.6
        self.assertAlmostEqual(compute_final_score(1.0, 0.0), 0.6)
        # Right answer, perfect reasoning -> 1.0
        self.assertAlmostEqual(compute_final_score(1.0, 1.0), 1.0)
        # Partial both
        self.assertAlmostEqual(compute_final_score(0.5, 0.5), 0.4)

    def test_shape_rl_reward(self):
        # No reasoning -> heavy penalty
        reward = shape_rl_reward(1.0, 0.5, "", has_reasoning=False, has_answer=True)
        self.assertLess(reward, 0.6)
        # No answer -> penalty
        reward = shape_rl_reward(1.0, 0.5, "", has_reasoning=True, has_answer=False)
        self.assertLess(reward, 0.6)
        # Both present, high quality -> bonus
        reward = shape_rl_reward(1.0, 0.9, "verify check", has_reasoning=True, has_answer=True)
        self.assertGreater(reward, 0.95)

    def test_reasoning_quality_coverage(self):
        # High coverage
        response = "<reasoning>\ntrace cascade first bug second bug third bug data flow verify test\n</reasoning>"
        rq, breakdown = score_reasoning_quality(
            response,
            expected_concepts=["trace", "cascade", "first bug", "second bug",
                             "third bug", "data flow", "verify", "test"],
            token_budget=600,
            correctness=1.0,
        )
        self.assertGreater(breakdown.coverage, 0.9)
        self.assertGreater(rq, 0.5)

    def test_reasoning_quality_filler_penalty(self):
        # With filler
        response_with_filler = (
            "<reasoning>\nLet me think about this. This is an interesting problem. "
            "I need to analyze the code. trace cascade verify test\n</reasoning>"
        )
        rq_filler, breakdown_filler = score_reasoning_quality(
            response_with_filler,
            expected_concepts=["trace", "cascade", "verify", "test"],
            token_budget=600,
            correctness=1.0,
        )
        # Without filler
        response_no_filler = (
            "<reasoning>\ntrace cascade verify test\n</reasoning>"
        )
        rq_no_filler, breakdown_no_filler = score_reasoning_quality(
            response_no_filler,
            expected_concepts=["trace", "cascade", "verify", "test"],
            token_budget=600,
            correctness=1.0,
        )
        self.assertGreater(breakdown_filler.filler_penalty, 0)
        self.assertEqual(breakdown_no_filler.filler_penalty, 0)

    def test_reasoning_quality_verification(self):
        # With verification
        response_verified = (
            "<reasoning>\ntrace the code. Let me verify by checking the output. "
            "Let me trace through the test case.\n</reasoning>"
        )
        rq, breakdown = score_reasoning_quality(
            response_verified,
            expected_concepts=["trace"],
            token_budget=600,
            correctness=1.0,
        )
        self.assertGreater(breakdown.verification, 0)
        self.assertGreater(len(breakdown.verification_evidence), 0)

    def test_reasoning_quality_efficiency(self):
        # Within budget
        short_reasoning = "<reasoning>\ntrace verify test\n</reasoning>"
        rq_short, bd_short = score_reasoning_quality(
            short_reasoning,
            expected_concepts=["trace", "verify", "test"],
            token_budget=600,
            correctness=1.0,
        )
        # Way over budget
        long_reasoning = "<reasoning>\n" + "trace verify test " * 500 + "\n</reasoning>"
        rq_long, bd_long = score_reasoning_quality(
            long_reasoning,
            expected_concepts=["trace", "verify", "test"],
            token_budget=600,
            correctness=1.0,
        )
        self.assertGreater(bd_short.token_efficiency, bd_long.token_efficiency)

    def test_wrong_answer_no_efficiency_credit(self):
        # Wrong answer -> no efficiency credit even if concise
        response = "<reasoning>\ntrace verify test\n</reasoning>"
        rq, breakdown = score_reasoning_quality(
            response,
            expected_concepts=["trace", "verify", "test"],
            token_budget=600,
            correctness=0.0,  # wrong answer
        )
        self.assertEqual(breakdown.token_efficiency, 0.0)


class TestInstanceGeneration(unittest.TestCase):
    """Test full instance generation."""

    def test_generate_instance(self):
        for task_class in ALL_TASKS:
            task = task_class()
            with self.subTest(task=task.task_id):
                inst = task.generate_instance()
                self.assertIn('task_id', inst)
                self.assertIn('reasoning_skill', inst)
                self.assertIn('codebase', inst)
                self.assertIn('task', inst)
                self.assertIn('solution', inst)
                self.assertIn('reasoning', inst)
                self.assertEqual(inst['task_id'], task.task_id)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Long-Horizon tasks")
    parser.add_argument("--task", type=str, help="test specific task only")
    args = parser.parse_args()

    if args.task:
        # Run tests for specific task only
        task_class = register_long_horizon[args.task]
        task = task_class()
        print(f"\nTesting task: {task.task_id}")
        print(f"  Skill: {task.reasoning_skill}")
        print(f"  Failure mode: {task.failure_mode}")
        print(f"  Token budget: {task.token_budget}")
        print(f"  Expected concepts: {task.expected_concepts}")

        codebase = task.gen_codebase()
        print(f"\n  Codebase: {len(codebase)} files")
        for fname, content in codebase.items():
            print(f"    {fname}: {len(content)} chars, {len(content.splitlines())} lines")

        perfect = task.build_perfect_response()
        score, breakdown = task.grade(codebase, perfect)
        print(f"\n  Perfect response score: {score:.3f}")
        print(f"    Correctness: {breakdown['correctness']:.3f}")
        print(f"    Reasoning quality: {breakdown['reasoning_quality']:.3f}")
        print(f"    Coverage: {breakdown['reasoning_details']['coverage']:.3f}")
        print(f"    Token efficiency: {breakdown['reasoning_details']['token_efficiency']:.3f}")
        print(f"    Verification: {breakdown['reasoning_details']['verification']:.3f}")

        bad = ("<reasoning>\nI think we need to change something.\n</reasoning>\n"
               "<answer>\n```python:main.py\npass\n```\n</answer>")
        score_bad, _ = task.grade(codebase, bad)
        print(f"\n  Bad response score: {score_bad:.3f}")

    else:
        # Run all tests
        unittest.main(verbosity=2)


if __name__ == '__main__':
    main()
