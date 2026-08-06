"""
Task: cascading_bug_chain

Reasoning skill: Multi-step reasoning where each step depends on the previous.

A 3-module data processing pipeline (input_parser -> transformer ->
output_formatter) has three bugs stacked in sequence.  Each bug is only visible
after the previous one is fixed, because the downstream module masks the
symptom until the upstream data is correct.

Bug 1 (input_parser): off-by-one in field extraction that drops the last
character of each field.
Bug 2 (transformer): only visible after bug 1 is fixed -- a type coercion
error that converts "3.0" to int 3 instead of float 3.0 when the input string
has no decimal point.
Bug 3 (output_formatter): only visible after bug 2 is fixed -- a format string
that uses %d instead of %f, truncating float values.

The model must fix all 3 bugs in sequence.  Fixing only 1 or 2 gives partial
credit.  The tests are designed so that fixing bugs in the wrong order does not
help.

Failure mode: small models fix the first bug they see and miss cascading
failures that only surface after the upstream fix.
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer, extract_reasoning, parse_code_blocks,
    apply_code_changes, run_tests, compute_test_score,
)


@register_long_horizon
class CascadingBugChain(LongHorizonEnv):
    task_id = "cascading_bug_chain"
    reasoning_skill = "Multi-step reasoning where each step depends on the previous"
    failure_mode = (
        "Small models fix the first bug they see and miss cascading failures "
        "that only surface after the upstream fix is applied."
    )
    token_budget = 800
    expected_concepts = [
        "trace", "cascade", "first bug", "second bug", "third bug",
        "data flow", "verify", "test",
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        input_parser = textwrap.dedent('''
            """Input parser — splits raw CSV-like lines into named fields.

            Each input line looks like:  name,age,score
            The parser returns a dict with keys "name", "age", "score".
            """

            FIELDS = ("name", "age", "score")


            def parse_line(line: str) -> dict:
                """Parse a single comma-separated line into a field dict.

                BUG 1: the field extraction uses range(1, len(parts)) which
                drops the last character of each field value.  This is an
                off-by-one error in the slice logic.
                """
                parts = line.split(",")
                result = {}
                for i, field in enumerate(FIELDS):
                    if i < len(parts):
                        # BUG: drops the last character of every field.
                        # Should be parts[i], not parts[i][:-1].
                        result[field] = parts[i][:-1]
                return result


            def parse_lines(lines: list[str]) -> list[dict]:
                """Parse multiple lines into a list of field dicts."""
                return [parse_line(line) for line in lines]
        ''').strip()

        transformer = textwrap.dedent('''
            """Transformer — converts parsed field dicts into typed records.

            Takes a dict with string fields and produces a dict with proper
            Python types: name stays str, age becomes int, score becomes float.
            """

            def transform_record(record: dict) -> dict:
                """Transform a single parsed record into typed form.

                BUG 2: the score coercion uses int() when the string has no
                decimal point, losing the float type.  A value like "30"
                should become 30.0 (float), not 30 (int).  This bug is masked
                when bug 1 is present because the truncated score string is
                often malformed and the test never reaches the type check.
                """
                result = {}
                result["name"] = record.get("name", "")
                result["age"] = int(record.get("age", "0"))
                score_str = record.get("score", "0.0")
                # BUG: uses int() when there is no decimal point.
                if "." not in score_str:
                    result["score"] = int(score_str)
                else:
                    result["score"] = float(score_str)
                return result


            def transform_records(records: list[dict]) -> list[dict]:
                """Transform multiple parsed records."""
                return [transform_record(r) for r in records]
        ''').strip()

        output_formatter = textwrap.dedent('''
            """Output formatter — renders typed records as formatted strings.

            Produces lines like:  name=alice age=30 score=95.50
            """

            def format_record(record: dict) -> str:
                """Format a single typed record as a display string.

                BUG 3: the score field uses %d (integer format) instead of %f,
                truncating float values.  This bug is masked when bug 2 is
                present because the score is already an int, so %d works by
                accident.  Only after bug 2 is fixed (score is float) does %d
                truncate the fractional part.
                """
                # BUG: %d truncates floats; should be %.2f.
                return "name=%s age=%d score=%d" % (
                    record["name"],
                    record["age"],
                    record["score"],
                )


            def format_records(records: list[dict]) -> list[str]:
                """Format multiple typed records."""
                return [format_record(r) for r in records]
        ''').strip()

        pipeline = textwrap.dedent('''
            """Pipeline orchestrator — chains parser, transformer, formatter."""

            from input_parser import parse_lines
            from transformer import transform_records
            from output_formatter import format_records


            def run_pipeline(lines: list[str]) -> list[str]:
                """Run the full pipeline: parse -> transform -> format."""
                parsed = parse_lines(lines)
                transformed = transform_records(parsed)
                formatted = format_records(transformed)
                return formatted
        ''').strip()

        tests = textwrap.dedent('''
            from input_parser import parse_line, parse_lines
            from transformer import transform_record, transform_records
            from output_formatter import format_record, format_records
            from pipeline import run_pipeline


            # ── Tests that pass with the base (buggy) code ──
            # These exercise code paths that are not affected by the bugs.

            def test_parse_empty_line():
                result = parse_line("")
                # Empty line -> one part (empty string), truncated to empty.
                assert result.get("name") == ""

            def test_parse_single_field():
                result = parse_line("alice")
                # Single field: "alice" truncated to "alic" by bug 1.
                # This test checks the (buggy) behavior so it passes on base.
                assert "name" in result

            # ── Tests that pass only after bug 1 is fixed ──

            def test_parse_preserves_full_name():
                result = parse_line("alice,30,95.5")
                assert result["name"] == "alice", f"name was {result.get('name')!r}"

            def test_parse_preserves_full_age():
                result = parse_line("alice,30,95.5")
                assert result["age"] == "30", f"age was {result.get('age')!r}"

            def test_parse_preserves_full_score():
                result = parse_line("alice,30,95.5")
                assert result["score"] == "95.5", f"score was {result.get('score')!r}"

            # ── Tests that pass only after bug 2 is fixed ──

            def test_transform_score_is_float_with_decimal():
                record = {"name": "alice", "age": "30", "score": "95.5"}
                result = transform_record(record)
                assert isinstance(result["score"], float), f"score type was {type(result['score'])}"
                assert result["score"] == 95.5

            def test_transform_score_is_float_without_decimal():
                record = {"name": "bob", "age": "25", "score": "30"}
                result = transform_record(record)
                assert isinstance(result["score"], float), f"score type was {type(result['score'])}"
                assert result["score"] == 30.0

            # ── Test that passes only after bug 3 is fixed ──

            def test_format_preserves_float_score():
                record = {"name": "alice", "age": 30, "score": 95.5}
                line = format_record(record)
                # After bug 3 fix, score should show as 95.50 (or 95.5).
                assert "95.5" in line, f"line was {line!r}"

            # ── Full pipeline integration test ──

            def test_full_pipeline_end_to_end():
                lines = ["alice,30,95.5"]
                output = run_pipeline(lines)
                assert len(output) == 1
                line = output[0]
                assert "alice" in line, f"line was {line!r}"
                assert "30" in line, f"line was {line!r}"
                assert "95.5" in line, f"line was {line!r}"
        ''').strip()

        return {
            "input_parser.py": input_parser,
            "transformer.py": transformer,
            "output_formatter.py": output_formatter,
            "pipeline.py": pipeline,
            "test_cascading.py": tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''
            You are given a 3-module data processing pipeline:

              input_parser.py  ->  transformer.py  ->  output_formatter.py

            The pipeline parses comma-separated lines, transforms the fields
            into typed records, and formats them as display strings.

            There are THREE bugs in this pipeline, stacked so that each bug
            is only visible after the previous one is fixed:

            1. Bug 1 in `input_parser.py`: an off-by-one error that drops the
               last character of each field.
            2. Bug 2 in `transformer.py`: a type coercion error that converts
               score strings without a decimal point to int instead of float.
               This is masked by bug 1 because the truncated score strings
               are malformed.
            3. Bug 3 in `output_formatter.py`: a format string that uses %d
               instead of %f for the score, truncating float values.  This is
               masked by bug 2 because the score is already an int.

            Fix all three bugs in sequence.  All 8 tests in
            `test_cascading.py` must pass.  Fixing only 1 or 2 bugs will give
            partial credit (some tests will still fail).

            Return your solution as code blocks tagged with the filename:

            ```python:input_parser.py
            ...
            ```
        ''').strip()

    # ── Correct solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        input_parser = textwrap.dedent('''
            """Input parser — splits raw CSV-like lines into named fields."""

            FIELDS = ("name", "age", "score")


            def parse_line(line: str) -> dict:
                """Parse a single comma-separated line into a field dict."""
                parts = line.split(",")
                result = {}
                for i, field in enumerate(FIELDS):
                    if i < len(parts):
                        result[field] = parts[i]
                return result


            def parse_lines(lines: list[str]) -> list[dict]:
                """Parse multiple lines into a list of field dicts."""
                return [parse_line(line) for line in lines]
        ''').strip()

        transformer = textwrap.dedent('''
            """Transformer — converts parsed field dicts into typed records."""

            def transform_record(record: dict) -> dict:
                """Transform a single parsed record into typed form."""
                result = {}
                result["name"] = record.get("name", "")
                result["age"] = int(record.get("age", "0"))
                score_str = record.get("score", "0.0")
                result["score"] = float(score_str)
                return result


            def transform_records(records: list[dict]) -> list[dict]:
                """Transform multiple parsed records."""
                return [transform_record(r) for r in records]
        ''').strip()

        output_formatter = textwrap.dedent('''
            """Output formatter — renders typed records as formatted strings."""

            def format_record(record: dict) -> str:
                """Format a single typed record as a display string."""
                return "name=%s age=%d score=%.2f" % (
                    record["name"],
                    record["age"],
                    record["score"],
                )


            def format_records(records: list[dict]) -> list[str]:
                """Format multiple typed records."""
                return [format_record(r) for r in records]
        ''').strip()

        return {
            "input_parser.py": input_parser,
            "transformer.py": transformer,
            "output_formatter.py": output_formatter,
        }

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''
            Let me trace the data flow through this pipeline step by step,
            starting from the first module and moving forward.  The key
            challenge is that the bugs cascade — each bug masks the next, so I
            must fix them in order and re-trace after each fix.

            Step 1 — Trace the input_parser:
            I read `parse_line` and see `result[field] = parts[i][:-1]`.  The
            slice `[:-1]` drops the last character of every field.  For input
            "alice,30,95.5", this produces name="alic", age="3", score="95."
            — clearly wrong.  This is the first bug: an off-by-one in field
            extraction.  The fix is `result[field] = parts[i]` (no slicing).
            Let me verify: with the fix, "alice,30,95.5" produces
            name="alice", age="30", score="95.5".  Correct.

            Step 2 — Trace the transformer with correct input:
            Now that bug 1 is fixed, the transformer receives name="alice",
            age="30", score="95.5".  I read `transform_record` and see the
            score logic: `if "." not in score_str: result["score"] =
            int(score_str)`.  For "95.5" the else branch runs and produces
            float 95.5 — that works.  But for a score like "30" (no decimal
            point), the if branch runs and produces int 30, not float 30.0.
            This is the second bug: a type coercion error.  The score should
            always be a float.  The fix is `result["score"] =
            float(score_str)` unconditionally — float("30") returns 30.0.
            Let me verify: float("30") == 30.0 (type float), float("95.5")
            == 95.5 (type float).  Both correct.

            Why was this bug masked before?  When bug 1 was present, the
            score string was truncated.  For "95.5" it became "95." which
            float() would parse, but the test for float type was never
            reached because the name/age tests failed first.  The cascade
            hides the downstream bug until the upstream is fixed.

            Step 3 — Trace the output_formatter with correct typed data:
            Now that bug 2 is fixed, the formatter receives score=95.5
            (float).  I read `format_record` and see `"score=%d" %
            record["score"]`.  The %d format specifier expects an integer
            and truncates floats — 95.5 becomes 95.  This is the third bug:
            a format string using %d instead of %f.  The fix is
            `"score=%.2f" % record["score"]` which formats 95.5 as "95.50".
            Let me verify: "%.2f" % 95.5 == "95.50", and "95.5" is in
            "95.50".  The test checks `"95.5" in line`, which passes.

            Why was this bug masked before?  When bug 2 was present, the
            score was int 30 (for input "30"), and %d on an int works fine.
            Only after bug 2 is fixed (score becomes float) does %d
            truncate the fractional part.  This is the cascade: bug 3 is
            invisible until bug 2 is fixed.

            Step 4 — Verify the full pipeline end to end:
            Input: "alice,30,95.5"
            - parse_line: name="alice", age="30", score="95.5" (bug 1 fixed)
            - transform_record: name="alice", age=30, score=95.5 (bug 2 fixed)
            - format_record: "name=alice age=30 score=95.50" (bug 3 fixed)
            The test checks "alice" in line, "30" in line, "95.5" in line —
            all present.  Let me also check a score without a decimal:
            Input: "bob,25,30"
            - parse: name="bob", age="25", score="30"
            - transform: name="bob", age=25, score=30.0 (float, bug 2 fixed)
            - format: "name=bob age=25 score=30.00"
            The test test_transform_score_is_float_without_decimal checks
            isinstance(score, float) == True.  Correct.

            Step 5 — Confirm test progression:
            - test_parse_empty_line, test_parse_single_field: pass on base
              (2 tests).
            - test_parse_preserves_full_name/age/score: pass after bug 1
              fix (3 more tests).
            - test_transform_score_is_float_with/without_decimal: pass
              after bug 2 fix (2 more tests).
            - test_format_preserves_float_score, test_full_pipeline: pass
              after bug 3 fix (1 more test + integration).
            Total: 8 tests, all pass after all 3 fixes.  Fixing bugs in the
            wrong order does not help because each downstream test depends
            on the upstream fix.

            To confirm: I have traced the data flow through all three
            modules, identified each bug by examining the code and tracing
            the actual data values, fixed them in cascade order, and
            verified every test passes by mentally stepping through the
            pipeline.  The key insight is that cascading bugs require
            sequential debugging — fix the first bug, re-trace, find the
            next, repeat.
        ''').strip()

    # ── Grader ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        answer = extract_answer(response)
        changes = parse_code_blocks(answer)
        if not changes:
            reasoning = extract_reasoning(response)
            changes = parse_code_blocks(reasoning)
        if not changes:
            return 0.0, {"reason": "no code blocks found in response"}

        new_codebase = apply_code_changes(codebase, changes)
        test_code = codebase.get("test_cascading.py", "")
        results = run_tests(new_codebase, test_code, timeout=15.0)
        score, breakdown = compute_test_score(results)
        breakdown["results"] = results.get("results", [])
        breakdown["method"] = "run_tests"
        return score, breakdown
