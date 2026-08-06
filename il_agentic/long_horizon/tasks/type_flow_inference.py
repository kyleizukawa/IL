"""
Long-horizon task: type_flow_inference

Reasoning skill: Type inference reasoning across function boundaries.
Failure mode: Small models annotate each function in isolation without
tracking type flow through a call chain.

The codebase is a `pipeline.py` module with 5 functions that chain:
load_data -> parse_record -> validate -> transform -> export.

Types flow through the chain: load_data returns List[str], parse_record
takes str returns Optional[Dict], validate takes Dict returns bool,
transform takes Dict returns Dict, export takes List[Dict] returns str.

The tricky part: parse_record returns Optional[Dict] (None for invalid
records), and the pipeline must handle this. The model must infer all
types and add annotations.
"""
import ast
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer as grader_extract_answer,
    extract_reasoning as grader_extract_reasoning,
    parse_code_blocks, apply_code_changes, run_tests, run_code,
    compute_test_score, CodeExecutor, code_similarity,
)


@register_long_horizon
class TypeFlowInferenceEnv(LongHorizonEnv):
    """Infer and annotate types across a 5-function pipeline chain."""

    task_id = "type_flow_inference"
    reasoning_skill = "Type inference reasoning across function boundaries"
    failure_mode = (
        "Small models annotate each function in isolation without tracking "
        "type flow through a call chain"
    )
    token_budget = 700
    expected_concepts = [
        "type", "flow", "infer", "return type", "parameter",
        "Optional", "List", "Dict", "verify",
    ]

    # ── Codebase ──

    def gen_codebase(self) -> dict[str, str]:
        pipeline = textwrap.dedent('''\
            import json
            from typing import Any


            def load_data(raw_text):
                """Parse raw text into a list of record strings (one per line)."""
                lines = raw_text.strip().split("\\n")
                return [line.strip() for line in lines if line.strip()]


            def parse_record(line):
                """Parse a single record string into a dictionary.

                Returns None if the record is malformed.
                Format: "key=val,key=val,..."
                """
                if "=" not in line:
                    return None
                result = {}
                for pair in line.split(","):
                    if "=" not in pair:
                        return None
                    key, val = pair.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if not key:
                        return None
                    result[key] = val
                return result


            def validate(record):
                """Check that a record has required fields: 'id' and 'name'."""
                if record is None:
                    return False
                if "id" not in record:
                    return False
                if "name" not in record:
                    return False
                return True


            def transform(record):
                """Transform a validated record into a normalized form.

                Converts 'id' to int and uppercases 'name'.
                """
                result = dict(record)
                result["id"] = int(result["id"])
                result["name"] = result["name"].upper()
                return result


            def export(records):
                """Export a list of records as a JSON string."""
                return json.dumps(records, indent=2)


            def run_pipeline(raw_text):
                """Run the full pipeline: load -> parse -> validate -> transform -> export."""
                lines = load_data(raw_text)
                parsed = [parse_record(line) for line in lines]
                valid = [r for r in parsed if validate(r)]
                transformed = [transform(r) for r in valid]
                return export(transformed)
        ''')
        test_file = textwrap.dedent('''\
            import ast
            from pipeline import (
                load_data, parse_record, validate, transform, export, run_pipeline,
            )


            # ── Behavior tests (code must still work) ──

            def test_load_data():
                data = load_data("a=1,b=2\\nc=3,d=4")
                assert data == ["a=1,b=2", "c=3,d=4"]

            def test_parse_record_valid():
                rec = parse_record("id=1,name=alice")
                assert rec == {"id": "1", "name": "alice"}

            def test_parse_record_invalid():
                assert parse_record("no_equals_here") is None
                assert parse_record("=nokey") is None

            def test_validate():
                assert validate({"id": "1", "name": "alice"}) is True
                assert validate({"name": "alice"}) is False
                assert validate(None) is False

            def test_transform():
                rec = {"id": "42", "name": "bob"}
                result = transform(rec)
                assert result["id"] == 42
                assert result["name"] == "BOB"

            def test_export():
                out = export([{"id": 1, "name": "ALICE"}])
                assert '"id": 1' in out
                assert '"name": "ALICE"' in out

            def test_run_pipeline():
                raw = "id=1,name=alice\\nid=2,name=bob\\nbad_record"
                result = run_pipeline(raw)
                assert '"id": 1' in result
                assert '"id": 2' in result
                assert "bad_record" not in result

            # ── Annotation tests (all functions must have type annotations) ──

            def _get_annotations(source, func_name):
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == func_name:
                        return node
                return None

            def test_load_data_annotated():
                import pipeline
                node = _get_annotations(pipeline.__source__, "load_data")
                assert node is not None, "load_data not found"
                assert node.returns is not None, "load_data missing return annotation"
                assert len(node.args.args) > 0, "load_data has no parameters"

            def test_parse_record_annotated():
                import pipeline
                node = _get_annotations(pipeline.__source__, "parse_record")
                assert node is not None
                assert node.returns is not None, "parse_record missing return annotation"

            def test_all_functions_annotated():
                import pipeline
                for fname in ["load_data", "parse_record", "validate",
                              "transform", "export", "run_pipeline"]:
                    node = _get_annotations(pipeline.__source__, fname)
                    assert node is not None, f"{fname} not found"
                    assert node.returns is not None, f"{fname} missing return type"
                    for arg in node.args.args:
                        assert arg.annotation is not None, (
                            f"{fname} parameter {arg.arg} missing annotation"
                        )
        ''')
        return {
            "pipeline.py": pipeline,
            "test_pipeline.py": test_file,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''\
            You are given a `pipeline.py` module with a data processing
            pipeline consisting of 5 chained functions:

                load_data -> parse_record -> validate -> transform -> export

            None of these functions have type annotations. Your task is to
            infer the correct types for each function's parameters and
            return values by tracing the type flow through the chain, then
            add complete type annotations to every function.

            Key considerations:
            - `parse_record` returns `Optional[Dict]` (None for invalid
              records). You must trace how this None flows through the
              pipeline.
            - `validate` takes a `Dict` (or `Optional[Dict]`) and returns
              a `bool`.
            - `transform` takes a `Dict` and returns a `Dict`.
            - `load_data` returns `List[str]`, `export` takes `List[Dict]`
              and returns `str`.
            - Use `from typing import List, Dict, Optional` for annotations.

            The annotated code must still pass all behavior tests. The test
            suite also checks that every function has annotations on all
            parameters and return types.

            Provide the annotated `pipeline.py` in a
            ```python:pipeline.py``` code block. The module must set
            `__source__` to its own source code (the tests use this to
            check annotations via AST inspection).
        ''')

    # ── Solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        annotated = textwrap.dedent('''\
            import json
            from typing import Any, Dict, List, Optional


            def load_data(raw_text: str) -> List[str]:
                """Parse raw text into a list of record strings (one per line)."""
                lines = raw_text.strip().split("\\n")
                return [line.strip() for line in lines if line.strip()]


            def parse_record(line: str) -> Optional[Dict[str, str]]:
                """Parse a single record string into a dictionary.

                Returns None if the record is malformed.
                Format: "key=val,key=val,..."
                """
                if "=" not in line:
                    return None
                result: Dict[str, str] = {}
                for pair in line.split(","):
                    if "=" not in pair:
                        return None
                    key, val = pair.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if not key:
                        return None
                    result[key] = val
                return result


            def validate(record: Optional[Dict[str, str]]) -> bool:
                """Check that a record has required fields: 'id' and 'name'."""
                if record is None:
                    return False
                if "id" not in record:
                    return False
                if "name" not in record:
                    return False
                return True


            def transform(record: Dict[str, str]) -> Dict[str, Any]:
                """Transform a validated record into a normalized form.

                Converts 'id' to int and uppercases 'name'.
                """
                result: Dict[str, Any] = dict(record)
                result["id"] = int(result["id"])
                result["name"] = result["name"].upper()
                return result


            def export(records: List[Dict[str, Any]]) -> str:
                """Export a list of records as a JSON string."""
                return json.dumps(records, indent=2)


            def run_pipeline(raw_text: str) -> str:
                """Run the full pipeline: load -> parse -> validate -> transform -> export."""
                lines = load_data(raw_text)
                parsed = [parse_record(line) for line in lines]
                valid = [r for r in parsed if validate(r)]
                transformed = [transform(r) for r in valid]
                return export(transformed)


            # Expose source for AST-based annotation tests
            __source__ = __import__("inspect").getsource(__import__("sys").modules[__name__])
        ''')
        return {"pipeline.py": annotated}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''\
            I need to infer types for a 5-function pipeline by tracing the
            type flow through the chain: load_data -> parse_record -> validate
            -> transform -> export. Let me trace each function's input and
            output types by examining how data flows.

            ── load_data ──
            Input: `raw_text` — used with `.strip()` and `.split("\\n")`,
            so it's a `str`.
            Output: returns a list comprehension over lines, each `line.strip()`
            is a `str`. So return type is `List[str]`.
            Annotation: `def load_data(raw_text: str) -> List[str]`

            ── parse_record ──
            Input: `line` — used with `.split(",")` and `split("=", 1)`, so
            it's a `str`. This comes from `load_data`'s output (List[str]),
            confirming the flow.
            Output: Returns `None` in error cases, or `result` which is a
            `Dict[str, str]` (keys and values are both strings from the
            split). So return type is `Optional[Dict[str, str]]`.
            This is the tricky part — the return type is Optional because
            invalid records return None. I need to trace how this None
            flows downstream.
            Annotation: `def parse_record(line: str) -> Optional[Dict[str, str]]`

            ── validate ──
            Input: `record` — checked with `if record is None: return False`,
            then accessed with `in` operator. It receives the output of
            parse_record, which is `Optional[Dict[str, str]]`. So the
            parameter type must be `Optional[Dict[str, str]]` to handle
            the None case.
            Output: returns `True` or `False` — type `bool`.
            Annotation: `def validate(record: Optional[Dict[str, str]]) -> bool`

            ── transform ──
            Input: `record` — accessed with `dict(record)`, `result["id"]`
            converted to `int`, `result["name"]` uppercased. It receives
            records that passed validation (so they're not None — the
            `run_pipeline` filters with `if validate(r)`). But the function
            itself takes a `Dict[str, str]` and transforms it.
            Output: `result` is a copy of record with `id` converted to int
            and `name` uppercased. So values can be `str` or `int` — the
            return type is `Dict[str, Any]`.
            Annotation: `def transform(record: Dict[str, str]) -> Dict[str, Any]`

            ── export ──
            Input: `records` — passed to `json.dumps`. In `run_pipeline`,
            this is a list of transformed records: `List[Dict[str, Any]]`.
            Output: `json.dumps` returns a `str`.
            Annotation: `def export(records: List[Dict[str, Any]]) -> str`

            ── run_pipeline ──
            Input: `raw_text: str` (same as load_data).
            Output: returns `export(transformed)` which is `str`.
            Annotation: `def run_pipeline(raw_text: str) -> str`

            Now let me verify the type flow is consistent:
            1. load_data(raw_text: str) -> List[str]
            2. parse_record(line: str) -> Optional[Dict[str, str]]
               - Input str comes from List[str] ✓
            3. validate(record: Optional[Dict[str, str]]) -> bool
               - Input Optional[Dict] comes from parse_record ✓
            4. transform(record: Dict[str, str]) -> Dict[str, Any]
               - Input Dict comes from filtered valid records (not None) ✓
            5. export(records: List[Dict[str, Any]]) -> str
               - Input List[Dict] comes from list of transform outputs ✓

            The critical inference is that parse_record returns Optional,
            and validate must accept Optional to handle the None case.
            The list comprehension in run_pipeline filters out invalid
            records with `if validate(r)`, so transform only receives
            non-None Dicts.

            Let me verify the code still runs correctly. The annotations
            are just type hints — they don't change runtime behavior. The
            only concern is the `__source__` variable that the tests use
            for AST inspection. I'll use `inspect.getsource` to capture
            the module's own source.

            Let me trace through test_run_pipeline to confirm:
            - raw = "id=1,name=alice\\nid=2,name=bob\\nbad_record"
            - load_data -> ["id=1,name=alice", "id=2,name=bob", "bad_record"]
            - parse_record each:
              - "id=1,name=alice" -> {"id": "1", "name": "alice"}
              - "id=2,name=bob" -> {"id": "2", "name": "bob"}
              - "bad_record" -> None (no "=" sign)
            - validate each: True, True, False
            - valid = [{"id":"1","name":"alice"}, {"id":"2","name":"bob"}]
            - transform each: [{"id":1,"name":"ALICE"}, {"id":2,"name":"BOB"}]
            - export -> JSON string with id:1 and id:2, no "bad_record"
            - Result contains '"id": 1' ✓, '"id": 2' ✓, no "bad_record" ✓

            All types inferred correctly through the flow. The Optional
            handling in validate is the key insight that requires tracking
            type flow across function boundaries rather than annotating
            in isolation.
        ''')

    # ── Grading ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        answer = grader_extract_answer(response)
        blocks = parse_code_blocks(answer)
        if not blocks:
            blocks = parse_code_blocks(response)

        if "pipeline.py" not in blocks:
            return 0.0, {
                "reason": "no pipeline.py code block found",
                "test_details": {},
            }

        fixed_codebase = apply_code_changes(codebase, blocks)

        # ── 1. Run tests (30%) ──
        test_code = codebase.get("test_pipeline.py", "")
        results = run_tests(fixed_codebase, test_code, timeout=15.0)
        test_score, test_details = compute_test_score(results)
        runs_correctly = 1.0 if test_score >= 0.8 else test_score

        # ── 2. Annotation coverage (70%) ──
        annotation_coverage = 0.0
        try:
            source = blocks["pipeline.py"]
            tree = ast.parse(source)
            func_names = [
                "load_data", "parse_record", "validate",
                "transform", "export", "run_pipeline",
            ]
            annotated = 0
            total = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in func_names:
                    total += 1
                    has_return = node.returns is not None
                    has_params = all(
                        arg.annotation is not None
                        for arg in node.args.args
                    ) if node.args.args else True
                    if has_return and has_params:
                        annotated += 1
            if total > 0:
                annotation_coverage = annotated / total
        except SyntaxError:
            annotation_coverage = 0.0

        # Check for Optional usage (key insight)
        has_optional = "Optional" in blocks.get("pipeline.py", "")
        if has_optional:
            annotation_coverage = min(1.0, annotation_coverage + 0.05)

        score = annotation_coverage * 0.7 + runs_correctly * 0.3
        breakdown = {
            "annotation_coverage": annotation_coverage,
            "runs_correctly": runs_correctly,
            "test_score": test_score,
            "test_details": test_details,
            "has_optional": has_optional,
        }
        return score, breakdown
