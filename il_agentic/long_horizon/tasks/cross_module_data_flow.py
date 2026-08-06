"""
Task: cross_module_data_flow

Reasoning skill: Sustained reasoning over long code paths (5+ modules).

A 5-module data pipeline (loader -> validator -> enricher -> aggregator ->
exporter) has a single bug in the enricher module: it looks up scores using
the wrong key ("id" instead of "item_id").  The symptom (wrong output) only
appears in the exporter's final output, so the model must trace backwards
through all 5 modules to find the root cause.

The bug only affects items where id != item_id, which makes it subtle — some
items produce correct output and some do not, depending on whether the two
keys happen to match.

Failure mode: small models lose track of data transformations across modules
and fix symptoms in the exporter instead of tracing to the root cause in the
enricher.
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer, extract_reasoning, parse_code_blocks,
    apply_code_changes, run_tests, compute_test_score,
)


@register_long_horizon
class CrossModuleDataFlow(LongHorizonEnv):
    task_id = "cross_module_data_flow"
    reasoning_skill = "Sustained reasoning over long code paths (5+ modules)"
    failure_mode = (
        "Small models lose track of data transformations across modules and "
        "fix symptoms in the final module instead of tracing to the root cause."
    )
    token_budget = 900
    expected_concepts = [
        "trace", "data flow", "module", "transform",
        "corrupt", "where", "root cause", "verify",
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        loader = textwrap.dedent('''
            """Loader — reads raw item records from a data source.

            Each raw record has: id, item_id, name, category, value.
            The id is an internal sequence number; item_id is the
            business identifier.  They may differ.
            """

            def load_items(raw_data: list[dict]) -> list[dict]:
                """Load raw item dicts from the data source.

                Returns a list of dicts with keys: id, item_id, name,
                category, value.
                """
                items = []
                for record in raw_data:
                    items.append({
                        "id": record["id"],
                        "item_id": record["item_id"],
                        "name": record["name"],
                        "category": record["category"],
                        "value": record["value"],
                    })
                return items
        ''').strip()

        validator = textwrap.dedent('''
            """Validator — checks that all required fields are present.

            Drops items that fail validation and returns the rest unchanged.
            """

            REQUIRED_FIELDS = ("id", "item_id", "name", "category", "value")


            def validate_items(items: list[dict]) -> list[dict]:
                """Validate that each item has all required fields.

                Returns only the items that pass validation.
                """
                valid = []
                for item in items:
                    if all(f in item for f in REQUIRED_FIELDS):
                        valid.append(item)
                return valid
        ''').strip()

        enricher = textwrap.dedent('''
            """Enricher — adds a "score" field to each item by looking up
            the item_id in a reference table.

            BUG: uses "id" instead of "item_id" as the lookup key.  This
            causes wrong scores for items where id != item_id.  Items where
            id == item_id get the correct score by coincidence.
            """

            # Reference table: item_id -> score
            SCORE_TABLE = {
                "A100": 95,
                "B200": 80,
                "C300": 70,
                "D400": 60,
            }


            def enrich_items(items: list[dict]) -> list[dict]:
                """Add a score field to each item based on its item_id.

                BUG: looks up item["id"] instead of item["item_id"].
                When id != item_id, the wrong score (or no score) is
                assigned.
                """
                enriched = []
                for item in items:
                    new_item = dict(item)
                    # BUG: should be item["item_id"], not item["id"].
                    lookup_key = item["id"]
                    new_item["score"] = SCORE_TABLE.get(lookup_key, 0)
                    enriched.append(new_item)
                return enriched
        ''').strip()

        aggregator = textwrap.dedent('''
            """Aggregator — groups items by category and computes summary
            statistics (count, total_value, avg_score).
            """

            def aggregate_items(items: list[dict]) -> dict:
                """Aggregate items by category.

                Returns {category: {count, total_value, avg_score}}.
            """
                groups = {}
                for item in items:
                    cat = item["category"]
                    if cat not in groups:
                        groups[cat] = {
                            "count": 0,
                            "total_value": 0,
                            "scores": [],
                        }
                    groups[cat]["count"] += 1
                    groups[cat]["total_value"] += item["value"]
                    groups[cat]["scores"].append(item["score"])

                result = {}
                for cat, data in groups.items():
                    avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
                    result[cat] = {
                        "count": data["count"],
                        "total_value": data["total_value"],
                        "avg_score": avg_score,
                    }
                return result
        ''').strip()

        exporter = textwrap.dedent('''
            """Exporter — formats the aggregated data as a report string.

            This is where the symptom appears: wrong scores from the enricher
            propagate through the aggregator and show up as incorrect
            avg_score values in the final report.
            """

            def export_report(aggregated: dict) -> str:
                """Export the aggregated data as a formatted report string."""
                lines = []
                lines.append("Category Report")
                lines.append("-" * 40)
                for cat in sorted(aggregated.keys()):
                    data = aggregated[cat]
                    lines.append(
                        "%s: count=%d total_value=%d avg_score=%.1f"
                        % (cat, data["count"], data["total_value"], data["avg_score"])
                    )
                return "\\n".join(lines)
        ''').strip()

        pipeline = textwrap.dedent('''
            """Pipeline orchestrator — chains all 5 modules."""

            from loader import load_items
            from validator import validate_items
            from enricher import enrich_items
            from aggregator import aggregate_items
            from exporter import export_report


            def run_pipeline(raw_data: list[dict]) -> str:
                """Run the full 5-module pipeline."""
                items = load_items(raw_data)
                items = validate_items(items)
                items = enrich_items(items)
                aggregated = aggregate_items(items)
                report = export_report(aggregated)
                return report
        ''').strip()

        tests = textwrap.dedent('''
            from loader import load_items
            from validator import validate_items
            from enricher import enrich_items
            from aggregator import aggregate_items
            from exporter import export_report
            from pipeline import run_pipeline


            RAW_DATA = [
                {"id": "1", "item_id": "A100", "name": "widget",
                 "category": "tools", "value": 100},
                {"id": "2", "item_id": "B200", "name": "gadget",
                 "category": "tools", "value": 50},
                {"id": "3", "item_id": "C300", "name": "gizmo",
                 "category": "electronics", "value": 200},
                {"id": "A100", "item_id": "A100", "name": "special",
                 "category": "tools", "value": 75},
            ]


            # ── Individual module tests (all pass on base code) ──

            def test_loader_loads_all_fields():
                items = load_items(RAW_DATA)
                assert len(items) == 4
                assert items[0]["item_id"] == "A100"
                assert items[0]["name"] == "widget"

            def test_validator_keeps_valid_items():
                items = load_items(RAW_DATA)
                valid = validate_items(items)
                assert len(valid) == 4

            def test_aggregator_groups_by_category():
                items = [
                    {"id": "1", "item_id": "A100", "name": "w",
                     "category": "tools", "value": 100, "score": 95},
                    {"id": "2", "item_id": "B200", "name": "g",
                     "category": "tools", "value": 50, "score": 80},
                ]
                result = aggregate_items(items)
                assert "tools" in result
                assert result["tools"]["count"] == 2
                assert result["tools"]["total_value"] == 150

            # ── Full pipeline tests (fail because of enricher bug) ──

            def test_enricher_uses_item_id_not_id():
                items = load_items(RAW_DATA)
                items = validate_items(items)
                enriched = enrich_items(items)
                # Item with id="1", item_id="A100" should get score for A100 (95).
                item1 = enriched[0]
                assert item1["score"] == 95, f"score was {item1['score']}"

            def test_enricher_correct_for_mismatched_ids():
                items = load_items(RAW_DATA)
                items = validate_items(items)
                enriched = enrich_items(items)
                # Item with id="2", item_id="B200" should get score 80.
                item2 = enriched[1]
                assert item2["score"] == 80, f"score was {item2['score']}"
                # Item with id="3", item_id="C300" should get score 70.
                item3 = enriched[2]
                assert item3["score"] == 70, f"score was {item3['score']}"

            def test_pipeline_report_has_correct_scores():
                report = run_pipeline(RAW_DATA)
                # The tools category has items A100(95), B200(80), A100(95).
                # avg_score = (95 + 80 + 95) / 3 = 90.0
                assert "90.0" in report, f"report was:\\n{report}"
        ''').strip()

        return {
            "loader.py": loader,
            "validator.py": validator,
            "enricher.py": enricher,
            "aggregator.py": aggregator,
            "exporter.py": exporter,
            "pipeline.py": pipeline,
            "test_cross_module.py": tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''
            You are given a 5-module data processing pipeline:

              loader.py -> validator.py -> enricher.py -> aggregator.py
                          -> exporter.py

            The pipeline loads raw item records, validates them, enriches
            each with a score from a reference table, aggregates by
            category, and exports a report.

            There is a bug somewhere in the pipeline.  The symptom is that
            the final report (from `exporter.py`) has wrong avg_score values
            for some categories.  The bug only affects items where the "id"
            field differs from the "item_id" field.

            You must trace the data flow backwards from the wrong output
            through all 5 modules to find the root cause, then fix it.

            All 6 tests in `test_cross_module.py` must pass.  The 3
            individual module tests already pass; the 3 full pipeline tests
            fail because of the bug.

            Return your solution as a code block tagged with the filename:

            ```python:enricher.py
            ...
            ```
        ''').strip()

    # ── Correct solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        enricher = textwrap.dedent('''
            """Enricher — adds a "score" field to each item by looking up
            the item_id in a reference table.
            """

            # Reference table: item_id -> score
            SCORE_TABLE = {
                "A100": 95,
                "B200": 80,
                "C300": 70,
                "D400": 60,
            }


            def enrich_items(items: list[dict]) -> list[dict]:
                """Add a score field to each item based on its item_id."""
                enriched = []
                for item in items:
                    new_item = dict(item)
                    lookup_key = item["item_id"]
                    new_item["score"] = SCORE_TABLE.get(lookup_key, 0)
                    enriched.append(new_item)
                return enriched
        ''').strip()

        return {"enricher.py": enricher}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''
            Let me trace the data flow backwards from the wrong output to
            find the root cause.  The symptom is in the exporter's report:
            avg_score values are wrong for some categories.  I need to trace
            where the score data comes from and which module corrupt it.

            Step 1 — Start at the exporter:
            I read `exporter.py` and see `export_report` formats the
            aggregated dict.  It reads `data["avg_score"]` from each
            category group.  The exporter does not compute scores — it just
            displays them.  So the wrong avg_score must come from the
            aggregator's input, not the exporter itself.  The exporter is
            not where the bug is.

            Step 2 — Trace back to the aggregator:
            I read `aggregator.py` and see `aggregate_items` groups items
            by category and computes `avg_score = sum(scores) /
            len(scores)`.  The scores come from `item["score"]` on each
            item.  The aggregator does not modify scores — it just averages
            them.  So if avg_score is wrong, the individual item["score"]
            values must be wrong.  The aggregator is not where the bug is
            either.

            Step 3 — Trace back to the enricher:
            I read `enricher.py` and see `enrich_items` adds a "score"
            field to each item.  The score is looked up from SCORE_TABLE
            using a key.  I see `lookup_key = item["id"]`.  But the
            SCORE_TABLE is keyed by item_id values (A100, B200, C300,
            D400).  The code uses `item["id"]` instead of
            `item["item_id"]`.  This is the root cause.

            When id == item_id (e.g., the 4th item where both are "A100"),
            the lookup works by coincidence and the score is correct.  But
            when id != item_id (e.g., the 1st item: id="1",
            item_id="A100"), the lookup key "1" is not in SCORE_TABLE, so
            the score defaults to 0 instead of 95.  This is why only some
            items have wrong scores — the bug is data-dependent.

            Step 4 — Confirm the data flow:
            Let me trace the specific items through the pipeline:
            - Item 1: id="1", item_id="A100".  enricher looks up "1" ->
              not found -> score=0.  Should be 95 (for A100).  BUG.
            - Item 2: id="2", item_id="B200".  enricher looks up "2" ->
              not found -> score=0.  Should be 80.  BUG.
            - Item 3: id="3", item_id="C300".  enricher looks up "3" ->
              not found -> score=0.  Should be 70.  BUG.
            - Item 4: id="A100", item_id="A100".  enricher looks up
              "A100" -> found -> score=95.  Correct (by coincidence).

            The tools category has items 1, 2, 4 with scores 0, 0, 95.
            avg_score = (0+0+95)/3 = 31.7.  But the correct scores should
            be 95, 80, 95 -> avg = 90.0.  That matches the test
            `test_pipeline_report_has_correct_scores` which checks for
            "90.0" in the report.

            Step 5 — Fix the enricher:
            Change `lookup_key = item["id"]` to `lookup_key =
            item["item_id"]`.  This makes the lookup use the business
            identifier that matches the SCORE_TABLE keys.

            Step 6 — Verify by tracing forward:
            With the fix:
            - Item 1: item_id="A100" -> score=95.  Correct.
            - Item 2: item_id="B200" -> score=80.  Correct.
            - Item 3: item_id="C300" -> score=70.  Correct.
            - Item 4: item_id="A100" -> score=95.  Correct.

            Tools category: items 1,2,4 with scores 95,80,95.
            avg_score = (95+80+95)/3 = 90.0.  The report will contain
            "90.0".  Test passes.

            Electronics category: item 3 with score 70.
            avg_score = 70.0.  Correct.

            Let me also verify the individual module tests still pass:
            - test_loader_loads_all_fields: loader is unchanged.  OK.
            - test_validator_keeps_valid_items: validator is unchanged.
              OK.
            - test_aggregator_groups_by_category: aggregator is unchanged.
              OK.
            - test_enricher_uses_item_id_not_id: item1 score == 95.  OK.
            - test_enricher_correct_for_mismatched_ids: item2 score == 80,
              item3 score == 70.  OK.
            - test_pipeline_report_has_correct_scores: "90.0" in report.
              OK.

            To confirm: the root cause was in the enricher module, not the
            exporter where the symptom appeared.  By tracing the data flow
            backwards through all 5 modules — exporter (displays) ->
            aggregator (averages) -> enricher (assigns) — I found that the
            enricher used the wrong key ("id" instead of "item_id") for
            the score lookup.  The fix is a one-line change in the enricher,
            and I have verified all 6 tests pass by tracing the corrected
            data flow forward through the pipeline.
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
        test_code = codebase.get("test_cross_module.py", "")
        results = run_tests(new_codebase, test_code, timeout=15.0)
        score, breakdown = compute_test_score(results)
        breakdown["results"] = results.get("results", [])
        breakdown["method"] = "run_tests"
        return score, breakdown
