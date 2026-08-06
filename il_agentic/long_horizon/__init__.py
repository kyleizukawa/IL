"""
Long-Horizon Agentic Coding Environments — 20 hand-crafted tasks.

Built in the mechanize.work style: each task is a single, rich, hand-crafted
scenario with a real multi-file codebase (100-400 lines), designed to test
specific long-horizon reasoning capabilities in small models.

Key differences from the procedural il_agentic environments:
- Hand-crafted: each task is unique, not generated from templates
- Long-horizon: reasoning must be sustained over 500-2000 tokens
- Efficiency-aware: graders measure reasoning quality, not just correctness
- Failure-mode targeted: each task exposes a specific reasoning weakness

Efficiency-aware reward shaping:
    final_score = correctness * (0.6 + 0.4 * reasoning_quality)
    where reasoning_quality = coverage * 0.4 + efficiency * 0.3
                              + verification * 0.2 + (1 - filler) * 0.1

This means:
- Wrong answers always get 0 (no credit for efficient wrong answers)
- Right answers with lazy reasoning get 0.6 × correctness
- Right answers with thorough, relevant, verified reasoning get 1.0 × correctness
- The model is incentivized to reason WELL, not just get the right answer
"""
from .base import LongHorizonEnv, register_long_horizon
from .efficiency import (
    score_reasoning_quality, ReasoningBreakdown,
    EXPECTED_CONCEPTS, VERIFICATION_KEYWORDS, FILLER_PHRASES,
)

# Import all 20 tasks to trigger registration
from .tasks import (
    cascading_bug_chain, cross_module_data_flow, invariant_preservation,
    complexity_optimization, api_contract_compliance, race_condition_detection,
    recursive_repair, type_flow_inference, property_based_tests,
    design_pattern_selection, error_propagation_analysis, state_machine_impl,
    reachability_analysis, bottleneck_isolation, backward_compat_evolution,
    differential_analysis, spec_compliance_audit, minimal_change_identification,
    coverage_gap_analysis, security_audit,
)

ALL_TASKS = list(register_long_horizon._registry.values())
