"""
IL Agentic — 15 RL environments for agentic coding, codebase reasoning,
and killing laziness in small models.

Built in the style of mechanize.work: each environment is a self-contained
software engineering task with a deterministic grader that provides
informative reward signals for reinforcement learning.

Environment interface (see base.py):
  - gen_params(rng, difficulty)    → task parameters
  - gen_codebase(params, rng)      → {filename: content} mini-codebase
  - gen_task(params, codebase)     → task description string
  - gen_solution(params, codebase) → correct solution {filename: content}
  - gen_reasoning(params, codebase, solution) → teacher reasoning trace
  - grade(params, codebase, response) → (score, breakdown)

Model response format:
  <reasoning>
  ...analysis of the codebase, tracing code, identifying issues...
  </reasoning>
  <answer>
  ```python:filename.py
  ...the code...
  ```
  </answer>

For Q&A tasks (codebase_nav, code_review), the answer is plain text.
"""
from .base import AgenticEnv, register_env, ENV_REGISTRY
from .graders import (
    run_tests, run_code, CodeExecutor, parse_code_blocks,
    apply_code_changes, extract_answer, extract_reasoning
)

# Import all environments to trigger registration
from .environments import (
    bug_localization, feature_impl, refactor_preserve, test_writing,
    api_client, perf_optimize, codebase_nav, type_annotate,
    doc_gen, config_fix, error_handling, data_transform,
    algorithm_impl, code_review, stacktrace_debug,
)

ALL_ENVIRONMENTS = list(ENV_REGISTRY.values())
