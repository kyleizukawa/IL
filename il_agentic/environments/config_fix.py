"""
Environment 10: Configuration Fix

Skill: Fixing broken configuration files.

The model is given a Python config file (using dictionaries/dataclasses) with
errors (wrong keys, invalid values, missing required fields, type mismatches)
and must fix it. The grader runs a validation function against the model's
fixed config and returns the fraction of checks that pass.

Difficulty scaling:
- easy: 1-2 errors, obvious (wrong value, missing required key)
- medium: 3-4 errors, subtle (type mismatch, wrong nested key)
- hard: 5+ errors across multiple config sections
"""
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, parse_code_blocks, apply_code_changes,
    extract_reasoning, run_code,
)


# ── Domain templates ──
# Each domain has: broken config, correct config, validation function

DOMAINS = {
    "database": {
        "config_var": "DATABASE_CONFIG",
        "broken": textwrap.dedent('''
            DATABASE_CONFIG = {
                "host": "localhost",
                "port": "5432",
                "database": "myapp",
                "user": "admin",
                "password": "",
                "pool_size": -1,
                "timeout": 30,
                "ssl_mode": "maybe",
                "max_connections": 100,
            }
        ''').strip(),
        "correct": textwrap.dedent('''
            DATABASE_CONFIG = {
                "host": "localhost",
                "port": 5432,
                "database": "myapp",
                "user": "admin",
                "password": "secret",
                "pool_size": 10,
                "timeout": 30,
                "ssl_mode": "require",
                "max_connections": 100,
            }
        ''').strip(),
        "validation": textwrap.dedent('''
            def validate_config(config):
                checks = []

                # Check required keys exist
                required = ["host", "port", "database", "user", "password", "pool_size", "timeout", "ssl_mode"]
                for key in required:
                    checks.append(("has_" + key, key in config))

                # Check port is int and in valid range
                checks.append(("port_is_int", isinstance(config.get("port"), int)))
                checks.append(("port_range", 1 <= config.get("port", 0) <= 65535))

                # Check password is non-empty string
                checks.append(("password_nonempty", isinstance(config.get("password"), str) and len(config.get("password", "")) > 0))

                # Check pool_size is positive int
                checks.append(("pool_size_positive", isinstance(config.get("pool_size"), int) and config.get("pool_size", 0) > 0))

                # Check ssl_mode is valid
                valid_ssl = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
                checks.append(("ssl_mode_valid", config.get("ssl_mode") in valid_ssl))

                # Check timeout is positive
                checks.append(("timeout_positive", isinstance(config.get("timeout"), (int, float)) and config.get("timeout", 0) > 0))

                return checks
        ''').strip(),
        "errors": [
            "port is a string '5432' instead of int 5432",
            "password is empty string instead of non-empty",
            "pool_size is -1 instead of a positive integer",
            "ssl_mode is 'maybe' instead of a valid mode like 'require'",
        ],
    },
    "web_server": {
        "config_var": "SERVER_CONFIG",
        "broken": textwrap.dedent('''
            SERVER_CONFIG = {
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 0,
                "debug": "true",
                "cors_origins": ["http://localhost:3000"],
                "rate_limit": {
                    "enabled": True,
                    "requests_per_minute": -100,
                    "burst_size": 50,
                },
                "static_files": {
                    "enabled": True,
                    "dir": "/var/www/static",
                    "max_age": "3600",
                },
            }
        ''').strip(),
        "correct": textwrap.dedent('''
            SERVER_CONFIG = {
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 4,
                "debug": True,
                "cors_origins": ["http://localhost:3000"],
                "rate_limit": {
                    "enabled": True,
                    "requests_per_minute": 100,
                    "burst_size": 50,
                },
                "static_files": {
                    "enabled": True,
                    "dir": "/var/www/static",
                    "max_age": 3600,
                },
            }
        ''').strip(),
        "validation": textwrap.dedent('''
            def validate_config(config):
                checks = []

                # Top-level checks
                checks.append(("has_host", "host" in config))
                checks.append(("port_int", isinstance(config.get("port"), int)))
                checks.append(("port_range", 1 <= config.get("port", 0) <= 65535))

                # workers must be positive int
                checks.append(("workers_positive", isinstance(config.get("workers"), int) and config.get("workers", 0) > 0))

                # debug must be bool
                checks.append(("debug_is_bool", isinstance(config.get("debug"), bool)))

                # cors_origins must be a list
                checks.append(("cors_is_list", isinstance(config.get("cors_origins"), list)))

                # rate_limit nested config
                rl = config.get("rate_limit", {})
                checks.append(("has_rate_limit", isinstance(rl, dict)))
                checks.append(("rl_enabled_bool", isinstance(rl.get("enabled"), bool)))
                checks.append(("rl_rpm_positive", isinstance(rl.get("requests_per_minute"), int) and rl.get("requests_per_minute", 0) > 0))
                checks.append(("rl_burst_positive", isinstance(rl.get("burst_size"), int) and rl.get("burst_size", 0) > 0))

                # static_files nested config
                sf = config.get("static_files", {})
                checks.append(("has_static_files", isinstance(sf, dict)))
                checks.append(("sf_enabled_bool", isinstance(sf.get("enabled"), bool)))
                checks.append(("sf_dir_is_str", isinstance(sf.get("dir"), str) and len(sf.get("dir", "")) > 0))
                checks.append(("sf_max_age_int", isinstance(sf.get("max_age"), int) and sf.get("max_age", -1) >= 0))

                return checks
        ''').strip(),
        "errors": [
            "workers is 0 instead of a positive integer",
            "debug is string 'true' instead of boolean True",
            "rate_limit.requests_per_minute is -100 instead of positive",
            "static_files.max_age is string '3600' instead of int 3600",
        ],
    },
    "logging": {
        "config_var": "LOGGING_CONFIG",
        "broken": textwrap.dedent('''
            LOGGING_CONFIG = {
                "level": "VERBOSE",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": {
                    "console": {
                        "class": "StreamHandler",
                        "level": "INFO",
                        "stream": "ext://sys.stdout",
                    },
                    "file": {
                        "class": "FileHandler",
                        "level": "DEBUG",
                        "filename": "",
                        "encoding": 123,
                        "max_bytes": "10485760",
                        "backup_count": 5,
                    },
                },
                "root": {
                    "level": "WARNING",
                    "handlers": ["console"],
                },
            }
        ''').strip(),
        "correct": textwrap.dedent('''
            LOGGING_CONFIG = {
                "level": "DEBUG",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": {
                    "console": {
                        "class": "StreamHandler",
                        "level": "INFO",
                        "stream": "ext://sys.stdout",
                    },
                    "file": {
                        "class": "RotatingFileHandler",
                        "level": "DEBUG",
                        "filename": "app.log",
                        "encoding": "utf-8",
                        "max_bytes": 10485760,
                        "backup_count": 5,
                    },
                },
                "root": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                },
            }
        ''').strip(),
        "validation": textwrap.dedent('''
            def validate_config(config):
                checks = []

                valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

                # Top-level level must be valid
                checks.append(("level_valid", config.get("level") in valid_levels))

                # format must be non-empty string
                checks.append(("format_str", isinstance(config.get("format"), str) and len(config.get("format", "")) > 0))

                # handlers must be a dict
                handlers = config.get("handlers", {})
                checks.append(("handlers_is_dict", isinstance(handlers, dict)))
                checks.append(("has_console_handler", "console" in handlers))
                checks.append(("has_file_handler", "file" in handlers))

                # console handler
                ch = handlers.get("console", {})
                checks.append(("console_level_valid", ch.get("level") in valid_levels))
                checks.append(("console_class", isinstance(ch.get("class"), str) and len(ch.get("class", "")) > 0))

                # file handler
                fh = handlers.get("file", {})
                checks.append(("file_level_valid", fh.get("level") in valid_levels))
                checks.append(("file_class", isinstance(fh.get("class"), str) and len(fh.get("class", "")) > 0))
                checks.append(("file_filename_nonempty", isinstance(fh.get("filename"), str) and len(fh.get("filename", "")) > 0))
                checks.append(("file_encoding_str", isinstance(fh.get("encoding"), str)))
                checks.append(("file_max_bytes_int", isinstance(fh.get("max_bytes"), int) and fh.get("max_bytes", 0) > 0))
                checks.append(("file_backup_int", isinstance(fh.get("backup_count"), int) and fh.get("backup_count", 0) >= 0))

                # root config
                root = config.get("root", {})
                checks.append(("root_level_valid", root.get("level") in valid_levels))
                checks.append(("root_handlers_list", isinstance(root.get("handlers"), list)))
                checks.append(("root_has_handlers", len(root.get("handlers", [])) > 0))

                return checks
        ''').strip(),
        "errors": [
            "level is 'VERBOSE' instead of a valid level like 'DEBUG'",
            "file.filename is empty string instead of a real filename",
            "file.encoding is int 123 instead of string 'utf-8'",
            "file.max_bytes is string '10485760' instead of int",
            "root.handlers only has ['console'] but should include 'file' too",
        ],
    },
    "feature_flags": {
        "config_var": "FEATURE_FLAGS",
        "broken": textwrap.dedent('''
            FEATURE_FLAGS = {
                "new_ui": "yes",
                "beta_features": True,
                "max_retries": 3.5,
                "experiment_bucket": "control",
                "rollout_percentage": 150,
                "disabled_features": ["old_dashboard", "legacy_api"],
                "legacy_api": True,
                "env_overrides": {
                    "production": {
                        "new_ui": False,
                        "beta_features": True,
                    },
                    "staging": None,
                },
            }
        ''').strip(),
        "correct": textwrap.dedent('''
            FEATURE_FLAGS = {
                "new_ui": True,
                "beta_features": False,
                "max_retries": 3,
                "experiment_bucket": "treatment",
                "rollout_percentage": 50,
                "disabled_features": ["old_dashboard"],
                "legacy_api": False,
                "env_overrides": {
                    "production": {
                        "new_ui": False,
                        "beta_features": False,
                    },
                    "staging": {
                        "new_ui": True,
                        "beta_features": True,
                    },
                },
            }
        ''').strip(),
        "validation": textwrap.dedent('''
            def validate_config(config):
                checks = []

                # Boolean flags must be actual bools
                checks.append(("new_ui_is_bool", isinstance(config.get("new_ui"), bool)))
                checks.append(("beta_features_is_bool", isinstance(config.get("beta_features"), bool)))
                checks.append(("legacy_api_is_bool", isinstance(config.get("legacy_api"), bool)))

                # legacy_api should be False (it's in disabled_features)
                checks.append(("legacy_api_disabled", config.get("legacy_api") == False))

                # max_retries must be int
                checks.append(("max_retries_int", isinstance(config.get("max_retries"), int)))

                # experiment_bucket must be valid
                valid_buckets = ["control", "treatment", "holdout"]
                checks.append(("bucket_valid", config.get("experiment_bucket") in valid_buckets))

                # rollout_percentage must be 0-100
                rp = config.get("rollout_percentage", -1)
                checks.append(("rollout_int", isinstance(rp, int)))
                checks.append(("rollout_range", 0 <= rp <= 100))

                # disabled_features must be a list
                checks.append(("disabled_is_list", isinstance(config.get("disabled_features"), list)))

                # env_overrides must be a dict with valid env keys
                eo = config.get("env_overrides", {})
                checks.append(("env_overrides_is_dict", isinstance(eo, dict)))
                checks.append(("has_production", "production" in eo and isinstance(eo.get("production"), dict)))
                checks.append(("has_staging", "staging" in eo and isinstance(eo.get("staging"), dict)))
                checks.append(("staging_not_none", eo.get("staging") is not None))

                # production overrides should have bool values
                prod = eo.get("production", {})
                if isinstance(prod, dict):
                    checks.append(("prod_new_ui_bool", isinstance(prod.get("new_ui"), bool)))
                    checks.append(("prod_beta_bool", isinstance(prod.get("beta_features"), bool)))
                else:
                    checks.append(("prod_new_ui_bool", False))
                    checks.append(("prod_beta_bool", False))

                return checks
        ''').strip(),
        "errors": [
            "new_ui is 'yes' (string) instead of boolean True",
            "beta_features is True but should be False",
            "max_retries is 3.5 (float) instead of int 3",
            "rollout_percentage is 150 instead of 0-100 range",
            "legacy_api is True but should be False (it's in disabled_features)",
            "env_overrides.staging is None instead of a dict",
        ],
    },
}


# ── Distractor code ──

DISTRACTORS = [
    textwrap.dedent('''
        def format_duration(seconds):
            units = [("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]
            parts = []
            for unit, size in units:
                if seconds >= size:
                    count = int(seconds // size)
                    parts.append(f"{count}{unit}")
                    seconds %= size
            return " ".join(parts) if parts else "0s"
    ''').strip(),
    textwrap.dedent('''
        def deep_merge(a, b):
            result = dict(a)
            for key, value in b.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
    ''').strip(),
    textwrap.dedent('''
        def interpolate(values, target_idx):
            if not values:
                return 0
            if target_idx < 0:
                return values[0]
            if target_idx >= len(values) - 1:
                return values[-1]
            lower = int(target_idx)
            frac = target_idx - lower
            return values[lower] * (1 - frac) + values[lower + 1] * frac
    ''').strip(),
]


@register_env
class ConfigFixEnv(AgenticEnv):
    name = "config_fix"
    skill = "Fixing broken configuration files"
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
        codebase = {f"{main_module}.py": domain["broken"]}

        # Add validation as a separate file
        codebase["validate.py"] = (
            f"from {main_module} import *\n"
            f"{domain['validation']}\n\n"
            f"if __name__ == \"__main__\":\n"
            f"    config = {domain['config_var']}\n"
            f"    checks = validate_config(config)\n"
            f"    passed = sum(1 for _, ok in checks if ok)\n"
            f"    total = len(checks)\n"
            f"    for name, ok in checks:\n"
            f"        status = \"PASS\" if ok else \"FAIL\"\n"
            f"        print(f\"{{status}}: {{name}}\")\n"
            f"    print(f\"\\n{{passed}}/{{total}} checks passed\")\n"
        )

        for idx in params.get("distractor_indices", []):
            codebase[f"helper_{idx}.py"] = DISTRACTORS[idx]

        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]

        lines = []
        lines.append("You are a software engineer fixing a broken configuration file.")
        lines.append("")
        lines.append("The configuration file has several errors:")
        lines.append("- Wrong values (incorrect data)")
        lines.append("- Type mismatches (string instead of int, etc.)")
        lines.append("- Invalid enum values")
        lines.append("- Missing required fields")
        lines.append("")
        lines.append("A validation function is provided in `validate.py` that checks the config.")
        lines.append("Your task is to:")
        lines.append("1. Read the validation function to understand what the config should look like")
        lines.append("2. Identify all errors in the config file")
        lines.append("3. Write the corrected config file")
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")

        lines.append("Provide your fixed config in the following format:")
        lines.append("<reasoning>")
        lines.append("...read the validation function, identify each error, explain the fix...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append(f"```python:{main_module}.py")
        lines.append("# the corrected config")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]
        return {f"{main_module}.py": domain["correct"]}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        main_module = params["domain"]
        errors = domain["errors"]

        lines = []
        lines.append(f"I need to fix the broken configuration in {main_module}.py.")
        lines.append("Let me first read the validation function to understand what the correct config should look like.")
        lines.append("")
        lines.append("Reading validate.py carefully:")
        lines.append("- The validation function checks each field for type, value range, and validity.")
        lines.append("- I need to match each check against the current config values.")
        lines.append("")
        lines.append("Now let me compare the current config against the validation checks:")
        lines.append("")

        for i, error in enumerate(errors, 1):
            lines.append(f"Error {i}: {error}")
            lines.append(f"  - I can see this will fail the corresponding validation check.")
            lines.append(f"  - The fix is to correct the value to match what the validator expects.")
            lines.append("")

        lines.append(f"I've identified {len(errors)} errors total. Let me write the corrected config file")
        lines.append("with all fixes applied.")

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

        target_file = f"{main_module}.py"
        if target_file not in code_changes:
            # Try to find any .py file
            for fname in code_changes:
                if main_module in fname:
                    target_file = fname
                    break

        if target_file not in code_changes:
            return 0.0, {
                "reason": "target config file not found in response",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        # Build the validation script
        modified = apply_code_changes(codebase, code_changes)

        # Run validation
        validation_script = (
            f"import json\n"
            f"from {main_module} import *\n"
            f"{domain['validation']}\n\n"
            f"# Find the config variable\n"
            f"config = None\n"
            f"import {main_module} as _mod\n"
            f"for attr in dir(_mod):\n"
            f"    val = getattr(_mod, attr)\n"
            f"    if isinstance(val, dict) and not attr.startswith('_'):\n"
            f"        config = val\n"
            f"        break\n\n"
            f"if config is None:\n"
            f"    print(json.dumps({{\"error\": \"no config dict found\"}}))\n"
            f"else:\n"
            f"    checks = validate_config(config)\n"
            f"    results = []\n"
            f"    for name, ok in checks:\n"
            f"        results.append({{\"name\": name, \"passed\": ok}})\n"
            f"    passed = sum(1 for _, ok in checks if ok)\n"
            f"    total = len(checks)\n"
            f"    print(json.dumps({{\"results\": results, \"passed\": passed, \"total\": total}}))\n"
        )

        run_result = run_code(validation_script, codebase=modified, timeout=10.0)

        if run_result["returncode"] != 0:
            return 0.0, {
                "reason": f"validation script failed: {run_result.get('stderr', '')[:200]}",
                "has_reasoning": bool(extract_reasoning(response)),
                "files_changed": list(code_changes.keys()),
            }

        import json
        try:
            # Find JSON in output
            output = run_result["stdout"].strip()
            json_line = None
            for line in output.split("\n"):
                line = line.strip()
                if line.startswith("{") and "passed" in line:
                    json_line = line
                    break
            if json_line:
                data = json.loads(json_line)
            else:
                return 0.0, {
                    "reason": "no JSON output from validation",
                    "has_reasoning": bool(extract_reasoning(response)),
                }
        except (json.JSONDecodeError, KeyError):
            return 0.0, {
                "reason": "failed to parse validation output",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        passed = data.get("passed", 0)
        total = data.get("total", 0)
        score = passed / total if total > 0 else 0.0

        breakdown = {
            "passed": passed,
            "total": total,
            "score": score,
            "check_results": data.get("results", []),
            "has_reasoning": bool(extract_reasoning(response)),
            "files_changed": list(code_changes.keys()),
            "changed_target": target_file in code_changes,
        }

        return score, breakdown
