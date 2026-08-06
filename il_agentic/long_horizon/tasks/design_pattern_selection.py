"""
Long-horizon task: design_pattern_selection

Reasoning skill: Pattern recognition + application.
Failure mode: Small models don't recognize when a design pattern applies
and fail to refactor away from anti-patterns.

The codebase is a `shipping_calculator.py` module that calculates shipping
costs for different carriers (UPS, FedEx, DHL) using a giant if/elif chain.
This violates the open/closed principle. The model must refactor to use
the Strategy pattern: define a ShippingStrategy interface, implement one
strategy per carrier, and use a registry/dispatch.

Grader: 0.6 * test_pass_rate + 0.4 * structural_score where structural_score
checks for strategy classes, no if/elif on carrier type, and registry pattern.
"""
import ast
import re
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer as grader_extract_answer,
    extract_reasoning as grader_extract_reasoning,
    parse_code_blocks, apply_code_changes, run_tests, run_code,
    compute_test_score, CodeExecutor, code_similarity,
)


@register_long_horizon
class DesignPatternSelection(LongHorizonEnv):
    """Refactor an if/elif chain into the Strategy pattern."""

    task_id = "design_pattern_selection"
    reasoning_skill = "Pattern recognition + application"
    failure_mode = (
        "Small models don't recognize when a design pattern applies and "
        "fail to refactor away from anti-patterns"
    )
    token_budget = 800
    expected_concepts = [
        "strategy", "pattern", "interface", "polymorphism",
        "refactor", "dispatch", "verify", "open/closed",
    ]

    # ── Codebase ──

    def gen_codebase(self) -> dict[str, str]:
        shipping = textwrap.dedent('''\
            class ShippingCalculator:
                """Calculates shipping costs for different carriers.

                Currently uses a giant if/elif chain to dispatch based on
                carrier name. This violates the open/closed principle —
                adding a new carrier requires modifying this class.
                """

                def __init__(self):
                    self.supported_carriers = ["UPS", "FedEx", "DHL"]

                def calculate_cost(self, carrier, weight, distance):
                    """Calculate shipping cost for a given carrier.

                    Args:
                        carrier: carrier name ("UPS", "FedEx", "DHL")
                        weight: package weight in kg
                        distance: shipping distance in km

                    Returns:
                        Shipping cost as a float.

                    Raises:
                        ValueError: if carrier is not supported.
                    """
                    if carrier == "UPS":
                        base_rate = 5.0
                        per_kg = 0.50
                        per_km = 0.10
                        # UPS has a discount for heavy packages
                        if weight > 10:
                            per_kg = 0.40
                        cost = base_rate + weight * per_kg + distance * per_km
                        return round(cost, 2)

                    elif carrier == "FedEx":
                        base_rate = 7.0
                        per_kg = 0.45
                        per_km = 0.08
                        # FedEx has a surcharge for long distances
                        if distance > 500:
                            base_rate += 3.0
                        cost = base_rate + weight * per_kg + distance * per_km
                        return round(cost, 2)

                    elif carrier == "DHL":
                        base_rate = 6.0
                        per_kg = 0.55
                        per_km = 0.12
                        # DHL has a flat discount for light packages
                        if weight < 2:
                            base_rate = 4.0
                        cost = base_rate + weight * per_kg + distance * per_km
                        return round(cost, 2)

                    else:
                        raise ValueError(f"Unsupported carrier: {carrier}")

                def get_supported_carriers(self):
                    """Return list of supported carrier names."""
                    return list(self.supported_carriers)

                def add_carrier(self, name, base_rate, per_kg, per_km):
                    """Add a new carrier. Currently not supported properly."""
                    raise NotImplementedError(
                        "Cannot add carrier — refactor to Strategy pattern needed"
                    )


            # Convenience function
            def calculate_shipping(carrier, weight, distance):
                """Calculate shipping cost using the default calculator."""
                calc = ShippingCalculator()
                return calc.calculate_cost(carrier, weight, distance)
        ''')
        test_file = textwrap.dedent('''\
            from shipping_calculator import ShippingCalculator, calculate_shipping


            # ── Behavior tests (must still pass after refactor) ──

            def test_ups_basic():
                calc = ShippingCalculator()
                cost = calc.calculate_cost("UPS", 5, 100)
                # base=5, 5*0.50=2.5, 100*0.10=10 -> 17.5
                assert cost == 17.5, f"Expected 17.5, got {cost}"

            def test_ups_heavy_discount():
                calc = ShippingCalculator()
                cost = calc.calculate_cost("UPS", 15, 100)
                # base=5, 15*0.40=6.0, 100*0.10=10 -> 21.0
                assert cost == 21.0, f"Expected 21.0, got {cost}"

            def test_fedex_basic():
                calc = ShippingCalculator()
                cost = calc.calculate_cost("FedEx", 5, 100)
                # base=7, 5*0.45=2.25, 100*0.08=8 -> 17.25
                assert cost == 17.25, f"Expected 17.25, got {cost}"

            def test_fedex_long_distance():
                calc = ShippingCalculator()
                cost = calc.calculate_cost("FedEx", 5, 600)
                # base=7+3=10, 5*0.45=2.25, 600*0.08=48 -> 60.25
                assert cost == 60.25, f"Expected 60.25, got {cost}"

            def test_dhl_light_package():
                calc = ShippingCalculator()
                cost = calc.calculate_cost("DHL", 1, 50)
                # base=4, 1*0.55=0.55, 50*0.12=6 -> 10.55
                assert cost == 10.55, f"Expected 10.55, got {cost}"

            def test_unsupported_carrier():
                calc = ShippingCalculator()
                try:
                    calc.calculate_cost("Unknown", 5, 100)
                    assert False, "Should have raised ValueError"
                except ValueError:
                    pass

            def test_convenience_function():
                cost = calculate_shipping("UPS", 5, 100)
                assert cost == 17.5

            def test_supported_carriers():
                calc = ShippingCalculator()
                carriers = calc.get_supported_carriers()
                assert "UPS" in carriers
                assert "FedEx" in carriers
                assert "DHL" in carriers

            # ── Structural tests (check Strategy pattern is used) ──

            def test_no_if_elif_on_carrier():
                """The calculate_cost method should not use if/elif on carrier name."""
                import shipping_calculator
                import inspect
                source = inspect.getsource(shipping_calculator)
                # Check that there's no if/elif chain checking carrier == "..."
                lines = source.split("\\n")
                carrier_checks = [l for l in lines
                                  if 'carrier ==' in l]
                # Allow at most 1 (for error handling), not a chain
                assert len(carrier_checks) <= 1, (
                    f"Found {len(carrier_checks)} carrier checks - "
                    "should use Strategy pattern dispatch instead"
                )

            def test_can_add_carrier_without_modification():
                """Adding a new carrier should not require modifying existing code."""
                calc = ShippingCalculator()
                # If the Strategy pattern is used, we should be able to
                # register a new carrier strategy
                assert hasattr(calc, 'register_strategy') or hasattr(calc, 'add_carrier'), (
                    "Calculator should support registering new strategies"
                )
                # Try adding a new carrier
                try:
                    if hasattr(calc, 'register_strategy'):
                        # Strategy pattern: register a new strategy
                        class TestCarrier:
                            def calculate(self, weight, distance):
                                return weight * 1.0 + distance * 0.5
                        calc.register_strategy("TestCo", TestCarrier())
                        cost = calc.calculate_cost("TestCo", 10, 20)
                        assert cost == 20.0, f"Expected 20.0, got {cost}"
                    elif hasattr(calc, 'add_carrier'):
                        calc.add_carrier("TestCo", base_rate=0, per_kg=1.0, per_km=0.5)
                        cost = calc.calculate_cost("TestCo", 10, 20)
                        assert cost == 20.0, f"Expected 20.0, got {cost}"
                except NotImplementedError:
                    assert False, "add_carrier should work after Strategy refactor"
        ''')
        return {
            "shipping_calculator.py": shipping,
            "test_shipping.py": test_file,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''\
            You are given a `shipping_calculator.py` module that calculates
            shipping costs for different carriers (UPS, FedEx, DHL).

            The current implementation uses a giant if/elif chain to dispatch
            based on carrier name. This violates the open/closed principle:
            adding a new carrier requires modifying the `calculate_cost`
            method, which risks breaking existing carriers.

            Your task:
            1. Recognize that the Strategy pattern is the appropriate design
               pattern for this problem.
            2. Refactor the code to use the Strategy pattern:
               - Define a `ShippingStrategy` interface (abstract base class
                 with a `calculate` method).
               - Implement one strategy class per carrier (UPSStrategy,
                 FedExStrategy, DHLStrategy).
               - Use a registry/dispatch mechanism in `ShippingCalculator`
                 that maps carrier names to strategy instances.
               - Support registering new strategies without modifying
                 existing code.
            3. Preserve all existing behavior — the behavior tests must pass.
            4. Support adding new carriers via `register_strategy` or
               `add_carrier` without modifying existing strategy classes.

            The refactored code must pass all tests in `test_shipping.py`,
            including the structural tests that verify the Strategy pattern
            is actually used (no if/elif on carrier type, new carriers can
            be added without modification).

            Provide the refactored `shipping_calculator.py` in a
            ```python:shipping_calculator.py``` code block.
        ''')

    # ── Solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        refactored = textwrap.dedent('''\
            from abc import ABC, abstractmethod


            class ShippingStrategy(ABC):
                """Abstract interface for shipping cost calculation strategies."""

                @abstractmethod
                def calculate(self, weight, distance):
                    """Calculate shipping cost given weight and distance.

                    Args:
                        weight: package weight in kg
                        distance: shipping distance in km

                    Returns:
                        Shipping cost as a float.
                    """
                    ...


            class UPSStrategy(ShippingStrategy):
                """UPS shipping strategy with heavy package discount."""

                def calculate(self, weight, distance):
                    base_rate = 5.0
                    per_kg = 0.50
                    per_km = 0.10
                    if weight > 10:
                        per_kg = 0.40
                    cost = base_rate + weight * per_kg + distance * per_km
                    return round(cost, 2)


            class FedExStrategy(ShippingStrategy):
                """FedEx shipping strategy with long-distance surcharge."""

                def calculate(self, weight, distance):
                    base_rate = 7.0
                    per_kg = 0.45
                    per_km = 0.08
                    if distance > 500:
                        base_rate += 3.0
                    cost = base_rate + weight * per_kg + distance * per_km
                    return round(cost, 2)


            class DHLStrategy(ShippingStrategy):
                """DHL shipping strategy with light package discount."""

                def calculate(self, weight, distance):
                    base_rate = 6.0
                    per_kg = 0.55
                    per_km = 0.12
                    if weight < 2:
                        base_rate = 4.0
                    cost = base_rate + weight * per_kg + distance * per_km
                    return round(cost, 2)


            class ShippingCalculator:
                """Calculates shipping costs using the Strategy pattern.

                Uses a registry to map carrier names to strategy instances.
                New carriers can be added via register_strategy() without
                modifying existing code — following the open/closed principle.
                """

                def __init__(self):
                    self._strategies = {}
                    # Register built-in strategies
                    self.register_strategy("UPS", UPSStrategy())
                    self.register_strategy("FedEx", FedExStrategy())
                    self.register_strategy("DHL", DHLStrategy())

                def register_strategy(self, name, strategy):
                    """Register a new shipping strategy.

                    Args:
                        name: carrier name
                        strategy: ShippingStrategy instance
                    """
                    self._strategies[name] = strategy

                def add_carrier(self, name, base_rate, per_kg, per_km):
                    """Add a simple carrier with flat rates.

                    Creates a generic strategy from the given parameters.
                    """
                    class GenericStrategy(ShippingStrategy):
                        def calculate(self, weight, distance):
                            cost = base_rate + weight * per_kg + distance * per_km
                            return round(cost, 2)
                    self.register_strategy(name, GenericStrategy())

                def calculate_cost(self, carrier, weight, distance):
                    """Calculate shipping cost for a given carrier.

                    Dispatches to the registered strategy via polymorphism.
                    """
                    strategy = self._strategies.get(carrier)
                    if strategy is None:
                        raise ValueError(f"Unsupported carrier: {carrier}")
                    return strategy.calculate(weight, distance)

                def get_supported_carriers(self):
                    """Return list of supported carrier names."""
                    return list(self._strategies.keys())


            def calculate_shipping(carrier, weight, distance):
                """Calculate shipping cost using the default calculator."""
                calc = ShippingCalculator()
                return calc.calculate_cost(carrier, weight, distance)
        ''')
        return {"shipping_calculator.py": refactored}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''\
            Let me analyze the `shipping_calculator.py` module to identify
            the design problem and determine the appropriate pattern.

            ── Problem Analysis ──

            The `calculate_cost` method uses a giant if/elif chain:
                if carrier == "UPS": ...
                elif carrier == "FedEx": ...
                elif carrier == "DHL": ...
                else: raise ValueError

            Each branch contains carrier-specific logic (base rates, per-kg
            charges, per-km charges, and special conditions like discounts
            and surcharges). This has several problems:

            1. **Violates open/closed principle**: Adding a new carrier
               (e.g., "USPS") requires modifying the `calculate_cost`
               method itself. The class is not open for extension without
               modification.

            2. **Hard to maintain**: Each carrier's logic is embedded in
               a single method. Changing UPS's pricing requires editing
               a method that also contains FedEx and DHL logic.

            3. **No polymorphism**: The dispatch is based on string
               comparison, not on object behavior. There's no interface
               that strategies can implement.

            ── Pattern Recognition ──

            This is a classic case for the **Strategy pattern**. The
            pattern applies when:
            - You have multiple algorithms for the same task (calculating
              shipping cost for different carriers).
            - You need to switch between algorithms at runtime (based on
              carrier name).
            - You want to add new algorithms without modifying existing
              code (open/closed principle).

            The Strategy pattern solution:
            1. Define an interface (`ShippingStrategy`) with a `calculate`
               method.
            2. Implement one concrete strategy per carrier (UPSStrategy,
               FedExStrategy, DHLStrategy).
            3. Use a registry in `ShippingCalculator` that maps carrier
               names to strategy instances.
            4. `calculate_cost` dispatches to the registered strategy via
               polymorphism — it calls `strategy.calculate(weight, distance)`
               without knowing which carrier it is.

            ── Implementation ──

            I'll use `abc.ABC` and `@abstractmethod` to define the
            `ShippingStrategy` interface. Each carrier gets its own class
            implementing `calculate(self, weight, distance)`.

            The `ShippingCalculator` maintains a `_strategies` dict. In
            `__init__`, it registers the three built-in strategies. The
            `register_strategy` method allows adding new strategies at
            runtime — this is the key to the open/closed principle.

            The `calculate_cost` method simply looks up the strategy and
            calls `strategy.calculate(weight, distance)`. This is
            polymorphic dispatch — the calculator doesn't need to know
            the carrier type, it just calls the interface method.

            I'll also keep `add_carrier` as a convenience method that
            creates a generic strategy from flat rate parameters, and
            `calculate_shipping` as the convenience function.

            ── Verification ──

            Let me verify behavior is preserved by tracing the tests:

            test_ups_basic: UPS, weight=5, distance=100
            - UPSStrategy.calculate(5, 100): base=5, per_kg=0.50, per_km=0.10
            - weight(5) <= 10, no discount
            - cost = 5 + 5*0.50 + 100*0.10 = 5 + 2.5 + 10 = 17.5

            test_ups_heavy_discount: UPS, weight=15, distance=100
            - weight(15) > 10, per_kg=0.40
            - cost = 5 + 15*0.40 + 100*0.10 = 5 + 6 + 10 = 21.0

            test_fedex_long_distance: FedEx, weight=5, distance=600
            - distance(600) > 500, base_rate=7+3=10
            - cost = 10 + 5*0.45 + 600*0.08 = 10 + 2.25 + 48 = 60.25

            test_dhl_light_package: DHL, weight=1, distance=50
            - weight(1) < 2, base_rate=4
            - cost = 4 + 1*0.55 + 50*0.12 = 4 + 0.55 + 6 = 10.55

            test_unsupported_carrier: "Unknown" not in registry
            - strategy = None, raises ValueError

            test_no_if_elif_on_carrier: The refactored calculate_cost
            has no `if carrier ==` checks (only `if strategy is None`
            for error handling).

            test_can_add_carrier_without_modification:
            - register_strategy("TestCo", TestCarrier())
            - calculate_cost("TestCo", 10, 20) -> 10*1.0 + 20*0.5 = 20.0

            All behavior preserved, structural tests pass. The refactor
            follows the open/closed principle: new carriers can be added
            by registering new strategies without modifying existing
            strategy classes or the calculator's dispatch logic. The
            polymorphism of the Strategy interface enables clean dispatch.

            Let me verify the test_no_if_elif_on_carrier test more carefully.
            The refactored source has `if strategy is None` which does NOT
            contain `carrier ==`, so the carrier_checks filter finds 0
            matches. The test asserts len(carrier_checks) <= 1, which
            passes with 0.

            To confirm: the Strategy pattern refactor uses polymorphism
            for dispatch, a registry for carrier lookup, and an interface
            for the strategy contract. This satisfies the open/closed
            principle — the system is open for extension (new strategies
            can be registered) but closed for modification (existing
            strategy classes don't need to change).
        ''')

    # ── Grading ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        answer = grader_extract_answer(response)
        blocks = parse_code_blocks(answer)
        if not blocks:
            blocks = parse_code_blocks(response)

        if "shipping_calculator.py" not in blocks:
            return 0.0, {
                "reason": "no shipping_calculator.py code block found",
                "test_details": {},
            }

        fixed_codebase = apply_code_changes(codebase, blocks)
        test_code = codebase.get("test_shipping.py", "")
        results = run_tests(fixed_codebase, test_code, timeout=15.0)
        test_score, test_details = compute_test_score(results)

        # ── Structural score (40%) ──
        structural_score = 0.0
        structural_checks = {}
        try:
            source = blocks["shipping_calculator.py"]
            tree = ast.parse(source)

            # Check 1: Has strategy classes (classes ending in "Strategy" or
            # inheriting from an ABC/interface)
            class_names = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            ]
            has_strategy_classes = any(
                "Strategy" in name for name in class_names
            )
            structural_checks["has_strategy_classes"] = has_strategy_classes

            # Check 2: No if/elif chain on carrier type
            source_lines = source.split("\n")
            carrier_checks = [
                line for line in source_lines
                if 'carrier ==' in line
            ]
            no_if_elif_chain = len(carrier_checks) <= 1
            structural_checks["no_if_elif_chain"] = no_if_elif_chain

            # Check 3: Has registry/dispatch pattern (dict mapping names to strategies)
            has_registry = (
                "_strategies" in source or
                "registry" in source.lower() or
                "register" in source.lower()
            )
            structural_checks["has_registry"] = has_registry

            # Check 4: Has abstract base class / interface
            has_interface = (
                "ABC" in source or
                "abstractmethod" in source or
                "abstract" in source.lower()
            )
            structural_checks["has_interface"] = has_interface

            # Check 5: Has register_strategy or add_carrier method
            has_register = (
                "register_strategy" in source or
                "add_carrier" in source
            )
            structural_checks["has_register_method"] = has_register

            checks_passed = sum([
                has_strategy_classes, no_if_elif_chain, has_registry,
                has_interface, has_register,
            ])
            structural_score = checks_passed / 5.0
        except SyntaxError:
            structural_checks = {"error": "syntax error in code"}

        score = 0.6 * test_score + 0.4 * structural_score
        breakdown = {
            "test_score": test_score,
            "structural_score": structural_score,
            "structural_checks": structural_checks,
            "test_details": test_details,
        }
        return score, breakdown
