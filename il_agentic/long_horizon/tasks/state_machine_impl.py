"""
Task: state_machine_impl

Reasoning skill: State reasoning — reasoning about transitions and guards.

The model is given a formal specification for an order-processing finite state
machine and a skeleton implementation.  It must implement all valid transitions
and the guards that reject illegal ones.

States: NEW -> PENDING -> PAID -> SHIPPED -> DELIVERED, plus CANCELLED
(reachable from NEW, PENDING, PAID).  Guards: can only pay if the order has
items, can only ship if paid and address verified, can cancel only before
shipping.

Failure mode: small models miss transition guards or allow illegal states
(e.g. shipping an unpaid order, cancelling a shipped order).
"""
import textwrap

from ..base import LongHorizonEnv, register_long_horizon
from ...graders import (
    extract_answer, extract_reasoning, parse_code_blocks,
    apply_code_changes, run_tests, compute_test_score,
)


@register_long_horizon
class StateMachineImpl(LongHorizonEnv):
    task_id = "state_machine_impl"
    reasoning_skill = "State reasoning — reasoning about transitions and guards"
    failure_mode = (
        "Small models miss transition guards or allow illegal states, "
        "e.g. shipping an unpaid order or cancelling a shipped order."
    )
    token_budget = 800
    expected_concepts = [
        "state", "transition", "guard", "initial",
        "terminal", "illegal", "verify", "finite",
    ]

    # ── Codebase generation ──

    def gen_codebase(self) -> dict[str, str]:
        spec = textwrap.dedent('''
            # Order Processing State Machine — Specification

            ## States
            - NEW        : initial state, order just created
            - PENDING    : order has items and is awaiting payment
            - PAID       : payment received
            - SHIPPED    : order handed to carrier
            - DELIVERED  : order received by customer (terminal)
            - CANCELLED  : order cancelled (terminal)

            ## Valid transitions
            NEW      -> PENDING     (add items)
            PENDING  -> PAID        (pay)
            PAID     -> SHIPPED     (ship)
            SHIPPED  -> DELIVERED   (deliver)
            NEW      -> CANCELLED   (cancel)
            PENDING  -> CANCELLED   (cancel)
            PAID     -> CANCELLED   (cancel)

            ## Guards
            - NEW -> PENDING:       order must have at least one item
            - PENDING -> PAID:      order must have at least one item
            - PAID -> SHIPPED:      address must be verified
            - * -> CANCELLED:       current state must be NEW, PENDING, or PAID
                                    (cannot cancel after shipping)

            ## Terminal states
            - DELIVERED : no outgoing transitions
            - CANCELLED : no outgoing transitions
        ''').strip()

        skeleton = textwrap.dedent('''
            """Order processing state machine — implementation skeleton.

            Fill in the transition methods and guards according to the spec.
            Every method must raise IllegalTransitionError when called from a
            state that does not permit the transition or when a guard fails.
            """

            class IllegalTransitionError(Exception):
                """Raised when a transition is not allowed from the current state."""
                pass


            class Order:
                STATES = ("NEW", "PENDING", "PAID", "SHIPPED", "DELIVERED", "CANCELLED")
                TERMINAL = ("DELIVERED", "CANCELLED")

                def __init__(self, order_id: str):
                    self.order_id = order_id
                    self.state = "NEW"
                    self.items: list[str] = []
                    self.address_verified = False

                # ── Helpers ──

                def _require_state(self, *allowed):
                    if self.state not in allowed:
                        raise IllegalTransitionError(
                            f"cannot transition from {self.state}"
                        )

                # ── Transitions (implement these) ──

                def add_item(self, item: str) -> None:
                    """Add an item and move NEW -> PENDING."""
                    # TODO: implement
                    raise NotImplementedError

                def pay(self) -> None:
                    """Pay for the order: PENDING -> PAID."""
                    # TODO: implement
                    raise NotImplementedError

                def verify_address(self) -> None:
                    """Mark the address as verified (no state change)."""
                    # TODO: implement
                    raise NotImplementedError

                def ship(self) -> None:
                    """Ship the order: PAID -> SHIPPED.  Requires verified address."""
                    # TODO: implement
                    raise NotImplementedError

                def deliver(self) -> None:
                    """Mark delivered: SHIPPED -> DELIVERED."""
                    # TODO: implement
                    raise NotImplementedError

                def cancel(self) -> None:
                    """Cancel the order: NEW/PENDING/PAID -> CANCELLED."""
                    # TODO: implement
                    raise NotImplementedError
        ''').strip()

        tests = textwrap.dedent('''
            from order_machine import Order, IllegalTransitionError


            # ── Valid transition tests ──

            def test_new_to_pending():
                o = Order("1")
                o.add_item("widget")
                assert o.state == "PENDING"
                assert "widget" in o.items

            def test_pending_to_paid():
                o = Order("2")
                o.add_item("widget")
                o.pay()
                assert o.state == "PAID"

            def test_paid_to_shipped():
                o = Order("3")
                o.add_item("widget")
                o.pay()
                o.verify_address()
                o.ship()
                assert o.state == "SHIPPED"

            def test_shipped_to_delivered():
                o = Order("4")
                o.add_item("widget")
                o.pay()
                o.verify_address()
                o.ship()
                o.deliver()
                assert o.state == "DELIVERED"

            def test_cancel_from_new():
                o = Order("5")
                o.cancel()
                assert o.state == "CANCELLED"

            def test_cancel_from_pending():
                o = Order("6")
                o.add_item("widget")
                o.cancel()
                assert o.state == "CANCELLED"

            # ── Invalid transition tests (guards must reject) ──

            def test_pay_without_items_rejected():
                o = Order("7")
                # Cannot go NEW -> PAID directly; must add items first.
                try:
                    o.pay()
                    assert False, "should have raised"
                except IllegalTransitionError:
                    pass
                assert o.state == "NEW"

            def test_ship_without_address_rejected():
                o = Order("8")
                o.add_item("widget")
                o.pay()
                try:
                    o.ship()
                    assert False, "should have raised"
                except IllegalTransitionError:
                    pass
                assert o.state == "PAID"

            def test_cancel_after_ship_rejected():
                o = Order("9")
                o.add_item("widget")
                o.pay()
                o.verify_address()
                o.ship()
                try:
                    o.cancel()
                    assert False, "should have raised"
                except IllegalTransitionError:
                    pass
                assert o.state == "SHIPPED"

            def test_deliver_from_paid_rejected():
                o = Order("10")
                o.add_item("widget")
                o.pay()
                try:
                    o.deliver()
                    assert False, "should have raised"
                except IllegalTransitionError:
                    pass
                assert o.state == "PAID"

            def test_ship_from_new_rejected():
                o = Order("11")
                try:
                    o.ship()
                    assert False, "should have raised"
                except IllegalTransitionError:
                    pass
                assert o.state == "NEW"

            def test_delivered_is_terminal():
                o = Order("12")
                o.add_item("widget")
                o.pay()
                o.verify_address()
                o.ship()
                o.deliver()
                try:
                    o.cancel()
                    assert False, "should have raised"
                except IllegalTransitionError:
                    pass
                assert o.state == "DELIVERED"
        ''').strip()

        return {
            "spec.md": spec,
            "order_machine.py": skeleton,
            "test_state_machine.py": tests,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''
            You are given a formal specification (`spec.md`) for an order
            processing finite state machine and a skeleton implementation in
            `order_machine.py`.

            Implement every transition method in the skeleton so that:

            - Valid transitions update the state correctly.
            - Guards reject illegal transitions by raising
              `IllegalTransitionError`.
            - Terminal states (DELIVERED, CANCELLED) reject all further
              transitions.

            States: NEW -> PENDING -> PAID -> SHIPPED -> DELIVERED, plus
            CANCELLED (reachable from NEW, PENDING, PAID).

            Guards:
            - add_item: NEW -> PENDING (order must end up with >= 1 item)
            - pay: PENDING -> PAID (order must have items)
            - ship: PAID -> SHIPPED (address must be verified)
            - deliver: SHIPPED -> DELIVERED
            - cancel: only from NEW, PENDING, or PAID

            All 12 tests in `test_state_machine.py` must pass — 6 valid
            transitions and 6 invalid transitions that must be rejected.

            Return your solution as a code block tagged with the filename:

            ```python:order_machine.py
            ...
            ```
        ''').strip()

    # ── Correct solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        impl = textwrap.dedent('''
            """Order processing state machine — implementation."""

            class IllegalTransitionError(Exception):
                """Raised when a transition is not allowed from the current state."""
                pass


            class Order:
                STATES = ("NEW", "PENDING", "PAID", "SHIPPED", "DELIVERED", "CANCELLED")
                TERMINAL = ("DELIVERED", "CANCELLED")

                def __init__(self, order_id: str):
                    self.order_id = order_id
                    self.state = "NEW"
                    self.items: list[str] = []
                    self.address_verified = False

                # ── Helpers ──

                def _require_state(self, *allowed):
                    if self.state not in allowed:
                        raise IllegalTransitionError(
                            f"cannot transition from {self.state}"
                        )

                # ── Transitions ──

                def add_item(self, item: str) -> None:
                    """Add an item and move NEW -> PENDING."""
                    self._require_state("NEW")
                    if not item:
                        raise IllegalTransitionError("cannot add empty item")
                    self.items.append(item)
                    self.state = "PENDING"

                def pay(self) -> None:
                    """Pay for the order: PENDING -> PAID."""
                    self._require_state("PENDING")
                    if not self.items:
                        raise IllegalTransitionError("cannot pay without items")
                    self.state = "PAID"

                def verify_address(self) -> None:
                    """Mark the address as verified (no state change)."""
                    self.address_verified = True

                def ship(self) -> None:
                    """Ship the order: PAID -> SHIPPED.  Requires verified address."""
                    self._require_state("PAID")
                    if not self.address_verified:
                        raise IllegalTransitionError("address not verified")
                    self.state = "SHIPPED"

                def deliver(self) -> None:
                    """Mark delivered: SHIPPED -> DELIVERED."""
                    self._require_state("SHIPPED")
                    self.state = "DELIVERED"

                def cancel(self) -> None:
                    """Cancel the order: NEW/PENDING/PAID -> CANCELLED."""
                    self._require_state("NEW", "PENDING", "PAID")
                    self.state = "CANCELLED"
        ''').strip()

        return {"order_machine.py": impl}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''
            Let me reason through this finite state machine specification
            carefully, mapping every state, transition, and guard before
            writing any code.

            Step 1 — Enumerate the states:
            NEW (initial), PENDING, PAID, SHIPPED, DELIVERED (terminal),
            CANCELLED (terminal).  The initial state is NEW — every Order
            starts there.  DELIVERED and CANCELLED are terminal: no outgoing
            transitions are legal from them.

            Step 2 — Map the valid transitions:
            NEW -> PENDING (add_item)
            PENDING -> PAID (pay)
            PAID -> SHIPPED (ship)
            SHIPPED -> DELIVERED (deliver)
            NEW -> CANCELLED (cancel)
            PENDING -> CANCELLED (cancel)
            PAID -> CANCELLED (cancel)

            So the linear happy path is NEW -> PENDING -> PAID -> SHIPPED ->
            DELIVERED, and CANCELLED branches off from NEW, PENDING, or PAID.
            Once SHIPPED, you cannot cancel — the order is physically in
            transit.  This is the key guard small models miss.

            Step 3 — Enumerate the guards:
            - add_item: must be in NEW.  After adding, state becomes PENDING.
              The order must have at least one item for the transition to make
              sense, so I append the item then transition.
            - pay: must be in PENDING and must have items.  The items guard is
              mostly belt-and-suspenders since you can only reach PENDING by
              adding an item, but the spec requires it and a caller might
              clear items in a future version.
            - ship: must be in PAID and address_verified must be True.  The
              address verification is a separate method (verify_address) that
              does not change state — it just flips a flag.
            - deliver: must be in SHIPPED.  No additional guard.
            - cancel: must be in NEW, PENDING, or PAID.  Not SHIPPED, not
              DELIVERED, not CANCELLED.  This is the guard that prevents
              cancelling an order already in transit or delivered.

            Step 4 — Implement using _require_state helper:
            Each method calls _require_state with the allowed source states,
            then checks any additional guard, then sets self.state.  The
            helper raises IllegalTransitionError if the current state is not
            allowed, which covers both "wrong state" and "terminal state"
            cases (terminal states simply are not in any allowed list).

            Step 5 — Verify by tracing through each test case:
            - test_new_to_pending: NEW, add_item("widget") -> items non-empty,
              state PENDING. OK.
            - test_pending_to_paid: add_item then pay -> PENDING -> PAID. OK.
            - test_paid_to_shipped: add, pay, verify_address, ship ->
              PAID -> SHIPPED. OK.
            - test_shipped_to_delivered: ...ship then deliver -> DELIVERED. OK.
            - test_cancel_from_new: NEW -> cancel -> CANCELLED. OK.
            - test_cancel_from_pending: add_item -> PENDING -> cancel ->
              CANCELLED. OK.
            - test_pay_without_items_rejected: NEW, pay() -> _require_state
              sees NEW not in ("PENDING",) -> raises. state stays NEW. OK.
            - test_ship_without_address_rejected: add, pay, ship without
              verify_address -> address_verified False -> raises. state PAID.
              OK.
            - test_cancel_after_ship_rejected: ...ship -> SHIPPED, cancel ->
              _require_state sees SHIPPED not in (NEW,PENDING,PAID) -> raises.
              state SHIPPED. OK.
            - test_deliver_from_paid_rejected: PAID, deliver ->
              _require_state sees PAID not in ("SHIPPED",) -> raises. OK.
            - test_ship_from_new_rejected: NEW, ship -> not in ("PAID",) ->
              raises. OK.
            - test_delivered_is_terminal: ...deliver -> DELIVERED, cancel ->
              not in (NEW,PENDING,PAID) -> raises. state DELIVERED. OK.

            Let me double check the terminal property: DELIVERED and CANCELLED
            are terminal because no method lists them as an allowed source
            state.  _require_state will reject any transition from them.  I
            have verified this covers test_delivered_is_terminal.

            To confirm: the implementation maps every valid transition, every
            guard, and every terminal-state rejection from the spec.  The
            _require_state helper centralizes the state check so guards are
            consistent and easy to verify.
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
        test_code = codebase.get("test_state_machine.py", "")
        results = run_tests(new_codebase, test_code, timeout=15.0)
        score, breakdown = compute_test_score(results)
        breakdown["results"] = results.get("results", [])
        breakdown["method"] = "run_tests"
        return score, breakdown
