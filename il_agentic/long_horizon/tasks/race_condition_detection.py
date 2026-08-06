"""
Long-horizon task: race_condition_detection

Reasoning skill: Concurrency reasoning about execution interleavings.
Failure mode: Small models can't reason about non-deterministic execution order.

This is a Q&A task — the model explains the race condition and the fix in text,
not code. The answer should contain both an explanation and a code fix.

The codebase is a BankAccount class with a race condition in transfer():
the balance check and withdrawal are not atomic. The model must identify the
race condition, explain a specific interleaving that causes a problem, and
provide a fix using a lock around the critical section.
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
class RaceConditionDetectionEnv(LongHorizonEnv):
    """Identify and fix a race condition in a threaded BankAccount class."""

    task_id = "race_condition_detection"
    reasoning_skill = "Concurrency reasoning about execution interleavings"
    failure_mode = (
        "Small models can't reason about non-deterministic execution order "
        "and fail to identify check-then-act race conditions"
    )
    token_budget = 700
    expected_concepts = [
        "race condition", "interleaving", "atomic", "lock",
        "thread", "critical section", "non-deterministic", "verify",
    ]

    # ── Codebase ──

    def gen_codebase(self) -> dict[str, str]:
        bank_account = textwrap.dedent('''\
            import threading


            class BankAccount:
                """A simple bank account supporting concurrent transfers."""

                def __init__(self, owner, balance=0):
                    self.owner = owner
                    self.balance = balance

                def deposit(self, amount):
                    if amount <= 0:
                        raise ValueError("Deposit must be positive")
                    self.balance += amount

                def withdraw(self, amount):
                    if amount <= 0:
                        raise ValueError("Withdrawal must be positive")
                    if self.balance >= amount:
                        self.balance -= amount
                        return True
                    return False

                def transfer(self, target, amount):
                    """Transfer `amount` from this account to `target`.

                    This method is intended to be safe for concurrent use
                    from multiple threads, but it currently has a subtle
                    bug that can cause incorrect balances under load.
                    """
                    # Check that we have enough money
                    if self.balance >= amount:
                        # Simulate some processing delay between check and action
                        import time
                        time.sleep(0.001)
                        # Withdraw from this account
                        self.balance -= amount
                        # Deposit into target account
                        target.balance += amount
                        return True
                    return False


            def run_concurrent_transfers(account_a, account_b, n_threads, amount):
                """Run n_threads concurrent transfers from a to b."""
                threads = []
                for _ in range(n_threads):
                    t = threading.Thread(
                        target=account_a.transfer, args=(account_b, amount)
                    )
                    threads.append(t)
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        ''')
        test_file = textwrap.dedent('''\
            from bank_account import BankAccount, run_concurrent_transfers


            def test_transfer_basic():
                a = BankAccount("alice", 1000)
                b = BankAccount("bob", 0)
                assert a.transfer(b, 100) is True
                assert a.balance == 900
                assert b.balance == 100

            def test_transfer_insufficient():
                a = BankAccount("alice", 50)
                b = BankAccount("bob", 0)
                assert a.transfer(b, 100) is False
                assert a.balance == 50
                assert b.balance == 0

            def test_concurrent_no_negative_balance():
                \"\"\"Under concurrency, balance must never go negative.\"\"\"
                a = BankAccount("alice", 1000)
                b = BankAccount("bob", 0)
                run_concurrent_transfers(a, b, n_threads=10, amount=100)
                # Total transferred should be exactly 1000 (10 * 100)
                # and alice should never go below 0.
                assert a.balance >= 0, f"Balance went negative: {a.balance}"
                assert a.balance + b.balance == 1000, (
                    f"Conservation violated: a={a.balance}, b={b.balance}"
                )

            def test_concurrent_conservation():
                \"\"\"Total money is conserved across concurrent transfers.\"\"\"
                a = BankAccount("alice", 5000)
                b = BankAccount("bob", 0)
                run_concurrent_transfers(a, b, n_threads=50, amount=100)
                assert a.balance + b.balance == 5000, (
                    f"Money lost/created: a={a.balance}, b={b.balance}"
                )
                assert a.balance >= 0
        ''')
        return {
            "bank_account.py": bank_account,
            "test_bank_account.py": test_file,
        }

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''\
            You are given a `bank_account.py` module with a `BankAccount` class
            that supports concurrent transfers between accounts using Python's
            `threading` module.

            The `transfer()` method is documented as "safe for concurrent use"
            but it contains a race condition that can cause incorrect balances
            under concurrent load.

            Your task:
            1. Identify the race condition in the `transfer()` method.
            2. Explain a specific thread interleaving that causes the bug —
               describe the exact sequence of operations across two threads
               that leads to an incorrect result.
            3. Provide a corrected version of `bank_account.py` that fixes
               the race condition.

            Your response must include:
            - A clear explanation of the race condition (what makes it
              non-deterministic, which operations must be atomic).
            - A specific interleaving showing how two threads cause the bug.
            - The fixed code in a ```python:bank_account.py``` block.

            The fixed code must pass all tests in `test_bank_account.py`,
            including the concurrency tests that would fail with the race
            condition present.
        ''')

    # ── Solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        fixed = textwrap.dedent('''\
            import threading


            class BankAccount:
                """A simple bank account supporting concurrent transfers."""

                def __init__(self, owner, balance=0):
                    self.owner = owner
                    self.balance = balance
                    self._lock = threading.Lock()

                def deposit(self, amount):
                    if amount <= 0:
                        raise ValueError("Deposit must be positive")
                    with self._lock:
                        self.balance += amount

                def withdraw(self, amount):
                    if amount <= 0:
                        raise ValueError("Withdrawal must be positive")
                    with self._lock:
                        if self.balance >= amount:
                            self.balance -= amount
                            return True
                        return False

                def transfer(self, target, amount):
                    """Transfer `amount` from this account to `target`.

                    Thread-safe: the check-then-act sequence is protected
                    by a lock so that no other thread can interleave between
                    the balance check and the withdrawal.
                    """
                    with self._lock:
                        if self.balance >= amount:
                            self.balance -= amount
                            # Deposit into target (target has its own lock)
                            with target._lock:
                                target.balance += amount
                            return True
                        return False


            def run_concurrent_transfers(account_a, account_b, n_threads, amount):
                """Run n_threads concurrent transfers from a to b."""
                threads = []
                for _ in range(n_threads):
                    t = threading.Thread(
                        target=account_a.transfer, args=(account_b, amount)
                    )
                    threads.append(t)
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        ''')
        return {"bank_account.py": fixed}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''\
            Let me trace through the `transfer()` method in `bank_account.py`
            to understand the concurrency bug.

            The transfer method does:
                if self.balance >= amount:       # check
                    time.sleep(0.001)            # delay
                    self.balance -= amount       # action: withdraw
                    target.balance += amount     # action: deposit
                    return True
                return False

            This is a classic check-then-act pattern. The check
            (`self.balance >= amount`) and the action (`self.balance -= amount`)
            are two separate operations with no lock protecting the critical
            section between them. This means the operations are not atomic —
            a thread can be preempted between the check and the action.

            Let me construct a specific interleaving that causes a race
            condition. Suppose account_a has balance 100 and two threads
            each try to transfer 100 to account_b:

            Thread 1                          Thread 2
            ---------                          ---------
            check: balance >= 100? YES
                                              check: balance >= 100? YES
            (both threads see balance=100)
            sleep(0.001)                      sleep(0.001)
            balance -= 100  -> balance=0
                                              balance -= 100  -> balance=-100
            target += 100
                                              target += 100

            After this interleaving, account_a.balance = -100 (negative!)
            and account_b.balance = 200. The total money went from 100 to
            100, but account_a went negative, which should never happen.
            The invariant "balance >= 0" is violated.

            The root cause is non-deterministic execution order: the OS
            scheduler can interleave thread execution at any point, and
            without a lock, both threads read the same stale balance value
            before either writes. The check and the withdrawal must be
            atomic — no other thread should be able to read or write
            the balance between the check and the withdrawal.

            The fix is to add a `threading.Lock` around the critical section.
            I'll add a `self._lock = threading.Lock()` in `__init__`, and
            wrap the entire check-then-act sequence in `with self._lock:`.
            This makes the check and withdrawal atomic — once a thread
            acquires the lock, no other thread can enter the critical
            section until it releases.

            For the deposit into target, I use `with target._lock:` to
            protect target's balance as well. This prevents a concurrent
            deposit/transfer into target from corrupting its balance.

            Let me verify the fix by tracing through the same interleaving:
            Thread 1 acquires self._lock, checks balance=100 (YES), withdraws
            100, balance=0, releases lock. Thread 2 was blocked waiting for
            the lock; now it acquires it, checks balance=0 (NO, 0 < 100),
            returns False. Balance never goes negative. The invariant holds.

            Let me also verify the test `test_concurrent_no_negative_balance`:
            10 threads each transfer 100 from an account with 1000. With the
            lock, at most 10 transfers succeed (1000/100=10), balance ends
            at 0, never negative. Conservation: 0 + 1000 = 1000. Correct.

            I should also verify there's no deadlock risk. The transfer
            acquires self._lock then target._lock. If two accounts transfer
            to each other simultaneously, there's a potential deadlock
            (A locks A, B locks B, A waits for B, B waits for A). For this
            task the tests only transfer in one direction (a->b), so it's
            safe. A production fix would use lock ordering, but that's
            beyond the scope here.

            To confirm: the fix adds a lock around the critical section,
            making the check-then-act atomic, eliminating the race
            condition caused by non-deterministic thread interleaving.
        ''')

    # ── Grading ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        answer = grader_extract_answer(response)
        reasoning = grader_extract_reasoning(response)
        blocks = parse_code_blocks(answer)
        if not blocks:
            blocks = parse_code_blocks(response)

        # ── 1. Code fix score (70%) ──
        code_score = 0.0
        test_details = {}
        if "bank_account.py" in blocks:
            fixed_codebase = apply_code_changes(codebase, blocks)
            test_code = codebase.get("test_bank_account.py", "")
            results = run_tests(fixed_codebase, test_code, timeout=20.0)
            test_score, test_details = compute_test_score(results)
            # Require the concurrency tests to pass specifically
            concurrency_ok = True
            for r in results.get("results", []):
                if "concurrent" in r.get("name", "") and r.get("status") != "pass":
                    concurrency_ok = False
            if concurrency_ok and test_score >= 0.75:
                code_score = test_score
            else:
                code_score = test_score * 0.5
        else:
            test_details = {"reason": "no bank_account.py code block found"}

        # ── 2. Explanation score (30%) ──
        full_text = (reasoning + " " + answer).lower()
        explanation_concepts = [
            "race condition", "interleaving", "atomic", "lock",
            "thread", "critical section", "non-deterministic",
        ]
        found = [c for c in explanation_concepts if c in full_text]
        explanation_score = len(found) / len(explanation_concepts)

        # Bonus: mentions a specific interleaving with two threads
        if re.search(r"thread\s*1.*thread\s*2|thread\s+a.*thread\s+b", full_text):
            explanation_score = min(1.0, explanation_score + 0.1)
        # Bonus: mentions "check" and "act" or "check-then-act"
        if "check" in full_text and "act" in full_text:
            explanation_score = min(1.0, explanation_score + 0.05)

        score = code_score * 0.7 + explanation_score * 0.3
        breakdown = {
            "code_score": code_score,
            "explanation_score": explanation_score,
            "concepts_found": found,
            "test_details": test_details,
            "has_code_block": "bank_account.py" in blocks,
        }
        return score, breakdown
