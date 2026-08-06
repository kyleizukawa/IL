"""
Long-horizon task: property_based_tests

Reasoning skill: Abstract reasoning about mathematical properties.
Failure mode: Small models write example-based tests, not property-based
tests that capture mathematical invariants.

The codebase is a `matrix_ops.py` module with a `Matrix` class supporting
add, multiply, and transpose. The model must write property-based tests
that capture mathematical properties (commutativity, associativity,
identity, involution) rather than specific examples.

The grader runs the model's tests against correct code (must pass) and
against mutated code (must fail). Score = mutation kill rate.
"""
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
class PropertyBasedTestsEnv(LongHorizonEnv):
    """Write property-based tests for matrix operations that catch mutants."""

    task_id = "property_based_tests"
    reasoning_skill = "Abstract reasoning about mathematical properties"
    failure_mode = (
        "Small models write example-based tests, not property-based tests "
        "that capture mathematical invariants"
    )
    token_budget = 700
    expected_concepts = [
        "property", "associative", "commutative", "identity",
        "inverse", "invariant", "edge case", "verify",
    ]

    # ── Codebase ──

    def gen_codebase(self) -> dict[str, str]:
        matrix_ops = textwrap.dedent('''\
            import random


            class Matrix:
                """A simple 2D matrix supporting add, multiply, and transpose."""

                def __init__(self, data):
                    """Create a matrix from a list of lists."""
                    if not data or not data[0]:
                        raise ValueError("Matrix must have at least one element")
                    self.rows = len(data)
                    self.cols = len(data[0])
                    for row in data:
                        if len(row) != self.cols:
                            raise ValueError("All rows must have same length")
                    self.data = [list(row) for row in data]

                @classmethod
                def random(cls, rows, cols, min_val=-10, max_val=10):
                    """Generate a random matrix."""
                    data = [[random.randint(min_val, max_val) for _ in range(cols)]
                            for _ in range(rows)]
                    return cls(data)

                @classmethod
                def zeros(cls, rows, cols):
                    """Create a zero matrix."""
                    return cls([[0] * cols for _ in range(rows)])

                @classmethod
                def identity(cls, n):
                    """Create an n×n identity matrix."""
                    data = [[1 if i == j else 0 for j in range(n)]
                            for i in range(n)]
                    return cls(data)

                def shape(self):
                    return (self.rows, self.cols)

                def __eq__(self, other):
                    if not isinstance(other, Matrix):
                        return False
                    return self.data == other.data

                def __add__(self, other):
                    if self.shape() != other.shape():
                        raise ValueError("Shape mismatch for addition")
                    result = [[self.data[i][j] + other.data[i][j]
                               for j in range(self.cols)]
                              for i in range(self.rows)]
                    return Matrix(result)

                def __mul__(self, other):
                    """Matrix multiplication (not element-wise)."""
                    if self.cols != other.rows:
                        raise ValueError("Shape mismatch for multiplication")
                    result = [[sum(self.data[i][k] * other.data[k][j]
                                   for k in range(self.cols))
                               for j in range(other.cols)]
                              for i in range(self.rows)]
                    return Matrix(result)

                def transpose(self):
                    result = [[self.data[j][i] for j in range(self.rows)]
                              for i in range(self.cols)]
                    return Matrix(result)

                def __repr__(self):
                    return f"Matrix({self.data})"
        ''')
        # The test file is a template — the model writes the actual tests
        test_template = textwrap.dedent('''\
            from matrix_ops import Matrix
            import random

            # TODO: Write property-based tests for Matrix operations.
            #
            # Properties to test:
            # 1. Addition is commutative: A + B == B + A
            # 2. Addition is associative: (A + B) + C == A + (B + C)
            # 3. Multiplication is associative: (A * B) * C == A * (B * C)
            # 4. Transpose of transpose is identity: (A^T)^T == A
            # 5. A + zeros == A (additive identity)
            # 6. A * identity == A (multiplicative identity)
            #
            # Write tests that generate RANDOM matrices and check these
            # properties hold. Do NOT just test with specific hardcoded
            # examples — use random generation to test the property
            # across many cases.
            #
            # Your tests will be run against correct code AND mutated code.
            # Good property-based tests should catch the mutations.
        ''')
        return {
            "matrix_ops.py": matrix_ops,
            "test_matrix_ops.py": test_template,
        }

    # ── Mutations for grading ──

    def _get_mutations(self) -> list[dict]:
        """Return list of mutations to test against."""
        return [
            {
                "name": "add_swapped_operands",
                "desc": "Addition does other + self instead of self + other (breaks commutativity test if it checks element order)",
                "code": textwrap.dedent('''\
                    import random


                    class Matrix:
                        def __init__(self, data):
                            if not data or not data[0]:
                                raise ValueError("Matrix must have at least one element")
                            self.rows = len(data)
                            self.cols = len(data[0])
                            for row in data:
                                if len(row) != self.cols:
                                    raise ValueError("All rows must have same length")
                            self.data = [list(row) for row in data]

                        @classmethod
                        def random(cls, rows, cols, min_val=-10, max_val=10):
                            data = [[random.randint(min_val, max_val) for _ in range(cols)]
                                    for _ in range(rows)]
                            return cls(data)

                        @classmethod
                        def zeros(cls, rows, cols):
                            return cls([[0] * cols for _ in range(rows)])

                        @classmethod
                        def identity(cls, n):
                            data = [[1 if i == j else 0 for j in range(n)]
                                    for i in range(n)]
                            return cls(data)

                        def shape(self):
                            return (self.rows, self.cols)

                        def __eq__(self, other):
                            if not isinstance(other, Matrix):
                                return False
                            return self.data == other.data

                        def __add__(self, other):
                            if self.shape() != other.shape():
                                raise ValueError("Shape mismatch")
                            # MUTATION: swapped to other - self (subtraction)
                            result = [[other.data[i][j] - self.data[i][j]
                                       for j in range(self.cols)]
                                      for i in range(self.rows)]
                            return Matrix(result)

                        def __mul__(self, other):
                            if self.cols != other.rows:
                                raise ValueError("Shape mismatch")
                            result = [[sum(self.data[i][k] * other.data[k][j]
                                           for k in range(self.cols))
                                       for j in range(other.cols)]
                                      for i in range(self.rows)]
                            return Matrix(result)

                        def transpose(self):
                            result = [[self.data[j][i] for j in range(self.rows)]
                                      for i in range(self.cols)]
                            return Matrix(result)

                        def __repr__(self):
                            return f"Matrix({self.data})"
                '''),
            },
            {
                "name": "mul_no_sum",
                "desc": "Multiplication uses first element instead of sum (breaks associativity)",
                "code": textwrap.dedent('''\
                    import random


                    class Matrix:
                        def __init__(self, data):
                            if not data or not data[0]:
                                raise ValueError("Matrix must have at least one element")
                            self.rows = len(data)
                            self.cols = len(data[0])
                            for row in data:
                                if len(row) != self.cols:
                                    raise ValueError("All rows must have same length")
                            self.data = [list(row) for row in data]

                        @classmethod
                        def random(cls, rows, cols, min_val=-10, max_val=10):
                            data = [[random.randint(min_val, max_val) for _ in range(cols)]
                                    for _ in range(rows)]
                            return cls(data)

                        @classmethod
                        def zeros(cls, rows, cols):
                            return cls([[0] * cols for _ in range(rows)])

                        @classmethod
                        def identity(cls, n):
                            data = [[1 if i == j else 0 for j in range(n)]
                                    for i in range(n)]
                            return cls(data)

                        def shape(self):
                            return (self.rows, self.cols)

                        def __eq__(self, other):
                            if not isinstance(other, Matrix):
                                return False
                            return self.data == other.data

                        def __add__(self, other):
                            if self.shape() != other.shape():
                                raise ValueError("Shape mismatch")
                            result = [[self.data[i][j] + other.data[i][j]
                                       for j in range(self.cols)]
                                      for i in range(self.rows)]
                            return Matrix(result)

                        def __mul__(self, other):
                            if self.cols != other.rows:
                                raise ValueError("Shape mismatch")
                            # MUTATION: uses first product instead of sum
                            result = [[self.data[i][0] * other.data[0][j]
                                       for j in range(other.cols)]
                                      for i in range(self.rows)]
                            return Matrix(result)

                        def transpose(self):
                            result = [[self.data[j][i] for j in range(self.rows)]
                                      for i in range(self.cols)]
                            return Matrix(result)

                        def __repr__(self):
                            return f"Matrix({self.data})"
                '''),
            },
            {
                "name": "transpose_no_swap",
                "desc": "Transpose returns original (breaks involution property)",
                "code": textwrap.dedent('''\
                    import random


                    class Matrix:
                        def __init__(self, data):
                            if not data or not data[0]:
                                raise ValueError("Matrix must have at least one element")
                            self.rows = len(data)
                            self.cols = len(data[0])
                            for row in data:
                                if len(row) != self.cols:
                                    raise ValueError("All rows must have same length")
                            self.data = [list(row) for row in data]

                        @classmethod
                        def random(cls, rows, cols, min_val=-10, max_val=10):
                            data = [[random.randint(min_val, max_val) for _ in range(cols)]
                                    for _ in range(rows)]
                            return cls(data)

                        @classmethod
                        def zeros(cls, rows, cols):
                            return cls([[0] * cols for _ in range(rows)])

                        @classmethod
                        def identity(cls, n):
                            data = [[1 if i == j else 0 for j in range(n)]
                                    for i in range(n)]
                            return cls(data)

                        def shape(self):
                            return (self.rows, self.cols)

                        def __eq__(self, other):
                            if not isinstance(other, Matrix):
                                return False
                            return self.data == other.data

                        def __add__(self, other):
                            if self.shape() != other.shape():
                                raise ValueError("Shape mismatch")
                            result = [[self.data[i][j] + other.data[i][j]
                                       for j in range(self.cols)]
                                      for i in range(self.rows)]
                            return Matrix(result)

                        def __mul__(self, other):
                            if self.cols != other.rows:
                                raise ValueError("Shape mismatch")
                            result = [[sum(self.data[i][k] * other.data[k][j]
                                           for k in range(self.cols))
                                       for j in range(other.cols)]
                                      for i in range(self.rows)]
                            return Matrix(result)

                        def transpose(self):
                            # MUTATION: returns self instead of transposed
                            return Matrix([list(row) for row in self.data])

                        def __repr__(self):
                            return f"Matrix({self.data})"
                '''),
            },
            {
                "name": "identity_wrong",
                "desc": "Identity matrix has 2 on diagonal (breaks multiplicative identity)",
                "code": textwrap.dedent('''\
                    import random


                    class Matrix:
                        def __init__(self, data):
                            if not data or not data[0]:
                                raise ValueError("Matrix must have at least one element")
                            self.rows = len(data)
                            self.cols = len(data[0])
                            for row in data:
                                if len(row) != self.cols:
                                    raise ValueError("All rows must have same length")
                            self.data = [list(row) for row in data]

                        @classmethod
                        def random(cls, rows, cols, min_val=-10, max_val=10):
                            data = [[random.randint(min_val, max_val) for _ in range(cols)]
                                    for _ in range(rows)]
                            return cls(data)

                        @classmethod
                        def zeros(cls, rows, cols):
                            return cls([[0] * cols for _ in range(rows)])

                        @classmethod
                        def identity(cls, n):
                            # MUTATION: 2 on diagonal instead of 1
                            data = [[2 if i == j else 0 for j in range(n)]
                                    for i in range(n)]
                            return cls(data)

                        def shape(self):
                            return (self.rows, self.cols)

                        def __eq__(self, other):
                            if not isinstance(other, Matrix):
                                return False
                            return self.data == other.data

                        def __add__(self, other):
                            if self.shape() != other.shape():
                                raise ValueError("Shape mismatch")
                            result = [[self.data[i][j] + other.data[i][j]
                                       for j in range(self.cols)]
                                      for i in range(self.rows)]
                            return Matrix(result)

                        def __mul__(self, other):
                            if self.cols != other.rows:
                                raise ValueError("Shape mismatch")
                            result = [[sum(self.data[i][k] * other.data[k][j]
                                           for k in range(self.cols))
                                       for j in range(other.cols)]
                                      for i in range(self.rows)]
                            return Matrix(result)

                        def transpose(self):
                            result = [[self.data[j][i] for j in range(self.rows)]
                                      for i in range(self.cols)]
                            return Matrix(result)

                        def __repr__(self):
                            return f"Matrix({self.data})"
                '''),
            },
            {
                "name": "add_not_associative",
                "desc": "Addition multiplies by 2 (breaks associativity)",
                "code": textwrap.dedent('''\
                    import random


                    class Matrix:
                        def __init__(self, data):
                            if not data or not data[0]:
                                raise ValueError("Matrix must have at least one element")
                            self.rows = len(data)
                            self.cols = len(data[0])
                            for row in data:
                                if len(row) != self.cols:
                                    raise ValueError("All rows must have same length")
                            self.data = [list(row) for row in data]

                        @classmethod
                        def random(cls, rows, cols, min_val=-10, max_val=10):
                            data = [[random.randint(min_val, max_val) for _ in range(cols)]
                                    for _ in range(rows)]
                            return cls(data)

                        @classmethod
                        def zeros(cls, rows, cols):
                            return cls([[0] * cols for _ in range(rows)])

                        @classmethod
                        def identity(cls, n):
                            data = [[1 if i == j else 0 for j in range(n)]
                                    for i in range(n)]
                            return cls(data)

                        def shape(self):
                            return (self.rows, self.cols)

                        def __eq__(self, other):
                            if not isinstance(other, Matrix):
                                return False
                            return self.data == other.data

                        def __add__(self, other):
                            if self.shape() != other.shape():
                                raise ValueError("Shape mismatch")
                            # MUTATION: doubles the sum
                            result = [[2 * (self.data[i][j] + other.data[i][j])
                                       for j in range(self.cols)]
                                      for i in range(self.rows)]
                            return Matrix(result)

                        def __mul__(self, other):
                            if self.cols != other.rows:
                                raise ValueError("Shape mismatch")
                            result = [[sum(self.data[i][k] * other.data[k][j]
                                           for k in range(self.cols))
                                       for j in range(other.cols)]
                                      for i in range(self.rows)]
                            return Matrix(result)

                        def transpose(self):
                            result = [[self.data[j][i] for j in range(self.rows)]
                                      for i in range(self.cols)]
                            return Matrix(result)

                        def __repr__(self):
                            return f"Matrix({self.data})"
                '''),
            },
        ]

    # ── Task description ──

    def gen_task(self, codebase: dict[str, str]) -> str:
        return textwrap.dedent('''\
            You are given a `matrix_ops.py` module with a `Matrix` class
            that supports addition (`__add__`), multiplication (`__mul__`),
            and transpose.

            Your task is to write property-based tests that capture the
            mathematical properties of these operations. Do NOT write
            example-based tests with hardcoded values — instead, generate
            random matrices and verify that mathematical invariants hold.

            Properties to test:
            1. **Commutativity of addition**: A + B == B + A
            2. **Associativity of addition**: (A + B) + C == A + (B + C)
            3. **Associativity of multiplication**: (A * B) * C == A * (B * C)
            4. **Transpose involution**: (A^T)^T == A
            5. **Additive identity**: A + zeros == A
            6. **Multiplicative identity**: A * identity == A

            For each property, generate random matrices of appropriate
            shapes and verify the property holds. Run each check multiple
            times with different random matrices.

            Your tests will be evaluated by running them against:
            - The correct implementation (they must all pass)
            - Several mutated implementations (they should catch the bugs)

            Score = fraction of mutations caught by your tests.

            Write your tests in a ```python:test_matrix_ops.py``` block.
            The tests should import from `matrix_ops` and define functions
            starting with `test_`.
        ''')

    # ── Solution ──

    def gen_solution(self, codebase: dict[str, str]) -> dict[str, str]:
        tests = textwrap.dedent('''\
            from matrix_ops import Matrix
            import random


            def test_addition_commutative():
                \"\"\"A + B == B + A for random matrices.\"\"\"
                for _ in range(20):
                    rows, cols = random.randint(1, 5), random.randint(1, 5)
                    A = Matrix.random(rows, cols)
                    B = Matrix.random(rows, cols)
                    assert A + B == B + A, f"Commutativity failed for {A}, {B}"


            def test_addition_associative():
                \"\"\"(A + B) + C == A + (B + C) for random matrices.\"\"\"
                for _ in range(20):
                    rows, cols = random.randint(1, 5), random.randint(1, 5)
                    A = Matrix.random(rows, cols)
                    B = Matrix.random(rows, cols)
                    C = Matrix.random(rows, cols)
                    assert (A + B) + C == A + (B + C), "Associativity failed"


            def test_multiplication_associative():
                \"\"\"(A * B) * C == A * (B * C) for random compatible matrices.\"\"\"
                for _ in range(20):
                    n = random.randint(1, 5)
                    m = random.randint(1, 5)
                    p = random.randint(1, 5)
                    A = Matrix.random(n, m)
                    B = Matrix.random(m, p)
                    C = Matrix.random(p, random.randint(1, 5))
                    assert (A * B) * C == A * (B * C), "Mul associativity failed"


            def test_transpose_involution():
                \"\"\"(A^T)^T == A for random matrices.\"\"\"
                for _ in range(20):
                    rows, cols = random.randint(1, 5), random.randint(1, 5)
                    A = Matrix.random(rows, cols)
                    assert A.transpose().transpose() == A, "Transpose involution failed"


            def test_transpose_swaps_dimensions():
                \"\"\"A^T has swapped dimensions and A[i][j] == A^T[j][i].\"\"\"
                for _ in range(20):
                    rows, cols = random.randint(2, 5), random.randint(2, 5)
                    A = Matrix.random(rows, cols)
                    AT = A.transpose()
                    assert AT.shape() == (cols, rows), (
                        f"Transpose shape wrong: {AT.shape()} != {(cols, rows)}"
                    )
                    for i in range(rows):
                        for j in range(cols):
                            assert A.data[i][j] == AT.data[j][i], (
                                f"Transpose element mismatch at ({i},{j})"
                            )


            def test_additive_identity():
                \"\"\"A + zeros == A for random matrices.\"\"\"
                for _ in range(20):
                    rows, cols = random.randint(1, 5), random.randint(1, 5)
                    A = Matrix.random(rows, cols)
                    Z = Matrix.zeros(rows, cols)
                    assert A + Z == A, "Additive identity failed"


            def test_multiplicative_identity():
                \"\"\"A * identity == A for random square matrices.\"\"\"
                for _ in range(20):
                    n = random.randint(1, 5)
                    A = Matrix.random(n, n)
                    I = Matrix.identity(n)
                    assert A * I == A, "Multiplicative identity failed"


            def test_edge_case_1x1():
                \"\"\"Edge case: 1x1 matrices should work for all operations.\"\"\"
                A = Matrix([[5]])
                B = Matrix([[3]])
                assert A + B == Matrix([[8]])
                assert A * B == Matrix([[15]])
                assert A.transpose() == A
        ''')
        return {"test_matrix_ops.py": tests}

    # ── Teacher reasoning ──

    def gen_reasoning(self, codebase: dict[str, str],
                      solution: dict[str, str]) -> str:
        return textwrap.dedent('''\
            I need to write property-based tests for matrix operations. The
            key insight is that property-based tests verify mathematical
            invariants that hold for ALL valid inputs, not just specific
            examples. This means I should generate random matrices and
            check that each property holds.

            Let me identify the mathematical properties of each operation:

            ── Addition properties ──
            1. Commutative: A + B == B + A. This is a fundamental property
               of matrix addition. For any two matrices of the same shape,
               swapping the order gives the same result.
            2. Associative: (A + B) + C == A + (B + C). Grouping doesn't
               matter for addition.
            3. Identity: A + 0 == A. Adding a zero matrix of the same shape
               leaves A unchanged. The zero matrix is the additive identity.

            ── Multiplication properties ──
            4. Associative: (A * B) * C == A * (B * C). Note: multiplication
               is NOT commutative (A * B != B * A in general), so I should
               NOT test commutativity for multiplication.
            5. Identity: A * I == A. Multiplying by the identity matrix
               leaves A unchanged. The identity matrix is the multiplicative
               identity. This only works for square matrices (n×n).

            ── Transpose properties ──
            6. Involution: (A^T)^T == A. Transposing twice gives back the
               original matrix. This is an inverse property — transpose is
               its own inverse.

            For each property, I'll generate random matrices of appropriate
            shapes. For addition, both matrices need the same shape. For
            multiplication, the inner dimensions must match (A is m×n,
            B is n×p). For the identity property, I need square matrices.

            Let me think about edge cases. A 1×1 matrix is a valid edge
            case — all operations should work. I should also test with
            non-square matrices for transpose (a 2×3 matrix transposed
            becomes 3×2, and transposing again gives back 2×3).

            Now, why will these tests catch mutations? Let me reason about
            each mutation:

            - If addition does subtraction (A + B = B - A), the commutative
              test catches it: A + B would be B - A, but B + A would be
              A - B. These are different unless A == B.

            - If multiplication doesn't sum (uses first product), the
              associativity test catches it: the sum is essential for
              correct matrix multiplication, and skipping it changes
              the result for any matrix with cols > 1.

            - If transpose returns the original, the involution test
              catches it: (A^T)^T would be A^T (which is A since transpose
              is a no-op), but for non-square matrices A^T != A, so
              A^T^T = A^T != A.

            Wait, actually if transpose returns self, then (A^T)^T = A^T = A
            (since A^T is just A). So the involution test would PASS for
            square matrices but FAIL for non-square matrices. I need to
              make sure I test with non-square matrices. My random
              generation uses random rows and cols independently, so
              non-square matrices will be generated. Good.

            - If identity has 2 on the diagonal, A * I = 2A != A, so the
              multiplicative identity test catches it.

            - If addition doubles the sum, associativity catches it:
              (A+B)+C = 2(2(A+B)+C) = 4A+4B+2C, while A+(B+C) = 2(A+2(B+C))
              = 2A+4B+4C. These differ.

            Let me verify my tests pass on the correct implementation by
            tracing through test_addition_commutative:
            - Generate A = random 3×4, B = random 3×4
            - A + B: element-wise sum
            - B + A: element-wise sum (same result)
            - Assert A + B == B + A: True (addition is commutative) ✓

            And test_transpose_involution with a 2×3 matrix:
            - A = random 2×3
            - A.transpose(): 3×2 matrix
            - A.transpose().transpose(): 2×3 matrix (same as A)
            - Assert == A: True ✓

            For the edge case test with 1×1 matrices:
            - A = [[5]], B = [[3]]
            - A + B = [[8]] ✓
            - A * B = [[15]] ✓
            - A.transpose() = [[5]] = A ✓

            All tests should pass on correct code and fail on mutated code.
            The property-based approach with random generation ensures
            the tests are robust — they check the invariant, not a
            specific value that might happen to work by coincidence.

            To verify: I've covered commutative, associative, identity,
            inverse (involution), and edge case properties. Each test
            generates random inputs and checks the invariant holds,
            which is the essence of property-based testing.
        ''')

    # ── Grading ──

    def grade_correctness(self, codebase: dict[str, str],
                          response: str) -> tuple[float, dict]:
        answer = grader_extract_answer(response)
        blocks = parse_code_blocks(answer)
        if not blocks:
            blocks = parse_code_blocks(response)

        # The model provides test_matrix_ops.py
        test_key = None
        for key in blocks:
            if "test" in key and "matrix" in key:
                test_key = key
                break
        if not test_key:
            # Accept any test file
            for key in blocks:
                if "test" in key.lower():
                    test_key = key
                    break
        if not test_key:
            return 0.0, {
                "reason": "no test file found in response",
                "mutation_results": [],
            }

        model_tests = blocks[test_key]
        correct_codebase = codebase  # original matrix_ops.py is correct

        # ── 1. Tests must pass on correct code ──
        results_correct = run_tests(correct_codebase, model_tests, timeout=20.0)
        if results_correct.get("total", 0) == 0:
            return 0.0, {
                "reason": "no tests ran or syntax error in test code",
                "mutation_results": [],
            }
        pass_rate_correct = results_correct.get("passed", 0) / results_correct.get("total", 1)
        if pass_rate_correct < 0.5:
            # Tests must mostly pass on correct code
            return pass_rate_correct * 0.1, {
                "reason": "tests fail on correct code",
                "correct_results": results_correct,
                "mutation_results": [],
            }

        # ── 2. Run tests against each mutation ──
        mutations = self._get_mutations()
        mutation_results = []
        killed = 0
        for mut in mutations:
            mutated_codebase = dict(correct_codebase)
            mutated_codebase["matrix_ops.py"] = mut["code"]
            results_mut = run_tests(mutated_codebase, model_tests, timeout=20.0)
            mut_pass_rate = results_mut.get("passed", 0) / max(1, results_mut.get("total", 1))
            # Mutation is "killed" if at least one test fails
            killed_mut = mut_pass_rate < 1.0
            if killed_mut:
                killed += 1
            mutation_results.append({
                "name": mut["name"],
                "desc": mut["desc"],
                "killed": killed_mut,
                "pass_rate": mut_pass_rate,
                "failed_tests": results_mut.get("failed", 0),
            })

        kill_rate = killed / len(mutations) if mutations else 0.0

        # Final score: kill rate, scaled by how well tests pass on correct code
        score = kill_rate * pass_rate_correct
        breakdown = {
            "kill_rate": kill_rate,
            "mutations_killed": killed,
            "total_mutations": len(mutations),
            "pass_rate_correct": pass_rate_correct,
            "correct_results": {
                "total": results_correct.get("total", 0),
                "passed": results_correct.get("passed", 0),
            },
            "mutation_results": mutation_results,
        }
        return score, breakdown
