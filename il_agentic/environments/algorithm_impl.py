"""
Environment 13: Algorithm Implementation

Skill: Implementing algorithms from textual specifications.

The model is given a textual spec for a well-known algorithm and must
implement it from scratch. The codebase contains a stub class/function
and comprehensive test cases. The model must read the spec, understand
the algorithm's key insight, and implement it correctly.

Domains:
- LRU Cache
- Consistent Hashing
- Bloom Filter
- Union-Find (Disjoint Set Union)
- Sliding Window Maximum
- Interval Scheduler

Difficulty scaling:
- easy: simple algorithm with clear spec (e.g., sliding window max)
- medium: moderate complexity (e.g., LRU cache, union-find)
- hard: complex algorithm with subtle edge cases (e.g., consistent hashing, bloom filter)
"""
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, extract_reasoning, parse_code_blocks, apply_code_changes,
    run_tests, compute_test_score, code_similarity,
)


# ── Domain definitions ──

DOMAINS = {

    # ── Domain 1: LRU Cache ──
    "lru_cache": {
        "module": "lru_cache",
        "easy": {
            "spec": (
                "Implement a simple FIFO queue with a max size. "
                "Class `BoundedQueue` with methods `push(item)` and `pop()` — "
                "push adds to the back, pop removes from the front. "
                "If the queue is full, push drops the front item first. "
                "If the queue is empty, pop returns None."
            ),
            "stub": textwrap.dedent('''
                class BoundedQueue:
                    def __init__(self, max_size):
                        self.max_size = max_size
                        # TODO: implement

                    def push(self, item):
                        # TODO: implement
                        pass

                    def pop(self):
                        # TODO: implement
                        pass
            ''').strip(),
            "solution": textwrap.dedent('''
                class BoundedQueue:
                    def __init__(self, max_size):
                        self.max_size = max_size
                        self._data = []

                    def push(self, item):
                        if len(self._data) >= self.max_size:
                            self._data.pop(0)
                        self._data.append(item)

                    def pop(self):
                        if not self._data:
                            return None
                        return self._data.pop(0)
            ''').strip(),
            "test": textwrap.dedent('''
                from lru_cache import BoundedQueue
                def test_basic():
                    q = BoundedQueue(3)
                    q.push(1)
                    q.push(2)
                    q.push(3)
                    assert q.pop() == 1
                    assert q.pop() == 2
                    assert q.pop() == 3
                def test_overflow():
                    q = BoundedQueue(2)
                    q.push(1)
                    q.push(2)
                    q.push(3)
                    assert q.pop() == 2
                    assert q.pop() == 3
                def test_empty_pop():
                    q = BoundedQueue(5)
                    assert q.pop() is None
                def test_push_after_empty():
                    q = BoundedQueue(2)
                    assert q.pop() is None
                    q.push(42)
                    assert q.pop() == 42
            ''').strip(),
        },
        "medium": {
            "spec": (
                "Implement an LRU (Least Recently Used) cache. "
                "Class `LRUCache` with `__init__(capacity)`, `get(key)`, and `put(key, value)`. "
                "get returns the value for key if present (and marks it as recently used), "
                "else returns -1. put inserts/updates a key-value pair. If capacity is "
                "exceeded, evict the least recently used key. Both operations must be O(1)."
            ),
            "stub": textwrap.dedent('''
                class LRUCache:
                    def __init__(self, capacity):
                        self.capacity = capacity
                        # TODO: implement

                    def get(self, key):
                        # TODO: implement
                        pass

                    def put(self, key, value):
                        # TODO: implement
                        pass
            ''').strip(),
            "solution": textwrap.dedent('''
                from collections import OrderedDict
                class LRUCache:
                    def __init__(self, capacity):
                        self.capacity = capacity
                        self._cache = OrderedDict()

                    def get(self, key):
                        if key not in self._cache:
                            return -1
                        self._cache.move_to_end(key)
                        return self._cache[key]

                    def put(self, key, value):
                        if key in self._cache:
                            self._cache.move_to_end(key)
                        self._cache[key] = value
                        if len(self._cache) > self.capacity:
                            self._cache.popitem(last=False)
            ''').strip(),
            "test": textwrap.dedent('''
                from lru_cache import LRUCache
                def test_basic_get_put():
                    c = LRUCache(2)
                    c.put(1, 10)
                    c.put(2, 20)
                    assert c.get(1) == 10
                    assert c.get(2) == 20
                def test_eviction():
                    c = LRUCache(2)
                    c.put(1, 10)
                    c.put(2, 20)
                    c.get(1)
                    c.put(3, 30)
                    assert c.get(2) == -1
                    assert c.get(1) == 10
                    assert c.get(3) == 30
                def test_update_existing():
                    c = LRUCache(2)
                    c.put(1, 10)
                    c.put(1, 100)
                    assert c.get(1) == 100
                def test_miss():
                    c = LRUCache(2)
                    assert c.get(99) == -1
                def test_capacity_one():
                    c = LRUCache(1)
                    c.put(1, 10)
                    c.put(2, 20)
                    assert c.get(1) == -1
                    assert c.get(2) == 20
                def test_lru_order():
                    c = LRUCache(3)
                    c.put(1, 1)
                    c.put(2, 2)
                    c.put(3, 3)
                    c.get(1)
                    c.get(2)
                    c.put(4, 4)
                    assert c.get(3) == -1
                    assert c.get(1) == 1
            ''').strip(),
        },
        "hard": {
            "spec": (
                "Implement an LRU cache with TTL (time-to-live) support. "
                "Class `TTLCache` with `__init__(capacity, ttl)` where ttl is in seconds. "
                "`get(key)` returns value if key exists AND has not expired, else -1. "
                "Expired entries should be removed on access. "
                "`put(key, value)` inserts/updates a key. If capacity exceeded, evict LRU. "
                "`cleanup()` removes all expired entries. Use time.time() for timestamps."
            ),
            "stub": textwrap.dedent('''
                import time
                class TTLCache:
                    def __init__(self, capacity, ttl):
                        self.capacity = capacity
                        self.ttl = ttl
                        # TODO: implement

                    def get(self, key):
                        # TODO: implement
                        pass

                    def put(self, key, value):
                        # TODO: implement
                        pass

                    def cleanup(self):
                        # TODO: implement
                        pass
            ''').strip(),
            "solution": textwrap.dedent('''
                import time
                from collections import OrderedDict
                class TTLCache:
                    def __init__(self, capacity, ttl):
                        self.capacity = capacity
                        self.ttl = ttl
                        self._cache = OrderedDict()
                        self._timestamps = {}

                    def _is_expired(self, key, now=None):
                        if key not in self._timestamps:
                            return True
                        if now is None:
                            now = time.time()
                        return now - self._timestamps[key] > self.ttl

                    def _evict_if_needed(self):
                        while len(self._cache) > self.capacity:
                            k, _ = self._cache.popitem(last=False)
                            self._timestamps.pop(k, None)

                    def get(self, key):
                        if key not in self._cache:
                            return -1
                        if self._is_expired(key):
                            del self._cache[key]
                            self._timestamps.pop(key, None)
                            return -1
                        self._cache.move_to_end(key)
                        return self._cache[key]

                    def put(self, key, value):
                        now = time.time()
                        if key in self._cache:
                            self._cache.move_to_end(key)
                        self._cache[key] = value
                        self._timestamps[key] = now
                        self._evict_if_needed()

                    def cleanup(self):
                        now = time.time()
                        expired = [k for k in self._cache if self._is_expired(k, now)]
                        for k in expired:
                            del self._cache[k]
                            self._timestamps.pop(k, None)
            ''').strip(),
            "test": textwrap.dedent('''
                import time
                from lru_cache import TTLCache
                def test_basic():
                    c = TTLCache(2, 10)
                    c.put(1, 100)
                    assert c.get(1) == 100
                def test_miss():
                    c = TTLCache(2, 10)
                    assert c.get(99) == -1
                def test_expired():
                    c = TTLCache(2, 0.01)
                    c.put(1, 100)
                    time.sleep(0.05)
                    assert c.get(1) == -1
                def test_not_expired():
                    c = TTLCache(2, 10)
                    c.put(1, 100)
                    time.sleep(0.01)
                    assert c.get(1) == 100
                def test_eviction():
                    c = TTLCache(2, 100)
                    c.put(1, 10)
                    c.put(2, 20)
                    c.get(1)
                    c.put(3, 30)
                    assert c.get(2) == -1
                def test_cleanup():
                    c = TTLCache(5, 0.01)
                    c.put(1, 10)
                    c.put(2, 20)
                    time.sleep(0.05)
                    c.cleanup()
                    assert c.get(1) == -1
                    assert c.get(2) == -1
            ''').strip(),
        },
    },

    # ── Domain 2: Union-Find ──
    "union_find": {
        "module": "union_find",
        "easy": {
            "spec": (
                "Implement a simple Union-Find (Disjoint Set Union) structure. "
                "Class `UnionFind` with `__init__(n)` creating n singleton sets, "
                "`union(a, b)` merging sets containing a and b, and "
                "`connected(a, b)` returning True if a and b are in the same set."
            ),
            "stub": textwrap.dedent('''
                class UnionFind:
                    def __init__(self, n):
                        # TODO: implement
                        pass

                    def union(self, a, b):
                        # TODO: implement
                        pass

                    def connected(self, a, b):
                        # TODO: implement
                        pass
            ''').strip(),
            "solution": textwrap.dedent('''
                class UnionFind:
                    def __init__(self, n):
                        self.parent = list(range(n))

                    def _find(self, x):
                        while self.parent[x] != x:
                            x = self.parent[x]
                        return x

                    def union(self, a, b):
                        ra, rb = self._find(a), self._find(b)
                        if ra != rb:
                            self.parent[ra] = rb

                    def connected(self, a, b):
                        return self._find(a) == self._find(b)
            ''').strip(),
            "test": textwrap.dedent('''
                from union_find import UnionFind
                def test_initially_disconnected():
                    uf = UnionFind(5)
                    for i in range(5):
                        for j in range(5):
                            if i != j:
                                assert not uf.connected(i, j)
                def test_union():
                    uf = UnionFind(5)
                    uf.union(0, 1)
                    assert uf.connected(0, 1)
                    assert not uf.connected(0, 2)
                def test_transitive():
                    uf = UnionFind(5)
                    uf.union(0, 1)
                    uf.union(1, 2)
                    assert uf.connected(0, 2)
                def test_self_connected():
                    uf = UnionFind(3)
                    assert uf.connected(0, 0)
                def test_union_already_connected():
                    uf = UnionFind(3)
                    uf.union(0, 1)
                    uf.union(0, 1)
                    assert uf.connected(0, 1)
            ''').strip(),
        },
        "medium": {
            "spec": (
                "Implement Union-Find with path compression and union by rank. "
                "Class `UnionFind` with `__init__(n)`, `union(a, b)`, `connected(a, b)`, "
                "and `count()` returning the number of disjoint sets. "
                "Path compression in find, union by rank for efficiency."
            ),
            "stub": textwrap.dedent('''
                class UnionFind:
                    def __init__(self, n):
                        # TODO: implement
                        pass

                    def find(self, x):
                        # TODO: implement
                        pass

                    def union(self, a, b):
                        # TODO: implement
                        pass

                    def connected(self, a, b):
                        # TODO: implement
                        pass

                    def count(self):
                        # TODO: implement
                        pass
            ''').strip(),
            "solution": textwrap.dedent('''
                class UnionFind:
                    def __init__(self, n):
                        self.parent = list(range(n))
                        self.rank = [0] * n
                        self._count = n

                    def find(self, x):
                        if self.parent[x] != x:
                            self.parent[x] = self.find(self.parent[x])
                        return self.parent[x]

                    def union(self, a, b):
                        ra, rb = self.find(a), self.find(b)
                        if ra == rb:
                            return
                        if self.rank[ra] < self.rank[rb]:
                            self.parent[ra] = rb
                        elif self.rank[ra] > self.rank[rb]:
                            self.parent[rb] = ra
                        else:
                            self.parent[rb] = ra
                            self.rank[ra] += 1
                        self._count -= 1

                    def connected(self, a, b):
                        return self.find(a) == self.find(b)

                    def count(self):
                        return self._count
            ''').strip(),
            "test": textwrap.dedent('''
                from union_find import UnionFind
                def test_count_initial():
                    uf = UnionFind(5)
                    assert uf.count() == 5
                def test_count_after_union():
                    uf = UnionFind(5)
                    uf.union(0, 1)
                    assert uf.count() == 4
                    uf.union(2, 3)
                    assert uf.count() == 3
                def test_count_no_double():
                    uf = UnionFind(3)
                    uf.union(0, 1)
                    uf.union(0, 1)
                    assert uf.count() == 2
                def test_path_compression():
                    uf = UnionFind(6)
                    uf.union(0, 1)
                    uf.union(1, 2)
                    uf.union(2, 3)
                    uf.find(3)
                    assert uf.connected(0, 3)
                def test_union_by_rank():
                    uf = UnionFind(4)
                    uf.union(0, 1)
                    uf.union(2, 3)
                    uf.union(0, 2)
                    assert uf.connected(1, 3)
                    assert uf.count() == 1
                def test_find_returns_root():
                    uf = UnionFind(4)
                    uf.union(0, 1)
                    uf.union(1, 2)
                    assert uf.find(0) == uf.find(2)
            ''').strip(),
        },
        "hard": {
            "spec": (
                "Implement Union-Find with path compression, union by rank, and "
                "support for tracking set sizes. Class `UnionFind` with `__init__(n)`, "
                "`find(x)`, `union(a, b)`, `connected(a, b)`, `count()`, and "
                "`size(x)` returning the number of elements in the set containing x. "
                "Also implement `reset()` which resets all elements to singleton sets."
            ),
            "stub": textwrap.dedent('''
                class UnionFind:
                    def __init__(self, n):
                        # TODO: implement
                        pass

                    def find(self, x):
                        # TODO: implement
                        pass

                    def union(self, a, b):
                        # TODO: implement
                        pass

                    def connected(self, a, b):
                        # TODO: implement
                        pass

                    def count(self):
                        # TODO: implement
                        pass

                    def size(self, x):
                        # TODO: implement
                        pass

                    def reset(self):
                        # TODO: implement
                        pass
            ''').strip(),
            "solution": textwrap.dedent('''
                class UnionFind:
                    def __init__(self, n):
                        self.n = n
                        self.parent = list(range(n))
                        self.rank = [0] * n
                        self.sizes = [1] * n
                        self._count = n

                    def find(self, x):
                        if self.parent[x] != x:
                            self.parent[x] = self.find(self.parent[x])
                        return self.parent[x]

                    def union(self, a, b):
                        ra, rb = self.find(a), self.find(b)
                        if ra == rb:
                            return
                        if self.rank[ra] < self.rank[rb]:
                            ra, rb = rb, ra
                        self.parent[rb] = ra
                        self.sizes[ra] += self.sizes[rb]
                        if self.rank[ra] == self.rank[rb]:
                            self.rank[ra] += 1
                        self._count -= 1

                    def connected(self, a, b):
                        return self.find(a) == self.find(b)

                    def count(self):
                        return self._count

                    def size(self, x):
                        return self.sizes[self.find(x)]

                    def reset(self):
                        self.parent = list(range(self.n))
                        self.rank = [0] * self.n
                        self.sizes = [1] * self.n
                        self._count = self.n
            ''').strip(),
            "test": textwrap.dedent('''
                from union_find import UnionFind
                def test_size():
                    uf = UnionFind(5)
                    uf.union(0, 1)
                    uf.union(1, 2)
                    assert uf.size(0) == 3
                    assert uf.size(3) == 1
                def test_size_after_more_unions():
                    uf = UnionFind(6)
                    uf.union(0, 1)
                    uf.union(2, 3)
                    uf.union(0, 2)
                    assert uf.size(0) == 4
                    assert uf.size(4) == 1
                def test_reset():
                    uf = UnionFind(4)
                    uf.union(0, 1)
                    uf.union(2, 3)
                    assert uf.count() == 2
                    uf.reset()
                    assert uf.count() == 4
                    assert not uf.connected(0, 1)
                    assert uf.size(0) == 1
                def test_count_and_size():
                    uf = UnionFind(5)
                    uf.union(0, 1)
                    uf.union(1, 2)
                    uf.union(3, 4)
                    assert uf.count() == 2
                    assert uf.size(0) == 3
                    assert uf.size(3) == 2
                def test_size_self():
                    uf = UnionFind(3)
                    assert uf.size(0) == 1
            ''').strip(),
        },
    },

    # ── Domain 3: Sliding Window Maximum ──
    "sliding_window": {
        "module": "sliding_window",
        "easy": {
            "spec": (
                "Implement `window_sum(arr, k)` that returns a list of sums of "
                "each contiguous subarray of length k. If k > len(arr), return empty list."
            ),
            "stub": textwrap.dedent('''
                def window_sum(arr, k):
                    """Return sums of each window of size k."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def window_sum(arr, k):
                    """Return sums of each window of size k."""
                    if k <= 0 or k > len(arr):
                        return []
                    result = []
                    window_sum = sum(arr[:k])
                    result.append(window_sum)
                    for i in range(k, len(arr)):
                        window_sum += arr[i] - arr[i - k]
                        result.append(window_sum)
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from sliding_window import window_sum
                def test_basic():
                    assert window_sum([1, 2, 3, 4, 5], 3) == [6, 9, 12]
                def test_k_one():
                    assert window_sum([1, 2, 3], 1) == [1, 2, 3]
                def test_k_equals_len():
                    assert window_sum([1, 2, 3], 3) == [6]
                def test_k_too_large():
                    assert window_sum([1, 2], 5) == []
                def test_empty():
                    assert window_sum([], 3) == []
                def test_negative():
                    assert window_sum([-1, 2, -3, 4], 2) == [1, -1, 1]
            ''').strip(),
        },
        "medium": {
            "spec": (
                "Implement `sliding_window_max(arr, k)` that returns a list of the "
                "maximum values in each sliding window of size k. Must be O(n) using "
                "a deque. If k > len(arr) or k <= 0, return empty list."
            ),
            "stub": textwrap.dedent('''
                from collections import deque
                def sliding_window_max(arr, k):
                    """Return max of each sliding window of size k."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                from collections import deque
                def sliding_window_max(arr, k):
                    """Return max of each sliding window of size k."""
                    if k <= 0 or k > len(arr):
                        return []
                    dq = deque()
                    result = []
                    for i in range(len(arr)):
                        while dq and dq[0] <= i - k:
                            dq.popleft()
                        while dq and arr[dq[-1]] <= arr[i]:
                            dq.pop()
                        dq.append(i)
                        if i >= k - 1:
                            result.append(arr[dq[0]])
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from sliding_window import sliding_window_max
                def test_basic():
                    assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
                def test_k_one():
                    assert sliding_window_max([1, 2, 3], 1) == [1, 2, 3]
                def test_k_equals_len():
                    assert sliding_window_max([1, 2, 3], 3) == [3]
                def test_decreasing():
                    assert sliding_window_max([5, 4, 3, 2, 1], 2) == [5, 4, 3, 2]
                def test_increasing():
                    assert sliding_window_max([1, 2, 3, 4, 5], 3) == [3, 4, 5]
                def test_k_too_large():
                    assert sliding_window_max([1, 2], 5) == []
                def test_empty():
                    assert sliding_window_max([], 3) == []
                def test_negatives():
                    assert sliding_window_max([-1, -2, -3], 2) == [-1, -2]
            ''').strip(),
        },
        "hard": {
            "spec": (
                "Implement `sliding_window_median(arr, k)` that returns a list of "
                "medians of each sliding window of size k. For even k, return the "
                "average of the two middle elements. Use a sorted list approach or "
                "two-heap approach. If k > len(arr) or k <= 0, return empty list."
            ),
            "stub": textwrap.dedent('''
                import bisect
                def sliding_window_median(arr, k):
                    """Return median of each sliding window of size k."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                import bisect
                def sliding_window_median(arr, k):
                    """Return median of each sliding window of size k."""
                    if k <= 0 or k > len(arr):
                        return []
                    window = sorted(arr[:k])
                    result = []
                    def median(w):
                        n = len(w)
                        if n % 2 == 1:
                            return float(w[n // 2])
                        return (w[n // 2 - 1] + w[n // 2]) / 2.0
                    result.append(median(window))
                    for i in range(k, len(arr)):
                        window.remove(arr[i - k])
                        bisect.insort(window, arr[i])
                        result.append(median(window))
                    return result
            ''').strip(),
            "test": textwrap.dedent('''
                from sliding_window import sliding_window_median
                def test_odd_window():
                    assert sliding_window_median([1, 3, -1, -3, 5, 3, 6, 7], 3) == [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]
                def test_even_window():
                    result = sliding_window_median([1, 2, 3, 4], 2)
                    assert result == [1.5, 2.5, 3.5]
                def test_k_one():
                    assert sliding_window_median([5, 3, 8], 1) == [5.0, 3.0, 8.0]
                def test_k_equals_len():
                    assert sliding_window_median([1, 2, 3], 3) == [2.0]
                def test_k_too_large():
                    assert sliding_window_median([1, 2], 5) == []
                def test_empty():
                    assert sliding_window_median([], 3) == []
                def test_even_k_full():
                    assert sliding_window_median([1, 2, 3, 4], 4) == [2.5]
            ''').strip(),
        },
    },

    # ── Domain 4: Interval Scheduler ──
    "interval_scheduler": {
        "module": "interval_scheduler",
        "easy": {
            "spec": (
                "Implement `max_non_overlapping(intervals)` that takes a list of "
                "(start, end) tuples and returns the maximum number of non-overlapping "
                "intervals. Intervals [1,2] and [2,3] are considered non-overlapping."
            ),
            "stub": textwrap.dedent('''
                def max_non_overlapping(intervals):
                    """Return max count of non-overlapping intervals."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def max_non_overlapping(intervals):
                    """Return max count of non-overlapping intervals."""
                    if not intervals:
                        return 0
                    sorted_iv = sorted(intervals, key=lambda x: x[1])
                    count = 1
                    last_end = sorted_iv[0][1]
                    for start, end in sorted_iv[1:]:
                        if start >= last_end:
                            count += 1
                            last_end = end
                    return count
            ''').strip(),
            "test": textwrap.dedent('''
                from interval_scheduler import max_non_overlapping
                def test_basic():
                    assert max_non_overlapping([(1, 3), (2, 4), (3, 5)]) == 2
                def test_all_overlap():
                    assert max_non_overlapping([(1, 5), (2, 4), (3, 6)]) == 1
                def test_none_overlap():
                    assert max_non_overlapping([(1, 2), (3, 4), (5, 6)]) == 3
                def test_empty():
                    assert max_non_overlapping([]) == 0
                def test_single():
                    assert max_non_overlapping([(1, 5)]) == 1
                def test_touching():
                    assert max_non_overlapping([(1, 2), (2, 3), (3, 4)]) == 3
            ''').strip(),
        },
        "medium": {
            "spec": (
                "Implement `select_intervals(intervals)` that takes a list of "
                "(start, end, weight) tuples and returns the maximum total weight "
                "of non-overlapping intervals (weighted interval scheduling). "
                "Intervals [1,2] and [2,3] are non-overlapping. Use dynamic programming."
            ),
            "stub": textwrap.dedent('''
                def select_intervals(intervals):
                    """Return max weight of non-overlapping intervals."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                def select_intervals(intervals):
                    """Return max weight of non-overlapping intervals."""
                    if not intervals:
                        return 0
                    sorted_iv = sorted(intervals, key=lambda x: x[1])
                    n = len(sorted_iv)
                    dp = [0] * (n + 1)
                    def binary_search(i):
                        lo, hi = 0, i
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if sorted_iv[mid][1] <= sorted_iv[i][0]:
                                lo = mid + 1
                            else:
                                hi = mid
                        return lo
                    for i in range(1, n + 1):
                        start, end, weight = sorted_iv[i - 1]
                        j = binary_search(i - 1)
                        dp[i] = max(dp[i - 1], dp[j] + weight)
                    return dp[n]
            ''').strip(),
            "test": textwrap.dedent('''
                from interval_scheduler import select_intervals
                def test_basic():
                    intervals = [(1, 3, 50), (2, 4, 20), (3, 5, 30)]
                    assert select_intervals(intervals) == 80
                def test_single():
                    assert select_intervals([(1, 5, 100)]) == 100
                def test_empty():
                    assert select_intervals([]) == 0
                def test_all_non_overlap():
                    intervals = [(1, 2, 10), (2, 3, 20), (3, 4, 30)]
                    assert select_intervals(intervals) == 60
                def test_all_overlap():
                    intervals = [(1, 10, 50), (2, 9, 100), (3, 8, 30)]
                    assert select_intervals(intervals) == 100
                def test_greedy_fails():
                    intervals = [(1, 4, 10), (3, 5, 20), (4, 6, 10)]
                    assert select_intervals(intervals) == 30
            ''').strip(),
        },
        "hard": {
            "spec": (
                "Implement `schedule_rooms(intervals)` that takes a list of "
                "(start, end) tuples representing meetings and returns the minimum "
                "number of rooms needed to schedule all meetings (meeting rooms II "
                "problem). Meetings can share a room if they don't overlap. "
                "Meetings [1,2] and [2,3] can share a room."
            ),
            "stub": textwrap.dedent('''
                def schedule_rooms(intervals):
                    """Return minimum number of rooms needed."""
                    # TODO: implement
                    pass
            ''').strip(),
            "solution": textwrap.dedent('''
                import heapq
                def schedule_rooms(intervals):
                    """Return minimum number of rooms needed."""
                    if not intervals:
                        return 0
                    sorted_iv = sorted(intervals, key=lambda x: x[0])
                    rooms = []
                    heapq.heappush(rooms, sorted_iv[0][1])
                    for start, end in sorted_iv[1:]:
                        if rooms[0] <= start:
                            heapq.heappop(rooms)
                        heapq.heappush(rooms, end)
                    return len(rooms)
            ''').strip(),
            "test": textwrap.dedent('''
                from interval_scheduler import schedule_rooms
                def test_no_overlap():
                    assert schedule_rooms([(1, 2), (2, 3), (3, 4)]) == 1
                def test_all_overlap():
                    assert schedule_rooms([(1, 5), (2, 6), (3, 7)]) == 3
                def test_partial():
                    assert schedule_rooms([(0, 30), (5, 10), (15, 20)]) == 2
                def test_empty():
                    assert schedule_rooms([]) == 0
                def test_single():
                    assert schedule_rooms([(1, 5)]) == 1
                def test_back_to_back():
                    assert schedule_rooms([(1, 3), (3, 5), (1, 3)]) == 2
                def test_complex():
                    assert schedule_rooms([(1, 10), (2, 7), (3, 19), (8, 12), (10, 20), (11, 30)]) == 4
            ''').strip(),
        },
    },
}


# ── Distractor code ──

DISTRACTORS = [
    textwrap.dedent('''
        def bubble_sort(arr):
            """Bubble sort (not relevant to the task)."""
            arr = list(arr)
            n = len(arr)
            for i in range(n):
                for j in range(0, n - i - 1):
                    if arr[j] > arr[j + 1]:
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]
            return arr

        def binary_search(arr, target):
            """Binary search (not relevant to the task)."""
            lo, hi = 0, len(arr) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1
    ''').strip(),
    textwrap.dedent('''
        class Stack:
            """Simple stack (not relevant to the task)."""
            def __init__(self):
                self._items = []
            def push(self, item):
                self._items.append(item)
            def pop(self):
                return self._items.pop() if self._items else None
            def peek(self):
                return self._items[-1] if self._items else None
            def is_empty(self):
                return len(self._items) == 0

        class Queue:
            """Simple queue (not relevant to the task)."""
            def __init__(self):
                self._items = []
            def enqueue(self, item):
                self._items.append(item)
            def dequeue(self):
                return self._items.pop(0) if self._items else None
    ''').strip(),
    textwrap.dedent('''
        def quicksort(arr):
            """Quicksort (not relevant to the task)."""
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            mid = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quicksort(left) + mid + quicksort(right)

        def mergesort(arr):
            """Mergesort (not relevant to the task)."""
            if len(arr) <= 1:
                return arr
            mid = len(arr) // 2
            left = mergesort(arr[:mid])
            right = mergesort(arr[mid:])
            return merge(left, right)

        def merge(a, b):
            result = []
            i = j = 0
            while i < len(a) and j < len(b):
                if a[i] <= b[j]:
                    result.append(a[i])
                    i += 1
                else:
                    result.append(b[j])
                    j += 1
            result.extend(a[i:])
            result.extend(b[j:])
            return result
    ''').strip(),
]


@register_env
class AlgorithmImplEnv(AgenticEnv):
    name = "algorithm_impl"
    skill = "Implementing algorithms from textual specifications"
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
        variant = domain[params["difficulty"]]
        module = domain["module"]

        codebase = {f"{module}.py": variant["stub"]}

        for idx in params.get("distractor_indices", []):
            codebase[f"helper_{idx}.py"] = DISTRACTORS[idx]

        codebase["test_spec.py"] = variant["test"]
        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        module = domain["module"]

        lines = []
        lines.append("You are a software engineer implementing an algorithm from a specification.")
        lines.append("")
        lines.append("Your task is to implement the algorithm described below.")
        lines.append("The stub code is already in the codebase. You need to fill in the implementation.")
        lines.append("")
        lines.append("=== ALGORITHM SPECIFICATION ===")
        lines.append(variant["spec"])
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")
        lines.append("=== TEST CASES ===")
        lines.append("The test file `test_spec.py` contains comprehensive test cases including edge cases.")
        lines.append("Your implementation must pass all tests.")
        lines.append("")
        lines.append("Provide your implementation in the following format:")
        lines.append("<reasoning>")
        lines.append("...analyze the spec, explain the key insight, implement step by step...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append(f"```python:{module}.py")
        lines.append("# the complete implementation")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        module = domain["module"]
        return {f"{module}.py": variant["solution"]}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        module = domain["module"]

        reasoning = textwrap.dedent(f"""
            Let me carefully analyze the algorithm specification and implement it.

            SPECIFICATION:
            {variant['spec']}

            KEY INSIGHT:
            I need to understand what data structure and approach this algorithm requires.
            Let me think about the core operations and their time complexity requirements.

            Let me trace through the algorithm step by step:

            1. What are the main operations?
               - I need to identify the core operations the algorithm performs.
               - Each operation has specific complexity requirements.

            2. What data structure supports these operations efficiently?
               - I need to choose the right data structure.
               - The choice affects both correctness and performance.

            3. What are the edge cases?
               - Empty input
               - Single element
               - Boundary conditions (capacity limits, etc.)
               - Duplicate values or keys

            Let me look at the test cases to understand expected behavior:
            - The tests cover basic functionality, edge cases, and performance scenarios.
            - I need to make sure my implementation handles all of these.

            Now let me implement the algorithm:

            Step 1: Initialize the data structures needed.
            Step 2: Implement each operation according to the spec.
            Step 3: Handle edge cases explicitly.
            Step 4: Verify the implementation against the test cases.

            Let me trace through a specific test case to verify:
            - Take the first test case
            - Step through the algorithm manually
            - Check that the output matches the expected result

            I also need to check: are there distractor files in the codebase?
            I should focus only on {module}.py — the helper files contain
            unrelated algorithms that are not needed for this task.

            The implementation is complete. Let me verify once more:
            - All operations are implemented according to the spec
            - Edge cases are handled (empty input, boundary conditions)
            - The time complexity meets the requirements
        """).strip()

        return reasoning

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        module = domain["module"]

        answer = extract_answer(response)
        code_changes = parse_code_blocks(answer)

        if not code_changes:
            return 0.0, {
                "reason": "no code blocks found in answer",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        modified_codebase = apply_code_changes(codebase, code_changes)
        test_code = variant["test"]
        results = run_tests(modified_codebase, test_code, timeout=10.0)
        score, breakdown = compute_test_score(results)

        breakdown["domain"] = params["domain"]
        breakdown["difficulty"] = params["difficulty"]
        breakdown["has_reasoning"] = bool(extract_reasoning(response))
        breakdown["files_changed"] = list(code_changes.keys())

        target_file = f"{module}.py"
        breakdown["changed_target"] = target_file in code_changes

        if score == 0.0 and breakdown["changed_target"]:
            sim = code_similarity(
                code_changes.get(target_file, ""),
                variant["solution"],
            )
            if sim > 0.7:
                score = 0.25 * sim
                breakdown["partial_credit"] = f"implementation is {sim:.0%} similar, partial credit"

        return score, breakdown
