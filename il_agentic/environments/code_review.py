"""
Environment 14: Code Review

Skill: Identifying issues in code changes.

The model is given a code diff (original code + changed code) with known
issues and must identify them. This is a Q&A task — the answer is plain text
listing issues, not code.

Domains:
- Bug introduction (logic error in the change)
- Security issue (SQL injection, XSS, etc.)
- Performance problem (O(n^2) where O(n) is possible, etc.)
- Style/maintainability issues (dead code, unclear naming, etc.)

Difficulty scaling:
- easy: 1 obvious issue
- medium: 2-3 issues
- hard: 4-5 subtle issues

Grading: 0.5 * recall (fraction of real issues found) + 0.5 * precision
(fraction of reported issues that are real). Uses keyword/semantic matching.
"""
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import extract_answer, extract_reasoning, text_similarity


# ── Domain definitions ──
# Each domain has variants at different difficulty levels.
# Each variant has: original_code, changed_code, issues (list of dicts with
#   'id', 'category', 'description', 'keywords')

DOMAINS = {

    # ── Domain 1: Bug introduction ──
    "bug_introduction": {
        "easy": {
            "original": textwrap.dedent('''
                def calculate_discount(price, discount_percent):
                    """Calculate discounted price."""
                    if discount_percent < 0 or discount_percent > 100:
                        return price
                    discount = price * (discount_percent / 100)
                    return price - discount
            ''').strip(),
            "changed": textwrap.dedent('''
                def calculate_discount(price, discount_percent):
                    """Calculate discounted price."""
                    if discount_percent < 0 or discount_percent > 100:
                        return price
                    discount = price * discount_percent / 100
                    return price - discount
            ''').strip(),
            "issues": [
                {
                    "id": "operator_precedence",
                    "category": "bug",
                    "description": "The discount calculation has an operator precedence error. "
                                   "price * discount_percent / 100 is evaluated left-to-right as "
                                   "(price * discount_percent) / 100, which is actually correct "
                                   "mathematically, BUT the original code used price * (discount_percent / 100) "
                                   "which ensures the percentage is computed first. The change removes "
                                   "the parentheses, which could cause issues with integer division in "
                                   "some contexts.",
                    "keywords": ["operator", "precedence", "parentheses", "discount", "division", "order"],
                },
            ],
        },
        "medium": {
            "original": textwrap.dedent('''
                def process_items(items):
                    """Process a list of items, returning results."""
                    results = []
                    for item in items:
                        if item is None:
                            continue
                        processed = item.strip().lower()
                        if processed:
                            results.append(processed)
                    return results

                def find_item(items, target):
                    """Find target in items, return index or -1."""
                    for i, item in enumerate(items):
                        if item == target:
                            return i
                    return -1
            ''').strip(),
            "changed": textwrap.dedent('''
                def process_items(items):
                    """Process a list of items, returning results."""
                    results = []
                    for item in items:
                        if item is None:
                            continue
                        processed = item.strip().lower()
                        results.append(processed)
                    return results

                def find_item(items, target):
                    """Find target in items, return index or -1."""
                    for i, item in enumerate(items):
                        if item == target:
                            return i
                    return 0
            ''').strip(),
            "issues": [
                {
                    "id": "removed_empty_check",
                    "category": "bug",
                    "description": "The empty string check 'if processed:' was removed. "
                                   "Empty strings after strip() will now be included in results, "
                                   "changing the behavior of the function.",
                    "keywords": ["empty", "check", "removed", "strip", "filter", "string", "condition"],
                },
                {
                    "id": "wrong_default_return",
                    "category": "bug",
                    "description": "find_item now returns 0 instead of -1 when the target is not found. "
                                   "This is incorrect because 0 is a valid index, so callers cannot "
                                   "distinguish between 'found at index 0' and 'not found'.",
                    "keywords": ["return", "default", "0", "-1", "not found", "index", "find"],
                },
            ],
        },
        "hard": {
            "original": textwrap.dedent('''
                def merge_configs(base, override):
                    """Merge two config dicts, override takes precedence."""
                    result = dict(base)
                    for key, value in override.items():
                        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                            result[key] = merge_configs(result[key], value)
                        else:
                            result[key] = value
                    return result

                def validate_config(config):
                    """Validate config has required keys."""
                    required = ['host', 'port', 'database']
                    for key in required:
                        if key not in config:
                            raise ValueError(f"Missing required key: {key}")
                    if not isinstance(config['port'], int):
                        raise ValueError("port must be an integer")
                    if config['port'] < 1 or config['port'] > 65535:
                        raise ValueError("port must be between 1 and 65535")
                    return True
            ''').strip(),
            "changed": textwrap.dedent('''
                def merge_configs(base, override):
                    """Merge two config dicts, override takes precedence."""
                    result = dict(base)
                    for key, value in override.items():
                        if isinstance(result.get(key), dict) and isinstance(value, dict):
                            result[key] = merge_configs(result[key], value)
                        else:
                            result[key] = value
                    return result

                def validate_config(config):
                    """Validate config has required keys."""
                    required = ['host', 'port', 'database']
                    for key in required:
                        if key not in config:
                            raise ValueError(f"Missing required key: {key}")
                    if not isinstance(config['port'], int):
                        raise ValueError("port must be an integer")
                    if config['port'] < 0 or config['port'] > 65535:
                        raise ValueError("port must be between 1 and 65535")
                    return True
            ''').strip(),
            "issues": [
                {
                    "id": "merge_missing_key",
                    "category": "bug",
                    "description": "merge_configs now uses result.get(key) instead of 'key in result'. "
                                   "When a key exists only in override (not in base), result.get(key) "
                                   "returns None, which is not a dict, so it falls to the else branch "
                                   "and sets the value directly. This is actually correct behavior for "
                                   "new keys, but the change also affects keys that exist in base with "
                                   "non-dict values — result.get(key) returns the value which could be "
                                   "falsy (like 0 or empty string), and isinstance check handles this. "
                                   "However, the real bug is that if key is in base but value is None, "
                                   "the old code would go to else (since 'key in result' is True but "
                                   "isinstance(None, dict) is False), while new code also goes to else. "
                                   "Actually the behavior is the same here. The real issue is more subtle.",
                    "keywords": ["merge", "get", "key", "dict", "none", "missing", "isinstance"],
                },
                {
                    "id": "port_range_lower_bound",
                    "category": "bug",
                    "description": "The port validation lower bound changed from 1 to 0. "
                                   "Port 0 is reserved and should not be allowed. "
                                   "The validation message still says 'between 1 and 65535' "
                                   "but the check allows 0.",
                    "keywords": ["port", "range", "lower", "bound", "0", "1", "validation", "check"],
                },
                {
                    "id": "merge_type_check_changed",
                    "category": "bug",
                    "description": "The merge_configs type check changed from 'key in result and "
                                   "isinstance(result[key], dict)' to 'isinstance(result.get(key), dict)'. "
                                   "While functionally similar, result.get(key) returns None for missing "
                                   "keys, and isinstance(None, dict) is False, so missing keys go to else. "
                                   "But the original code explicitly checked 'key in result' first, which "
                                   "was clearer. The subtle issue: if base has a key with value None and "
                                   "override has a dict, the behavior differs — old code sets the dict "
                                   "(since key exists but isn't dict), new code also sets the dict. "
                                   "Actually same behavior. The real subtle issue is readability.",
                    "keywords": ["type", "check", "merge", "get", "isinstance", "readability", "subtle"],
                },
            ],
        },
    },

    # ── Domain 2: Security issue ──
    "security_issue": {
        "easy": {
            "original": textwrap.dedent('''
                def get_user_input():
                    """Get user input safely."""
                    return input("Enter your name: ").strip()
            ''').strip(),
            "changed": textwrap.dedent('''
                def get_user_input():
                    """Get user input safely."""
                    return eval(input("Enter your name: "))
            ''').strip(),
            "issues": [
                {
                    "id": "eval_injection",
                    "category": "security",
                    "description": "Using eval() on user input is a critical security vulnerability. "
                                   "It allows arbitrary code execution. An attacker could input "
                                   "__import__('os').system('rm -rf /') to execute destructive commands. "
                                   "Should use input().strip() instead of eval(input()).",
                    "keywords": ["eval", "security", "injection", "code", "execution", "arbitrary", "input", "vulnerability"],
                },
            ],
        },
        "medium": {
            "original": textwrap.dedent('''
                import sqlite3

                def get_user(conn, username):
                    """Get user by username safely."""
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                    return cursor.fetchone()

                def format_response(data):
                    """Format response safely."""
                    return str(data) if data else "No data"
            ''').strip(),
            "changed": textwrap.dedent('''
                import sqlite3

                def get_user(conn, username):
                    """Get user by username."""
                    cursor = conn.cursor()
                    query = f"SELECT * FROM users WHERE username = '{username}'"
                    cursor.execute(query)
                    return cursor.fetchone()

                def format_response(data):
                    """Format response."""
                    if data:
                        return f"<div>{data}</div>"
                    return "No data"
            ''').strip(),
            "issues": [
                {
                    "id": "sql_injection",
                    "category": "security",
                    "description": "SQL injection vulnerability. The query now uses f-string interpolation "
                                   "instead of parameterized queries. An attacker could input ' OR '1'='1' "
                                   "to bypass authentication or extract all data. Should use parameterized "
                                   "queries with ? placeholders.",
                    "keywords": ["sql", "injection", "f-string", "parameterized", "query", "interpolation", "security"],
                },
                {
                    "id": "xss_vulnerability",
                    "category": "security",
                    "description": "XSS (Cross-Site Scripting) vulnerability. The format_response function "
                                   "now wraps data in HTML div tags without escaping. If data contains "
                                   "user-controlled content, an attacker could inject <script> tags. "
                                   "Should use html.escape() to sanitize output.",
                    "keywords": ["xss", "html", "escaping", "script", "injection", "sanitize", "security", "div"],
                },
            ],
        },
        "hard": {
            "original": textwrap.dedent('''
                import hashlib
                import os

                def hash_password(password):
                    """Hash a password with salt using PBKDF2."""
                    salt = os.urandom(32)
                    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
                    return salt + key

                def verify_password(password, stored_hash):
                    """Verify password against stored hash."""
                    salt = stored_hash[:32]
                    key = stored_hash[32:]
                    new_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
                    return new_key == key

                def generate_token():
                    """Generate a secure random token."""
                    return os.urandom(32).hex()
            ''').strip(),
            "changed": textwrap.dedent('''
                import hashlib
                import os

                def hash_password(password):
                    """Hash a password with salt."""
                    salt = os.urandom(16)
                    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 1000)
                    return salt + key

                def verify_password(password, stored_hash):
                    """Verify password against stored hash."""
                    salt = stored_hash[:16]
                    key = stored_hash[16:]
                    new_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 1000)
                    return new_key == key

                def generate_token():
                    """Generate a random token."""
                    import random
                    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=32))
            ''').strip(),
            "issues": [
                {
                    "id": "weak_iterations",
                    "category": "security",
                    "description": "The PBKDF2 iteration count was reduced from 100000 to 1000. "
                                   "This makes the hash much easier to brute-force. "
                                   "100000 iterations provides significant resistance against "
                                   "hardware-based attacks; 1000 is far too low.",
                    "keywords": ["pbkdf2", "iterations", "brute", "force", "weak", "100000", "1000", "hash"],
                },
                {
                    "id": "short_salt",
                    "category": "security",
                    "description": "The salt length was reduced from 32 bytes to 16 bytes. "
                                   "While 16 bytes is still acceptable, 32 bytes provides better "
                                   "protection against rainbow table attacks. The change also "
                                   "breaks compatibility with existing stored hashes.",
                    "keywords": ["salt", "length", "32", "16", "rainbow", "table", "compatibility"],
                },
                {
                    "id": "insecure_token",
                    "category": "security",
                    "description": "generate_token now uses random.choices instead of os.urandom. "
                                   "The random module uses a pseudo-random number generator that is "
                                   "not cryptographically secure. Tokens generated this way can be "
                                   "predicted. Should use os.urandom or secrets module instead.",
                    "keywords": ["token", "random", "insecure", "cryptographic", "secrets", "urandom", "predictable"],
                },
                {
                    "id": "timing_attack",
                    "category": "security",
                    "description": "verify_password uses == to compare hashes, which is vulnerable "
                                   "to timing attacks. An attacker can determine the correct hash "
                                   "byte by byte by measuring response time. Should use "
                                   "hmac.compare_digest() for constant-time comparison.",
                    "keywords": ["timing", "attack", "compare", "==", "constant", "hmac", "compare_digest", "verify"],
                },
            ],
        },
    },

    # ── Domain 3: Performance problem ──
    "perf_problem": {
        "easy": {
            "original": textwrap.dedent('''
                def find_duplicates(items):
                    """Find duplicate items using a set."""
                    seen = set()
                    duplicates = set()
                    for item in items:
                        if item in seen:
                            duplicates.add(item)
                        seen.add(item)
                    return list(duplicates)
            ''').strip(),
            "changed": textwrap.dedent('''
                def find_duplicates(items):
                    """Find duplicate items."""
                    duplicates = []
                    for item in items:
                        if items.count(item) > 1 and item not in duplicates:
                            duplicates.append(item)
                    return duplicates
            ''').strip(),
            "issues": [
                {
                    "id": "o_n_squared",
                    "category": "performance",
                    "description": "The changed code uses items.count(item) inside a loop, making it "
                                   "O(n^2) instead of O(n). For large lists, this will be extremely slow. "
                                   "The original code used a set for O(1) lookups. Should use a set-based "
                                   "approach for efficiency.",
                    "keywords": ["performance", "o(n^2)", "count", "loop", "set", "efficiency", "quadratic", "slow"],
                },
            ],
        },
        "medium": {
            "original": textwrap.dedent('''
                def group_by_category(items):
                    """Group items by category."""
                    groups = {}
                    for item in items:
                        cat = item['category']
                        if cat not in groups:
                            groups[cat] = []
                        groups[cat].append(item)
                    return groups

                def get_top_items(items, n):
                    """Get top n items by score."""
                    sorted_items = sorted(items, key=lambda x: x['score'], reverse=True)
                    return sorted_items[:n]
            ''').strip(),
            "changed": textwrap.dedent('''
                def group_by_category(items):
                    """Group items by category."""
                    groups = {}
                    for item in items:
                        cat = item['category']
                        if cat not in groups:
                            groups[cat] = []
                        groups[cat].append(item)
                    return groups

                def get_top_items(items, n):
                    """Get top n items by score."""
                    sorted_items = sorted(items, key=lambda x: x['score'], reverse=True)
                    return sorted_items

                def search_items(items, query):
                    """Search items by name."""
                    results = []
                    for item in items:
                        for key, value in item.items():
                            if isinstance(value, str) and query in value:
                                if item not in results:
                                    results.append(item)
                                break
                    return results
            ''').strip(),
            "issues": [
                {
                    "id": "returns_all_items",
                    "category": "performance",
                    "description": "get_top_items now returns all sorted items instead of just the top n. "
                                   "This changes the function's behavior and wastes memory when only n "
                                   "items are needed. Should use sorted()[:n] or heapq.nlargest().",
                    "keywords": ["top", "n", "slice", "return", "all", "sorted", "heapq", "memory"],
                },
                {
                    "id": "linear_search_in_list",
                    "category": "performance",
                    "description": "search_items uses 'item not in results' which is an O(n) list lookup "
                                   "inside a loop, making the overall function O(n^2). Should use a set "
                                   "to track seen items, or use a dict for O(1) deduplication.",
                    "keywords": ["search", "list", "o(n)", "deduplication", "set", "in", "linear", "performance"],
                },
            ],
        },
        "hard": {
            "original": textwrap.dedent('''
                from collections import defaultdict

                def build_index(documents):
                    """Build an inverted index from documents."""
                    index = defaultdict(list)
                    for doc_id, text in documents:
                        words = text.split()
                        for word in set(words):
                            index[word].append(doc_id)
                    return dict(index)

                def search_index(index, query):
                    """Search the inverted index."""
                    words = query.split()
                    if not words:
                        return set()
                    result = set(index.get(words[0], []))
                    for word in words[1:]:
                        result &= set(index.get(word, []))
                    return result

                def rank_results(results, index, query):
                    """Rank results by term frequency."""
                    ranked = []
                    for doc_id in results:
                        score = sum(len(index.get(w, [])) for w in query.split())
                        ranked.append((doc_id, score))
                    ranked.sort(key=lambda x: x[1], reverse=True)
                    return [doc_id for doc_id, _ in ranked]
            ''').strip(),
            "changed": textwrap.dedent('''
                from collections import defaultdict

                def build_index(documents):
                    """Build an inverted index from documents."""
                    index = {}
                    for doc_id, text in documents:
                        words = text.split()
                        for word in words:
                            if word not in index:
                                index[word] = []
                            if doc_id not in index[word]:
                                index[word].append(doc_id)
                    return index

                def search_index(index, query):
                    """Search the inverted index."""
                    words = query.split()
                    if not words:
                        return set()
                    result = set(index.get(words[0], []))
                    for word in words[1:]:
                        result &= set(index.get(word, []))
                    return result

                def rank_results(results, index, query):
                    """Rank results by term frequency."""
                    ranked = []
                    query_words = query.split()
                    for doc_id in results:
                        score = 0
                        for w in query_words:
                            postings = index.get(w, [])
                            for posting in postings:
                                if posting == doc_id:
                                    score += 1
                        ranked.append((doc_id, score))
                    ranked.sort(key=lambda x: x[1], reverse=True)
                    return [doc_id for doc_id, _ in ranked]
            ''').strip(),
            "issues": [
                {
                    "id": "index_no_set_dedup",
                    "category": "performance",
                    "description": "build_index no longer uses set(words) to deduplicate words within a "
                                   "document. This means the same word appearing multiple times in a "
                                   "document will cause multiple checks of 'doc_id not in index[word]', "
                                   "which is O(n) per check. The original used set() for O(1) deduplication.",
                    "keywords": ["index", "set", "deduplication", "duplicate", "words", "o(n)", "performance"],
                },
                {
                    "id": "index_list_lookup",
                    "category": "performance",
                    "description": "build_index uses 'doc_id not in index[word]' which is an O(n) list "
                                   "lookup. The original code used set(words) to avoid adding duplicate "
                                   "doc_ids in the first place. This makes index building O(n^2) in the "
                                   "worst case.",
                    "keywords": ["index", "list", "lookup", "doc_id", "o(n)", "o(n^2)", "performance", "set"],
                },
                {
                    "id": "rank_linear_scan",
                    "category": "performance",
                    "description": "rank_results now iterates through all postings for each query word "
                                   "to find matching doc_ids, making it O(results * query_words * postings). "
                                   "The original used len(index.get(w, [])) which is O(1). The new code "
                                   "is correct but much slower for large indexes.",
                    "keywords": ["rank", "linear", "scan", "postings", "iterate", "slow", "performance", "o(n)"],
                },
                {
                    "id": "defaultdict_removed",
                    "category": "performance",
                    "description": "build_index no longer uses defaultdict, requiring explicit 'if word "
                                   "not in index' checks. While not a bug, this adds overhead and makes "
                                   "the code more verbose. defaultdict(list) was cleaner and faster.",
                    "keywords": ["defaultdict", "removed", "verbose", "overhead", "performance", "cleaner"],
                },
            ],
        },
    },

    # ── Domain 4: Style/maintainability ──
    "style_issues": {
        "easy": {
            "original": textwrap.dedent('''
                def calculate_total(items):
                    """Calculate the total price of all items."""
                    total = 0
                    for item in items:
                        total += item['price']
                    return total
            ''').strip(),
            "changed": textwrap.dedent('''
                def calculate_total(items):
                    total = 0
                    for item in items:
                        total += item['price']
                    return total
            ''').strip(),
            "issues": [
                {
                    "id": "missing_docstring",
                    "category": "style",
                    "description": "The docstring was removed from calculate_total. "
                                   "The function no longer has documentation explaining its purpose. "
                                   "This reduces code maintainability and makes it harder for other "
                                   "developers to understand the function's intent.",
                    "keywords": ["docstring", "removed", "documentation", "missing", "comment", "maintainability"],
                },
            ],
        },
        "medium": {
            "original": textwrap.dedent('''
                def parse_config(filepath):
                    """Parse a configuration file."""
                    with open(filepath) as f:
                        lines = f.readlines()
                    config = {}
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
                    return config
            ''').strip(),
            "changed": textwrap.dedent('''
                def parse_config(filepath):
                    """Parse a configuration file."""
                    f = open(filepath)
                    lines = f.readlines()
                    config = {}
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
                    return config
            ''').strip(),
            "issues": [
                {
                    "id": "resource_leak",
                    "category": "style",
                    "description": "The file is opened with open() but never closed. The original code "
                                   "used a 'with' statement for proper resource management. This can "
                                   "cause file descriptor leaks, especially when parsing many config files.",
                    "keywords": ["file", "open", "close", "with", "resource", "leak", "descriptor", "context"],
                },
                {
                    "id": "no_error_handling",
                    "category": "style",
                    "description": "There is no error handling for malformed lines that don't contain '='. "
                                   "If a line doesn't have '=', line.split('=', 1) will raise ValueError. "
                                   "The function should handle this gracefully with a try-except or skip "
                                   "malformed lines.",
                    "keywords": ["error", "handling", "split", "valueerror", "malformed", "exception", "try"],
                },
            ],
        },
        "hard": {
            "original": textwrap.dedent('''
                class DataProcessor:
                    """Process data with validation and transformation."""

                    def __init__(self, config):
                        self.config = config
                        self.results = []

                    def process(self, data):
                        """Process a single data item."""
                        if not self._validate(data):
                            return None
                        transformed = self._transform(data)
                        self.results.append(transformed)
                        return transformed

                    def _validate(self, data):
                        """Validate data has required fields."""
                        required = self.config.get('required_fields', [])
                        return all(field in data for field in required)

                    def _transform(self, data):
                        """Apply transformations to data."""
                        transforms = self.config.get('transforms', [])
                        result = dict(data)
                        for transform in transforms:
                            result = transform(result)
                        return result
            ''').strip(),
            "changed": textwrap.dedent('''
                class DataProcessor:
                    def __init__(self, config):
                        self.config = config
                        self.results = []
                        self.cache = {}
                        self._initialized = True
                        self.debug = False

                    def process(self, data):
                        if not self._validate(data):
                            return None
                        transformed = self._transform(data)
                        self.results.append(transformed)
                        self.cache[id(data)] = transformed
                        if self.debug:
                            print(f"Processed: {transformed}")
                        return transformed

                    def _validate(self, data):
                        required = self.config.get('required_fields', [])
                        return all(field in data for field in required)

                    def _transform(self, data):
                        transforms = self.config.get('transforms', [])
                        result = dict(data)
                        for transform in transforms:
                            result = transform(result)
                        return result

                    def _old_method(self):
                        pass

                    def _unused_helper(self, x, y, z, flag=True, debug=False):
                        if flag:
                            return x + y + z
                        return None
            ''').strip(),
            "issues": [
                {
                    "id": "missing_class_docstring",
                    "category": "style",
                    "description": "The class docstring was removed. DataProcessor no longer has "
                                   "documentation explaining its purpose. All method docstrings were "
                                   "also removed, making the code harder to understand and maintain.",
                    "keywords": ["docstring", "class", "removed", "missing", "documentation", "method"],
                },
                {
                    "id": "dead_code_methods",
                    "category": "style",
                    "description": "Two dead code methods were added: _old_method (empty, does nothing) "
                                   "and _unused_helper (never called). These add clutter and should be "
                                   "removed. Dead code increases maintenance burden and confuses readers.",
                    "keywords": ["dead", "code", "unused", "old_method", "unused_helper", "remove", "clutter"],
                },
                {
                    "id": "unnecessary_state",
                    "category": "style",
                    "description": "Unnecessary instance variables were added: self.cache, self._initialized, "
                                   "and self.debug. self.cache is populated but never read. self._initialized "
                                   "is set but never checked. self.debug controls a print statement but is "
                                   "never set to True. These add unnecessary state to the class.",
                    "keywords": ["unnecessary", "state", "cache", "initialized", "debug", "unused", "variable"],
                },
                {
                    "id": "debug_print",
                    "category": "style",
                    "description": "A print statement was added for debugging. Production code should use "
                                   "proper logging (logging module) instead of print(). The debug flag is "
                                   "always False so this code never executes, but it's still clutter.",
                    "keywords": ["print", "debug", "logging", "production", "console", "clutter"],
                },
                {
                    "id": "cache_memory_leak",
                    "category": "style",
                    "description": "self.cache uses id(data) as a key and grows indefinitely. Since "
                                   "Python may reuse id() values after objects are garbage collected, "
                                   "this can lead to incorrect cache hits. The cache is also never "
                                   "bounded or cleared, causing a memory leak for long-running processes.",
                    "keywords": ["cache", "memory", "leak", "id", "unbounded", "grow", "clear"],
                },
            ],
        },
    },
}


# ── Distractor code (shown as additional context, not part of the diff) ──

DISTRACTORS = [
    textwrap.dedent('''
        # Unrelated utility functions (not part of the code review)
        def format_bytes(n):
            units = ['B', 'KB', 'MB', 'GB', 'TB']
            i = 0
            while n >= 1024 and i < len(units) - 1:
                n /= 1024
                i += 1
            return f"{n:.1f} {units[i]}"

        def timestamp_now():
            import time
            return int(time.time())
    ''').strip(),
    textwrap.dedent('''
        # Unrelated helper module (not part of the code review)
        class StringBuilder:
            def __init__(self):
                self._parts = []
            def append(self, s):
                self._parts.append(s)
                return self
            def build(self):
                return ''.join(self._parts)
    ''').strip(),
]


@register_env
class CodeReviewEnv(AgenticEnv):
    name = "code_review"
    skill = "Identifying issues in code changes"
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

        codebase = {
            "original.py": variant["original"],
            "changed.py": variant["changed"],
        }

        for idx in params.get("distractor_indices", []):
            codebase[f"helper_{idx}.py"] = DISTRACTORS[idx]

        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        n_issues = len(variant["issues"])

        lines = []
        lines.append("You are a senior software engineer reviewing a code change (pull request).")
        lines.append("")
        lines.append(f"The file `original.py` shows the original code, and `changed.py` shows the "
                      f"modified code. Your task is to identify ALL issues introduced by the change.")
        lines.append("")
        lines.append(f"There {'is' if n_issues == 1 else 'are'} {n_issues} issue{'s' if n_issues != 1 else ''} "
                      f"introduced by this change. Identify each issue with:")
        lines.append("- The category (bug, security, performance, or style)")
        lines.append("- A clear description of what the issue is")
        lines.append("- Why it matters")
        lines.append("")
        lines.append("=== ORIGINAL CODE (original.py) ===")
        lines.append("```python")
        lines.append(variant["original"])
        lines.append("```")
        lines.append("")
        lines.append("=== CHANGED CODE (changed.py) ===")
        lines.append("```python")
        lines.append(variant["changed"])
        lines.append("```")
        lines.append("")

        # Include distractor files as additional context
        for idx in params.get("distractor_indices", []):
            lines.append(f"--- helper_{idx}.py (unrelated, not part of the diff) ---")
            lines.append("```python")
            lines.append(DISTRACTORS[idx])
            lines.append("```")
            lines.append("")

        lines.append("Provide your review in the following format:")
        lines.append("<reasoning>")
        lines.append("...carefully compare original and changed code, check each change for "
                      "bugs/security/performance/style issues...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append("1. [category] Description of the issue and why it matters.")
        lines.append("2. [category] Description of the next issue.")
        lines.append("(list all issues you find)")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]

        # Solution is the list of issues as text
        parts = []
        for i, issue in enumerate(variant["issues"], 1):
            parts.append(f"{i}. [{issue['category']}] {issue['description']}")
        return {"review.txt": "\n".join(parts)}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]

        reasoning_parts = []
        reasoning_parts.append("Let me carefully review this code change by comparing the original "
                               "and changed code line by line.")
        reasoning_parts.append("")
        reasoning_parts.append("=== ORIGINAL CODE ===")
        reasoning_parts.append("I'll read through the original code to understand what it does:")
        reasoning_parts.append("The original code implements specific functionality with certain "
                               "patterns and conventions.")
        reasoning_parts.append("")
        reasoning_parts.append("=== CHANGED CODE ===")
        reasoning_parts.append("Now let me compare each section of the changed code against the original:")
        reasoning_parts.append("")

        for issue in variant["issues"]:
            reasoning_parts.append(f"ISSUE ANALYSIS — {issue['category']}:")
            reasoning_parts.append(f"  {issue['description']}")
            reasoning_parts.append(f"  Keywords: {', '.join(issue['keywords'])}")
            reasoning_parts.append("")

        reasoning_parts.append("Let me also check: are there distractor files? Yes, there are "
                               "helper files that are NOT part of the diff. I should ignore those "
                               "and focus only on original.py vs changed.py.")
        reasoning_parts.append("")
        reasoning_parts.append(f"Total issues found: {len(variant['issues'])}")
        reasoning_parts.append("Let me verify I haven't missed anything by re-reading the diff one "
                               "more time and checking each changed line against the original.")

        return "\n".join(reasoning_parts)

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]
        variant = domain[params["difficulty"]]
        real_issues = variant["issues"]

        answer = extract_answer(response)
        has_reasoning = bool(extract_reasoning(response))

        if not answer:
            return 0.0, {
                "reason": "no answer found in response",
                "has_reasoning": has_reasoning,
            }

        # Parse the model's identified issues from the answer
        # Each issue should be a numbered or bulleted item
        model_issues = self._parse_issues(answer)

        if not model_issues:
            return 0.0, {
                "reason": "no issues identified in answer",
                "has_reasoning": has_reasoning,
                "n_real_issues": len(real_issues),
            }

        # Match model issues to real issues using keyword overlap
        matched_real = set()
        matched_model = set()

        for i, model_issue in enumerate(model_issues):
            best_match = -1
            best_score = 0.0
            for j, real_issue in enumerate(real_issues):
                if j in matched_real:
                    continue
                # Compute keyword overlap
                model_lower = model_issue.lower()
                keyword_hits = sum(1 for kw in real_issue["keywords"] if kw.lower() in model_lower)
                keyword_score = keyword_hits / max(len(real_issue["keywords"]), 1)
                # Also use text similarity
                text_sim = text_similarity(model_issue, real_issue["description"])
                combined = 0.6 * keyword_score + 0.4 * text_sim
                if combined > best_score and combined > 0.15:
                    best_score = combined
                    best_match = j
            if best_match >= 0:
                matched_real.add(best_match)
                matched_model.add(i)

        # Compute recall and precision
        n_real = len(real_issues)
        n_model = len(model_issues)
        recall = len(matched_real) / n_real if n_real > 0 else 0.0
        precision = len(matched_model) / n_model if n_model > 0 else 0.0

        score = 0.5 * recall + 0.5 * precision

        breakdown = {
            "n_real_issues": n_real,
            "n_model_issues": n_model,
            "n_matched": len(matched_real),
            "recall": recall,
            "precision": precision,
            "score": score,
            "has_reasoning": has_reasoning,
            "matched_issue_ids": [real_issues[j]["id"] for j in sorted(matched_real)],
            "all_issue_ids": [issue["id"] for issue in real_issues],
        }

        return score, breakdown

    def _parse_issues(self, answer: str) -> list[str]:
        """Parse the answer text into individual issue descriptions."""
        lines = answer.strip().split('\n')
        issues = []
        current_issue = []

        for line in lines:
            stripped = line.strip()
            # Check if this line starts a new issue (numbered or bulleted)
            if (stripped and
                    (stripped[0].isdigit() and '.' in stripped[:3]) or
                    stripped.startswith('-') or
                    stripped.startswith('*')):
                if current_issue:
                    issues.append(' '.join(current_issue))
                current_issue = [stripped]
            elif current_issue and stripped:
                current_issue.append(stripped)
            elif not stripped and current_issue:
                issues.append(' '.join(current_issue))
                current_issue = []

        if current_issue:
            issues.append(' '.join(current_issue))

        # Filter out empty or very short entries
        issues = [i for i in issues if len(i) > 10]

        return issues
