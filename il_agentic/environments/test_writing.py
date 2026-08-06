"""
Environment 4: Test Writing

Skill: Writing effective tests that catch bugs.

The model is given a function with correct behavior and must write tests for it.
The grader runs the model's tests against the correct version (must pass) and
against N mutated (buggy) versions (should fail). Score = mutation kill rate.

If tests fail on the correct version, score = 0 (tests are wrong).

Difficulty scaling:
- easy: single function, simple behavior, 3 mutants
- medium: single function with edge cases, 4-5 mutants
- hard: multi-function module with distractors, 5-6 mutants
"""
import random
import textwrap
import re
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, parse_code_blocks, apply_code_changes,
    run_tests, extract_reasoning, CodeExecutor,
)


# ── Domain definitions ──
# Each domain has:
#   - "module": the module name
#   - "correct": the correct implementation
#   - "spec": description of what the function does (shown to model)
#   - "mutants": list of buggy versions (each is a full module content)
#   - "distractor_note": whether to include distractor functions

DOMAINS = {
    "password_validator": {
        "module": "password",
        "correct": textwrap.dedent('''
            def validate_password(password):
                """Validate a password. Returns list of error messages (empty = valid).

                Rules:
                - At least 8 characters long
                - Contains at least one uppercase letter
                - Contains at least one lowercase letter
                - Contains at least one digit
                - Contains at least one special character from !@#$%^&*
                """
                errors = []
                if len(password) < 8:
                    errors.append("Password must be at least 8 characters")
                if not any(c.isupper() for c in password):
                    errors.append("Password must contain an uppercase letter")
                if not any(c.islower() for c in password):
                    errors.append("Password must contain a lowercase letter")
                if not any(c.isdigit() for c in password):
                    errors.append("Password must contain a digit")
                if not any(c in "!@#$%^&*" for c in password):
                    errors.append("Password must contain a special character")
                return errors
        ''').strip(),
        "spec": (
            "validate_password(password) - Validates a password and returns a list of "
            "error messages. An empty list means the password is valid.\n"
            "Rules: at least 8 chars, at least one uppercase, one lowercase, one digit, "
            "and one special character from the set !@#$%^&*"
        ),
        "mutants": [
            textwrap.dedent('''
                def validate_password(password):
                    errors = []
                    if len(password) < 8:
                        errors.append("Password must be at least 8 characters")
                    if not any(c.isupper() for c in password):
                        errors.append("Password must contain an uppercase letter")
                    if not any(c.islower() for c in password):
                        errors.append("Password must contain a lowercase letter")
                    if not any(c.isdigit() for c in password):
                        errors.append("Password must contain a digit")
                    # BUG: removed special character check
                    return errors
            ''').strip(),
            textwrap.dedent('''
                def validate_password(password):
                    errors = []
                    if len(password) < 7:  # BUG: should be 8
                        errors.append("Password must be at least 8 characters")
                    if not any(c.isupper() for c in password):
                        errors.append("Password must contain an uppercase letter")
                    if not any(c.islower() for c in password):
                        errors.append("Password must contain a lowercase letter")
                    if not any(c.isdigit() for c in password):
                        errors.append("Password must contain a digit")
                    if not any(c in "!@#$%^&*" for c in password):
                        errors.append("Password must contain a special character")
                    return errors
            ''').strip(),
            textwrap.dedent('''
                def validate_password(password):
                    errors = []
                    if len(password) < 8:
                        errors.append("Password must be at least 8 characters")
                    if not any(c.isupper() for c in password):
                        errors.append("Password must contain an uppercase letter")
                    if not any(c.islower() for c in password):
                        errors.append("Password must contain a lowercase letter")
                    if not any(c.isdigit() for c in password):
                        errors.append("Password must contain a digit")
                    if not any(c in "!@#$%^&*abcdef" for c in password):  # BUG: extra chars
                        errors.append("Password must contain a special character")
                    return errors
            ''').strip(),
            textwrap.dedent('''
                def validate_password(password):
                    errors = []
                    if len(password) <= 8:  # BUG: should be < 8, not <= 8
                        errors.append("Password must be at least 8 characters")
                    if not any(c.isupper() for c in password):
                        errors.append("Password must contain an uppercase letter")
                    if not any(c.islower() for c in password):
                        errors.append("Password must contain a lowercase letter")
                    if not any(c.isdigit() for c in password):
                        errors.append("Password must contain a digit")
                    if not any(c in "!@#$%^&*" for c in password):
                        errors.append("Password must contain a special character")
                    return errors
            ''').strip(),
            textwrap.dedent('''
                def validate_password(password):
                    errors = []
                    if len(password) < 8:
                        errors.append("Password must be at least 8 characters")
                    if not any(c.isupper() for c in password):
                        errors.append("Password must contain an uppercase letter")
                    # BUG: removed lowercase check
                    if not any(c.isdigit() for c in password):
                        errors.append("Password must contain a digit")
                    if not any(c in "!@#$%^&*" for c in password):
                        errors.append("Password must contain a special character")
                    return errors
            ''').strip(),
        ],
    },
    "date_parser": {
        "module": "dateparser",
        "correct": textwrap.dedent('''
            def parse_date(date_str):
                """Parse a date string in YYYY-MM-DD format.

                Returns a dict with 'year', 'month', 'day' keys.
                Raises ValueError for invalid formats or impossible dates.
                """
                parts = date_str.split("-")
                if len(parts) != 3:
                    raise ValueError("Date must be in YYYY-MM-DD format")
                try:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    raise ValueError("Date components must be integers")
                if month < 1 or month > 12:
                    raise ValueError("Month must be between 1 and 12")
                if day < 1 or day > 31:
                    raise ValueError("Day must be between 1 and 31")
                # Check days in month
                days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                # Leap year check
                if month == 2:
                    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                        days_in_month[2] = 29
                if day > days_in_month[month]:
                    raise ValueError(f"Invalid day {day} for month {month}")
                return {"year": year, "month": month, "day": day}
        ''').strip(),
        "spec": (
            "parse_date(date_str) - Parse a date string in YYYY-MM-DD format.\n"
            "Returns a dict with 'year', 'month', 'day' keys (all integers).\n"
            "Raises ValueError for: wrong format, non-integer components, "
            "month not in 1-12, day not in 1-31, or day exceeding days in that month "
            "(including leap year handling for February)."
        ),
        "mutants": [
            textwrap.dedent('''
                def parse_date(date_str):
                    parts = date_str.split("-")
                    if len(parts) != 3:
                        raise ValueError("Date must be in YYYY-MM-DD format")
                    try:
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    except ValueError:
                        raise ValueError("Date components must be integers")
                    if month < 1 or month > 12:
                        raise ValueError("Month must be between 1 and 12")
                    if day < 1 or day > 31:  # BUG: doesn't check days per month
                        raise ValueError("Day must be between 1 and 31")
                    return {"year": year, "month": month, "day": day}
            ''').strip(),
            textwrap.dedent('''
                def parse_date(date_str):
                    parts = date_str.split("-")
                    if len(parts) != 3:
                        raise ValueError("Date must be in YYYY-MM-DD format")
                    try:
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    except ValueError:
                        raise ValueError("Date components must be integers")
                    if month < 0 or month > 12:  # BUG: allows month 0
                        raise ValueError("Month must be between 1 and 12")
                    if day < 1 or day > 31:
                        raise ValueError("Day must be between 1 and 31")
                    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                    if month == 2:
                        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                            days_in_month[2] = 29
                    if day > days_in_month[month]:
                        raise ValueError(f"Invalid day {day} for month {month}")
                    return {"year": year, "month": month, "day": day}
            ''').strip(),
            textwrap.dedent('''
                def parse_date(date_str):
                    parts = date_str.split("-")
                    if len(parts) != 3:
                        raise ValueError("Date must be in YYYY-MM-DD format")
                    try:
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    except ValueError:
                        raise ValueError("Date components must be integers")
                    if month < 1 or month > 12:
                        raise ValueError("Month must be between 1 and 12")
                    if day < 1 or day > 31:
                        raise ValueError("Day must be between 1 and 31")
                    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                    # BUG: no leap year check - Feb always 28
                    if day > days_in_month[month]:
                        raise ValueError(f"Invalid day {day} for month {month}")
                    return {"year": year, "month": month, "day": day}
            ''').strip(),
            textwrap.dedent('''
                def parse_date(date_str):
                    parts = date_str.split("/")
                    if len(parts) != 3:  # BUG: uses / instead of -
                        raise ValueError("Date must be in YYYY-MM-DD format")
                    try:
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    except ValueError:
                        raise ValueError("Date components must be integers")
                    if month < 1 or month > 12:
                        raise ValueError("Month must be between 1 and 12")
                    if day < 1 or day > 31:
                        raise ValueError("Day must be between 1 and 31")
                    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                    if month == 2:
                        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                            days_in_month[2] = 29
                    if day > days_in_month[month]:
                        raise ValueError(f"Invalid day {day} for month {month}")
                    return {"year": year, "month": month, "day": day}
            ''').strip(),
            textwrap.dedent('''
                def parse_date(date_str):
                    parts = date_str.split("-")
                    if len(parts) != 3:
                        raise ValueError("Date must be in YYYY-MM-DD format")
                    try:
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    except ValueError:
                        raise ValueError("Date components must be integers")
                    if month < 1 or month > 12:
                        raise ValueError("Month must be between 1 and 12")
                    if day < 0 or day > 31:  # BUG: allows day 0
                        raise ValueError("Day must be between 1 and 31")
                    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                    if month == 2:
                        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                            days_in_month[2] = 29
                    if day > days_in_month[month]:
                        raise ValueError(f"Invalid day {day} for month {month}")
                    return {"year": year, "month": month, "day": day}
            ''').strip(),
        ],
    },
    "url_parser": {
        "module": "urlparser",
        "correct": textwrap.dedent('''
            def parse_url(url):
                """Parse a URL into its components.

                Returns dict with keys: 'scheme', 'host', 'port', 'path', 'query', 'fragment'.
                Missing components have default values:
                - scheme: None if not present
                - host: None if not present
                - port: None if not present (default port)
                - path: '/' if not present
                - query: None if not present
                - fragment: None if not present

                Raises ValueError if no host is present after scheme.
                """
                result = {"scheme": None, "host": None, "port": None,
                          "path": "/", "query": None, "fragment": None}

                # Extract fragment
                if "#" in url:
                    url, fragment = url.split("#", 1)
                    result["fragment"] = fragment

                # Extract query
                if "?" in url:
                    url, query = url.split("?", 1)
                    result["query"] = query

                # Extract scheme
                if "://" in url:
                    scheme, rest = url.split("://", 1)
                    result["scheme"] = scheme
                    url = rest

                # Extract path
                if "/" in url:
                    host_part, path = url.split("/", 1)
                    result["path"] = "/" + path
                    url = host_part

                # Extract port
                if ":" in url:
                    host, port_str = url.rsplit(":", 1)
                    try:
                        result["port"] = int(port_str)
                    except ValueError:
                        raise ValueError(f"Invalid port: {port_str}")
                    result["host"] = host
                else:
                    result["host"] = url if url else None

                if not result["host"]:
                    raise ValueError("URL must have a host")

                return result
        ''').strip(),
        "spec": (
            "parse_url(url) - Parse a URL string into components.\n"
            "Returns a dict with keys: 'scheme', 'host', 'port', 'path', 'query', 'fragment'.\n"
            "Defaults: scheme=None, host=None, port=None, path='/', query=None, fragment=None.\n"
            "Raises ValueError if no host is present, or if port is not a valid integer."
        ),
        "mutants": [
            textwrap.dedent('''
                def parse_url(url):
                    result = {"scheme": None, "host": None, "port": None,
                              "path": "/", "query": None, "fragment": None}
                    if "#" in url:
                        url, fragment = url.split("#", 1)
                        result["fragment"] = fragment
                    if "?" in url:
                        url, query = url.split("?", 1)
                        result["query"] = query
                    if "://" in url:
                        scheme, rest = url.split("://", 1)
                        result["scheme"] = scheme
                        url = rest
                    if "/" in url:
                        host_part, path = url.split("/", 1)
                        result["path"] = "/" + path
                        url = host_part
                    if ":" in url:
                        host, port_str = url.rsplit(":", 1)
                        try:
                            result["port"] = int(port_str)
                        except ValueError:
                            raise ValueError(f"Invalid port: {port_str}")
                        result["host"] = host
                    else:
                        result["host"] = url if url else None
                    # BUG: removed host validation
                    return result
            ''').strip(),
            textwrap.dedent('''
                def parse_url(url):
                    result = {"scheme": None, "host": None, "port": None,
                              "path": "/", "query": None, "fragment": None}
                    if "#" in url:
                        url, fragment = url.split("#", 1)
                        result["fragment"] = fragment
                    if "?" in url:
                        url, query = url.split("?", 1)
                        result["query"] = query
                    if "://" in url:
                        scheme, rest = url.split("://", 1)
                        result["scheme"] = scheme
                        url = rest
                    if "/" in url:
                        host_part, path = url.split("/", 1)
                        result["path"] = path  # BUG: missing leading /
                        url = host_part
                    if ":" in url:
                        host, port_str = url.rsplit(":", 1)
                        try:
                            result["port"] = int(port_str)
                        except ValueError:
                            raise ValueError(f"Invalid port: {port_str}")
                        result["host"] = host
                    else:
                        result["host"] = url if url else None
                    if not result["host"]:
                        raise ValueError("URL must have a host")
                    return result
            ''').strip(),
            textwrap.dedent('''
                def parse_url(url):
                    result = {"scheme": None, "host": None, "port": None,
                              "path": "/", "query": None, "fragment": None}
                    if "#" in url:
                        url, fragment = url.split("#", 1)
                        result["fragment"] = fragment
                    # BUG: query extracted after path (wrong order)
                    if "://" in url:
                        scheme, rest = url.split("://", 1)
                        result["scheme"] = scheme
                        url = rest
                    if "/" in url:
                        host_part, path = url.split("/", 1)
                        result["path"] = "/" + path
                        url = host_part
                    if "?" in url:
                        url, query = url.split("?", 1)
                        result["query"] = query
                    if ":" in url:
                        host, port_str = url.rsplit(":", 1)
                        try:
                            result["port"] = int(port_str)
                        except ValueError:
                            raise ValueError(f"Invalid port: {port_str}")
                        result["host"] = host
                    else:
                        result["host"] = url if url else None
                    if not result["host"]:
                        raise ValueError("URL must have a host")
                    return result
            ''').strip(),
            textwrap.dedent('''
                def parse_url(url):
                    result = {"scheme": None, "host": None, "port": None,
                              "path": "/", "query": None, "fragment": None}
                    if "#" in url:
                        url, fragment = url.split("#", 1)
                        result["fragment"] = fragment
                    if "?" in url:
                        url, query = url.split("?", 1)
                        result["query"] = query
                    if "://" in url:
                        scheme, rest = url.split("://", 1)
                        result["scheme"] = scheme
                        url = rest
                    if "/" in url:
                        host_part, path = url.split("/", 1)
                        result["path"] = "/" + path
                        url = host_part
                    if ":" in url:
                        host, port_str = url.rsplit(":", 1)
                        # BUG: doesn't validate port is integer
                        result["port"] = port_str
                        result["host"] = host
                    else:
                        result["host"] = url if url else None
                    if not result["host"]:
                        raise ValueError("URL must have a host")
                    return result
            ''').strip(),
            textwrap.dedent('''
                def parse_url(url):
                    result = {"scheme": None, "host": None, "port": None,
                              "path": "/", "query": None, "fragment": None}
                    if "#" in url:
                        url, fragment = url.split("#", 1)
                        result["fragment"] = fragment
                    if "?" in url:
                        url, query = url.split("?", 1)
                        result["query"] = query
                    if "://" in url:
                        scheme, rest = url.split("://", 1)
                        result["scheme"] = scheme
                        url = rest
                    if "/" in url:
                        host_part, path = url.split("/", 1)
                        result["path"] = "/" + path
                        url = host_part
                    if ":" in url:
                        host, port_str = url.rsplit(":", 1)
                        try:
                            result["port"] = int(port_str)
                        except ValueError:
                            raise ValueError(f"Invalid port: {port_str}")
                        result["host"] = host
                    else:
                        result["host"] = url if url else None
                    # BUG: empty string host is allowed (should raise)
                    if result["host"] is None:
                        raise ValueError("URL must have a host")
                    return result
            ''').strip(),
        ],
    },
    "poker_evaluator": {
        "module": "poker",
        "correct": textwrap.dedent('''
            def evaluate_hand(cards):
                """Evaluate a 5-card poker hand.

                Args:
                    cards: list of (rank, suit) tuples where rank is 2-14 (14=Ace)
                           and suit is a string.

                Returns:
                    Hand rank as integer (higher is better):
                    9: Straight flush, 8: Four of a kind, 7: Full house,
                    6: Flush, 5: Straight, 4: Three of a kind, 3: Two pair,
                    2: One pair, 1: High card
                """
                ranks = sorted([c[0] for c in cards], reverse=True)
                suits = [c[1] for c in cards]

                is_flush = len(set(suits)) == 1

                # Check straight (including wheel: A-2-3-4-5)
                unique_ranks = sorted(set(ranks), reverse=True)
                is_straight = False
                if len(unique_ranks) == 5:
                    if unique_ranks[0] - unique_ranks[4] == 4:
                        is_straight = True
                    elif unique_ranks == [14, 5, 4, 3, 2]:
                        is_straight = True

                # Count rank occurrences
                from collections import Counter
                rank_counts = Counter(ranks)
                counts = sorted(rank_counts.values(), reverse=True)

                if is_straight and is_flush:
                    return 9
                if counts == [4, 1]:
                    return 8
                if counts == [3, 2]:
                    return 7
                if is_flush:
                    return 6
                if is_straight:
                    return 5
                if counts == [3, 1, 1]:
                    return 4
                if counts == [2, 2, 1]:
                    return 3
                if counts == [2, 1, 1, 1]:
                    return 2
                return 1
        ''').strip(),
        "spec": (
            "evaluate_hand(cards) - Evaluate a 5-card poker hand.\n"
            "Input: list of (rank, suit) tuples, rank is 2-14 (14=Ace), suit is a string.\n"
            "Returns hand rank as integer: 9=Straight flush, 8=Four of a kind, "
            "7=Full house, 6=Flush, 5=Straight, 4=Three of a kind, 3=Two pair, "
            "2=One pair, 1=High card.\n"
            "Note: A-2-3-4-5 (the 'wheel') is a valid straight."
        ),
        "mutants": [
            textwrap.dedent('''
                def evaluate_hand(cards):
                    ranks = sorted([c[0] for c in cards], reverse=True)
                    suits = [c[1] for c in cards]
                    is_flush = len(set(suits)) == 1
                    unique_ranks = sorted(set(ranks), reverse=True)
                    is_straight = False
                    if len(unique_ranks) == 5:
                        if unique_ranks[0] - unique_ranks[4] == 4:
                            is_straight = True
                        # BUG: missing wheel straight check
                    from collections import Counter
                    rank_counts = Counter(ranks)
                    counts = sorted(rank_counts.values(), reverse=True)
                    if is_straight and is_flush:
                        return 9
                    if counts == [4, 1]:
                        return 8
                    if counts == [3, 2]:
                        return 7
                    if is_flush:
                        return 6
                    if is_straight:
                        return 5
                    if counts == [3, 1, 1]:
                        return 4
                    if counts == [2, 2, 1]:
                        return 3
                    if counts == [2, 1, 1, 1]:
                        return 2
                    return 1
            ''').strip(),
            textwrap.dedent('''
                def evaluate_hand(cards):
                    ranks = sorted([c[0] for c in cards], reverse=True)
                    suits = [c[1] for c in cards]
                    is_flush = len(set(suits)) == 1
                    unique_ranks = sorted(set(ranks), reverse=True)
                    is_straight = False
                    if len(unique_ranks) == 5:
                        if unique_ranks[0] - unique_ranks[4] == 4:
                            is_straight = True
                        elif unique_ranks == [14, 5, 4, 3, 2]:
                            is_straight = True
                    from collections import Counter
                    rank_counts = Counter(ranks)
                    counts = sorted(rank_counts.values(), reverse=True)
                    if is_straight and is_flush:
                        return 9
                    if counts == [4, 1]:
                        return 8
                    if counts == [3, 2]:
                        return 7
                    if is_flush:
                        return 6
                    if is_straight:
                        return 5
                    if counts == [3, 1, 1]:
                        return 4
                    if counts == [2, 2, 1]:
                        return 3
                    if counts == [2, 1, 1, 1]:
                        return 2
                    return 2  # BUG: high card returns 2 (one pair) instead of 1
            ''').strip(),
            textwrap.dedent('''
                def evaluate_hand(cards):
                    ranks = sorted([c[0] for c in cards], reverse=True)
                    suits = [c[1] for c in cards]
                    is_flush = len(set(suits)) == 1
                    unique_ranks = sorted(set(ranks), reverse=True)
                    is_straight = False
                    if len(unique_ranks) == 5:
                        if unique_ranks[0] - unique_ranks[4] == 4:
                            is_straight = True
                        elif unique_ranks == [14, 5, 4, 3, 2]:
                            is_straight = True
                    from collections import Counter
                    rank_counts = Counter(ranks)
                    counts = sorted(rank_counts.values(), reverse=True)
                    if is_straight and is_flush:
                        return 9
                    if counts == [4, 1]:
                        return 8
                    if counts == [3, 2]:
                        return 7
                    if is_flush:
                        return 6
                    if is_straight:
                        return 5
                    if counts == [3, 1, 1]:
                        return 4
                    if counts == [2, 2, 1]:
                        return 3
                    # BUG: one pair returns 1 (high card) instead of 2
                    return 1
            ''').strip(),
            textwrap.dedent('''
                def evaluate_hand(cards):
                    ranks = sorted([c[0] for c in cards], reverse=True)
                    suits = [c[1] for c in cards]
                    is_flush = len(set(suits)) == 1
                    unique_ranks = sorted(set(ranks), reverse=True)
                    is_straight = False
                    if len(unique_ranks) == 5:
                        if unique_ranks[0] - unique_ranks[4] == 4:
                            is_straight = True
                        elif unique_ranks == [14, 5, 4, 3, 2]:
                            is_straight = True
                    from collections import Counter
                    rank_counts = Counter(ranks)
                    counts = sorted(rank_counts.values(), reverse=True)
                    if is_straight and is_flush:
                        return 9
                    if counts == [4, 1]:
                        return 8
                    if counts == [3, 2]:
                        return 7
                    if is_flush:
                        return 6
                    if is_straight:
                        return 5
                    if counts == [3, 1, 1]:
                        return 3  # BUG: returns 3 (two pair) instead of 4 (three of a kind)
                    if counts == [2, 2, 1]:
                        return 4  # BUG: returns 4 (three of a kind) instead of 3 (two pair)
                    if counts == [2, 1, 1, 1]:
                        return 2
                    return 1
            ''').strip(),
            textwrap.dedent('''
                def evaluate_hand(cards):
                    ranks = sorted([c[0] for c in cards], reverse=True)
                    suits = [c[1] for c in cards]
                    is_flush = len(suits) == 5  # BUG: always True (5 cards)
                    unique_ranks = sorted(set(ranks), reverse=True)
                    is_straight = False
                    if len(unique_ranks) == 5:
                        if unique_ranks[0] - unique_ranks[4] == 4:
                            is_straight = True
                        elif unique_ranks == [14, 5, 4, 3, 2]:
                            is_straight = True
                    from collections import Counter
                    rank_counts = Counter(ranks)
                    counts = sorted(rank_counts.values(), reverse=True)
                    if is_straight and is_flush:
                        return 9
                    if counts == [4, 1]:
                        return 8
                    if counts == [3, 2]:
                        return 7
                    if is_flush:
                        return 6
                    if is_straight:
                        return 5
                    if counts == [3, 1, 1]:
                        return 4
                    if counts == [2, 2, 1]:
                        return 3
                    if counts == [2, 1, 1, 1]:
                        return 2
                    return 1
            ''').strip(),
        ],
    },
}


# ── Distractor code ──

DISTRACTORS = [
    textwrap.dedent('''
        def format_table(rows):
            """Format rows as a text table (not relevant to the task)."""
            if not rows:
                return ""
            widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
            lines = []
            for row in rows:
                lines.append(" | ".join(str(val).ljust(w) for val, w in zip(row, widths)))
            return "\\n".join(lines)
    ''').strip(),
    textwrap.dedent('''
        def colorize(text, color):
            """Add ANSI color codes (not relevant to the task)."""
            colors = {"red": 31, "green": 32, "yellow": 33, "blue": 34}
            code = colors.get(color, 0)
            return f"\\033[{code}m{text}\\033[0m" if code else text
    ''').strip(),
    textwrap.dedent('''
        def memoize(func):
            """Memoize a function (not relevant to the task)."""
            cache = {}
            def wrapper(*args):
                if args not in cache:
                    cache[args] = func(*args)
                return cache[args]
            return wrapper
    ''').strip(),
]


@register_env
class TestWritingEnv(AgenticEnv):
    name = "test_writing"
    skill = "Writing effective tests that catch bugs"
    difficulty_tiers = ["easy", "medium", "hard"]

    def gen_params(self, rng, difficulty="medium"):
        domain_name = rng.choice(list(DOMAINS.keys()))
        domain = DOMAINS[domain_name]
        n_mutants = {"easy": 3, "medium": 4, "hard": 5}[difficulty]
        n_mutants = min(n_mutants, len(domain["mutants"]))
        n_distractors = {"easy": 0, "medium": 1, "hard": 2}[difficulty]
        distractors = rng.sample(DISTRACTORS, n_distractors) if n_distractors else []

        # Select mutant indices
        mutant_indices = rng.sample(range(len(domain["mutants"])), n_mutants)

        return {
            "domain": domain_name,
            "difficulty": difficulty,
            "n_mutants": n_mutants,
            "mutant_indices": mutant_indices,
            "n_distractors": n_distractors,
            "distractor_indices": [DISTRACTORS.index(d) for d in distractors] if distractors else [],
            "seed": rng.randint(0, 999999),
        }

    def gen_codebase(self, params, rng):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]
        codebase = {f"{module}.py": domain["correct"]}

        for idx in params.get("distractor_indices", []):
            distractor = DISTRACTORS[idx]
            codebase[f"helper_{idx}.py"] = distractor

        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]

        lines = []
        lines.append("You are a software engineer writing tests for an existing function.")
        lines.append("")
        lines.append("You are given a correct implementation. Your task is to write")
        lines.append("comprehensive tests that:")
        lines.append("1. Verify the correct behavior of the function")
        lines.append("2. Cover edge cases and boundary conditions")
        lines.append("3. Would catch common bugs (mutations) if introduced")
        lines.append("")
        lines.append("=== FUNCTION SPECIFICATION ===")
        lines.append(domain["spec"])
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")

        lines.append("Write your tests as a Python file with test_ functions (using assert).")
        lines.append("Your tests should import from the module and test thoroughly.")
        lines.append("")
        lines.append("Provide your tests in the following format:")
        lines.append("<reasoning>")
        lines.append("...analyze the function, identify edge cases,")
        lines.append("explain what each test checks and why...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append("```python:test_model.py")
        lines.append("# your test functions here")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]
        # Generate example tests that would catch all mutants
        # These are for SFT - showing thorough test writing
        example_tests = self._generate_example_tests(params, module)
        return {"test_model.py": example_tests}

    def _generate_example_tests(self, params, module):
        """Generate example tests that would catch mutants for this domain."""
        domain = DOMAINS[params["domain"]]
        if params["domain"] == "password_validator":
            return textwrap.dedent(f'''
                from {module} import validate_password

                def test_valid_password():
                    assert validate_password("Abc123!x") == []

                def test_short_password():
                    errors = validate_password("Ab1!x")
                    assert "Password must be at least 8 characters" in errors

                def test_exactly_8_chars():
                    assert validate_password("Abcdef1!") == []

                def test_no_uppercase():
                    errors = validate_password("abcdef1!")
                    assert "Password must contain an uppercase letter" in errors

                def test_no_lowercase():
                    errors = validate_password("ABCDEF1!")
                    assert "Password must contain a lowercase letter" in errors

                def test_no_digit():
                    errors = validate_password("Abcdefg!")
                    assert "Password must contain a digit" in errors

                def test_no_special_char():
                    errors = validate_password("Abcdef12")
                    assert "Password must contain a special character" in errors

                def test_empty_password():
                    errors = validate_password("")
                    assert len(errors) == 5

                def test_all_rules_violated():
                    errors = validate_password("a")
                    assert len(errors) >= 4

                def test_special_chars_set():
                    for ch in "!@#$%^&*":
                        pw = "Abcdef1" + ch
                        assert validate_password(pw) == []

                def test_other_chars_not_special():
                    errors = validate_password("Abcdef12(")
                    assert "Password must contain a special character" in errors
            ''').strip()
        elif params["domain"] == "date_parser":
            return textwrap.dedent(f'''
                from {module} import parse_date

                def test_valid_date():
                    result = parse_date("2024-01-15")
                    assert result == {{"year": 2024, "month": 1, "day": 15}}

                def test_leap_year_feb29():
                    result = parse_date("2024-02-29")
                    assert result["day"] == 29

                def test_non_leap_feb29():
                    try:
                        parse_date("2023-02-29")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_century_non_leap():
                    try:
                        parse_date("1900-02-29")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_year400_leap():
                    result = parse_date("2000-02-29")
                    assert result["day"] == 29

                def test_invalid_month():
                    try:
                        parse_date("2024-13-01")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_month_zero():
                    try:
                        parse_date("2024-00-15")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_invalid_day():
                    try:
                        parse_date("2024-01-32")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_day_zero():
                    try:
                        parse_date("2024-01-00")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_april_31():
                    try:
                        parse_date("2024-04-31")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_wrong_format():
                    try:
                        parse_date("01/15/2024")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_non_integer():
                    try:
                        parse_date("2024-aa-15")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_two_parts():
                    try:
                        parse_date("2024-01")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass
            ''').strip()
        elif params["domain"] == "url_parser":
            return textwrap.dedent(f'''
                from {module} import parse_url

                def test_full_url():
                    result = parse_url("http://example.com:8080/path?q=1#frag")
                    assert result["scheme"] == "http"
                    assert result["host"] == "example.com"
                    assert result["port"] == 8080
                    assert result["path"] == "/path"
                    assert result["query"] == "q=1"
                    assert result["fragment"] == "frag"

                def test_scheme_and_host():
                    result = parse_url("https://example.com")
                    assert result["scheme"] == "https"
                    assert result["host"] == "example.com"
                    assert result["port"] is None
                    assert result["path"] == "/"

                def test_host_and_path():
                    result = parse_url("example.com/path")
                    assert result["host"] == "example.com"
                    assert result["path"] == "/path"
                    assert result["scheme"] is None

                def test_with_port():
                    result = parse_url("http://host:3000/api")
                    assert result["port"] == 3000

                def test_with_query():
                    result = parse_url("http://host/path?a=b&c=d")
                    assert result["query"] == "a=b&c=d"

                def test_with_fragment():
                    result = parse_url("http://host/path#section")
                    assert result["fragment"] == "section"

                def test_default_path():
                    result = parse_url("http://host")
                    assert result["path"] == "/"

                def test_no_host():
                    try:
                        parse_url("/just/a/path")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_invalid_port():
                    try:
                        parse_url("http://host:abc/path")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_empty_host_with_scheme():
                    try:
                        parse_url("http:///path")
                        assert False, "Should raise ValueError"
                    except ValueError:
                        pass

                def test_path_with_slash():
                    result = parse_url("http://host/a/b/c")
                    assert result["path"] == "/a/b/c"
            ''').strip()
        else:  # poker_evaluator
            return textwrap.dedent(f'''
                from {module} import evaluate_hand

                def test_high_card():
                    cards = [(14, "H"), (10, "D"), (8, "C"), (6, "S"), (3, "H")]
                    assert evaluate_hand(cards) == 1

                def test_one_pair():
                    cards = [(10, "H"), (10, "D"), (8, "C"), (6, "S"), (3, "H")]
                    assert evaluate_hand(cards) == 2

                def test_two_pair():
                    cards = [(10, "H"), (10, "D"), (8, "C"), (8, "S"), (3, "H")]
                    assert evaluate_hand(cards) == 3

                def test_three_of_a_kind():
                    cards = [(10, "H"), (10, "D"), (10, "C"), (6, "S"), (3, "H")]
                    assert evaluate_hand(cards) == 4

                def test_straight():
                    cards = [(6, "H"), (5, "D"), (4, "C"), (3, "S"), (2, "H")]
                    assert evaluate_hand(cards) == 5

                def test_wheel_straight():
                    cards = [(14, "H"), (5, "D"), (4, "C"), (3, "S"), (2, "H")]
                    assert evaluate_hand(cards) == 5

                def test_flush():
                    cards = [(14, "H"), (10, "H"), (8, "H"), (6, "H"), (3, "H")]
                    assert evaluate_hand(cards) == 6

                def test_full_house():
                    cards = [(10, "H"), (10, "D"), (10, "C"), (6, "S"), (6, "H")]
                    assert evaluate_hand(cards) == 7

                def test_four_of_a_kind():
                    cards = [(10, "H"), (10, "D"), (10, "C"), (10, "S"), (3, "H")]
                    assert evaluate_hand(cards) == 8

                def test_straight_flush():
                    cards = [(9, "H"), (8, "H"), (7, "H"), (6, "H"), (5, "H")]
                    assert evaluate_hand(cards) == 9

                def test_not_straight():
                    cards = [(10, "H"), (8, "D"), (6, "C"), (4, "S"), (2, "H")]
                    assert evaluate_hand(cards) == 1

                def test_not_flush():
                    cards = [(14, "H"), (10, "D"), (8, "C"), (6, "S"), (3, "H")]
                    assert evaluate_hand(cards) == 1
            ''').strip()

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]

        reasoning = textwrap.dedent(f"""
            Let me carefully analyze the function to write comprehensive tests that catch bugs.

            First, I'll read the implementation in {module}.py to understand the exact behavior:

            {domain['spec']}

            Let me trace through the code line by line to identify all the edge cases and
            boundary conditions I need to test:

            1. Normal/happy path: I need to test the basic case where inputs are valid and
            the function should work correctly. This establishes a baseline.

            2. Boundary conditions: I need to test the edges of each condition:
            - For length checks: test exactly at the boundary (e.g., exactly 8 characters)
            - For range checks: test the minimum and maximum valid values
            - For empty inputs: test what happens with empty strings or lists

            3. Error cases: I need to test that the function raises errors when it should:
            - Invalid inputs that should raise ValueError
            - Missing required components
            - Out-of-range values

            4. Mutation-catching tests: I need to think about what bugs could be introduced
            and write tests that would catch them:
            - Off-by-one errors (e.g., < vs <=)
            - Missing checks (e.g., a validation rule removed)
            - Wrong operators (e.g., > instead of <)
            - Changed constants (e.g., different character sets)
            - Logic inversions (e.g., checking the wrong condition)

            Let me think about each specific edge case for this function:

            Looking at the implementation carefully, I can see several conditions that
            could be mutated:
            - Length comparisons could use wrong operators
            - Character set checks could be modified
            - Validation steps could be removed entirely
            - Order of operations could be changed
            - Return values could be wrong

            For each potential mutation, I need a test that would fail if that mutation
            were introduced. This means my tests need to:
            - Exercise each code path
            - Test boundary values precisely
            - Verify exact return values, not just types
            - Test both valid and invalid inputs

            Now let me write the tests, making sure each one targets a specific behavior
            that could be broken by a mutation. I'll organize them by category:
            - Valid input tests (should pass on correct code)
            - Edge case tests (boundary conditions)
            - Error case tests (should raise exceptions)
            - Mutation-catching tests (specific inputs that distinguish correct from buggy)

            Each test has a clear purpose and would fail if the corresponding code path
            were modified incorrectly.
        """).strip()

        return reasoning

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]

        answer = extract_answer(response)
        code_changes = parse_code_blocks(answer)

        if not code_changes:
            return 0.0, {
                "reason": "no code blocks found in answer",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        # Find the test file the model wrote
        test_code = None
        test_filename = None
        for fname, content in code_changes.items():
            if "test" in fname.lower() or fname.endswith(".py"):
                test_code = content
                test_filename = fname
                break

        if test_code is None:
            return 0.0, {
                "reason": "no test file found in answer",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        # Step 1: Run tests against the CORRECT implementation
        # Tests MUST pass on correct code, otherwise score = 0
        correct_codebase = dict(codebase)
        correct_results = run_tests(correct_codebase, test_code, timeout=10.0)

        correct_passed = correct_results.get("passed", 0)
        correct_total = correct_results.get("total", 0)

        if correct_total == 0:
            return 0.0, {
                "reason": "no tests ran",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        if correct_passed < correct_total:
            # Tests fail on correct code - tests are wrong
            return 0.0, {
                "reason": f"tests fail on correct code ({correct_passed}/{correct_total} passed)",
                "correct_results": correct_results,
                "has_reasoning": bool(extract_reasoning(response)),
            }

        # Step 2: Run tests against each mutant
        mutants_killed = 0
        mutant_results = []
        for idx in params["mutant_indices"]:
            mutant_code = domain["mutants"][idx]
            mutant_codebase = dict(codebase)
            mutant_codebase[f"{module}.py"] = mutant_code

            mutant_results_data = run_tests(mutant_codebase, test_code, timeout=10.0)
            mutant_passed = mutant_results_data.get("passed", 0)
            mutant_total = mutant_results_data.get("total", 0)

            # Mutant is "killed" if at least one test fails
            killed = mutant_passed < mutant_total
            if killed:
                mutants_killed += 1

            mutant_results.append({
                "mutant_idx": idx,
                "passed": mutant_passed,
                "total": mutant_total,
                "killed": killed,
            })

        n_mutants = len(params["mutant_indices"])
        kill_rate = mutants_killed / n_mutants if n_mutants > 0 else 0.0

        breakdown = {
            "correct_tests_passed": correct_passed,
            "correct_tests_total": correct_total,
            "mutants_killed": mutants_killed,
            "mutants_total": n_mutants,
            "kill_rate": kill_rate,
            "mutant_results": mutant_results,
            "has_reasoning": bool(extract_reasoning(response)),
            "test_filename": test_filename,
            "score": kill_rate,
        }

        return kill_rate, breakdown
