"""
Environment 2: Feature Implementation

Skill: Implementing a feature from a specification in an existing codebase.

The model is given a codebase with existing functionality and a feature spec.
It must implement the feature following existing patterns in the codebase.

Difficulty scaling:
- easy: single file, simple feature, clear pattern to follow
- medium: 2 files, moderate feature, need to understand cross-file patterns
- hard: 3 files with distractors, complex feature requiring understanding of
  the full codebase architecture
"""
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, parse_code_blocks, apply_code_changes,
    run_tests, compute_test_score, extract_reasoning,
)


# ── Domain definitions ──
# Each domain has:
#   - "existing": the codebase the model starts with
#   - "feature_code": the code that implements the new feature (the solution)
#   - "feature_spec": the specification for the new feature
#   - "existing_tests": tests for existing functionality (must still pass)
#   - "feature_tests": tests for the new feature (must pass after implementation)
#   - "module": the main module name
#   - "supporting": optional supporting files

DOMAINS = {
    "todo_manager": {
        "module": "todo",
        "existing": textwrap.dedent('''
            class TodoList:
                """A simple todo list manager."""

                def __init__(self):
                    self._items = []
                    self._next_id = 1

                def add(self, title):
                    """Add a todo item, return its id."""
                    item = {"id": self._next_id, "title": title, "done": False}
                    self._items.append(item)
                    self._next_id += 1
                    return item["id"]

                def complete(self, item_id):
                    """Mark a todo item as done."""
                    for item in self._items:
                        if item["id"] == item_id:
                            item["done"] = True
                            return True
                    return False

                def remove(self, item_id):
                    """Remove a todo item by id."""
                    self._items = [i for i in self._items if i["id"] != item_id]

                def list_all(self):
                    """Return all todo items."""
                    return list(self._items)

                def list_pending(self):
                    """Return only incomplete todo items."""
                    return [i for i in self._items if not i["done"]]

                def list_completed(self):
                    """Return only completed todo items."""
                    return [i for i in self._items if i["done"]]
        ''').strip(),
        "feature_code": textwrap.dedent('''
                def search(self, query):
                    """Return items whose title contains the query (case-insensitive)."""
                    query_lower = query.lower()
                    return [i for i in self._items if query_lower in i["title"].lower()]

                def sort_by(self, key, reverse=False):
                    """Return items sorted by key ('title', 'id', or 'done')."""
                    valid_keys = {"title", "id", "done"}
                    if key not in valid_keys:
                        raise ValueError(f"Invalid sort key: {key}. Must be one of {valid_keys}")
                    return sorted(self._items, key=lambda i: i[key], reverse=reverse)

                def clear_completed(self):
                    """Remove all completed items, return count removed."""
                    before = len(self._items)
                    self._items = [i for i in self._items if not i["done"]]
                    return before - len(self._items)
        ''').strip(),
        "feature_spec": (
            "Add the following methods to the TodoList class:\n"
            "1. search(query) - Return items whose title contains the query string "
            "(case-insensitive). Return a list of matching item dicts.\n"
            "2. sort_by(key, reverse=False) - Return all items sorted by the given key. "
            "Valid keys are 'title', 'id', and 'done'. Raise ValueError for invalid keys. "
            "Returns a new sorted list (does not mutate internal state).\n"
            "3. clear_completed() - Remove all completed items from the list and return "
            "the count of items removed."
        ),
        "existing_tests": textwrap.dedent('''
            from todo import TodoList
            def test_add():
                t = TodoList()
                assert t.add("Buy milk") == 1
                assert t.add("Walk dog") == 2
            def test_complete():
                t = TodoList()
                tid = t.add("Task")
                assert t.complete(tid) == True
                assert t.complete(999) == False
            def test_list_pending():
                t = TodoList()
                t.add("A")
                tid = t.add("B")
                t.complete(tid)
                pending = t.list_pending()
                assert len(pending) == 1
                assert pending[0]["title"] == "A"
            def test_remove():
                t = TodoList()
                tid = t.add("A")
                t.remove(tid)
                assert t.list_all() == []
        ''').strip(),
        "feature_tests": textwrap.dedent('''
            from todo import TodoList
            def test_search_case_insensitive():
                t = TodoList()
                t.add("Buy Milk")
                t.add("Walk Dog")
                t.add("buy bread")
                results = t.search("buy")
                assert len(results) == 2
            def test_search_no_match():
                t = TodoList()
                t.add("A")
                assert t.search("xyz") == []
            def test_search_empty_query():
                t = TodoList()
                t.add("A")
                t.add("B")
                results = t.search("")
                assert len(results) == 2
            def test_sort_by_title():
                t = TodoList()
                t.add("Cherry")
                t.add("Apple")
                t.add("Banana")
                result = t.sort_by("title")
                assert [i["title"] for i in result] == ["Apple", "Banana", "Cherry"]
            def test_sort_by_id_reverse():
                t = TodoList()
                t.add("A")
                t.add("B")
                t.add("C")
                result = t.sort_by("id", reverse=True)
                assert [i["id"] for i in result] == [3, 2, 1]
            def test_sort_by_invalid_key():
                t = TodoList()
                t.add("A")
                try:
                    t.sort_by("invalid")
                    assert False, "Should have raised ValueError"
                except ValueError:
                    pass
            def test_clear_completed():
                t = TodoList()
                t.add("A")
                t.add("B")
                t.add("C")
                t.complete(1)
                t.complete(3)
                removed = t.clear_completed()
                assert removed == 2
                assert len(t.list_all()) == 1
            def test_clear_completed_none():
                t = TodoList()
                t.add("A")
                assert t.clear_completed() == 0
                assert len(t.list_all()) == 1
        ''').strip(),
    },
    "shopping_cart": {
        "module": "cart",
        "existing": textwrap.dedent('''
            class ShoppingCart:
                """A shopping cart with items and quantities."""

                def __init__(self):
                    self._items = {}

                def add_item(self, name, price, quantity=1):
                    """Add an item to the cart."""
                    if name in self._items:
                        self._items[name]["quantity"] += quantity
                    else:
                        self._items[name] = {"price": price, "quantity": quantity}

                def remove_item(self, name):
                    """Remove an item from the cart."""
                    if name in self._items:
                        del self._items[name]

                def get_item(self, name):
                    """Get an item dict by name, or None."""
                    return self._items.get(name)

                def total(self):
                    """Compute the total price of all items."""
                    return sum(i["price"] * i["quantity"] for i in self._items.values())

                def item_count(self):
                    """Return total number of items (sum of quantities)."""
                    return sum(i["quantity"] for i in self._items.values())

                def list_items(self):
                    """Return list of item names."""
                    return list(self._items.keys())
        ''').strip(),
        "feature_code": textwrap.dedent('''
                def apply_discount(self, percent):
                    """Apply a percentage discount to all items. Returns discount amount."""
                    if not 0 <= percent <= 100:
                        raise ValueError("Discount must be between 0 and 100")
                    before = self.total()
                    for item in self._items.values():
                        item["price"] = round(item["price"] * (1 - percent / 100), 2)
                    return round(before - self.total(), 2)

                def export_csv(self):
                    """Export cart contents as CSV string."""
                    lines = ["name,price,quantity"]
                    for name, info in self._items.items():
                        lines.append(f"{name},{info['price']},{info['quantity']}")
                    return "\\n".join(lines)

                def merge(self, other_cart):
                    """Merge another cart's items into this one."""
                    for name, info in other_cart._items.items():
                        self.add_item(name, info["price"], info["quantity"])
        ''').strip(),
        "feature_spec": (
            "Add the following methods to the ShoppingCart class:\n"
            "1. apply_discount(percent) - Apply a percentage discount (0-100) to all "
            "item prices in place. Raise ValueError if percent is not in [0, 100]. "
            "Round each new price to 2 decimal places. Return the total discount amount "
            "(before_total - after_total, rounded to 2 decimals).\n"
            "2. export_csv() - Return a CSV string with header 'name,price,quantity' "
            "followed by one row per item.\n"
            "3. merge(other_cart) - Merge items from another ShoppingCart instance into "
            "this one. Items that exist in both should have their quantities added together."
        ),
        "existing_tests": textwrap.dedent('''
            from cart import ShoppingCart
            def test_add_and_total():
                c = ShoppingCart()
                c.add_item("apple", 1.50, 3)
                c.add_item("banana", 0.50, 2)
                assert abs(c.total() - 5.50) < 0.01
            def test_add_existing():
                c = ShoppingCart()
                c.add_item("apple", 1.00, 2)
                c.add_item("apple", 1.00, 3)
                assert c.item_count() == 5
            def test_remove():
                c = ShoppingCart()
                c.add_item("apple", 1.00)
                c.remove_item("apple")
                assert c.list_items() == []
            def test_get_item():
                c = ShoppingCart()
                c.add_item("apple", 1.00, 2)
                item = c.get_item("apple")
                assert item["quantity"] == 2
                assert c.get_item("missing") is None
        ''').strip(),
        "feature_tests": textwrap.dedent('''
            from cart import ShoppingCart
            def test_apply_discount():
                c = ShoppingCart()
                c.add_item("apple", 10.00, 2)
                discount = c.apply_discount(10)
                assert abs(discount - 2.00) < 0.01
                assert abs(c.total() - 18.00) < 0.01
            def test_apply_discount_zero():
                c = ShoppingCart()
                c.add_item("x", 5.00)
                assert c.apply_discount(0) == 0
            def test_apply_discount_invalid():
                c = ShoppingCart()
                c.add_item("x", 5.00)
                try:
                    c.apply_discount(150)
                    assert False, "Should raise ValueError"
                except ValueError:
                    pass
            def test_export_csv():
                c = ShoppingCart()
                c.add_item("apple", 1.50, 3)
                c.add_item("banana", 0.50, 2)
                csv = c.export_csv()
                lines = csv.split("\\n")
                assert lines[0] == "name,price,quantity"
                assert len(lines) == 3
            def test_export_csv_empty():
                c = ShoppingCart()
                csv = c.export_csv()
                assert csv == "name,price,quantity"
            def test_merge():
                c1 = ShoppingCart()
                c1.add_item("apple", 1.00, 2)
                c2 = ShoppingCart()
                c2.add_item("apple", 1.00, 3)
                c2.add_item("banana", 0.50, 1)
                c1.merge(c2)
                assert c1.item_count() == 6
                assert "banana" in c1.list_items()
        ''').strip(),
    },
    "temp_converter": {
        "module": "temperature",
        "existing": textwrap.dedent('''
            def celsius_to_fahrenheit(c):
                """Convert Celsius to Fahrenheit."""
                return c * 9 / 5 + 32

            def fahrenheit_to_celsius(f):
                """Convert Fahrenheit to Celsius."""
                return (f - 32) * 5 / 9

            def celsius_to_kelvin(c):
                """Convert Celsius to Kelvin."""
                return c + 273.15

            def kelvin_to_celsius(k):
                """Convert Kelvin to Celsius."""
                return k - 273.15

            def format_temperature(value, unit, decimals=1):
                """Format a temperature with unit symbol."""
                symbol = {"C": "°C", "F": "°F", "K": "K"}.get(unit, "?")
                return f"{value:.{decimals}f}{symbol}"
        ''').strip(),
        "feature_code": textwrap.dedent('''
            def convert(value, from_unit, to_unit):
                """Convert between any two temperature units (C, F, K)."""
                valid = {"C", "F", "K"}
                if from_unit not in valid or to_unit not in valid:
                    raise ValueError(f"Units must be one of {valid}")
                if from_unit == to_unit:
                    return value
                # Convert to Celsius first
                if from_unit == "F":
                    celsius = fahrenheit_to_celsius(value)
                elif from_unit == "K":
                    celsius = kelvin_to_celsius(value)
                else:
                    celsius = value
                # Convert from Celsius to target
                if to_unit == "F":
                    return celsius_to_fahrenheit(celsius)
                elif to_unit == "K":
                    return celsius_to_kelvin(celsius)
                return celsius

            def convert_batch(temperatures, from_unit, to_unit):
                """Convert a list of temperatures from one unit to another."""
                return [convert(t, from_unit, to_unit) for t in temperatures]

            def temperature_range(start, end, step, unit):
                """Generate a list of temperatures from start to end (exclusive)
                with given step, all in the specified unit."""
                result = []
                current = start
                if step <= 0:
                    raise ValueError("step must be positive")
                if start <= end:
                    while current < end:
                        result.append(round(current, 10))
                        current += step
                else:
                    while current > end:
                        result.append(round(current, 10))
                        current -= step
                return result
        ''').strip(),
        "feature_spec": (
            "Add the following functions to temperature.py:\n"
            "1. convert(value, from_unit, to_unit) - Convert a temperature between "
            "any two units ('C', 'F', 'K'). Raise ValueError for invalid units. "
            "If from_unit == to_unit, return value unchanged.\n"
            "2. convert_batch(temperatures, from_unit, to_unit) - Convert a list of "
            "temperatures. Returns a list of converted values.\n"
            "3. temperature_range(start, end, step, unit) - Generate a list of "
            "temperatures from start to end (exclusive) with given step. Raise "
            "ValueError if step <= 0. Handle both ascending and descending ranges."
        ),
        "existing_tests": textwrap.dedent('''
            from temperature import celsius_to_fahrenheit, fahrenheit_to_celsius
            from temperature import celsius_to_kelvin, kelvin_to_celsius
            from temperature import format_temperature
            def test_c_to_f():
                assert abs(celsius_to_fahrenheit(0) - 32) < 0.01
                assert abs(celsius_to_fahrenheit(100) - 212) < 0.01
            def test_f_to_c():
                assert abs(fahrenheit_to_celsius(32) - 0) < 0.01
                assert abs(fahrenheit_to_celsius(212) - 100) < 0.01
            def test_c_to_k():
                assert abs(celsius_to_kelvin(0) - 273.15) < 0.01
            def test_format():
                assert format_temperature(23.5, "C") == "23.5°C"
                assert format_temperature(70, "F", 0) == "70°F"
        ''').strip(),
        "feature_tests": textwrap.dedent('''
            from temperature import convert, convert_batch, temperature_range
            def test_convert_c_to_f():
                assert abs(convert(0, "C", "F") - 32) < 0.01
            def test_convert_f_to_k():
                assert abs(convert(32, "F", "K") - 273.15) < 0.01
            def test_convert_k_to_f():
                assert abs(convert(273.15, "K", "F") - 32) < 0.01
            def test_convert_same_unit():
                assert convert(25, "C", "C") == 25
            def test_convert_invalid_unit():
                try:
                    convert(0, "C", "X")
                    assert False, "Should raise ValueError"
                except ValueError:
                    pass
            def test_convert_batch():
                result = convert_batch([0, 100], "C", "F")
                assert len(result) == 2
                assert abs(result[0] - 32) < 0.01
                assert abs(result[1] - 212) < 0.01
            def test_convert_batch_empty():
                assert convert_batch([], "C", "F") == []
            def test_temp_range_ascending():
                result = temperature_range(0, 10, 2, "C")
                assert result == [0, 2, 4, 6, 8]
            def test_temp_range_descending():
                result = temperature_range(10, 0, 2, "C")
                assert result == [10, 8, 6, 4, 2]
            def test_temp_range_invalid_step():
                try:
                    temperature_range(0, 10, 0, "C")
                    assert False, "Should raise ValueError"
                except ValueError:
                    pass
        ''').strip(),
    },
    "text_formatter": {
        "module": "formatter",
        "existing": textwrap.dedent('''
            def wrap_text(text, width):
                """Wrap text to the given width, breaking on spaces."""
                words = text.split()
                lines = []
                current = []
                current_len = 0
                for word in words:
                    if current_len + len(word) + (1 if current else 0) > width:
                        lines.append(" ".join(current))
                        current = [word]
                        current_len = len(word)
                    else:
                        current.append(word)
                        current_len += len(word) + (1 if len(current) > 1 else 0)
                if current:
                    lines.append(" ".join(current))
                return lines

            def indent_lines(lines, spaces=4):
                """Indent each line by the given number of spaces."""
                prefix = " " * spaces
                return [prefix + line if line else line for line in lines]

            def center_text(text, width):
                """Center text within given width."""
                if len(text) >= width:
                    return text
                total_pad = width - len(text)
                left = total_pad // 2
                right = total_pad - left
                return " " * left + text + " " * right

            def count_words(text):
                """Count words in text."""
                return len(text.split())
        ''').strip(),
        "feature_code": textwrap.dedent('''
            def justify_text(text, width):
                """Justify text to exact width by distributing extra spaces."""
                words = text.split()
                if len(words) == 1:
                    return words[0].ljust(width)
                if len(words) == 0:
                    return ""
                total_chars = sum(len(w) for w in words)
                total_spaces = width - total_chars
                gaps = len(words) - 1
                if total_spaces < gaps:
                    return " ".join(words)
                base_spaces = total_spaces // gaps
                extra = total_spaces % gaps
                result = []
                for i, word in enumerate(words):
                    result.append(word)
                    if i < gaps:
                        spaces = base_spaces + (1 if i < extra else 0)
                        result.append(" " * spaces)
                return "".join(result)

            def title_case(text):
                """Convert text to title case, respecting small words."""
                small_words = {"a", "an", "the", "and", "but", "or", "for",
                               "nor", "on", "at", "to", "by", "in", "of"}
                words = text.split()
                result = []
                for i, word in enumerate(words):
                    lower = word.lower()
                    if i > 0 and i < len(words) - 1 and lower in small_words:
                        result.append(lower)
                    else:
                        result.append(word.capitalize())
                return " ".join(result)

            def truncate_lines(lines, max_width, suffix="..."):
                """Truncate each line to max_width, adding suffix if truncated."""
                result = []
                for line in lines:
                    if len(line) <= max_width:
                        result.append(line)
                    else:
                        result.append(line[:max_width - len(suffix)] + suffix)
                return result
        ''').strip(),
        "feature_spec": (
            "Add the following functions to formatter.py:\n"
            "1. justify_text(text, width) - Justify a single line of text to exactly "
            "'width' characters by distributing extra spaces between words. If there's "
            "only one word, left-justify it. If text is empty, return empty string.\n"
            "2. title_case(text) - Convert text to title case, but keep small words "
            "(a, an, the, and, but, or, for, nor, on, at, to, by, in, of) lowercase "
            "unless they are the first or last word.\n"
            "3. truncate_lines(lines, max_width, suffix='...') - Truncate each line in "
            "a list to max_width characters. If a line exceeds max_width, truncate it "
            "and append the suffix (total length = max_width). Lines within limit are "
            "returned unchanged."
        ),
        "existing_tests": textwrap.dedent('''
            from formatter import wrap_text, indent_lines, center_text, count_words
            def test_wrap_text():
                result = wrap_text("hello world foo bar", 10)
                assert all(len(line) <= 10 for line in result)
                assert " ".join(result) == "hello world foo bar"
            def test_indent_lines():
                result = indent_lines(["a", "b"], 2)
                assert result == ["  a", "  b"]
            def test_center_text():
                assert center_text("hi", 6) == "  hi  "
            def test_count_words():
                assert count_words("one two three") == 3
        ''').strip(),
        "feature_tests": textwrap.dedent('''
            from formatter import justify_text, title_case, truncate_lines
            def test_justify_basic():
                result = justify_text("hello world", 15)
                assert len(result) == 15
                assert result.startswith("hello")
                assert result.endswith("world")
            def test_justify_single_word():
                result = justify_text("hello", 10)
                assert result == "hello     "
            def test_justify_empty():
                assert justify_text("", 10) == ""
            def test_justify_exact():
                result = justify_text("a b c", 5)
                assert result == "a b c"
            def test_title_case_basic():
                assert title_case("the lord of the rings") == "The Lord of the Rings"
            def test_title_case_first_last():
                assert title_case("the end") == "The End"
            def test_title_case_middle_small():
                assert title_case("war and peace") == "War and Peace"
            def test_truncate_lines():
                lines = ["short", "this is a very long line"]
                result = truncate_lines(lines, 10)
                assert result[0] == "short"
                assert len(result[1]) == 10
                assert result[1].endswith("...")
            def test_truncate_custom_suffix():
                lines = ["abcdefghij"]
                result = truncate_lines(lines, 6, suffix="!")
                assert result[0] == "abcde!"
        ''').strip(),
    },
}


# ── Distractor code (irrelevant functions to test if model can focus) ──

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

        def parse_csv(text):
            """Parse CSV text into rows (not relevant to the task)."""
            lines = text.strip().split("\\n")
            return [line.split(",") for line in lines]
    ''').strip(),
    textwrap.dedent('''
        def colorize(text, color):
            """Add ANSI color codes (not relevant to the task)."""
            colors = {"red": 31, "green": 32, "yellow": 33, "blue": 34}
            code = colors.get(color, 0)
            return f"\\033[{code}m{text}\\033[0m" if code else text

        def pad_left(text, width, char=" "):
            """Pad text on the left (not relevant to the task)."""
            return char * max(0, width - len(text)) + text
    ''').strip(),
    textwrap.dedent('''
        def debounce(func, delay):
            """Debounce a function call (not relevant to the task)."""
            import time
            last_call = [0]
            def wrapper(*args, **kwargs):
                now = time.time()
                if now - last_call[0] >= delay:
                    last_call[0] = now
                    return func(*args, **kwargs)
            return wrapper

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
class FeatureImplEnv(AgenticEnv):
    name = "feature_impl"
    skill = "Implementing a feature from a specification in an existing codebase"
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

    def _get_solution_code(self, params):
        """Return the full module code with the feature implemented."""
        domain = DOMAINS[params["domain"]]
        existing = domain["existing"]
        feature = domain["feature_code"]
        # For class-based domains, insert feature methods before the last line
        # For function-based domains, append at the end
        if "class " in existing:
            # Find the indentation level of methods (4 spaces typically)
            # Insert feature code at the same indentation, before the end of the class
            lines = existing.split("\n")
            # Find last non-empty line that's part of the class
            insert_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() and not lines[i].startswith("class "):
                    insert_idx = i + 1
                    break
            # Re-indent feature code to match method indentation (4 spaces)
            feature_lines = feature.strip().split("\n")
            reindented = "\n".join("    " + line if line.strip() else line
                                   for line in feature_lines)
            result = "\n".join(lines[:insert_idx]) + "\n" + reindented
            if insert_idx < len(lines):
                result += "\n" + "\n".join(lines[insert_idx:])
            return result.strip()
        else:
            return existing + "\n\n" + feature.strip()

    def gen_codebase(self, params, rng):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]
        codebase = {f"{module}.py": domain["existing"]}

        # Add distractor modules
        for idx in params.get("distractor_indices", []):
            distractor = DISTRACTORS[idx]
            codebase[f"helper_{idx}.py"] = distractor

        # Add existing test file
        codebase["test_existing.py"] = domain["existing_tests"]

        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]

        lines = []
        lines.append("You are a software engineer adding a new feature to an existing codebase.")
        lines.append("")
        lines.append("The codebase has existing functionality with tests in `test_existing.py`.")
        lines.append("Your task is to:")
        lines.append("1. Read the existing code carefully to understand the patterns and conventions")
        lines.append("2. Implement the new feature described in the spec below")
        lines.append("3. Follow existing code style and patterns")
        lines.append("4. Ensure existing tests still pass AND new feature tests pass")
        lines.append("")
        lines.append("=== FEATURE SPECIFICATION ===")
        lines.append(domain["feature_spec"])
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")

        lines.append("Provide your implementation in the following format:")
        lines.append("<reasoning>")
        lines.append("...read the existing code, explain the patterns you see,")
        lines.append("describe how the new feature should follow those patterns...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append(f"```python:{module}.py")
        lines.append("# the complete file with existing code + new feature")
        lines.append("```")
        lines.append("</answer>")

        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]
        return {f"{module}.py": self._get_solution_code(params)}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]

        reasoning = textwrap.dedent(f"""
            Let me carefully read the existing codebase to understand the patterns before implementing the feature.

            First, I'll examine {module}.py to understand the existing code structure and conventions:

            Looking at the existing code, I can see the following patterns:
            - The code uses clear docstrings for all functions/methods
            - Error handling uses ValueError for invalid inputs
            - Methods return appropriate types (lists, dicts, booleans, ints)
            - The code follows a consistent naming convention

            Now let me read the feature specification carefully:
            {domain['feature_spec']}

            Let me analyze each feature I need to implement:

            1. First feature: I need to understand what the spec asks for and how it relates
            to existing code. Looking at the existing methods/functions, I see they follow
            a specific pattern. My new feature should follow the same pattern.

            2. Second feature: The spec describes another capability. I need to make sure
            it integrates with the existing code and follows the same conventions for
            error handling and return types.

            3. Third feature: This one modifies state. I need to be careful about side effects
            and make sure the return value matches what the spec describes.

            Let me also check the existing tests to understand expected behavior:
            - test_existing.py shows how the existing functions are called and what they return
            - This tells me the expected interface and behavior patterns

            Now I'll implement the feature, making sure to:
            - Keep all existing code intact (don't break existing tests)
            - Add the new methods/functions at the appropriate location
            - Follow the same docstring and error handling patterns
            - Make sure the new code is syntactically correct and integrates properly

            Let me write the complete file with both existing code and the new feature.
            I need to be careful about indentation - if this is a class, the new methods
            need to be at the same indentation level as existing methods.

            After implementing, let me mentally trace through the feature tests to verify:
            - Each test should pass with my implementation
            - Edge cases (empty inputs, invalid inputs) are handled
            - The return types match what tests expect

            The implementation follows the existing patterns and should pass all tests.
        """).strip()

        return reasoning

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]
        module = domain["module"]

        # Parse the model's response
        answer = extract_answer(response)
        code_changes = parse_code_blocks(answer)

        if not code_changes:
            return 0.0, {
                "reason": "no code blocks found in answer",
                "has_reasoning": bool(extract_reasoning(response)),
            }

        # Apply changes to codebase
        modified_codebase = apply_code_changes(codebase, code_changes)

        # Run existing tests (must still pass)
        existing_results = run_tests(modified_codebase, domain["existing_tests"], timeout=10.0)
        existing_score, existing_breakdown = compute_test_score(existing_results)

        # Run feature tests (new functionality)
        feature_results = run_tests(modified_codebase, domain["feature_tests"], timeout=10.0)
        feature_score, feature_breakdown = compute_test_score(feature_results)

        # Combined score: existing tests must pass (weight 0.3), feature tests (weight 0.7)
        # If existing tests break, penalize heavily
        if existing_score < 1.0:
            # Existing functionality broken - cap score
            score = feature_score * 0.3 * existing_score
        else:
            score = 0.3 + 0.7 * feature_score

        breakdown = {
            "existing_tests": existing_breakdown,
            "feature_tests": feature_breakdown,
            "existing_score": existing_score,
            "feature_score": feature_score,
            "has_reasoning": bool(extract_reasoning(response)),
            "files_changed": list(code_changes.keys()),
            "changed_target": f"{module}.py" in code_changes,
            "score": score,
        }

        return score, breakdown
