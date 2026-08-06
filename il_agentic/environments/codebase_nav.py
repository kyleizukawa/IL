"""
Environment 7: Codebase Navigation & Understanding

Skill: Understanding and navigating a multi-file codebase.

The model is given a multi-file codebase and must answer questions about it:
- Where is function X defined?
- What does module Y import?
- Trace the data flow: when user calls Z, which functions are called?
- What does function W return?

This is a Q&A task — the answer is plain text, not code.

Difficulty scaling:
- easy: 2 files, direct questions
- medium: 3-4 files, trace questions
- hard: 5+ files with distractors, multi-hop trace questions
"""
import random
import textwrap
from ..base import AgenticEnv, register_env
from ..graders import (
    extract_answer, extract_reasoning, text_similarity,
)


# ── Domain definitions ──
# Each domain has:
#   files: {filename: content} the multi-file codebase
#   questions: list of {question, answer, keywords, trace}
#   distractor_files: extra irrelevant files for medium/hard

DOMAINS = {
    # ── Domain 1: Web App (routes/controllers/models) ──
    "web_app": {
        "files": {
            "routes.py": textwrap.dedent('''
                """URL routing for the web application."""
                from controllers.user_controller import UserController
                from controllers.order_controller import OrderController

                user_controller = UserController()
                order_controller = OrderController()

                def route_request(method, path, body=None):
                    """Route an HTTP request to the appropriate controller."""
                    if path == "/users" and method == "GET":
                        return user_controller.list_users()
                    elif path == "/users" and method == "POST":
                        return user_controller.create_user(body)
                    elif path.startswith("/users/") and method == "GET":
                        user_id = int(path.split("/")[-1])
                        return user_controller.get_user(user_id)
                    elif path == "/orders" and method == "GET":
                        return order_controller.list_orders()
                    elif path == "/orders" and method == "POST":
                        return order_controller.create_order(body)
                    elif path.startswith("/orders/") and method == "DELETE":
                        order_id = int(path.split("/")[-1])
                        return order_controller.cancel_order(order_id)
                    else:
                        return {"status": 404, "body": {"error": "not found"}}
            ''').strip(),
            "controllers/user_controller.py": textwrap.dedent('''
                """User controller handling user-related requests."""
                from models.user import User
                from models.database import db

                class UserController:
                    def list_users(self):
                        users = db.query(User).all()
                        return {"status": 200, "body": [u.to_dict() for u in users]}

                    def get_user(self, user_id):
                        user = db.query(User).get(user_id)
                        if not user:
                            return {"status": 404, "body": {"error": "user not found"}}
                        return {"status": 200, "body": user.to_dict()}

                    def create_user(self, body):
                        user = User(name=body["name"], email=body["email"])
                        db.save(user)
                        return {"status": 201, "body": user.to_dict()}
            ''').strip(),
            "controllers/order_controller.py": textwrap.dedent('''
                """Order controller handling order-related requests."""
                from models.order import Order
                from models.user import User
                from models.database import db

                class OrderController:
                    def list_orders(self):
                        orders = db.query(Order).all()
                        return {"status": 200, "body": [o.to_dict() for o in orders]}

                    def create_order(self, body):
                        user = db.query(User).get(body["user_id"])
                        if not user:
                            return {"status": 400, "body": {"error": "invalid user"}}
                        order = Order(user_id=body["user_id"], items=body["items"])
                        db.save(order)
                        return {"status": 201, "body": order.to_dict()}

                    def cancel_order(self, order_id):
                        order = db.query(Order).get(order_id)
                        if not order:
                            return {"status": 404, "body": {"error": "order not found"}}
                        order.status = "cancelled"
                        db.save(order)
                        return {"status": 200, "body": order.to_dict()}
            ''').strip(),
            "models/user.py": textwrap.dedent('''
                """User model."""
                class User:
                    def __init__(self, name, email, id=None):
                        self.id = id
                        self.name = name
                        self.email = email

                    def to_dict(self):
                        return {"id": self.id, "name": self.name, "email": self.email}
            ''').strip(),
            "models/order.py": textwrap.dedent('''
                """Order model."""
                class Order:
                    def __init__(self, user_id, items, id=None, status="pending"):
                        self.id = id
                        self.user_id = user_id
                        self.items = items
                        self.status = status

                    def to_dict(self):
                        return {"id": self.id, "user_id": self.user_id,
                                "items": self.items, "status": self.status}
            ''').strip(),
            "models/database.py": textwrap.dedent('''
                """Simple in-memory database."""
                class Database:
                    def __init__(self):
                        self._data = {}
                        self._next_id = 1

                    def query(self, model):
                        return Query(self, model)

                    def save(self, obj):
                        if obj.id is None:
                            obj.id = self._next_id
                            self._next_id += 1
                        self._data[(type(obj).__name__, obj.id)] = obj

                class Query:
                    def __init__(self, db, model):
                        self.db = db
                        self.model = model

                    def all(self):
                        return [v for (name, _), v in self.db._data.items() if name == self.model.__name__]

                    def get(self, obj_id):
                        return next((v for (name, i), v in self.db._data.items()
                                     if name == self.model.__name__ and i == obj_id), None)

                db = Database()
            ''').strip(),
        },
        "questions": [
            {
                "question": "Where is the function `route_request` defined?",
                "answer": "The function `route_request` is defined in `routes.py`.",
                "keywords": ["routes.py", "route_request"],
            },
            {
                "question": "What does `UserController.create_user` return on success?",
                "answer": "It returns a dict with status 201 and the user data in the body, like {\"status\": 201, \"body\": user.to_dict()}.",
                "keywords": ["201", "user", "to_dict", "body"],
            },
            {
                "question": "Trace the data flow: when a GET request comes in for /users/{id}, which functions are called in order?",
                "answer": "1. route_request in routes.py receives the request, 2. it calls user_controller.get_user(user_id), 3. get_user calls db.query(User).get(user_id), 4. if found, it calls user.to_dict() and returns the result.",
                "keywords": ["route_request", "get_user", "db.query", "to_dict", "User"],
            },
            {
                "question": "What modules does `routes.py` import?",
                "answer": "routes.py imports UserController from controllers.user_controller and OrderController from controllers.order_controller.",
                "keywords": ["UserController", "OrderController", "controllers.user_controller", "controllers.order_controller"],
            },
            {
                "question": "When creating an order, what validation does OrderController.create_order perform?",
                "answer": "It checks if the user with the given user_id exists by calling db.query(User).get(body['user_id']). If the user is not found, it returns a 400 error with 'invalid user'.",
                "keywords": ["user", "db.query", "User", "get", "400", "invalid user"],
            },
            {
                "question": "Trace the data flow: when a POST request to /orders is received, which functions are called in order?",
                "answer": "1. route_request in routes.py, 2. order_controller.create_order(body), 3. db.query(User).get(body['user_id']) to validate the user, 4. if valid, creates an Order and calls db.save(order), 5. returns order.to_dict().",
                "keywords": ["route_request", "create_order", "db.query", "User", "get", "Order", "db.save", "to_dict"],
            },
        ],
    },

    # ── Domain 2: CLI Tool (commands/handlers/utils) ──
    "cli_tool": {
        "files": {
            "main.py": textwrap.dedent('''
                """CLI entry point."""
                import sys
                from commands.parse_args import parse_args
                from commands.dispatcher import dispatch

                def main(argv=None):
                    """Main entry point for the CLI tool."""
                    if argv is None:
                        argv = sys.argv[1:]
                    args = parse_args(argv)
                    result = dispatch(args)
                    print(result)
                    return 0

                if __name__ == "__main__":
                    sys.exit(main())
            ''').strip(),
            "commands/parse_args.py": textwrap.dedent('''
                """Argument parsing for the CLI tool."""
                def parse_args(argv):
                    """Parse command-line arguments into a dict."""
                    if not argv:
                        return {"command": "help"}
                    command = argv[0]
                    args = {}
                    i = 1
                    while i < len(argv):
                        if argv[i].startswith("--"):
                            key = argv[i][2:]
                            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                                args[key] = argv[i + 1]
                                i += 2
                            else:
                                args[key] = True
                                i += 1
                        else:
                            i += 1
                    args["command"] = command
                    return args
            ''').strip(),
            "commands/dispatcher.py": textwrap.dedent('''
                """Command dispatcher."""
                from handlers.file_handler import FileHandler
                from handlers.search_handler import SearchHandler
                from handlers.stats_handler import StatsHandler

                def dispatch(args):
                    """Dispatch to the appropriate handler based on command."""
                    command = args.get("command", "help")
                    if command == "file":
                        return FileHandler().handle(args)
                    elif command == "search":
                        return SearchHandler().handle(args)
                    elif command == "stats":
                        return StatsHandler().handle(args)
                    elif command == "help":
                        return "Available commands: file, search, stats"
                    else:
                        return f"Unknown command: {command}"
            ''').strip(),
            "handlers/file_handler.py": textwrap.dedent('''
                """Handler for file operations."""
                from utils.file_utils import read_file, write_file, count_lines

                class FileHandler:
                    def handle(self, args):
                        action = args.get("action", "read")
                        if action == "read":
                            path = args.get("path", "")
                            content = read_file(path)
                            return content
                        elif action == "write":
                            path = args.get("path", "")
                            content = args.get("content", "")
                            write_file(path, content)
                            return f"Wrote to {path}"
                        elif action == "count":
                            path = args.get("path", "")
                            return f"{count_lines(path)} lines"
                        else:
                            return f"Unknown file action: {action}"
            ''').strip(),
            "handlers/search_handler.py": textwrap.dedent('''
                """Handler for search operations."""
                from utils.search_utils import linear_search, binary_search

                class SearchHandler:
                    def handle(self, args):
                        algorithm = args.get("algorithm", "linear")
                        data = args.get("data", "").split(",")
                        target = args.get("target", "")
                        if algorithm == "binary":
                            result = binary_search(sorted(data), target)
                        else:
                            result = linear_search(data, target)
                        return f"Found at index: {result}" if result >= 0 else "Not found"
            ''').strip(),
            "handlers/stats_handler.py": textwrap.dedent('''
                """Handler for statistics operations."""
                from utils.stats_utils import mean, median, mode

                class StatsHandler:
                    def handle(self, args):
                        data = [float(x) for x in args.get("data", "").split(",")]
                        operation = args.get("operation", "mean")
                        if operation == "mean":
                            return f"Mean: {mean(data)}"
                        elif operation == "median":
                            return f"Median: {median(data)}"
                        elif operation == "mode":
                            return f"Mode: {mode(data)}"
                        else:
                            return f"Unknown operation: {operation}"
            ''').strip(),
            "utils/file_utils.py": textwrap.dedent('''
                """File utility functions."""
                def read_file(path):
                    with open(path, "r") as f:
                        return f.read()

                def write_file(path, content):
                    with open(path, "w") as f:
                        f.write(content)

                def count_lines(path):
                    with open(path, "r") as f:
                        return sum(1 for _ in f)
            ''').strip(),
            "utils/search_utils.py": textwrap.dedent('''
                """Search utility functions."""
                def linear_search(data, target):
                    for i, item in enumerate(data):
                        if item == target:
                            return i
                    return -1

                def binary_search(data, target):
                    lo, hi = 0, len(data) - 1
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if data[mid] == target:
                            return mid
                        elif data[mid] < target:
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    return -1
            ''').strip(),
            "utils/stats_utils.py": textwrap.dedent('''
                """Statistics utility functions."""
                def mean(data):
                    return sum(data) / len(data) if data else 0

                def median(data):
                    if not data:
                        return 0
                    sorted_data = sorted(data)
                    n = len(sorted_data)
                    if n % 2 == 0:
                        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
                    return sorted_data[n//2]

                def mode(data):
                    if not data:
                        return 0
                    from collections import Counter
                    return Counter(data).most_common(1)[0][0]
            ''').strip(),
        },
        "questions": [
            {
                "question": "Where is the function `dispatch` defined?",
                "answer": "The function `dispatch` is defined in `commands/dispatcher.py`.",
                "keywords": ["dispatcher.py", "dispatch"],
            },
            {
                "question": "What does `parse_args` return when no arguments are provided?",
                "answer": "It returns a dict with command set to 'help': {\"command\": \"help\"}.",
                "keywords": ["help", "command", "dict"],
            },
            {
                "question": "Trace the data flow: when the user runs `main(['stats', '--data', '1,2,3', '--operation', 'median']`, which functions are called in order?",
                "answer": "1. main() in main.py, 2. parse_args() in commands/parse_args.py, 3. dispatch() in commands/dispatcher.py, 4. StatsHandler().handle() in handlers/stats_handler.py, 5. median() in utils/stats_utils.py.",
                "keywords": ["main", "parse_args", "dispatch", "StatsHandler", "handle", "median"],
            },
            {
                "question": "What modules does `commands/dispatcher.py` import?",
                "answer": "It imports FileHandler from handlers.file_handler, SearchHandler from handlers.search_handler, and StatsHandler from handlers.stats_handler.",
                "keywords": ["FileHandler", "SearchHandler", "StatsHandler", "handlers"],
            },
            {
                "question": "What does `FileHandler.handle` do when action is 'count'?",
                "answer": "It calls count_lines(path) from utils.file_utils and returns a string like '{n} lines' where n is the line count.",
                "keywords": ["count_lines", "lines", "file_utils"],
            },
            {
                "question": "Trace the data flow: when the user runs `main(['search', '--algorithm', 'binary', '--data', 'a,b,c', '--target', 'b'])`, which functions are called in order?",
                "answer": "1. main() in main.py, 2. parse_args() in commands/parse_args.py, 3. dispatch() in commands/dispatcher.py, 4. SearchHandler().handle() in handlers/search_handler.py, 5. sorted() on the data, 6. binary_search() in utils/search_utils.py.",
                "keywords": ["main", "parse_args", "dispatch", "SearchHandler", "handle", "binary_search", "sorted"],
            },
        ],
    },

    # ── Domain 3: Data Pipeline (extract/transform/load) ──
    "data_pipeline": {
        "files": {
            "pipeline.py": textwrap.dedent('''
                """Main data pipeline orchestrator."""
                from extract.csv_extractor import CSVExtractor
                from extract.json_extractor import JSONExtractor
                from transform.cleaner import DataCleaner
                from transform.aggregator import DataAggregator
                from load.db_loader import DBLoader
                from load.file_loader import FileLoader

                def run_pipeline(source_type, source_path, config):
                    """Run the full ETL pipeline."""
                    # Extract
                    if source_type == "csv":
                        extractor = CSVExtractor()
                    elif source_type == "json":
                        extractor = JSONExtractor()
                    else:
                        raise ValueError(f"Unknown source type: {source_type}")
                    raw_data = extractor.extract(source_path)

                    # Transform
                    cleaner = DataCleaner(config.get("clean_rules", {}))
                    cleaned = cleaner.clean(raw_data)

                    aggregator = DataAggregator(config.get("group_by", []))
                    aggregated = aggregator.aggregate(cleaned)

                    # Load
                    if config.get("output_type") == "db":
                        loader = DBLoader(config.get("db_url", ""))
                    else:
                        loader = FileLoader(config.get("output_path", "output.json"))
                    loader.load(aggregated)

                    return len(aggregated)
            ''').strip(),
            "extract/csv_extractor.py": textwrap.dedent('''
                """CSV data extractor."""
                import csv

                class CSVExtractor:
                    def extract(self, path):
                        """Extract data from a CSV file."""
                        with open(path, "r") as f:
                            reader = csv.DictReader(f)
                            return list(reader)
            ''').strip(),
            "extract/json_extractor.py": textwrap.dedent('''
                """JSON data extractor."""
                import json

                class JSONExtractor:
                    def extract(self, path):
                        """Extract data from a JSON file."""
                        with open(path, "r") as f:
                            return json.load(f)
            ''').strip(),
            "transform/cleaner.py": textwrap.dedent('''
                """Data cleaning transformer."""
                class DataCleaner:
                    def __init__(self, rules):
                        self.rules = rules

                    def clean(self, data):
                        """Apply cleaning rules to the data."""
                        cleaned = []
                        for row in data:
                            row = dict(row)
                            for field, rule in self.rules.items():
                                if field in row:
                                    if rule == "strip":
                                        row[field] = row[field].strip()
                                    elif rule == "lower":
                                        row[field] = row[field].lower()
                                    elif rule == "int":
                                        row[field] = int(row[field])
                            if None not in row.values():
                                cleaned.append(row)
                        return cleaned
            ''').strip(),
            "transform/aggregator.py": textwrap.dedent('''
                """Data aggregation transformer."""
                from collections import defaultdict

                class DataAggregator:
                    def __init__(self, group_by):
                        self.group_by = group_by

                    def aggregate(self, data):
                        """Group data by specified fields and count."""
                        if not self.group_by:
                            return data
                        groups = defaultdict(list)
                        for row in data:
                            key = tuple(row.get(f) for f in self.group_by)
                            groups[key].append(row)
                        result = []
                        for key, rows in groups.items():
                            entry = dict(zip(self.group_by, key))
                            entry["count"] = len(rows)
                            result.append(entry)
                        return result
            ''').strip(),
            "load/db_loader.py": textwrap.dedent('''
                """Database loader."""
                class DBLoader:
                    def __init__(self, db_url):
                        self.db_url = db_url

                    def load(self, data):
                        """Load data into the database."""
                        # In a real implementation, this would connect to a DB
                        # For now, we just simulate it
                        return len(data)
            ''').strip(),
            "load/file_loader.py": textwrap.dedent('''
                """File loader."""
                import json

                class FileLoader:
                    def __init__(self, output_path):
                        self.output_path = output_path

                    def load(self, data):
                        """Load data into a JSON file."""
                        with open(self.output_path, "w") as f:
                            json.dump(data, f)
                        return len(data)
            ''').strip(),
        },
        "questions": [
            {
                "question": "Where is the function `run_pipeline` defined?",
                "answer": "The function `run_pipeline` is defined in `pipeline.py`.",
                "keywords": ["pipeline.py", "run_pipeline"],
            },
            {
                "question": "What does `CSVExtractor.extract` return?",
                "answer": "It returns a list of dicts, where each dict represents a row from the CSV file (using csv.DictReader).",
                "keywords": ["list", "dict", "DictReader", "row"],
            },
            {
                "question": "Trace the data flow: when run_pipeline is called with source_type='csv', which functions are called in order?",
                "answer": "1. run_pipeline() in pipeline.py, 2. CSVExtractor().extract(source_path) in extract/csv_extractor.py, 3. DataCleaner().clean(raw_data) in transform/cleaner.py, 4. DataAggregator().aggregate(cleaned) in transform/aggregator.py, 5. DBLoader or FileLoader .load(aggregated) depending on config.",
                "keywords": ["run_pipeline", "CSVExtractor", "extract", "DataCleaner", "clean", "DataAggregator", "aggregate", "load"],
            },
            {
                "question": "What modules does `pipeline.py` import?",
                "answer": "It imports CSVExtractor from extract.csv_extractor, JSONExtractor from extract.json_extractor, DataCleaner from transform.cleaner, DataAggregator from transform.aggregator, DBLoader from load.db_loader, and FileLoader from load.file_loader.",
                "keywords": ["CSVExtractor", "JSONExtractor", "DataCleaner", "DataAggregator", "DBLoader", "FileLoader"],
            },
            {
                "question": "What does `DataCleaner.clean` do when a rule is 'strip'?",
                "answer": "It calls row[field].strip() to remove leading and trailing whitespace from that field's value.",
                "keywords": ["strip", "whitespace", "row", "field"],
            },
            {
                "question": "How does `DataAggregator.aggregate` work when group_by is specified?",
                "answer": "It groups rows by the specified fields using a defaultdict(list), creating a key tuple from the group_by fields for each row. Then for each group, it creates an entry dict with the group_by field values and a 'count' field containing the number of rows in that group.",
                "keywords": ["defaultdict", "group", "key", "tuple", "count", "group_by"],
            },
        ],
    },
}


# ── Distractor files (irrelevant code to test if model can focus) ──

DISTRACTORS = {
    "utils/logger.py": textwrap.dedent('''
        """Logging utility (not relevant to the task)."""
        import datetime

        class Logger:
            def __init__(self, name):
                self.name = name
                self.entries = []

            def log(self, level, message):
                entry = f"[{datetime.datetime.now()}] {level}: {message}"
                self.entries.append(entry)
                return entry

            def get_entries(self):
                return self.entries
    ''').strip(),
    "utils/validator.py": textwrap.dedent('''
        """Data validation utility (not relevant to the task)."""
        def validate_email(email):
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, email))

        def validate_phone(phone):
            digits = ''.join(c for c in phone if c.isdigit())
            return len(digits) == 10
    ''').strip(),
    "utils/cache.py": textwrap.dedent('''
        """Simple cache utility (not relevant to the task)."""
        class Cache:
            def __init__(self, max_size=100):
                self.max_size = max_size
                self._store = {}
                self._order = []

            def get(self, key):
                if key in self._store:
                    return self._store[key]
                return None

            def set(self, key, value):
                if key not in self._store and len(self._store) >= self.max_size:
                    oldest = self._order.pop(0)
                    del self._store[oldest]
                self._store[key] = value
                self._order.append(key)
    ''').strip(),
    "utils/formatter.py": textwrap.dedent('''
        """Text formatting utility (not relevant to the task)."""
        def format_table(headers, rows):
            widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]
            lines = [" | ".join(str(h).ljust(w) for h, w in zip(headers, widths))]
            for row in rows:
                lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))
            return "\\n".join(lines)
    ''').strip(),
}


@register_env
class CodebaseNavEnv(AgenticEnv):
    name = "codebase_nav"
    skill = "Understanding and navigating a multi-file codebase"
    difficulty_tiers = ["easy", "medium", "hard"]

    def gen_params(self, rng, difficulty="medium"):
        domain_name = rng.choice(list(DOMAINS.keys()))
        domain = DOMAINS[domain_name]
        n_questions = {"easy": 3, "medium": 4, "hard": 6}[difficulty]
        n_distractors = {"easy": 0, "medium": 1, "hard": 2}[difficulty]

        # Select questions based on difficulty
        all_questions = domain["questions"]
        if difficulty == "easy":
            # Pick simpler questions (definition, import questions)
            selected = [q for q in all_questions if "Trace" not in q["question"]][:n_questions]
            if len(selected) < n_questions:
                selected = all_questions[:n_questions]
        elif difficulty == "medium":
            selected = all_questions[:n_questions]
        else:
            selected = all_questions[:n_questions]

        distractor_keys = rng.sample(list(DISTRACTORS.keys()), n_distractors) if n_distractors else []

        return {
            "domain": domain_name,
            "difficulty": difficulty,
            "n_questions": len(selected),
            "question_indices": [all_questions.index(q) for q in selected],
            "distractor_keys": distractor_keys,
            "seed": rng.randint(0, 999999),
        }

    def gen_codebase(self, params, rng):
        domain = DOMAINS[params["domain"]]
        codebase = dict(domain["files"])
        for key in params.get("distractor_keys", []):
            codebase[key] = DISTRACTORS[key]
        return codebase

    def gen_task(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        questions = [domain["questions"][i] for i in params["question_indices"]]

        lines = []
        lines.append("You are a software engineer analyzing an unfamiliar codebase.")
        lines.append("")
        lines.append("You are given a multi-file Python codebase. Your task is to:")
        lines.append("1. Read through all the files carefully")
        lines.append("2. Answer the questions below about the codebase")
        lines.append("3. Provide specific file names, function names, and line references in your answers")
        lines.append("4. For trace questions, list the functions called in order")
        lines.append("")
        lines.append("=== CODEBASE FILES ===")
        lines.append("")
        for filename, content in sorted(codebase.items()):
            lines.append(f"--- {filename} ---")
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")
        lines.append("=== QUESTIONS ===")
        lines.append("")
        for i, q in enumerate(questions, 1):
            lines.append(f"Q{i}: {q['question']}")
            lines.append("")
        lines.append("Answer ALL questions. For each question, provide a clear, specific answer.")
        lines.append("Format your answer as a numbered list matching the questions.")
        lines.append("")
        lines.append("Provide your answers in the following format:")
        lines.append("<reasoning>")
        lines.append("...read each relevant file, trace the code, explain your analysis...")
        lines.append("</reasoning>")
        lines.append("<answer>")
        lines.append("1. [answer to Q1]")
        lines.append("2. [answer to Q2]")
        lines.append("...")
        lines.append("</answer>")
        return "\n".join(lines)

    def gen_solution(self, params, codebase):
        domain = DOMAINS[params["domain"]]
        questions = [domain["questions"][i] for i in params["question_indices"]]
        answer_lines = []
        for i, q in enumerate(questions, 1):
            answer_lines.append(f"{i}. {q['answer']}")
        return {"answers.txt": "\n".join(answer_lines)}

    def gen_reasoning(self, params, codebase, solution):
        domain = DOMAINS[params["domain"]]
        questions = [domain["questions"][i] for i in params["question_indices"]]
        files = codebase

        reasoning_lines = [
            "Let me carefully read through each file in the codebase to answer the questions.",
            "",
        ]

        # List all files and summarize
        reasoning_lines.append("Files in the codebase:")
        for filename in sorted(files.keys()):
            content = files[filename]
            # Extract function/class definitions
            defs = []
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("def ") or stripped.startswith("class "):
                    defs.append(stripped.split("(")[0] + "(" if "(" in stripped else stripped)
            if defs:
                reasoning_lines.append(f"  {filename}: defines {', '.join(defs)}")
            else:
                reasoning_lines.append(f"  {filename}: utility/config file")
        reasoning_lines.append("")

        # Answer each question with evidence
        for i, q in enumerate(questions, 1):
            reasoning_lines.append(f"Question {i}: {q['question']}")
            reasoning_lines.append("")

            if "Where is" in q["question"]:
                # Find the function definition
                func_name = q["question"].split("`")[1] if "`" in q["question"] else ""
                for filename, content in sorted(files.items()):
                    for line_num, line in enumerate(content.split("\n"), 1):
                        if f"def {func_name}" in line or f"class {func_name}" in line:
                            reasoning_lines.append(f"  I found `def {func_name}` in {filename} at line {line_num}.")
                            reasoning_lines.append(f"  The line is: {line.strip()}")
                            break
                reasoning_lines.append(f"  Answer: {q['answer']}")
            elif "import" in q["question"].lower():
                # Find imports in the relevant file
                target_file = q["question"].split("`")[1] if "`" in q["question"] else ""
                if target_file in files:
                    reasoning_lines.append(f"  Looking at {target_file}, I check the import statements at the top:")
                    for line in files[target_file].split("\n"):
                        if line.strip().startswith("import ") or line.strip().startswith("from "):
                            reasoning_lines.append(f"    {line.strip()}")
                reasoning_lines.append(f"  Answer: {q['answer']}")
            elif "Trace" in q["question"]:
                reasoning_lines.append("  To trace the data flow, I need to follow the call chain:")
                reasoning_lines.append(f"  Answer: {q['answer']}")
            elif "return" in q["question"].lower() or "What does" in q["question"]:
                reasoning_lines.append(f"  I need to read the relevant function carefully:")
                reasoning_lines.append(f"  Answer: {q['answer']}")
            else:
                reasoning_lines.append(f"  Answer: {q['answer']}")

            reasoning_lines.append("")

        reasoning_lines.append("I have now answered all questions with specific file references and function names.")
        return "\n".join(reasoning_lines)

    def grade(self, params, codebase, response):
        domain = DOMAINS[params["domain"]]
        questions = [domain["questions"][i] for i in params["question_indices"]]

        answer = extract_answer(response)
        has_reasoning = bool(extract_reasoning(response))

        if not answer:
            return 0.0, {
                "reason": "no answer found in response",
                "has_reasoning": has_reasoning,
            }

        # Parse numbered answers from the response
        # Try to split by numbered lines
        answer_lines = []
        current_num = None
        current_text = []

        for line in answer.split("\n"):
            stripped = line.strip()
            # Check if line starts with a number followed by . or )
            if stripped and stripped[0].isdigit():
                parts = stripped.split(".", 1)
                if parts[0].isdigit():
                    if current_num is not None:
                        answer_lines.append(" ".join(current_text).strip())
                    current_num = int(parts[0])
                    current_text = [parts[1].strip()] if len(parts) > 1 else []
                    continue
            current_text.append(stripped)

        if current_num is not None:
            answer_lines.append(" ".join(current_text).strip())

        # If we couldn't parse numbered answers, try splitting by double newlines
        if len(answer_lines) < len(questions):
            answer_lines = [a.strip() for a in answer.split("\n\n") if a.strip()]

        # Score each question
        total_score = 0.0
        question_scores = []

        for i, q in enumerate(questions):
            model_answer = answer_lines[i] if i < len(answer_lines) else ""

            # Text similarity score
            sim = text_similarity(model_answer, q["answer"])

            # Keyword matching score
            keywords = q["keywords"]
            model_lower = model_answer.lower()
            keywords_found = sum(1 for kw in keywords if kw.lower() in model_lower)
            keyword_score = keywords_found / len(keywords) if keywords else 0

            # Combined score for this question: 0.4 * similarity + 0.6 * keyword_score
            q_score = 0.4 * sim + 0.6 * keyword_score
            question_scores.append({
                "question": q["question"],
                "model_answer": model_answer[:200],
                "expected_answer": q["answer"][:200],
                "similarity": round(sim, 3),
                "keyword_score": round(keyword_score, 3),
                "keywords_found": keywords_found,
                "keywords_total": len(keywords),
                "score": round(q_score, 3),
            })
            total_score += q_score

        # Normalize to [0, 1]
        final_score = total_score / len(questions) if questions else 0.0

        breakdown = {
            "domain": params["domain"],
            "difficulty": params["difficulty"],
            "n_questions": len(questions),
            "n_answered": len(answer_lines),
            "has_reasoning": has_reasoning,
            "question_scores": question_scores,
            "final_score": round(final_score, 3),
        }

        return final_score, breakdown
