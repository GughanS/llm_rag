import sqlite3
import datetime
import threading
from pathlib import Path

# Thread-local storage since sqlite3 connections aren't thread-safe
_local = threading.local()

def get_db():
    if not hasattr(_local, "db"):
        db_path = Path("logs")
        db_path.mkdir(exist_ok=True)
        conn = sqlite3.connect(db_path / "requests.db", check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL") # High concurrency
        conn.execute('''
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_name TEXT,
                prompt TEXT,
                response TEXT,
                latency_ms REAL,
                token_count INTEGER,
                status_code INTEGER
            )
        ''')
        conn.commit()
        _local.db = conn
    return _local.db

def log_request_async(model_name: str, prompt: str, response: str, latency_ms: float, token_count: int, status_code: int):
    """Log to SQLite in a fire-and-forget background thread to avoid blocking the API response."""
    def _insert():
        try:
            db = get_db()
            db.execute(
                """INSERT INTO request_logs 
                   (timestamp, model_name, prompt, response, latency_ms, token_count, status_code) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (datetime.datetime.now(datetime.timezone.utc).isoformat(), model_name, prompt, response, latency_ms, token_count, status_code)
            )
            db.commit()
        except Exception as e:
            print(f"Failed to log to SQLite: {e}")
            
    threading.Thread(target=_insert, daemon=True).start()
