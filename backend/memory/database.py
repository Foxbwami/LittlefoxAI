import os
import sqlite3

from backend.core import config

legacy_path = os.path.join(config.BASE_DIR, "memory.db")
db_path = config.PROFILE_DB_PATH
if os.path.exists(legacy_path) and not os.path.exists(db_path):
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.replace(legacy_path, db_path)
    except Exception:
        db_path = legacy_path

os.makedirs(os.path.dirname(db_path), exist_ok=True)
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    personality TEXT
)
"""
)
conn.commit()


def save_user(user_id, name, personality):
    cursor.execute(
        "INSERT OR REPLACE INTO users VALUES (?, ?, ?)",
        (user_id, name, personality),
    )
    conn.commit()


def get_user(user_id):
    cursor.execute(
        "SELECT name, personality FROM users WHERE user_id=?",
        (user_id,),
    )
    return cursor.fetchone()
