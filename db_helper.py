
import sqlite3
from datetime import datetime
import hashlib

DB_NAME = "smartland.db"

# -----------------------------
# UTILS
# -----------------------------
def get_db():
    return sqlite3.connect(DB_NAME)

# -----------------------------
# INIT DB
# -----------------------------
def init_db():
    conn = get_db()
    c = conn.cursor()

    # USERS TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            password_hash TEXT
        )
    """)

    # PREDICTIONS TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            degree TEXT,
            major TEXT,
            skill1 TEXT,
            skill2 TEXT,
            certification TEXT,
            experience_years INTEGER,
            project_count INTEGER,
            internship TEXT,
            experience_level TEXT,
            predicted_label TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # FEEDBACK TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            rating INTEGER,
            comments TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

# -----------------------------
# USER FUNCTIONS
# -----------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                  (username, email, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[1] == hash_password(password):
        return user[0]
    return None

# -----------------------------
# PREDICTION FUNCTIONS
# -----------------------------
def insert_prediction(user_id, row):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (
            user_id, timestamp, degree, major, skill1, skill2, certification,
            experience_years, project_count, internship, experience_level, predicted_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        row.get("Degree"),
        row.get("Major"),
        row.get("Skill1"),
        row.get("Skill2"),
        row.get("Certification"),
        row.get("ExperienceYears"),
        row.get("ProjectCount"),
        row.get("Internship"),
        row.get("ExperienceLevel"),
        row.get("predicted_label")
    ))
    conn.commit()
    conn.close()

def fetch_history(user_id, limit=100):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return rows

# -----------------------------
# FEEDBACK FUNCTIONS
# -----------------------------
def insert_feedback(user_id, rating, comments):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO feedback (user_id, timestamp, rating, comments) VALUES (?, ?, ?, ?)",
              (user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rating, comments))
    conn.commit()
    conn.close()
def clear_history(user_id):
    """
    Deletes all prediction records for a given user.
    
    Args:
        user_id (int): The ID of the logged-in user.
    """
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        conn.commit()
    except Exception as e:
        print(f"Error clearing history: {e}")
    finally:
        conn.close()

