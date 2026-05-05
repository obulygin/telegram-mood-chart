"""
Database Manager for Telegram Mood Chart.
Handles SQLite operations: schema creation, upserting messages, caching embeddings, and retrieving stats.
"""
import sqlite3
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path: str = "mood.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self):
        """Create tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Table: Contacts (Interlocutors)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                is_bot BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table: Messages (Raw data + basic parsing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT, -- ID from JSON export
                contact_id INTEGER,
                date TIMESTAMP,
                text TEXT,
                has_media BOOLEAN DEFAULT FALSE,
                is_edit BOOLEAN DEFAULT FALSE,
                raw_json JSON,
                inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(contact_id) REFERENCES contacts(id),
                UNIQUE(external_id, contact_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_date ON messages(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_contact ON messages(contact_id)")

        # Table: Analysis Results (Scores per message)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_scores (
                message_id INTEGER PRIMARY KEY,
                sentiment_score REAL,
                anxiety_score REAL,
                stress_score REAL,
                i_focus_score REAL,
                we_focus_score REAL,
                future_ratio REAL,
                social_breadth REAL,
                initiation_score REAL,
                lexical_diversity REAL,
                question_ratio REAL,
                embedding_vector BLOB, -- Stored as bytes (pickle or numpy tobytes)
                model_version TEXT,
                is_llm_analyzed BOOLEAN DEFAULT FALSE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
            )
        """)

        # Table: Daily Aggregates (Pre-calculated stats)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date DATE PRIMARY KEY,
                avg_sentiment REAL,
                avg_anxiety REAL,
                avg_stress REAL,
                msg_count INTEGER,
                ci_lower_sentiment REAL,
                ci_upper_sentiment REAL,
                bootstrap_iterations INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table: Change Points (Detected events)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS change_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_date DATE,
                metric_name TEXT,
                direction TEXT, -- 'up' or 'down'
                significance_p_value REAL,
                magnitude REAL,
                algorithm TEXT, -- 'PELT', 'BinarySegmentation'
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table: Processing Log (To track incremental updates)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                file_hash TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                new_messages_count INTEGER,
                status TEXT -- 'success', 'partial', 'error'
            )
        """)

        self.conn.commit()

    def get_or_create_contact(self, name: str, is_bot: bool = False) -> int:
        """Get contact ID or create if not exists."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM contacts WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return row['id']
        
        cursor.execute("INSERT INTO contacts (name, is_bot) VALUES (?, ?)", (name, is_bot))
        self.conn.commit()
        return cursor.lastrowid

    def message_exists(self, external_id: str, contact_id: int) -> bool:
        """Check if message already in DB."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM messages WHERE external_id = ? AND contact_id = ?",
            (external_id, contact_id)
        )
        return cursor.fetchone() is not None

    def insert_message(self, external_id: str, contact_id: int, date: datetime, 
                       text: str, has_media: bool, is_edit: bool, raw_json: str = None) -> int:
        """Insert a new message. Returns message_id."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO messages (external_id, contact_id, date, text, has_media, is_edit, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(external_id, contact_id) DO NOTHING
            """, (external_id, contact_id, date, text, has_media, is_edit, raw_json))
            self.conn.commit()
            
            # Get the ID (either newly inserted or existing if conflict happened but we ignored it)
            cursor.execute(
                "SELECT id FROM messages WHERE external_id = ? AND contact_id = ?",
                (external_id, contact_id)
            )
            row = cursor.fetchone()
            return row['id'] if row else -1
        except sqlite3.IntegrityError:
            # Race condition or duplicate, fetch existing
            cursor.execute(
                "SELECT id FROM messages WHERE external_id = ? AND contact_id = ?",
                (external_id, contact_id)
            )
            row = cursor.fetchone()
            return row['id'] if row else -1

    def get_unanalyzed_messages(self, limit: int = 1000) -> List[Dict]:
        """Fetch messages that haven't been analyzed yet."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT m.id, m.text, m.date, c.name as contact_name
            FROM messages m
            JOIN contacts c ON m.contact_id = c.id
            LEFT JOIN message_scores s ON m.id = s.message_id
            WHERE s.message_id IS NULL
            ORDER BY m.date ASC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]

    def save_analysis_result(self, message_id: int, scores: Dict[str, float], 
                             embedding: Optional[bytes] = None, is_llm: bool = False):
        """Save analysis scores for a message."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO message_scores 
            (message_id, sentiment_score, anxiety_score, stress_score, i_focus_score, 
             we_focus_score, future_ratio, social_breadth, initiation_score, 
             lexical_diversity, question_ratio, embedding_vector, is_llm_analyzed, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            message_id,
            scores.get('sentiment', 0.0),
            scores.get('anxiety', 0.0),
            scores.get('stress', 0.0),
            scores.get('i_focus', 0.0),
            scores.get('we_focus', 0.0),
            scores.get('future_ratio', 0.0),
            scores.get('social_breadth', 0.0),
            scores.get('initiation', 0.0),
            scores.get('lexical_diversity', 0.0),
            scores.get('question_ratio', 0.0),
            embedding,
            is_llm
        ))
        self.conn.commit()

    def get_daily_aggregates(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Retrieve daily stats as DataFrame."""
        query = "SELECT * FROM daily_stats"
        params = []
        
        if start_date:
            query += " WHERE date >= ?"
            params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
        elif end_date:
            query += " WHERE date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date ASC"
        
        return pd.read_sql_query(query, self.conn, params=params)

    def save_daily_stats(self, date_str: str, stats: Dict[str, Any]):
        """Save or update daily aggregated statistics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO daily_stats 
            (date, avg_sentiment, avg_anxiety, avg_stress, msg_count, 
             ci_lower_sentiment, ci_upper_sentiment, bootstrap_iterations, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            date_str,
            stats.get('avg_sentiment'),
            stats.get('avg_anxiety'),
            stats.get('avg_stress'),
            stats.get('msg_count'),
            stats.get('ci_lower'),
            stats.get('ci_upper'),
            stats.get('bootstrap_iters', 1000)
        ))
        self.conn.commit()

    def log_processing_run(self, source_file: str, file_hash: str, count: int, status: str):
        """Log a processing run."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO processing_log (source_file, file_hash, new_messages_count, status)
            VALUES (?, ?, ?, ?)
        """, (source_file, file_hash, count, status))
        self.conn.commit()

    def get_total_message_count(self) -> int:
        """Get total number of messages."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages")
        return cursor.fetchone()[0]

    def get_analyzed_count(self) -> int:
        """Get number of analyzed messages."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM message_scores")
        return cursor.fetchone()[0]

    def close(self):
        self.conn.close()
