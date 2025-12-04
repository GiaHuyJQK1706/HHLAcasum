"""
Database Layer - SQLite for managing summaries and texts
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading
import uuid


class DataAccess:
    """SQLite database management for summaries"""
    
    def __init__(self, db_path: str = "./data/summarizer.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables"""
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Create users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create texts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS texts (
                        text_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        length INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                """)
                
                # Create summaries table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS summaries (
                        summary_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        text_id TEXT NOT NULL,
                        original_content TEXT NOT NULL,
                        summary_content TEXT NOT NULL,
                        summary_length TEXT NOT NULL,
                        original_length INTEGER NOT NULL,
                        summary_text_length INTEGER NOT NULL,
                        compression_ratio REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id),
                        FOREIGN KEY (text_id) REFERENCES texts(text_id)
                    )
                """)
                
                conn.commit()
                conn.close()
                
                print("✅ Database initialized")
        except Exception as e:
            raise Exception(f"Failed to initialize database: {str(e)}")
    
    def get_or_create_user(self, user_id: Optional[str] = None) -> str:
        """
        Get or create a user
        
        Args:
            user_id: User ID (if None, generate new)
            
        Returns:
            User ID
        """
        try:
            with self._lock:
                if user_id is None:
                    user_id = str(uuid.uuid4())
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Check if user exists
                cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
                if cursor.fetchone():
                    conn.close()
                    return user_id
                
                # Create new user
                cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
                conn.commit()
                conn.close()
                
                return user_id
        except Exception as e:
            raise Exception(f"Failed to get or create user: {str(e)}")
    
    def save_summary(self, user_id: str, original_text: str, summary_text: str, 
                    summary_length: str) -> str:
        """
        Save summary to database
        
        Args:
            user_id: User ID
            original_text: Original text
            summary_text: Generated summary
            summary_length: "short" or "long"
            
        Returns:
            Summary ID
        """
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Generate IDs
                text_id = str(uuid.uuid4())
                summary_id = str(uuid.uuid4())
                
                # Calculate metrics
                original_length = len(original_text)
                summary_text_length = len(summary_text)
                compression_ratio = (summary_text_length / original_length * 100) if original_length > 0 else 0
                
                # Save original text
                cursor.execute("""
                    INSERT INTO texts (text_id, user_id, content, length)
                    VALUES (?, ?, ?, ?)
                """, (text_id, user_id, original_text, original_length))
                
                # Save summary
                cursor.execute("""
                    INSERT INTO summaries 
                    (summary_id, user_id, text_id, original_content, summary_content, 
                     summary_length, original_length, summary_text_length, compression_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (summary_id, user_id, text_id, original_text, summary_text,
                      summary_length, original_length, summary_text_length, compression_ratio))
                
                conn.commit()
                conn.close()
                
                return summary_id
        except Exception as e:
            raise Exception(f"Failed to save summary: {str(e)}")
    
    def get_summaries_by_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get summaries for a user - IMPROVED: Return full content
        
        Args:
            user_id: User ID
            limit: Max number of records
            
        Returns:
            List of summary records
        """
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        summary_id,
                        original_content,
                        summary_content,
                        summary_length,
                        original_length,
                        summary_text_length,
                        compression_ratio,
                        created_at
                    FROM summaries
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
                
                rows = cursor.fetchall()
                conn.close()
                
                summaries = []
                for row in rows:
                    summaries.append({
                        "summary_id": row["summary_id"],
                        "original_text": row["original_content"],  # Full original text
                        "summary_text": row["summary_content"],    # Full summary
                        "summary_length": row["summary_length"],
                        "original_length": row["original_length"],
                        "summary_length_chars": row["summary_text_length"],
                        "compression_ratio": round(row["compression_ratio"], 2),
                        "created_at": row["created_at"]
                    })
                
                return summaries
        except Exception as e:
            raise Exception(f"Failed to get summaries: {str(e)}")
    
    def get_summary_by_id(self, summary_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific summary
        
        Args:
            summary_id: Summary ID
            user_id: User ID (for security)
            
        Returns:
            Summary data or None
        """
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        summary_id,
                        original_content,
                        summary_content,
                        summary_length,
                        original_length,
                        summary_text_length,
                        compression_ratio,
                        created_at
                    FROM summaries
                    WHERE summary_id = ? AND user_id = ?
                """, (summary_id, user_id))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return {
                        "summary_id": row["summary_id"],
                        "original_text": row["original_content"],
                        "summary_text": row["summary_content"],
                        "summary_length": row["summary_length"],
                        "original_length": row["original_length"],
                        "summary_length_chars": row["summary_text_length"],
                        "compression_ratio": round(row["compression_ratio"], 2),
                        "created_at": row["created_at"]
                    }
                
                return None
        except Exception as e:
            raise Exception(f"Failed to get summary: {str(e)}")
    
    def delete_summary(self, summary_id: str, user_id: str) -> bool:
        """
        Delete a summary
        
        Args:
            summary_id: Summary ID
            user_id: User ID (for security)
            
        Returns:
            True if deleted
        """
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Get text_id
                cursor.execute("SELECT text_id FROM summaries WHERE summary_id = ? AND user_id = ?", 
                              (summary_id, user_id))
                row = cursor.fetchone()
                
                if not row:
                    conn.close()
                    return False
                
                text_id = row["text_id"]
                
                # Delete summary
                cursor.execute("DELETE FROM summaries WHERE summary_id = ?", (summary_id,))
                
                # Delete text if no other summaries reference it
                cursor.execute("SELECT COUNT(*) as count FROM summaries WHERE text_id = ?", (text_id,))
                if cursor.fetchone()["count"] == 0:
                    cursor.execute("DELETE FROM texts WHERE text_id = ?", (text_id,))
                
                conn.commit()
                conn.close()
                
                return True
        except Exception as e:
            raise Exception(f"Failed to delete summary: {str(e)}")
    
    def search_summaries(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        """
        Search summaries by keyword - IMPROVED: Return full content
        
        Args:
            user_id: User ID
            query: Search query
            
        Returns:
            List of matching summaries
        """
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                search_query = f"%{query}%"
                cursor.execute("""
                    SELECT 
                        summary_id,
                        original_content,
                        summary_content,
                        summary_length,
                        original_length,
                        summary_text_length,
                        compression_ratio,
                        created_at
                    FROM summaries
                    WHERE user_id = ? 
                    AND (original_content LIKE ? OR summary_content LIKE ?)
                    ORDER BY created_at DESC
                    LIMIT 50
                """, (user_id, search_query, search_query))
                
                rows = cursor.fetchall()
                conn.close()
                
                summaries = []
                for row in rows:
                    summaries.append({
                        "summary_id": row["summary_id"],
                        "original_text": row["original_content"],  # Full original text
                        "summary_text": row["summary_content"],    # Full summary
                        "summary_length": row["summary_length"],
                        "original_length": row["original_length"],
                        "compression_ratio": round(row["compression_ratio"], 2),
                        "created_at": row["created_at"]
                    })
                
                return summaries
        except Exception as e:
            raise Exception(f"Failed to search summaries: {str(e)}")
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a user
        
        Args:
            user_id: User ID
            
        Returns:
            User statistics
        """
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Total summaries
                cursor.execute("SELECT COUNT(*) as count FROM summaries WHERE user_id = ?", (user_id,))
                total_summaries = cursor.fetchone()["count"]
                
                # Short vs long
                cursor.execute("""
                    SELECT summary_length, COUNT(*) as count 
                    FROM summaries 
                    WHERE user_id = ? 
                    GROUP BY summary_length
                """, (user_id,))
                
                length_counts = {}
                for row in cursor.fetchall():
                    length_counts[row["summary_length"]] = row["count"]
                
                # Average compression ratio
                cursor.execute("""
                    SELECT AVG(compression_ratio) as avg_ratio 
                    FROM summaries 
                    WHERE user_id = ?
                """, (user_id,))
                
                avg_ratio = cursor.fetchone()["avg_ratio"] or 0
                
                conn.close()
                
                return {
                    "total_summaries": total_summaries,
                    "short_summaries": length_counts.get("short", 0),
                    "long_summaries": length_counts.get("long", 0),
                    "average_compression_ratio": round(avg_ratio, 2)
                }
        except Exception as e:
            raise Exception(f"Failed to get user stats: {str(e)}")