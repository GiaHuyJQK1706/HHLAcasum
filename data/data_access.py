"""
Data Access Layer for storing and retrieving summarization history
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading


class DataAccess:
    """Manages data persistence for summarization history"""
    
    def __init__(self, storage_dir: str = "./data/storage"):
        self.storage_dir = Path(storage_dir)
        self.history_file = self.storage_dir / "history.json"
        self.max_history = 10
        self._lock = threading.Lock()
        self._ensure_storage_exists()
    
    def _ensure_storage_exists(self) -> None:
        """Ensure storage directory and files exist"""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            if not self.history_file.exists():
                self._save_history([])
        except Exception as e:
            raise Exception(f"Failed to initialize storage: {str(e)}")
    
    def save_text_to_storage(self, filepath: str) -> str:
        """
        Save extracted text to storage
        
        Args:
            filepath: Path to text file
            
        Returns:
            Saved file path
        """
        try:
            with self._lock:
                text_storage = self.storage_dir / "texts"
                text_storage.mkdir(parents=True, exist_ok=True)
                
                filename = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                save_path = text_storage / filename
                
                with open(filepath, 'r', encoding='utf-8') as src:
                    content = src.read()
                
                with open(save_path, 'w', encoding='utf-8') as dst:
                    dst.write(content)
                
                return str(save_path)
        except Exception as e:
            raise Exception(f"Failed to save text to storage: {str(e)}")
    
    def add_summary_record(self, original_text: str, summary: str, 
                          summary_length: str) -> None:
        """
        Add a new summary record to history
        
        Args:
            original_text: Original input text
            summary: Generated summary text
            summary_length: Length type (short/long)
        """
        try:
            with self._lock:
                history = self._load_history()
                
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "original_text": original_text[:500],  # Store first 500 chars
                    "summary": summary,
                    "summary_length": summary_length,
                    "original_length": len(original_text),
                    "summary_ratio": round(len(summary) / len(original_text) * 100, 2) if original_text else 0
                }
                
                history.insert(0, record)
                history = history[:self.max_history]
                
                self._save_history(history)
        except Exception as e:
            raise Exception(f"Failed to add summary record: {str(e)}")
    
    def get_summary_history(self) -> List[Dict[str, Any]]:
        """
        Get summary history (max 10 records)
        
        Returns:
            List of summary records
        """
        try:
            with self._lock:
                history = self._load_history()
                return history[:self.max_history]
        except Exception as e:
            raise Exception(f"Failed to retrieve history: {str(e)}")
    
    def clear_history(self) -> None:
        """Clear all history records"""
        try:
            with self._lock:
                self._save_history([])
        except Exception as e:
            raise Exception(f"Failed to clear history: {str(e)}")
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history from file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            raise Exception(f"Failed to load history: {str(e)}")
    
    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        """Save history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise Exception(f"Failed to save history: {str(e)}")