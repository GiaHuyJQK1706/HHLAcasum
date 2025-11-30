"""
Data Controller - Manages data persistence with SQLite
"""
from typing import List, Dict, Any
from data.database import Database


class DataController:
    """Manages all data persistence operations using SQLite"""
    
    def __init__(self):
        self.db = Database()
        self.current_user_id = None
    
    def initialize_user(self, user_id: str = None) -> str:
        """
        Initialize or get user
        
        Args:
            user_id: User ID (if None, auto-generate)
            
        Returns:
            User ID
        """
        try:
            self.current_user_id = self.db.get_or_create_user(user_id)
            return self.current_user_id
        except Exception as e:
            raise Exception(f"Failed to initialize user: {str(e)}")
    
    def add_summary_record(self, original_text: str, summary: str, 
                          summary_length: str) -> Dict[str, Any]:
        """
        Add a new summary record to database (UC02, UC05)
        
        Args:
            original_text: Original input text
            summary: Generated summary text
            summary_length: Length type (short/long)
            
        Returns:
            Status dictionary
        """
        try:
            if not self.current_user_id:
                self.initialize_user()
            
            summary_id = self.db.save_summary(
                user_id=self.current_user_id,
                original_text=original_text,
                summary_text=summary,
                summary_length=summary_length
            )
            
            return {
                "success": True,
                "summary_id": summary_id,
                "message": "Summary record saved successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to save summary: {str(e)}"
            }
    
    def get_summary_history(self) -> List[Dict[str, Any]]:
        """
        Get summary history from database (UC05)
        
        Returns:
            List of summary records
        """
        try:
            if not self.current_user_id:
                return []
            
            return self.db.get_summaries_by_user(self.current_user_id, limit=50)
        except Exception as e:
            print(f"Error retrieving history: {str(e)}")
            return []
    
    def get_summary_by_id(self, summary_id: str) -> Dict[str, Any]:
        """
        Get a specific summary
        
        Args:
            summary_id: Summary ID
            
        Returns:
            Summary data
        """
        try:
            if not self.current_user_id:
                return None
            
            return self.db.get_summary_by_id(summary_id, self.current_user_id)
        except Exception as e:
            print(f"Error retrieving summary: {str(e)}")
            return None
    
    def delete_summary(self, summary_id: str) -> Dict[str, Any]:
        """
        Delete a summary
        
        Args:
            summary_id: Summary ID
            
        Returns:
            Status dictionary
        """
        try:
            if not self.current_user_id:
                return {"success": False, "message": "User not initialized"}
            
            if self.db.delete_summary(summary_id, self.current_user_id):
                return {
                    "success": True,
                    "message": "Summary deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "message": "Summary not found"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete summary: {str(e)}"
            }
    
    def search_summaries(self, query: str) -> List[Dict[str, Any]]:
        """
        Search summaries by keyword
        
        Args:
            query: Search query
            
        Returns:
            List of matching summaries
        """
        try:
            if not self.current_user_id:
                return []
            
            return self.db.search_summaries(self.current_user_id, query)
        except Exception as e:
            print(f"Error searching summaries: {str(e)}")
            return []
    
    def get_user_stats(self) -> Dict[str, Any]:
        """
        Get user statistics
        
        Returns:
            User statistics
        """
        try:
            if not self.current_user_id:
                return {}
            
            return self.db.get_user_stats(self.current_user_id)
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {}
    
    def clear_history(self) -> Dict[str, Any]:
        """
        Clear all history records
        
        Returns:
            Status dictionary
        """
        try:
            if not self.current_user_id:
                return {"success": False, "message": "User not initialized"}
            
            history = self.get_summary_history()
            deleted_count = 0
            
            for record in history:
                if self.db.delete_summary(record["summary_id"], self.current_user_id):
                    deleted_count += 1
            
            return {
                "success": True,
                "message": f"Cleared {deleted_count} summaries",
                "count": deleted_count
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to clear history: {str(e)}"
            }