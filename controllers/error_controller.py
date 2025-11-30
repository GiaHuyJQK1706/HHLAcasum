"""
Error Controller - Centralized error handling
"""
from typing import Dict, Any
from datetime import datetime


class ErrorController:
    """Handles error logging and reporting"""
    
    def __init__(self):
        self.errors = []
        self.max_errors_stored = 100
    
    def log_error(self, error_message: str, error_type: str = "general") -> Dict[str, Any]:
        """
        Log an error
        
        Args:
            error_message: Error message
            error_type: Type of error (validation, file, processing, model, system)
            
        Returns:
            Error record
        """
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message
        }
        
        self.errors.append(error_record)
        
        # Keep only last N errors
        if len(self.errors) > self.max_errors_stored:
            self.errors.pop(0)
        
        return error_record
    
    def get_error_message(self, error: Exception) -> str:
        """
        Get user-friendly error message
        
        Args:
            error: Exception object
            
        Returns:
            User-friendly error message
        """
        error_str = str(error)
        
        # Provide user-friendly messages for common errors
        if "CUDA" in error_str or "cuda" in error_str:
            return "GPU not available. Running on CPU."
        elif "out of memory" in error_str.lower():
            return "Out of memory. Please try with shorter text or restart."
        elif "file not found" in error_str.lower():
            return "File not found. Please check the file path."
        elif "permission denied" in error_str.lower():
            return "Permission denied. Please check file permissions."
        elif "encoding" in error_str.lower():
            return "File encoding error. Please ensure file is UTF-8 encoded."
        elif "model" in error_str.lower():
            return "Model error. Please check your model configuration."
        else:
            return f"An error occurred: {error_str}"
    
    def get_errors(self, limit: int = 10) -> list:
        """
        Get recent errors
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of error records
        """
        return self.errors[-limit:]
    
    def clear_errors(self) -> None:
        """Clear all stored errors"""
        self.errors = []
    
    def get_error_report(self) -> Dict[str, Any]:
        """
        Get error report
        
        Returns:
            Error report dictionary
        """
        return {
            "total_errors": len(self.errors),
            "recent_errors": self.get_errors(5),
            "error_types": self._categorize_errors()
        }
    
    def _categorize_errors(self) -> Dict[str, int]:
        """Categorize errors by type"""
        categories = {}
        for error in self.errors:
            error_type = error.get("type", "unknown")
            categories[error_type] = categories.get(error_type, 0) + 1
        return categories