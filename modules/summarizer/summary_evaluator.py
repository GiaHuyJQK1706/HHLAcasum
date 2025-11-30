"""
Summary evaluation and quality assessment module
"""
from typing import Dict


class SummaryEvaluator:
    """Evaluates quality of generated summaries"""
    
    def __init__(self):
        pass
    
    def calculate_compression_ratio(self, original_text: str, summary: str) -> float:
        """
        Calculate compression ratio
        
        Args:
            original_text: Original text
            summary: Generated summary
            
        Returns:
            Compression ratio as percentage
        """
        try:
            if not original_text:
                return 0.0
            
            ratio = (len(summary) / len(original_text)) * 100
            return round(ratio, 2)
        except Exception as e:
            raise Exception(f"Failed to calculate compression ratio: {str(e)}")
    
    def evaluate_summary(self, original_text: str, summary: str) -> Dict:
        """
        Evaluate the quality of generated summary
        
        Args:
            original_text: Original text
            summary: Generated summary
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            metrics = {
                "original_length": len(original_text),
                "summary_length": len(summary),
                "compression_ratio": self.calculate_compression_ratio(original_text, summary),
                "word_count_original": len(original_text.split()),
                "word_count_summary": len(summary.split()),
                "is_valid": len(summary) > 0 and len(summary) < len(original_text)
            }
            return metrics
        except Exception as e:
            raise Exception(f"Summary evaluation failed: {str(e)}")
    
    def validate_summary(self, summary: str, min_length: int = 10) -> Dict:
        """
        Validate if summary is acceptable
        
        Args:
            summary: Summary text to validate
            min_length: Minimum acceptable length
            
        Returns:
            Dictionary with validation result
        """
        try:
            result = {
                "valid": True,
                "issues": []
            }
            
            if not summary or not summary.strip():
                result["valid"] = False
                result["issues"].append("Summary is empty")
            
            if len(summary) < min_length:
                result["valid"] = False
                result["issues"].append(f"Summary too short (minimum {min_length} characters)")
            
            if not summary[0].isupper():
                result["issues"].append("Summary doesn't start with capital letter")
            
            return result
        except Exception as e:
            raise Exception(f"Summary validation failed: {str(e)}")