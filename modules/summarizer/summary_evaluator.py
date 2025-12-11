"""
@ file modules/summarizer/summary_evaluator.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
"""
from typing import Dict


class SummaryEvaluator:
    """Evaluates quality of generated summaries"""
    
    def __init__(self):
        pass
    
    def calculate_compression_ratio(self, original_text: str, summary: str) -> float:
        """
        Calculate compression ratio
        Cong thuc: (length of summary / length of original text) * 100
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
        
        Danh gia chat luong tom tat su dung cac chi so co ban
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
        Thuat toan: Kiem tra do dai toi thieu va cau chuan
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
        