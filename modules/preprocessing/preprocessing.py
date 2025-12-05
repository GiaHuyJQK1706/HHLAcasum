"""
@ file modules/preprocessing/preprocessing.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
"""
import re       # Regular expressions for text processing
from typing import List
from modules.module_configs import ModuleConfigs


class Preprocessing:
    """Handles text preprocessing operations"""
    
    def __init__(self, config: ModuleConfigs = None):
        self.config = config or ModuleConfigs()
    
    def normalize_text(self, text: str) -> str:
        """ Normalize text by removing extra whitespace and special characters """
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove multiple punctuation
            text = re.sub(r'([.!?]){2,}', r'\1', text)
            
            # Normalize quotes
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Text normalization failed: {str(e)}")
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Thuat toan: Su dung bieu thuc chinh quy de tach cau
        """
        try:
            # Split by common sentence endings
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences
        except Exception as e:
            raise Exception(f"Sentence splitting failed: {str(e)}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Thuat toan: Su dung bieu thuc chinh quy de tach tu
        """
        try:
            # Simple tokenization by splitting on whitespace and punctuation
            tokens = re.findall(r'\b\w+\b|[.!?]', text.lower())
            return tokens
        except Exception as e:
            raise Exception(f"Tokenization failed: {str(e)}")
    
    def process(self, text: str) -> str:
        """ Process text through all preprocessing steps        """
        try:
            # Step 1: Normalize text
            text = self.normalize_text(text)
            
            # Validate text length
            if len(text) < self.config.MIN_TEXT_LENGTH:
                raise ValueError(
                    f"Text is too short. Minimum length: {self.config.MIN_TEXT_LENGTH} characters"
                )
            
            if len(text) > self.config.MAX_TEXT_LENGTH:
                text = text[:self.config.MAX_TEXT_LENGTH]
            
            return text
        except Exception as e:
            raise Exception(f"Text processing failed: {str(e)}")
        