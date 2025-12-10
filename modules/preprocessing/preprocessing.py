"""
@ file modules/preprocessing/preprocessing.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
@ v0.98: Enhanced with section detection support
"""
import re
from typing import List
from modules.module_configs import ModuleConfigs
from modules.preprocessing.section_detector import SectionDetector


class Preprocessing:
    """Enhanced text preprocessing with structure awareness"""
    
    def __init__(self, config: ModuleConfigs = None):
        self.config = config or ModuleConfigs()
        self.section_detector = SectionDetector()
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and special characters"""
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove multiple punctuation
            text = re.sub(r'([.!?]){2,}', r'\1', text)
            
            # Normalize quotes and escape characters
            text = (text
                    .replace("\\", "\\\\")
                    .replace(""", '"').replace(""", '"')
                    .replace("'", "'").replace("'", "'")
                    .replace('"', '\\"')
                    .replace("'", "\\'"))

            return text.strip()
        except Exception as e:
            raise Exception(f"Text normalization failed: {str(e)}")
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        try:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences
        except Exception as e:
            raise Exception(f"Sentence splitting failed: {str(e)}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            tokens = re.findall(r'\b\w+\b|[.!?]', text.lower())
            return tokens
        except Exception as e:
            raise Exception(f"Tokenization failed: {str(e)}")
    
    def detect_sections(self, text: str):
        """Detect document sections (wrapper for SectionDetector)"""
        try:
            return self.section_detector.detect_sections(text)
        except Exception as e:
            raise Exception(f"Section detection failed: {str(e)}")
    
    def process(self, text: str) -> str:
        """Process text through all preprocessing steps"""
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
        