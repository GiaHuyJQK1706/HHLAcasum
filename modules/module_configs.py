"""
Configuration for modules
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModuleConfigs:
    """Configuration settings for preprocessing and summarization modules"""
    
    # Model configuration
    MODEL_NAME: str = "hhlai/hhlai_academic_textsum"
    MODEL_LOCAL_PATH: str = "./models/hhlai_academic_textsum"  # Local model path
    DEVICE: str = "cpu"  # "cpu" or "cuda"
    USE_LOCAL_MODEL: bool = True  # Load from local path if available
    
    # Preprocessing configuration
    MAX_TEXT_LENGTH: int = 1024
    MIN_TEXT_LENGTH: int = 30
    
    # Tokenizer configuration
    MAX_TOKENS: int = 512
    
    # Summarization configuration - FIXED: Different lengths for short vs long
    SUMMARY_MIN_LENGTH_SHORT: int = 30
    SUMMARY_MAX_LENGTH_SHORT: int = 80
    SUMMARY_MIN_LENGTH_LONG: int = 100
    SUMMARY_MAX_LENGTH_LONG: int = 255
    
    # Generation parameters
    NUM_BEAMS: int = 4
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    
    # Validation configuration
    ALLOWED_FILE_TYPES: list = None
    MAX_FILE_SIZE_MB: int = 50
    
    def __post_init__(self):
        if self.ALLOWED_FILE_TYPES is None:
            self.ALLOWED_FILE_TYPES = [".txt", ".pdf", ".docx"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model_name": self.MODEL_NAME,
            "device": self.DEVICE,
            "max_text_length": self.MAX_TEXT_LENGTH,
            "min_text_length": self.MIN_TEXT_LENGTH,
            "max_tokens": self.MAX_TOKENS,
            "summary_lengths": {
                "short": {
                    "min": self.SUMMARY_MIN_LENGTH_SHORT,
                    "max": self.SUMMARY_MAX_LENGTH_SHORT
                },
                "long": {
                    "min": self.SUMMARY_MIN_LENGTH_LONG,
                    "max": self.SUMMARY_MAX_LENGTH_LONG
                }
            },
            "generation_params": {
                "num_beams": self.NUM_BEAMS,
                "temperature": self.TEMPERATURE,
                "top_p": self.TOP_P
            }
        }