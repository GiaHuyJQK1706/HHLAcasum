"""
@ file modules/module_configs.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModuleConfigs:
    """Adaptive configuration for T5 base and fine-tuned models"""
    
    # ============================================================
    # MODEL CONFIGURATION
    # ============================================================
    
    MODEL_NAME: str = "hhlai/hhlai_academic_textsum"
    MODEL_LOCAL_PATH: str = "./models/hhlai_academic_textsum"
    DEVICE: str = "cpu"
    USE_LOCAL_MODEL: bool = True
    
    # CRITICAL: Set this based on your model type
    IS_FINETUNED_MODEL: bool = True  # True if using fine-tuned model
    
    # Preprocessing configuration
    MAX_TEXT_LENGTH: int = 16384
    MIN_TEXT_LENGTH: int = 30
    MAX_TOKENS: int = 512
    
    # ============================================================
    # SUMMARY LENGTHS
    # ============================================================
    
    SUMMARY_MIN_LENGTH_SHORT: int = 60    
    SUMMARY_MAX_LENGTH_SHORT: int = 150   
    SUMMARY_MIN_LENGTH_LONG: int = 300    
    SUMMARY_MAX_LENGTH_LONG: int = 600    
    
    # ============================================================
    # GENERATION PARAMETERS
    # Adaptive based on model type and summary length
    # ============================================================
    
    # Short summary - more conservative
    SHORT_LENGTH_PENALTY: float = 0.9     
    SHORT_NUM_BEAMS: int = 4               
    SHORT_REPETITION_PENALTY: float = 1.3  
    SHORT_EARLY_STOPPING: bool = True     
    SHORT_NO_REPEAT_NGRAM: int = 3
    SHORT_DO_SAMPLE: bool = False  # Greedy for consistency
    
    # Long summary - more flexible
    LONG_LENGTH_PENALTY: float = 1.5       
    LONG_NUM_BEAMS: int = 4                
    LONG_REPETITION_PENALTY: float = 1.1
    LONG_EARLY_STOPPING: bool = True
    LONG_NO_REPEAT_NGRAM: int = 2
    LONG_TEMPERATURE: float = 0.8
    LONG_TOP_P: float = 0.95
    LONG_DO_SAMPLE: bool = True  # Sampling for diversity
    
    # ============================================================
    # ADVANCED FEATURES
    # Effectiveness depends on model type
    # ============================================================
    
    USE_STRUCTURE_AWARE: bool = True
    USE_HIERARCHICAL_SUMMARIZATION: bool = True  # More effective with fine-tuned
    USE_TWO_PASS_SUMMARIZATION: bool = False
    
    # Section detection
    ENABLE_SECTION_DETECTION: bool = True
    MIN_SECTION_LENGTH: int = 150
    MERGE_SMALL_SECTIONS: bool = True
    MIN_SECTION_WORDS: int = 30
    
    # Hierarchical settings
    HIERARCHICAL_SECTION_MAX_LENGTH: int = 500
    HIERARCHICAL_POLISH_PASS: bool = True
    
    # ============================================================
    # PROMPT ENGINEERING - ADAPTIVE
    # Different strategies for base vs fine-tuned models
    # ============================================================
    
    USE_PREFIX_CONDITIONING: bool = True
    
    # Simple prompts for FINE-TUNED models
    FINETUNED_PREFIX_SHORT: str = "summarize: "
    FINETUNED_PREFIX_LONG: str = "summarize: "
    
    # ============================================================
    # QUALITY CONTROL
    # ============================================================
    
    ENABLE_QUALITY_VALIDATION: bool = True
    QUALITY_THRESHOLD: float = 0.6
    AUTO_FIX_ISSUES: bool = True
    RETRY_ON_LOW_QUALITY: bool = True
    MAX_RETRIES: int = 2
    
    # ============================================================
    # CHUNKING
    # ============================================================
    
    USE_SMART_CHUNKING: bool = True
    USE_SECTION_BASED_CHUNKING: bool = True
    
    CHUNK_SIZE: int = 3000          
    CHUNK_OVERLAP: int = 300        
    MAX_CHUNKS: int = 4             
    
    SHORT_INPUT_MAX_CHARS: int = 4096
    LONG_INPUT_MAX_CHARS: int = 16384 
    
    # ============================================================
    # POST-PROCESSING
    # ============================================================
    
    ENABLE_SPELL_CHECK: bool = True
    COMMON_FIXES: Dict[str, str] = None
    
    ENABLE_CACHING: bool = True
    ENSURE_COMPLETE_SENTENCES: bool = True
    ALLOWED_END_PUNCTUATION: tuple = (".", "!", "?")
    
    # ============================================================
    # VALIDATION
    # ============================================================
    
    ALLOWED_FILE_TYPES: list = None
    MAX_FILE_SIZE_MB: int = 50
    
    def __post_init__(self):
        if self.ALLOWED_FILE_TYPES is None:
            self.ALLOWED_FILE_TYPES = [".txt", ".pdf", ".docx"]
        
        if self.COMMON_FIXES is None:
            self.COMMON_FIXES = {
                "teh": "the", "taht": "that", "wich": "which", "wth": "with",
                "adn": "and", "nad": "and", "hte": "the", "thet": "that",
                "occured": "occurred", "ocurred": "occurred",
                "recieve": "receive", "recieved": "received",
                "seperate": "separate", "seperated": "separated",
                "definately": "definitely", "defiantly": "definitely",
                "untill": "until", "sucessful": "successful",
            }
    
    def get_summary_config(self, summary_length: str) -> Dict[str, Any]:
        """
        Get adaptive configuration based on:
        1. Model type (base vs fine-tuned)
        2. Summary length (short vs long)
        """
        
        is_long = (summary_length == "long")
        
        # Base configuration
        config = {
            "min_length": self.SUMMARY_MIN_LENGTH_LONG if is_long else self.SUMMARY_MIN_LENGTH_SHORT,
            "max_length": self.SUMMARY_MAX_LENGTH_LONG if is_long else self.SUMMARY_MAX_LENGTH_SHORT,
            "length_penalty": self.LONG_LENGTH_PENALTY if is_long else self.SHORT_LENGTH_PENALTY,
            "num_beams": self.LONG_NUM_BEAMS if is_long else self.SHORT_NUM_BEAMS,
            "repetition_penalty": self.LONG_REPETITION_PENALTY if is_long else self.SHORT_REPETITION_PENALTY,
            "early_stopping": self.LONG_EARLY_STOPPING if is_long else self.SHORT_EARLY_STOPPING,
            "no_repeat_ngram_size": self.LONG_NO_REPEAT_NGRAM if is_long else self.SHORT_NO_REPEAT_NGRAM,
            "do_sample": self.LONG_DO_SAMPLE if is_long else self.SHORT_DO_SAMPLE,
            "input_max_chars": self.LONG_INPUT_MAX_CHARS if is_long else self.SHORT_INPUT_MAX_CHARS,
            "summary_type": "long" if is_long else "short",
            "is_finetuned": self.IS_FINETUNED_MODEL
        }
        
        # Adaptive: Only add sampling parameters if enabled
        if config["do_sample"]:
            config["temperature"] = self.LONG_TEMPERATURE if is_long else 0.8
            config["top_p"] = self.LONG_TOP_P if is_long else 0.9
        
        # Adaptive: Chunking strategy
        # Fine-tuned models can handle structure better
        config["use_chunking"] = self.USE_SMART_CHUNKING and is_long
        config["prefer_hierarchical"] = self.IS_FINETUNED_MODEL and self.USE_HIERARCHICAL_SUMMARIZATION
        
        # Adaptive: Prefix selection
        if self.USE_PREFIX_CONDITIONING:
            if self.IS_FINETUNED_MODEL:
                # Fine-tuned: Simple prefix is enough
                config["prefix"] = self.FINETUNED_PREFIX_LONG if is_long else self.FINETUNED_PREFIX_SHORT
            else:
                # Base model: Need detailed instructions
                config["prefix"] = self.FINETUNED_PREFIX_LONG if is_long else self.FINETUNED_PREFIX_SHORT
        else:
            config["prefix"] = ""

        return config
    
    def get_strategy_recommendation(self, text_length: int, num_sections: int) -> str:
        """
        Recommend best strategy based on model type and text characteristics
        """
        
        if self.IS_FINETUNED_MODEL:
            # Fine-tuned model: Can use advanced strategies
            if num_sections >= 3 and text_length > 3000:
                return "hierarchical"  # Best for structured docs
            elif text_length > 5000:
                return "extractive_abstractive"  # Good for long docs
            else:
                return "standard"  # Simple and fast
        else:
            # Base model: Use simpler strategies
            if text_length > 5000:
                return "extractive_abstractive"  # Safer for long docs
            else:
                return "standard"  # Most reliable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model_name": self.MODEL_NAME,
            "model_type": "T5 Fine-tuned" if self.IS_FINETUNED_MODEL else "T5 Base",
            "device": self.DEVICE,
            "features": {
                "structure_aware": self.USE_STRUCTURE_AWARE,
                "hierarchical": self.USE_HIERARCHICAL_SUMMARIZATION,
                "quality_validation": self.ENABLE_QUALITY_VALIDATION,
                "section_detection": self.ENABLE_SECTION_DETECTION,
            },
            "optimization_level": "High (Fine-tuned)" if self.IS_FINETUNED_MODEL else "Standard (Base)"
        }


# ============================================================
# VI DU SU DUNG
# ============================================================

"""
Change the configuration based on whether you're using a fine-tuned model or the base T5 model.
# For FINE-TUNED model:
config = ModuleConfigs(
    MODEL_NAME="hhlai/hhlai_academic_textsum",
    IS_FINETUNED_MODEL=True  # ← This tells the system
)

# For BASE T5 model:
config = ModuleConfigs(
    MODEL_NAME="t5-base",
    IS_FINETUNED_MODEL=False  # ← Different strategies will be used
)
"""