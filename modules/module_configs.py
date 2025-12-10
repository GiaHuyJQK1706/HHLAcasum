"""
@ file modules/module_configs.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
@ v0.98: Advanced structure-aware summarization
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModuleConfigs:
    """Configuration settings with advanced structure-aware features"""
    
    # Model configuration
    MODEL_NAME: str = "hhlai/hhlai_academic_textsum"
    MODEL_LOCAL_PATH: str = "./models/hhlai_academic_textsum"
    DEVICE: str = "cpu"
    USE_LOCAL_MODEL: bool = True
    
    # Preprocessing configuration
    MAX_TEXT_LENGTH: int = 16384
    MIN_TEXT_LENGTH: int = 30
    MAX_TOKENS: int = 512
    
    # ============================================================
    # SUMMARY LENGTHS
    # ============================================================
    
    SUMMARY_MIN_LENGTH_SHORT: int = 80    
    SUMMARY_MAX_LENGTH_SHORT: int = 200   
    SUMMARY_MIN_LENGTH_LONG: int = 280    
    SUMMARY_MAX_LENGTH_LONG: int = 600    
    
    # ============================================================
    # GENERATION PARAMETERS
    # ============================================================
    
    # Short summary
    SHORT_LENGTH_PENALTY: float = 0.9     
    SHORT_NUM_BEAMS: int = 4               
    SHORT_REPETITION_PENALTY: float = 1.3  
    SHORT_EARLY_STOPPING: bool = False     
    SHORT_NO_REPEAT_NGRAM: int = 3
    SHORT_TEMPERATURE: float = 0.7
    SHORT_TOP_P: float = 0.9
    SHORT_DO_SAMPLE: bool = False
    
    # Long summary
    LONG_LENGTH_PENALTY: float = 2.2       
    LONG_NUM_BEAMS: int = 4                
    LONG_REPETITION_PENALTY: float = 1.1
    LONG_EARLY_STOPPING: bool = False
    LONG_NO_REPEAT_NGRAM: int = 2
    LONG_TEMPERATURE: float = 0.8
    LONG_TOP_P: float = 0.95
    LONG_DO_SAMPLE: bool = True
    
    # ============================================================
    # ADVANCED: STRUCTURE-AWARE SUMMARIZATION
    # ============================================================
    
    # Enable/disable advanced features
    USE_STRUCTURE_AWARE: bool = True
    USE_HIERARCHICAL_SUMMARIZATION: bool = True
    USE_TWO_PASS_SUMMARIZATION: bool = False  # Slower but better quality
    
    # Section detection
    ENABLE_SECTION_DETECTION: bool = True
    MIN_SECTION_LENGTH: int = 100  # Minimum chars for a valid section
    MERGE_SMALL_SECTIONS: bool = True
    
    # Hierarchical summarization
    HIERARCHICAL_SECTION_MAX_LENGTH: int = 500  # Max length per section summary
    HIERARCHICAL_POLISH_PASS: bool = True  # Final polish pass
    
    # Two-pass settings
    TWO_PASS_EXTRACTIVE_RATIO: float = 0.3  # Extract 30% of sentences
    TWO_PASS_ABSTRACTIVE_MIN_LENGTH: int = 100
    
    # ============================================================
    # PROMPT ENGINEERING
    # ============================================================
    
    USE_PREFIX_CONDITIONING: bool = True
    USE_ADVANCED_PROMPTS: bool = True
    
    # Basic prompts (used when USE_ADVANCED_PROMPTS = False)
    T5_PREFIX_SHORT: str = "summarize: "
    T5_PREFIX_LONG: str = "summarize in detail: "
    
    # Advanced prompts (structure-aware)
    ADVANCED_PROMPTS: Dict[str, str] = None
    
    # ============================================================
    # QUALITY VALIDATION
    # ============================================================
    
    ENABLE_QUALITY_VALIDATION: bool = True
    QUALITY_THRESHOLD: float = 0.6  # Minimum quality score
    AUTO_FIX_ISSUES: bool = True  # Auto-fix common problems
    RETRY_ON_LOW_QUALITY: bool = True
    MAX_RETRIES: int = 2
    
    # ============================================================
    # CHUNKING
    # ============================================================
    
    USE_SMART_CHUNKING: bool = True
    USE_SECTION_BASED_CHUNKING: bool = True  # NEW: Chunk by sections
    
    CHUNK_SIZE: int = 3000          
    CHUNK_OVERLAP: int = 300        
    MAX_CHUNKS: int = 4             
    
    SHORT_INPUT_MAX_CHARS: int = 4096
    LONG_INPUT_MAX_CHARS: int = 16384 
    
    # ============================================================
    # POST-PROCESSING
    # ============================================================
    
    ENABLE_SPELL_CHECK: bool = True
    USE_ADVANCED_SPELL_CHECK: bool = False
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
        
        # Common fixes dictionary
        if self.COMMON_FIXES is None:
            self.COMMON_FIXES = {
                "teh": "the", "taht": "that", "wich": "which", "wth": "with",
                "adn": "and", "nad": "and", "hte": "the", "thet": "that",
                "occured": "occurred", "ocurred": "occurred",
                "recieve": "receive", "recieved": "received",
                "seperate": "separate", "seperated": "separated",
                "definately": "definitely", "defiantly": "definitely",
                "untill": "until", "sucessful": "successful", "succesful": "successful",
                "similiar": "similar", "simular": "similar",
                "algorythm": "algorithm", "algoritm": "algorithm",
                "seperation": "separation", "independant": "independent",
                "occassion": "occasion", "accomodate": "accommodate",
                "apparant": "apparent", "concious": "conscious",
                "existance": "existence", "persistant": "persistent",
                "complier": "compiler", "compilar": "compiler",
                "funcionality": "functionality", "functonality": "functionality",
                "performace": "performance", "peformance": "performance",
                "efficent": "efficient", "effecient": "efficient",
                "optmization": "optimization", "optimisation": "optimization"
            }
        
        # Advanced prompts
        if self.ADVANCED_PROMPTS is None:
            self.ADVANCED_PROMPTS = {
                'short_structured': """Create a concise, well-structured summary that:
                - Starts with the main topic
                - Covers key points in logical order
                - Ends with conclusions or implications
                Text: """,
                                
                'long_structured': """Generate a comprehensive summary maintaining the document structure:
                - Begin with an overview of the main topic and motivation
                - Present key concepts and components in a clear, organized manner
                - Include important methods, findings, or features
                - Conclude with significance, implications, or future directions
                Ensure coherent flow between sections.
                Text: """,
                                
                'academic': """Summarize this academic text following this structure:
                1. Overview and motivation (2-3 sentences)
                2. Key concepts and methodology (3-4 sentences)
                3. Main findings or components (3-4 sentences)
                4. Conclusions or future directions (1-2 sentences)
                Text: """,
                
                'rewrite_coherent': """Rewrite this into a well-structured, coherent summary maintaining logical flow: """,
            }
    
    def get_summary_config(self, summary_length: str) -> Dict[str, Any]:
        """Get optimal configuration for summary type"""
        
        is_long = (summary_length == "long")
        
        config = {
            "min_length": self.SUMMARY_MIN_LENGTH_LONG if is_long else self.SUMMARY_MIN_LENGTH_SHORT,
            "max_length": self.SUMMARY_MAX_LENGTH_LONG if is_long else self.SUMMARY_MAX_LENGTH_SHORT,
            "length_penalty": self.LONG_LENGTH_PENALTY if is_long else self.SHORT_LENGTH_PENALTY,
            "num_beams": self.LONG_NUM_BEAMS if is_long else self.SHORT_NUM_BEAMS,
            "repetition_penalty": self.LONG_REPETITION_PENALTY if is_long else self.SHORT_REPETITION_PENALTY,
            "early_stopping": self.LONG_EARLY_STOPPING if is_long else self.SHORT_EARLY_STOPPING,
            "no_repeat_ngram_size": self.LONG_NO_REPEAT_NGRAM if is_long else self.SHORT_NO_REPEAT_NGRAM,
            "temperature": self.LONG_TEMPERATURE if is_long else self.SHORT_TEMPERATURE,
            "top_p": self.LONG_TOP_P if is_long else self.SHORT_TOP_P,
            "do_sample": self.LONG_DO_SAMPLE if is_long else self.SHORT_DO_SAMPLE,
            "input_max_chars": self.LONG_INPUT_MAX_CHARS if is_long else self.SHORT_INPUT_MAX_CHARS,
            "use_chunking": self.USE_SMART_CHUNKING and is_long,
            "summary_type": "long" if is_long else "short"
        }
        
        # Add prefix/prompt
        if self.USE_PREFIX_CONDITIONING:
            if self.USE_ADVANCED_PROMPTS:
                config["prefix"] = self.ADVANCED_PROMPTS['long_structured' if is_long else 'short_structured']
            else:
                config["prefix"] = self.T5_PREFIX_LONG if is_long else self.T5_PREFIX_SHORT
        else:
            config["prefix"] = ""
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model_name": self.MODEL_NAME,
            "model_type": "T5 Fine-tuned - Advanced Structure-Aware",
            "device": self.DEVICE,
            "features": {
                "structure_aware": self.USE_STRUCTURE_AWARE,
                "hierarchical": self.USE_HIERARCHICAL_SUMMARIZATION,
                "two_pass": self.USE_TWO_PASS_SUMMARIZATION,
                "quality_validation": self.ENABLE_QUALITY_VALIDATION,
                "section_detection": self.ENABLE_SECTION_DETECTION,
            },
            "summary_lengths": {
                "short": {
                    "min": self.SUMMARY_MIN_LENGTH_SHORT,
                    "max": self.SUMMARY_MAX_LENGTH_SHORT,
                    "beams": self.SHORT_NUM_BEAMS,
                    "expected_words": "60-150 words"
                },
                "long": {
                    "min": self.SUMMARY_MIN_LENGTH_LONG,
                    "max": self.SUMMARY_MAX_LENGTH_LONG,
                    "beams": self.LONG_NUM_BEAMS,
                    "expected_words": "210-450 words"
                }
            },
            "performance": "Advanced: Structure-aware + Quality validation + Auto-fixing"
        }
        