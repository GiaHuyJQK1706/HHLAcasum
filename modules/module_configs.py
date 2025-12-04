"""
Configuration for modules - PERFORMANCE OPTIMIZED VERSION
Tối ưu tốc độ trong khi giữ chất lượng cao
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModuleConfigs:
    """Configuration settings optimized for speed and quality"""
    
    # Model configuration
    MODEL_NAME: str = "hhlai/hhlai_academic_textsum"
    MODEL_LOCAL_PATH: str = "./models/hhlai_academic_textsum"
    DEVICE: str = "cpu" # "cuda" nếu máy có GPU
    USE_LOCAL_MODEL: bool = True
    
    # Preprocessing configuration
    MAX_TEXT_LENGTH: int = 16384
    MIN_TEXT_LENGTH: int = 30
    
    # Tokenizer configuration
    MAX_TOKENS: int = 512
    
    # ============================================================
    # PERFORMANCE: Cân bằng độ dài và tốc độ
    # ============================================================
    
    # Short summary: Ngắn gọn nhưng đầy đủ (60-150 tokens ~ 45-115 words)
    SUMMARY_MIN_LENGTH_SHORT: int = 60
    SUMMARY_MAX_LENGTH_SHORT: int = 180
    
    # Long summary: Chi tiết (220-500 tokens ~ 165-375 words)
    SUMMARY_MIN_LENGTH_LONG: int = 220
    SUMMARY_MAX_LENGTH_LONG: int = 500
    
    # SPEED OPTIMIZATION: Tắt multi-pass, tối ưu beams
    USE_MULTI_PASS: bool = False  # Tắt để tăng tốc 2x
    USE_PREFIX_CONDITIONING: bool = True
    
    # Prefixes tối ưu cho T5
    T5_PREFIX_SHORT: str = "summarize: "  # Đơn giản hơn
    T5_PREFIX_LONG: str = "summarize in detail: "
    
    # Spell check configuration
    ENABLE_SPELL_CHECK: bool = True
    USE_ADVANCED_SPELL_CHECK: bool = False  # Tắt library nặng
    
    # Common fixes - mở rộng hơn
    COMMON_FIXES: Dict[str, str] = None
    
    # Short summary parameters - OPTIMIZED
    SHORT_LENGTH_PENALTY: float = 0.7
    SHORT_NUM_BEAMS: int = 3  # Giảm beams để tăng tốc độ, tăng beam giúp giữ chất lượng
    SHORT_REPETITION_PENALTY: float = 1.2
    SHORT_EARLY_STOPPING: bool = True
    SHORT_NO_REPEAT_NGRAM: int = 3
    
    # Long summary parameters - OPTIMIZED  
    LONG_LENGTH_PENALTY: float = 2.0
    LONG_NUM_BEAMS: int = 3  # Giảm beam để tăng tốc độ, chunking giúp giữ chất lượng
    LONG_REPETITION_PENALTY: float = 1.1
    LONG_EARLY_STOPPING: bool = False
    LONG_NO_REPEAT_NGRAM: int = 2
    
    # ============================================================
    # SMART INPUT PROCESSING: Chunking để giảm mất thông tin
    # ============================================================
    USE_SMART_CHUNKING: bool = True
    
    # Thay vì truncate, chia input thành chunks và tóm tắt từng chunk
    CHUNK_SIZE: int = 2500  # chars per chunk
    CHUNK_OVERLAP: int = 200  # overlap để không mất context
    MAX_CHUNKS: int = 3  # Tối đa 3 chunks (không quá lâu)
    
    # Input limits ()
    SHORT_INPUT_MAX_CHARS: int = 30
    LONG_INPUT_MAX_CHARS: int = 15000  # Tăng để giữ info, dùng chunking
    
    # ============================================================
    # CACHING: Cache tokenization để tăng tốc
    # ============================================================
    ENABLE_CACHING: bool = True
    
    # Sentence completion
    ENSURE_COMPLETE_SENTENCES: bool = True
    ALLOWED_END_PUNCTUATION: tuple = (".", "!", "?")
    
    # Validation configuration
    ALLOWED_FILE_TYPES: list = None
    MAX_FILE_SIZE_MB: int = 50
    
    def __post_init__(self):
        if self.ALLOWED_FILE_TYPES is None:
            self.ALLOWED_FILE_TYPES = [".txt", ".pdf", ".docx"]
        
        # Expanded common fixes
        if self.COMMON_FIXES is None:
            self.COMMON_FIXES = {
                # Basic typos
                "teh": "the", "taht": "that", "wich": "which", "wth": "with",
                "adn": "and", "nad": "and", "hte": "the", "thet": "that",
                
                # Common misspellings
                "occured": "occurred", "ocurred": "occurred",
                "recieve": "receive", "recieved": "received",
                "seperate": "separate", "seperated": "separated",
                "definately": "definitely", "defiantly": "definitely",
                "untill": "until", "untill": "until",
                "sucessful": "successful", "succesful": "successful",
                "similiar": "similar", "simular": "similar",
                
                # Academic terms
                "algorythm": "algorithm", "algoritm": "algorithm",
                "seperation": "separation", "independant": "independent",
                "occassion": "occasion", "accomodate": "accommodate",
                "apparant": "apparent", "concious": "conscious",
                "existance": "existence", "persistant": "persistent",
                
                # Technical terms
                "complier": "compiler", "compilar": "compiler",
                "funcionality": "functionality", "functonality": "functionality",
                "performace": "performance", "peformance": "performance",
                "efficent": "efficient", "effecient": "efficient",
                "optmization": "optimization", "optimisation": "optimization"
            }
    
    def get_summary_config(self, summary_length: str) -> Dict[str, Any]:
        if summary_length == "long":
            return {
                "min_length": self.SUMMARY_MIN_LENGTH_LONG,
                "max_length": self.SUMMARY_MAX_LENGTH_LONG,
                "length_penalty": self.LONG_LENGTH_PENALTY,
                "num_beams": self.LONG_NUM_BEAMS,
                "repetition_penalty": self.LONG_REPETITION_PENALTY,
                "early_stopping": self.LONG_EARLY_STOPPING,
                "no_repeat_ngram_size": self.LONG_NO_REPEAT_NGRAM,
                "input_max_chars": self.LONG_INPUT_MAX_CHARS,
                "prefix": self.T5_PREFIX_LONG if self.USE_PREFIX_CONDITIONING else "",
                "use_chunking": self.USE_SMART_CHUNKING
            }
        else:  # short
            return {
                "min_length": self.SUMMARY_MIN_LENGTH_SHORT,
                "max_length": self.SUMMARY_MAX_LENGTH_SHORT,
                "length_penalty": self.SHORT_LENGTH_PENALTY,
                "num_beams": self.SHORT_NUM_BEAMS,
                "repetition_penalty": self.SHORT_REPETITION_PENALTY,
                "early_stopping": self.SHORT_EARLY_STOPPING,
                "no_repeat_ngram_size": self.SHORT_NO_REPEAT_NGRAM,
                "input_max_chars": self.SHORT_INPUT_MAX_CHARS,
                "prefix": self.T5_PREFIX_SHORT if self.USE_PREFIX_CONDITIONING else "",
                "use_chunking": False  # Short không cần chunking
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model_name": self.MODEL_NAME,
            "model_type": "T5 Fine-tuned - Performance Optimized",
            "device": self.DEVICE,
            "optimizations": {
                "multi_pass": self.USE_MULTI_PASS,
                "smart_chunking": self.USE_SMART_CHUNKING,
                "reduced_beams": True,
                "fast_spell_check": self.ENABLE_SPELL_CHECK and not self.USE_ADVANCED_SPELL_CHECK,
                "caching": self.ENABLE_CACHING
            },
            "summary_lengths": {
                "short": {
                    "min": self.SUMMARY_MIN_LENGTH_SHORT,
                    "max": self.SUMMARY_MAX_LENGTH_SHORT,
                    "beams": self.SHORT_NUM_BEAMS,
                    "strategy": "Fast single-pass"
                },
                "long": {
                    "min": self.SUMMARY_MIN_LENGTH_LONG,
                    "max": self.SUMMARY_MAX_LENGTH_LONG,
                    "beams": self.LONG_NUM_BEAMS,
                    "strategy": "Chunked processing"
                }
            },
            "expected_speedup": "2-3x faster than multi-pass version"
        }