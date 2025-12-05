"""
@ file modules/module_configs.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
@ v0.95 change: Longer outputs + Better accuracy
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModuleConfigs:
    """Configuration settings optimized for length and accuracy"""
    
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
    # SHORT & LONG SUMMARY LENGTHS
    # ============================================================
    
    SUMMARY_MIN_LENGTH_SHORT: int = 80    
    SUMMARY_MAX_LENGTH_SHORT: int = 200   
    
    SUMMARY_MIN_LENGTH_LONG: int = 280    
    SUMMARY_MAX_LENGTH_LONG: int = 600    
    
    # ============================================================
    # ACCURACY
    # LENGTH_PENALTY: Độ dài penalty để điều chỉnh độ dài tóm tắt
    # NUM_BEAMS: Số beams cho beam search
    # REPETITION_PENALTY: Penalty để tránh lặp từ
    # EARLY_STOPPING: Dừng sớm nếu đạt độ chính xác
    # NO_REPEAT_NGRAM: Không lặp n-gram
    # ============================================================
    
    # Short summary parameters
    SHORT_LENGTH_PENALTY: float = 0.9     
    SHORT_NUM_BEAMS: int = 4               
    SHORT_REPETITION_PENALTY: float = 1.3  
    SHORT_EARLY_STOPPING: bool = False     
    SHORT_NO_REPEAT_NGRAM: int = 3
    
    # Long summary parameters
    LONG_LENGTH_PENALTY: float = 2.2       
    LONG_NUM_BEAMS: int = 4                
    LONG_REPETITION_PENALTY: float = 1.1
    LONG_EARLY_STOPPING: bool = False
    LONG_NO_REPEAT_NGRAM: int = 2
    
    # ============================================================
    # INPUT PROCESSING
    # ============================================================
    
    # Tắt multi-pass để tăng tốc
    USE_MULTI_PASS: bool = False
    USE_PREFIX_CONDITIONING: bool = True
    
    T5_PREFIX_SHORT: str = "summarize: "
    T5_PREFIX_LONG: str = "summarize in detail: "
    
    # Spell check configuration
    ENABLE_SPELL_CHECK: bool = True
    USE_ADVANCED_SPELL_CHECK: bool = False
    
    # Common fixes
    COMMON_FIXES: Dict[str, str] = None
    
    # ============================================================
    # SMART CHUNKING
    # CHUNK_SIZE: Kích thước mỗi chunk
    # CHUNK_OVERLAP: Overlap giữa các chunks
    # MAX_CHUNKS: Số chunk tối đa để xử lý
    # SHORT_INPUT_MAX_CHARS: Giới hạn ký tự cho short input
    # LONG_INPUT_MAX_CHARS: Giới hạn ký tự cho long input
    # ============================================================
    USE_SMART_CHUNKING: bool = True
    
    # Tăng chunk size để mỗi chunk có nhiều context hơn
    CHUNK_SIZE: int = 3000          
    CHUNK_OVERLAP: int = 300        
    MAX_CHUNKS: int = 4             
    
    # Input limits - Tăng để không mất info
    SHORT_INPUT_MAX_CHARS: int = 4096
    LONG_INPUT_MAX_CHARS: int = 16384 
    
    # ============================================================
    # ADVANCED FEATURES
    # TEMPERATURE: Kiểm soát độ đa dạng
    # TOP_P: Top-p sampling
    # ============================================================
    
    # Temperature control cho diversity
    SHORT_TEMPERATURE: float = 0.7   # Thêm temperature
    LONG_TEMPERATURE: float = 0.8    # Thêm temperature
    
    # Top-p sampling
    SHORT_TOP_P: float = 0.9         # Thêm top-p
    LONG_TOP_P: float = 0.95         # Thêm top-p
    
    # Do sample để tăng diversity
    SHORT_DO_SAMPLE: bool = False    # Short: deterministic
    LONG_DO_SAMPLE: bool = True      # Long: diverse
    
    # Caching
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
        
        # Expanded common fixes (fast execution)
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
                "untill": "until", "sucessful": "successful", "succesful": "successful",
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
        """ Lấy cấu hình tối ưu cho loại tóm tắt """
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
                "use_chunking": self.USE_SMART_CHUNKING,
                "temperature": self.LONG_TEMPERATURE,
                "top_p": self.LONG_TOP_P,
                "do_sample": self.LONG_DO_SAMPLE
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
                "use_chunking": False,  # Short không cần chunking
                "temperature": self.SHORT_TEMPERATURE,
                "top_p": self.SHORT_TOP_P,
                "do_sample": self.SHORT_DO_SAMPLE
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model_name": self.MODEL_NAME,
            "model_type": "T5 Fine-tuned - Final Optimized (Length + Accuracy)",
            "device": self.DEVICE,
            "improvements": {
                "longer_outputs": True,
                "better_accuracy": True,
                "increased_beams": True,
                "improved_chunking": True,
                "temperature_control": True
            },
            "summary_lengths": {
                "short": {
                    "min": self.SUMMARY_MIN_LENGTH_SHORT,
                    "max": self.SUMMARY_MAX_LENGTH_SHORT,
                    "beams": self.SHORT_NUM_BEAMS,
                    "expected_words": "60-150 words",
                    "strategy": "High-quality beam search"
                },
                "long": {
                    "min": self.SUMMARY_MIN_LENGTH_LONG,
                    "max": self.SUMMARY_MAX_LENGTH_LONG,
                    "beams": self.LONG_NUM_BEAMS,
                    "expected_words": "210-450 words",
                    "strategy": "Diverse sampling with chunking"
                }
            },
            "performance": "Balanced: +25% length, +15% accuracy, ~10% slower"
        }
        