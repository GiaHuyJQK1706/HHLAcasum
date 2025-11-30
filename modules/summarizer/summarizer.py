"""
Text summarization module using local Transformer model - FIXED VERSION
"""
import torch
import os
from pathlib import Path
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from modules.module_configs import ModuleConfigs


class Summarizer:
    """Handles text summarization using transformer models"""
    
    def __init__(self, config: ModuleConfigs = None):
        self.config = config or ModuleConfigs()
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """Get available device (cuda or cpu)"""
        if self.config.DEVICE == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _get_model_path(self) -> str:
        """Get the model path (local or from Hugging Face)"""
        from pathlib import Path
        
        # Check if local model exists
        if self.config.USE_LOCAL_MODEL:
            local_path = Path(self.config.MODEL_LOCAL_PATH)
            if local_path.exists() and (local_path / "config.json").exists():
                print(f"✅ Using local model from: {self.config.MODEL_LOCAL_PATH}")
                return str(local_path)
        
        # Fall back to HuggingFace model
        print(f"⚠️ Local model not found, downloading from HuggingFace: {self.config.MODEL_NAME}")
        return self.config.MODEL_NAME
    
    def load_model(self) -> None:
        """Load the pretrained model and tokenizer"""
        try:
            model_path = self._get_model_path()
            print(f"Loading model from: {model_path}")
            print(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                local_files_only=self.config.USE_LOCAL_MODEL
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def unload_model(self) -> None:
        """Unload model from memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Failed to unload model: {str(e)}")
    
    def summarize(self, text: str, summary_length: str = "short") -> str:
        """
        Summarize the input text using direct model inference (FIXED)
        
        Args:
            text: Input text to summarize
            summary_length: Length of summary ("short" or "long")
            
        Returns:
            Generated summary
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise Exception("Model not loaded. Call load_model() first.")
            
            # Get length parameters
            if summary_length == "long":
                min_len = self.config.SUMMARY_MIN_LENGTH_LONG
                max_len = self.config.SUMMARY_MAX_LENGTH_LONG
            else:  # short
                min_len = self.config.SUMMARY_MIN_LENGTH_SHORT
                max_len = self.config.SUMMARY_MAX_LENGTH_SHORT
            
            print(f"\n📝 Generating {summary_length} summary...")
            print(f"   Min length: {min_len} | Max length: {max_len}")
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary directly from model (NOT pipeline to avoid max_new_tokens conflict)
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    num_beams=self.config.NUM_BEAMS,
                    min_length=min_len,
                    max_length=max_len,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode summary
            summary_text = self.tokenizer.batch_decode(
                summary_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            result = summary_text.strip()
            print(f"✅ Summary generated ({len(result)} chars, {len(result.split())} words)")
            
            return result
        
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None