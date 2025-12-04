import torch
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from modules.module_configs import ModuleConfigs


class Summarizer:
    """High-performance T5 summarizer with smart chunking"""
    
    def __init__(self, config: ModuleConfigs = None):
        self.config = config or ModuleConfigs()
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
        # Performance: Cache
        self._token_cache = {} if self.config.ENABLE_CACHING else None
        self._stats = {
            "total_time": 0,
            "tokenization_time": 0,
            "generation_time": 0,
            "postprocess_time": 0
        }
    
    def _get_device(self) -> str:
        """Get available device (cuda or cpu)"""
        if self.config.DEVICE == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _get_model_path(self) -> str:
        """Get the model path (local or from Hugging Face)"""
        if self.config.USE_LOCAL_MODEL:
            local_path = Path(self.config.MODEL_LOCAL_PATH)
            if local_path.exists() and (local_path / "config.json").exists():
                print(f"✅ Using local model from: {self.config.MODEL_LOCAL_PATH}")
                return str(local_path)
        
        print(f"⚠️ Local model not found, downloading from HuggingFace: {self.config.MODEL_NAME}")
        return self.config.MODEL_NAME
    
    def load_model(self) -> None:
        """Load the pretrained model and tokenizer"""
        try:
            model_path = self._get_model_path()
            print(f"Loading T5 model from: {model_path}")
            print(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                local_files_only=self.config.USE_LOCAL_MODEL
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            print(f"   Optimizations: Chunking={'✓' if self.config.USE_SMART_CHUNKING else '✗'}, "
                  f"Beams={self.config.SHORT_NUM_BEAMS}/{self.config.LONG_NUM_BEAMS}, "
                  f"Caching={'✓' if self.config.ENABLE_CACHING else '✗'}")
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
            if self._token_cache:
                self._token_cache.clear()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Failed to unload model: {str(e)}")
    
    def _split_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Nếu không phải chunk cuối, tìm sentence boundary
            if end < len(text):
                # Tìm dấu câu gần nhất trong 100 chars cuối của chunk
                search_start = max(end - 100, start)
                last_period = max(
                    text.rfind('.', search_start, end),
                    text.rfind('!', search_start, end),
                    text.rfind('?', search_start, end)
                )
                
                if last_period > start:
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Tính start mới với overlap
            start = end - overlap if end < len(text) else end
            
            # Giới hạn số chunks
            if len(chunks) >= self.config.MAX_CHUNKS:
                # Thêm phần còn lại vào chunk cuối
                if start < len(text):
                    remaining = text[start:].strip()
                    if remaining:
                        chunks[-1] = chunks[-1] + " " + remaining
                break
        
        return chunks
    
    def _merge_chunk_summaries(self, chunk_summaries: List[str]) -> str:
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        
        # Merge bằng cách nối lại với smooth transition
        merged = " ".join(chunk_summaries)
        
        # Clean up duplicates và redundancy
        sentences = re.split(r'(?<=[.!?])\s+', merged)
        unique_sentences = []
        seen = set()
        
        for sent in sentences:
            # Normalize để check duplicate
            normalized = ' '.join(sent.lower().split())
            if normalized not in seen and len(normalized) > 20:
                unique_sentences.append(sent)
                seen.add(normalized)
        
        return ' '.join(unique_sentences)
    
    def _fast_spell_check(self, text: str) -> str:
        if not self.config.ENABLE_SPELL_CHECK or not self.config.COMMON_FIXES:
            return text
        
        words = text.split()
        fixed_words = []
        
        for word in words:
            # Tách punctuation
            clean_word = word.strip('.,!?;:"\'-')
            punct = word[len(clean_word):] if len(word) > len(clean_word) else ''
            
            # Check và sửa
            lower_word = clean_word.lower()
            if lower_word in self.config.COMMON_FIXES:
                fixed = self.config.COMMON_FIXES[lower_word]
                # Preserve capitalization
                if clean_word and clean_word[0].isupper():
                    fixed = fixed.capitalize()
                if clean_word.isupper():
                    fixed = fixed.upper()
                fixed_words.append(fixed + punct)
            else:
                fixed_words.append(word)
        
        return ' '.join(fixed_words)
    
    # Ensure summary ends with complete sentence
    def _ensure_complete_sentence(self, text: str) -> str:
        if not text or not text.strip():
            return text
        
        text = text.strip()
        
        if text[-1] in self.config.ALLOWED_END_PUNCTUATION:
            return text
        
        # Tìm dấu câu cuối cùng
        last_period = max(
            text.rfind('.'),
            text.rfind('!'),
            text.rfind('?')
        )
        
        if last_period > len(text) * 0.7:  # Chỉ cắt nếu gần cuối
            return text[:last_period + 1].strip()
        
        return text + "."
    
    def _clean_summary_text(self, text: str) -> str:
        text = ' '.join(text.split())
        # Sửa dấu câu bị dính
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        # Xóa dấu câu lặp
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        # Sửa khoảng trắng trước dấu câu
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()
    
    def _tokenize_with_cache(self, text: str, max_length: int) -> Dict:
        if not self._token_cache:
            # No caching
            return self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
        
        # Check cache
        cache_key = hash(text[:100])  # Cache key từ 100 chars đầu
        
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]
        
        # Tokenize và cache
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Giới hạn cache size
        if len(self._token_cache) > 50:
            # Xóa entry cũ nhất
            self._token_cache.pop(next(iter(self._token_cache)))
        
        self._token_cache[cache_key] = inputs
        return inputs
    
    def _generate_summary_single(self, text: str, config: Dict) -> str:
        # Prefix
        if config.get('prefix'):
            input_text = config['prefix'] + text
        else:
            input_text = text
        
        # Tokenize với caching
        start_time = time.time()
        inputs = self._tokenize_with_cache(input_text, 512)
        self._stats["tokenization_time"] += time.time() - start_time
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                min_length=config['min_length'],
                max_length=config['max_length'] + 20,
                num_beams=config['num_beams'],
                length_penalty=config['length_penalty'],
                repetition_penalty=config['repetition_penalty'],
                no_repeat_ngram_size=config['no_repeat_ngram_size'],
                early_stopping=config['early_stopping'],
                do_sample=False
            )
        self._stats["generation_time"] += time.time() - start_time
        
        # Decode
        summary = self.tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return summary
    
    def _post_process_summary(self, summary: str) -> str:
        start_time = time.time()
        
        # 1. Clean
        summary = self._clean_summary_text(summary)
        
        # 2. Fast spell check
        summary = self._fast_spell_check(summary)
        
        # 3. Complete sentence
        if self.config.ENSURE_COMPLETE_SENTENCES:
            summary = self._ensure_complete_sentence(summary)
        
        self._stats["postprocess_time"] += time.time() - start_time
        
        return summary
    
    def summarize(self, text: str, summary_length: str = "short") -> str:
        try:
            if self.model is None or self.tokenizer is None:
                raise Exception("Model not loaded. Call load_model() first.")
            
            total_start = time.time()
            
            # Lấy cấu hình
            config = self.config.get_summary_config(summary_length)
            
            print(f"\n📝 Generating {summary_length} summary (Performance Mode)...")
            print(f"   Target: {config['min_length']}-{config['max_length']} tokens")
            print(f"   Beams: {config['num_beams']} (optimized)")
            
            # ============================================================
            # SMART CHUNKING: Xử lý văn bản dài
            # ============================================================
            if config.get('use_chunking') and len(text) > config['input_max_chars']:
                print(f"   Input: {len(text)} chars → Using smart chunking")
                
                # Chia thành chunks
                chunks = self._split_into_chunks(
                    text,
                    self.config.CHUNK_SIZE,
                    self.config.CHUNK_OVERLAP
                )
                
                print(f"   Chunks: {len(chunks)} ({[len(c) for c in chunks]} chars)")
                
                # Tóm tắt từng chunk
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    print(f"   Processing chunk {i+1}/{len(chunks)}...")
                    summary = self._generate_summary_single(chunk, config)
                    chunk_summaries.append(summary)
                
                # Merge summaries
                summary_text = self._merge_chunk_summaries(chunk_summaries)
                print(f"   ✓ Merged {len(chunk_summaries)} chunk summaries")
                
            else:
                # Simple truncate cho văn bản ngắn
                if len(text) > config['input_max_chars']:
                    text = text[:config['input_max_chars']]
                    print(f"   Input: {len(text)} chars (truncated)")
                else:
                    print(f"   Input: {len(text)} chars")
                
                # Sinh summary
                summary_text = self._generate_summary_single(text, config)
            
            # ============================================================
            # POST-PROCESSING: Fast cleaning
            # ============================================================
            result = self._post_process_summary(summary_text)
            
            # Stats
            total_time = time.time() - total_start
            self._stats["total_time"] = total_time
            
            word_count = len(result.split())
            char_count = len(result)
            
            print(f"✅ Summary completed in {total_time:.2f}s:")
            print(f"   {char_count} chars | {word_count} words")
            print(f"   Breakdown: Token={self._stats['tokenization_time']:.2f}s, "
                  f"Gen={self._stats['generation_time']:.2f}s, "
                  f"Post={self._stats['postprocess_time']:.2f}s")
            print(f"   Ends with: '{result[-1]}' ✓")
            
            # Reset stats
            self._stats = {k: 0 for k in self._stats}
            
            return result
        
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self._stats.copy()