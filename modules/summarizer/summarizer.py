"""
@ file modules/summarizer/summarizer.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
@ v1.0: Fixed prompt leakage, generation parameters, and section headers
"""
import torch
import re
import time
from pathlib import Path
from typing import Optional, Dict, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from modules.module_configs import ModuleConfigs
from modules.preprocessing.section_detector import SectionDetector, Section
from modules.summarizer.summary_validator import SummaryValidator


class Summarizer:
    
    def __init__(self, config: ModuleConfigs = None):
        self.config = config or ModuleConfigs()
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
        self.section_detector = SectionDetector()
        self.validator = SummaryValidator()
        
        self._token_cache = {} if self.config.ENABLE_CACHING else None
        self._stats = {
            "total_time": 0,
            "tokenization_time": 0,
            "generation_time": 0,
            "postprocess_time": 0,
            "validation_time": 0
        }
    
    def _get_device(self) -> str:
        if self.config.DEVICE == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _get_model_path(self) -> str:
        if self.config.USE_LOCAL_MODEL:
            local_path = Path(self.config.MODEL_LOCAL_PATH)
            if local_path.exists() and (local_path / "config.json").exists():
                print(f"✅ Using local model from: {self.config.MODEL_LOCAL_PATH}")
                return str(local_path)
        
        print(f"⚠️ Local model not found, downloading from HuggingFace: {self.config.MODEL_NAME}")
        return self.config.MODEL_NAME
    
    def load_model(self) -> None:
        try:
            model_path = self._get_model_path()
            print(f"Loading T5 model from: {model_path}")
            print(f"Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                local_files_only=self.config.USE_LOCAL_MODEL
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def unload_model(self) -> None:
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
    
    # ================================================================
    # HAM TOM TAT CHINH
    # ================================================================
    
    def summarize(self, text: str, summary_length: str = "short") -> str:
        """
        Generate natural flowing summary
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise Exception("Model not loaded. Call load_model() first.")
            
            total_start = time.time()
            config = self.config.get_summary_config(summary_length)
            
            print(f"\n📝 Generating {summary_length} summary...")
            print(f"   Target: {config['min_length']}-{config['max_length']} tokens")
            print(f"   Strategy: Hybrid (Extractive + Abstractive)")
            
            # Chien luoc tom tat co cau truc
            if self.config.USE_STRUCTURE_AWARE and len(text) > 2000:
                sections = self.section_detector.detect_sections(text)
                structure = self.section_detector.get_document_structure(sections)
                
                # Check neu van ban co cau truc thuc su
                has_structure = (
                    len(sections) >= 2 and 
                    not (len(sections) == 1 and sections[0].title == "Document")
                )
                
                if has_structure:
                    # Show detected sections
                    section_titles = [s.title for s in sections[:5]]
                    titles_str = ", ".join(section_titles)
                    if len(sections) > 5:
                        titles_str += f", ... (+{len(sections)-5} more)"
                    print(f"   ✅ Structured document: {len(sections)} sections detected")
                    print(f"      Sections: {titles_str}")
                    
                    # Use hierarchical HYBRID strategy
                    print(f"   📋 Mode: Hierarchical Hybrid")
                    print(f"      → Extract key sentences per section")
                    print(f"      → Combine and generate abstractive summary")
                    summary = self._hierarchical_content_only(sections, config)
                else:
                    # No clear structure - use standard HYBRID
                    print(f"   📄 Document: No clear section structure")
                    print(f"   📋 Mode: Standard Hybrid")
                    print(f"      → Extract key sentences from full text")
                    print(f"      → Generate abstractive summary")
                    summary = self._standard_generation(text, config)
            else:
                print(f"   📋 Mode: Standard Hybrid (short text)")
                summary = self._standard_generation(text, config)
            
            # Hau xu ly ket qua de loai bo cac thanh phan khong can thiet
            summary = self._remove_all_section_markers(summary)
            summary = self._ensure_clean_output(summary)
            
            # Danh gia va sua loi neu can
            if self.config.ENABLE_QUALITY_VALIDATION:
                summary = self._validate_and_fix(summary, config)
            
            # Final polish
            summary = self._final_polish(summary)
            
            # Stats
            total_time = time.time() - total_start
            self._print_stats(summary, total_time, config)
            self._reset_stats()
            
            return summary
        
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")
    
    # ================================================================
    # STRATEGY 1: HIERARCHICAL 
    # ================================================================
    
    def _hierarchical_content_only(self, sections: List[Section], config: Dict) -> str:
        """
        Tiep can theo phan cap (Hybrid): Extract tu moi phan, sau do moi tao noi dung. 
        Phuong phap nay dam bao chung ta nam bat duoc cac chi tiet quan trong trong khi van duy tri duoc luong thong tin tu nhien.
        """
        print(f"   Phase 1/3: Extracting key information from each section...")
        
        # Extract content from each section
        section_contents = []
        for i, section in enumerate(sections, 1):
            if len(section.content.strip()) < 50:
                continue
            
            print(f"      [{i}/{len(sections)}] {section.title}: {len(section.content)} chars")
            
            # HYBRID: Extract key sentences from each section
            # Viec nay giup giam kich thuoc dau vao cho viec tom tat sau do ma van giu nguyen thong tin quan trong
            num_sentences = 5 if config['summary_type'] == 'long' else 3
            
            key_sentences = self.section_detector.extract_key_sentences(
                section.content,
                top_k=num_sentences
            )
            
            section_contents.append(' '.join(key_sentences))
        
        print(f"   Phase 2/3: Combining extracted content...")
        
        # Ket hop noi dung da extract
        combined_content = ' '.join(section_contents)
        original_len = sum(len(s.content) for s in sections if len(s.content.strip()) >= 50)
        
        print(f"      Reduced from {original_len} to {len(combined_content)} chars")
        
        # Generate abstractive summary from extracted content
        print(f"   Phase 3/3: Generating abstractive summary...")
        
        summary = self._generate_clean_summary(combined_content, config)
        
        return summary
    
    # ================================================================
    # STRATEGY 2: STANDARD GENERATION (HYBRID)
    # ================================================================
    
    def _standard_generation(self, text: str, config: Dict) -> str:
        """
        Tom tat tieu chuan voi phuong phap HYBRID (extractive + abstractive)
        Dam bao chat luong tom tat voi cac van ban bat ky
        """
        
        # Very long text - need chunking with extraction
        if len(text) > config['input_max_chars'] * 2:
            print(f"   Input: {len(text)} chars → Multi-chunk Hybrid")
            return self._multi_chunk_hybrid(text, config)
        
        # Long text - single extraction + generation
        elif len(text) > config['input_max_chars']:
            print(f"   Input: {len(text)} chars → Extractive-Abstractive Hybrid")
            
            # STEP 1: Extract key sentences (reduce input size)
            num_sentences = 15 if config['summary_type'] == 'long' else 10
            print(f"      Phase 1: Extracting {num_sentences} key sentences...")
            
            key_sentences = self.section_detector.extract_key_sentences(text, top_k=num_sentences)
            extracted_text = ' '.join(key_sentences)
            
            print(f"      Reduced from {len(text)} to {len(extracted_text)} chars")
            
            # STEP 2: Abstractive summarization on extracted content
            print(f"      Phase 2: Generating abstractive summary...")
            summary = self._generate_clean_summary(extracted_text, config)
            
            return summary
        
        # Medium text - light extraction for quality
        elif len(text) > 4096:
            print(f"   Input: {len(text)} chars → Light extraction + generation")
            
            # Extract top sentences to focus on key content
            num_sentences = 15 if config['summary_type'] == 'long' else 10
            key_sentences = self.section_detector.extract_key_sentences(text, top_k=num_sentences)
            extracted_text = ' '.join(key_sentences)
            
            print(f"      Focused on {len(key_sentences)} key sentences")
            return self._generate_clean_summary(extracted_text, config)
        
        # Short text - direct generation
        else:
            print(f"   Input: {len(text)} chars → Direct generation")
            return self._generate_clean_summary(text, config)
    
    # Ham nay chi xu ly cho truong hop van ban rat dai
    def _multi_chunk_hybrid(self, text: str, config: Dict) -> str:
        """
        Handle very long texts with chunking + hybrid approach
        Each chunk: extract key sentences -> summarize -> merge
        """
        print(f"      Strategy: Multi-chunk with extraction")
        
        # Split into chunks
        chunks = self._split_smart_chunks(text, config['input_max_chars'])
        print(f"      Split into {len(chunks)} chunks")
        
        # Process each chunk with extraction
        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"         Chunk {i}/{len(chunks)}: {len(chunk)} chars")
            
            # Extract key sentences from chunk
            num_sentences = 10 if config['summary_type'] == 'long' else 6
            key_sentences = self.section_detector.extract_key_sentences(chunk, top_k=num_sentences)
            extracted = ' '.join(key_sentences)
            
            # Generate summary from extracted content
            summary = self._generate_clean_summary(extracted, config)
            chunk_summaries.append(summary)
        
        # Merge all chunk summaries
        if len(chunk_summaries) > 1:
            print(f"      Merging {len(chunk_summaries)} chunk summaries...")
            
            # Combine all summaries
            merged = ' '.join(chunk_summaries)
            
            # Extract key sentences from merged summaries
            num_final = 15 if config['summary_type'] == 'long' else 10
            final_key = self.section_detector.extract_key_sentences(merged, top_k=num_final)
            final_extracted = ' '.join(final_key)
            
            # Final generation
            final_summary = self._generate_clean_summary(final_extracted, config)
            return final_summary
        else:
            return chunk_summaries[0]
    
    # ================================================================
    # CORE GENERATION (FIXED)
    # ================================================================
    
    def _generate_clean_summary(self, text: str, config: Dict) -> str:
        """
        Generate summary with FIXED prompt handling
        NO prompt leakage, NO generation warnings
        """
        
        # Prepare input with simple prefix
        prefix = config.get('prefix', '')
        input_text = prefix + text
        
        # Tokenize
        start_time = time.time()
        inputs = self._tokenize_with_cache(input_text, 512)
        self._stats["tokenization_time"] += time.time() - start_time
        
        # Generation parameters (FIXED)
        start_time = time.time()
        
        generation_params = {
            "input_ids": inputs["input_ids"],
            "min_length": config['min_length'],
            "max_length": config['max_length'],
            "num_beams": config['num_beams'],
            "length_penalty": config['length_penalty'],
            "repetition_penalty": config['repetition_penalty'],
            "no_repeat_ngram_size": config['no_repeat_ngram_size'],
            "early_stopping": config['early_stopping']
        }
        
        # CRITICAL FIX: Only add temperature/top_p if sampling is enabled
        if config.get('do_sample', False):
            generation_params['do_sample'] = True
            generation_params['temperature'] = config.get('temperature', 0.8)
            generation_params['top_p'] = config.get('top_p', 0.9)
        else:
            # Greedy decoding - no sampling parameters
            generation_params['do_sample'] = False
        
        # Generate
        with torch.no_grad():
            summary_ids = self.model.generate(**generation_params)
        
        self._stats["generation_time"] += time.time() - start_time
        
        # Decode
        summary = self.tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return summary
    
    # ================================================================
    # POST-PROCESSING (ENHANCED)
    # ================================================================
    
    def _remove_all_section_markers(self, text: str) -> str:
        """
        CRITICAL: Remove ALL section markers, numbers, and headers
        This prevents prompt leakage and section artifacts
        """
        
        # Remove numbered sections (1., 1.1., I., etc.)
        text = re.sub(r'\b\d+\.\d+\.\d+\b', '', text)  # 1.1.1
        text = re.sub(r'\b\d+\.\d+\b', '', text)        # 1.1
        text = re.sub(r'\b\d+\.', '', text)             # 1.
        text = re.sub(r'\b[IVX]+\.', '', text)          # Roman numerals
        
        # Remove markdown headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove bullet points and list markers
        text = re.sub(r'^[-•*]\s+', '', text, flags=re.MULTILINE)
        
        # Remove section titles (capitalized phrases followed by colon)
        text = re.sub(r'^[A-Z][A-Za-z\s]{2,40}:\s*', '', text, flags=re.MULTILINE)
        
        # Remove ALL CAPS headers
        text = re.sub(r'^[A-Z\s]{3,40}$', '', text, flags=re.MULTILINE)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', ' ', text)  # Multiple newlines
        text = re.sub(r'\s+', ' ', text)       # Multiple spaces
        
        return text.strip()
    
    def _ensure_clean_output(self, text: str) -> str:
        """
        Ensure output is clean prose without instructions or artifacts
        """
        
        # CRITICAL: Remove any prompt instructions that leaked through
        instruction_patterns = [
            r'create a.*?paragraph.*?text:',
            r'rewrite.*?text:',
            r'generate.*?summary.*?text:',
            r'summarize.*?text:',
            r'no section numbers',
            r'no bullet points',
            r'smooth narrative',
            r'complete sentences',
            r'preserve.*?information',
        ]
        
        for pattern in instruction_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove section-related phrases
        section_phrases = [
            'the following',
            'as follows',
            'section 1',
            'section 2',
            'first section',
            'next section',
            'in section',
        ]
        
        for phrase in section_phrases:
            text = re.sub(r'\b' + phrase + r'\b', '', text, flags=re.IGNORECASE)
        
        # Clean up resulting whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    def _validate_and_fix(self, summary: str, config: Dict) -> str:
        """Validate and fix common issues"""
        
        start_time = time.time()
        
        expected_min = config['min_length'] // 1.5
        validation = self.validator.validate_quality(summary, int(expected_min))
        
        self._stats["validation_time"] = time.time() - start_time
        
        print(f"   Quality: score={validation['score']:.2f}")
        
        if validation['issues']:
            print(f"      Auto-fixing {len(validation['issues'])} issues...")
            summary = self.validator.fix_common_issues(summary)
            
            # Additional cleaning
            summary = self._remove_all_section_markers(summary)
            summary = self._ensure_clean_output(summary)
        
        return summary
    
    def _final_polish(self, summary: str) -> str:
        """Final polishing for perfect output"""
        
        # Ensure complete sentences
        summary = summary.strip()
        
        if summary and summary[-1] not in '.!?':
            # Find last valid sentence ending
            last_period = max(
                summary.rfind('.'),
                summary.rfind('!'),
                summary.rfind('?')
            )
            
            # Only truncate if ending is close (within last 35%)
            if last_period > len(summary) * 0.65:
                summary = summary[:last_period + 1].strip()
            else:
                # Add period if no good ending found
                summary += '.'
        
        # Final cleanup
        summary = re.sub(r'\s+', ' ', summary)
        summary = re.sub(r'\s+([.,!?;:])', r'\1', summary)
        
        # Ensure proper sentence spacing
        summary = re.sub(r'([.!?])([A-Z])', r'\1 \2', summary)
        
        # Spell check
        if self.config.ENABLE_SPELL_CHECK:
            summary = self._fast_spell_check(summary)
        
        return summary.strip()
    
    # ================================================================
    # UTILITIES
    # ================================================================
    
    def _split_smart_chunks(self, text: str, max_size: int) -> List[str]:
        """Smart chunking preserving paragraph boundaries"""
        
        if len(text) <= max_size:
            return [text]
        
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    # Single paragraph too long, split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sent in sentences:
                        if len(current_chunk) + len(sent) > max_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent
                        else:
                            current_chunk += " " + sent
            else:
                current_chunk += " " + para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _tokenize_with_cache(self, text: str, max_length: int) -> Dict:
        """Tokenize with caching for efficiency"""
        
        if not self._token_cache:
            return self.tokenizer(
                text, max_length=max_length,
                truncation=True, return_tensors="pt"
            ).to(self.device)
        
        cache_key = hash(text[:100])
        
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]
        
        inputs = self.tokenizer(
            text, max_length=max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        
        # Limit cache size
        if len(self._token_cache) > 50:
            self._token_cache.pop(next(iter(self._token_cache)))
        
        self._token_cache[cache_key] = inputs
        return inputs
    
    def _fast_spell_check(self, text: str) -> str:
        """Fast spell checking using common fixes"""
        
        if not self.config.COMMON_FIXES:
            return text
        
        words = text.split()
        fixed_words = []
        
        for word in words:
            # Preserve punctuation
            clean_word = word.strip('.,!?;:"\'-')
            punct = word[len(clean_word):] if len(word) > len(clean_word) else ''
            
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
    
    def _print_stats(self, summary: str, total_time: float, config: Dict):
        """Print statistics"""
        
        word_count = len(summary.split())
        char_count = len(summary)
        
        print(f"\n✅ Summary completed in {total_time:.2f}s:")
        print(f"   📊 Output: {char_count} chars | {word_count} words")
        print(f"   ✨ Quality: Hybrid approach (Extractive + Abstractive)")
        print(f"   ✓ Clean: No sections, No prompts, Complete sentences")
    
    def _reset_stats(self):
        """Reset statistics"""
        self._stats = {k: 0 for k in self._stats}
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self._stats.copy()
    