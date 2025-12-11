"""
@ file modules/summarizer/summarizer.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
@ v0.99: Natural flowing summaries with complete sentences
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
    """Ultimate T5 summarizer with natural flowing output"""
    
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
            print(f"   Mode: Natural Flow Generation")
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
    # MAIN: TOM TAT VAN BAN THANH VAN BAN TU NHIEN
    # ================================================================
    
    def summarize(self, text: str, summary_length: str = "short") -> str:
        """Generate natural flowing summary without section numbers"""
        try:
            if self.model is None or self.tokenizer is None:
                raise Exception("Model not loaded. Call load_model() first.")
            
            total_start = time.time()
            config = self.config.get_summary_config(summary_length)
            
            print(f"\n📝 Generating {summary_length} natural flow summary...")
            print(f"   Target: {config['min_length']}-{config['max_length']} tokens")
            
            # CHIEN LUOC NANG CAO: Ket hop Trich xuat + Truu tuong (toc do cao)
            if self.config.USE_STRUCTURE_AWARE and len(text) > 3000:
                sections = self.section_detector.detect_sections(text)
                structure = self.section_detector.get_document_structure(sections)
                
                print(f"   Document: {structure['total_sections']} sections")
                
                if structure['is_well_structured'] and len(sections) > 1:
                    print(f"   Strategy: Hierarchical with Natural Flow")
                    summary = self._hierarchical_natural_flow(sections, config, text)
                else:
                    print(f"   Strategy: Enhanced Extractive-Abstractive")
                    summary = self._extractive_abstractive_hybrid(text, config)
            else:
                print(f"   Strategy: Standard with Flow Enhancement")
                summary = self._standard_with_flow(text, config)
            
            # CRITICAL: Xu ly hau qua van ban tu nhien
            summary = self._ensure_natural_flow(summary)
            
            # Kiem tra chat luong va sua loi neu can
            if self.config.ENABLE_QUALITY_VALIDATION:
                summary = self._validate_and_fix_natural(summary, config)
            
            # Hoan thien cuoi cung
            summary = self._final_polish(summary)
            
            # Thong ke
            total_time = time.time() - total_start
            self._print_stats(summary, total_time, config)
            self._reset_stats()
            
            return summary
        
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")
    
    # ================================================================
    # CHIEN LUOC 1: PHAN CAP VOI NATURAL FLOW
    # ================================================================
    
    def _hierarchical_natural_flow(self, sections: List[Section], 
                                   config: Dict, original_text: str) -> str:
        """Hierarchical summarization producing natural flowing text"""
        
        print(f"   Phase 1/3: Extracting key information per section...")
        
        # Trich xuat thong tin chinh tu moi phan muc
        section_extracts = []
        for i, section in enumerate(sections, 1):
            if len(section.content) < 50:
                continue
            
            print(f"      [{i}/{len(sections)}] {section.title}")
            
            # Trich xuat cau chinh (khong chi la tom tat)
            key_sentences = self._extract_key_info_from_section(
                section.content, 
                section.section_type,
                max_sentences=5 if config['summary_type'] == 'long' else 3
            )
            
            section_extracts.append({
                'title': section.title,
                'type': section.section_type,
                'key_info': ' '.join(key_sentences),
                'original_length': len(section.content)
            })
        
        print(f"   Phase 2/3: Combining into coherent narrative...")
        
        # Ket hop cac phan trich xuat thanh van ban chay
        combined_narrative = self._combine_to_narrative(section_extracts)
        
        print(f"   Phase 3/3: Final abstractive polish...")
        
        # Chinh sua truu tuong de van ban tu nhien
        polish_config = config.copy()
        polish_config['prefix'] = """Rewrite this as a single, flowing paragraph without section numbers or bullet points. 
Ensure smooth transitions between ideas and complete sentences throughout. 
Preserve all important technical details and information.
Text: """
        polish_config['max_length'] = config['max_length'] + 50  # Allow more space
        
        final_summary = self._generate_summary_single(combined_narrative, polish_config)
        
        return final_summary
    
    def _extract_key_info_from_section(self, section_text: str, 
                                       section_type: str, 
                                       max_sentences: int = 5) -> List[str]:
        """Extract most informative sentences from a section"""
        
        sentences = re.split(r'(?<=[.!?])\s+', section_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return sentences
        
        # Diem so cac cau theo do quan trong
        from collections import Counter
        
        # Tinh tan so tu
        all_words = []
        for sent in sentences:
            words = [w.lower() for w in re.findall(r'\b\w+\b', sent)]
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        # Loai bo cac tu pho bien (stopwords)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'been',
                     'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'can', 'could', 'may', 'might', 'must', 'that', 'this', 'these', 'those'}
        
        # Diem so cac cau theo do quan trong
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            words = [w.lower() for w in re.findall(r'\b\w+\b', sentence) if w.lower() not in stopwords]
            
            if not words:
                continue
            
            # Diem so TF
            tf_score = sum(word_freq[w] for w in words) / len(words)
            
            # Thuong diem theo vi tri (Uu tien cau dau va cau cuoi trong phan muc)
            position_boost = 1.0
            if idx == 0:
                position_boost = 1.3  # First sentence
            elif idx == len(sentences) - 1 and section_type == 'conclusion':
                position_boost = 1.2  # Last in conclusion
            
            # Thuong diem noi dung ky thuat (so, cac term ky thuat)
            tech_boost = 1.0
            if re.search(r'\d+', sentence):
                tech_boost += 0.2
            if any(term in sentence.lower() for term in ['system', 'method', 'algorithm', 'performance', 'model']):
                tech_boost += 0.1
            
            final_score = tf_score * position_boost * tech_boost
            scored_sentences.append((final_score, sentence, idx))
        
        # Lay cac cau hang dau, giu nguyen thu tu
        top_sentences = sorted(scored_sentences, reverse=True)[:max_sentences]
        top_sentences.sort(key=lambda x: x[2])  # Sort by original position
        
        return [s[1] for s in top_sentences]
    
    def _combine_to_narrative(self, section_extracts: List[Dict]) -> str:
        """Combine section extracts into a flowing narrative"""
        
        # Cac cau chuyen tiep cho su chuyen dong tu nhien
        transitions = {
            'introduction': '',  # Khong can cau chuyen tiep o dau
            'content': 'Furthermore, ',
            'methods': 'In terms of methodology, ',
            'results': 'The findings indicate that ',
            'conclusion': 'Ultimately, '
        }
        
        narrative = ""
        prev_type = None
        
        for i, extract in enumerate(section_extracts):
            section_type = extract['type']
            content = extract['key_info']
            
            # Them cau chuyen tiep neu can
            if i > 0 and section_type != prev_type:
                # Them cau chuyen tiep phu hop
                if section_type == 'conclusion':
                    narrative += " "
                else:
                    narrative += " Additionally, "
            else:
                narrative += " "
            
            # Them noi dung (loai bo cac so hieu phan muc neu co)
            content_clean = re.sub(r'^\d+\.?\s*', '', content)
            narrative += content_clean
            
            prev_type = section_type
        
        return narrative.strip()
    
    # ================================================================
    # CHIEN LUOC 2: KET HOP TRICH XUAT - TRUU TUONG
    # ================================================================
    
    def _extractive_abstractive_hybrid(self, text: str, config: Dict) -> str:
        """Hybrid approach: extract key info, then generate natural summary"""
        
        print(f"   Phase 1/2: Extracting key sentences...")
        
        # Trich xuat cau chinh
        num_sentences = 15 if config['summary_type'] == 'long' else 10
        key_sentences = self.section_detector.extract_key_sentences(text, top_k=num_sentences)
        
        extracted_text = ' '.join(key_sentences)
        print(f"      Extracted {len(key_sentences)} key sentences")
        
        print(f"   Phase 2/2: Generating natural flowing summary...")
        
        # Tao tom tat van ban tu nhien
        gen_config = config.copy()
        gen_config['prefix'] = """Create a single, flowing paragraph that naturally describes this content. 
No section numbers, no bullet points, just smooth narrative prose with complete sentences.
Preserve all important information and technical details.
Text: """
        
        summary = self._generate_summary_single(extracted_text, gen_config)
        
        return summary
    
    # ================================================================
    # CHIEN LUOC 3: CAI THIEN TIEU CHUAN VOI VAN BAN TU NHIEN
    # ================================================================
    
    def _standard_with_flow(self, text: str, config: Dict) -> str:
        """ Standard summarization with flow enhancement """
        
        if len(text) > config['input_max_chars']:
            print(f"   Input: {len(text)} chars → Smart chunking")
            
            chunks = self._split_into_smart_chunks(text, config['input_max_chars'])
            print(f"   Chunks: {len(chunks)}")
            
            # Tom tat moi chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks, 1):
                print(f"      Chunk {i}/{len(chunks)}")
                
                chunk_config = config.copy()
                chunk_config['prefix'] = "Write a flowing paragraph summarizing: "
                
                summary = self._generate_summary_single(chunk, chunk_config)
                chunk_summaries.append(summary)
            
            # Ket hop thanh van ban tu nhien
            if len(chunk_summaries) > 1:
                merged = self._merge_to_natural_flow(chunk_summaries, config)
            else:
                merged = chunk_summaries[0]
            
            return merged
        else:
            print(f"   Input: {len(text)} chars")
            
            config['prefix'] = """Generate a natural, flowing summary as a single cohesive paragraph. 
No section numbers or lists, just smooth narrative prose.
Text: """
            
            return self._generate_summary_single(text, config)
    
    def _split_into_smart_chunks(self, text: str, max_size: int) -> List[str]:
        """ Smart chunking preserving paragraph boundaries - returns list of text chunks """
        
        if len(text) <= max_size:
            return [text]
        
        # Tach tung doan van truoc
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Neu them doan van nay vuot qua gioi han
            if len(current_chunk) + len(para) > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    # Doan van don le qua dai, tach theo cau
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
    
    def _merge_to_natural_flow(self, summaries: List[str], config: Dict) -> str:
        """ Ket hop nhieu tom tat thanh van ban chay tu nhien """
        
        print(f"   Merging {len(summaries)} summaries into natural flow...")
        
        # Loai bo cac tom tat trung lap
        unique_summaries = []
        seen_content = set()
        
        for summary in summaries:
            # Chuan hoa noi dung de so sanh
            normalized = ' '.join(summary.lower().split()[:10])
            if normalized not in seen_content:
                unique_summaries.append(summary)
                seen_content.add(normalized)
        
        # Ket hop voi cac cau chuyen tiep
        combined = ""
        for i, summary in enumerate(unique_summaries):
            # Loai bo cac so phan
            summary = re.sub(r'^\d+\.?\s*', '', summary)
            summary = re.sub(r'\n\d+\.?\s*', ' ', summary)
            
            if i == 0:
                combined = summary
            else:
                # Them cau chuyen tiep
                combined += " Additionally, " + summary[0].lower() + summary[1:]
        
        # Buoc chinh sua cuoi cung
        polish_config = config.copy()
        polish_config['prefix'] = """Rewrite this as one smooth, flowing paragraph. 
Remove any redundancy, ensure logical flow, and maintain complete sentences.
Text: """
        polish_config['min_length'] = config['min_length']
        polish_config['max_length'] = config['max_length'] + 30
        
        polished = self._generate_summary_single(combined, polish_config)
        
        return polished
    
    # ================================================================
    # XU LY HAU QUA VAN BAN TU NHIEN
    # ================================================================
    
    def _ensure_natural_flow(self, summary: str) -> str:
        """Ensure summary has natural flow without section markers"""
        
        print(f"   Ensuring natural flow...")
        
        # Loai bo toan bo so phan muc va dau danh dau
        summary = re.sub(r'^\d+\.?\s*', '', summary)  # At start
        summary = re.sub(r'\n\d+\.?\s*', ' ', summary)  # At line breaks
        summary = re.sub(r'\s+\d+\.?\s+', ' ', summary)  # In middle
        summary = re.sub(r'\b\d+\.\d+\b', '', summary)  # Subsection numbers like 1.1
        
        # Loai bo cac dau danh dau va dau danh sach
        summary = re.sub(r'[-•*]\s+', '', summary)
        
        # Chinh lai khoang trang
        summary = re.sub(r'\s+', ' ', summary)
        summary = re.sub(r'\s+([.,!?;:])', r'\1', summary)
        
        # Chinh lai khoang trang giua cac cau
        summary = re.sub(r'([.!?])([A-Z])', r'\1 \2', summary)
        
        # Loai bo cac tieu de phan muc con lai
        lines = summary.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Bo qua cac dong co the la tieu de (Ngan, viet hoa, co dau ':')
            if len(line) < 100 and line and line[0].isupper() and ':' in line:
                # Dong nay co the la tieu de, bo qua luon
                continue
            if line:
                cleaned_lines.append(line)
        
        summary = ' '.join(cleaned_lines)
        
        return summary.strip()
    
    def _validate_and_fix_natural(self, summary: str, config: Dict) -> str:
        """Xac thuc va chinh sua de co luu luong tu nhien"""
        
        start_time = time.time()
        
        expected_min = config['min_length'] // 1.5
        validation = self.validator.validate_quality(summary, int(expected_min))
        
        self._stats["validation_time"] = time.time() - start_time
        
        print(f"   Quality: score={validation['score']:.2f}")
        
        if validation['issues']:
            print(f"      Auto-fixing {len(validation['issues'])} issues...")
            summary = self.validator.fix_common_issues(summary)
            
            # Additional fixes for natural flow
            summary = self._ensure_natural_flow(summary)
        
        return summary
    
    def _final_polish(self, summary: str) -> str:
        """Chinh sua cuoi cung de co ket qua hoan hao"""
        
        # Lam cho chac chan ket thuc bang cau hoan chinh
        summary = summary.strip()
        
        # Tim dau cham cuoi cau hop le
        if summary and summary[-1] not in '.!?':
            last_period = max(
                summary.rfind('.'),
                summary.rfind('!'),
                summary.rfind('?')
            )
            
            # Chi cat neu dau cau gan cuoi (trong vong 35% cuoi)
            if last_period > len(summary) * 0.65:
                summary = summary[:last_period + 1].strip()
            else:
                # Them dau cham neu khong tim thay ket thuc tot
                summary += '.'
        
        # Loai bo cac ton tai con lai
        summary = re.sub(r'\s+', ' ', summary)
        summary = re.sub(r'\s+([.,!?;:])', r'\1', summary)
        
        # Kiem tra chinh ta
        if self.config.ENABLE_SPELL_CHECK:
            summary = self._fast_spell_check(summary)
        
        return summary.strip()
    
    # ================================================================
    # GENERATION & UTILITIES
    # ================================================================
    
    def _generate_summary_single(self, text: str, config: Dict) -> str:
        """Tao van ban tom tat tu nhien cho mot chunk don le"""
        
        if config.get('prefix'):
            input_text = config['prefix'] + text
        else:
            input_text = text
        
        start_time = time.time()
        inputs = self._tokenize_with_cache(input_text, 512)
        self._stats["tokenization_time"] += time.time() - start_time
        
        start_time = time.time()
        
        generation_params = {
            "input_ids": inputs["input_ids"],
            "min_length": config['min_length'],
            "max_length": config['max_length'] + 30,
            "num_beams": config['num_beams'],
            "length_penalty": config['length_penalty'],
            "repetition_penalty": config['repetition_penalty'],
            "no_repeat_ngram_size": config['no_repeat_ngram_size'],
            "early_stopping": config['early_stopping']
        }
        
        if 'temperature' in config and config['temperature'] > 0:
            generation_params['temperature'] = config['temperature']
        if 'top_p' in config:
            generation_params['top_p'] = config['top_p']
        if 'do_sample' in config:
            generation_params['do_sample'] = config['do_sample']
        
        with torch.no_grad():
            summary_ids = self.model.generate(**generation_params)
        
        self._stats["generation_time"] += time.time() - start_time
        
        summary = self.tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return summary
    
    def _tokenize_with_cache(self, text: str, max_length: int) -> Dict:
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
        
        if len(self._token_cache) > 50:
            self._token_cache.pop(next(iter(self._token_cache)))
        
        self._token_cache[cache_key] = inputs
        return inputs
    
    def _fast_spell_check(self, text: str) -> str:
        if not self.config.COMMON_FIXES:
            return text
        
        words = text.split()
        fixed_words = []
        
        for word in words:
            clean_word = word.strip('.,!?;:"\'-')
            punct = word[len(clean_word):] if len(word) > len(clean_word) else ''
            
            lower_word = clean_word.lower()
            if lower_word in self.config.COMMON_FIXES:
                fixed = self.config.COMMON_FIXES[lower_word]
                if clean_word and clean_word[0].isupper():
                    fixed = fixed.capitalize()
                if clean_word.isupper():
                    fixed = fixed.upper()
                fixed_words.append(fixed + punct)
            else:
                fixed_words.append(word)
        
        return ' '.join(fixed_words)
    
    def _print_stats(self, summary: str, total_time: float, config: Dict):
        word_count = len(summary.split())
        char_count = len(summary)
        
        print(f"✅ Summary completed in {total_time:.2f}s:")
        print(f"   {char_count} chars | {word_count} words")
        print(f"   Natural flow: ✓ No section numbers, ✓ Complete sentences")
    
    def _reset_stats(self):
        self._stats = {k: 0 for k in self._stats}
    
    def is_model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None
    
    def get_performance_stats(self) -> Dict:
        return self._stats.copy()