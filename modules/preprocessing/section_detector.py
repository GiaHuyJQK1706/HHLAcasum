"""
@ file modules/preprocessing/section_detector.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
@ v1: Fixed: extract key sentences (TF-IDF -> PyTextRank) 
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Section:
    """Represents a section in the document"""
    title: str
    content: str
    level: int
    section_number: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0
    section_type: str = "content"


class SectionDetector:
    
    def __init__(self):
        # Patterns ordered by priority (most specific first)
        self.patterns = [
            # Subsections with multiple dots (2.1, 2.1.1, etc.)
            (r'^(\d+\.\d+(?:\.\d+)*)\s+(.+)$', 'numbered_sub'),
            
            # Main numbered sections (1., 2., 3.)
            (r'^(\d+)\.\s+(.+)$', 'numbered'),
            
            # Roman numerals (I., II., III.)
            (r'^([IVX]+)\.\s+(.+)$', 'roman'),
            
            # Markdown headers (## Header)
            (r'^(#{1,6})\s+(.+)$', 'markdown'),
            
            # ALL CAPS HEADERS
            (r'^([A-Z][A-Z\s]{2,50})$', 'caps'),
        ]
        
        # Known section keywords
        self.section_keywords = {
            'introduction': ['introduction', 'overview', 'background', 'motivation', 'abstract', 'preface'],
            'methods': ['methods', 'methodology', 'approach', 'implementation', 'design', 'architecture'],
            'results': ['results', 'findings', 'evaluation', 'experiments', 'performance', 'benchmarking'],
            'discussion': ['discussion', 'analysis', 'interpretation', 'challenges'],
            'conclusion': ['conclusion', 'summary', 'future', 'conclusions', 'final', 'closing'],
        }
    
    def detect_sections(self, text: str) -> List[Section]:
        """
        Detect all sections - FIXED to not over-filter
        """
        if not text or not text.strip():
            return []
        
        lines = text.split('\n')
        sections = []
        current_section = None
        
        print(f"\n🔍 Detecting sections in document ({len(lines)} lines)...")
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Try to match as section header
            section_info = self._match_section_header(line_stripped, i, lines)
            
            if section_info:
                # Debug print
                print(f"   ✓ Line {i}: Found section '{section_info['title']}' (level {section_info['level']})")
                
                # Save previous section
                if current_section and current_section.content.strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = Section(
                    title=section_info['title'],
                    content="",
                    level=section_info['level'],
                    section_number=section_info.get('number'),
                    start_pos=i,
                    section_type=self._classify_section_type(section_info['title'])
                )
            else:
                # Add content to current section
                if current_section:
                    current_section.content += line_stripped + " "
                else:
                    # First content before any header
                    if len(line_stripped) > 20:
                        current_section = Section(
                            title="Document Content",
                            content=line_stripped + " ",
                            level=0,
                            start_pos=i,
                            section_type="content"
                        )
        
        # Add last section
        if current_section and current_section.content.strip():
            sections.append(current_section)
        
        print(f"   → Initial detection: {len(sections)} sections")
        
        # CRITICAL: Only remove sections that are TRULY too small
        # Don't be too aggressive with filtering
        valid_sections = []
        for section in sections:
            word_count = len(section.content.split())
            # Very lenient threshold - only remove if really empty
            if word_count >= 5:  # Changed from 10 to 5
                valid_sections.append(section)
            else:
                print(f"   ⚠ Removed too-small section: '{section.title}' ({word_count} words)")
        
        print(f"   → After filtering: {len(valid_sections)} sections")
        
        # IMPORTANT: Don't treat as unstructured if we found multiple sections
        # Only return single "Document" section if we truly found NOTHING
        if len(valid_sections) == 0:
            print(f"   → No valid sections, treating as plain document")
            return [Section(
                title="Document",
                content=text,
                level=0,
                section_type="content"
            )]
        
        # If we found sections, return them!
        print(f"   ✅ Final: {len(valid_sections)} sections detected\n")
        return valid_sections
    
    def _match_section_header(self, line: str, line_num: int, all_lines: List[str]) -> Optional[Dict]:
        """
        Match section header - SIMPLIFIED and more lenient
        """
        
        # Skip very long lines (clearly not headers)
        if len(line) > 200:
            return None
        
        # Try each pattern
        for pattern, pattern_type in self.patterns:
            match = re.match(pattern, line)
            
            if not match:
                continue
            
            # NUMBERED SUBSECTION (1.1, 2.3, 3.2.1, etc.)
            if pattern_type == 'numbered_sub':
                section_num = match.group(1)
                title = match.group(2).strip()
                
                # Basic validation
                if len(title.split()) > 20:  # Title too long
                    continue
                
                # Must have SOME content after (very lenient)
                if not self._has_minimal_content_after(line_num, all_lines):
                    continue
                
                level = len(section_num.split('.'))
                return {
                    'title': title,
                    'number': section_num,
                    'level': level,
                    'type': pattern_type
                }
            
            # SIMPLE NUMBERED SECTION (1., 2., 3., etc.)
            elif pattern_type == 'numbered':
                section_num = match.group(1)
                title = match.group(2).strip()
                
                # Basic validation
                if len(title.split()) > 20:
                    continue
                
                # Must have content after
                if not self._has_minimal_content_after(line_num, all_lines):
                    continue
                
                return {
                    'title': title,
                    'number': section_num,
                    'level': 1,
                    'type': pattern_type
                }
            
            # ROMAN NUMERALS
            elif pattern_type == 'roman':
                section_num = match.group(1)
                title = match.group(2).strip()
                
                if len(title.split()) > 20:
                    continue
                
                if not self._has_minimal_content_after(line_num, all_lines):
                    continue
                
                return {
                    'title': title,
                    'number': section_num,
                    'level': 1,
                    'type': pattern_type
                }
            
            # MARKDOWN HEADERS
            elif pattern_type == 'markdown':
                hashes = match.group(1)
                title = match.group(2).strip()
                
                if len(title.split()) > 20:
                    continue
                
                level = len(hashes)
                return {
                    'title': title,
                    'number': None,
                    'level': level,
                    'type': pattern_type
                }
            
            # ALL CAPS
            elif pattern_type == 'caps':
                title = match.group(1).strip()
                word_count = len(title.split())
                
                # Must be reasonable length
                if not (2 <= word_count <= 10):
                    continue
                
                if not self._has_minimal_content_after(line_num, all_lines):
                    continue
                
                return {
                    'title': title.title(),
                    'number': None,
                    'level': 1,
                    'type': pattern_type
                }
        
        return None
    
    def _has_minimal_content_after(self, line_num: int, lines: List[str]) -> bool:
        """
        VERY LENIENT check - just verify there's SOMETHING after this line
        This prevents treating the last line as a header
        """
        # Look at next few lines
        for i in range(line_num + 1, min(line_num + 3, len(lines))):
            line = lines[i].strip()
            
            # If we find any non-empty line with reasonable length, it's valid
            if line and len(line) > 10:
                return True
        
        # No content found
        return False
    
    def _classify_section_type(self, title: str) -> str:
        """Classify section based on title"""
        title_lower = title.lower()
        
        for section_type, keywords in self.section_keywords.items():
            if any(kw in title_lower for kw in keywords):
                return section_type
        
        return 'content'
    
    def get_document_structure(self, sections: List[Section]) -> Dict:
        """Analyze document structure"""
        structure = {
            'total_sections': len(sections),
            'has_introduction': False,
            'has_conclusion': False,
            'section_types': {},
            'average_section_length': 0,
            'is_well_structured': False
        }
        
        if not sections:
            return structure
        
        # Count section types
        for section in sections:
            structure['section_types'][section.section_type] = \
                structure['section_types'].get(section.section_type, 0) + 1
            
            if section.section_type == 'introduction':
                structure['has_introduction'] = True
            elif section.section_type == 'conclusion':
                structure['has_conclusion'] = True
        
        # Calculate average section length
        total_length = sum(len(s.content) for s in sections)
        structure['average_section_length'] = total_length // len(sections) if sections else 0
        
        structure['is_well_structured'] = len(sections) >= 2     
        return structure
    
    def extract_key_sentences(self, text: str, top_k: int = 10) -> List[str]:
        try:
            import spacy
            import pytextrank
            
            # Load spacy model with pytextrank
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("⚠️ Downloading spacy model...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                nlp = spacy.load("en_core_web_sm")
            
            # Add PyTextRank to the pipeline
            if "textrank" not in nlp.pipe_names:
                nlp.add_pipe("textrank")
            
            # Split into sentences for validation
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) <= top_k:
                return sentences
            
            # Process text with PyTextRank
            doc = nlp(text)
            
            # Extract top sentences
            top_sentences = []
            for sent in doc._.textrank.summary(limit_sentences=top_k):
                sentence_text = sent.text.strip()
                if len(sentence_text) > 20:
                    top_sentences.append(sentence_text)
            
            # If not enough sentences, add more from original
            if len(top_sentences) < top_k:
                extracted_set = set(top_sentences)
                for sent in sentences:
                    if sent not in extracted_set and len(top_sentences) < top_k:
                        top_sentences.append(sent)
            
            # Sort by original order in text
            sentence_positions = {}
            for sent in top_sentences:
                pos = text.find(sent)
                if pos != -1:
                    sentence_positions[sent] = pos
            
            sorted_sentences = sorted(
                top_sentences, 
                key=lambda s: sentence_positions.get(s, float('inf'))
            )
            
            return sorted_sentences[:top_k]
            
        except ImportError as e:
            print(f"⚠️ PyTextRank not installed: {str(e)}")
            print("   Install with: pip install pytextrank spacy")
            print("   Then run: python -m spacy download en_core_web_sm")
            print("   Falling back to simple extraction...")
            
            # Fallback
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            return sentences[:min(top_k, len(sentences))]
            
        except Exception as e:
            print(f"⚠️ PyTextRank extraction failed: {str(e)}")
            print("   Falling back to simple extraction...")
            
            # Fallback
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            return sentences[:min(top_k, len(sentences))]
    