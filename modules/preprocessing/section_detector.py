"""
@ file modules/preprocessing/section_detector.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
@ Advanced section detection for structured documents
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Section:
    """Represents a document section"""
    title: str
    content: str
    level: int
    section_number: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0
    section_type: str = "content"  # content, introduction, conclusion, etc.


class SectionDetector:
    """Advanced document section detection"""
    
    def __init__(self):
        # Section patterns - ordered by priority
        self.patterns = [
            # Pattern 1: Numbered sections (1., 1.1, 1.1.1)
            (r'^(\d+(?:\.\d+)*)\.\s+(.+?)$', 'numbered'),
            
            # Pattern 2: Named sections (Introduction, Methods, etc.)
            (r'^([A-Z][A-Za-z\s]{2,40})\s*$', 'named'),
            
            # Pattern 3: ALL CAPS headings
            (r'^([A-Z\s]{3,40})$', 'caps'),
            
            # Pattern 4: Markdown-style headers (# Header)
            (r'^(#{1,6})\s+(.+?)$', 'markdown'),
        ]
        
        # Known section types for classification
        self.section_keywords = {
            'introduction': ['introduction', 'overview', 'background', 'motivation'],
            'methods': ['methods', 'methodology', 'approach', 'implementation'],
            'results': ['results', 'findings', 'evaluation', 'experiments'],
            'discussion': ['discussion', 'analysis', 'interpretation'],
            'conclusion': ['conclusion', 'summary', 'future work', 'conclusions'],
        }
    
    def detect_sections(self, text: str) -> List[Section]:
        """
        Detect all sections in document
        
        Returns:
            List of Section objects with metadata
        """
        if not text or not text.strip():
            return []
        
        lines = text.split('\n')
        sections = []
        current_section = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Try to match section header
            section_info = self._match_section_header(line_stripped, i)
            
            if section_info:
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
                # Add to current section content
                if current_section:
                    current_section.content += line_stripped + " "
                else:
                    # No section yet, create default intro section
                    current_section = Section(
                        title="Introduction",
                        content=line_stripped + " ",
                        level=0,
                        start_pos=i,
                        section_type="introduction"
                    )
        
        # Add last section
        if current_section and current_section.content.strip():
            sections.append(current_section)
        
        # Post-process: merge very small sections
        sections = self._merge_small_sections(sections)
        
        return sections
    
    def _match_section_header(self, line: str, line_num: int) -> Optional[Dict]:
        """Try to match line against section patterns"""
        
        for pattern, pattern_type in self.patterns:
            match = re.match(pattern, line)
            
            if match:
                if pattern_type == 'numbered':
                    section_num = match.group(1)
                    title = match.group(2)
                    level = len(section_num.split('.'))
                    return {
                        'title': title,
                        'number': section_num,
                        'level': level,
                        'type': pattern_type
                    }
                
                elif pattern_type == 'named':
                    title = match.group(1).strip()
                    # Only consider as header if it's short and title-like
                    if len(title.split()) <= 8 and title[0].isupper():
                        return {
                            'title': title,
                            'number': None,
                            'level': 1,
                            'type': pattern_type
                        }
                
                elif pattern_type == 'caps':
                    title = match.group(1).strip()
                    if len(title.split()) <= 6:
                        return {
                            'title': title.title(),  # Convert to Title Case
                            'number': None,
                            'level': 1,
                            'type': pattern_type
                        }
                
                elif pattern_type == 'markdown':
                    hashes = match.group(1)
                    title = match.group(2)
                    level = len(hashes)
                    return {
                        'title': title,
                        'number': None,
                        'level': level,
                        'type': pattern_type
                    }
        
        return None
    
    def _classify_section_type(self, title: str) -> str:
        """Classify section based on title keywords"""
        title_lower = title.lower()
        
        for section_type, keywords in self.section_keywords.items():
            if any(kw in title_lower for kw in keywords):
                return section_type
        
        return 'content'
    
    def _merge_small_sections(self, sections: List[Section], 
                              min_length: int = 100) -> List[Section]:
        """Merge sections that are too small"""
        if not sections:
            return sections
        
        merged = []
        i = 0
        
        while i < len(sections):
            current = sections[i]
            
            # If section is too small and not the last one
            if len(current.content) < min_length and i < len(sections) - 1:
                # Merge with next section
                next_section = sections[i + 1]
                next_section.content = current.content + " " + next_section.content
                i += 1  # Skip current
            else:
                merged.append(current)
                i += 1
        
        return merged
    
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
        
        # Calculate average length
        total_length = sum(len(s.content) for s in sections)
        structure['average_section_length'] = total_length // len(sections)
        
        # Determine if well-structured (has intro + conclusion + 2+ sections)
        structure['is_well_structured'] = (
            structure['has_introduction'] and 
            structure['has_conclusion'] and 
            len(sections) >= 3
        )
        
        return structure
    
    def extract_key_sentences(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract key sentences using simple TF-IDF scoring
        
        Args:
            text: Input text
            top_k: Number of sentences to extract
        
        Returns:
            List of key sentences in original order
        """
        from collections import Counter
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= top_k:
            return sentences
        
        # Calculate word frequencies
        all_words = []
        for sentence in sentences:
            words = [w.lower() for w in re.findall(r'\b\w+\b', sentence)]
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        # Score sentences
        sentence_scores = []
        for sentence in sentences:
            words = [w.lower() for w in re.findall(r'\b\w+\b', sentence)]
            if not words:
                continue
            
            # TF score (normalized by sentence length)
            score = sum(word_freq[w] for w in words) / len(words)
            
            # Boost score for sentences with numbers/stats
            if re.search(r'\d+', sentence):
                score *= 1.2
            
            # Boost for sentences at beginning (likely important)
            position_boost = 1.0 - (sentences.index(sentence) / len(sentences)) * 0.3
            score *= position_boost
            
            sentence_scores.append((score, sentence, sentences.index(sentence)))
        
        # Get top-k sentences
        top_sentences = sorted(sentence_scores, reverse=True)[:top_k]
        
        # Sort by original order
        top_sentences.sort(key=lambda x: x[2])
        
        return [s[1] for s in top_sentences]