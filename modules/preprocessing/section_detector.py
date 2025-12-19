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
    """Dai dien mot phan muc trong van ban"""
    title: str
    content: str
    level: int
    section_number: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0
    section_type: str = "content"  # content, introduction, conclusion, etc.


class SectionDetector:
    """Phat hien phan muc van ban nang cao"""
    
    def __init__(self):
        # Cac mau de phat hien tieu de phan muc
        self.patterns = [
            # Numbered sections (1., 1.1, 1.1.1)
            (r'^(\d+(?:\.\d+)*)\.\s+(.+?)$', 'numbered'),
            
            # Named sections (Introduction, Methods, etc.)
            (r'^([A-Z][A-Za-z\s]{2,40})\s*$', 'named'),
            
            # ALL CAPS headings
            (r'^([A-Z\s]{3,40})$', 'caps'),
            
            # Markdown-style headers (# Header)
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
        Phat hien cac phan muc trong van ban
        Tra ve danh sach cac doi tuong Section kem theo thong tin chi tiet (metadata)
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
            
            # Thu phan tieu de phan muc
            section_info = self._match_section_header(line_stripped, i)
            
            if section_info:
                # Luu phan muc truoc do
                if current_section and current_section.content.strip():
                    sections.append(current_section)
                
                # Bat dau phan muc moi
                current_section = Section(
                    title=section_info['title'],
                    content="",
                    level=section_info['level'],
                    section_number=section_info.get('number'),
                    start_pos=i,
                    section_type=self._classify_section_type(section_info['title'])
                )
            else:
                # Them vao noi dung phan muc hien tai
                if current_section:
                    current_section.content += line_stripped + " "
                else:
                    # Chua co phan muc nao, tao phan muc gioi thieu mac dinh
                    current_section = Section(
                        title="Introduction",
                        content=line_stripped + " ",
                        level=0,
                        start_pos=i,
                        section_type="introduction"
                    )
        
        # Them phan muc cuoi cung
        if current_section and current_section.content.strip():
            sections.append(current_section)
        
        # Xu ly hau qua: gop cac phan muc rat nho
        sections = self._merge_small_sections(sections)
        
        return sections
    
    def _match_section_header(self, line: str, line_num: int) -> Optional[Dict]:
        """Thu phan tieu de phan muc"""
        
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
        """Phan loai phan muc dua tren tu khoa trong tieu de"""
        title_lower = title.lower()
        
        for section_type, keywords in self.section_keywords.items():
            if any(kw in title_lower for kw in keywords):
                return section_type
        
        return 'content'
    
    def _merge_small_sections(self, sections: List[Section], 
                              min_length: int = 100) -> List[Section]:
        """Gop cac phan muc qua nho"""
        if not sections:
            return sections
        
        merged = []
        i = 0
        
        while i < len(sections):
            current = sections[i]
            
            # Neu phan muc hien tai qua ngan, gop voi phan muc tiep theo
            if len(current.content) < min_length and i < len(sections) - 1:
                # Gop voi phan muc tiep theo
                next_section = sections[i + 1]
                next_section.content = current.content + " " + next_section.content
                i += 1  # Bo qua phan muc hien tai
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
        
        # Dem loai phan muc
        for section in sections:
            structure['section_types'][section.section_type] = \
                structure['section_types'].get(section.section_type, 0) + 1
            
            if section.section_type == 'introduction':
                structure['has_introduction'] = True
            elif section.section_type == 'conclusion':
                structure['has_conclusion'] = True
        
        # Tinh do dai trung binh phan muc
        total_length = sum(len(s.content) for s in sections)
        structure['average_section_length'] = total_length // len(sections)
        
        # Xac dinh xem co cau truc tot (co gioi thieu + ket luan + 2+ phan muc)
        structure['is_well_structured'] = (
            structure['has_introduction'] and 
            structure['has_conclusion'] and 
            len(sections) >= 3
        )
        
        return structure
    
    def extract_key_sentences(self, text: str, top_k: int = 10) -> List[str]:
        """
        Trích xuất câu quan trọng sử dụng TF-IDF
        """
        from collections import Counter
        import math
        
        # Tách văn bản thành các câu
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= top_k:
            return sentences
        
        # Tính document frequency (số câu chứa mỗi từ)
        doc_freq = Counter()
        for sentence in sentences:
            words = set(w.lower() for w in re.findall(r'\b\w+\b', sentence))
            for word in words:
                doc_freq[word] += 1
        
        # Tính TF-IDF cho từng câu
        N = len(sentences)  # Tổng số câu
        sentence_scores = []
        
        for idx, sentence in enumerate(sentences):
            words = [w.lower() for w in re.findall(r'\b\w+\b', sentence)]
            if not words:
                continue
            
            # Tính TF cho từng từ trong câu
            tf = Counter(words)
            
            # Tính TF-IDF score cho câu
            tfidf_score = 0
            for word in set(words):
                # TF: tần suất từ trong câu
                term_freq = tf[word] / len(words)
                
                # IDF: log(N / df)
                idf = math.log(N / doc_freq[word]) if doc_freq[word] > 0 else 0
                
                # TF-IDF
                tfidf_score += term_freq * idf
            
            # Chuẩn hóa theo số từ unique
            tfidf_score /= len(set(words))
            
            # Các yếu tố bổ trợ
            if re.search(r'\d+', sentence): # Nếu câu có số, tăng điểm
                tfidf_score *= 1.2
            
            position_boost = 1.0 - (idx / N) * 0.3 # Câu đầu có điểm cao hơn
            tfidf_score *= position_boost
            
            sentence_scores.append((tfidf_score, sentence, idx))
        
        # Lấy top-k câu
        top_sentences = sorted(sentence_scores, reverse=True)[:top_k]
        top_sentences.sort(key=lambda x: x[2])
        
        return [s[1] for s in top_sentences]
    