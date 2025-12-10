"""
@ file modules/summarizer/summary_validator.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
@ Summary quality validation and coherence checking
"""
import re
from typing import Dict, List, Tuple


class SummaryValidator:
    """Validates summary quality and coherence"""
    
    def __init__(self):
        self.min_word_count = 30
        self.required_punctuation = '.!?'
    
    def validate_quality(self, summary: str, 
                        expected_min_words: int = 50) -> Dict:
        """
        Comprehensive quality validation
        
        Returns:
            Dict with validation results and issues
        """
        issues = []
        warnings = []
        
        # Check 1: Not empty
        if not summary or not summary.strip():
            return {
                'valid': False,
                'score': 0.0,
                'issues': ['Summary is empty'],
                'warnings': []
            }
        
        summary = summary.strip()
        words = summary.split()
        word_count = len(words)
        
        # Check 2: Minimum length
        if word_count < self.min_word_count:
            issues.append(f'Too short: {word_count} words (minimum: {self.min_word_count})')
        elif word_count < expected_min_words:
            warnings.append(f'Shorter than expected: {word_count} words (expected: {expected_min_words}+)')
        
        # Check 3: Proper ending
        if summary[-1] not in self.required_punctuation:
            issues.append('Does not end with proper punctuation')
        
        # Check 4: No incomplete section numbers
        incomplete_sections = self._check_incomplete_sections(summary)
        if incomplete_sections:
            issues.append(f'Incomplete sections found: {incomplete_sections}')
        
        # Check 5: No section number jumps
        section_jumps = self._check_section_jumps(summary)
        if section_jumps:
            issues.append(f'Section number jumps detected: {section_jumps}')
        
        # Check 6: Coherence score
        coherence_score = self._calculate_coherence(summary)
        if coherence_score < 0.5:
            warnings.append(f'Low coherence score: {coherence_score:.2f}')
        
        # Check 7: Repetition
        repetition_score = self._check_repetition(summary)
        if repetition_score > 0.3:
            warnings.append(f'High repetition detected: {repetition_score:.2%}')
        
        # Calculate overall score
        score = self._calculate_overall_score(
            word_count, expected_min_words, coherence_score, 
            repetition_score, len(issues), len(warnings)
        )
        
        return {
            'valid': len(issues) == 0,
            'score': score,
            'issues': issues,
            'warnings': warnings,
            'metrics': {
                'word_count': word_count,
                'coherence': coherence_score,
                'repetition': repetition_score
            }
        }
    
    def _check_incomplete_sections(self, text: str) -> List[str]:
        """Check for incomplete section headers (e.g., ends with '2.')"""
        incomplete = []
        
        # Check if text ends with section number
        if re.search(r'\d+\.\s*$', text):
            incomplete.append('Text ends with section number')
        
        # Check for section headers without content
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if re.match(r'^\d+\.', line.strip()):
                # Check if next line exists and has content
                if i == len(lines) - 1 or (i+1 < len(lines) and len(lines[i+1].strip()) < 10):
                    incomplete.append(f'Section header without content: {line.strip()[:30]}')
        
        return incomplete[:3]  # Return max 3 examples
    
    def _check_section_jumps(self, text: str) -> List[str]:
        """Check for section number discontinuities (1, 3, 5 instead of 1, 2, 3)"""
        # Extract all section numbers
        numbers = []
        for match in re.finditer(r'^(\d+)\.', text, re.MULTILINE):
            numbers.append(int(match.group(1)))
        
        if not numbers:
            return []
        
        jumps = []
        for i in range(len(numbers) - 1):
            if numbers[i+1] - numbers[i] > 1:
                jumps.append(f'{numbers[i]} → {numbers[i+1]}')
        
        return jumps[:3]  # Return max 3 examples
    
    def _calculate_coherence(self, text: str) -> float:
        """
        Calculate coherence score based on:
        - Sentence connectivity
        - Use of transition words
        - Consistent tense
        """
        score = 1.0
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            return 1.0
        
        # Transition words (simple check)
        transitions = [
            'however', 'moreover', 'furthermore', 'therefore', 'thus',
            'additionally', 'consequently', 'meanwhile', 'subsequently',
            'in addition', 'as a result', 'for example', 'in contrast'
        ]
        
        transition_count = sum(1 for sent in sentences 
                              if any(t in sent.lower() for t in transitions))
        transition_ratio = transition_count / len(sentences)
        
        # More transitions = better coherence (up to 0.4 ratio)
        score *= (1.0 + min(transition_ratio, 0.4))
        
        # Penalize very short sentences
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_length < 10:
            score *= 0.8
        
        return min(score, 1.0)
    
    def _check_repetition(self, text: str) -> float:
        """Calculate repetition score (0 = no repetition, 1 = high repetition)"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) < 2:
            return 0.0
        
        # Check for repeated sentences
        unique_sentences = set()
        repeated = 0
        
        for sent in sentences:
            # Normalize sentence
            normalized = ' '.join(sent.lower().split())
            
            if normalized in unique_sentences:
                repeated += 1
            else:
                unique_sentences.add(normalized)
        
        return repeated / len(sentences) if sentences else 0.0
    
    def _calculate_overall_score(self, word_count: int, expected_min: int,
                                 coherence: float, repetition: float,
                                 num_issues: int, num_warnings: int) -> float:
        """Calculate overall quality score (0-1)"""
        score = 1.0
        
        # Length factor
        if word_count < expected_min:
            score *= (word_count / expected_min)
        
        # Coherence factor
        score *= coherence
        
        # Repetition penalty
        score *= (1.0 - repetition * 0.5)
        
        # Issues penalty
        score *= (1.0 - num_issues * 0.2)
        
        # Warnings penalty
        score *= (1.0 - num_warnings * 0.05)
        
        return max(0.0, min(1.0, score))
    
    def fix_common_issues(self, summary: str) -> str:
        """Auto-fix common issues in summary"""
        
        # Fix 1: Remove incomplete sections at the end
        summary = re.sub(r'\d+\.\s*$', '', summary)
        
        # Fix 2: Remove section headers without content at the end
        lines = summary.split('\n')
        while lines and re.match(r'^\d+\.', lines[-1].strip()):
            lines.pop()
        summary = '\n'.join(lines)
        
        # Fix 3: Ensure proper ending
        summary = summary.strip()
        if summary and summary[-1] not in '.!?':
            # Find last proper sentence ending
            last_period = max(
                summary.rfind('.'),
                summary.rfind('!'),
                summary.rfind('?')
            )
            if last_period > len(summary) * 0.65:
                summary = summary[:last_period + 1]
            else:
                summary += '.'
        
        # Fix 4: Remove duplicate consecutive sentences
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        unique = []
        prev = None
        
        for sent in sentences:
            normalized = ' '.join(sent.lower().split())
            if normalized != prev:
                unique.append(sent)
                prev = normalized
        
        summary = ' '.join(unique)
        
        # Fix 5: Clean up spacing
        summary = re.sub(r'\s+', ' ', summary)
        summary = re.sub(r'\s+([.,!?;:])', r'\1', summary)
        
        return summary.strip()
    