"""
Input file processing module for handling various file formats
"""
from pathlib import Path
from typing import Tuple
from modules.module_configs import ModuleConfigs


class InputProcessing:
    """Handles input file processing and text extraction"""
    
    def __init__(self, config: ModuleConfigs = None):
        self.config = config or ModuleConfigs()
    
    def validate_input(self, text: str) -> dict:
        """
        Validate input text
        
        Args:
            text: Input text to validate
            
        Returns:
            Dictionary with validation result and message
        """
        try:
            if not text or not text.strip():
                return {
                    "valid": False,
                    "message": "Input text cannot be empty"
                }
            
            text_length = len(text)
            if text_length < self.config.MIN_TEXT_LENGTH:
                return {
                    "valid": False,
                    "message": f"Text is too short. Minimum {self.config.MIN_TEXT_LENGTH} characters required. Current: {text_length}"
                }
            
            if text_length > self.config.MAX_TEXT_LENGTH:
                return {
                    "valid": True,
                    "message": f"Text length ({text_length}) exceeds maximum ({self.config.MAX_TEXT_LENGTH}). Will be truncated.",
                    "truncated": True
                }
            
            return {
                "valid": True,
                "message": "Input text is valid"
            }
        except Exception as e:
            return {
                "valid": False,
                "message": f"Validation error: {str(e)}"
            }
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Remove control characters
            text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')
            return text.strip()
        except Exception as e:
            raise Exception(f"Text cleaning failed: {str(e)}")
    
    def extract_from_file(self, filepath: str) -> str:
        """
        Extract text from uploaded file
        
        Args:
            filepath: Path to the file
            
        Returns:
            Extracted text
        """
        try:
            path = Path(filepath)
            
            # Check file extension
            if path.suffix.lower() not in self.config.ALLOWED_FILE_TYPES:
                raise ValueError(
                    f"File type not supported. Allowed types: {', '.join(self.config.ALLOWED_FILE_TYPES)}"
                )
            
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.MAX_FILE_SIZE_MB:
                raise ValueError(
                    f"File too large. Maximum size: {self.config.MAX_FILE_SIZE_MB}MB"
                )
            
            # Extract based on file type
            if path.suffix.lower() == ".txt":
                return self._extract_txt(filepath)
            elif path.suffix.lower() == ".pdf":
                return self._extract_pdf(filepath)
            elif path.suffix.lower() == ".docx":
                return self._extract_docx(filepath)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        
        except Exception as e:
            raise Exception(f"File extraction failed: {str(e)}")
    
    def _extract_txt(self, filepath: str) -> str:
        """Extract text from .txt file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return self.clean_text(text)
        except Exception as e:
            raise Exception(f"Failed to extract from .txt file: {str(e)}")
    
    def _extract_pdf(self, filepath: str) -> str:
        """Extract text from .pdf file"""
        try:
            import PyPDF2
            text = ""
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + " "
            return self.clean_text(text)
        except ImportError:
            raise Exception("PyPDF2 library not installed. Install with: pip install PyPDF2")
        except Exception as e:
            raise Exception(f"Failed to extract from .pdf file: {str(e)}")
    
    def _extract_docx(self, filepath: str) -> str:
        """Extract text from .docx file"""
        try:
            import docx
            doc = docx.Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return self.clean_text(text)
        except ImportError:
            raise Exception("python-docx library not installed. Install with: pip install python-docx")
        except Exception as e:
            raise Exception(f"Failed to extract from .docx file: {str(e)}")
    
    def extract_from_filepath(self, filepath: str) -> str:
        """
        Extract text from file path
        
        Args:
            filepath: Path to the file
            
        Returns:
            Extracted text
        """
        return self.extract_from_file(filepath)