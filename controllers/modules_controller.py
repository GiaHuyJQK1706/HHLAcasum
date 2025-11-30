"""
Modules Controller for managing preprocessing and summarization modules
"""
from typing import Dict, Optional
from modules.module_configs import ModuleConfigs
from modules.preprocessing.preprocessing import Preprocessing
from modules.preprocessing.input_processing import InputProcessing
from modules.summarizer.summarizer import Summarizer


class ModulesController:
    """Manages all modules including preprocessing and summarization"""
    
    def __init__(self):
        self.config = ModuleConfigs()
        self.preprocessing = Preprocessing(self.config)
        self.input_processing = InputProcessing(self.config)
        self.summarizer = Summarizer(self.config)
        self.available_models = [self.config.MODEL_NAME]
        self.current_model = self.config.MODEL_NAME
        self._model_loaded = False
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        return self.available_models
    
    def change_model(self, model_name: str) -> bool:
        """
        Change the current model
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_name not in self.available_models:
                raise ValueError(f"Model {model_name} not available")
            
            self.current_model = model_name
            self.config.MODEL_NAME = model_name
            
            # Unload current model
            if self._model_loaded:
                self.summarizer.unload_model()
                self._model_loaded = False
            
            return True
        except Exception as e:
            print(f"Error changing model: {str(e)}")
            return False
    
    def preload_models(self) -> Dict[str, bool]:
        """
        Preload all models
        
        Returns:
            Dictionary with model load status
        """
        try:
            result = {}
            
            # Load summarization model
            self.summarizer.load_model()
            result['summarizer'] = True
            self._model_loaded = True
            
            return result
        except Exception as e:
            print(f"Error preloading models: {str(e)}")
            return {'summarizer': False}
    
    def scan_models(self) -> list:
        """Scan and return available models"""
        return self.available_models
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        try:
            return self.preprocessing.process(text)
        except Exception as e:
            raise Exception(f"Preprocessing failed: {str(e)}")
    
    def validate_and_extract_file(self, filepath: str) -> str:
        """
        Validate file format and extract text
        
        Args:
            filepath: Path to the file
            
        Returns:
            Extracted text
        """
        try:
            return self.input_processing.extract_from_filepath(filepath)
        except Exception as e:
            raise Exception(f"File extraction failed: {str(e)}")
    
    def validate_input_text(self, text: str) -> dict:
        """
        Validate input text
        
        Args:
            text: Text to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            return self.input_processing.validate_input(text)
        except Exception as e:
            return {
                "valid": False,
                "message": f"Validation error: {str(e)}"
            }
    
    def summarize_text(self, text: str, summary_length: str = "short") -> str:
        """
        Summarize input text
        
        Args:
            text: Text to summarize
            summary_length: Length of summary (short or long)
            
        Returns:
            Generated summary
        """
        try:
            if not self._model_loaded:
                raise Exception("Model not loaded. Call preload_models() first.")
            
            return self.summarizer.summarize(text, summary_length)
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")
    
    def is_model_ready(self) -> bool:
        """Check if model is ready for summarization"""
        return self._model_loaded and self.summarizer.is_model_loaded()