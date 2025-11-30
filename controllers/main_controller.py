"""
Main Controller - Orchestrates all operations
"""
from typing import Dict, Optional
from controllers.modules_controller import ModulesController
from controllers.data_controller import DataController
from modules.summarizer.summary_evaluator import SummaryEvaluator


class MainController:
    """Main controller orchestrating all operations"""
    
    def __init__(self):
        self.modules_controller = ModulesController()
        self.data_controller = DataController()
        self.evaluator = SummaryEvaluator()
        
        # Current session state
        self.current_text = None
        self.current_summary = None
        self.last_error = None
        self.model_ready = False
        self.user_id = None
        
        # Initialize user
        self.user_id = self.data_controller.initialize_user()
    
    def initialize_system(self) -> Dict:
        """
        Initialize the system and load models
        
        Returns:
            Initialization status
        """
        try:
            if self.model_ready:
                return {
                    "success": True,
                    "message": "Model already initialized"
                }
            
            print("\n⏳ Preloading models...")
            # Preload models
            model_status = self.modules_controller.preload_models()
            
            if model_status.get('summarizer'):
                self.model_ready = True
                print("✅ Models preloaded successfully!")
                return {
                    "success": True,
                    "message": "System initialized successfully"
                }
            else:
                raise Exception("Failed to load summarization model")
        
        except Exception as e:
            self.last_error = str(e)
            print(f"❌ {str(e)}")
            return {
                "success": False,
                "message": f"Initialization failed: {str(e)}"
            }
    
    def import_text_from_file(self, filepath: str) -> Dict:
        """
        Import text from file (UC01)
        
        Args:
            filepath: Path to the file
            
        Returns:
            Status and extracted text
        """
        try:
            # Extract text from file
            text = self.modules_controller.validate_and_extract_file(filepath)
            
            # Store current text
            self.current_text = text
            self.last_error = None
            
            return {
                "success": True,
                "text": text,
                "message": "Text imported successfully"
            }
        
        except Exception as e:
            self.last_error = str(e)
            return {
                "success": False,
                "message": f"Import failed: {str(e)}"
            }
    
    def import_text_direct(self, text: str) -> Dict:
        """
        Import text directly from input (UC01)
        
        Args:
            text: Input text
            
        Returns:
            Status and validation result
        """
        try:
            # Validate input
            validation = self.modules_controller.validate_input_text(text)
            
            if not validation.get("valid"):
                self.last_error = validation.get("message")
                return {
                    "success": False,
                    "message": validation.get("message")
                }
            
            # Store current text
            self.current_text = text
            self.last_error = None
            
            return {
                "success": True,
                "text": text,
                "message": "Text imported successfully"
            }
        
        except Exception as e:
            self.last_error = str(e)
            return {
                "success": False,
                "message": f"Import failed: {str(e)}"
            }
    
    def summarize_current_text(self, summary_length: str = "short") -> Dict:
        """
        Summarize the current imported text (UC02)
        
        Args:
            summary_length: Length of summary (short or long)
            
        Returns:
            Status and generated summary
        """
        try:
            if not self.current_text:
                raise Exception("No text to summarize. Please import text first.")
            
            # Check if model is ready, if not initialize
            if not self.model_ready:
                print("⏳ Model not ready, initializing...")
                init_result = self.initialize_system()
                if not init_result["success"]:
                    raise Exception(init_result["message"])
            
            # Check if model is still not ready
            if not self.modules_controller.is_model_ready():
                raise Exception("Model initialization failed. Please try again.")
            
            # Preprocess text
            preprocessed_text = self.modules_controller.preprocess_text(self.current_text)
            
            # Generate summary
            summary = self.modules_controller.summarize_text(preprocessed_text, summary_length)
            
            # Store current summary
            self.current_summary = summary
            
            # Evaluate summary
            metrics = self.evaluator.evaluate_summary(self.current_text, summary)
            
            # Save to history
            self.data_controller.add_summary_record(
                original_text=self.current_text,
                summary=summary,
                summary_length=summary_length
            )
            
            self.last_error = None
            
            return {
                "success": True,
                "summary": summary,
                "metrics": metrics,
                "message": "Summarization completed successfully"
            }
        
        except Exception as e:
            self.last_error = str(e)
            return {
                "success": False,
                "message": f"Summarization failed: {str(e)}"
            }
    
    def get_current_summary(self) -> str:
        """Get the current generated summary (UC03)"""
        return self.current_summary or ""
    
    def get_summary_history(self) -> list:
        """Get summary history (UC05)"""
        try:
            return self.data_controller.get_summary_history()
        except Exception as e:
            self.last_error = str(e)
            return []
    
    def search_summaries(self, query: str) -> list:
        """Search summaries by keyword"""
        try:
            return self.data_controller.search_summaries(query)
        except Exception as e:
            self.last_error = str(e)
            return []
    
    def get_user_stats(self) -> dict:
        """Get user statistics"""
        try:
            return self.data_controller.get_user_stats()
        except Exception as e:
            self.last_error = str(e)
            return {}
    
    def clear_history(self) -> dict:
        """Clear all history"""
        try:
            return self.data_controller.clear_history()
        except Exception as e:
            self.last_error = str(e)
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message (UC03)"""
        return self.last_error
    
    def clear_session(self) -> None:
        """Clear current session data"""
        self.current_text = None
        self.current_summary = None
        self.last_error = None