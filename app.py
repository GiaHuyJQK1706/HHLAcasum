"""
@ app.py: Main application entry point for Academic Text Summarizer
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
@ Update: Change model
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from controllers.main_controller import MainController
from controllers.error_controller import ErrorController
from ui.app_ui import AppUI
from ui.event_handles import EventHandles


def check_and_setup_models():
    """Check and setup models before launching"""
    try:
        from model_setup import check_local_model, download_and_setup_model
        
        MODEL_PATH = "./models/hhlai_acasum_t5_base"
        MODEL_NAME = "hhlai/hhlai_acasum_t5_base"
        
        print("\n" + "="*60)
        print("🔍 Checking Model Setup...")
        print("="*60)
        
        # Check if local model exists
        if check_local_model(MODEL_PATH):
            print(f"✅ Local model found at: {MODEL_PATH}\n")
            return True
        else:
            print(f"⚠️ Local model not found, attempting to download...")
            if download_and_setup_model(MODEL_NAME, MODEL_PATH):
                print(f"✅ Model downloaded successfully!\n")
                return True
            else:
                print(f"❌ Failed to setup model\n")
                return False
    
    except Exception as e:
        print(f"⚠️ Warning: Could not verify model: {str(e)}")
        print("The model will be downloaded on first use.\n")
        return True


def main():
    """Initialize and run the application"""
    try:
        # Check and setup models
        if not check_and_setup_models():
            print("❌ Model setup failed. Please check your installation.")
            sys.exit(1)
        
        print("="*60)
        print("🚀 Initializing Application...")
        print("="*60 + "\n")
        
        # Initialize controllers
        main_controller = MainController()
        error_controller = ErrorController()
        
        # Initialize event handlers
        event_handles = EventHandles(
            main_controller=main_controller,
            error_controller=error_controller
        )
        
        # Initialize UI
        app_ui = AppUI(event_handles=event_handles)
        
        # Launch the application
        app_ui.launch()
        
    except Exception as e:
        print(f"❌ Fatal error starting application: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    