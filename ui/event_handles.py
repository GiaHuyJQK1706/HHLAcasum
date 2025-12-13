"""
@ file ui/event_handles.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
"""
from typing import Optional, Tuple
from pathlib import Path
from controllers.main_controller import MainController
from controllers.error_controller import ErrorController


class EventHandles:
    """Handles all UI events"""
    
    def __init__(self, main_controller: MainController, error_controller: ErrorController):
        self.main_controller = main_controller
        self.error_controller = error_controller
        self.model_initialized = False
    
    def initialize_model(self) -> Tuple[bool, str]:
        try:
            if self.model_initialized:
                return True, "Model already initialized"
            
            print("⏳ Initializing model for first time...")
            result = self.main_controller.initialize_system()
            
            if result["success"]:
                self.model_initialized = True
                return True, result["message"]
            else:
                return False, result["message"]
        
        except Exception as e:
            error_msg = str(e)
            self.error_controller.log_error(error_msg, "system")
            return False, f"Model initialization failed: {error_msg}"
    
    def on_import_file_clicked(self, file_obj) -> Tuple[str, str, str]:
        try:
            if file_obj is None:
                return "⚠️ Please select a file", "", "❌ Error"
            
            result = self.main_controller.import_text_from_file(file_obj.name)
            
            if result["success"]:
                preview = result["text"][:500] + ("..." if len(result["text"]) > 16384 else "")
                return f"✅ {result['message']}", preview, "📄 Success"
            else:
                self.error_controller.log_error(result["message"], "file")
                return f"❌ {result['message']}", "", "❌ Error"
        
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            self.error_controller.log_error(str(e), "system")
            return f"❌ {error_msg}", "", "❌ Error"
    
    def on_import_text_direct(self, text: str) -> Tuple[str, str, str]:
        try:
            if not text or not text.strip():
                return "⚠️ Please enter some text", "", "❌ Error"
            
            result = self.main_controller.import_text_direct(text)
            
            if result["success"]:
                preview = result["text"][:16384] + ("..." if len(result["text"]) > 16384 else "")
                return f"✅ {result['message']}", preview, "📄 Success"
            else:
                self.error_controller.log_error(result["message"], "validation")
                return f"⚠️ {result['message']}", "", "❌ Error"
        
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            self.error_controller.log_error(str(e), "system")
            return f"❌ {error_msg}", "", "❌ Error"
    
    def on_summarize_clicked(self, summary_length: str) -> Tuple[str, str, str]:
        try:
            # Initialize model if not already done
            if not self.model_initialized:
                print("⏳ Loading model...")
                success, msg = self.initialize_model()
                if not success:
                    return f"❌ {msg}", "", "❌ Error"
            
            if summary_length not in ["short", "long"]:
                return "⚠️ Invalid summary length selected", "", "❌ Error"
            
            # Check if text is imported
            if not self.main_controller.current_text:
                return "⚠️ Please import text first before summarizing", "", "❌ Error"
            
            print(f"📝 Summarizing ({summary_length})...")
            result = self.main_controller.summarize_current_text(summary_length)
            
            if result["success"]:
                summary = result["summary"]
                metrics = result["metrics"]
                metrics_str = f"📊 Metrics: Original: {metrics['original_length']} chars | Summary: {metrics['summary_length']} chars | Ratio: {metrics['compression_ratio']}%"
                print(f"✅ {result['message']}")
                return f"✅ {result['message']}\n{metrics_str}", summary, "📝 Summary"
            else:
                self.error_controller.log_error(result["message"], "processing")
                return f"❌ {result['message']}", "", "❌ Error"
        
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            self.error_controller.log_error(str(e), "system")
            return f"❌ {error_msg}", "", "❌ Error"
    
    def on_view_result_clicked(self) -> str:
        try:
            summary = self.main_controller.get_current_summary()
            if not summary:
                return "No summary available. Please generate a summary first."
            return summary
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            return f"Error: {error_msg}"
    
    def on_view_error_clicked(self) -> str:
        try:
            error = self.main_controller.get_last_error()
            if not error:
                return "No errors recorded."
            return f"Last Error:\n{error}"
        except Exception as e:
            return f"Error retrieving error message: {str(e)}"
        
    def on_view_stats_clicked(self) -> str:
        try:
            stats = self.main_controller.get_user_stats()
            if not stats:
                return "No statistics available."
            return stats
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            return f"Error retrieving statistics: {error_msg}"
            
    
    def on_view_history_clicked(self) -> str:
        try:
            history = self.main_controller.get_summary_history()
            if not history:
                return "No history available."
            
            # Format as plain text (better for Textbox display)
            formatted = "=" * 80 + "\n"
            formatted += "SUMMARY HISTORY (Most Recent First)\n"
            formatted += "=" * 80 + "\n\n"
            
            for i, record in enumerate(history, 1):
                formatted += f"{'─' * 80}\n"
                formatted += f"RECORD #{i}\n"
                formatted += f"{'─' * 80}\n"
                formatted += f"Timestamp: {record['created_at']}\n"
                formatted += f"Type: {record['summary_length'].upper()}\n"
                formatted += f"Original Length: {record['original_length']} chars\n"
                formatted += f"Summary Length: {record['summary_length_chars']} chars\n"
                formatted += f"Compression Ratio: {record['compression_ratio']}%\n"
                formatted += f"\n{'─' * 80}\n"
                formatted += f"ORIGINAL TEXT:\n"
                formatted += f"{'─' * 80}\n"
                formatted += f"{record['original_text']}\n"  # Full original text
                formatted += f"\n{'─' * 80}\n"
                formatted += f"SUMMARY:\n"
                formatted += f"{'─' * 80}\n"
                formatted += f"{record['summary_text']}\n"  # Full summary
                formatted += f"\n\n"
            
            return formatted
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            return f"Error retrieving history: {error_msg}"
    
    def on_search_history_clicked(self, query: str) -> str:
        try:
            results = self.main_controller.search_summaries(query)
            if not results:
                return f"No results found for: '{query}'"
            
            formatted = "=" * 80 + "\n"
            formatted += f"SEARCH RESULTS FOR: '{query}'\n"
            formatted += "=" * 80 + "\n"
            formatted += f"Found {len(results)} result(s)\n\n"
            
            for i, record in enumerate(results, 1):
                formatted += f"{'─' * 80}\n"
                formatted += f"RESULT #{i}\n"
                formatted += f"{'─' * 80}\n"
                formatted += f"Timestamp: {record['created_at']}\n"
                formatted += f"Type: {record['summary_length'].upper()}\n"
                formatted += f"Original Length: {record['original_length']} chars\n"
                formatted += f"Compression: {record['compression_ratio']}%\n"
                formatted += f"\n{'─' * 80}\n"
                formatted += f"ORIGINAL TEXT:\n"
                formatted += f"{'─' * 80}\n"
                formatted += f"{record['original_text']}\n"  # Full original text
                formatted += f"\n{'─' * 80}\n"
                formatted += f"SUMMARY:\n"
                formatted += f"{'─' * 80}\n"
                formatted += f"{record['summary_text']}\n"  # Full summary
                formatted += f"\n\n"
            
            return formatted
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            return f"Error searching summaries: {error_msg}"
    
    def on_view_history_clicked_json(self) -> list:
        try:
            return self.main_controller.get_summary_history()
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            self.error_controller.log_error(str(e), "system")
            return []
    
    def on_search_history_clicked_json(self, query: str) -> list:
        try:
            if not query or not query.strip():
                return []
            
            results = self.main_controller.search_summaries(query)
            return results
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            self.error_controller.log_error(str(e), "system")
            return []
    
    def on_view_stats_clicked_json(self) -> str:
        """ View user statistics as formatted text (for Textbox display) """
        try:
            stats = self.main_controller.get_user_stats()
            
            formatted = """
════════════════════════════════════════════════════════════════
                     📊 YOUR STATISTICS
════════════════════════════════════════════════════════════════

Total Summaries Created:        {}
Short Summaries:                {}
Long Summaries:                 {}
Average Compression Ratio:      {}%

════════════════════════════════════════════════════════════════
""".format(
                stats.get('total_summaries', 0),
                stats.get('short_summaries', 0),
                stats.get('long_summaries', 0),
                stats.get('average_compression_ratio', 0)
            )
            return formatted
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            return f"Error retrieving statistics: {error_msg}"
        """
        View user statistics
        
        Returns:
            Formatted statistics as Markdown
        """
        try:
            stats = self.main_controller.get_user_stats()
            
            formatted = "## 📊 Your Statistics\n\n"
            formatted += f"- **Total Summaries**: {stats.get('total_summaries', 0)}\n"
            formatted += f"- **Short Summaries**: {stats.get('short_summaries', 0)}\n"
            formatted += f"- **Long Summaries**: {stats.get('long_summaries', 0)}\n"
            formatted += f"- **Average Compression**: {stats.get('average_compression_ratio', 0)}%\n"
            
            return formatted
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            return f"Error retrieving statistics: {error_msg}"
    
    def on_clear_history_clicked(self) -> str:
        try:
            result = self.main_controller.clear_history()
            if result.get("success"):
                return f"{result.get('message', 'History cleared')}"
            else:
                return f"{result.get('message', 'Failed to clear history')}"
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            return f"Error: {error_msg}"
    
    def on_export_txt(self) -> Tuple[str, str]:
        try:
            summary = self.main_controller.get_current_summary()
            if not summary:
                return None, "❌ No summary to export."
            
            import tempfile
            from datetime import datetime
            import os
            
            # Create temp directory if not exists
            export_dir = Path("./exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate safe filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{timestamp}.txt"
            filepath = export_dir / filename
            
            # Write summary to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"📥 File created: {filepath}")
            print(f"📊 File size: {filepath.stat().st_size} bytes")
            
            return str(filepath), f"✅ File ready for download: {filename}"
        
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            print(f"❌ Export error: {error_msg}")
            return None, f"❌ Export failed: {error_msg}"
    
    def on_clear_session(self) -> str:
        try:
            self.main_controller.clear_session()
            return "✅ Session cleared successfully"
        except Exception as e:
            error_msg = self.error_controller.get_error_message(e)
            return f"❌ Error clearing session: {error_msg}"