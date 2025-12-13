"""
@ file ui/app_ui.py
@ Copyright (C) 2025 by Gia-Huy Do & HHL Team
@ v0.99: Add HHL Theme, add About tab
"""
import gradio as gr
from ui.event_handles import EventHandles


class AppUI:
    """Main UI application using Gradio"""

    def __init__(self, event_handles: EventHandles):
        self.event_handles = event_handles
        self._initialize_model()
        self.interface = self._create_interface()

    def _initialize_model(self):
        """Initialize model at startup"""
        try:
            print("\n" + "=" * 60)
            print("📦 Loading Model at Startup...")
            print("=" * 60)
            success, message = self.event_handles.initialize_model()
            if success:
                print(f"✅ {message}\n")
            else:
                print(f"⚠️ {message}\n")
        except Exception as e:
            print(f"⚠️ Warning during model initialization: {str(e)}\n")

    def _create_interface(self):
        """Create the Gradio interface"""
        HHL_theme = gr.themes.Soft(
            primary_hue=gr.themes.colors.emerald,
            secondary_hue=gr.themes.colors.emerald,
            neutral_hue=gr.themes.colors.slate,
            radius_size=gr.themes.sizes.radius_md
        )

        with gr.Blocks(title="HHL Acasum", theme=HHL_theme) as interface:
            
            gr.Markdown("""
            # HHL Acasum

            A powerful application for summarizing academic texts using state-of-the-art Transformer models.
            """
            )

            with gr.Tab("Summarization"):
                gr.Markdown("### Import Text")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Upload File**")
                        file_input = gr.File(
                            label="Choose file (.txt, .pdf, .docx)",
                            file_types=[".txt", ".pdf", ".docx"]
                        )
                        import_file_btn = gr.Button("Import from File", variant="primary")

                    with gr.Column():
                        gr.Markdown("**or Direct Input**")
                        text_input = gr.Textbox(
                            label="Enter text directly",
                            lines=8,
                            max_lines=8,
                            placeholder="Paste your academic text here..."
                        )
                        import_text_btn = gr.Button("Import Text", variant="primary")

                # Preview section - fixed (no max_lines None)
                gr.Markdown("### Text Preview")
                preview_text = gr.Textbox(
                    label="Complete Input Text",
                    lines=5,
                    max_lines=9,
                    interactive=False,
                    placeholder="Text preview will appear here...",
                    show_copy_button=True
                )

                status_message = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready to import text"
                )

                # Summarization section
                gr.Markdown("### Summarize")
                with gr.Row():
                    summary_length = gr.Radio(
                        choices=["short", "long"],
                        value="short",
                        label="Summary Length"
                    )
                    summarize_btn = gr.Button("Generate Summary", variant="primary")

                # Result section
                gr.Markdown("### View Result & Download")
                with gr.Row():
                    view_result_btn = gr.Button("View Summary", scale=1)
                    view_error_btn = gr.Button("View Error", scale=1)
                    export_btn = gr.Button("Download", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Session", scale=1)

                result_display = gr.Textbox(
                    label="Result",
                    lines=10,
                    max_lines=25,
                    interactive=False,
                    placeholder="Result will appear here...",
                    show_copy_button=True
                )

                export_file = gr.File(label="Download File")
                export_status = gr.Textbox(label="Export Status", interactive=False)

                # Hidden dummy component — created in layout and reused in outputs
                hidden_box = gr.Textbox(visible=False)

                # Event handlers - use only components created above (no inline component creation)
                import_file_btn.click(
                    fn=self.event_handles.on_import_file_clicked,
                    inputs=[file_input],
                    outputs=[status_message, preview_text, hidden_box]
                )

                import_text_btn.click(
                    fn=self.event_handles.on_import_text_direct,
                    inputs=[text_input],
                    outputs=[status_message, preview_text, hidden_box]
                )

                summarize_btn.click(
                    fn=self.event_handles.on_summarize_clicked,
                    inputs=[summary_length],
                    outputs=[status_message, result_display, hidden_box]
                )

                view_result_btn.click(
                    fn=self.event_handles.on_view_result_clicked,
                    outputs=[result_display]
                )

                view_error_btn.click(
                    fn=self.event_handles.on_view_error_clicked,
                    outputs=[result_display]
                )

                export_btn.click(
                    fn=self.event_handles.on_export_txt,
                    outputs=[export_file, export_status]
                )

                clear_btn.click(
                    fn=self.event_handles.on_clear_session,
                    outputs=[status_message]
                )

            # History Tab - use HTML rendering for stability
            with gr.Tab("Summary History"):
                gr.Markdown("### Summary History")

                with gr.Row():
                    search_query = gr.Textbox(
                        label="Search summaries",
                        placeholder="Enter keyword to search...",
                        scale=4
                    )
                    
                with gr.Row():
                    search_btn = gr.Button("Search", variant="primary", scale=1)
                    stats_btn = gr.Button("Statistics", scale=1)
                    refresh_history_btn = gr.Button("Refresh History", scale=1) 
                    clear_history_btn = gr.Button("Clear All History", scale=1)

                # History display uses HTML string (safe and simple)
                history_display = gr.HTML(value="<p>Click 'Refresh' to load history</p>", label="History Results")

                def refresh_history():
                    history = self.event_handles.on_view_history_clicked_json()
                    return self._format_history_as_html(history)

                def search_history(query):
                    if not query or not query.strip():
                        return "<p>Please enter a search query</p>"
                    results = self.event_handles.on_search_history_clicked_json(query)
                    if not results:
                        return f"<p>No results found for: <strong>{query}</strong></p>"
                    return self._format_history_as_html(results)

                def show_stats():
                    stats = self.event_handles.on_view_stats_clicked()
                    # build simple HTML for stats
                    return f"""
                    <div style="background-color:#f7f9fb;padding:12px;border-radius:8px;">
                        <h3>Statistics</h3>
                        <p><strong>Total Summaries:</strong> {stats.get('total_summaries', 0)}</p>
                        <p><strong>Short Summaries:</strong> {stats.get('short_summaries', 0)}</p>
                        <p><strong>Long Summaries:</strong> {stats.get('long_summaries', 0)}</p>
                        <p><strong>Average Compression:</strong> {stats.get('average_compression_ratio', 0)}%</p>
                    </div>
                    """

                def clear_all():
                    result = self.event_handles.on_clear_history_clicked()

                    if "Error" in result or "Failed" in result:
                        return f"<p style='color:red;'>{result}</p>"
                    else:
                        return f"<p style='color:green;'>{result}</p>"


                refresh_history_btn.click(fn=refresh_history, outputs=[history_display])
                search_btn.click(fn=search_history, inputs=[search_query], outputs=[history_display])
                stats_btn.click(fn=show_stats, outputs=[history_display])
                clear_history_btn.click(fn=clear_all, outputs=[history_display])

            # About Tab
            with gr.Tab("About"):
                gr.Markdown("""
                ## About HHL Acasum - Academic Text Summarizer

                ### Features
                -  Full text display without truncation  
                -  Support for multiple file formats  
                -  Adjustable summary length (short/long)  
                -  Automatic history tracking  
                -  Download and search capabilities  
                -  Organized history view with separate boxes  

                ## Contact Information
                - **Developer**: Gia-Huy Do, HHL Team  
                - **Email**: [huydogiaccac@gmail.com](mailto:huydogiaccac@gmail.com)  
                - **GitHub**: [github.com/GiaHuyJQK1706/HHLAcasum](https://github.com/GiaHuyJQK1706/HHLAcasum)  
                - **Facebook**: [facebook.com/hhlteamlab](https://facebook.com/hhlteamlab)
                """)


        return interface

    def _format_history_as_html(self, history_list):
        """Format history records as HTML for display"""
        if not history_list:
            return "<p>No history to display</p>"

        html = "<div style='display:flex;flex-direction:column;gap:18px;'>"

        for i, record in enumerate(history_list, 1):
            original_text = (record.get('original_text') or "").replace("\n", "<br/>")
            summary_text = (record.get('summary_text') or "").replace("\n", "<br/>")
            created_at = record.get('created_at', 'N/A')
            summary_length = record.get('summary_length', 'N/A')
            original_length = record.get('original_length', 0)
            # summary_length_chars = record.get('summary_length_chars', 0)
            compression_ratio = record.get('compression_ratio', 0)

            block = f"""
            <div style="border:1px solid #e1e8f0;border-radius:8px;padding:12px;background:#fff;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <div>
                        <h4 style="margin:0 0 4px 0;">Record #{i}</h4>
                        <div style="font-size:13px;color:#555;">
                            <span><strong>Date:</strong> {created_at}</span> &nbsp;|&nbsp;
                            <span><strong>Type:</strong> {summary_length}</span>
                        </div>
                    </div>
                    <div style="text-align:right;font-size:13px;color:#333;">
                        <div>Original: <strong>{original_length}</strong> chars</div>
                        <div>Compression ratio: <strong>{compression_ratio}%</strong></div>
                    </div>
                </div>

                <div style="margin-bottom:10px;">
                    <h5 style="margin:0 0 6px 0;">Original Text</h5>
                    <div style="max-height:220px;overflow:auto;padding:10px;border:1px solid #f0f0f0;border-radius:6px;background:#fafafa;font-family:monospace;font-size:13px;">
                        {original_text}
                    </div>
                </div>

                <div>
                    <h5 style="margin:0 0 6px 0;">Generated Summary</h5>
                    <div style="max-height:180px;overflow:auto;padding:10px;border:1px solid #f0f0f0;border-radius:6px;background:#fff;font-family:monospace;font-size:13px;">
                        {summary_text}
                    </div>
                </div>
            </div>
            """
            html += block

        html += "</div>"
        return html

    def launch(self, share: bool = False, server_name: str = "127.0.0.1",
               server_port: int = 7860):
        """Launch the Gradio interface"""
        print(f"\n{'=' * 60}")
        print("🚀 Starting Academic Text Summarizer...")
        print(f"{'=' * 60}")
        print(f"Access the application at: http://{server_name}:{server_port}")
        print(f"{'=' * 60}\n")

        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True
        )
