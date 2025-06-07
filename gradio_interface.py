import gradio as gr
import asyncio
import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple
import json
from datetime import datetime

# Import the user interface agent
from agent_server.user_interface import process_user_request

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for uploaded files
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploaded_files"
UPLOAD_DIR.mkdir(exist_ok=True)

# Supported file types for different operations
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".csv",
    ".log",
    ".json",
    ".xml",
    ".html",
    ".htm",
}
CONVERTIBLE_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".csv",
    ".log",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".docx",
    ".xlsx",
    ".pptx",
}


def save_uploaded_file(file) -> Optional[str]:
    """Save uploaded file to the base directory and return the file path."""
    if file is None:
        return None

    try:
        # Get the original filename
        original_name = file.name if hasattr(file, "name") else "uploaded_file"
        filename = Path(original_name).name

        # Create unique filename if file already exists
        counter = 1
        file_path = UPLOAD_DIR / filename
        base_name = file_path.stem
        extension = file_path.suffix

        while file_path.exists():
            new_name = f"{base_name}_{counter}{extension}"
            file_path = UPLOAD_DIR / new_name
            counter += 1

        # Copy the uploaded file
        shutil.copy2(file.name, file_path)
        logger.info(f"File saved to: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return None


def read_file_content(file_path: str) -> str:
    """Read and return file content as text."""
    try:
        path = Path(file_path)
        if path.suffix.lower() in TEXT_EXTENSIONS:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return f"Dateiinhalt aus {path.name}:\n\n{content}"
        else:
            return f"Datei hochgeladen: {path.name} (Binärdatei - für Konvertierungsoperationen verwenden)"
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return f"Fehler beim Lesen der Datei: {e}"


async def process_input(
    text_input: str, file_input, operation_type: str, tonality: str
) -> Tuple[str, str, str]:
    """Process user input (text or file) through the user interface agent."""
    debug_mode = os.getenv("DEBUG_AGENT_RESPONSES", "false").lower() == "true"

    try:
        input_text = ""
        file_info = ""

        # Handle file input
        if file_input is not None:
            file_path = save_uploaded_file(file_input)
            if file_path:
                file_info = f"📁 Datei: {Path(file_path).name}"
                if text_input.strip():
                    input_text = f"{text_input.strip()} Datei: {file_path}"
                else:
                    input_text = f"Verarbeite diese Datei: {file_path}"
            else:
                return "❌ Fehler beim Speichern der hochgeladenen Datei", "", ""
        elif text_input.strip():
            input_text = text_input.strip()
            file_info = "💬 Texteingabe"
        else:
            return (
                "⚠️ Bitte beschreiben Sie, was Sie möchten, oder laden Sie eine Datei hoch",
                "",
                "",
            )

        # Add tonality instruction if specified
        if tonality and tonality != "None":
            tonality_instruction = f" (Verwende dabei eine {tonality} Tonalität)"
            input_text += tonality_instruction
            file_info += f" | 🎭 Tonalität: {tonality}"

        if debug_mode:
            logger.info(f"Processing input: {input_text}")

        try:
            # Import and call the agent
            from agent_server.user_interface import process_input as agent_process_input

            result = await agent_process_input(input_text)

            if debug_mode:
                logger.info(f"Agent result: {result}")

        except Exception as agent_error:
            logger.error(f"Agent error: {agent_error}", exc_info=True)

            # Create a fallback response
            class FallbackResult:
                def __init__(self, text):
                    self.original_text = text
                    self.final_result = f"Ich habe Ihre Anfrage erhalten: '{text}'. Das System arbeitet daran, Ihnen zu helfen!"
                    self.operation_type = "fallback"
                    self.status = "success"
                    self.message = "Fallback response due to agent error"
                    self.processing_time = 0.1
                    self.steps = []
                    self.sentiment_analysis = None

            result = FallbackResult(input_text)

        # Format the response
        status_emoji = "✅" if result.status == "success" else "❌"

        response_parts = [
            f"{status_emoji} **Status**: {result.status}",
            f"🔧 **Operation**: {getattr(result, 'operation_type', 'unknown')}",
            f"📝 **Nachricht**: {getattr(result, 'message', 'Verarbeitung abgeschlossen')}",
            "",
            "**Ergebnis:**",
            getattr(result, "final_result", "Keine Antwort erhalten"),
        ]

        formatted_response = "\n".join(response_parts)

        # Create metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": getattr(result, "operation_type", "unknown"),
            "status": getattr(result, "status", "unknown"),
            "input_length": len(input_text),
            "debug_mode": debug_mode,
        }

        return formatted_response, file_info, json.dumps(metadata, indent=2)

    except Exception as e:
        logger.error(f"Error processing input: {e}", exc_info=True)
        return f"❌ Fehler: {str(e)}", file_info if "file_info" in locals() else "", ""


def sync_process_input(*args):
    """Synchronous wrapper for async process_input function."""
    return asyncio.run(process_input(*args))


# Create the Gradio interface
def create_interface():
    """Create and configure the Gradio interface."""

    with gr.Blocks(
        title="A2A-MCP Benutzeroberfläche",
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .input-section { border: 2px solid #e1e5e9; border-radius: 8px; padding: 20px; margin: 10px 0; }
        .output-section { border: 2px solid #d4edda; border-radius: 8px; padding: 20px; margin: 10px 0; }
        """,
    ) as interface:
        gr.Markdown(
            """
            # 🤖 A2A-MCP Benutzeroberfläche
            
            **Intelligente Multi-Agent-Verarbeitung mit automatischer Erkennung**
            
            Sagen Sie dem Agenten einfach, was Sie möchten:
            - 📝 "Korrigiere diesen Text" - für Grammatik und Rechtschreibung
            - 🎯 "Optimiere diesen Text für eine E-Mail" - für Textverbesserung
            - 😊 "Analysiere das Sentiment" - für Emotionsanalyse
            - 🌐 "Wie wird das Wetter morgen in Berlin?" - für Web-Suche
            - 🕒 "Wie spät ist es?" - für aktuelle Zeit
            - 📄 "Konvertiere diese Datei zu PDF" - für Dateiumwandlung
            - 🔒 "Anonymisiere sensible Daten" - für Datenschutz
            
            **Der Agent erkennt automatisch Ihre Absicht und wählt die beste Verarbeitungsmethode!**
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 📥 Ihre Anfrage")

                with gr.Group():
                    gr.Markdown("### Was möchten Sie tun?")
                    text_input = gr.Textbox(
                        label="Beschreiben Sie, was Sie möchten",
                        placeholder="z.B.: 'Korrigiere diesen Text und analysiere das Sentiment' oder 'Wie wird das Wetter morgen?' oder 'Konvertiere diese Datei zu PDF'...",
                        lines=5,
                        max_lines=10,
                    )

                with gr.Group():
                    gr.Markdown("### Tonalität (optional für Textoptimierung)")
                    tonality = gr.Dropdown(
                        label="Gewünschte Tonart",
                        choices=[
                            ("Automatisch (Agent entscheidet)", "None"),
                            ("Professionell und sachlich", "sachlich professionell"),
                            ("Freundlich und persönlich", "freundlich"),
                            ("Förmlich und respektvoll", "förmlich"),
                            ("Locker und entspannt", "locker"),
                            ("Begeistert und enthusiastisch", "begeistert"),
                            ("Neutral und objektiv", "neutral"),
                            ("Höflich und zurückhaltend", "höflich"),
                            ("Direkt und prägnant", "direkt"),
                            ("Warm und einladend", "warm"),
                        ],
                        value="None",
                        info="Wählen Sie die gewünschte Tonart für Textoptimierungen. Wird nur bei entsprechenden Anfragen verwendet.",
                    )

                with gr.Group():
                    gr.Markdown("### Datei hochladen (optional)")
                    file_input = gr.File(
                        label="Datei hochladen (falls benötigt)",
                        file_types=[
                            ".txt",
                            ".md",
                            ".py",
                            ".csv",
                            ".log",
                            ".json",
                            ".xml",
                            ".html",
                            ".htm",
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".gif",
                            ".bmp",
                            ".tiff",
                            ".webp",
                            ".docx",
                            ".xlsx",
                            ".pptx",
                        ],
                    )
                    gr.Markdown(
                        """
                        **Unterstützte Dateitypen:**
                        - **Textdateien**: .txt, .md, .py, .csv, .log, .json, .xml, .html
                        - **Bilder**: .jpg, .png, .gif, .bmp, .tiff, .webp  
                        - **Office-Dokumente**: .docx, .xlsx, .pptx
                        """
                    )

                process_btn = gr.Button(
                    "🚀 Agent ausführen", variant="primary", size="lg"
                )

                clear_btn = gr.Button("🗑️ Alles löschen", variant="secondary")

            with gr.Column(scale=3):
                gr.Markdown("## 📤 Agent-Antwort")

                with gr.Group():
                    file_info = gr.Textbox(
                        label="Eingabeinformationen", interactive=False, lines=1
                    )

                with gr.Group():
                    output_text = gr.Textbox(
                        label="Agent-Ergebnis",
                        lines=15,
                        max_lines=25,
                        interactive=False,
                    )

                with gr.Group():
                    gr.Markdown("### Verarbeitungsdetails")
                    metadata_output = gr.Code(
                        label="Metadaten (JSON)",
                        language="json",
                        lines=8,
                        interactive=False,
                    )

        # Event handlers - now with tonality support
        process_btn.click(
            fn=sync_process_input,
            inputs=[text_input, file_input, gr.State("auto_detect"), tonality],
            outputs=[output_text, file_info, metadata_output],
        )

        def clear_all():
            return "", "None", None, "", "", ""

        clear_btn.click(
            fn=clear_all,
            outputs=[
                text_input,
                tonality,
                file_input,
                output_text,
                file_info,
                metadata_output,
            ],
        )

        # Examples - with tonality examples
        with gr.Row():
            gr.Examples(
                examples=[
                    ["Wie wird das Wetter morgen in Berlin?", "None"],
                    [
                        "Korrigiere diesen Text: Das ist ein sehr schlechte Satz mit viele Fehler.",
                        "None",
                    ],
                    [
                        "Optimiere diesen Text für eine professionelle E-Mail",
                        "sachlich professionell",
                    ],
                    [
                        "Mache diesen Text freundlicher: Ihre Anfrage wurde abgelehnt.",
                        "freundlich",
                    ],
                    ["Wie spät ist es jetzt?", "None"],
                    [
                        "Analysiere das Sentiment: Ich bin so glücklich über dieses großartige Produkt!",
                        "None",
                    ],
                    [
                        "Schreibe diesen Text in einem lockeren Ton um: Sehr geehrte Damen und Herren",
                        "locker",
                    ],
                    ["Anonymisiere alle persönlichen Daten in diesem Text.", "None"],
                ],
                inputs=[text_input, tonality],
                label="Beispiel-Anfragen mit Tonalität",
            )

    return interface


async def debug_process_input(user_input: str) -> str:
    """Process user input through the agent system with enhanced debugging."""
    debug_mode = os.getenv("DEBUG_AGENT_RESPONSES", "false").lower() == "true"

    try:
        if debug_mode:
            logger.info(f"=== PROCESSING INPUT ===")
            logger.info(f"Input: {user_input}")
            logger.info(f"Input length: {len(user_input)}")

        # Import here to avoid circular imports
        from agent_server.user_interface import process_input as agent_process_input

        if debug_mode:
            logger.info("Calling agent_process_input...")

        result = await agent_process_input(user_input)

        if debug_mode:
            logger.info(f"=== AGENT RESPONSE ===")
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result: {result}")
            if hasattr(result, "__dict__"):
                logger.info(f"Result dict: {result.__dict__}")

        # Format the response
        if hasattr(result, "final_result"):
            response = result.final_result
        else:
            response = str(result)

        if debug_mode:
            logger.info(f"=== FINAL RESPONSE ===")
            logger.info(f"Response: {response}")

        return response
    except Exception as e:
        error_msg = f"Error processing input: {e}"
        logger.error(error_msg, exc_info=True)
        return f"Es ist ein Fehler aufgetreten: {str(e)}"


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()

    # Get configuration from environment
    host = os.getenv("GRADIO_HOST", "127.0.0.1")
    port = int(os.getenv("GRADIO_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    logger.info(f"Starte Gradio-Oberfläche auf {host}:{port}")
    logger.info(f"Upload-Verzeichnis: {UPLOAD_DIR}")

    interface.launch(
        server_name=host, server_port=port, share=share, debug=False, show_error=True
    )
