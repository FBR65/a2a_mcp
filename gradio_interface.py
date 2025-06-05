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
    try:
        input_text = ""
        file_info = ""

        # Handle file input
        if file_input is not None:
            file_path = save_uploaded_file(file_input)
            if file_path:
                file_info = f"📁 Datei: {Path(file_path).name}"

                # For file operations, combine user instruction with file path
                if text_input.strip():
                    input_text = f"{text_input.strip()} Datei: {file_path}"
                else:
                    input_text = f"Verarbeite diese Datei: {file_path}"
            else:
                return "❌ Fehler beim Speichern der hochgeladenen Datei", "", ""

        # Handle text input
        elif text_input.strip():
            input_text = text_input.strip()
            file_info = "💬 Texteingabe"
        else:
            return (
                "⚠️ Bitte beschreiben Sie, was Sie möchten, oder laden Sie eine Datei hoch",
                "",
                "",
            )

        # Add tonality instruction if specified and not "None"
        if tonality and tonality != "None":
            tonality_instruction = f" (Verwende dabei eine {tonality} Tonalität)"
            input_text += tonality_instruction
            file_info += f" | 🎭 Tonalität: {tonality}"

        # Always use auto_detect - let the agent decide what to do
        result = await process_user_request(input_text)

        # Format the response
        status_emoji = "✅" if result.status == "success" else "❌"

        # Create detailed response
        response_parts = [
            f"{status_emoji} **Status**: {result.status}",
            f"🔧 **Operation**: {result.operation_type}",
            f"⏱️ **Verarbeitungszeit**: {result.processing_time:.2f}s"
            if result.processing_time
            else "",
            f"📝 **Nachricht**: {result.message}",
            "",
            "**Ergebnis:**",
            result.final_result,
        ]

        # Add processing steps if available
        if result.steps:
            response_parts.extend(["", "**Verarbeitungsschritte:**"])
            for i, step in enumerate(result.steps, 1):
                step_emoji = (
                    "✅"
                    if step.status == "success"
                    else "⚠️"
                    if step.status == "warning"
                    else "❌"
                )
                response_parts.append(
                    f"{i}. {step_emoji} **{step.step_name}**: {step.message}"
                )

        # Add sentiment analysis if available
        if result.sentiment_analysis:
            response_parts.extend(
                [
                    "",
                    "**Sentimentanalyse:**",
                    f"Kategorie: {result.sentiment_analysis.get('label', 'N/A')}",
                    f"Vertrauen: {result.sentiment_analysis.get('confidence', 'N/A')}",
                    f"Bewertung: {result.sentiment_analysis.get('score', 'N/A')}",
                ]
            )
            if result.sentiment_analysis.get("emotions"):
                response_parts.append(
                    f"Emotionen: {result.sentiment_analysis['emotions']}"
                )

        formatted_response = "\n".join(filter(None, response_parts))

        # Create metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": result.operation_type,
            "status": result.status,
            "processing_time": result.processing_time,
            "original_text_length": len(result.original_text),
            "final_result_length": len(result.final_result),
        }

        return formatted_response, file_info, json.dumps(metadata, indent=2)

    except Exception as e:
        logger.error(f"Error processing input: {e}")
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
