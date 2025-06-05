import asyncio
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import UserMessage, ModelMessage
from pydantic_ai.mcp import MCPServerHTTP
import os
from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path="../../.env")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_mcp_server_url() -> str:
    """Get MCP server URL from environment variables."""
    host = os.getenv("SERVER_HOST", "localhost")
    port = os.getenv("SERVER_PORT", "8000")
    scheme = os.getenv("SERVER_SCHEME", "http")
    return f"{scheme}://{host}:{port}"


class ProcessingStep(BaseModel):
    """Individual processing step result."""

    step_name: str
    input_text: str
    output_text: str
    status: str
    message: str


class UserInterfaceResponse(BaseModel):
    """Response model for user interface agent."""

    original_text: str
    final_result: str
    operation_type: str
    steps: List[ProcessingStep] = Field(default_factory=list)
    sentiment_analysis: Optional[Dict[str, Any]] = None
    status: str
    message: str
    processing_time: Optional[float] = None


# Context for the user interface agent
class UserInterfaceContext(BaseModel):
    """Context passed to the user interface agent."""

    request_id: str = Field(default="default")
    processing_start_time: float = Field(
        default_factory=lambda: __import__("time").time()
    )


# Create the main User Interface Agent FIRST
user_interface_agent = Agent(
    model=os.getenv("TEXT_OPT_MODEL", "gpt-4"),
    result_type=UserInterfaceResponse,
    system_prompt="""Sie sind ein intelligenter Benutzer-Interface-Agent, der als erster Ansprechpartner für alle Benutzeranfragen fungiert.

Ihre Hauptaufgaben:
1. INTELLIGENTE ANALYSE: Analysieren Sie jede Benutzeranfrage und entscheiden Sie autonom, welche Services benötigt werden
2. MCP-SERVICES: Nutzen Sie verfügbare MCP-Services für Web-Suche, Extraktion, Anonymisierung, PDF-Konvertierung
3. A2A-KOORDINATION: Koordinieren Sie mit spezialisierten Agenten für Textverarbeitung (Sentiment, Optimizer, Lektor, Query Refactor)
4. AUTONOME ENTSCHEIDUNGEN: Treffen Sie eigenständige Entscheidungen über die beste Verarbeitungsstrategie

Verfügbare MCP-Services:
- extract_website_text: Website-Text-Extraktion
- duckduckgo_search: Web-Suche und Wetterinformationen
- anonymize_text: Text-Anonymisierung
- convert_to_pdf: PDF-Konvertierung
- get_current_time: Aktuelle Zeit

A2A-Agents:
- sentiment: Sentiment-Analyse
- optimizer: Text-Optimierung
- lektor: Grammatikkorrektur
- query_ref: Query-Überarbeitung

WICHTIGE ENTSCHEIDUNGSREGELN:
1. Bei Zeit/Datum-Fragen (zeit, uhr, spät, datum, jetzt, heute): IMMER process_time_query verwenden
2. Bei Wetterfragen (wetter, weather, temperatur, regen, schnee, sonne, wind, morgen): search_web_information verwenden
3. Bei Informationsanfragen mit Fragewörtern (wie, was, wo, wann, warum): search_web_information verwenden
4. Bei Textverbesserung/Korrektur: coordinate_with_a2a_agents verwenden
5. Bei URL-Anfragen: extract_website_content verwenden
6. Bei PDF-Konvertierung: convert_file_to_pdf verwenden
7. Bei sensiblen Daten: anonymize_sensitive_text vor weiterer Verarbeitung

BEISPIEL für Zeit-/Datumsanfrage:
Eingabe: "Wie späte ist es jetzt und welches Datum haben wir?"
→ Erkenne: Zeit + Datum
→ Verwende: process_time_query
→ Antworte: Aktuelle UTC-Zeit und Datum mit Hinweis auf Zeitzone

Antworten Sie IMMER mit einem vollständigen UserInterfaceResponse-Objekt mit allen erforderlichen Feldern.""",
)


# Now define the tools AFTER the agent
@user_interface_agent.tool
async def search_web_information(
    ctx: RunContext[UserInterfaceContext], query: str, max_results: int = 5
) -> Dict[str, Any]:
    """Search for information on the web using DuckDuckGo."""
    mcp_server = MCPServerHTTP(_get_mcp_server_url())
    try:
        # Enhance weather queries
        search_query = query
        if any(word in query.lower() for word in ["wetter", "weather"]):
            if "weather" not in query.lower():
                search_query = f"weather {query}"

        result = await mcp_server.call_tool(
            "duckduckgo_search", {"query": search_query, "max_results": max_results}
        )
        return result
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {"results": [], "error": str(e)}
    finally:
        await mcp_server.close()


@user_interface_agent.tool
async def extract_website_content(
    ctx: RunContext[UserInterfaceContext], url: str
) -> Dict[str, Any]:
    """Extract text content from a website."""
    mcp_server = MCPServerHTTP(_get_mcp_server_url())
    try:
        result = await mcp_server.call_tool("extract_website_text", {"url": url})
        return result
    except Exception as e:
        logger.error(f"Website extraction failed: {e}")
        return {"error": str(e)}
    finally:
        await mcp_server.close()


@user_interface_agent.tool
async def anonymize_sensitive_text(
    ctx: RunContext[UserInterfaceContext], text: str
) -> Dict[str, Any]:
    """Anonymize sensitive information in text."""
    mcp_server = MCPServerHTTP(_get_mcp_server_url())
    try:
        result = await mcp_server.call_tool("anonymize_text", {"text": text})
        return result
    except Exception as e:
        logger.error(f"Text anonymization failed: {e}")
        return {"error": str(e)}
    finally:
        await mcp_server.close()


@user_interface_agent.tool
async def get_current_time_info(
    ctx: RunContext[UserInterfaceContext],
) -> Dict[str, Any]:
    """Get current time information."""
    mcp_server = MCPServerHTTP(_get_mcp_server_url())
    try:
        result = await mcp_server.call_tool("get_current_time", {})
        return result
    except Exception as e:
        logger.error(f"Time retrieval failed: {e}")
        return {"error": str(e)}
    finally:
        await mcp_server.close()


@user_interface_agent.tool
async def convert_file_to_pdf(
    ctx: RunContext[UserInterfaceContext],
    file_path: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a file to PDF format."""
    mcp_server = MCPServerHTTP(_get_mcp_server_url())
    try:
        result = await mcp_server.call_tool(
            "convert_to_pdf",
            {"input_filepath": file_path, "output_directory": output_dir},
        )
        return result
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return {"error": str(e)}
    finally:
        await mcp_server.close()


@user_interface_agent.tool
async def coordinate_with_a2a_agents(
    ctx: RunContext[UserInterfaceContext],
    text: str,
    operation: str,
    tonality: Optional[str] = None,
) -> Dict[str, Any]:
    """Coordinate with A2A agents for text processing."""
    try:
        # Import here to avoid circular imports
        import sys
        from pathlib import Path

        # Add parent directory to path for import
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from a2a_server import setup_a2a_server

        registry = await setup_a2a_server()
        results = {}

        try:
            if operation == "sentiment":
                message = [UserMessage(content=text)]
                result = await registry.call_agent("sentiment", message)
                results = {
                    "sentiment": {
                        "label": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "score": result.sentiment.score,
                        "emotions": result.emotions,
                    },
                    "status": result.status,
                    "message": result.message,
                }

            elif operation == "optimize":
                prompt = (
                    f"Optimize this text with tonality '{tonality}': {text}"
                    if tonality
                    else text
                )
                message = [UserMessage(content=prompt)]
                result = await registry.call_agent("optimizer", message)
                results = {
                    "optimized_text": result.optimized_text,
                    "status": result.status,
                    "message": result.message,
                }

            elif operation == "correct":
                message = [UserMessage(content=text)]
                result = await registry.call_agent("lektor", message)
                results = {
                    "corrected_text": result.corrected_text,
                    "status": result.status,
                    "message": result.message,
                }

            elif operation == "refactor":
                message = [UserMessage(content=text)]
                result = await registry.call_agent("query_ref", message)
                results = {
                    "refactored_query": result.query,
                    "status": "success",
                    "message": "Query successfully refactored",
                }

            elif operation == "full_pipeline":
                # Execute full pipeline: refactor → optimize → correct → sentiment
                current_text = text
                pipeline_steps = []

                # Step 1: Refactor
                message = [UserMessage(content=current_text)]
                refactor_result = await registry.call_agent("query_ref", message)
                current_text = refactor_result.query
                pipeline_steps.append({"step": "refactor", "result": current_text})

                # Step 2: Optimize
                optimize_prompt = (
                    f"Optimize this text with tonality '{tonality}': {current_text}"
                    if tonality
                    else current_text
                )
                message = [UserMessage(content=optimize_prompt)]
                optimize_result = await registry.call_agent("optimizer", message)
                current_text = optimize_result.optimized_text
                pipeline_steps.append({"step": "optimize", "result": current_text})

                # Step 3: Correct
                message = [UserMessage(content=current_text)]
                correct_result = await registry.call_agent("lektor", message)
                current_text = correct_result.corrected_text
                pipeline_steps.append({"step": "correct", "result": current_text})

                # Step 4: Sentiment
                message = [UserMessage(content=current_text)]
                sentiment_result = await registry.call_agent("sentiment", message)

                results = {
                    "final_text": current_text,
                    "pipeline_steps": pipeline_steps,
                    "sentiment": {
                        "label": sentiment_result.sentiment.label,
                        "confidence": sentiment_result.sentiment.confidence,
                        "score": sentiment_result.sentiment.score,
                        "emotions": sentiment_result.emotions,
                    },
                    "status": "success",
                    "message": "Full pipeline completed successfully",
                }

        finally:
            await registry.stop()

        return results

    except Exception as e:
        logger.error(f"A2A coordination failed: {e}")
        return {"error": str(e), "status": "error"}


@user_interface_agent.tool
async def analyze_user_request_intent(
    ctx: RunContext[UserInterfaceContext], user_text: str
) -> str:
    """Analyze user request to determine the best processing strategy."""
    text_lower = user_text.lower()

    # Time/Date keywords (German and English) - EXPLICIT time/date requests
    explicit_time_keywords = [
        "zeit",
        "time",
        "uhr",
        "uhrzeit",
        "spät",
        "late",
        "datum",
        "date",
        "wieviel",
        "wie spät",
        "what time",
        "current time",
        "jetzt",
        "now",
        "aktuell",
        "current",
    ]

    # Date-specific phrases that should always be time queries
    date_phrases = [
        "welches datum",
        "what date",
        "welcher tag",
        "what day",
        "datum heute",
        "date today",
        "datum morgen",
        "date tomorrow",
        "welcher monat",
        "what month",
        "welches jahr",
        "what year",
    ]

    # Weather keywords (German and English) - INCLUDING temporal context
    weather_keywords = [
        "wetter",
        "weather",
        "temperatur",
        "temperature",
        "regen",
        "rain",
        "schnee",
        "snow",
        "sonne",
        "sun",
        "wind",
        "wolken",
        "clouds",
        "morgen",
        "tomorrow",
        "übermorgen",
        "vorhersage",
        "forecast",
        "heute",
    ]

    # Weather-specific phrases that should override other detection
    weather_phrases = [
        "wetter morgen",
        "weather tomorrow",
        "wetter heute",
        "weather today",
        "wird das wetter",
        "how will the weather",
        "wie wird das wetter",
    ]

    # Location indicators that suggest weather queries
    location_indicators = [
        "in",
        "für",
        "bei",
        "um",
        "kiel",
        "hamburg",
        "berlin",
        "münchen",
        "germany",
        "deutschland",
        "stadt",
        "city",
    ]

    # PRIORITY 1: Explicit date/time queries (highest priority)
    has_explicit_time = any(keyword in text_lower for keyword in explicit_time_keywords)
    has_date_phrase = any(phrase in text_lower for phrase in date_phrases)

    if has_date_phrase or (
        has_explicit_time and ("datum" in text_lower or "date" in text_lower)
    ):
        return "time_query"

    # PRIORITY 2: Weather queries with clear context
    has_weather_keyword = any(keyword in text_lower for keyword in weather_keywords)
    has_weather_phrase = any(phrase in text_lower for phrase in weather_phrases)
    has_location = any(indicator in text_lower for indicator in location_indicators)

    # Weather detection logic: explicit weather terms OR weather phrases OR weather + location
    if (
        has_weather_phrase
        or (has_weather_keyword and has_location)
        or "wetter" in text_lower
    ):
        return "weather_search"

    # PRIORITY 3: Pure time queries (without explicit date context)
    if has_explicit_time:
        return "time_query"

    # PRIORITY 4: General information search
    if (
        any(
            word in text_lower
            for word in [
                "wie",
                "was",
                "wo",
                "wann",
                "warum",
                "how",
                "what",
                "where",
                "when",
                "why",
            ]
        )
        and "?" in user_text
    ):
        return "information_search"
    elif any(
        word in text_lower
        for word in [
            "korrigiere",
            "correct",
            "optimiere",
            "optimize",
            "verbessere",
            "improve",
        ]
    ):
        return "text_processing"
    elif "http" in text_lower or "www." in text_lower:
        return "url_extraction"
    else:
        return "sentiment_analysis"


@user_interface_agent.tool
async def process_time_query(
    ctx: RunContext[UserInterfaceContext], query: str
) -> UserInterfaceResponse:
    """Specifically handle time and date queries."""
    import time

    start_time = (
        ctx.deps.processing_start_time
        if hasattr(ctx.deps, "processing_start_time")
        else time.time()
    )

    try:
        # Get current time from MCP service
        time_result = await get_current_time_info(ctx)

        if time_result.get("error"):
            raise Exception(time_result["error"])

        current_time_utc = time_result.get("current_time_utc", "Unknown time")

        # Format the response based on what user asked
        text_lower = query.lower()

        if any(word in text_lower for word in ["datum", "date"]) and any(
            word in text_lower for word in ["zeit", "time", "uhr", "spät"]
        ):
            # User asked for both time and date
            response_text = f"Aktuelle Zeit und Datum (UTC): {current_time_utc}\n\nHinweis: Dies ist die koordinierte Weltzeit (UTC). Für Ihre lokale Zeit addieren Sie die entsprechende Zeitzone."
            message = "Aktuelle Zeit und Datum erfolgreich abgerufen"
        elif any(word in text_lower for word in ["datum", "date"]):
            # User asked only for date
            response_text = f"Aktuelles Datum (UTC): {current_time_utc.split('T')[0] if 'T' in current_time_utc else current_time_utc}\n\nHinweis: Dies ist das UTC-Datum."
            message = "Aktuelles Datum erfolgreich abgerufen"
        else:
            # User asked for time
            response_text = f"Aktuelle Zeit (UTC): {current_time_utc}\n\nHinweis: Dies ist die koordinierte Weltzeit (UTC). Für Ihre lokale Zeit addieren Sie die entsprechende Zeitzone."
            message = "Aktuelle Zeit erfolgreich abgerufen"

        return UserInterfaceResponse(
            original_text=query,
            final_result=response_text,
            operation_type="time_query",
            steps=[
                ProcessingStep(
                    step_name="Time Request Analysis",
                    input_text=query,
                    output_text="Erkannt: Zeit- und/oder Datumsanfrage",
                    status="success",
                    message="Benutzeranfrage als Zeit-/Datumsabfrage identifiziert",
                ),
                ProcessingStep(
                    step_name="MCP Time Service Call",
                    input_text="get_current_time request",
                    output_text=current_time_utc,
                    status="success",
                    message="Zeit erfolgreich vom NTP-Server abgerufen",
                ),
                ProcessingStep(
                    step_name="Response Formatting",
                    input_text=current_time_utc,
                    output_text="Formatierte Zeit-/Datumsantwort",
                    status="success",
                    message="Antwort benutzerfreundlich formatiert",
                ),
            ],
            status="success",
            message=message,
            processing_time=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"Time query processing failed: {e}")
        return UserInterfaceResponse(
            original_text=query,
            final_result=f"Entschuldigung, beim Abrufen der aktuellen Zeit ist ein Fehler aufgetreten: {str(e)}",
            operation_type="time_query",
            steps=[
                ProcessingStep(
                    step_name="Error Handling",
                    input_text=query,
                    output_text="Fehler aufgetreten",
                    status="error",
                    message=str(e),
                )
            ],
            status="error",
            message=f"Time query failed: {str(e)}",
            processing_time=time.time() - start_time,
        )


# A2A server function for user interface (standalone function, not a tool)
async def user_interface_a2a_function(
    messages: List[ModelMessage],
) -> UserInterfaceResponse:
    """A2A endpoint for user interface functionality."""
    if not messages:
        return UserInterfaceResponse(
            original_text="",
            final_result="",
            operation_type="a2a_call",
            status="error",
            message="No messages provided",
        )

    # Extract text from the last user message
    last_message = messages[-1]
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        text = last_message.content
    else:
        text = str(last_message)

    # Let the agent make autonomous decisions
    context = UserInterfaceContext(request_id="a2a_request")
    result = await user_interface_agent.run(text, ctx=context)
    return result.data


# Function to process user requests directly through the agent
async def process_user_request(text: str, **kwargs) -> UserInterfaceResponse:
    """Process user input through the intelligent agent."""
    context = UserInterfaceContext(request_id="direct_request")
    result = await user_interface_agent.run(text, ctx=context)
    return result.data


# Example usage functions for testing
async def main():
    """Main function to demonstrate User Interface Agent usage."""
    # Test different types of requests
    test_cases = [
        "Wie wird das Wetter morgen in Kiel?",  # Weather query
        "Also echt mal, dieses Produkt ist der letzte Schrott, total verbuggt und wer das kauft ist doch selber schuld, oder?",  # Text processing
        "Bitte korrigiere diesen Text: Das ist ein sehr schlechte Satz mit viele Fehler.",  # Grammar correction
        "Was ist die Hauptstadt von Deutschland?",  # Information query
        "Wie späte ist es jetzt und welches Datum haben wir?",  # Time and date query
        "Welches datum istmorgen?",  # Ambiguous date query
    ]

    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"Test Case {i}: {test_text}")
        print("=" * 60)

        result = await process_user_request(test_text)

        print(f"Operation: {result.operation_type}")
        print(f"Status: {result.status}")
        print(f"Final Result: {result.final_result[:200]}...")
        print(f"Processing Time: {result.processing_time:.2f}s")

        if result.steps:
            print("\nProcessing Steps:")
            for step in result.steps:
                print(f"  - {step.step_name}: {step.status}")

        if result.sentiment_analysis:
            print(f"\nSentiment: {result.sentiment_analysis}")


if __name__ == "__main__":
    asyncio.run(main())
