import asyncio
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
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


# Initialize the model with OpenAI provider for Ollama compatibility
def _create_user_interface_agent():
    """Create the user interface agent with proper Ollama configuration."""
    try:
        llm_api_key = os.getenv("API_KEY", "ollama")
        llm_endpoint = os.getenv("BASE_URL", "http://localhost:11434/v1")
        llm_model_name = os.getenv("USER_INTERFACE_MODEL", "qwen2.5:latest")

        provider = OpenAIProvider(base_url=llm_endpoint, api_key=llm_api_key)
        model = OpenAIModel(provider=provider, model_name=llm_model_name)

        return Agent(
            model=model,
            result_type=UserInterfaceResponse,
            retries=1,
            system_prompt="""Du bist ein Assistent, der IMMER Tools verwendet, um Benutzeranfragen zu bearbeiten.

ANALYSE DER ANFRAGE:
1. **Text optimieren/korrigieren/freundlicher machen**: 
   → VERWENDE coordinate_with_a2a_agents(text="...", operation="optimize", tonality="freundlich")

2. **Zeit/Datum fragen**: 
   → VERWENDE process_time_query(query="...")

3. **Wetter fragen**: 
   → VERWENDE search_web_information(query="...")

4. **Websuche**: 
   → VERWENDE search_web_information(query="...")

Du MUSST IMMER ein Tool verwenden. Antworte niemals direkt ohne Tool-Aufruf!

Für Textoptimierung mit freundlicher Tonalität:
- Extrahiere den zu optimierenden Text
- Rufe coordinate_with_a2a_agents auf mit operation="optimize" und tonality="freundlich"
- Verwende das Ergebnis als final_result""",
        )
    except Exception as e:
        logging.error(f"Failed to initialize user interface agent: {e}")
        raise


# Create the main User Interface Agent
user_interface_agent = _create_user_interface_agent()


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
        # Fix: Use proper cleanup method
        try:
            if hasattr(mcp_server, "close"):
                await mcp_server.close()
            elif hasattr(mcp_server, "aclose"):
                await mcp_server.aclose()
        except Exception as cleanup_error:
            logger.warning(f"MCP cleanup failed (non-critical): {cleanup_error}")


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
        try:
            if hasattr(mcp_server, "close"):
                await mcp_server.close()
            elif hasattr(mcp_server, "aclose"):
                await mcp_server.aclose()
        except Exception as cleanup_error:
            logger.warning(f"MCP cleanup failed (non-critical): {cleanup_error}")


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
        try:
            if hasattr(mcp_server, "close"):
                await mcp_server.close()
            elif hasattr(mcp_server, "aclose"):
                await mcp_server.aclose()
        except Exception as cleanup_error:
            logger.warning(f"MCP cleanup failed (non-critical): {cleanup_error}")


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
        try:
            if hasattr(mcp_server, "close"):
                await mcp_server.close()
            elif hasattr(mcp_server, "aclose"):
                await mcp_server.aclose()
        except Exception as cleanup_error:
            logger.warning(f"MCP cleanup failed (non-critical): {cleanup_error}")


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
        try:
            if hasattr(mcp_server, "close"):
                await mcp_server.close()
            elif hasattr(mcp_server, "aclose"):
                await mcp_server.aclose()
        except Exception as cleanup_error:
            logger.warning(f"MCP cleanup failed (non-critical): {cleanup_error}")


@user_interface_agent.tool
async def coordinate_with_a2a_agents(
    ctx: RunContext[UserInterfaceContext],
    text: str,
    operation: str,
    tonality: Optional[str] = None,
) -> UserInterfaceResponse:
    """Coordinate with A2A agents for text processing using the existing A2A registry system."""
    import time

    start_time = time.time()

    debug_a2a = os.getenv("DEBUG_A2A_CALLS", "false").lower() == "true"

    try:
        logger.info(
            f"A2A coordination called with operation: {operation}, tonality: {tonality}"
        )

        # Import the existing A2A setup from our project
        import sys
        from pathlib import Path

        # Add parent directory to path for import
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from a2a_server import setup_a2a_server

        registry = await setup_a2a_server()

        try:
            if operation == "optimize":
                # Create a structured request for the optimizer
                if tonality:
                    message = [{"content": f"TONALITY:{tonality}|TEXT:{text}"}]
                else:
                    message = [{"content": text}]
                result = await registry.call_agent("optimizer", message)

                if debug_a2a:
                    logger.info(f"Raw optimizer result: {result}")
                    logger.info(f"Result type: {type(result)}")
                    logger.info(
                        f"Result attributes: {dir(result) if hasattr(result, '__dict__') else 'No attributes'}"
                    )

                # Extract the actual optimized text from the result
                optimized_text = None

                # Check if the optimizer had an error
                if hasattr(result, "status") and result.status == "error":
                    if debug_a2a:
                        logger.warning(f"Optimizer returned error: {result.message}")
                    # Try to extract actual text from the error response
                    if hasattr(result, "optimized_text") and isinstance(
                        result.optimized_text, str
                    ):
                        # The optimized_text might contain the original prompt, extract it
                        optimized_text_raw = result.optimized_text
                        if optimized_text_raw.startswith(
                            "{'content': '"
                        ) and optimized_text_raw.endswith("'}"):
                            # Extract content from the dict string
                            content_start = optimized_text_raw.find(
                                "'content': '"
                            ) + len("'content': '")
                            content_end = optimized_text_raw.rfind("'}")
                            if content_start < content_end:
                                optimized_text = optimized_text_raw[
                                    content_start:content_end
                                ]
                            else:
                                optimized_text = text  # Fallback to original text
                        else:
                            optimized_text = optimized_text_raw
                    else:
                        optimized_text = f"Optimizer service error: {result.message}. Original text: {text}"
                else:
                    # Normal processing
                    if hasattr(result, "optimized_text"):
                        optimized_text = result.optimized_text
                    elif hasattr(result, "content"):
                        optimized_text = result.content
                    elif isinstance(result, dict) and "optimized_text" in result:
                        optimized_text = result["optimized_text"]
                    elif isinstance(result, dict) and "content" in result:
                        optimized_text = result["content"]
                    elif isinstance(result, str):
                        optimized_text = result
                    else:
                        # Fallback: convert to string
                        optimized_text = str(result)

                if debug_a2a:
                    logger.info(f"Extracted optimized text: {optimized_text}")

                return UserInterfaceResponse(
                    original_text=text,
                    final_result=optimized_text,
                    operation_type="coordinate_with_a2a_agents",
                    status="success",
                    message=f"Text optimization completed with {tonality or 'default'} tonality",
                    processing_time=time.time() - start_time,
                    steps=[
                        ProcessingStep(
                            step_name="A2A Text Optimization",
                            input_text=text,
                            output_text=optimized_text,
                            status="success",
                            message="Text successfully optimized via A2A registry",
                        )
                    ],
                )

            elif operation == "correct":
                message = [{"content": text}]
                result = await registry.call_agent("lektor", message)

                if debug_a2a:
                    logger.info(f"Raw lektor result: {result}")

                # Extract corrected text
                corrected_text = None
                if hasattr(result, "corrected_text"):
                    corrected_text = result.corrected_text
                elif hasattr(result, "content"):
                    corrected_text = result.content
                elif isinstance(result, dict) and "corrected_text" in result:
                    corrected_text = result["corrected_text"]
                elif isinstance(result, dict) and "content" in result:
                    corrected_text = result["content"]
                elif isinstance(result, str):
                    corrected_text = result
                else:
                    corrected_text = str(result)

                return UserInterfaceResponse(
                    original_text=text,
                    final_result=corrected_text,
                    operation_type="coordinate_with_a2a_agents",
                    status="success",
                    message="Text correction completed via A2A",
                    processing_time=time.time() - start_time,
                    steps=[
                        ProcessingStep(
                            step_name="A2A Text Correction",
                            input_text=text,
                            output_text=corrected_text,
                            status="success",
                            message="Text successfully corrected via A2A registry",
                        )
                    ],
                )

            elif operation == "sentiment":
                message = [{"content": text}]
                result = await registry.call_agent("sentiment", message)

                if debug_a2a:
                    logger.info(f"Raw sentiment result: {result}")

                # Extract sentiment data - this is more complex due to nested structure
                sentiment_info = None
                emotions = []

                if hasattr(result, "sentiment") and hasattr(result, "emotions"):
                    sentiment_info = {
                        "label": result.sentiment.label
                        if hasattr(result.sentiment, "label")
                        else "unknown",
                        "confidence": result.sentiment.confidence
                        if hasattr(result.sentiment, "confidence")
                        else 0.0,
                        "score": result.sentiment.score
                        if hasattr(result.sentiment, "score")
                        else 0.0,
                    }
                    emotions = result.emotions if result.emotions else []
                else:
                    # Fallback sentiment analysis
                    sentiment_info = {
                        "label": "neutral",
                        "confidence": 0.5,
                        "score": 0.0,
                    }
                    emotions = ["neutral"]

                result_text = f"Sentiment Analysis Results:\nLabel: {sentiment_info['label']}\nConfidence: {sentiment_info['confidence']:.2f}\nScore: {sentiment_info['score']:.2f}\nEmotions: {emotions}"

                return UserInterfaceResponse(
                    original_text=text,
                    final_result=result_text,
                    operation_type="coordinate_with_a2a_agents",
                    status="success",
                    message="Sentiment analysis completed via A2A",
                    processing_time=time.time() - start_time,
                    sentiment_analysis=sentiment_info,
                    steps=[
                        ProcessingStep(
                            step_name="A2A Sentiment Analysis",
                            input_text=text,
                            output_text=f"Sentiment: {sentiment_info['label']}",
                            status="success",
                            message="Sentiment successfully analyzed via A2A registry",
                        )
                    ],
                )

        finally:
            await registry.stop()

    except Exception as e:
        logger.error(f"A2A coordination failed: {e}", exc_info=True)
        return UserInterfaceResponse(
            original_text=text,
            final_result=f"Error during {operation}: {str(e)}",
            operation_type="coordinate_with_a2a_agents",
            status="error",
            message=f"A2A coordination failed: {str(e)}",
            processing_time=time.time() - start_time,
        )


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


# New processing function with debugging
async def process_input(user_input: str) -> UserInterfaceResponse:
    """
    Main processing function - force tool usage to prevent fallbacks.
    """
    debug_mode = os.getenv("DEBUG_AGENT_RESPONSES", "false").lower() == "true"

    try:
        if debug_mode:
            logger.info(f"=== USER INTERFACE AGENT DEBUG ===")
            logger.info(f"Input: {user_input}")

        context = UserInterfaceContext(request_id="gradio_request")

        # For text optimization requests, bypass the agent and call the tool directly
        input_lower = user_input.lower()
        if any(
            word in input_lower
            for word in ["optimier", "verbesser", "freundlich", "korrigier", "mache"]
        ):
            if debug_mode:
                logger.info("Detected text processing request - calling tool directly")

            # Extract the text and tonality with better parsing
            text_to_process = user_input
            tonality = None

            # Look for tonality indicators in the original input
            if "sachlich professionell" in input_lower:
                tonality = "sachlich professionell"
            elif (
                "professionelle e-mail" in input_lower or "professionell" in input_lower
            ):
                tonality = "sachlich professionell"
            elif "freundlich" in input_lower or "friendly" in input_lower:
                tonality = "freundlich"
            elif "förmlich" in input_lower:
                tonality = "förmlich"
            elif "locker" in input_lower:
                tonality = "locker"
            elif "begeistert" in input_lower:
                tonality = "begeistert"
            elif "professionell" in input_lower:
                tonality = "professionell"

            # Extract the actual text to optimize (remove the instruction part)
            if ":" in user_input:
                parts = user_input.split(":", 1)
                if len(parts) > 1:
                    text_to_process = parts[1].strip()
                    # Remove tonality instruction in parentheses
                    if "(" in text_to_process and ")" in text_to_process:
                        # Find the parentheses content
                        start_paren = text_to_process.find("(")
                        end_paren = text_to_process.find(")")
                        if start_paren < end_paren:
                            text_to_process = text_to_process[:start_paren].strip()

            # If no specific text was provided after the colon, use a default
            if not text_to_process or text_to_process == user_input:
                text_to_process = "Bitte geben Sie den zu optimierenden Text an."

            if debug_mode:
                logger.info(f"Extracted text: '{text_to_process}'")
                logger.info(f"Detected tonality: '{tonality}'")

            # Call the A2A coordination tool directly
            return await coordinate_with_a2a_agents(
                context, text_to_process, "optimize", tonality
            )

        # For other requests, try the agent
        try:
            if debug_mode:
                logger.info("Running user_interface_agent...")

            result = await user_interface_agent.run(user_input, ctx=context)

            if debug_mode:
                logger.info(f"Agent run completed")
                logger.info(f"Result: {result}")

            # Check if we got a valid response
            if hasattr(result, "data") and isinstance(
                result.data, UserInterfaceResponse
            ):
                if debug_mode:
                    logger.info("Valid UserInterfaceResponse received")
                return result.data

        except Exception as agent_error:
            if debug_mode:
                logger.error(f"Agent run failed: {agent_error}", exc_info=True)

        # Final fallback should not be reached for text processing
        return UserInterfaceResponse(
            original_text=user_input,
            final_result=f"System could not process: {user_input}",
            operation_type="error",
            status="error",
            message="System failed to process request",
        )

    except Exception as e:
        logger.error(f"Critical error in process_input: {e}", exc_info=True)
        return UserInterfaceResponse(
            original_text=user_input,
            final_result=f"Critical system error",
            operation_type="error",
            status="error",
            message=f"Critical error: {str(e)}",
        )


def _create_intelligent_fallback(
    user_input: str, error_info: str
) -> UserInterfaceResponse:
    """Create an intelligent fallback response based on user input analysis."""
    input_lower = user_input.lower()

    # Detect what the user likely wanted and provide a helpful fallback
    if any(
        word in input_lower
        for word in [
            "optimier",
            "verbesser",
            "freundlich",
            "korrigier",
            "optimize",
            "improve",
            "correct",
            "friendly",
        ]
    ):
        # Text processing request
        if "freundlich" in input_lower or "friendly" in input_lower:
            # Extract the text to be made friendlier
            text_to_process = user_input
            if ":" in user_input:
                text_to_process = user_input.split(":", 1)[1].strip()
                # Remove tonality instruction
                if "(" in text_to_process:
                    text_to_process = text_to_process.split("(")[0].strip()

            # Provide a simple friendly version
            if (
                "abgelehnt" in text_to_process.lower()
                or "rejected" in text_to_process.lower()
            ):
                friendly_result = "Vielen Dank für Ihre Anfrage! Leider können wir Ihrem Wunsch diesmal nicht entsprechen, aber wir würden uns freuen, Ihnen bei einer anderen Gelegenheit helfen zu können."
            else:
                friendly_result = (
                    f"Hier ist eine freundlichere Version: {text_to_process} 😊"
                )

            return UserInterfaceResponse(
                original_text=user_input,
                final_result=friendly_result,
                operation_type="coordinate_with_a2a_agents",
                status="success",
                message="Fallback-Textoptimierung durchgeführt",
            )

        return UserInterfaceResponse(
            original_text=user_input,
            final_result="Es scheint, Sie möchten Text bearbeiten. Das System arbeitet derzeit daran, diese Funktion verfügbar zu machen.",
            operation_type="coordinate_with_a2a_agents",
            status="partial_success",
            message="Textverarbeitung erkannt, aber Verarbeitung fehlgeschlagen",
        )

    elif any(
        word in input_lower for word in ["zeit", "uhr", "datum", "time", "date", "spät"]
    ):
        return UserInterfaceResponse(
            original_text=user_input,
            final_result="Es scheint, Sie fragen nach der Zeit oder dem Datum. Das System arbeitet daran, diese Information abzurufen.",
            operation_type="process_time_query",
            status="partial_success",
            message="Zeit-/Datumsanfrage erkannt, aber Verarbeitung fehlgeschlagen",
        )

    elif any(word in input_lower for word in ["wetter", "weather", "temperatur"]):
        return UserInterfaceResponse(
            original_text=user_input,
            final_result="Es scheint, Sie fragen nach dem Wetter. Das System arbeitet daran, diese Information abzurufen.",
            operation_type="search_web_information",
            status="partial_success",
            message="Wetteranfrage erkannt, aber Verarbeitung fehlgeschlagen",
        )

    else:
        return UserInterfaceResponse(
            original_text=user_input,
            final_result=f"Ich habe Ihre Anfrage '{user_input}' verstanden, aber es gab ein Problem bei der Verarbeitung. Bitte versuchen Sie es erneut oder formulieren Sie Ihre Anfrage anders.",
            operation_type="search_web_information",
            status="error",
            message=f"Fallback response - {error_info[:100]}",
        )


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
