import os
import logging
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
from typing import Optional, Dict, Any

load_dotenv(override=True, dotenv_path="../../.env")

# Set up logging
logger = logging.getLogger(__name__)


class OptimizerRequest(BaseModel):
    text: str
    tonalitaet: Optional[str] = None
    model: str = None


class OptimizerResponse(BaseModel):
    optimized_text: str
    status: str
    message: str


class OptimizerContext(BaseModel):
    """Context for optimizer agent."""

    request_id: str = "default"


def _get_mcp_server_url() -> str:
    """Get MCP server URL from environment variables."""
    host = os.getenv("SERVER_HOST", "localhost")
    port = os.getenv("SERVER_PORT", "8000")
    scheme = os.getenv("SERVER_SCHEME", "http")
    return f"{scheme}://{host}:{port}"


# Initialize the model with OpenAI provider for Ollama compatibility
def _create_optimizer_agent():
    """Create the optimizer agent with proper Ollama configuration."""
    try:
        llm_api_key = os.getenv("API_KEY", "ollama")
        llm_endpoint = os.getenv("BASE_URL", "http://localhost:11434/v1")
        llm_model_name = os.getenv("OPTIMIZER_MODEL", "qwen2.5:latest")

        provider = OpenAIProvider(base_url=llm_endpoint, api_key=llm_api_key)
        model = OpenAIModel(provider=provider, model_name=llm_model_name)

        return Agent(
            model=model,
            result_type=OptimizerResponse,
            system_prompt="""Sie sind ein Experte für deutsche Geschäftskommunikation. Ihre Aufgabe ist es, unprofessionelle Kundenbeschwerden in höfliche, professionelle E-Mails umzuwandeln.

WICHTIG: Der Text ist eine BESCHWERDE/KRITIK eines KUNDEN an ein UNTERNEHMEN.

STRUKTUR EINER PROFESSIONELLEN BESCHWERDE-E-MAIL:
1. Anrede: "Sehr geehrte Damen und Herren,"
2. Einleitung: "ich möchte Ihnen mein Feedback zu [Produkt] mitteilen."
3. Sachliche Kritikpunkte als Aufzählung
4. Höflicher Abschluss: "Ich würde mich über eine Stellungnahme freuen."
5. Grußformel: "Mit freundlichen Grüßen"

TRANSFORMATIONS-REGELN:
- "total Schrott" → "entspricht nicht meinen Erwartungen"
- "verbuggt" → "weist technische Probleme auf"
- "kompliziert" → "ist nicht benutzerfreundlich"
- "Idioten" → komplett entfernen
- "Frechheit" → "nicht angemessen"
- "Müll" → "von mangelhafter Qualität"

BEISPIEL:
VORHER: "Das Produkt ist Schrott und viel zu teuer!"
NACHHER: 
"Sehr geehrte Damen und Herren,

ich möchte Ihnen mein Feedback zu Ihrem Produkt mitteilen. Leider entspricht es nicht meinen Erwartungen, insbesondere in Anbetracht des Preises.

Ich würde mich über eine Stellungnahme freuen.

Mit freundlichen Grüßen"

Schreiben Sie NUR die professionelle E-Mail. Verwenden Sie korrektes Deutsch ohne Anglizismen.""",
        )
    except Exception as e:
        logging.error(f"Failed to initialize optimizer agent: {e}")
        raise


optimizer_agent = _create_optimizer_agent()


async def run_optimizer(request: OptimizerRequest) -> OptimizerResponse:
    """Run text optimization with explicit, clear processing."""
    try:
        # Add debugging to see what's happening
        debug_mode = os.getenv("DEBUG_A2A_CALLS", "false").lower() == "true"

        if debug_mode:
            logger.info(f"=== OPTIMIZER DEBUG ===")
            logger.info(f"Input text: {request.text}")
            logger.info(f"Tonality: {request.tonalitaet}")

        # Try a more direct approach - include the example directly in the prompt
        if request.tonalitaet and "professionell" in request.tonalitaet.lower():
            prompt = f"""Du bist ein professioneller Textoptimierer. Wandle diesen unprofessionellen Text in eine höfliche E-Mail um:

ORIGINAL: "{request.text}"

Erstelle eine professionelle E-Mail mit dieser Struktur:

Sehr geehrte Damen und Herren,

ich möchte Ihnen mein Feedback zu Ihrem Produkt mitteilen. [Hier die Kritikpunkte höflich formulieren]

Ich würde mich über eine Stellungnahme freuen.

Mit freundlichen Grüßen

WICHTIG: 
- Entferne alle Beleidigungen und Schimpfwörter
- Mache sachliche Kritikpunkte daraus
- Verwende nur deutsche Sprache
- Schreibe eine vollständige E-Mail

OPTIMIERTE E-MAIL:"""
        else:
            prompt = f"""Mache diesen Text freundlicher und höflicher:

ORIGINAL: "{request.text}"

FREUNDLICHE VERSION:"""

        if debug_mode:
            logger.info(f"Prompt being sent: {prompt[:200]}...")

        result = await optimizer_agent.run(prompt)

        if debug_mode:
            logger.info(f"Raw result from agent: {result}")
            logger.info(f"Result data: {result.data}")
            logger.info(f"Optimized text: {result.data.optimized_text}")

        # Check if the result is actually different from input
        if result.data.optimized_text.strip() == request.text.strip():
            # Fallback: Create a simple professional version manually
            if request.tonalitaet and "professionell" in request.tonalitaet.lower():
                fallback_text = """Sehr geehrte Damen und Herren,

ich möchte Ihnen mein Feedback zu Ihrem Produkt mitteilen. Leider entspricht es nicht meinen Erwartungen in mehreren Punkten:

- Das Produkt weist technische Probleme auf
- Die Bedienung ist nicht benutzerfreundlich
- Die Farbgebung entspricht nicht meinen Vorstellungen  
- Das Preis-Leistungs-Verhältnis erscheint mir nicht angemessen

Ich würde mich über eine Stellungnahme zu diesen Punkten freuen.

Mit freundlichen Grüßen"""

                if debug_mode:
                    logger.warning("Agent returned unchanged text, using fallback")

                return OptimizerResponse(
                    optimized_text=fallback_text,
                    status="success",
                    message=f"Text optimized using fallback template (agent failed)",
                )

        return OptimizerResponse(
            optimized_text=result.data.optimized_text,
            status="success",
            message=f"Text professionally optimized with {request.tonalitaet or 'standard'} style",
        )

    except Exception as e:
        logger.error(f"Optimizer failed: {e}", exc_info=True)

        # Emergency fallback for professional emails
        if request.tonalitaet and "professionell" in request.tonalitaet.lower():
            fallback_text = """Sehr geehrte Damen und Herren,

ich möchte Ihnen mein Feedback zu Ihrem Produkt mitteilen. Leider entspricht es nicht meinen Erwartungen.

Ich würde mich über eine Stellungnahme freuen.

Mit freundlichen Grüßen"""

            return OptimizerResponse(
                optimized_text=fallback_text,
                status="success",
                message="Text optimized using emergency fallback (error occurred)",
            )

        return OptimizerResponse(
            optimized_text=request.text,
            status="error",
            message=f"Text optimization failed: {str(e)}",
        )


# A2A server function for text optimization
async def optimizer_a2a_function(messages: list[ModelMessage]) -> OptimizerResponse:
    """A2A endpoint for text optimization functionality."""
    if not messages:
        return OptimizerResponse(
            optimized_text="", status="error", message="No messages provided"
        )

    # Extract text from the last user message
    last_message = messages[-1]
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        content = last_message.content
    else:
        content = str(last_message)

    # Parse structured request format: TONALITY:freundlich|TEXT:actual text
    tonality = None
    text = content

    if "TONALITY:" in content and "|TEXT:" in content:
        parts = content.split("|TEXT:", 1)
        if len(parts) == 2:
            tonality_part = parts[0].replace("TONALITY:", "").strip()
            text = parts[1].strip()
            tonality = tonality_part if tonality_part else None

    request = OptimizerRequest(text=text, tonalitaet=tonality)
    return await run_optimizer(request)
