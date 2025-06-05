import os
import logging
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.mcp import MCPServerHTTP
from openai import AsyncClient
from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path="../../.env")


class LektorRequest(BaseModel):
    text: str
    model: str = None


class LektorResponse(BaseModel):
    corrected_text: str
    status: str
    message: str


# Initialize the lektor agent with MCP integration
def _get_mcp_server_url() -> str:
    """Get MCP server URL from environment variables."""
    host = os.getenv("SERVER_HOST", "localhost")
    port = os.getenv("SERVER_PORT", "8000")
    scheme = os.getenv("SERVER_SCHEME", "http")
    return f"{scheme}://{host}:{port}"


lektor_agent = Agent(
    model=os.getenv("TEXT_OPT_MODEL", "gpt-4"),
    result_type=LektorResponse,
    system_prompt="""Ihre Aufgabe ist es, den vom Benutzer übergebenen Text direkt zu korrigieren.
Konzentrieren Sie sich auf die Korrektur von Grammatik, Rechtschreibung, Zeichensetzung, Syntax und Lesefluss des Originaltextes.
Das Ergebnis Ihrer Arbeit muss der korrigierte Originaltext sein.

**ANWEISUNG FÜR DIE AUSGABE:**
Geben Sie NUR den vollständig korrigierten Text zurück.
*   KEINE einleitenden Sätze oder Phrasen.
*   KEINE Erklärungen zu Ihren Korrekturen.
*   KEINE Kommentare über den Text oder Ihre Arbeit.
*   KEINE Zitate des Originaltextes.
*   Antworten Sie ausschließlich mit dem direkt korrigierten Text.""",
)


async def run_lektor(request: LektorRequest) -> LektorResponse:
    """Run lektor correction on the provided text."""
    try:
        # Get current time for timestamping using MCP
        mcp_server = MCPServerHTTP(_get_mcp_server_url())
        try:
            time_info = await mcp_server.call_tool("get_current_time", {})
            if "current_time_utc" in time_info:
                logging.info(
                    f"Lektor processing started at: {time_info['current_time_utc']}"
                )
        except Exception as e:
            logging.warning(f"Could not get current time from MCP: {e}")
        finally:
            await mcp_server.close()

        # Configure client if needed
        if os.getenv("BASE_URL"):
            client = AsyncClient(
                api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL")
            )
            # Update agent's model client
            lektor_agent._model.client = client

        result = await lektor_agent.run(request.text)

        return LektorResponse(
            corrected_text=result.data.corrected_text,
            status="success",
            message=f"Successfully corrected {len(request.text)} characters",
        )
    except Exception as e:
        return LektorResponse(
            corrected_text=request.text,
            status="error",
            message=f"Lektor correction failed: {str(e)}",
        )


# A2A server function for lektor
async def lektor_a2a_function(messages: list[ModelMessage]) -> LektorResponse:
    """A2A endpoint for lektor functionality."""
    if not messages:
        return LektorResponse(
            corrected_text="", status="error", message="No messages provided"
        )

    # Extract text from the last user message
    last_message = messages[-1]
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        text = last_message.content
    else:
        text = str(last_message)

    request = LektorRequest(text=text)
    return await run_lektor(request)
