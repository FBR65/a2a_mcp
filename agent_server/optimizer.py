import os
import logging
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.mcp import MCPServerHTTP
from openai import AsyncClient
from dotenv import load_dotenv
from typing import Optional

load_dotenv(override=True, dotenv_path="../../.env")


class OptimizerRequest(BaseModel):
    text: str
    tonalitaet: Optional[str] = None
    model: str = None


class OptimizerResponse(BaseModel):
    optimized_text: str
    status: str
    message: str


# Initialize the optimizer agent
def _get_mcp_server_url() -> str:
    """Get MCP server URL from environment variables."""
    host = os.getenv("SERVER_HOST", "localhost")
    port = os.getenv("SERVER_PORT", "8000")
    scheme = os.getenv("SERVER_SCHEME", "http")
    return f"{scheme}://{host}:{port}"


optimizer_agent = Agent(
    model=os.getenv("TEXT_OPT_MODEL", "gpt-4"),
    result_type=OptimizerResponse,
    system_prompt="""Sie sind ein Textoptimierer, der darauf spezialisiert ist, Texte zu verbessern und zu verfeinern. 
Ihre Aufgabe ist es, den bereitgestellten Text zu analysieren und eine optimierte Version auszugeben. 
Konzentrieren Sie sich bei der Optimierung auf Aspekte wie Klarheit, Prägnanz, Wirkung, Lesefluss und Kohärenz. 
Ziel ist es, den Text für den Leser verständlicher, ansprechender und effektiver zu gestalten. 
Geben Sie nur den optimierten Text aus.""",
)


async def run_optimizer(request: OptimizerRequest) -> OptimizerResponse:
    """Run text optimization on the provided text."""
    try:
        # MCP integration for enhanced capabilities
        mcp_server = MCPServerHTTP(_get_mcp_server_url())
        enhanced_context = ""

        try:
            # Get current time for context
            time_info = await mcp_server.call_tool("get_current_time", {})
            if "current_time_utc" in time_info:
                logging.info(
                    f"Optimizer processing started at: {time_info['current_time_utc']}"
                )

            # If the text mentions specific topics, enhance with web search
            if len(request.text) > 200:  # Only for longer texts
                try:
                    # Extract potential search terms (basic implementation)
                    words = request.text.split()
                    potential_topics = [
                        word for word in words if len(word) > 6 and word.isalpha()
                    ]
                    if potential_topics:
                        search_query = " ".join(
                            potential_topics[:3]
                        )  # Use first 3 long words
                        search_results = await mcp_server.call_tool(
                            "duckduckgo_search",
                            {"query": search_query, "max_results": 2},
                        )
                        if "results" in search_results and search_results["results"]:
                            enhanced_context = f"Current context from web: {search_results['results'][0].get('snippet', '')}"
                            logging.info("Enhanced optimizer with web context")
                except Exception as e:
                    logging.warning(f"Web enhancement failed: {e}")
        except Exception as e:
            logging.warning(f"MCP enhancement failed: {e}")
        finally:
            await mcp_server.close()

        # Configure client if needed
        if os.getenv("BASE_URL"):
            client = AsyncClient(
                api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL")
            )
            # Update agent's model client
            optimizer_agent._model.client = client

        # Build prompt with tonality if provided
        tonality_instruction = ""
        if request.tonalitaet:
            tonality_instruction = (
                f"Der Text soll in einem {request.tonalitaet} Ton verfasst sein. "
            )

        context_instruction = ""
        if enhanced_context:
            context_instruction = (
                f"Berücksichtige diesen aktuellen Kontext: {enhanced_context[:200]}... "
            )

        prompt = f"{tonality_instruction}{context_instruction}Optimieren Sie folgenden Text: {request.text}"

        result = await optimizer_agent.run(prompt)

        return OptimizerResponse(
            optimized_text=result.data.optimized_text,
            status="success",
            message=f"Successfully optimized {len(request.text)} characters",
        )
    except Exception as e:
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
        text = last_message.content
    else:
        text = str(last_message)

    request = OptimizerRequest(text=text)
    return await run_optimizer(request)
