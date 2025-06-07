import os
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional

import dotenv
import pydantic
import pydantic_ai
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (consider loading only once if used elsewhere)
# It's often better to load this at the application entry point.
dotenv.load_dotenv(override=True, dotenv_path="../../.env")


@dataclass
class RefactorDependencies:
    """Dependencies for the refactor agent (currently empty)."""

    pass


class RefactorResult(BaseModel):
    """Expected result structure from the refactor agent."""

    query: str = Field(description="Query for the retrieve agent")


# Initialize the model with OpenAI provider for Ollama compatibility
def _create_query_refactor_agent():
    """Create the query refactor agent with proper Ollama configuration."""
    try:
        llm_api_key = os.getenv("API_KEY", "ollama")
        llm_endpoint = os.getenv("BASE_URL", "http://localhost:11434/v1")
        llm_model_name = os.getenv("QUERY_REF_MODEL", "qwen2.5:latest")

        provider = OpenAIProvider(base_url=llm_endpoint, api_key=llm_api_key)
        model = OpenAIModel(provider=provider, model_name=llm_model_name)

        return Agent(
            model=model,
            result_type=RefactorResult,
            system_prompt="""Du bist ein Assistent zur Optimierung von Nutzeranfragen für Large Language Models (LLMs).
Deine Aufgabe ist es, die folgende Nutzeranfrage so zu überarbeiten, dass sie maximal klar, präzise und effizient für ein LLM ist.

**Anweisungen für die Überarbeitung:**
1.  **Klarheit und Präzision:** Formuliere die Anfrage unmissverständlich. Eliminiere Mehrdeutigkeiten und präzisiere vage Ausdrücke.
2.  **Effizienz:** Entferne alle unnötigen Füllwörter, Höflichkeitsfloskeln oder redundante Informationen. Die Anfrage soll direkt auf den Punkt kommen.
3.  **Struktur (optional):** Falls die Anfrage komplex ist, gliedere sie in logische Unterschritte, wenn dies die Verarbeitbarkeit für das LLM verbessert.
4.  **Informationsgehalt:** Stelle sicher, dass alle für die Beantwortung der Anfrage notwendigen Informationen der Originalanfrage erhalten bleiben.
5.  **Sinnbewahrung:** Der Kern und die Absicht der ursprünglichen Nutzeranfrage dürfen nicht verändert werden.

**Formatierungsanweisung für deine Ausgabe:**
Deine Antwort MUSS AUSSCHLIESSLICH die überarbeitete Nutzeranfrage als reinen Text enthalten.
Gib KEINERLEI zusätzliche Informationen aus, wie z.B.:
- Keine Einleitungssätze (z.B. "Hier ist die überarbeitete Anfrage:")
- Keine Erklärungen zu deinen Änderungen
- Keine Kommentare
- Keine Zusammenfassungen
- Keine Entschuldigungen

**Beispiel:**
Wenn die Nutzeranfrage lautet: "Erzähl mir doch mal was über Katzen, vielleicht so ihre Geschichte und so?"
Könnte deine Ausgabe (die überarbeitete Anfrage) sein: "Gib eine Zusammenfassung der Geschichte der Domestizierung von Katzen und ihrer Entwicklung."

Nutzeranfrage: {user_query}
Ausgabe (nur die überarbeitete Anfrage):""",
        )
    except Exception as e:
        logger.error(f"Failed to initialize query refactor agent: {e}")
        raise


# Create the agent
query_refactor_agent = _create_query_refactor_agent()


async def refactor_query_direct(user_query: str) -> str:
    """
    Refactors the given user query using the configured agent.

    Args:
        user_query: The user query string to refactor.

    Returns:
        The refactored query string.
    """
    if not user_query:
        logger.warning("Received empty user query.")
        return ""

    logger.info(f"Refactoring query: '{user_query}'")
    try:
        result = await query_refactor_agent.run(user_query)
        refactored_query = result.data.query
        logger.info(f"Refactored query: '{refactored_query}'")
        return refactored_query
    except Exception as e:
        logger.error(f"Error during agent run for query '{user_query}': {e}")
        return user_query  # Return original query if refactoring fails


class QueryRefactorAgent:
    """
    A class to encapsulate the query refactoring logic using a Pydantic AI Agent.
    """

    def __init__(self):
        """
        Initializes the QueryRefactorAgent.
        """
        logger.info("Initializing QueryRefactorAgent...")
        self.agent = query_refactor_agent
        logger.info("QueryRefactorAgent initialized successfully.")

    async def refactor_query(self, user_query: str) -> str:
        """
        Refactors the given user query using the configured agent.
        """
        return await refactor_query_direct(user_query)


# A2A server function for query refactoring (moved outside the class)
async def query_ref_a2a_function(messages: list) -> RefactorResult:
    """A2A endpoint for query refactoring functionality."""
    if not messages:
        return RefactorResult(query="")

    # Extract text from the last user message
    last_message = messages[-1]
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        text = last_message.content
    else:
        text = str(last_message)

    try:
        # Use the direct function instead of creating a new instance
        refactored_query = await refactor_query_direct(text)
        return RefactorResult(query=refactored_query)

    except Exception as e:
        logger.error(f"Query refactoring failed: {e}")
        # Return original text if refactoring fails
        return RefactorResult(query=text)


# --- Example Usage ---
async def main():
    """Main function to demonstrate QueryRefactorAgent usage."""
    print(f"Pydantic version {pydantic.__version__}")
    print(f"Pydantic AI version {pydantic_ai.__version__}")

    try:
        # Example query
        user_query = "Was soll gem der Bundesregierung im Datenschutz erreicht werden?"

        # Refactor the query
        refactored_query = await refactor_query_direct(user_query)

        print("\n--- Refactoring Result ---")
        print(f"Original Query: {user_query}")
        print(f"Refactored Query: {refactored_query}")

        # Example with an empty query
        print("\n--- Testing Empty Query ---")
        empty_result = await refactor_query_direct("")
        print(f"Result for empty query: '{empty_result}'")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
