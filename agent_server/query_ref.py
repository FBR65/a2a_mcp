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


class QueryRefactorAgent:
    """
    A class to encapsulate the query refactoring logic using a Pydantic AI Agent.
    """

    DEFAULT_SYSTEM_PROMPT = """Du bist ein Assistent zur Optimierung von Nutzeranfragen für Large Language Models (LLMs).
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

Nutzeranfrage: {{user_query}}
Ausgabe (nur die überarbeitete Anfrage):"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        """
        Initializes the QueryRefactorAgent.

        Args:
            api_key: OpenAI API key. Defaults to os.getenv("API_KEY").
            base_url: OpenAI base URL. Defaults to os.getenv("BASE_URL").
            model_name: OpenAI model name. Defaults to os.getenv("MODEL_NAME").
            system_prompt: The system prompt for the agent.
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.model_name = model_name or os.getenv("MODEL_NAME")
        self.system_prompt = system_prompt

        if not self.api_key:
            raise ValueError("API_KEY environment variable or argument is required.")
        if not self.base_url:
            raise ValueError("BASE_URL environment variable or argument is required.")
        if not self.model_name:
            raise ValueError("MODEL_NAME environment variable or argument is required.")

        logger.info("Initializing QueryRefactorAgent...")
        logger.info(f"Using Model: {self.model_name}, Base URL: {self.base_url}")

        try:
            provider = OpenAIProvider(base_url=self.base_url, api_key=self.api_key)
            model = OpenAIModel(
                provider=provider,
                model_name=self.model_name,
            )
            self.agent = Agent(
                name="Refactor Agent",
                model=model,
                deps_type=RefactorDependencies,
                result_type=RefactorResult,
                system_prompt=self.system_prompt,
            )
            logger.info("QueryRefactorAgent initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Pydantic AI components: {e}")
            raise

    async def refactor_query(self, user_query: str) -> str:
        """
        Refactors the given user query using the configured agent.

        Args:
            user_query: The user query string to refactor.

        Returns:
            The refactored query string.

        Raises:
            ValueError: If the agent fails to produce a valid result.
            Exception: For other underlying errors during agent execution.
        """
        if not user_query:
            logger.warning("Received empty user query.")
            return ""

        logger.info(f"Refactoring query: '{user_query}'")
        try:
            run_result = await self.agent.run(user_query=user_query)
            if isinstance(run_result.output, RefactorResult):
                refactored_query = run_result.output.query
                logger.info(f"Refactored query: '{refactored_query}'")
                return refactored_query
            else:
                # This case might indicate an issue with pydantic-ai or the LLM response
                logger.error(
                    f"Agent returned unexpected output type: {type(run_result.output)}"
                )
                logger.error(f"Raw output: {run_result.raw_output}")
                raise ValueError(
                    "Agent did not return the expected RefactorResult structure."
                )
        except ValidationError as e:
            logger.error(f"Validation error processing agent response: {e}")
            logger.error(
                f"Raw output that caused validation error: {getattr(e, 'raw_output', 'N/A')}"
            )  # Attempt to log raw output if available
            raise ValueError(f"Failed to validate agent response: {e}") from e
        except Exception as e:
            logger.error(f"Error during agent run for query '{user_query}': {e}")
            raise  # Re-raise the original exception


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
        # Initialize the agent with environment variables
        refactor_agent_instance = QueryRefactorAgent()

        # Refactor the query
        refactored_query = await refactor_agent_instance.refactor_query(text)

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
        # Initialize the agent (reads from .env by default)
        refactor_agent_instance = QueryRefactorAgent()

        # Example query
        user_query = "Was soll gem der Bundesregierung im Datenschutz erreicht werden?"

        # Refactor the query
        refactored_query = await refactor_agent_instance.refactor_query(user_query)

        print("\n--- Refactoring Result ---")
        print(f"Original Query: {user_query}")
        print(f"Refactored Query: {refactored_query}")

        # Example with an empty query
        print("\n--- Testing Empty Query ---")
        empty_result = await refactor_agent_instance.refactor_query("")
        print(f"Result for empty query: '{empty_result}'")

    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
