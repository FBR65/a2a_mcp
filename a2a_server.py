import asyncio
import logging

# from a2a_protocol import a2a_registry  # Comment out until a2a_protocol is implemented
from agent_server.lektor import lektor_a2a_function
from agent_server.sentiment import sentiment_a2a_function
from agent_server.optimizer import optimizer_a2a_function
from agent_server.query_ref import query_ref_a2a_function
from agent_server.user_interface import user_interface_a2a_function

logging.basicConfig(level=logging.INFO)


# Simple registry implementation for testing
class SimpleA2ARegistry:
    def __init__(self):
        self.agents = {}

    def register_a2a_agent(self, name: str, function):
        self.agents[name] = function

    async def call_agent(self, name: str, messages):
        if name not in self.agents:
            raise ValueError(f"Agent {name} not found")
        return await self.agents[name](messages)

    async def start(self):
        pass

    async def stop(self):
        pass


async def setup_a2a_server():
    """Setup and start the a2a server with all agents."""
    registry = SimpleA2ARegistry()

    # Register all agents
    registry.register_a2a_agent("lektor", lektor_a2a_function)
    registry.register_a2a_agent("sentiment", sentiment_a2a_function)
    registry.register_a2a_agent("optimizer", optimizer_a2a_function)
    registry.register_a2a_agent("query_ref", query_ref_a2a_function)
    registry.register_a2a_agent("user_interface", user_interface_a2a_function)

    # Start the registry
    await registry.start()

    return registry


async def run_text_processing_workflow(text: str, tonality: str = None):
    """Run the complete text processing workflow using a2a agents."""

    # Setup the server
    registry = await setup_a2a_server()

    try:
        # Step 1: Refactor query for better processing
        refactored_result = await registry.call_agent("query_ref", text)
        print(f"Refactored: {refactored_result.query}")

        # Step 2: Optimize text
        optimize_text = (
            f"Optimize this text with tonality '{tonality}': {refactored_result.query}"
        )
        optimized_result = await registry.call_agent("optimizer", optimize_text)
        print(f"Optimized: {optimized_result.optimized_text}")

        # Step 3: Lektor correction
        corrected_result = await registry.call_agent(
            "lektor", optimized_result.optimized_text
        )
        print(f"Corrected: {corrected_result.corrected_text}")

        # Step 4: Sentiment analysis
        sentiment_result = await registry.call_agent(
            "sentiment", corrected_result.corrected_text
        )
        print(
            f"Sentiment: {sentiment_result.sentiment.label} (confidence: {sentiment_result.sentiment.confidence})"
        )

        return {
            "original": text,
            "refactored": refactored_result.query,
            "optimized": optimized_result.optimized_text,
            "corrected": corrected_result.corrected_text,
            "sentiment": sentiment_result.sentiment.label,
        }

    finally:
        await registry.stop()


async def run_query_refactor_demo(query: str):
    """Demonstrate query refactoring functionality."""
    registry = await setup_a2a_server()

    try:
        result = await registry.call_agent("query_ref", query)

        print("\n--- Query Refactoring Demo ---")
        print(f"Original: {query}")
        print(f"Refactored: {result.query}")

        return result.query

    finally:
        await registry.stop()


async def run_user_interface_demo():
    """Demonstrate user interface agent functionality."""
    from agent_server.user_interface import user_interface_agent

    # Test different operations
    test_text = "Also echt mal, dieses Produkt ist der letzte Schrott, total verbuggt und wer das kauft ist doch selber schuld, oder?"

    # Test sentiment only
    from agent_server.user_interface import UserRequest

    request = UserRequest(
        text=test_text,
        operation="sentiment_only",
        detailed=True,
    )

    result = await user_interface_agent.process_user_request(request)

    print("\n--- User Interface Demo ---")
    print(f"Operation: {result.operation_type}")
    print(f"Result: {result.final_result}")
    print(f"Sentiment: {result.sentiment_analysis}")


async def main():
    """Main function demonstrating a2a agent communication."""

    # Demo 1: Query refactoring
    query = "Was soll gem der Bundesregierung im Datenschutz erreicht werden?"
    await run_query_refactor_demo(query)

    # Demo 2: Full text processing workflow
    text = "Also echt mal, dieses Produkt ist der letzte Schrott, total verbuggt und wer das kauft ist doch selber schuld, oder?"
    result = await run_text_processing_workflow(text, "sachlich professionell")

    print("\n--- Final Results ---")
    for key, value in result.items():
        print(f"{key}: {value}")

    # Demo 3: User interface agent
    await run_user_interface_demo()


if __name__ == "__main__":
    asyncio.run(main())
