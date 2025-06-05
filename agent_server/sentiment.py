from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP
from typing import Optional, Dict, Any, List
import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path="../../.env")


def _get_mcp_server_url() -> str:
    """Get MCP server URL from environment variables."""
    host = os.getenv("SERVER_HOST", "localhost")
    port = os.getenv("SERVER_PORT", "8000")
    scheme = os.getenv("SERVER_SCHEME", "http")
    return f"{scheme}://{host}:{port}"


class SentimentRequest(BaseModel):
    text: str
    language: Optional[str] = "en"
    detailed: Optional[bool] = False


class SentimentScore(BaseModel):
    label: str  # positive, negative, neutral
    confidence: float
    score: float  # -1 to 1


class SentimentResponse(BaseModel):
    sentiment: SentimentScore
    emotions: Optional[List[Dict[str, float]]] = None
    status: str
    message: str


class A2AMessage(BaseModel):
    agent_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float


sentiment_agent = Agent(
    model="openai:gpt-4",
    result_type=SentimentResponse,
    system_prompt="""You are a sentiment analysis agent.
    Analyze text sentiment and return structured sentiment data.
    Provide sentiment label (positive/negative/neutral), confidence score, and detailed analysis.""",
)


async def analyze_sentiment_direct(request: SentimentRequest) -> SentimentResponse:
    """Analyze sentiment of the provided text."""
    try:
        # Simulate sentiment analysis processing
        await asyncio.sleep(0.3)

        # Optionally anonymize text before processing if it contains sensitive info
        text_to_analyze = request.text
        if len(request.text) > 100:  # Only for longer texts
            try:
                mcp_server = MCPServerHTTP(_get_mcp_server_url())
                try:
                    anon_result = await mcp_server.call_tool(
                        "anonymize_text", {"text": request.text}
                    )
                    if (
                        "anonymized_text" in anon_result
                        and anon_result["anonymized_text"]
                    ):
                        text_to_analyze = anon_result["anonymized_text"]
                        logging.info("Text anonymized before sentiment analysis")
                except Exception as e:
                    logging.warning(f"Anonymization failed, using original text: {e}")
                finally:
                    await mcp_server.close()
            except Exception as e:
                logging.warning(f"MCP anonymization setup failed: {e}")

        # Simple sentiment analysis simulation
        text_lower = text_to_analyze.lower()
        positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "love",
            "like",
            "fantastic",
            "awesome",
            "brilliant",
            "perfect",
            "outstanding",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "hate",
            "dislike",
            "horrible",
            "worst",
            "schrott",
            "verbuggt",
            "schlecht",
            "furchtbar",  # German negative words
            "katastrophal",
            "grauenhaft",
            "miserabel",
            "enttäuschend",
            "ärgerlich",
            "frustrierend",
            "unbrauchbar",
            "mangelhaft",
            "fehlerhaft",
        ]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            label = "positive"
            score = min(0.8, 0.5 + (pos_count - neg_count) * 0.1)
        elif neg_count > pos_count:
            label = "negative"
            score = max(-0.8, -0.5 - (neg_count - pos_count) * 0.1)
        else:
            label = "neutral"
            score = 0.0

        confidence = min(0.95, 0.6 + abs(score) * 0.4)

        sentiment_score = SentimentScore(
            label=label, confidence=confidence, score=score
        )

        emotions = None
        if request.detailed:
            emotions = [
                {"joy": max(0, score * 0.8)},
                {"anger": max(0, -score * 0.6)},
                {"fear": max(0, -score * 0.4)},
                {"surprise": abs(score) * 0.3},
            ]

        return SentimentResponse(
            sentiment=sentiment_score,
            emotions=emotions,
            status="success",
            message=f"Analyzed sentiment for {len(request.text)} characters",
        )

    except Exception as e:
        return SentimentResponse(
            sentiment=SentimentScore(label="neutral", confidence=0.0, score=0.0),
            status="error",
            message=f"Sentiment analysis failed: {str(e)}",
        )


class SentimentA2AAgent:
    def __init__(self, agent_id: str = "sentiment_agent"):
        self.agent_id = agent_id
        self.agent = sentiment_agent

    async def process_a2a_message(self, message: A2AMessage) -> A2AMessage:
        """Process incoming a2a messages."""
        try:
            if message.message_type == "sentiment_request":
                request = SentimentRequest(**message.payload)
                response = await self.agent.run(
                    f"Analyze sentiment of this text: {request.text}"
                )

                return A2AMessage(
                    agent_id=self.agent_id,
                    message_type="sentiment_response",
                    payload=response.data.model_dump(),
                    timestamp=asyncio.get_event_loop().time(),
                )
            else:
                raise ValueError(f"Unknown message type: {message.message_type}")

        except Exception as e:
            return A2AMessage(
                agent_id=self.agent_id,
                message_type="error",
                payload={"error": str(e)},
                timestamp=asyncio.get_event_loop().time(),
            )

    async def analyze_text_sentiment(
        self, text: str, detailed: bool = False
    ) -> SentimentResponse:
        """Analyze sentiment directly."""
        request = SentimentRequest(text=text, detailed=detailed)
        return await analyze_sentiment_direct(request)


# A2A server function for sentiment analysis
async def sentiment_a2a_function(messages: list) -> SentimentResponse:
    """A2A endpoint for sentiment analysis functionality."""
    if not messages:
        return SentimentResponse(
            sentiment=SentimentScore(label="neutral", confidence=0.0, score=0.0),
            status="error",
            message="No messages provided",
        )

    # Extract text from the last user message
    last_message = messages[-1]
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        text = last_message.content
    else:
        text = str(last_message)

    request = SentimentRequest(text=text, detailed=True)
    return await analyze_sentiment_direct(request)
