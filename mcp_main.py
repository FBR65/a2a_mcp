import os
import uvicorn
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi_mcp import FastApiMCP
from typing import Optional
from pydantic import BaseModel, Field

# --- Import Service Classes ---
from mcp_services.mcp_website.headless_browser import HeadlessBrowserExtractor
from mcp_services.mcp_time.ntp_time import NtpTime
from mcp_services.mcp_search.duck_search import (
    DuckDuckGoSearcher,
    DuckDuckGoSearchResults,
)
from mcp_services.mcp_anonymizer.anonymiz import Anonymizer
from mcp_services.mcp_fileconverter.file2pdf import PDFConverter  # Add this import

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Use a consistent logger name if desired
logger = logging.getLogger("mcp_server.main")  # Changed logger name

# --- Load Environment Variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.warning(
        f".env file not found at {dotenv_path}. Relying on system environment variables."
    )


# --- Instantiate Services (Error handling remains important) ---
headless_browser: Optional[HeadlessBrowserExtractor] = None
ntp_time: Optional[NtpTime] = None
duck_searcher: Optional[DuckDuckGoSearcher] = None
anonymizer: Optional[Anonymizer] = None
pdf_converter: Optional[PDFConverter] = None  # Add this line

try:
    logger.info("Initializing HeadlessBrowserExtractor...")
    headless_browser = HeadlessBrowserExtractor()

    logger.info("Initializing NtpTime...")
    ntp_time = NtpTime()

    logger.info("Initializing DuckDuckGoSearcher...")
    duck_searcher = DuckDuckGoSearcher()

    # Anonymizer Config
    logger.info("Initializing Anonymizer...")
    use_llm_anon = os.getenv("ANONYMIZER_USE_LLM", "False").lower() == "true"
    llm_endpoint_anon = os.getenv("ANONYMIZER_LLM_ENDPOINT")
    llm_api_key_anon = os.getenv("ANONYMIZER_LLM_API_KEY")
    llm_model_anon = os.getenv("ANONYMIZER_LLM_MODEL")
    if use_llm_anon and (not llm_endpoint_anon or not llm_api_key_anon):
        logger.warning(
            "Anonymizer LLM use enabled, but endpoint or API key missing. Disabling LLM."
        )
        use_llm_anon = False
    anonymizer = Anonymizer(
        use_llm=use_llm_anon,
        llm_endpoint_url=llm_endpoint_anon,
        llm_api_key=llm_api_key_anon,
        llm_model_name=llm_model_anon,
    )

    # PDF Converter initialization
    logger.info("Initializing PDFConverter...")
    pdf_converter = PDFConverter()

except ValueError as e:
    logger.error(f"Configuration error during service initialization: {e}")
except ImportError as e:
    logger.error(f"Import error: {e}. Dependencies might be missing.")
    raise SystemExit(f"Failed to import service component: {e}") from e
except Exception as e:
    logger.exception(f"Unexpected error during service initialization: {e}")


# --- Pydantic Models for API Endpoints ---


# Website Extractor
class ExtractTextRequest(BaseModel):
    url: str = Field(..., description="URL of the website to extract text from.")


class ExtractTextResponse(BaseModel):
    url: str
    text_content: Optional[str] = None
    error: Optional[str] = None


# NTP Time
class NtpTimeResponse(BaseModel):
    current_time_utc: Optional[str] = None
    error: Optional[str] = None


# DuckDuckGo Search
class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query.")
    max_results: int = Field(
        5, description="Maximum number of search results to return.", gt=0, le=20
    )


# Note: Using the models directly from the search service module
# class DuckDuckGoSearchResults(BaseModel): ...
# class DuckDuckGoSearchResult(BaseModel): ...


# Anonymizer
class AnonymizeRequest(BaseModel):
    text: str = Field(..., description="Text to be anonymized.")
    # Add other options if the Anonymizer class supports them


class AnonymizeResponse(BaseModel):
    anonymized_text: Optional[str] = None
    error: Optional[str] = None


# PDF Converter
class ConvertToPdfRequest(BaseModel):
    input_filepath: str = Field(
        ..., description="Path to the file that should be converted to PDF."
    )
    output_directory: Optional[str] = Field(
        None, description="Optional directory where the PDF should be saved."
    )
    output_filename: Optional[str] = Field(
        None, description="Optional filename for the output PDF (without extension)."
    )


class ConvertToPdfResponse(BaseModel):
    output_filepath: Optional[str] = None
    error: Optional[str] = None


# --- FastAPI Setup ---
app = FastAPI(
    title="MCP Server with Integrated Services",
    description="Exposes various backend services via FastAPI. MCP integration provides tool access.",
    version="1.0.0",
)


# --- Standard FastAPI Routes (Hidden from Docs by default) ---
@app.get("/", include_in_schema=False)
async def root():
    return PlainTextResponse("MCP Server is running.")


@app.get("/health", include_in_schema=False)
async def health_check():
    return JSONResponse({"status": "ok"})


# --- Service Endpoints (Hidden from Docs, Exposed via MCP) ---


# Helper functions to check service availability
def check_service(service_instance, service_name: str):
    if service_instance is None:
        logger.error(f"Attempted to use unavailable {service_name} service.")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail=f"{service_name} service is not configured or failed to initialize.",
        )


# Website Extractor Endpoint
@app.post(
    "/extract-text",
    response_model=ExtractTextResponse,
    summary="Extract Text Content from URL",
    operation_id="extract_website_text",
    include_in_schema=False,
)
async def extract_text_endpoint(request_data: ExtractTextRequest):
    """Extracts the main textual content from a given website URL."""
    check_service(headless_browser, "Headless Browser")
    logger.info(f"API: Received request to extract text from URL: {request_data.url}")
    try:
        # The HeadlessBrowserExtractor likely doesn't have an async extract_text method
        text = headless_browser.extract_text(request_data.url)
        return ExtractTextResponse(url=request_data.url, text_content=text)
    except Exception as e:
        logger.exception(f"Error extracting text from {request_data.url}: {e}")
        return ExtractTextResponse(url=request_data.url, error=str(e))


# NTP Time Endpoint
@app.get(
    "/current-time",
    response_model=NtpTimeResponse,
    summary="Get Current UTC Time from NTP",
    operation_id="get_current_time",  # MCP tool name
    include_in_schema=False,
)
async def current_time_endpoint():
    """Gets the current accurate time from an NTP server."""
    check_service(ntp_time, "NTP Time")
    logger.info("API: Received request for current time.")
    try:
        # The NtpTime class has get_formatted_time and get_raw_time methods, not get_current_time_iso
        current_time = ntp_time.get_formatted_time("%Y-%m-%dT%H:%M:%S")
        return NtpTimeResponse(current_time_utc=current_time)
    except Exception as e:
        logger.exception(f"Error getting NTP time: {e}")
        return NtpTimeResponse(error=str(e))


# DuckDuckGo Search Endpoint
@app.post(
    "/search",
    response_model=DuckDuckGoSearchResults,
    tags=["Search and Weather Tools"],  # Use the model from the service
    summary="Perform DuckDuckGo and Weather Search",
    operation_id="duckduckgo_search",  # MCP tool name
    include_in_schema=False,
)
async def search_endpoint(request_data: SearchRequest):
    """Performs a web and weather search using DuckDuckGo."""
    check_service(duck_searcher, "DuckDuckGo Search")
    logger.info(f"API: Received search request for query: '{request_data.query}'")
    try:
        # From duck_search.py, the search method is not async
        results = duck_searcher.search(
            request_data.query, num_results=request_data.max_results
        )
        # The search method returns an instance of DuckDuckGoSearchResults
        return results
    except Exception as e:
        logger.exception(f"Error performing search for '{request_data.query}': {e}")
        # Return an empty result set with an error message
        return DuckDuckGoSearchResults(results=[])


# Anonymizer Endpoint
@app.post(
    "/anonymize",
    response_model=AnonymizeResponse,
    summary="Anonymize Text",
    operation_id="anonymize_text",  # MCP tool name
    include_in_schema=False,
)
async def anonymize_endpoint(request_data: AnonymizeRequest):
    """Anonymizes potentially sensitive information within the provided text."""
    check_service(anonymizer, "Anonymizer")
    logger.info("API: Received request to anonymize text.")
    try:
        # Use the batch method since that's what the Anonymizer class provides
        anonymized_list = anonymizer.anonymize_batch([request_data.text])
        if anonymized_list and len(anonymized_list) > 0:
            anonymized_text = anonymized_list[0]
        else:
            anonymized_text = (
                request_data.text
            )  # Return original if anonymization fails

        return AnonymizeResponse(anonymized_text=anonymized_text)
    except Exception as e:
        logger.exception(f"Error during anonymization: {e}")
        return AnonymizeResponse(error=str(e))


# PDF Converter Endpoint
@app.post(
    "/convert-to-pdf",
    response_model=ConvertToPdfResponse,
    summary="Convert File to PDF",
    operation_id="convert_to_pdf",  # MCP tool name
    include_in_schema=False,
)
async def convert_to_pdf_endpoint(request_data: ConvertToPdfRequest):
    """
    Converts various file types to PDF format.

    Supported file types:
    - Text files: .txt, .md, .py, .csv, .log, .json, .xml
    - Image files: .jpg, .png, .jpeg, .gif, .bmp, .tiff, .webp
    - Office documents: .docx, .xlsx, .pptx (requires LibreOffice)
    - HTML files: .html, .htm (requires WeasyPrint)
    """
    check_service(pdf_converter, "PDF Converter")
    logger.info(
        f"API: Received request to convert file to PDF: {request_data.input_filepath}"
    )
    try:
        output_path = pdf_converter.convert(
            request_data.input_filepath,
            output_directory=request_data.output_directory,
            output_filename=request_data.output_filename,
        )

        if output_path:
            return ConvertToPdfResponse(output_filepath=str(output_path))
        else:
            return ConvertToPdfResponse(
                error="Conversion failed. Check logs for details."
            )
    except Exception as e:
        logger.exception(f"Error during PDF conversion: {e}")
        return ConvertToPdfResponse(error=str(e))


# --- FastAPI MCP Integration ---
# Determine base URL (adjust logic as needed for your deployment)
server_host = os.environ.get("SERVER_HOST", "localhost")
server_port = os.environ.get("SERVER_PORT", "8000")
try:
    port_num = int(server_port)
except ValueError:
    port_num = 8000
    logger.warning(f"Invalid SERVER_PORT '{server_port}', using default {port_num}.")

# Construct base URL dynamically - adjust scheme if using HTTPS
# Use SERVER_SCHEME env var if available, default to http
server_scheme = os.environ.get("SERVER_SCHEME", "http")
base_url = f"{server_scheme}://{server_host}:{port_num}"
logger.info(f"Configuring MCP with base_url: {base_url}")

# Initialize MCP before using decorators
mcp = FastApiMCP(
    app,
    name="Integrated Services MCP",
    describe_full_response_schema=True,
    description="Provides tools for web interaction, time, search, anonymization, query refactoring, email, and SQLite database access.",
    include_operations=[
        "get_current_time",
        "duckduckgo_search",
        "anonymize_text",
        "process_document",
        "convert_to_pdf",  # Add this new operation
    ],
)

mcp.mount()  # This automatically discovers endpoints with operation_id


# --- Run Server ---
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on host 0.0.0.0:{port_num}")

    # Check if reload should be enabled (e.g., based on an environment variable)
    reload_enabled = os.getenv("UVICORN_RELOAD", "true").lower() == "true"
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info").lower()
    workers_count = int(os.getenv("UVICORN_WORKERS", "1"))

    # For reload or workers to work, we need to use import string format
    if reload_enabled or workers_count > 1:
        # Use import string format for reload/workers support
        uvicorn.run(
            "mcp_main:app",  # Import string format
            host="0.0.0.0",
            port=port_num,
            reload=reload_enabled
            if workers_count == 1
            else False,  # Can't use both reload and workers
            workers=workers_count
            if not reload_enabled
            else 1,  # Can't use both reload and workers
            log_level=log_level,
        )
    else:
        # Use app object directly for simple production deployment
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port_num,
            log_level=log_level,
        )
