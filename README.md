# A2A-MCP: Agent-to-Agent Model Control Protocol

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](License.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20with-uv-purple.svg)](https://github.com/astral-sh/uv)

**A comprehensive multi-agent AI system combining Agent-to-Agent (A2A) communication with Model Control Protocol (MCP) services for intelligent text processing, web interaction, and document management.**

## 🚀 Features

### Core Capabilities
- **🤖 Intelligent User Interface Agent**: Auto-detects user intent and coordinates appropriate services
- **🔄 Agent-to-Agent Communication**: Seamless coordination between specialized AI agents
- **🛠️ MCP Service Integration**: Standardized tool access for web scraping, search, and file conversion
- **🌐 Web Interface**: User-friendly Gradio interface for easy interaction
- **📊 Real-time Monitoring**: Service health monitoring and automatic restart capabilities

### Available Agents
- **📝 Lektor Agent**: Grammar and spelling correction
- **🎯 Optimizer Agent**: Text optimization with customizable tonality
- **😊 Sentiment Agent**: Emotion and sentiment analysis with anonymization support
- **🔄 Query Refactor Agent**: Query optimization for improved LLM processing
- **🧠 User Interface Agent**: Intelligent coordinator for all services

### MCP Services
- **🌐 Website Text Extraction**: Headless browser with JavaScript rendering
- **🔍 Web Search**: DuckDuckGo search with weather information
- **🔒 Text Anonymization**: PII detection and anonymization with optional LLM enhancement
- **📄 PDF Conversion**: Multi-format file to PDF conversion
- **🕒 NTP Time Service**: Accurate time synchronization

## 📋 Prerequisites

- **Python 3.11+** (Required for modern async features)
- **uv** package manager (⚠️ **Important**: Use `uv` instead of `pip` for dependency management)
- **Chrome/Chromium** browser (for headless browsing)
- **LibreOffice** (optional, for advanced document conversion)

## 🛠️ Installation

### 1. Install uv Package Manager

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Using pip (if uv is not available)
pip install uv
```

### 2. Clone and Setup Project

```bash
# Clone the repository
git clone <repository-url>
cd a2a_mcp

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies using uv (NOT pip!)
uv pip install -r requirements.txt

# Install spaCy German language model
uv pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration (Required)
API_KEY=your_api_key_here
BASE_URL=http://localhost:11434/v1/chat/completions  # For Ollama
MODEL_NAME=llama3.2:latest
TEXT_OPT_MODEL=llama3.2:latest

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=8000
SERVER_SCHEME=http

# Gradio Interface
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860
GRADIO_SHARE=false

# Anonymizer Configuration (Optional)
ANONYMIZER_USE_LLM=true
ANONYMIZER_LLM_ENDPOINT=http://localhost:11434/v1/chat/completions
ANONYMIZER_LLM_API_KEY=your_llm_api_key
ANONYMIZER_LLM_MODEL=llama3.2:latest

# Development Settings
UVICORN_RELOAD=true
UVICORN_LOG_LEVEL=info
```

### 4. Install Optional Dependencies

```bash
# For enhanced PDF conversion (LibreOffice required)
# Ubuntu/Debian
sudo apt-get install libreoffice

# macOS
brew install --cask libreoffice

# Windows: Download from https://www.libreoffice.org/
```

## 🚀 Quick Start

### Option 1: Launch All Services (Recommended)

```bash
# Start all services with the launcher
python launcher.py
```

This will start:
- A2A Agent Registry (embedded)
- MCP Server on http://localhost:8000
- Gradio Interface on http://127.0.0.1:7860

### Option 2: Manual Service Startup

```bash
# Terminal 1: Start MCP Server
python mcp_main.py

# Terminal 2: Start Gradio Interface
python gradio_interface.py

# A2A server runs embedded within the services
```

### Option 3: Python API Usage

```python
import asyncio
from agent_server.user_interface import process_user_request

async def main():
    # Process a user request
    result = await process_user_request(
        "Korrigiere diesen Text: Das ist ein sehr schlechte Satz."
    )
    print(f"Result: {result.final_result}")
    print(f"Operation: {result.operation_type}")

asyncio.run(main())
```

## 📖 Usage Examples

### Text Processing

```python
# Grammar correction
await process_user_request("Korrigiere: Das ist ein sehr schlechte Satz.")

# Text optimization with tonality
await process_user_request("Optimiere diesen Text für eine freundliche E-Mail")

# Sentiment analysis
await process_user_request("Analysiere das Sentiment: Ich liebe dieses Produkt!")
```

### Web Interaction

```python
# Web search
await process_user_request("Wie wird das Wetter morgen in Berlin?")

# Website content extraction
await process_user_request("Extrahiere Text von https://example.com")

# Current time
await process_user_request("Wie spät ist es?")
```

### File Operations

```python
# PDF conversion
await process_user_request("Konvertiere diese Datei zu PDF: /path/to/file.docx")

# Text anonymization
await process_user_request("Anonymisiere sensible Daten in dieser Datei")
```

### Agent Coordination

```python
# Full processing pipeline
await process_user_request(
    "Korrigiere, optimiere und analysiere das Sentiment dieses Textes"
)
```

## 🔧 Configuration

### LLM Providers

#### Ollama (Local)
```env
API_KEY=ollama
BASE_URL=http://localhost:11434/v1/chat/completions
MODEL_NAME=llama3.2:latest
```

#### OpenAI
```env
API_KEY=sk-your-openai-key
BASE_URL=https://api.openai.com/v1/chat/completions
MODEL_NAME=gpt-4
```

#### Custom Provider
```env
API_KEY=your_api_key
BASE_URL=https://your-provider.com/v1/chat/completions
MODEL_NAME=your_model_name
```

### Service Configuration

#### MCP Server Settings
```env
SERVER_HOST=0.0.0.0      # Listen on all interfaces
SERVER_PORT=8000         # MCP server port
SERVER_SCHEME=https      # Use HTTPS in production
```

#### Gradio Interface
```env
GRADIO_HOST=0.0.0.0      # Public access
GRADIO_PORT=7860         # Gradio port
GRADIO_SHARE=true        # Create public Gradio link
```

### Anonymization Configuration

```env
ANONYMIZER_USE_LLM=true                                    # Enable LLM enhancement
ANONYMIZER_LLM_ENDPOINT=http://localhost:11434/v1/chat/completions
ANONYMIZER_LLM_API_KEY=your_key
ANONYMIZER_LLM_MODEL=llama3.2:latest
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Gradio Web UI  │    │   Direct API    │    │   A2A Clients   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    User Interface Agent   │
                    │   (Intent Recognition)    │
                    └─────────────┬─────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
       ┌────────▼───────┐ ┌──────▼──────┐ ┌─────▼─────┐
       │  A2A Agents    │ │ MCP Services │ │ External  │
       │                │ │              │ │ Services  │
       │ • Lektor       │ │ • Web Search │ │ • Ollama  │
       │ • Optimizer    │ │ • Extraction │ │ • OpenAI  │
       │ • Sentiment    │ │ • Anonymizer │ │ • NTP     │
       │ • Query Ref    │ │ • PDF Conv   │ │ • Chrome  │
       └────────────────┘ └──────────────┘ └───────────┘
```

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
uv pip install pytest pytest-asyncio

# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Code Quality

```bash
# Install development tools
uv pip install black isort flake8 mypy

# Format code
black .
isort .

# Lint code
flake8 .
mypy .
```

### Adding New Agents

1. Create agent file in `agent_server/`:
```python
# agent_server/my_agent.py
from pydantic import BaseModel
from pydantic_ai import Agent

class MyAgentResponse(BaseModel):
    result: str
    status: str

async def my_agent_a2a_function(messages: list) -> MyAgentResponse:
    # Implementation
    pass
```

2. Register in `a2a_server.py`:
```python
from agent_server.my_agent import my_agent_a2a_function

# In setup_a2a_server()
registry.register_a2a_agent("my_agent", my_agent_a2a_function)
```

### Adding New MCP Services

1. Create service in `mcp_services/`:
```python
# mcp_services/my_service/service.py
class MyService:
    def process(self, data: str) -> str:
        # Implementation
        pass
```

2. Add endpoint in `mcp_main.py`:
```python
@app.post("/my-endpoint", operation_id="my_service")
async def my_endpoint(request: MyRequest):
    # Implementation
    pass
```

## 🐛 Troubleshooting

### Common Issues

#### spaCy Model Not Found
```bash
# Install German language model
uv pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl
```

#### ChromeDriver Issues
```bash
# ChromeDriver is automatically managed by webdriver-manager
# If issues persist, install Chrome/Chromium manually
```

#### Port Already in Use
```bash
# Check what's using the port
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Change port in .env file
SERVER_PORT=8001
GRADIO_PORT=7861
```

#### uv Installation Issues
```bash
# If uv is not available, install via pip first
pip install uv

# Then use uv for all other dependencies
uv pip install -r requirements.txt
```

### Logging and Debugging

```bash
# Enable debug logging
export UVICORN_LOG_LEVEL=debug

# View logs in real-time
tail -f logs/app.log  # If logging to file
```

### Performance Optimization

```bash
# Use production settings
export UVICORN_RELOAD=false
export UVICORN_WORKERS=4

# Optimize for memory usage
export ANONYMIZER_USE_LLM=false  # Disable LLM if not needed
```

## 📚 API Documentation

### MCP Services API

Once the server is running, access interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### A2A Agent Functions

All agents expose async functions accepting message lists and returning structured responses:

```python
async def agent_function(messages: List[ModelMessage]) -> AgentResponse:
    """
    Process messages and return structured response.
    
    Args:
        messages: List of user/system messages
        
    Returns:
        Structured response with status and results
    """
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes using `uv` for dependency management
4. Ensure tests pass: `pytest`
5. Format code: `black . && isort .`
6. Commit changes: `git commit -am 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Development Guidelines

- **Always use `uv`** instead of `pip` for dependency management
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write tests for new functionality
- Update documentation for API changes
- Use async/await for I/O operations

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPLv3) - see the [License.md](License.md) file for details.

### Key Points:
- ✅ **Free to use**, modify, and distribute
- ✅ **Commercial use** allowed
- ⚠️ **Copyleft**: Derivative works must also be AGPLv3
- ⚠️ **Network use**: If you run this as a service, you must provide source code

## 🙏 Acknowledgments

- **[Pydantic AI](https://github.com/pydantic/pydantic-ai)** - AI agent framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- **[Gradio](https://gradio.app/)** - ML web interfaces
- **[spaCy](https://spacy.io/)** - Natural language processing
- **[Trafilatura](https://trafilatura.readthedocs.io/)** - Web content extraction
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/a2a-mcp/issues)
- **Documentation**: [Project Wiki](https://github.com/your-username/a2a-mcp/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/a2a-mcp/discussions)

---

**Made with ❤️ by the A2A-MCP Team**

*Remember: Always use `uv` for dependency management to ensure consistent and fast package installation!*
