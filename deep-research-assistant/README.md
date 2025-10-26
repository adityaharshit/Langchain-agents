# Deep Research Assistant

A sophisticated multi-agent orchestration system built with LangGraph that combines retrieval-augmented generation (RAG), intelligent web scraping, and multi-hop reasoning to answer complex research queries.

## 🌟 Features

- **Multi-Agent Architecture**: 5 specialized agents working in coordination
- **🧠 LLM-Powered URL Generation**: GPT-4 intelligently suggests relevant sources for scraping
- **Intelligent RAG**: PostgreSQL + pgvector for semantic search
- **Adaptive Web Scraping**: Ethical scraping with robots.txt compliance
- **Real-time Streaming**: Server-Sent Events for progress tracking
- **Semantic Chunking**: Adaptive text segmentation with overlap
- **IEEE Citations**: Proper academic citation formatting
- **Provenance Tracking**: Full traceability from claims to sources
- **Language Filtering**: English-only content processing

## 🏗️ Architecture

### Multi-Agent System

The system uses **LangGraph** to orchestrate five specialized agents:

1. **Research Coordinator** - Query decomposition and orchestration
2. **Web Scraper Agent** - **LLM-powered URL generation** and document retrieval  
3. **Deep Analysis Agent** - Comparative, trend, and causal analysis
4. **Fact-Checking Agent** - Source validation and contradiction detection
5. **Output Formatting Agent** - Citation formatting and report generation

### 🧠 Intelligent URL Generation

The Web Scraper Agent now uses **GPT-4** to intelligently generate relevant URLs for web scraping instead of hardcoded patterns:

- **LLM-Powered**: Uses OpenAI GPT-4 to analyze research queries and suggest authoritative sources
- **Contextual**: Considers the main query and all subqueries for comprehensive URL generation
- **Authoritative Sources**: Prioritizes academic institutions, government agencies, and reputable organizations
- **Fallback System**: Automatically falls back to pattern-based URL generation if LLM fails
- **Diverse Content**: Generates URLs for academic papers, reports, news articles, and official statistics

### Agent Workflow

```mermaid
graph TB
    A[Research Coordinator] --> B{Confidence Check}
    B -->|Low| C[Web Scraper Agent]
    B -->|High| D[Deep Analysis Agent]
    C --> D
    D --> E[Fact-Checking Agent]
    E --> F{Fact Check}
    F -->|Low Confidence| A
    F -->|High Confidence| G[Output Formatting Agent]
    G --> H[Final Result]
```

### Technology Stack

- **Framework**: FastAPI + LangGraph
- **Database**: PostgreSQL with pgvector extension
- **Embeddings**: OpenAI text-embedding-3-small
- **Web Scraping**: aiohttp + Playwright (fallback)
- **Language**: Python 3.11+

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- PostgreSQL with pgvector extension
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd deep-research-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Configure PostgreSQL with pgvector**
   ```sql
   -- Connect to your PostgreSQL instance
   CREATE DATABASE vectordb;
   CREATE USER vectoruser WITH PASSWORD 'vectorpass';
   GRANT ALL PRIVILEGES ON DATABASE vectordb TO vectoruser;
   
   -- Enable pgvector extension
   \c vectordb
   CREATE EXTENSION vector;
   ```

5. **Run the application**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### Environment Configuration

Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults provided)
DATABASE_URL=postgresql://vectoruser:vectorpass@localhost:5433/vectordb
EMBEDDING_MODEL=text-embedding-3-small
LOG_LEVEL=INFO
```

## 📖 Usage

### Basic Query

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post("http://localhost:8000/api/v1/chat", json={
        "query": "How did COVID-19 impact renewable energy investment patterns?",
        "stream": True
    })
    
    task_id = response.json()["task_id"]
    
    # Stream progress
    async with client.stream("GET", f"http://localhost:8000/api/v1/stream/{task_id}") as stream:
        async for line in stream.aiter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                print(f"Progress: {event['step']}")
```

### API Endpoints

- **POST /api/v1/chat** - Submit research queries
- **GET /api/v1/stream/{task_id}** - Stream progress events (SSE)
- **GET /api/v1/status/{task_id}** - Get task status and results
- **GET /api/v1/health** - System health check
- **GET /api/v1/stats** - System statistics

### Example Response

```json
{
  "answer_markdown": "# Research Analysis: COVID-19 Impact on Renewable Energy\n\n## Key Findings\n- COVID-19 led to 23% decline in renewable investment in 2020...",
  "citations": [
    "[1] Renewable Energy Investment Trends 2019-2023, https://example.com/renewable-energy-2023",
    "[2] COVID-19 Economic Impact on Developing Countries, https://example.com/covid-impact"
  ],
  "provenance": [
    {
      "claim": "Information from Renewable Energy Investment Trends",
      "chunk_id": 123,
      "document_title": "Renewable Energy Investment Trends 2019-2023",
      "document_url": "https://example.com/renewable-energy-2023",
      "similarity_score": 0.89,
      "citation_number": 1
    }
  ],
  "confidence_score": 0.85,
  "total_sources": 5
}
```

## 🧪 Demo and Testing

### Run the Demo

The system includes a comprehensive demo with seed data:

```bash
# Run the end-to-end demo
python -m pytest tests/test_end_to_end.py::run_demo -v

# Or run as a standalone script
cd tests
python test_end_to_end.py
```

### Demo Query

The demo uses this multi-hop research question:

> "How did COVID-19 impact renewable energy investment patterns in developing countries, and what regulatory changes emerged as a result across different regions?"

### Expected Demo Output

The demo will:
1. ✅ Initialize the system and seed the database
2. ✅ Process the query through all 5 agents
3. ✅ Generate IEEE-style citations
4. ✅ Provide provenance mapping
5. ✅ Stream progress events in real-time

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_end_to_end.py -v
```

## 🔧 Configuration

### Database Configuration

The system requires PostgreSQL with the pgvector extension:

```sql
-- Required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For full-text search

-- Recommended settings for performance
SET shared_preload_libraries = 'pg_stat_statements';
SET max_connections = 100;
```

### Performance Tuning

Key configuration parameters:

```python
# In app/config.py
RETRIEVAL_K = 8                    # Number of chunks to retrieve
CONFIDENCE_THRESHOLD = 0.7         # Minimum confidence for analysis
MAX_CONCURRENT_SCRAPES = 5         # Concurrent web scraping limit
SCRAPE_RATE_LIMIT = 1.0           # Seconds between requests
EMBEDDING_BATCH_SIZE = 16          # Batch size for embeddings
```

### MCP Tools Configuration

All tools are implemented as MCP (Model Context Protocol) tools with agent-scoped access:

- **Research Coordinator**: Query decomposition, task prioritization, progress tracking
- **Web Scraper**: Keyword search, web scraping, content processing
- **Deep Analysis**: Comparative analysis, trend analysis, causal reasoning
- **Fact Checking**: Source credibility, cross-reference validation
- **Output Formatting**: Citation formatting, report structuring

## 📊 System Components

### Semantic Chunking

The system implements adaptive semantic chunking:

- **Chunk Size**: 150-800 tokens (adaptive)
- **Overlap**: 20% semantic overlap
- **Boundary Detection**: Headings, paragraphs, semantic transitions
- **Metadata**: Section, heading, position, confidence

### Vector Storage

Uses PostgreSQL with pgvector for efficient similarity search:

- **Embedding Model**: OpenAI text-embedding-3-small (1536 dimensions)
- **Index Type**: HNSW for approximate nearest neighbor search
- **Similarity Metric**: Cosine similarity
- **Search Methods**: Semantic, keyword, hybrid

### Intelligent URL Generation

The system uses GPT-4 to intelligently generate relevant URLs for web scraping:

- **LLM Analysis**: GPT-4 analyzes the research query and subqueries
- **Authoritative Sources**: Prioritizes academic, government, and reputable sources
- **Content Diversity**: Suggests academic papers, reports, news articles, statistics
- **Contextual Relevance**: Considers query context and research domain
- **Fallback System**: Pattern-based generation if LLM is unavailable

### Web Scraping

Ethical web scraping with multiple safeguards:

- **Robots.txt Compliance**: Automatic checking and respect
- **Rate Limiting**: Configurable delays between requests
- **User Agent**: Clear identification as research bot
- **Fallback**: Playwright for JavaScript-heavy pages
- **Language Detection**: English-only content filtering

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check PostgreSQL is running
   pg_ctl status
   
   # Verify pgvector extension
   psql -d vectordb -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
   ```

2. **OpenAI API Errors**
   ```bash
   # Verify API key
   export OPENAI_API_KEY=your_key_here
   python -c "import openai; print(openai.api_key)"
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   python -c "
   from app.workers.embedding import get_embedding_generator
   import asyncio
   asyncio.run(get_embedding_generator().get_cache_stats())
   "
   ```

### Performance Optimization

1. **Database Indexing**
   ```sql
   -- Ensure vector index exists
   CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks 
   USING hnsw (embedding vector_cosine_ops) 
   WITH (m = 16, ef_construction = 64);
   ```

2. **Connection Pooling**
   ```python
   # Adjust in app/db/database.py
   pool_size=10,
   max_overflow=20,
   pool_recycle=3600
   ```

3. **Embedding Caching**
   ```python
   # Clear cache if needed
   from app.workers.embedding import embedding_generator
   embedding_generator.clear_cache()
   ```

## 🤝 Contributing

### Development Setup

1. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

2. **Run code formatting**
   ```bash
   black app/ tests/
   isort app/ tests/
   ```

3. **Run type checking**
   ```bash
   mypy app/
   ```

### Architecture Guidelines

- **Agent Design**: Each agent should have a single responsibility
- **Tool Implementation**: All tools must use the `@mcp.tool()` decorator
- **Error Handling**: Implement graceful degradation and retry logic
- **Logging**: Use structured logging with appropriate levels
- **Testing**: Write tests for all new functionality

### Adding New Agents

1. Create agent class in `app/agents.py`
2. Implement required tools in `app/mcp_tools.py`
3. Update graph structure in `app/langgraph_agents.py`
4. Add tests in `tests/`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph** for multi-agent orchestration
- **pgvector** for efficient vector similarity search
- **OpenAI** for embedding models
- **FastAPI** for the web framework

## 📞 Support

For questions, issues, or contributions:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review existing [GitHub issues](https://github.com/deep-research-assistant/issues)
3. Create a new issue with detailed information

---

**Note on Library Versions**: This project uses the latest versions of all dependencies without version pinning to ensure access to the most recent security updates and improvements. The code is tested with Python 3.11+ and should work with newer versions as they become available.