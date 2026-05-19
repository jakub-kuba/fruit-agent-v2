# Fruit Market AI Agent v2

An AI-powered assistant for fruit market data, built with LangGraph and Azure OpenAI.

## What it does

- **Fruit prices** — latest prices in PLN from a live database
- **Currency conversion** — convert prices to any currency using NBP live rates
- **Price reports** — generate monthly or yearly reports with CSV download
- **Fruit knowledge** — answer questions about seasonality, origin, and producers using RAG

## Tech Stack

- **LangGraph** — agent orchestration and conversation memory
- **LangChain** — RAG pipeline
- **Azure OpenAI** (gpt-4o) — LLM
- **Azure AI Search** — vector store for knowledge base
- **FastAPI** — backend API
- **Streamlit** — frontend UI
- **MySQL** — conversation history and price data
- **LangSmith** — observability and tracing
- **Docker** — containerization
- **GitHub Actions** — CI/CD

## Architecture

```
Streamlit UI → FastAPI → LangGraph Agent
                              ├── RAG (Azure AI Search)
                              ├── Tools (MySQL, NBP API)
                              └── Memory (MemorySaver)
```

## Setup

1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your credentials
3. Index the knowledge base:
```bash
PYTHONPATH=. uv run python scripts/index_knowledge_base.py
```
4. Run with Docker:
```bash
docker-compose up --build
```
5. Run Streamlit:
```bash
uv run streamlit run ui/streamlit_app.py
```

## Tests

```bash
PYTHONPATH=. uv run pytest tests/ -v
```
