import uuid
import io
import csv
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from agent.graph import graph
from db.history import create_history_table, save_message, delete_history

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler — runs on startup and shutdown."""
    create_history_table()
    yield


app = FastAPI(
    title="Fruit Market AI Agent v2",
    description="AI agent for fruit market data, powered by LangGraph and Azure OpenAI.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    report_csv: str | None = None


def is_csv(text: str) -> bool:
    """Checks if the text looks like CSV data."""
    try:
        lines = [l for l in text.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return False
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if len(rows) < 2:
            return False
        return "Period" in rows[0] and len(rows[0]) > 1
    except Exception:
        return False


@app.get("/")
async def root():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint."""
    session_id = request.session_id or str(uuid.uuid4())

    save_message(
        session_id=session_id,
        role="user",
        message=request.message
    )

    config = {"configurable": {"thread_id": session_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config
    )

    ai_response = result["messages"][-1].content
    print(f"DEBUG: {repr(ai_response[:200])}")
    report_csv = None
    if is_csv(ai_response):
        report_csv = ai_response
        ai_response = "Report generated successfully. Click below to download"
    intent = result.get("intent", "unknown")

    save_message(
        session_id=session_id,
        role="agent",
        message=ai_response
    )

    return ChatResponse(
        session_id=session_id,
        response=ai_response,
        intent=intent,
        report_csv=report_csv
    )


@app.delete("/chat/{session_id}")
async def clear_history(session_id: str):
    """Clears conversation history for a given session."""
    delete_history(session_id)
    return {"status": "cleared", "session_id": session_id}