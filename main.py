import uuid
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
    intent = result.get("intent", "unknown")

    save_message(
        session_id=session_id,
        role="agent",
        message=ai_response
    )

    return ChatResponse(
        session_id=session_id,
        response=ai_response,
        intent=intent
    )


@app.delete("/chat/{session_id}")
async def clear_history(session_id: str):
    """Clears conversation history for a given session."""
    delete_history(session_id)
    return {"status": "cleared", "session_id": session_id}