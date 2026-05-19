from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import text
from db.engine import get_engine


class Message(BaseModel):
    role: str
    message: str
    created_at: datetime | None = None


class ConversationHistory(BaseModel):
    session_id: str
    messages: list[Message] = []


def create_history_table():
    """creates agent_connversations table if not exists"""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_conversation (
                          id INT AUTO_INCREMENT PRIMARY KEY,
                          session_id VARCHAR(36) NOT NULL,
                          role VARCHAR(10) NOT NULL,
                          message TEXT NOT NULL,
                          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()


def save_message(session_id: str, role: str, message: str):
    """Saves a single message to history."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO agent_conversation (session_id, role, message)
            VALUES (:session_id, :role, :message)
        """), {
            "session_id": session_id,
            "role": role,
            "message": message
        })
        conn.commit()


def get_history(session_id: str) -> ConversationHistory:
    """Retrieves conversation history for a given session."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT role, message, created_at
            FROM agent_conversation
            WHERE session_id = :session_id
            ORDER BY created_at ASC
        """), {"session_id": session_id})
        messages = [
            Message(
                role=row.role,
                message=row.message,
                created_at=row.created_at
            )
            for row in result
        ]
    return ConversationHistory(session_id=session_id, messages=messages)


def delete_history(session_id: str):
    """Deletes conversation history for a given session."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            DELETE FROM agent_conversation
            WHERE session_id = :session_id
        """), {"session_id": session_id})
        conn.commit()