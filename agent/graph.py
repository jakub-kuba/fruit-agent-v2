from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
import operator
from agent.nodes import (
    rag_node,
    agent_node,
    tool_node,
    should_continue,
    classify_intent
)


class AgentState(TypedDict):
    """State definition for the agent graph."""
    messages: Annotated[list[BaseMessage], operator.add]
    rag_context: str
    intent: str


def build_graph() -> StateGraph:
    """Builds and compiles the LangGraph agent graph."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classifier", classify_intent)
    workflow.add_node("rag", rag_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("classifier")

    # After classifier — route based on intent
    workflow.add_conditional_edges(
        "classifier",
        lambda state: state["intent"],
        {
            "knowledge": "rag",
            "tool": "agent",
            "off_topic": "agent",
        }
    )

    # After RAG — always go to agent
    workflow.add_edge("rag", "agent")

    # After agent — tools or end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )

    # After tools — back to agent
    workflow.add_edge("tools", "agent")

    # Compile with MemorySaver
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Single instance reused across requests
graph = build_graph()