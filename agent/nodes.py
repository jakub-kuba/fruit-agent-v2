import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from agent.rag import retrieve
from agent.tools import get_latest_price, convert_price, get_analytics_report
from dotenv import load_dotenv

load_dotenv()

TOOLS = [get_latest_price, convert_price, get_analytics_report]

SYSTEM_PROMPT = """You are a fruit market assistant. You help users check current fruit prices, 
convert them to other currencies, generate price reports, and answer questions about fruits.

You have access to the following tools:
- get_latest_price: to get the current price of a specific fruit in PLN
- convert_price: to convert a fruit price to another currency
- get_analytics_report: to generate monthly or yearly price reports

You also have access to a knowledge base about fruits (seasonality, producers, origin).

Rules:
- Do not make up prices or data. Always use tools for price information.
- Always include the date when providing fruit prices.
- Never show exchange rate values or calculation steps. Only show the final converted price.
- When presenting report data for multiple fruits, combine them into a single table.
- If asked about anything unrelated to fruits, politely decline.
- Do not add closing pleasantries.
- Always respond in English unless the user explicitly writes in another language.
"""


def get_llm():
    """Returns Azure OpenAI LLM with tools bound."""
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
    )
    return llm.bind_tools(TOOLS)


# LangGraph wbudowany ToolNode — obsługuje tool calls poprawnie
tool_node = ToolNode(TOOLS)


def classify_intent(state: dict) -> dict:
    """Classifies user intent to decide whether RAG is needed."""
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
    )

    last_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content="""Classify the user message into one of three categories:
- "knowledge": user asks about fruit facts, seasonality, origin, producers
- "tool": user asks about fruit prices, currency conversion, or reports
- "off_topic": user asks about something unrelated to fruits

Respond with only one word: knowledge, tool, or off_topic."""),
        HumanMessage(content=last_message)
    ])

    state["intent"] = response.content.strip().lower()
    return state


def rag_node(state: dict) -> dict:
    """Retrieves relevant context from knowledge base."""
    last_message = state["messages"][-1].content
    context = retrieve(last_message)
    state["rag_context"] = context
    return state


def agent_node(state: dict) -> dict:
    """Main agent node — calls LLM with tools and context."""
    llm = get_llm()

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if state.get("rag_context"):
        messages.append(SystemMessage(
            content=f"Relevant knowledge base context:\n{state['rag_context']}"
        ))

    messages += state["messages"]

    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: dict) -> str:
    """Edge function — decides whether to call tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"