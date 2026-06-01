import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI

load_dotenv()

client = Client()

DATASET_NAME = "fruit-agent-hallucination"

EXAMPLES = [
    {"input": "what is the price of apple?"},
    {"input": "what is the price of banana?"},
    {"input": "what is the price of orange?"},
    {"input": "how much does mango cost?"},
    {"input": "what is the price of strawberry?"},
]


async def _run_agent(question: str) -> dict:
    from agent.graph import graph

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": f"eval-{question}"}},
    )

    messages = result["messages"]
    final_response = messages[-1].content

    tool_outputs = [
        msg.content for msg in messages if isinstance(msg, ToolMessage)
    ]
    context = "\n".join(tool_outputs) if tool_outputs else "No tool was called"

    return {"output": final_response, "context": context}


def target(inputs: dict) -> dict:
    return asyncio.run(_run_agent(inputs["input"]))


def hallucination_evaluator(run, example) -> dict:
    output = run.outputs.get("output", "")
    context = run.outputs.get("context", "")

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
    )

    prompt = f"""You are evaluating whether an AI fruit market assistant hallucinated price information.

Tool output (ground truth from database):
{context}

Agent response:
{output}

Does the agent's response contain price information that is NOT supported by the tool output?
- If the agent made up a price not present in the tool output, answer: hallucination
- If the agent's price matches the tool output (or no price was stated), answer: grounded

Answer with only one word: hallucination or grounded"""

    verdict = llm.invoke(prompt).content.strip().lower()

    return {
        "key": "hallucination",
        "score": 0 if verdict == "hallucination" else 1,
        "comment": verdict,
    }


def setup_dataset():
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(
            DATASET_NAME,
            description="Price questions to test if agent hallucinates fruit prices.",
        )
        client.create_examples(
            inputs=[{"input": ex["input"]} for ex in EXAMPLES],
            dataset_id=dataset.id,
        )
        print(f"Created dataset: {DATASET_NAME}")
    else:
        print(f"Dataset already exists: {DATASET_NAME}")


if __name__ == "__main__":
    setup_dataset()

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[hallucination_evaluator],
        experiment_prefix="hallucination-check",
        description="Checks if agent states prices not returned by get_latest_price tool.",
        max_concurrency=0,
    )

    print("\n=== Results ===")
    for r in results:
        inp = r["example"].inputs["input"]
        score = r["evaluation_results"]["results"][0].score
        comment = r["evaluation_results"]["results"][0].comment
        label = "PASS" if score == 1 else "FAIL"
        print(f"[{label}] {inp} → {comment}")
