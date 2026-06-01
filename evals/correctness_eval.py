import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

client = Client()

DATASET_NAME = "fruit-agent-correctness"

EXAMPLES = [
    {
        "input": "Which country is the largest producer of apples in the world?",
        "reference": "China is the largest producer of apples, producing over 50% of global supply.",
    },
    {
        "input": "When are oranges in season in Europe?",
        "reference": "Southern European oranges are in season from November to April.",
    },
    {
        "input": "Where do bananas imported to Europe come from?",
        "reference": "The vast majority of bananas imported to Europe come from Ecuador, Colombia, Costa Rica, and Cameroon.",
    },
    {
        "input": "Which country dominates global chokeberry production?",
        "reference": "Poland dominates global chokeberry production, responsible for approximately 90% of the world's commercial supply.",
    },
    {
        "input": "What is the harvest season for strawberries in Europe?",
        "reference": "The main strawberry season in Europe runs from April to July, starting in southern countries like Spain and Italy.",
    },
    {
        "input": "Who are the top EU producers of blueberries?",
        "reference": "The top EU producers of blueberries are Poland, Germany, and the Netherlands.",
    },
    {
        "input": "Where does hazelnut imported to Europe mainly come from?",
        "reference": "Turkey supplies the vast majority of hazelnuts consumed in Europe, accounting for roughly 70% of global production.",
    },
    {
        "input": "When are tangerines in season in Europe?",
        "reference": "Tangerines are in season in Europe from October to March.",
    },
    {
        "input": "What are the top 3 world producers of cherries?",
        "reference": "The top 3 world producers of cherries are Turkey, United States, and Iran.",
    },
    {
        "input": "Which country produces the most watermelons in the world?",
        "reference": "China is the largest producer of watermelons, producing approximately 70% of the world's supply.",
    },
]


async def _run_agent(question: str) -> dict:
    from agent.graph import graph

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": f"eval-correctness-{question[:30]}"}},
    )
    return {"output": result["messages"][-1].content}


def target(inputs: dict) -> dict:
    return asyncio.run(_run_agent(inputs["input"]))


def correctness_evaluator(run, example) -> dict:
    output = run.outputs.get("output", "")
    reference = example.outputs.get("reference", "")
    question = example.inputs.get("input", "")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    prompt = f"""You are an expert evaluating whether an AI assistant's answer is factually correct.

Question: {question}

Reference answer (ground truth from knowledge base):
{reference}

Agent's answer:
{output}

Is the agent's answer factually correct and consistent with the reference answer?
- Score 1: answer is correct and covers the key facts
- Score 0: answer contains factual errors or contradicts the reference

Respond with only: correct or incorrect"""

    verdict = llm.invoke(prompt).content.strip().lower()

    return {
        "key": "correctness",
        "score": 1 if verdict == "correct" else 0,
        "comment": verdict,
    }


def setup_dataset():
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(
            DATASET_NAME,
            description="Knowledge questions to test RAG correctness against fruit_knowledge_base.md.",
        )
        client.create_examples(
            inputs=[{"input": ex["input"]} for ex in EXAMPLES],
            outputs=[{"reference": ex["reference"]} for ex in EXAMPLES],
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
        evaluators=[correctness_evaluator],
        experiment_prefix="correctness-check",
        description="Checks if agent answers knowledge questions correctly based on fruit_knowledge_base.md.",
        max_concurrency=0,
    )

    print("\n=== Results ===")
    for r in results:
        inp = r["example"].inputs["input"]
        score = r["evaluation_results"]["results"][0].score
        label = "PASS" if score == 1 else "FAIL"
        print(f"[{label}] {inp}")
