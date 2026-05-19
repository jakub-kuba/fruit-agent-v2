import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "data", "fruit_knowledge_base.md")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "fruit-knowledge")


def get_embeddings() -> AzureOpenAIEmbeddings:
    """Returns Azure OpenAI embeddings model."""
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )


def get_vector_store() -> AzureSearch:
    """Returns Azure AI Search vector store."""
    return AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
        index_name=AZURE_SEARCH_INDEX,
        embedding_function=get_embeddings(),
    )


def index_documents():
    """Loads knowledge base, splits into chunks and indexes in Azure AI Search."""
    loader = TextLoader(KNOWLEDGE_BASE_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["## ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks into Azure AI Search.")


def retrieve(query: str, k: int = 3) -> str:
    """Retrieves most relevant chunks from knowledge base for a given query."""
    vector_store = get_vector_store()
    results: list[Document] = vector_store.similarity_search(
        query=query,
        k=k,
    )

    if not results:
        return "No relevant information found in knowledge base."

    return "\n\n".join([doc.page_content for doc in results])