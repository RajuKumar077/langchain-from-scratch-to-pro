from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

CONNECTION = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name="lesson08",
)

results = vector_store.similarity_search(
    "AI",
    k=5,
)

for i, doc in enumerate(results, 1):
    print("=" * 60)
    print(f"Document {i}")
    print("=" * 60)
    print(doc.page_content)
    print("\nMetadata:")
    print(doc.metadata)
    