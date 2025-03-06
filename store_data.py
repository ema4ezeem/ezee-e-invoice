import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to ChromaDB (stored locally)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="einvoice_knowledge")

# Your e-invoicing knowledge base
knowledge_base = [
    {"question": "What is Malaysia's e-invoicing deadline?", "answer": "July 2025 for large taxpayers."},
    {"question": "Who needs to comply with e-invoicing?", "answer": "All businesses with revenue above RM100 million from 2024."},
    {"question": "What is the penalty for non-compliance?", "answer": "A fine of up to RM50,000 or imprisonment."}
]

# Store embeddings in ChromaDB
for idx, item in enumerate(knowledge_base):
    embedding = model.encode(item["question"]).tolist()
    collection.add(
        ids=[str(idx)],
        embeddings=[embedding],
        metadatas=[{"question": item["question"], "answer": item["answer"]}]
    )

print("✅ Knowledge base stored successfully!")
