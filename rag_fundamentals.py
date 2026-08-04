from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Employees receive 20 days of paid leave every year.",
    "Annual bonuses are paid in December.",
    "Health insurance covers hospitalization expenses."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(np.array(embeddings).astype("float32"))

query = "How many leave days do employees receive?"

query_embedding = model.encode([query])

D, I = index.search(
    np.array(query_embedding).astype("float32"),
    k=1
)

print(documents[I[0][0]])
