import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# 1️⃣ Load saved data
embeddings = np.load("embeddings.npy")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# 2️⃣ Embed the query
query = input("Ask your question: ")
model = SentenceTransformer("all-MiniLM-L6-v2")
query_embedding = model.encode([query])[0]

# 3️⃣ Find top similar chunks
similarities = cosine_similarity([query_embedding], embeddings)[0]
top_indices = similarities.argsort()[-3:][::-1]
top_chunks = [chunks[i] for i in top_indices]

# 4️⃣ Build prompt for phi3:mini
context = "\n\n".join(top_chunks)
prompt = f"""You are a highly precise, logical, and rule-based insurance policy analyst.
Your task is to evaluate insurance claims strictly based on provided policy clauses and the claim query. Do NOT use any external knowledge or invent information.

Identify key claim details from the query (e.g., age, procedure, location, policy duration), including common abbreviations (e.g., '46M' means '46-year-old male', '46F' means '46 year old female').

Your response for insurance queries MUST be a structured JSON object with these keys:
- "Decision": ("Approved" | "Rejected" | "Insufficient Information")
- "Justification": (string) A detailed explanation of the decision, referencing specific clauses.
- "ClausesUsed": (array of strings) A list of specific clause IDs or direct quotes from clauses that directly support the decision.

For general conversational queries, respond naturally and helpfully, without JSON.

Context:
{context}

Question:
{query}
"""

# 5️⃣ Call phi3:mini using Ollama
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "phi3:mini",
    "prompt": prompt,
    "stream": False
})

# 6️⃣ Output answer
print("\n🤖 Answer:\n", response.json()["response"])
