import os
import json
import pickle
import numpy as np
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load document based on file type
def load_document(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".docx":
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext in [".html", ".htm"]:
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Split text into word-based chunks
def split_into_chunks(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Embed chunks using SentenceTransformer
def embed_chunks(text_chunks, model):
    return model.encode(text_chunks)

# Find supported document in folder
def find_first_supported_file(assets_dir="assets"):
    supported_exts = [".pdf", ".docx", ".txt", ".html", ".htm"]
    for file in os.listdir(assets_dir):
        if os.path.splitext(file)[-1].lower() in supported_exts:
            return os.path.join(assets_dir, file)
    return None

# Main Pipeline
def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    assets_dir = "assets"
    save_dir = "insurance_db"
    os.makedirs(save_dir, exist_ok=True)

    file_path = find_first_supported_file(assets_dir)
    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(f"❌ No supported document found in '{assets_dir}'.")

    print(f"📄 Loading: {file_path}")
    text = load_document(file_path)
    print(f"✅ Loaded {len(text)} characters.")

    # Chunk and embed
    raw_chunks = split_into_chunks(text, max_words=100)
    chunks = [{"id": f"CL-{i+1:04d}", "text": chunk} for i, chunk in enumerate(raw_chunks)]
    print(f"✂️ Split into {len(chunks)} chunks.")

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_chunks(chunk_texts, model)
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)

    with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    metadata = {
        "source_file": os.path.basename(file_path),
        "model": "all-MiniLM-L6-v2",
        "num_chunks": len(chunks),
        "chunking_strategy": "word_count=100",
        "embedding_dim": embeddings.shape[1]
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n📦 Saved to:", save_dir)
    print("📌 First chunk:\n", chunks[0])
    print("\n🧠 First embedding:\n", embeddings[0])

    # Query Phase
    query = input("\n❓ Ask your insurance-related question: ")
    query_embedding = model.encode([query])[0]

    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    top_chunks = [chunks[i] for i in top_indices]

    # Build context for prompt
    context = "\n\n".join([f"[{chunk['id']}] {chunk['text']}" for chunk in top_chunks])

    # Load and format prompt from modelfile.txt
    modelfile_path = os.path.join(assets_dir, "modelfile.txt")
    if not os.path.isfile(modelfile_path):
        raise FileNotFoundError(f"❌ 'modelfile.txt' not found in {assets_dir}")

    with open(modelfile_path, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = template.replace("{{context}}", context).replace("{{question}}", query)

    # Send request to local phi3:mini model using Ollama
    print("\n⏳ Sending request to phi3:mini using prompt from modelfile.txt...")
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "phi3:mini",
        "prompt": prompt,
        "stream": False
    })

    result = response.json().get("response", "[No response returned]")
    print("\n🤖 Answer:\n", result)

if __name__ == "__main__":
    main()