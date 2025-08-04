import os
import json
import pickle
import hashlib
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Directories
ASSETS_DIR = "assets"
SAVE_DIR = "insurance_db"
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Utility: Hash file to avoid recomputing
def calculate_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Load file
def load_document(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".txt":
        return open(filepath, "r", encoding="utf-8").read()
    elif ext == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(filepath).pages)
    elif ext == ".docx":
        return "\n".join(p.text for p in Document(filepath).paragraphs)
    elif ext in [".html", ".htm"]:
        return BeautifulSoup(open(filepath, "r", encoding="utf-8"), "html.parser").get_text()
    else:
        raise ValueError("Unsupported file type")

# Chunk and embed
def split_chunks(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def embed_chunks(text_chunks):
    return model.encode(text_chunks)

# UI
st.title("📄 ClaimWise AI - Document Q&A")
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "html", "htm"])

if uploaded_file:
    file_path = os.path.join(ASSETS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    file_hash = calculate_file_hash(file_path)
    metadata_path = os.path.join(SAVE_DIR, "metadata.json")

    # Check if same doc
    recompute = True
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            existing_meta = json.load(f)
        if existing_meta.get("hash") == file_hash:
            recompute = False
            embeddings = np.load(os.path.join(SAVE_DIR, "embeddings.npy"))
            with open(os.path.join(SAVE_DIR, "chunks.pkl"), "rb") as f:
                chunks = pickle.load(f)

    if recompute:
        st.info("🔁 Processing new document...")
        text = load_document(file_path)
        raw_chunks = split_chunks(text)
        chunks = [{"id": f"CL-{i+1:04d}", "text": ch} for i, ch in enumerate(raw_chunks)]
        embeddings = embed_chunks([ch["text"] for ch in chunks])
        np.save(os.path.join(SAVE_DIR, "embeddings.npy"), embeddings)
        with open(os.path.join(SAVE_DIR, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)
        metadata = {
            "hash": file_hash,
            "source": uploaded_file.name,
            "num_chunks": len(chunks),
            "model": "all-MiniLM-L6-v2"
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        st.success(f"✅ Document processed with {len(chunks)} chunks.")

    # Q&A
    st.subheader("💬 Ask a question")
    user_query = st.text_input("Your question about the document:")
    if user_query:
        with st.spinner("🔍 Searching relevant content..."):
            query_embedding = model.encode([user_query])[0]
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            top_indices = similarities.argsort()[-3:][::-1]
            top_chunks = [chunks[i] for i in top_indices]
            context = "\n\n".join([f"[{ch['id']}] {ch['text']}" for ch in top_chunks])

        # Prepare prompt
        modelfile_path = os.path.join(ASSETS_DIR, "modelfile.txt")
        if not os.path.isfile(modelfile_path):
            st.error("❌ 'modelfile.txt' not found in assets.")
        else:
            template = open(modelfile_path, "r", encoding="utf-8").read()
            prompt = template.replace("{{context}}", context).replace("{{question}}", user_query)

            # Send to Ollama
            with st.spinner("🧠 Thinking (phi3:mini)..."):
                try:
                    res = requests.post("http://localhost:11434/api/generate", json={
                        "model": "phi3:mini",
                        "prompt": prompt,
                        "stream": False
                    })
                    answer = res.json().get("response", "[No answer returned]")
                    st.success("🤖 Answer:")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"API Error: {e}")
