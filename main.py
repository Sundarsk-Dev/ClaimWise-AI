import os
import pickle
import numpy as np
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

# 1️⃣ Load document based on file type
def load_document(filepath):
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif ext == ".docx":
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext in [".html", ".htm"]:
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text()

    else:
        raise ValueError(f"Unsupported file type: {ext}")

# 2️⃣ Split long text into chunks (by words)
def split_into_chunks(text, max_words=100):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]

# 3️⃣ Embed chunks using MiniLM
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)

# 4️⃣ Run everything together
def main():
    file_path = "BAJHLIP23020V012223.pdf"  # 📄 Change this to your file name
    print(f"📄 Loading: {file_path}")
    
    # Load and parse text
    text = load_document(file_path)
    print(f"✅ Loaded {len(text)} characters.")

    # Split into chunks
    chunks = split_into_chunks(text, max_words=100)
    print(f"✂️ Split into {len(chunks)} chunks.")

    # Embed
    embeddings = embed_chunks(chunks)
    print(f"🔗 Embedding shape: {embeddings.shape}")  # (num_chunks, 384)

    # Optional: print first chunk and its embedding
    print("\n📌 First chunk:\n", chunks[0])
    print("\n🧠 First embedding:\n", embeddings[0])

    # Save embeddings and chunks
    np.save("embeddings.npy", embeddings)
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("\n💾 Saved embeddings and chunks.")

if __name__ == "__main__":
    main()
