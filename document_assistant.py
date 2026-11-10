
import os
import json
import pymupdf
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# ---------------- STORAGE CONFIGURATION ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STORAGE_DIR = os.path.join(BASE_DIR, "vectorstore")

INDEX_FILE = os.path.join(STORAGE_DIR, "documents.index")
CHUNKS_FILE = os.path.join(STORAGE_DIR, "chunks.json")

os.makedirs(STORAGE_DIR, exist_ok=True)


# ---------------- LOAD EMBEDDING MODEL ----------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- PDF TEXT EXTRACTION ----------------

def extract_pdf_text(uploaded_files):

    documents = []

    for pdf in uploaded_files:

        pdf_document = pymupdf.open(
            stream=pdf.getvalue(),
            filetype="pdf"
        )

        for page_number, page in enumerate(pdf_document):

            text = page.get_text().strip()

            if text:
                documents.append({
                    "text": text,
                    "source": pdf.name,
                    "page": page_number + 1
                })

        pdf_document.close()

    return documents


# ---------------- DOCUMENT CHUNKING ----------------

def split_documents(documents, chunk_size=800, overlap=150):

    chunks = []

    for document in documents:

        text = document["text"]
        start = 0

        while start < len(text):

            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document["page"]
                })

            start += chunk_size - overlap

    return chunks


# ---------------- CREATE FAISS VECTOR DATABASE ----------------

def create_vector_database(chunks):

    if not chunks:
        raise ValueError("No readable PDF content found.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index


# ---------------- SAVE VECTOR DATABASE ----------------

def save_vector_database(index, chunks):

    os.makedirs(STORAGE_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_FILE)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Vector database saved successfully!")
    print("Storage location:", STORAGE_DIR)


# ---------------- LOAD SAVED VECTOR DATABASE ----------------

def load_vector_database():

    if not os.path.exists(INDEX_FILE):
        return None, []

    if not os.path.exists(CHUNKS_FILE):
        return None, []

    index = faiss.read_index(INDEX_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        return None, []

    print("Saved documents loaded successfully!")

    return index, chunks


# ---------------- SEARCH DOCUMENTS ----------------

def search_documents(question, index, chunks, top_k=4):

    if index is None or not chunks:
        return []

    embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embedding = np.asarray(embedding, dtype="float32")

    k = min(top_k, len(chunks))

    scores, indices = index.search(embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "score": float(score)
        })

    return results
