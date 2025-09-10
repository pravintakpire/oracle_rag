import os
import uuid
import re
import json
import numpy as np
import fitz  # PyMuPDF
import oracledb
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import OllamaEmbeddings
import array

# ---- Load environment
load_dotenv(find_dotenv(usecwd=True))

ORACLE_USER = os.environ["ORACLE_USER"]
ORACLE_PASSWORD = os.environ["ORACLE_PASSWORD"]
ORACLE_DSN = os.environ["ORACLE_DSN"]

# ---- Config ----
PDF_PATH = "human-nutrition-text.pdf"
# --- MODIFIED: New Doc ID and Table Name ---
DOC_ID = "nutrition-v1-bge"
TABLE_NAME = "chunks_bge"
EMBED_MODEL_NAME = "bge-m3" # The model name you used in 'ollama pull'
BATCH_INSERT = 200

# ... (all helper functions like clean_text, split_sentences, etc., remain unchanged) ...
def clean_text(t: str) -> str:
    t = t.replace("\r", " ")
    t = re.sub(r"-\s*\n\s*", "", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = t.replace("\n", " ").strip()
    return t
def split_sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]
def chunk_page_by_sentences(text: str, sents_per_chunk: int = 10, overlap: int = 2):
    sents = split_sentences(text)
    i = 0
    step = max(1, sents_per_chunk - overlap)
    while i < len(sents):
        piece = sents[i:i + sents_per_chunk]
        if not piece: break
        chunk = " ".join(piece)
        if chunk:
            yield chunk
        i += step
def pdf_pages(path: str):
    doc = fitz.open(path)
    try:
        for i in range(len(doc)):
            txt = doc[i].get_text("text") or ""
            yield (i + 1, clean_text(txt))
    finally:
        doc.close()

def main():
    print(f"Initializing local embedding model '{EMBED_MODEL_NAME}' via Ollama...")
    embeddings_model = OllamaEmbeddings(model=EMBED_MODEL_NAME)

    with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as connection:
        cursor = connection.cursor()
        print("✅ Connected to Oracle Database.")

        # --- MODIFIED: Use dynamic table name in the query ---
        print(f"Deleting existing chunks from table '{TABLE_NAME}' for doc_id='{DOC_ID}'...")
        cursor.execute(f"DELETE FROM {TABLE_NAME} WHERE doc_id = :doc_id", {"doc_id": DOC_ID})
        print(f"Deleted {cursor.rowcount} rows.")

        print("Reading PDF by pages...")
        pages = list(pdf_pages(PDF_PATH))

        inputs, metas = [], []
        print("Chunking...")
        for page, text in pages:
            if not text: continue
            for chunk in chunk_page_by_sentences(text):
                inputs.append(chunk)
                metas.append({"page": page, "source": PDF_PATH})

        print(f"✅ Built {len(inputs)} chunks from {PDF_PATH}")

        print("Generating embeddings with local model (this may take a while)...")
        vectors = embeddings_model.embed_documents(inputs)

        rows_to_insert = []
        for idx, (content, emb, meta) in enumerate(zip(inputs, vectors, metas)):
            rows_to_insert.append({
                "doc_id": DOC_ID,
                "chunk_index": idx,
                "content": content,
                "metadata": json.dumps(meta),
                "embedding": array.array("f", emb)
            })

        # --- MODIFIED: Use dynamic table name in the query ---
        print(f"Uploading to Oracle DB table '{TABLE_NAME}'...")
        insert_sql = f"""
            INSERT INTO {TABLE_NAME} (doc_id, chunk_index, content, metadata, embedding)
            VALUES (:doc_id, :chunk_index, :content, :metadata, :embedding)
        """
        for j in tqdm(range(0, len(rows_to_insert), BATCH_INSERT), desc="Uploading"):
            batch_rows = rows_to_insert[j:j + BATCH_INSERT]
            cursor.executemany(insert_sql, batch_rows)

        connection.commit()
        print(f"🎉 Done! Inserted {len(rows_to_insert)} chunks into {TABLE_NAME}.")

if __name__ == "__main__":
    main()