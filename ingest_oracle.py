import os
import uuid
import re
import json
import numpy as np  # --- MODIFIED: Import NumPy ---
import fitz  # PyMuPDF
import tiktoken
import oracledb
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
import array  # --- MODIFIED: Import array module ---

# ---- Load environment
load_dotenv(find_dotenv(usecwd=True))

# --- Use your existing Oracle environment variables ---
ORACLE_USER = os.environ["ORACLE_USER"]
ORACLE_PASSWORD = os.environ["ORACLE_PASSWORD"]
ORACLE_DSN = os.environ["ORACLE_DSN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# ---- Config (unchanged)
PDF_PATH = "human-nutrition-text.pdf"
DOC_ID = "nutrition-v1"
EMBED_MODEL = "text-embedding-3-small"
BATCH_EMBED = 100
BATCH_INSERT = 200

# Sentence chunking params (unchanged)
SENTS_PER_CHUNK = 10
SENT_OVERLAP = 2
MAX_TOKENS = 1300
MIN_TOKENS = 50

enc = tiktoken.get_encoding("cl100k_base")


# --- All helper functions (clean_text, split_sentences, etc.) remain unchanged ---
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


def chunk_page_by_sentences(text: str,
                            sents_per_chunk: int = SENTS_PER_CHUNK,
                            overlap: int = SENT_OVERLAP,
                            max_tokens: int = MAX_TOKENS,
                            min_tokens: int = MIN_TOKENS):
    sents = split_sentences(text)
    i = 0
    step = max(1, sents_per_chunk - overlap)
    while i < len(sents):
        piece = sents[i:i + sents_per_chunk]
        if not piece: break
        chunk = " ".join(piece)
        ids = enc.encode(chunk)
        while max_tokens and len(ids) > max_tokens and len(piece) > 1:
            piece = piece[:-1]
            chunk = " ".join(piece)
            ids = enc.encode(chunk)
        if len(ids) >= min_tokens:
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
    client = OpenAI(api_key=OPENAI_API_KEY)

    with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as connection:
        cursor = connection.cursor()
        print("✅ Connected to Oracle Database.")

        print(f"Deleting existing chunks for doc_id='{DOC_ID}'...")
        cursor.execute("DELETE FROM chunks WHERE doc_id = :doc_id", {"doc_id": DOC_ID})
        print(f"Deleted {cursor.rowcount} rows.")

        print("Reading PDF by pages...")
        pages = list(pdf_pages(PDF_PATH))

        inputs, metas = [], []
        print("Chunking (10 sentences per chunk, 2 overlap)...")
        for page, text in pages:
            if not text: continue
            for chunk in chunk_page_by_sentences(text):
                inputs.append(chunk)
                metas.append({"page": page, "source": PDF_PATH})

        print(f"✅ Built {len(inputs)} chunks from {PDF_PATH}")

        vectors = []
        print("Generating embeddings...")
        for i in tqdm(range(0, len(inputs), BATCH_EMBED), desc="Embedding"):
            batch = inputs[i:i + BATCH_EMBED]
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            vectors.extend([d.embedding for d in resp.data])

        rows_to_insert = []
        for idx, (content, emb, meta) in enumerate(zip(inputs, vectors, metas)):
            rows_to_insert.append({
                "doc_id": DOC_ID,
                "chunk_index": idx,
                "content": content,
                "metadata": json.dumps(meta),
                # --- MODIFIED: Use NumPy to create a float32 array ---
                #"embedding": np.array(emb, dtype=np.float32)
                "embedding": array.array('f', emb)
            })

        print("Uploading to Oracle DB...")
        for j in tqdm(range(0, len(rows_to_insert), BATCH_INSERT), desc="Uploading"):
            batch_rows = rows_to_insert[j:j + BATCH_INSERT]
            cursor.executemany("""
                INSERT INTO chunks (doc_id, chunk_index, content, metadata, embedding)
                VALUES (:doc_id, :chunk_index, :content, :metadata, :embedding)
            """, batch_rows)

        connection.commit()
        print(f"🎉 Done! Inserted {len(rows_to_insert)} chunks for doc_id={DOC_ID}")


if __name__ == "__main__":
    main()