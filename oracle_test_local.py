# oracle_test_embedding.py  — test RAG against Oracle 23ai using bge-m3 (Ollama) and table 'chunks_bge'

import os, textwrap, array, numpy as np
import oracledb
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import OllamaEmbeddings

# ---- Load env (Oracle creds only; no OpenAI needed)
load_dotenv(find_dotenv(usecwd=True))
ORACLE_USER = os.environ["ORACLE_USER"]
ORACLE_PASSWORD = os.environ["ORACLE_PASSWORD"]
ORACLE_DSN = os.environ["ORACLE_DSN"]

# Must match ingest_oracle_local.py
TABLE_NAME = "chunks_bge"
EMBED_MODEL_NAME = "bge-m3"                 # ingest used bge-m3 (1024-dim)
PDF_PATH = "human-nutrition-text.pdf"       # used as a filter in metadata
TOP_K = 3

queries = [
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins",
    "What are micronutrients?"
]

def to_float32_vector(x):
    a = np.asarray(x, dtype=np.float32).ravel()
    return array.array('f', a.tolist())

def main():
    # Embeddings via local Ollama
    embedder = OllamaEmbeddings(model=EMBED_MODEL_NAME)

    # Connect to Oracle
    conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)

    for q in queries:
        # 1) Embed the query with the SAME model used for ingest (bge-m3 → 1024 dims)
        e = embedder.embed_query(q)            # list[float], length 1024
        qvec = to_float32_vector(e)            # -> array('f') for Oracle VECTOR bind

        # 2) Nearest-neighbor search (COSINE distance)
        sql = f"""
            SELECT doc_id, chunk_index, content, metadata,
                   VECTOR_DISTANCE(embedding, :qvec, COSINE) AS dist
            FROM {TABLE_NAME}
            WHERE metadata LIKE :pdf_filter         -- keep it simple; matches your ingest
            ORDER BY dist
            FETCH FIRST :k ROWS ONLY
        """

        rows = []
        with conn.cursor() as cur:
            cur.execute(sql, qvec=qvec, pdf_filter=f'%{PDF_PATH}%', k=TOP_K)
            for r in cur.fetchall():
                rows.append(r)

        print("\n" + "="*90)
        print(f"QUERY: {q}")
        if not rows:
            print("  (no matches)")
            continue

        for rank, r in enumerate(rows, start=1):
            doc_id, chunk_index, content, metadata, dist = r
            sim = 1.0 - float(dist)  # cosine distance -> similarity (higher is better)
            content_str = str(content) if content is not None else ""
            preview = textwrap.shorten(content_str.replace("\n", " "), width=160)
            print(f"  [{rank}] chunk_index={chunk_index}  sim={sim:.3f}")
            print(f"      {preview}")

if __name__ == "__main__":
    main()