# test_oracle.py
import os, textwrap, array, numpy as np
import oracledb
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# ---- Load env
load_dotenv(find_dotenv(usecwd=True))
ORACLE_USER = os.environ["ORACLE_USER"]
ORACLE_PASSWORD = os.environ["ORACLE_PASSWORD"]
ORACLE_DSN = os.environ["ORACLE_DSN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

EMBED_MODEL = "text-embedding-3-small"  # must match what you ingested with
PDF_PATH = "human-nutrition-text.pdf"   # used as a filter
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
    return array.array("f", a.tolist())

def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    # connect to Oracle
    conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)

    for q in queries:
        # embed query
        e = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding
        qvec = to_float32_vector(e)

        # nearest neighbor search
        sql = f"""
            SELECT doc_id, chunk_index, content, metadata,
                   VECTOR_DISTANCE(embedding, :qvec, COSINE) AS dist
            FROM chunks
            WHERE metadata LIKE :pdf_filter
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
            sim = 1.0 - float(dist)  # cosine distance → similarity
            #preview = textwrap.shorten(content.replace("\n"," "), width=160)
            preview = textwrap.shorten(str(content).replace("\n", " "), width=160)
            print(f"  [{rank}] chunk_index={chunk_index}  sim={sim:.3f}")
            print(f"      {preview}")

if __name__ == "__main__":
    main()