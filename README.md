# Oracle 23ai Vector Search with Python

This project demonstrates how to implement a Retrieval-Augmented Generation (RAG) pipeline using Python and Oracle Database 23ai's vector search capabilities. It includes scripts for ingesting documents (PDFs), generating vector embeddings using different models, and querying the database to find semantically similar text chunks.

The repository provides examples for connecting to both local/on-premise Oracle databases and Oracle Autonomous Database (ADB), as well as using embeddings from OpenAI or a local model via Ollama.

---

## Features

- **Document Ingestion**: Reads a PDF file, splits it into text chunks, and generates vector embeddings.
- **Multiple DB Support**:
  - `ingest_oracle.py` / `oracle_test_embedding.py`: Connects to a standard on-premise/local Oracle DB.
  - `ingest_oracle_adb.py` / `oracle_test_adb.py`: Connects to an Oracle Autonomous Database (ADB) using a wallet.
- **Multiple Embedding Models**:
  - **OpenAI**: Uses `text-embedding-3-small` for high-quality embeddings.
  - **Local Models**: Uses `bge-m3` via Ollama for a private, local-first approach.
- **Vector Search**: Performs cosine similarity searches on the stored vectors to retrieve relevant document chunks based on a query.

---

## Prerequisites

- Python 3.8+
- Oracle Database 23ai instance (local, on-premise, or ADB)
- For local embeddings: Ollama running with a downloaded model (e.g., `ollama pull bge-m3`)
- For ADB connection: Oracle Wallet files

---

## Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root of the project by copying the example below. Fill in your specific credentials.

```env
# For On-Premise/Local Oracle DB
ORACLE_USER="your_local_user"
ORACLE_PASSWORD="your_local_password"
ORACLE_DSN="localhost:1521/FREEPDB1"

# For OpenAI Embeddings
OPENAI_API_KEY="sk-..."

# For Oracle Autonomous Database (ADB)
ORACLE_USER_ADB="your_adb_user"
ORACLE_PASSWORD_ADB="your_adb_password"
ORACLE_DSN_ADB="your_adb_dsn_from_tnsnames"
TNS_ADMIN="/path/to/your/wallet_directory"
wallet_password="your_wallet_password"
```

---

## Database Table Setup
Use SQL_for_ORACLE_DB.txt to connect and create objects


## Add a PDF

Place the PDF file you want to ingest in the root directory and ensure the `PDF_PATH` variable in the scripts matches its filename (e.g., `human-nutrition-text.pdf`).
You can download file using https://www.google.com/search?q=http://pressbooks.oer.hawaii.edu/humannutrition/open/download%3Ftype%3Dpdf

---

## Usage

Choose the scripts that match your database and desired embedding model.

### 1. Ingest Data

Run an ingestion script to process the PDF and load the data into your Oracle database.

- **Local Oracle DB with OpenAI embeddings:**
  ```bash
  python ingest_oracle.py
  ```
- **Oracle ADB with OpenAI embeddings:**
  ```bash
  python ingest_oracle_adb.py
  ```
- **Local Oracle DB with local Ollama embeddings:**
  ```bash
  python ingest_oracle_local.py
  ```

### 2. Run Queries

After ingestion is complete, run the corresponding test script to perform vector similarity searches.

- **Query Local Oracle DB (OpenAI embeddings):**
  ```bash
  python oracle_test_embedding.py
  ```
- **Query Oracle ADB (OpenAI embeddings):**
  ```bash
  python oracle_test_adb.py
  ```
- **Query Local Oracle DB (local Ollama embeddings):**
  ```bash
  python oracle_test_local.py
  ```

---


