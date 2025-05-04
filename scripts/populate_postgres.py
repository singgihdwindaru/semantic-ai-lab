import psycopg2
import psycopg2.extras
from datetime import datetime
from uuid import uuid4
from embeddings.encoder import TextEncoder  # adjust to your actual encoder
from typing import List, Dict

DB_URL = "postgresql://user:password@localhost:5432/your_db"
TABLE_NAME = "documents"

# documents = [
#     "The moon orbits the Earth.",
#     "Artificial intelligence simulates human cognition.",
#     "Water boils at 100 degrees Celsius.",
# ]
# Example source data with metadata
documents = [
    {
        "id": str(uuid4()),
        "title": "AI and Human Cognition",
        "date": "2023-11-01",
        "tags": ["AI", "cognition"],
        "content": "Artificial intelligence simulates human cognition processes. It is evolving rapidly."
    },
    {
        "id": str(uuid4()),
        "title": "Water Facts",
        "date": "2024-01-15",
        "tags": ["science", "chemistry"],
        "content": "Water boils at 100 degrees Celsius. It's a fundamental chemical compound."
    },
]

def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    # naive split by sentence, or upgrade to nltk/spacy for better segmentation
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def create_table_if_not_exists(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id UUID NOT NULL,
            chunk_index INT NOT NULL,
            title TEXT,
            date DATE,
            tags TEXT[],
            content TEXT NOT NULL,
            embedding vector(768),
            PRIMARY KEY (id, chunk_index)
        );
        """)
        conn.commit()

def insert_documents(conn, encoder: TextEncoder, docs: List[Dict]):
    with conn.cursor() as cur:
        rows = []
        for doc in docs:
            chunks = chunk_text(doc["content"])
            embeddings = encoder.encode(chunks)
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                rows.append((
                    doc["id"], idx, doc["title"], doc["date"], doc["tags"], chunk, emb
                ))

        psycopg2.extras.execute_batch(
            cur,
            f"""
            INSERT INTO {TABLE_NAME} (id, chunk_index, title, date, tags, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id, chunk_index)
            DO UPDATE SET
                title = EXCLUDED.title,
                date = EXCLUDED.date,
                tags = EXCLUDED.tags,
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding;
            """,
            rows
        )
        conn.commit()
def get_full_document(conn, doc_id: str):
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(f"""
            SELECT
                id,
                title,
                date,
                tags,
                string_agg(content, ' ' ORDER BY chunk_index) AS full_content
            FROM {TABLE_NAME}
            WHERE id = %s
            GROUP BY id, title, date, tags
        """, (doc_id,))
        return cur.fetchone()
def semantic_search(conn, query_embedding: List[float], k: int = 5):
    # Hybrid Search (Metadata + Vector)
#     SELECT *
# FROM documents
# WHERE tags && ARRAY['AI']::text[]
# ORDER BY embedding <=> %s
# LIMIT 5;
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(f"""
            SELECT
                id,
                title,
                date,
                tags,
                content,
                embedding <=> %s AS distance
            FROM {TABLE_NAME}
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, (query_embedding, query_embedding, k))
        return cur.fetchall()

def main():
    encoder = TextEncoder()
    conn = psycopg2.connect(DB_URL)
    try:
        create_table_if_not_exists(conn)
        insert_documents(conn, encoder, documents)
        print("✅ Documents inserted with chunking, metadata, and upserts.")

        # query = "What is artificial intelligence?"
        # query_embedding = encoder.encode([query])[0]
        # results = semantic_search(conn, query_embedding)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
