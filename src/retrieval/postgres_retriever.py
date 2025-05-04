from typing import List
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool

"""
🧠 <#> = cosine distance (requires pgvector installed via: CREATE EXTENSION vector;)
db_url = "postgresql://user:password@localhost:5432/your_db"
retriever = PostgresRetriever(db_url)
top_results = retriever.search(query_embedding, top_k=3)
"""
class PostgresRetriever():
    def __init__(self, db_url: str, table_name: str = "documents", vector_column: str = "embedding"):
        self.pool = SimpleConnectionPool(minconn=1, maxconn=5, dsn=db_url)
        self.table_name = table_name
        self.vector_column = vector_column

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT content, {self.vector_column} <#> %s AS distance
                    FROM {self.table_name}
                    ORDER BY {self.vector_column} <#> %s
                    LIMIT %s;
                    """,
                    (query_embedding, query_embedding, top_k)
                )
                results = cur.fetchall()
                return [row["content"] for row in results]
        finally:
            self.pool.putconn(conn)
