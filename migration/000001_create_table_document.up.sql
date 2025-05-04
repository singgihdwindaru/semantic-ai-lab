
CREATE EXTENSION IF NOT EXISTS vector;
CREATE INDEX IF NOT EXISTS idx_embedding_vector
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)  -- or whatever your model output size is
);
