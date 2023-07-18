CREATE TABLE image_vectors
(
    id        bigserial PRIMARY KEY,
    img_path  varchar,
    embedding vector(1536)
);
