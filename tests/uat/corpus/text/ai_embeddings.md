# Vector Embeddings and Semantic Search

Vector embeddings convert text, images, and other data into dense numerical representations in high-dimensional space. Similar items are mapped to nearby vectors, enabling semantic similarity search using cosine distance or dot product.

Popular embedding models include OpenAI's text-embedding-3, Cohere Embed, and open-source alternatives like E5, BGE, and Qwen3-VL-Embedding. Vision-language models produce embeddings that align text and images in the same vector space, enabling cross-modal retrieval.

Vector databases like LanceDB, Pinecone, Weaviate, and Milvus store and index these embeddings for fast approximate nearest neighbor (ANN) search. Combined with BM25 full-text search, this creates powerful hybrid retrieval systems.
