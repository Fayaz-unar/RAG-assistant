import faiss
import numpy as np


def create_vector_store(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index.add(embeddings)

    return index


def search_vector_store(index, query_embedding, top_k=5):
    query_embedding = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    return scores[0], indices[0]
