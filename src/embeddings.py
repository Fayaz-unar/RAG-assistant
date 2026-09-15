from sentence_transformers import SentenceTransformer
import streamlit as st


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(texts):
    model = load_embedding_model()
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
