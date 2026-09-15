import streamlit as st
from groq import Groq

from src.document_loader import load_pdf
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.rag import retrieve_context, generate_answer


st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def create_groq_client():
    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


st.title("📚 RAG Study Assistant")

st.write(
    "Upload your study material and ask questions about it."
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Study Material")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("Process Documents"):

            all_chunks = []

            with st.spinner("Processing documents..."):

                for uploaded_file in uploaded_files:

                    pages = load_pdf(uploaded_file)

                    for page in pages:

                        text = page["text"]

                        # Basic chunking
                        chunk_size = 1000

                        for start in range(
                            0,
                            len(text),
                            chunk_size
                        ):

                            chunk_text = text[
                                start:start + chunk_size
                            ]

                            if chunk_text.strip():

                                all_chunks.append({
                                    "text": chunk_text.strip(),
                                    "page": page["page"],
                                    "document": uploaded_file.name
                                })

                if all_chunks:

                    texts = [
                        chunk["text"]
                        for chunk in all_chunks
                    ]

                    embeddings = create_embeddings(texts)

                    index = create_vector_store(
                        embeddings
                    )

                    st.session_state.chunks = all_chunks
                    st.session_state.vector_store = index

                    st.success(
                        f"Processed {len(all_chunks)} chunks."
                    )

                else:

                    st.error(
                        "No readable text was found."
                    )


# --------------------------------------------------
# Document Information
# --------------------------------------------------

if st.session_state.chunks:

    st.info(
        f"{len(st.session_state.chunks)} study chunks "
        "are currently available."
    )


# --------------------------------------------------
# Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

question = st.chat_input(
    "Ask something about your study material..."
)


if question:

    if not st.session_state.chunks:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                client = create_groq_client()

                retrieved_chunks = retrieve_context(
                    question,
                    st.session_state.vector_store,
                    st.session_state.chunks,
                    top_k=5
                )

                answer = generate_answer(
                    client,
                    question,
                    retrieved_chunks
                )

                st.markdown(answer)

                with st.expander("Sources"):

                    for chunk in retrieved_chunks:

                        st.write(
                            f"**{chunk['document']}** "
                            f"— Page {chunk['page']}"
                        )

                        st.caption(
                            chunk["text"][:500]
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )
