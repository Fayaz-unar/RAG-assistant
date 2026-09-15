```python
import streamlit as st
from groq import Groq

from src.document_loader import load_pdf
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.rag import retrieve_context, generate_answer
from src.study_tools import generate_summary, generate_questions


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.app-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.app-subtitle {
    font-size: 18px;
    opacity: 0.7;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

.source {
    padding: 10px;
    border-radius: 8px;
    border: 1px solid rgba(128,128,128,0.2);
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Groq Client
# --------------------------------------------------

@st.cache_resource
def create_groq_client():
    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
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
# Header
# --------------------------------------------------

st.markdown(
    '<div class="app-title">📚 RAG Study Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Your AI-powered study companion for documents, summaries, '
    'and exam preparation.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📂 Study Material")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button(
            "Process Documents",
            use_container_width=True
        ):

            all_chunks = []

            with st.spinner("Processing your study material..."):

                for uploaded_file in uploaded_files:

                    pages = load_pdf(uploaded_file)

                    for page in pages:

                        text = page["text"]

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
                        f"{len(all_chunks)} chunks processed."
                    )

                else:

                    st.error(
                        "No readable text was found."
                    )

    st.divider()

    st.header("🛠️ Study Tools")

    tool = st.radio(
        "Choose a tool",
        [
            "Chat",
            "Summarize Document",
            "Generate Questions"
        ]
    )

    st.divider()

    if st.session_state.chunks:

        st.metric(
            "Study Chunks",
            len(st.session_state.chunks)
        )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# Check Documents
# --------------------------------------------------

if not st.session_state.chunks:

    st.info(
        "👈 Upload your study material from the sidebar "
        "and click **Process Documents** to begin."
    )

    st.stop()


client = create_groq_client()


# --------------------------------------------------
# CHAT
# --------------------------------------------------

if tool == "Chat":

    st.subheader("💬 Ask Your Study Material")

    st.caption(
        "Ask questions about the documents you uploaded."
    )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching your study material..."):

                try:

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

                    with st.expander("📚 Sources"):

                        for chunk in retrieved_chunks:

                            st.markdown(
                                f"""
                                <div class="source">
                                <b>{chunk['document']}</b>
                                — Page {chunk['page']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(
                        f"Something went wrong: {str(e)}"
                    )


# --------------------------------------------------
# SUMMARIZE
# --------------------------------------------------

elif tool == "Summarize Document":

    st.subheader("📄 Document Summary")

    st.caption(
        "Generate an exam-friendly summary from your study material."
    )

    if st.button(
        "Generate Summary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing your study material..."
        ):

            try:

                # Limit context to avoid unnecessarily
                # huge requests.
                context = "\n\n".join(
                    chunk["text"]
                    for chunk in st.session_state.chunks
                )

                summary = generate_summary(
                    client,
                    context
                )

                st.markdown(summary)

            except Exception as e:

                st.error(
                    f"Could not generate summary: {str(e)}"
                )


# --------------------------------------------------
# GENERATE QUESTIONS
# --------------------------------------------------

elif tool == "Generate Questions":

    st.subheader("📝 Generate Exam Questions")

    st.caption(
        "Create questions based on your uploaded study material."
    )

    col1, col2 = st.columns(2)

    with col1:

        question_type = st.selectbox(
            "Question Type",
            [
                "short",
                "long"
            ]
        )

    with col2:

        number = st.number_input(
            "Number of Questions",
            min_value=1,
            max_value=20,
            value=10
        )

    if st.button(
        "Generate Questions",
        use_container_width=True
    ):

        with st.spinner(
            "Generating questions..."
        ):

            try:

                context = "\n\n".join(
                    chunk["text"]
                    for chunk in st.session_state.chunks
                )

                questions = generate_questions(
                    client,
                    context,
                    question_type,
                    number
                )

                st.markdown(questions)

            except Exception as e:

                st.error(
                    f"Could not generate questions: {str(e)}"
                )
```
