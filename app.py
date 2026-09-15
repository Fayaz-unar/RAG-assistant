import streamlit as st
from groq import Groq

from src.document_loader import load_pdf
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.rag import retrieve_context, generate_answer
from src.study_tools import generate_summary, generate_questions


st.set_page_config(
    page_title="StudyAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# CSS
# =========================

st.markdown("""
<style>

.stApp {
    background-color: #07111f;
    color: #e5e7eb;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background-color: #0b1626;
    border-right: 1px solid #1e334d;
}

.brand {
    font-size: 26px;
    font-weight: 800;
}

.brand span {
    color: #38bdf8;
}

.subtitle {
    color: #7f8da1;
    font-size: 13px;
}

.hero {
    padding: 25px 0;
}

.hero-label {
    color: #38bdf8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-top: 8px;
}

.hero-text {
    color: #8996a8;
    font-size: 15px;
    max-width: 650px;
    line-height: 1.6;
}

.tool-card {
    background-color: #0d1a2b;
    border: 1px solid #1d344e;
    border-radius: 16px;
    padding: 20px;
    min-height: 145px;
}

.tool-icon {
    font-size: 28px;
}

.tool-title {
    font-size: 18px;
    font-weight: 700;
    margin-top: 8px;
}

.tool-text {
    color: #7f8da1;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 6px;
}

.workspace {
    background-color: #0c1726;
    border: 1px solid #1d344e;
    border-radius: 16px;
    padding: 25px;
    margin-top: 25px;
}

.workspace-title {
    font-size: 23px;
    font-weight: 700;
}

.workspace-text {
    color: #7f8da1;
    font-size: 14px;
    margin-bottom: 20px;
}

.stat {
    background-color: #0d1a2b;
    border: 1px solid #1d344e;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}

.stat-number {
    color: #38bdf8;
    font-size: 22px;
    font-weight: 800;
}

.stat-label {
    color: #718096;
    font-size: 12px;
}

.source {
    background-color: #101f31;
    border: 1px solid #1d344e;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
}

.stButton > button {
    border-radius: 9px;
    background-color: #10243a;
    border: 1px solid #244766;
    color: #e5edf7;
    font-weight: 600;
}

.stButton > button:hover {
    border-color: #38bdf8;
    color: #38bdf8;
}

</style>
""", unsafe_allow_html=True)


# =========================
# GROQ
# =========================

@st.cache_resource
def get_client():
    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# =========================
# SESSION STATE
# =========================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tool" not in st.session_state:
    st.session_state.tool = "Chat"

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "questions" not in st.session_state:
    st.session_state.questions = ""


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.markdown(
        '<div class="brand">📚 Study<span>AI</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI study assistant</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 📂 Upload Material")

    files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if files:

        if st.button(
            "⚡ Process Documents",
            use_container_width=True
        ):

            chunks = []
            names = []

            with st.spinner("Reading documents..."):

                for file in files:

                    names.append(file.name)

                    pages = load_pdf(file)

                    for page in pages:

                        text = page["text"]

                        chunk_size = 1000
                        overlap = 150
                        step = chunk_size - overlap

                        for start in range(
                            0,
                            len(text),
                            step
                        ):

                            part = text[
                                start:start + chunk_size
                            ]

                            if part.strip():

                                chunks.append({
                                    "text": part.strip(),
                                    "page": page["page"],
                                    "document": file.name
                                })

            if not chunks:

                st.error(
                    "No readable text was found."
                )

            else:

                with st.spinner(
                    "Creating knowledge base..."
                ):

                    texts = [
                        item["text"]
                        for item in chunks
                    ]

                    embeddings = create_embeddings(
                        texts
                    )

                    index = create_vector_store(
                        embeddings
                    )

                st.session_state.chunks = chunks
                st.session_state.vector_store = index
                st.session_state.documents = names
                st.session_state.messages = []
                st.session_state.summary = ""
                st.session_state.questions = ""

                st.success(
                    f"{len(chunks)} chunks processed."
                )

    if st.session_state.documents:

        st.divider()

        st.markdown("### 📄 Your Documents")

        for name in st.session_state.documents:

            st.caption(
                "📄 " + name
            )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================
# HERO
# =========================

st.markdown("""
<div class="hero">

    <div class="hero-label">
        AI STUDY PLATFORM
    </div>

    <div class="hero-title">
        Study smarter. Learn faster.
    </div>

    <div class="hero-text">
        Upload your study material and use AI to
        ask questions, summarize documents, and
        generate exam questions.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================
# STUDY TOOLS
# =========================

st.markdown("### Study Tools")

st.caption(
    "Choose what you want to do with your study material."
)

c1, c2, c3 = st.columns(3)


with c1:

    st.markdown("""
    <div class="tool-card">

        <div class="tool-icon">💬</div>

        <div class="tool-title">
            Ask Questions
        </div>

        <div class="tool-text">
            Ask questions and get answers
            from your uploaded documents.
        </div>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Open Chat",
        key="chat_tool",
        use_container_width=True
    ):

        st.session_state.tool = "Chat"
        st.rerun()


with c2:

    st.markdown("""
    <div class="tool-card">

        <div class="tool-icon">📄</div>

        <div class="tool-title">
            Summarize Document
        </div>

        <div class="tool-text">
            Create simple and useful
            summaries for revision.
        </div>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Open Summarizer",
        key="summary_tool",
        use_container_width=True
    ):

        st.session_state.tool = "Summary"
        st.rerun()


with c3:

    st.markdown("""
    <div class="tool-card">

        <div class="tool-icon">📝</div>

        <div class="tool-title">
            Exam Questions
        </div>

        <div class="tool-text">
            Generate short or long questions
            from your study material.
        </div>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Open Generator",
        key="questions_tool",
        use_container_width=True
    ):

        st.session_state.tool = "Questions"
        st.rerun()


# =========================
# NO DOCUMENT
# =========================

if not st.session_state.chunks:

    st.markdown("""
    <div class="workspace">

        <div class="workspace-title">
            👋 Welcome to StudyAI
        </div>

        <div class="workspace-text">
            Upload a PDF from the sidebar and click
            "Process Documents" to begin.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.stop()


# =========================
# STATISTICS
# =========================

s1, s2, s3 = st.columns(3)


with s1:

    st.markdown(f"""
    <div class="stat">

        <div class="stat-number">
            {len(st.session_state.documents)}
        </div>

        <div class="stat-label">
            Documents
        </div>

    </div>
    """, unsafe_allow_html=True)


with s2:

    st.markdown(f"""
    <div class="stat">

        <div class="stat-number">
            {len(st.session_state.chunks)}
        </div>

        <div class="stat-label">
            Knowledge Chunks
        </div>

    </div>
    """, unsafe_allow_html=True)


with s3:

    st.markdown("""
    <div class="stat">

        <div class="stat-number">
            GPT-OSS 120B
        </div>

        <div class="stat-label">
            Groq AI
        </div>

    </div>
    """, unsafe_allow_html=True)


# =========================
# CLIENT
# =========================

client = get_client()


# =========================
# CHAT
# =========================

if st.session_state.tool == "Chat":

    st.markdown("""
    <div class="workspace">
        <div class="workspace-title">
            💬 Study Chat
        </div>

        <div class="workspace-text">
            Ask anything about your uploaded study material.
        </div>
    </div>
    """, unsafe_allow_html=True)

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask a question about your material..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching your material..."
            ):

                try:

                    retrieved = retrieve_context(
                        question,
                        st.session_state.vector_store,
                        st.session_state.chunks,
                        top_k=5
                    )

                    answer = generate_answer(
                        client,
                        question,
                        retrieved
                    )

                    st.markdown(answer)

                    with st.expander(
                        "📚 Sources"
                    ):

                        for item in retrieved:

                            st.markdown(
                                f"""
                                <div class="source">
                                    📄 <b>{item["document"]}</b>
                                    — Page {item["page"]}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:

                    st.error(
                        "Error: " + str(error)
                    )


# =========================
# SUMMARY
# =========================

elif st.session_state.tool == "Summary":

    st.markdown("""
    <div class="workspace">

        <div class="workspace-title">
            📄 Document Summarizer
        </div>

        <div class="workspace-text">
            Generate a clear summary of your uploaded material.
        </div>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "✨ Generate Summary",
        key="summary_generate",
        use_container_width=True
    ):

        context = "\n\n".join(
            item["text"]
            for item in st.session_state.chunks
        )

        with st.spinner(
            "Generating summary..."
        ):

            try:

                st.session_state.summary = generate_summary(
                    client,
                    context
                )

            except Exception as error:

                st.error(
                    "Error: " + str(error)
                )

    if st.session_state.summary:

        st.markdown("### 📖 Summary")

        st.markdown(
            st.session_state.summary
        )


# =========================
# QUESTIONS
# =========================

elif st.session_state.tool == "Questions":

    st.markdown("""
    <div class="workspace">

        <div class="workspace-title">
            📝 Exam Question Generator
        </div>

        <div class="workspace-text">
            Generate university-level questions
            from your uploaded material.
        </div>

    </div>
    """, unsafe_allow_html=True)

    q1, q2 = st.columns(2)

    with q1:

        qtype = st.selectbox(
            "Question Type",
            ["short", "long"],
            key="qtype"
        )

    with q2:

        amount = st.number_input(
            "Number of Questions",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key="amount"
        )

    if st.button(
        "✨ Generate Questions",
        key="questions_generate",
        use_container_width=True
    ):

        context = "\n\n".join(
            item["text"]
            for item in st.session_state.chunks
        )

        with st.spinner(
            "Generating questions..."
        ):

            try:

                st.session_state.questions = generate_questions(
                    client,
                    context,
                    qtype,
                    amount
                )

            except Exception as error:

                st.error(
                    "Error: " + str(error)
                )

    if st.session_state.questions:

        st.markdown(
            "### 📋 Generated Questions"
        )

        st.markdown(
            st.session_state.questions
        )
