import re

import streamlit as st
from groq import Groq

from src.document_loader import load_pdf
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.rag import retrieve_context, generate_answer
from src.study_tools import generate_summary, generate_questions


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="StudyAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HELPERS
# ============================================================

def clean_ai_output(text):
    """Prevent accidental HTML from AI output from appearing in the UI."""
    if text is None:
        return ""

    text = str(text)

    # Remove complete HTML tags.
    text = re.sub(r"<\s*/?\s*[a-zA-Z][^>]*>", "", text)

    # Remove common HTML entities.
    replacements = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def render_ai_text(text):
    """Render model output as Markdown, never as custom HTML."""
    cleaned = clean_ai_output(text)

    if cleaned:
        st.markdown(cleaned)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background-color: #07111f;
    color: #e7edf5;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background-color: #0b1626;
    border-right: 1px solid #1c3047;
}

section[data-testid="stSidebar"] * {
    color: #d7e0eb;
}

.brand {
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.brand-blue {
    color: #38bdf8;
}

.brand-sub {
    color: #7d8da2;
    font-size: 13px;
    margin-top: 4px;
}

.hero {
    padding: 25px 0 20px 0;
}

.hero-label {
    color: #38bdf8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    color: #f1f5f9;
    font-size: 42px;
    font-weight: 800;
    line-height: 1.15;
    margin-top: 8px;
}

.hero-text {
    color: #8795a8;
    font-size: 15px;
    line-height: 1.6;
    max-width: 680px;
    margin-top: 12px;
}

.section-title {
    color: #f1f5f9;
    font-size: 22px;
    font-weight: 700;
    margin-top: 10px;
}

.section-text {
    color: #7e8da2;
    font-size: 13px;
    margin-top: 4px;
    margin-bottom: 18px;
}

.tool-card {
    background-color: #0d1a2b;
    border: 1px solid #1c3853;
    border-radius: 16px;
    padding: 22px;
    min-height: 150px;
    margin-bottom: 10px;
}

.tool-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.tool-title {
    color: #edf4fb;
    font-size: 18px;
    font-weight: 700;
}

.tool-text {
    color: #7f8da1;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 7px;
}

.stat-card {
    background-color: #0d1a2b;
    border: 1px solid #1c3853;
    border-radius: 13px;
    padding: 16px;
    text-align: center;
    margin-top: 18px;
}

.stat-number {
    color: #38bdf8;
    font-size: 22px;
    font-weight: 800;
}

.stat-label {
    color: #718096;
    font-size: 12px;
    margin-top: 4px;
}

.workspace {
    background-color: #0c1726;
    border: 1px solid #1c3853;
    border-radius: 17px;
    padding: 25px;
    margin-top: 24px;
}

.workspace-title {
    color: #edf4fb;
    font-size: 23px;
    font-weight: 700;
}

.workspace-text {
    color: #7e8da2;
    font-size: 14px;
    line-height: 1.6;
    margin-top: 6px;
}

.source {
    background-color: #101f31;
    border: 1px solid #1d344c;
    border-radius: 9px;
    padding: 10px 12px;
    margin-bottom: 8px;
    color: #cbd5e1;
    font-size: 13px;
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid #244867;
    background-color: #10243a;
    color: #e2edf8;
    font-weight: 600;
    min-height: 42px;
}

.stButton > button:hover {
    border-color: #38bdf8;
    color: #38bdf8;
    background-color: #112a43;
}

div[data-testid="stFileUploader"] {
    background-color: #0d1a2b;
    border-radius: 10px;
}

[data-testid="stChatMessage"] {
    background-color: #0c1726;
    border-radius: 12px;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# GROQ
# ============================================================

@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "chunks": [],
    "vector_store": None,
    "documents": [],
    "messages": [],
    "tool": "Chat",
    "summary": "",
    "questions": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
<div class="brand">📚 Study<span class="brand-blue">AI</span></div>
<div class="brand-sub">Your personal AI study assistant</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 📂 Upload Material")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button(
            "⚡ Process Documents",
            use_container_width=True,
        ):
            all_chunks = []
            document_names = []

            with st.spinner("Reading your documents..."):
                for uploaded_file in uploaded_files:
                    document_names.append(uploaded_file.name)

                    pages = load_pdf(uploaded_file)

                    for page in pages:
                        text = page.get("text", "")

                        chunk_size = 1000
                        overlap = 150
                        step = chunk_size - overlap

                        for start in range(0, len(text), step):
                            chunk_text = text[start:start + chunk_size]

                            if chunk_text.strip():
                                all_chunks.append(
                                    {
                                        "text": chunk_text.strip(),
                                        "page": page["page"],
                                        "document": uploaded_file.name,
                                    }
                                )

            if not all_chunks:
                st.error("No readable text was found in the uploaded PDF.")
            else:
                with st.spinner("Building AI knowledge base..."):
                    texts = [chunk["text"] for chunk in all_chunks]
                    embeddings = create_embeddings(texts)
                    index = create_vector_store(embeddings)

                st.session_state.chunks = all_chunks
                st.session_state.vector_store = index
                st.session_state.documents = document_names
                st.session_state.messages = []
                st.session_state.summary = ""
                st.session_state.questions = ""

                st.success(
                    f"{len(all_chunks)} knowledge chunks ready."
                )

    if st.session_state.documents:
        st.divider()
        st.markdown("### 📄 Your Documents")

        for document in st.session_state.documents:
            st.caption("📄 " + document)

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-label">AI STUDY PLATFORM</div>
    <div class="hero-title">Study smarter.<br>Learn faster.</div>
    <div class="hero-text">
        Upload your study material and use AI to understand
        concepts, create summaries, and prepare for exams.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# STUDY TOOLS
# ============================================================

st.markdown(
    '<div class="section-title">Study Tools</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-text">Choose a tool to start studying.</div>',
    unsafe_allow_html=True,
)

tool1, tool2, tool3 = st.columns(3)


with tool1:
    st.markdown(
        """
<div class="tool-card">
    <div class="tool-icon">💬</div>
    <div class="tool-title">Ask Questions</div>
    <div class="tool-text">
        Ask questions about your documents and get answers using retrieval augmented generation.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button(
        "Open Chat",
        key="open_chat",
        use_container_width=True,
    ):
        st.session_state.tool = "Chat"
        st.rerun()


with tool2:
    st.markdown(
        """
<div class="tool-card">
    <div class="tool-icon">📄</div>
    <div class="tool-title">Summarize Document</div>
    <div class="tool-text">
        Turn lengthy study material into clear, structured and exam-friendly notes.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button(
        "Open Summarizer",
        key="open_summary",
        use_container_width=True,
    ):
        st.session_state.tool = "Summary"
        st.rerun()


with tool3:
    st.markdown(
        """
<div class="tool-card">
    <div class="tool-icon">📝</div>
    <div class="tool-title">Exam Questions</div>
    <div class="tool-text">
        Generate short and long questions directly from your uploaded study material.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button(
        "Open Generator",
        key="open_questions",
        use_container_width=True,
    ):
        st.session_state.tool = "Questions"
        st.rerun()


# ============================================================
# NO DOCUMENTS
# ============================================================

if not st.session_state.chunks:
    st.markdown(
        """
<div class="workspace">
    <div class="workspace-title">👋 Welcome to StudyAI</div>
    <div class="workspace-text">
        Your study workspace is ready. Upload a PDF from the sidebar and click "Process Documents" to begin.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# STATISTICS
# ============================================================

stat1, stat2, stat3 = st.columns(3)

with stat1:
    st.markdown(
        f"""
<div class="stat-card">
    <div class="stat-number">{len(st.session_state.documents)}</div>
    <div class="stat-label">Documents</div>
</div>
""",
        unsafe_allow_html=True,
    )

with stat2:
    st.markdown(
        f"""
<div class="stat-card">
    <div class="stat-number">{len(st.session_state.chunks)}</div>
    <div class="stat-label">Knowledge Chunks</div>
</div>
""",
        unsafe_allow_html=True,
    )

with stat3:
    st.markdown(
        """
<div class="stat-card">
    <div class="stat-number">GPT-OSS 120B</div>
    <div class="stat-label">Powered by Groq</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# GROQ CLIENT
# ============================================================

try:
    client = get_client()
except Exception as error:
    st.error(
        "Groq API key is missing or invalid. "
        "Please add GROQ_API_KEY to Streamlit secrets."
    )
    st.stop()


# ============================================================
# CHAT
# ============================================================

if st.session_state.tool == "Chat":
    st.markdown(
        """
<div class="workspace">
    <div class="workspace-title">💬 Study Chat</div>
    <div class="workspace-text">Ask questions and explore your uploaded material.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_ai_text(message["content"])

    question = st.chat_input(
        "Ask something about your study material..."
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your study material..."):
                try:
                    retrieved_chunks = retrieve_context(
                        question,
                        st.session_state.vector_store,
                        st.session_state.chunks,
                        top_k=5,
                    )

                    answer = generate_answer(
                        client,
                        question,
                        retrieved_chunks,
                    )

                    render_ai_text(answer)

                    with st.expander("📚 View Sources"):
                        for chunk in retrieved_chunks:
                            document = str(chunk.get("document", "Unknown"))
                            page = str(chunk.get("page", "?"))

                            st.markdown(
                                f"""
<div class="source">
    📄 <b>{document}</b> &nbsp; • &nbsp; Page {page}
</div>
""",
                                unsafe_allow_html=True,
                            )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": clean_ai_output(answer),
                        }
                    )

                except Exception as error:
                    st.error(
                        "Something went wrong: " + str(error)
                    )


# ============================================================
# SUMMARY
# ============================================================

elif st.session_state.tool == "Summary":
    st.markdown(
        """
<div class="workspace">
    <div class="workspace-title">📄 Document Summarizer</div>
    <div class="workspace-text">
        Generate a clear and exam-focused summary from your uploaded material.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button(
        "✨ Generate Summary",
        key="generate_summary",
        use_container_width=True,
    ):
        context = "\n\n".join(
            chunk["text"]
            for chunk in st.session_state.chunks
        )

        with st.spinner("Creating your summary..."):
            try:
                st.session_state.summary = generate_summary(
                    client,
                    context,
                )
            except Exception as error:
                st.error(
                    "Could not generate summary: " + str(error)
                )

    if st.session_state.summary:
        st.markdown("### 📖 Summary")
        render_ai_text(st.session_state.summary)


# ============================================================
# QUESTIONS
# ============================================================

elif st.session_state.tool == "Questions":
    st.markdown(
        """
<div class="workspace">
    <div class="workspace-title">📝 Exam Question Generator</div>
    <div class="workspace-text">
        Generate university-level questions from your uploaded study material.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    q1, q2 = st.columns(2)

    with q1:
        question_type = st.selectbox(
            "Question Type",
            ["short", "long"],
            key="question_type",
        )

    with q2:
        number = st.number_input(
            "Number of Questions",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key="question_number",
        )

    if st.button(
        "✨ Generate Questions",
        key="generate_questions",
        use_container_width=True,
    ):
        context = "\n\n".join(
            chunk["text"]
            for chunk in st.session_state.chunks
        )

        with st.spinner("Generating questions..."):
            try:
                st.session_state.questions = generate_questions(
                    client,
                    context,
                    question_type,
                    number,
                )
            except Exception as error:
                st.error(
                    "Could not generate questions: " + str(error)
                )

    if st.session_state.questions:
        st.markdown("### 📋 Generated Questions")
        render_ai_text(st.session_state.questions)
