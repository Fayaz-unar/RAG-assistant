import streamlit as st
from groq import Groq

from src.document_loader import load_pdf
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.rag import retrieve_context, generate_answer
from src.study_tools import generate_summary, generate_questions


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="StudyAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #080d16;
    color: #e8edf5;
}

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background-color: #0b1220;
    border-right: 1px solid #1c2a3d;
}

.brand {
    font-size: 26px;
    font-weight: 800;
}

.brand span {
    color: #38bdf8;
}

.brand-sub {
    color: #718096;
    font-size: 13px;
}

.hero {
    padding: 25px 0 20px 0;
}

.hero-small {
    color: #38bdf8;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    line-height: 1.1;
    margin-top: 8px;
}

.hero-description {
    color: #8b98aa;
    font-size: 16px;
    margin-top: 12px;
    max-width: 700px;
    line-height: 1.6;
}

.tool-card {
    background: linear-gradient(145deg, #101a2b, #0c1422);
    border: 1px solid #1d3048;
    border-radius: 16px;
    padding: 22px;
    min-height: 145px;
}

.tool-icon {
    font-size: 28px;
}

.tool-title {
    font-size: 18px;
    font-weight: 700;
    margin-top: 10px;
}

.tool-description {
    color: #7f8da1;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 6px;
}

.stat-card {
    background-color: #0e1725;
    border: 1px solid #1c2b40;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}

.stat-number {
    color: #38bdf8;
    font-size: 24px;
    font-weight: 800;
}

.stat-label {
    color: #748196;
    font-size: 12px;
    margin-top: 4px;
}

.workspace {
    background-color: #0d1624;
    border: 1px solid #1c2b40;
    border-radius: 18px;
    padding: 24px;
    margin-top: 24px;
}

.workspace-title {
    font-size: 23px;
    font-weight: 700;
}

.workspace-description {
    color: #7e8b9e;
    font-size: 14px;
    margin-top: 5px;
    margin-bottom: 20px;
}

.source-card {
    background-color: #101a29;
    border: 1px solid #1c3048;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 8px;
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid #23415f;
    background-color: #102238;
    color: #dbeafe;
    font-weight: 600;
}

.stButton > button:hover {
    border-color: #38bdf8;
    color: #38bdf8;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# GROQ CLIENT
# ==================================================

@st.cache_resource
def create_groq_client():
    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# ==================================================
# SESSION STATE
# ==================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "tool" not in st.session_state:
    st.session_state.tool = "Chat"

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "questions" not in st.session_state:
    st.session_state.questions = ""


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("""
    <div class="brand">
        📚 Study<span>AI</span>
    </div>

    <div class="brand-sub">
        Your personal AI study assistant
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload your study material",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button(
            "⚡ Process Documents",
            use_container_width=True
        ):

            all_chunks = []
            document_names = []

            with st.spinner("Processing documents..."):

                for uploaded_file in uploaded_files:

                    document_names.append(
                        uploaded_file.name
                    )

                    pages = load_pdf(uploaded_file)

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

                with st.spinner(
                    "Building knowledge base..."
                ):

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
                    st.session_state.documents = document_names

                st.success(
                    f"{len(all_chunks)} chunks ready"
                )

            else:

                st.error(
                    "No readable text found in the PDF."
                )

    if st.session_state.documents:

        st.divider()

        st.markdown("### Your Files")

        for document in st.session_state.documents:

            st.caption(
                f"📄 {document}"
            )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==================================================
# HERO
# ==================================================

st.markdown("""
<div class="hero">

    <div class="hero-small">
        AI STUDY PLATFORM
    </div>

    <div class="hero-title">
        Study smarter.<br>
        Learn faster.
    </div>

    <div class="hero-description">
        Upload your study material and use AI to
        understand, summarize, and prepare for exams.
    </div>

</div>
""", unsafe_allow_html=True)


# ==================================================
# STUDY TOOLS
# ==================================================

st.markdown("### Choose your study tool")

st.caption(
    "Select a tool to work with your uploaded study material."
)

col1, col2, col3 = st.columns(3)


# --------------------------------------------------
# CHAT CARD
# --------------------------------------------------

with col1:

    st.markdown("""
    <div class="tool-card">

        <div class="tool-icon">💬</div>

        <div class="tool-title">
            Ask Questions
        </div>

        <div class="tool-description">
            Ask questions about your documents
            and get answers using RAG.
        </div>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Open Chat",
        key="open_chat",
        use_container_width=True
    ):

        st.session_state.tool = "Chat"
        st.rerun()


# --------------------------------------------------
# SUMMARY CARD
# --------------------------------------------------

with col2:

    st.markdown("""
    <div class="tool-card">

        <div class="tool-icon">📄</div>

        <div class="tool-title">
            Summarize Document
        </div>

        <div class="tool-description">
            Convert your study material into
            simple and exam-focused notes.
        </div>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Open Summarizer",
        key="open_summary",
        use_container_width=True
    ):

        st.session_state.tool = "Summarize Document"
        st.rerun()


# --------------------------------------------------
# QUESTIONS CARD
# --------------------------------------------------

with col3:

    st.markdown("""
    <div class="tool-card">

        <div class="tool-icon">📝</div>

        <div class="tool-title">
            Exam Questions
        </div>

        <div class="tool-description">
            Generate short and long questions
            from your uploaded study material.
        </div>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Open Generator",
        key="open_questions",
        use_container_width=True
    ):

        st.session_state.tool = "Generate Questions"
        st.rerun()


# ==================================================
# DOCUMENT CHECK
# ==================================================

if not st.session_state.chunks:

    st.markdown("""
    <div class="workspace">

        <div class="workspace-title">
            👋 Welcome to StudyAI
        </div>

        <div class="workspace-description">
            Upload a PDF from the sidebar and click
            "Process Documents" to start studying.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ==================================================
# STATISTICS
# ==================================================

st.markdown("<br>", unsafe_allow_html=True)

stat1, stat2, stat3 = st.columns(3)

with stat1:

    st.markdown(f"""
    <div class="stat-card">

        <div class="stat-number">
            {len(st.session_state.documents)}
        </div>

        <div class="stat-label">
            Documents
        </div>

    </div>
    """, unsafe_allow_html=True)


with stat2:

    st.markdown(f"""
    <div class="stat-card">

        <div class="stat-number">
            {len(st.session_state.chunks)}
        </div>

        <div class="stat-label">
            Knowledge Chunks
        </div>

    </div>
    """, unsafe_allow_html=True)


with stat3:

    st.markdown("""
    <div class="stat-card">

        <div class="stat-number">
            GPT-OSS 120B
        </div>

        <div class="stat-label">
            Powered by Groq
        </div>

    </div>
    """, unsafe_allow_html=True)


# ==================================================
# GROQ CLIENT
# ==================================================

client = create_groq_client()


# ==================================================
# MAIN WORKSPACE
# ==================================================

st.markdown(
    '<div class="workspace">',
    unsafe_allow_html=True
)


# ==================================================
# CHAT
# ==================================================

if st.session_state.tool == "Chat":

    st.markdown(
        '<div class="workspace-title">💬 Study Chat</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="workspace-description">'
        'Ask questions and explore your uploaded material.'
        '</div>',
        unsafe_allow_html=True
    )

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask something about your material..."
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
                "Searching your study material..."
            ):

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

                    with st.expander(
                        "📚 View Sources"
                    ):

                        for chunk in retrieved_chunks:

                            st.markdown(
                                f"""
                                <div class="source-card">
                                    📄 <b>{chunk["document"]}</b>
                                    &nbsp; • &nbsp;
                                    Page {chunk["page"]}
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


# ==================================================
# SUMMARY
# ==================================================

elif st.session_state.tool == "Summarize Document":

    st.markdown(
        '<div class="workspace-title">'
        '📄 Document Summarizer'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="workspace-description">'
        'Create concise and exam-friendly notes.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "✨ Generate Summary",
        key="generate_summary",
        use_container_width=True
    ):

        with st.spinner(
            "Creating your summary..."
        ):

            try:

                context = "\n\n".join(
                    chunk["text"]
                    for chunk in st.session_state.chunks
                )

                st.session_state.summary = generate_summary(
                    client,
                    context
                )

            except Exception as e:

                st.error(
                    f"Could not generate summary: {str(e)}"
                )

    if st.session_state.summary:

        st.markdown("### 📖 Summary")

        st.markdown(
            st.session_state.summary
        )


# ==================================================
# QUESTIONS
# ==================================================

elif st.session_state.tool == "Generate Questions":

    st.markdown(
        '<div class="workspace-title">'
        '📝 Exam Question Generator'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="workspace-description">'
        'Generate university-level questions from your material.'
        '</div>',
        unsafe_allow_html=True
    )

    qcol1, qcol2 = st.columns(2)

    with qcol1:

        question_type = st.selectbox(
            "Question Type",
            ["short", "long"],
            key="question_type"
        )

    with qcol2:

        number = st.number_input(
            "Number of Questions",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key="question_number"
        )

    if st.button(
        "✨ Generate Questions",
        key="generate_questions",
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

                st.session_state.questions = generate_questions(
                    client,
                    context,
                    question_type,
                    number
                )

            except Exception as e:

                st.error(
                    f"Could not generate questions: {str(e)}"
                )

    if st.session_state.questions:

        st.markdown(
            "### 📋 Generated Questions"
        )

        st.markdown(
            st.session_state.questions
        )


# ==================================================
# CLOSE WORKSPACE
# ==================================================

st.markdown(
    "</div>",
    unsafe_allow_html=True
)
