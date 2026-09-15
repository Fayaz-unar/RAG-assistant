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

st.markdown(
    """
    <style>

    /* ==============================================
       GLOBAL
       ============================================== */

    .stApp {
        background: #080d16;
        color: #e8edf5;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ==============================================
       SIDEBAR
       ============================================== */

    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1c2a3d;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem;
    }

    /* ==============================================
       BRAND
       ============================================== */

    .brand {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .brand span {
        color: #38bdf8;
    }

    .brand-sub {
        color: #718096;
        font-size: 13px;
        margin-top: 3px;
    }

    /* ==============================================
       HERO
       ============================================== */

    .hero {
        padding: 25px 0 15px 0;
    }

    .hero-small {
        color: #38bdf8;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -1.5px;
        margin: 0;
    }

    .hero-description {
        color: #8b98aa;
        font-size: 16px;
        margin-top: 12px;
        max-width: 700px;
        line-height: 1.6;
    }

    /* ==============================================
       TOOL CARDS
       ============================================== */

    .tool-card {
        background: linear-gradient(
            145deg,
            #101a2b,
            #0c1422
        );

        border: 1px solid #1d3048;
        border-radius: 16px;

        padding: 22px;

        height: 145px;

        transition: all 0.2s ease;
    }

    .tool-icon {
        font-size: 26px;
        margin-bottom: 12px;
    }

    .tool-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .tool-description {
        color: #7f8da1;
        font-size: 13px;
        line-height: 1.45;
    }

    /* ==============================================
       STATS
       ============================================== */

    .stat-card {
        background: #0e1725;
        border: 1px solid #1c2b40;
        border-radius: 14px;
        padding: 16px 18px;
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
        margin-top: 3px;
    }

    /* ==============================================
       WORKSPACE
       ============================================== */

    .workspace {
        background: #0d1624;
        border: 1px solid #1c2b40;
        border-radius: 18px;
        padding: 24px;
        margin-top: 22px;
    }

    .workspace-title {
        font-size: 23px;
        font-weight: 750;
    }

    .workspace-description {
        color: #7e8b9e;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* ==============================================
       SOURCES
       ============================================== */

    .source-card {
        background: #101a29;
        border: 1px solid #1c3048;
        border-radius: 10px;
        padding: 12px 15px;
        margin-bottom: 8px;
        color: #c8d2df;
    }

    /* ==============================================
       DIVIDER
       ============================================== */

    .divider {
        height: 1px;
        background: #1b2a3d;
        margin: 25px 0;
    }

    /* ==============================================
       BUTTONS
       ============================================== */

    .stButton > button {
        border-radius: 9px;
        border: 1px solid #23415f;
        background: #102238;
        color: #dbeafe;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #38bdf8;
        color: #38bdf8;
    }

    /* ==============================================
       CHAT
       ============================================== */

    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    /* ==============================================
       HIDE STREAMLIT BRANDING
       ============================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# GROQ
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


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            📚 Study<span>AI</span>
        </div>

        <div class="brand-sub">
            Your personal AI study assistant
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ------------------------------
    # Upload
    # ------------------------------

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

            with st.spinner(
                "Processing documents..."
            ):

                for uploaded_file in uploaded_files:

                    document_names.append(
                        uploaded_file.name
                    )

                    pages = load_pdf(
                        uploaded_file
                    )

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

                                all_chunks.append(
                                    {
                                        "text": chunk_text.strip(),
                                        "page": page["page"],
                                        "document": uploaded_file.name
                                    }
                                )

            if all_chunks:

                with st.spinner(
                    "Building knowledge base..."
                ):

                    texts = [
                        chunk["text"]
                        for chunk in all_chunks
                    ]

                    embeddings = create_embeddings(
                        texts
                    )

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
                    "No readable text found."
                )

    # ------------------------------
    # Documents
    # ------------------------------

    if st.session_state.documents:

        st.divider()

        st.markdown(
            "### Your Files"
        )

        for document in st.session_state.documents:

            st.caption(
                f"📄 {document}"
            )

    # ------------------------------
    # Clear
    # ------------------------------

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==================================================
# HERO
# ==================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-small">
            AI STUDY PLATFORM
        </div>

        <div class="hero-title">
            Study smarter.<br>
            Learn faster.
        </div>

        <div class="hero-description">
            Upload your study material and use AI to understand,
            summarize, and prepare for your exams.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# TOOL SELECTION
# ==================================================

st.markdown(
    "### Choose your study tool"
)

st.caption(
    "Select what you want to do with your study material."
)

tool_col1, tool_col2, tool_col3 = st.columns(3)


# CHAT TOOL

with tool_col1:

    st.markdown(
        """
        <div class="tool-card">

            <div class="tool-icon">💬</div>

            <div class="tool-title">
                Ask

