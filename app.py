"""
app.py
======
AI Resume Analyzer & Job Matcher - Streamlit application entry point.

Pipeline:
1. User uploads a PDF resume and pastes a job description.
2. `resume_parser` extracts structured info (name, email, skills, etc.).
3. `llm_handler` scores the resume against the job description via the
   Groq LLM and produces strengths/weaknesses/suggestions.
4. `vector_search` finds the most semantically similar job postings via
   ChromaDB + Sentence-Transformers.
5. Results are rendered in a clean, modern Streamlit UI.
"""

import os
import logging

import streamlit as st
from dotenv import load_dotenv

from utils.resume_parser import parse_resume
from utils.llm_handler import LLMHandler, LLMHandlerError
from utils.vector_search import VectorSearchEngine

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Resume Analyzer & Job Matcher",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom styling - keeps the UI feeling modern and "designed" rather than
# like a default Streamlit form.
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main-header {
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 1.6rem;
    }
    .main-header h1 { margin-bottom: 0.3rem; font-size: 2.1rem; }
    .main-header p { margin: 0; opacity: 0.92; font-size: 1.02rem; }

    .section-card {
        background: var(--background-color, #ffffff);
        border: 1px solid rgba(120, 120, 120, 0.15);
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1.1rem;
    }

    .skill-badge {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        border: 1px solid #c7d2fe;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        margin: 0.18rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .keyword-badge {
        display: inline-block;
        background: #fff7ed;
        color: #c2410c;
        border: 1px solid #fed7aa;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        margin: 0.18rem;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .job-card {
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        background: rgba(99, 102, 241, 0.04);
    }
    .job-card h4 { margin: 0 0 0.2rem 0; }
    .job-company { color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem; }

    .footer-note {
        text-align: center;
        color: #9ca3af;
        font-size: 0.82rem;
        margin-top: 2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached resource loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading semantic search engine...")
def get_vector_engine() -> VectorSearchEngine:
    """
    Initialize (once per server process) the ChromaDB-backed vector search
    engine and seed it with sample job postings.
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    engine = VectorSearchEngine(persist_directory=persist_dir)
    engine.seed_sample_jobs()
    return engine


def get_llm_handler():
    """
    Build an LLMHandler using the configured Groq API key.

    Returns:
        An LLMHandler instance, or None if the API key is missing/invalid
        (an error is shown to the user in that case).
    """
    try:
        return LLMHandler()
    except LLMHandlerError as e:
        st.error(f"⚠️ {e}")
        return None


def score_color(score: int) -> str:
    """Map a 0-100 match score to a semantic color for display."""
    if score >= 75:
        return "#16a34a"   # green
    if score >= 50:
        return "#d97706"   # amber
    return "#dc2626"       # red


def score_label(score: int) -> str:
    """Map a 0-100 match score to a human-friendly label."""
    if score >= 75:
        return "Strong Match"
    if score >= 50:
        return "Moderate Match"
    return "Needs Improvement"


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🧠 AI Resume Analyzer & Job Matcher</h1>
        <p>Parse your resume, score it against any job description, and discover
        the best-fitting roles - all powered by NLP, an LLM, and semantic search.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup & Info")

    api_key_present = bool(os.getenv("GROQ_API_KEY"))
    if api_key_present:
        st.success("Groq API key detected ✅")
    else:
        st.error("Groq API key missing ❌")
        st.caption(
            "Add `GROQ_API_KEY=your_key` to a `.env` file in the project "
            "root (see `.env.example`), then restart the app."
        )

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. **Upload** your resume (PDF)\n"
        "2. **Paste** a target job description\n"
        "3. Click **Analyze Resume**\n"
        "4. Review your **match score**, **feedback**, and **best-fit jobs**"
    )

    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown(
        "- 🐍 Python + Streamlit\n"
        "- 🧾 PyPDF2 + spaCy (parsing)\n"
        "- 🤖 Groq LLM (scoring)\n"
        "- 🔎 ChromaDB + Sentence-Transformers (matching)"
    )

    st.markdown("---")
    st.caption("Built with ❤️ as an end-to-end AI portfolio project.")

# --------------------------------------------------------------------------
# Main input area
# --------------------------------------------------------------------------
left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown("#### 📄 Upload Resume")
    uploaded_resume = st.file_uploader(
        "Upload a PDF resume",
        type=["pdf"],
        help="Only PDF files are supported. Max recommended size: 5 MB.",
        label_visibility="collapsed",
    )
    if uploaded_resume is not None:
        st.success(f"Uploaded: **{uploaded_resume.name}**")

with right_col:
    st.markdown("#### 📋 Paste Job Description")
    job_description = st.text_area(
        "Job description",
        height=200,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
    )

st.write("")
analyze_clicked = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)

# Persist results across reruns (e.g. when expanding sections) using session state.
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "parsed_resume" not in st.session_state:
    st.session_state.parsed_resume = None
if "job_matches" not in st.session_state:
    st.session_state.job_matches = None

# --------------------------------------------------------------------------
# Analysis pipeline
# --------------------------------------------------------------------------
if analyze_clicked:
    if uploaded_resume is None:
        st.warning("Please upload a PDF resume before analyzing.")
    elif not job_description.strip():
        st.warning("Please paste a job description before analyzing.")
    else:
        # Step 1: Parse the resume.
        try:
            with st.spinner("📄 Parsing resume..."):
                parsed = parse_resume(uploaded_resume)
            st.session_state.parsed_resume = parsed
        except ValueError as e:
            st.error(f"⚠️ {e}")
            st.session_state.parsed_resume = None
            st.session_state.analysis_result = None
            st.session_state.job_matches = None
            parsed = None

        # Step 2: Score against the job description via the LLM.
        if st.session_state.parsed_resume is not None:
            llm_handler = get_llm_handler()
            if llm_handler is not None:
                try:
                    with st.spinner("🤖 Scoring resume with AI..."):
                        result = llm_handler.analyze_resume(
                            parsed["raw_text"], job_description
                        )
                    st.session_state.analysis_result = result
                except LLMHandlerError as e:
                    st.error(f"⚠️ {e}")
                    st.session_state.analysis_result = None

            # Step 3: Semantic job matching via ChromaDB.
            try:
                with st.spinner("🔎 Finding matching job postings..."):
                    engine = get_vector_engine()
                    matches = engine.find_matching_jobs(parsed["raw_text"], top_k=3)
                st.session_state.job_matches = matches
            except RuntimeError as e:
                st.error(f"⚠️ {e}")
                st.session_state.job_matches = None

# --------------------------------------------------------------------------
# Results: Parsed Resume Info
# --------------------------------------------------------------------------
parsed = st.session_state.parsed_resume
if parsed is not None:
    st.markdown("### 🗂️ Parsed Resume Information")
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown(f"**Name:** {parsed['name'] or '_Not detected_'}")
            st.markdown(f"**Email:** {parsed['email'] or '_Not detected_'}")
            st.markdown(f"**Phone:** {parsed['phone'] or '_Not detected_'}")
            st.markdown(
                f"**Experience:** {parsed['experience_years'] or '_Not detected_'}"
            )

        with info_col2:
            st.markdown("**Education:**")
            if parsed["education"]:
                for edu in parsed["education"]:
                    st.markdown(f"- {edu}")
            else:
                st.markdown("_Not detected_")

        st.markdown("**Detected Skills:**")
        if parsed["skills"]:
            badges_html = "".join(
                f'<span class="skill-badge">{skill}</span>' for skill in parsed["skills"]
            )
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.markdown("_No skills from our vocabulary were detected._")

        with st.expander("View extracted raw text"):
            st.text_area(
                "Raw resume text", parsed["raw_text"], height=250, label_visibility="collapsed"
            )
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Results: Match Score & Feedback
# --------------------------------------------------------------------------
result = st.session_state.analysis_result
if result is not None:
    st.markdown("### 🎯 Match Score & Feedback")

    score = result["match_score"]
    color = score_color(score)
    label = score_label(score)

    score_col, summary_col = st.columns([1, 2])
    with score_col:
        st.markdown(
            f"""
            <div class="section-card" style="text-align:center;">
                <div style="font-size:3rem; font-weight:800; color:{color};">{score}</div>
                <div style="color:{color}; font-weight:600;">{label}</div>
                <div style="margin-top:0.6rem;">
            """,
            unsafe_allow_html=True,
        )
        st.progress(score / 100)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with summary_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Overall Assessment**")
        st.write(result["summary"])
        st.markdown("</div>", unsafe_allow_html=True)

    strength_col, weakness_col = st.columns(2)
    with strength_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### ✅ Strengths")
        for item in result["strengths"]:
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    with weakness_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Weaknesses")
        for item in result["weaknesses"]:
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 🔑 Missing Keywords")
    if result["missing_keywords"]:
        badges_html = "".join(
            f'<span class="keyword-badge">{kw}</span>' for kw in result["missing_keywords"]
        )
        st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.markdown("_No major missing keywords detected - nice work!_")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 💡 Actionable Suggestions")
    for i, suggestion in enumerate(result["suggestions"], start=1):
        st.checkbox(suggestion, key=f"suggestion_{i}")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Results: Job Matching (ChromaDB)
# --------------------------------------------------------------------------
job_matches = st.session_state.job_matches
if job_matches is not None:
    st.markdown("### 🧭 Best-Fit Job Postings (Semantic Search)")
    if not job_matches:
        st.info("No job postings are available to match against yet.")
    else:
        for job in job_matches:
            st.markdown(
                f"""
                <div class="job-card">
                    <h4>{job['title']}</h4>
                    <div class="job-company">🏢 {job['company']} &nbsp;•&nbsp;
                        <strong style="color:{score_color(job['similarity_score'])}">
                            {job['similarity_score']}% match
                        </strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(job["similarity_score"], 100) / 100)
            with st.expander("View job description"):
                st.write(job["description"])

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    '<div class="footer-note">AI Resume Analyzer & Job Matcher · '
    "Built with Streamlit, Groq, ChromaDB & Sentence-Transformers</div>",
    unsafe_allow_html=True,
)
