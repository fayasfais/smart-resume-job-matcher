# 🧠 AI Resume Analyzer & Job Matcher

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/LLM-Groq-orange?logo=lightning&logoColor=white" alt="Groq">
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-6e56cf" alt="ChromaDB">
  <img src="https://img.shields.io/badge/NLP-spaCy-09a3d5?logo=spacy&logoColor=white" alt="spaCy">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

<p align="center">
  An end-to-end AI tool that parses your resume, scores it against any job
  description using an LLM, and semantically matches you against multiple
  job postings using vector search — all wrapped in a clean Streamlit UI.
</p>

---

## ✨ Features

- 📄 **PDF Resume Parsing** — Extracts raw text and structured fields (name, email, phone, skills, education, years of experience) using PyPDF2 and spaCy NLP.
- 🤖 **LLM-Powered Scoring** — Sends your resume and a target job description to a Groq-hosted LLM, which returns a 0–100 match score, an overall assessment, strengths, weaknesses, and missing keywords.
- 💡 **Actionable Suggestions** — Specific, checkable recommendations for improving your resume (not generic advice).
- 🔎 **Semantic Job Matching** — Uses ChromaDB with Sentence-Transformers embeddings to find the most relevant job postings for your resume, ranked by similarity score.
- 🎨 **Modern Streamlit UI** — A polished, card-based interface with skill badges, color-coded scores, and progress indicators.
- 🛡️ **Robust Error Handling** — Graceful handling of corrupted/encrypted PDFs, missing API keys, malformed LLM responses, and empty inputs.

---

## 🏗️ Architecture

```text
                ┌─────────────────┐
   PDF Resume → │  resume_parser   │ → structured fields + raw text
                └─────────────────┘
                          │
                          ▼
┌──────────────┐   ┌─────────────┐   ┌──────────────────┐
│ Job Desc.    │ → │ llm_handler │ → │ score, strengths, │
│ (pasted)     │   │ (Groq LLM)  │   │ weaknesses, tips  │
└──────────────┘   └─────────────┘   └──────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ vector_search    │ → top-k matching job postings
                │ (ChromaDB +      │   (semantic similarity)
                │  Sentence-Trans.)│
                └─────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │   app.py (UI)    │ → renders everything in Streamlit
                └─────────────────┘
```

---

## 📁 Project Structure

```text
AI-Resume-Analyzer/
├── app.py                     # Streamlit UI & orchestration
├── requirements.txt           # Python dependencies
├── README.md
├── .gitignore
├── .env.example                # Template for required environment variables
├── utils/
│   ├── __init__.py
│   ├── resume_parser.py        # PDF text extraction + NLP field extraction
│   ├── llm_handler.py          # Groq LLM integration & JSON parsing
│   ├── vector_search.py        # ChromaDB semantic job matching
│   └── prompts.py              # Centralized LLM prompt templates
└── data/
    └── sample_resumes/         # Drop sample PDF resumes here for testing
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configure your environment variables

```bash
cp .env.example .env
```

Then open `.env` and add your free Groq API key (get one at
[console.groq.com/keys](https://console.groq.com/keys)):

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🖥️ Usage

1. Upload a PDF resume in the **Upload Resume** panel.
2. Paste the target **Job Description** in the text area.
3. Click **🚀 Analyze Resume**.
4. Review:
   - Parsed resume information (name, contact info, skills, education).
   - Your **match score** with a color-coded rating.
   - **Strengths**, **weaknesses**, and **missing keywords**.
   - A checklist of **actionable suggestions**.
   - The top semantically matched **job postings** from the built-in vector store.

---

## 📸 Screenshots

> Add your own screenshots/GIFs here once you run the app locally.

| Resume Upload & Job Description | Match Score & Feedback | Job Matching |
|---|---|---|
| `screenshots/upload.png` | `screenshots/score.png` | `screenshots/jobs.png` |

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyPDF2 |
| NLP | spaCy (`en_core_web_sm`) |
| LLM Scoring | Groq API (`llama-3.3-70b-versatile` by default) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | ChromaDB |
| Config | python-dotenv |

---

## 🔧 Configuration Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Your Groq API key |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model used for scoring |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | Disk path for persisting the vector store. Remove/unset to run fully in-memory |

---

## 🛣️ Future Improvements

- [ ] Support DOCX and TXT resume uploads in addition to PDF
- [ ] Allow users to upload/import their own job postings (CSV/JSON) into ChromaDB
- [ ] OCR fallback for scanned/image-based resumes
- [ ] Multi-resume batch analysis and side-by-side comparison
- [ ] Resume rewriting assistant that drafts improved bullet points directly
- [ ] Export the analysis report as a downloadable PDF
- [ ] User accounts with analysis history
- [ ] Swap in a larger skills taxonomy / fine-tuned NER model for extraction
- [ ] Deploy a hosted demo (Streamlit Community Cloud / Hugging Face Spaces)

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull
request for bug fixes, new features, or documentation improvements.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the
[LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for fast LLM inference
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Sentence-Transformers](https://www.sbert.net/) for embeddings
- [spaCy](https://spacy.io/) for NLP
- [Streamlit](https://streamlit.io/) for the UI framework
