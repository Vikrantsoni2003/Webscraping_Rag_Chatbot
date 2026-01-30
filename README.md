# 🤖 Humanli.ai – Website-Based Chatbot Using Embeddings

**Role:** AI/ML Engineer Assignment  
**Duration:** 2–3 Days  
**Company:** Humanli.ai  
**Status:** Production Ready ✅

## 🎯 Project Overview

An intelligent, **Retrieval-Augmented Generation (RAG)** chatbot that transforms static website content into interactive conversational intelligence. The system strictly adheres to website context, eliminating hallucinations while maintaining conversational memory.

**Key Capabilities:**
- 🌐 **URL Intelligence**: Extracts, cleans, and processes any public website
- 🧠 **Context-Aware**: Maintains short-term conversational memory
- 🚫 **Zero Hallucination**: Strictly answers only from provided website content
- 💾 **Persistent Storage**: Vector embeddings saved locally for instant reload
- ⚙️ **Configurable**: User-adjustable chunking parameters for optimal performance

---

## 🏗️ Architecture & Workflow

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  WebBaseLoader   │────▶│ BeautifulSoup4  │
│   (User Input)  │     │  (Crawling)      │     │ (HTML Cleaning) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
│
▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Gemini Flash  │◀────│  FAISS Vector    │◀────│ RecursiveText   │
│   (LLM Engine)  │     │  Store (Indexed) │     │ Splitter        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
│                                               │
│         ┌──────────────────┐                 │
└────────▶│  HuggingFace     │◀────────────────┘
│  Embeddings      │
│ (all-MiniLM-L6-v2)│
└──────────────────┘

### Detailed Pipeline:

1.  **Input & Validation**: URL validation with `urllib.parse`, anti-blocking headers
2.  **Smart Crawling**: `WebBaseLoader` + `SoupStrainer` filters out nav, footer, ads, scripts
3.  **Deduplication**: MD5 hash-based duplicate content removal
4.  **Semantic Chunking**: `RecursiveCharacterTextSplitter` with configurable size/overlap
5.  **Metadata Preservation**: Source URL and page titles retained in every chunk
6.  **Vectorization**: Local HuggingFace embeddings (no API latency)
7.  **Retrieval**: FAISS similarity search with top-k chunk retrieval
8.  **Generation**: Gemini with strict system prompt enforcing context-only answers

---

## 🛠️ Technology Stack & Justifications

### 1. Large Language Model (LLM)
*   **Model Used**: **Google Gemini Flash** (Specifically: `gemini-3-flash-preview` / `gemini-1.5-flash`)
*   **Justification**:
    *   **Speed**: Optimized for low-latency retrieval tasks (sub-second response)
    *   **Context Window**: 1M+ tokens support, allowing comprehensive website context
    *   **Cost**: Generous free tier suitable for prototyping and production demos
    *   **Quality**: Superior instruction-following for strict "context-only" constraints

### 2. Vector Database
*   **Database Used**: **FAISS (Facebook AI Similarity Search)**
*   **Justification**:
    *   **Local-First**: File-based persistence (`faiss_index/` directory) - no external dependencies
    *   **Performance**: C++ optimized similarity search on CPU
    *   **Memory Efficient**: Suitable for single-website indices (1000s of chunks)
    *   **LangChain Native**: Seamless `load_local()` / `save_local()` integration

### 3. Embedding Strategy
*   **Model Used**: **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`**
*   **Justification**:
    *   **Dimensionality**: 384-dimensional vectors (optimal balance of quality vs. storage)
    *   **Speed**: ~1000 chunks/second on CPU
    *   **Accuracy**: SOTA for semantic similarity on English text
    *   **Offline Capability**: Zero external API dependency for embeddings

### 4. Orchestration Framework
*   **Framework**: **LangChain** (v0.1.0+)
*   **Components Used**:
    *   `create_retrieval_chain` for RAG pipeline
    *   `MessagesPlaceholder` for conversational memory
    *   `RecursiveCharacterTextSplitter` for semantic chunking

---

## 📋 Setup & Run Instructions

### Prerequisites
*   Python 3.9+
*   Google API Key ([Get here](https://makersuite.google.com/app/apikey))

### Local Installation

1.  **Clone Repository**
    ```bash
    git clone <repository_url>
    cd humanli-chatbot
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup**
    Create `.env` file:
    ```bash
    GOOGLE_API_KEY=AIzaSy...
    ```
    
    Or for Streamlit Cloud, create `.streamlit/secrets.toml`:
    ```toml
    GOOGLE_API_KEY = "AIzaSy..."
    ```

4.  **Run Application**
    ```bash
    streamlit run app.py
    ```
    Access at: `http://localhost:8501`

### Streamlit Cloud Deployment
1. Push code to GitHub
2. Connect repo to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add `GOOGLE_API_KEY` in Secrets Management (Advanced Settings)
4. Deploy!

---

## 🎮 Usage Guide

### 1. Website Ingestion
- Enter URL in sidebar (e.g., `https://docs.streamlit.io`)
- Adjust **Chunk Size** (500-2000) and **Overlap** (50-500) in Advanced Settings
- Click **Process Website**
- Wait for "✅ Website indexed successfully!" confirmation

### 2. Chat Interface
- Type questions in natural language
- System maintains context across questions (short-term memory)
- Responses strictly cite website content only

### 3. Management Tools
- **Download Chat**: Export full conversation history as `.txt`
- **Clear Chat**: Reset conversation context (retains index)
- **Reset System**: Delete FAISS index and start fresh

---

## 🎯 Assignment Requirements Compliance

| Requirement | Implementation Status | Evidence |
|-------------|---------------------|----------|
| **URL Input** | ✅ Complete | Validation with `is_valid_url()`, error handling for 403/404 |
| **Content Extraction** | ✅ Complete | `SoupStrainer` filters nav, footer, ads, scripts |
| **Duplicate Removal** | ✅ Complete | MD5 hash deduplication in `remove_duplicate_documents()` |
| **Configurable Chunking** | ✅ Complete | Streamlit sliders for size/overlap (500-2000, 50-500) |
| **Metadata Preservation** | ✅ Complete | Source URL & page title stored in `doc.metadata` |
| **Embeddings** | ✅ Complete | HuggingFace `all-MiniLM-L6-v2` (local, fast) |
| **Vector Persistence** | ✅ Complete | FAISS `save_local()` / `load_local()` with `allow_dangerous_deserialization` |
| **Strict Q&A Logic** | ✅ Complete | System prompt enforces: *"The answer is not available on the provided website."* |
| **Short-term Memory** | ✅ Complete | `MessagesPlaceholder` with last 10 message context |
| **Streamlit UI** | ✅ Complete | Responsive layout, status indicators, download functionality |

---

## ⚠️ Assumptions, Limitations & Future Scope

### Assumptions
*   Target websites are **publicly accessible** (no auth walls)
*   Primary content is **text-based HTML** (not heavy JS SPAs)
*   **English language** content for optimal embedding performance
*   Single-session focus (one active website at a time)

### Current Limitations
1.  **JavaScript Rendering**: Basic HTTP scraping may miss content loaded via `React`, `Vue`, or `Angular` SPAs
2.  **Session Volatility**: Chat history resets on browser refresh (vector index persists)
3.  **Single Domain**: Optimized for one website at a time (multi-URL requires index reset)

### Future Improvements
1.  **Hybrid Search**: Combine FAISS (semantic) + BM25 (keyword) for better acronym recognition
2.  **Dynamic Crawling**: Integrate `Playwright` for JavaScript-heavy modern web apps
3.  **Source Attribution**: Highlight specific webpage sections in UI (context highlighting)
4.  **Multi-Modal**: Support for images/charts extraction and description
5.  **Long-term Memory**: Redis/Sqlite backend for persistent chat history across sessions

---

## 📁 Project Structure

humanli-chatbot/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore             # Excludes faiss_index/, .env
├── faiss_index/           # Auto-generated vector storage (gitignored)
└── README.md              # This file
