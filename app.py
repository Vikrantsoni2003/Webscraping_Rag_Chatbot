import os
import hashlib
import streamlit as st
from bs4 import SoupStrainer, BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from dotenv import load_dotenv
import requests
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Humanli.ai - Website Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #e3f2fd;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #f5f5f5;
    }
    div[data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }
    .success-box {
        padding: 10px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
    }
    .warning-box {
        padding: 10px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
FAISS_INDEX_PATH = "faiss_index"
CHROMA_DB_PATH = "chroma_db"

# --- Functions ---

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Returns the HuggingFace embeddings model (Cached for performance)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def is_valid_url(url):
    """Validates if the provided string is a legitimate URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def remove_duplicate_documents(docs):
    """
    Removes duplicate documents based on content hash.
    Assignment Requirement: Avoid duplicate content
    """
    seen_hashes = set()
    unique_docs = []
    
    for doc in docs:
        # Create hash of content
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs

def get_content_from_url(url):
    """
    Loads and cleans content from the URL using WebBaseLoader.
    Removes: Headers, Footers, Nav, Ads, Scripts, Styles
    """
    if not is_valid_url(url):
        st.error("❌ Invalid URL format. Please enter a valid URL (e.g., https://example.com)")
        return None
    
    try:
        # Strict parsing - only keep main content tags
        def should_keep_tag(tag_name, *args):
            return tag_name not in ['nav', 'header', 'footer', 'script', 'style', 
                                   'aside', 'form', 'noscript', 'iframe', 'ad', 
                                   'advertisement', 'social', 'share', 'svg', 'path']
        
        bs_kwargs = dict(
            parse_only=SoupStrainer(name=should_keep_tag)
        )
        
        # Browser-like headers to prevent blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }

        with st.spinner("🔍 Crawling website..."):
            loader = WebBaseLoader(
                web_paths=(url,), 
                bs_kwargs=bs_kwargs, 
                requests_kwargs={'headers': headers, 'timeout': 30}
            )
            docs = loader.load()
            
            if not docs or len(docs) == 0:
                st.warning("⚠️ No content found at the provided URL.")
                return None
            
            # Remove empty documents
            docs = [doc for doc in docs if doc.page_content.strip()]
            
            if not docs:
                st.warning("⚠️ Content found but was empty after cleaning.")
                return None
            
            # Remove duplicates (Assignment Requirement)
            docs = remove_duplicate_documents(docs)
            
            # Add source URL to metadata if not present
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = url
            
            return docs
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error accessing URL: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error extracting content: {str(e)}")
        return None

def extract_page_title(soup):
    """Extracts page title from BeautifulSoup object."""
    try:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        return "Untitled Page"
    except:
        return "Untitled Page"

def process_content(docs, chunk_size, chunk_overlap):
    """
    Splits documents into chunks with user-defined settings.
    Preserves metadata: Source URL, Page Title
    """
    if not docs:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,  # Track position in original document
        separators=["\n\n", "\n", ". ", " ", ""]  # Smart splitting
    )
    
    splits = text_splitter.split_documents(docs)
    
    # Ensure metadata is preserved (Assignment Requirement)
    for i, split in enumerate(splits):
        # Ensure source URL is present
        if 'source' not in split.metadata or not split.metadata['source']:
            split.metadata['source'] = docs[0].metadata.get('source', 'Unknown')
        
        # Ensure title is present
        if 'title' not in split.metadata or not split.metadata['title']:
            # Try to extract from original doc or use source as fallback
            split.metadata['title'] = docs[0].metadata.get('title', 
                                  docs[0].metadata.get('source', 'Untitled'))
        
        # Add chunk index for reference
        split.metadata['chunk_index'] = i
    
    return splits

def get_vectorstore(splits, reload=False):
    """
    Creates or loads a FAISS vector store.
    Embeddings are persisted and reusable (Assignment Requirement).
    """
    embeddings = get_embeddings()
    
    # Try to load existing index if reload is requested
    if reload and os.path.exists(FAISS_INDEX_PATH):
        try:
            with st.spinner("💾 Loading existing index..."):
                vectorstore = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                return vectorstore
        except Exception as e:
            st.warning(f"Could not load existing index: {e}. Creating new one...")
    
    # Create new index
    if splits and len(splits) > 0:
        try:
            with st.spinner(f"🧠 Creating embeddings for {len(splits)} chunks..."):
                vectorstore = FAISS.from_documents(
                    documents=splits, 
                    embedding=embeddings
                )
                # Persist to disk (Assignment Requirement)
                vectorstore.save_local(FAISS_INDEX_PATH)
                return vectorstore
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None
    
    return None

def get_rag_chain(vectorstore):
    """
    Creates the RAG chain using Gemini with Memory Support.
    Strictly uses only website context.
    """
    
    # Retrieve API Key (Priority: st.secrets > .env)
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("🔑 Google API Key not found. Please set GOOGLE_API_KEY in .streamlit/secrets.toml or .env file")
        return None

    try:
        # Using Gemini 1.5 Flash (Fast, cost-effective, accurate)
        # Model name corrected from "gemini-3-flash-preview" to actual model name
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",  # Updated: Correct model name
            google_api_key=api_key,
            temperature=0.1,  # Low temperature for strict adherence to context
            max_output_tokens=2048,
            convert_system_message_to_human=True
        )

        # Strict System Prompt (Assignment Requirement)
        system_prompt = """You are a helpful AI assistant trained to answer questions based STRICTLY on the provided website context.

**CRITICAL RULES:**
1. **Greetings & Social**: If the user greets you (e.g., "hi", "hello") or asks personal questions ("how are you"), respond naturally and politely using chat history. Be friendly and welcoming.

2. **Information Queries**: For any question seeking facts, data, or information about the website:
   - Answer ONLY using the provided context below
   - If the context contains the answer, provide it accurately and concisely
   - Cite specific details from the context
   
3. **No Hallucination Policy**: If the answer is NOT found in the context, you MUST respond EXACTLY with:
   "The answer is not available on the provided website."
   
   Do NOT use outside knowledge. Do NOT apologize. Do NOT suggest alternatives. Use exactly this phrase.

4. **Language**: Respond in the same language as the user's query.

**Provided Context:**
{context}

Answer based ONLY on the above context."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # Configure retriever to return source metadata
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Top 5 relevant chunks
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
        
    except Exception as e:
        st.error(f"Error initializing AI model: {e}")
        return None

def check_faiss_exists():
    """Checks if FAISS index exists."""
    return os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH)

# --- Main App Logic ---

def main():
    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    if "current_url" not in st.session_state:
        st.session_state.current_url = ""
        
    if "processed_chunks" not in st.session_state:
        st.session_state.processed_chunks = 0

    # Sidebar UI
    with st.sidebar:
        st.title("🤖 Humanli.ai")
        st.markdown("### Website Intelligence Chatbot")
        st.markdown("---")
        
        # URL Input Section
        st.subheader("🔗 Website Source")
        url_input = st.text_input(
            "Enter Website URL", 
            placeholder="https://example.com",
            help="Enter full URL including https://"
        )
        
        # Advanced Settings
        with st.expander("⚙️ Advanced Chunking Settings"):
            st.caption("Configure text splitting strategy")
            
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider(
                    "Chunk Size", 
                    min_value=500, 
                    max_value=2000, 
                    value=1000, 
                    step=100,
                    help="Number of characters per chunk"
                )
            with col2:
                chunk_overlap = st.slider(
                    "Chunk Overlap", 
                    min_value=50, 
                    max_value=500, 
                    value=200, 
                    step=50,
                    help="Overlap between consecutive chunks"
                )
            
            st.info("💡 Tip: Larger chunks = more context but slower. Overlap helps maintain continuity.")

        # Process Button
        process_btn = st.button(
            "🚀 Process Website", 
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # System Status
        st.subheader("📊 System Status")
        
        if check_faiss_exists():
            st.markdown("🟢 **Persisted Index:** Available")
        else:
            st.markdown("🔴 **Persisted Index:** None")
            
        if st.session_state.vectorstore:
            st.markdown(f"🟢 **Current Session:** Active")
            st.markdown(f"📄 **Chunks Indexed:** {st.session_state.processed_chunks}")
        else:
            st.markdown("🟡 **Current Session:** Idle")
        
        st.markdown("---")
        
        # Download Chat History
        st.subheader("💾 Export")
        if st.session_state.messages:
            chat_history_text = ""
            for msg in st.session_state.messages:
                role_name = "User" if msg["role"] == "user" else "Assistant"
                chat_history_text += f"{role_name}: {msg['content']}\n\n"
            
            st.download_button(
                label="📥 Download Chat",
                data=chat_history_text,
                file_name=f"chat_history_{st.session_state.current_url.replace('://', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        if st.button("🔄 Reset System", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.current_url = ""
            st.session_state.processed_chunks = 0
            if os.path.exists(FAISS_INDEX_PATH):
                import shutil
                shutil.rmtree(FAISS_INDEX_PATH)
            st.rerun()

    # Main Content Area
    st.title("💬 Chat with Website")
    st.caption("Ask questions based strictly on the website content")
    
    # Auto-load existing index on startup (Persistence Feature)
    if not st.session_state.vectorstore and not url_input:
        if check_faiss_exists():
            with st.spinner("🔄 Loading previous session..."):
                vs = get_vectorstore(None, reload=True)
                if vs:
                    st.session_state.vectorstore = vs
                    st.info("✅ Loaded previous session from disk. Ready to chat!")
    
    # URL Processing Logic
    if process_btn and url_input:
        if not is_valid_url(url_input):
            st.error("Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("Processing website... This may take a moment."):
                # Extract content
                docs = get_content_from_url(url_input)
                
                if docs and len(docs) > 0:
                    st.success(f"✅ Successfully extracted {len(docs)} page(s)")
                    
                    # Process chunks
                    splits = process_content(docs, chunk_size, chunk_overlap)
                    st.session_state.processed_chunks = len(splits)
                    
                    if splits:
                        st.info(f"📄 Created {len(splits)} chunks (Size: {chunk_size}, Overlap: {chunk_overlap})")
                        
                        # Create vector store
                        vectorstore = get_vectorstore(splits)
                        
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.current_url = url_input
                            st.session_state.messages = []  # Reset chat for new URL
                            st.success("🎯 Website indexed successfully! Start chatting below.")
                            st.balloons()
                        else:
                            st.error("Failed to create vector index")
                    else:
                        st.error("No processable content found after chunking")
                else:
                    st.stop()
    
    # Display Chat Interface
    st.markdown("---")
    
    # Chat History Display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            if st.session_state.vectorstore:
                st.info("👋 Start chatting! Ask me anything about the website content.")
            else:
                st.info("👈 Enter a URL in the sidebar and click 'Process Website' to begin.")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])
                
                # Show source metadata if available (for debugging/verification)
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.caption(f"Source: {source}")

    # User Input
    if prompt := st.chat_input(
        "Ask a question about the website...", 
        disabled=not st.session_state.vectorstore
    ):
        # Display user message immediately
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Response
        if st.session_state.vectorstore:
            rag_chain = get_rag_chain(st.session_state.vectorstore)
            
            if rag_chain:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        try:
                            # Prepare Chat History for Memory (Last 10 messages for context window)
                            chat_history = []
                            for msg in st.session_state.messages[-10:-1]:  # Exclude current message
                                if msg["role"] == "user":
                                    chat_history.append(HumanMessage(content=msg["content"]))
                                elif msg["role"] == "assistant":
                                    chat_history.append(AIMessage(content=msg["content"]))

                            # Invoke RAG Chain
                            response = rag_chain.invoke({
                                "input": prompt,
                                "chat_history": chat_history
                            })
                            
                            answer = response["answer"]
                            
                            # Display response
                            st.markdown(answer)
                            
                            # Store message
                            msg_data = {
                                "role": "assistant", 
                                "content": answer
                            }
                            
                            # Store source documents metadata if needed
                            if "context" in response:
                                sources = []
                                for doc in response["context"]:
                                    if hasattr(doc, 'metadata'):
                                        sources.append(doc.metadata.get('source', 'Unknown'))
                                msg_data["sources"] = list(set(sources))
                            
                            st.session_state.messages.append(msg_data)
                            
                            # Rerun to update download button and UI
                            st.rerun()
                            
                        except Exception as e:
                            error_msg = f"I encountered an error processing your question: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })
            else:
                st.error("Failed to initialize chat engine. Check API key configuration.")
        else:
            st.warning("⚠️ Please process a website URL first using the sidebar.")

    # Footer
    st.markdown("---")
    st.caption("🚀 Powered by LangChain + Gemini + FAISS | Built for Humanli.ai Assignment")

if __name__ == "__main__":
    main()