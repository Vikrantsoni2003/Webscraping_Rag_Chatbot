import os
import hashlib
import streamlit as st
from bs4 import BeautifulSoup
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
</style>
""", unsafe_allow_html=True)

# --- Constants ---
FAISS_INDEX_PATH = "faiss_index"

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
    """Removes duplicate documents based on content hash."""
    seen_hashes = set()
    unique_docs = []
    
    for doc in docs:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs

def get_content_from_url(url):
    """
    ROBUST content extraction - tries multiple methods
    """
    if not is_valid_url(url):
        st.error("❌ Invalid URL format. Please enter a valid URL (e.g., https://example.com)")
        return None
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }
    
    # Method 1: Try WebBaseLoader first (good for static sites)
    try:
        with st.spinner("🔍 Trying WebBaseLoader..."):
            # Simple tag filter - remove common noise
            loader = WebBaseLoader(
                web_paths=(url,), 
                requests_kwargs={'headers': headers, 'timeout': 15}
            )
            docs = loader.load()
            
            if docs and len(docs) > 0:
                # Clean loaded docs
                cleaned_docs = []
                for doc in docs:
                    if hasattr(doc, 'page_content') and doc.page_content.strip():
                        # Remove extra whitespace
                        doc.page_content = ' '.join(doc.page_content.split())
                        if 'source' not in doc.metadata:
                            doc.metadata['source'] = url
                        cleaned_docs.append(doc)
                
                if cleaned_docs:
                    return remove_duplicate_documents(cleaned_docs)
    except Exception as e:
        st.warning(f"WebBaseLoader failed: {str(e)[:100]}. Trying fallback...")
    
    # Method 2: Fallback to requests + BeautifulSoup (better for blocked sites)
    try:
        with st.spinner("🔍 Using fallback method..."):
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get title
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            
            # Remove unwanted tags completely
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 
                           'form', 'noscript', 'iframe', 'svg', 'canvas', 'ad']):
                tag.decompose()
            
            # Get text from body mainly
            body = soup.find('body')
            if body:
                text = body.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            if len(text) < 100:
                st.warning("⚠️ Website returned very little content. Might be JavaScript-heavy or protected.")
                return None
            
            doc = Document(
                page_content=text[:50000],  # Limit to 50k chars for safety
                metadata={"source": url, "title": title}
            )
            return [doc]
            
    except Exception as e:
        st.error(f"❌ Failed to extract content: {str(e)}")
        return None

def process_content(docs, chunk_size, chunk_overlap):
    """Splits documents into chunks with user-defined settings."""
    if not docs:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    splits = text_splitter.split_documents(docs)
    
    # Ensure metadata is preserved
    for i, split in enumerate(splits):
        if 'source' not in split.metadata:
            split.metadata['source'] = docs[0].metadata.get('source', 'Unknown')
        if 'title' not in split.metadata:
            split.metadata['title'] = docs[0].metadata.get('title', 'Untitled')
        split.metadata['chunk_index'] = i
    
    return splits

def get_vectorstore(splits, reload=False):
    """Creates or loads a FAISS vector store."""
    embeddings = get_embeddings()
    
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
            st.warning(f"Could not load existing index: {e}")
    
    if splits and len(splits) > 0:
        try:
            with st.spinner(f"🧠 Creating embeddings for {len(splits)} chunks..."):
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                vectorstore.save_local(FAISS_INDEX_PATH)
                return vectorstore
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None
    
    return None

def get_rag_chain(vectorstore):
    """Creates the RAG chain using Gemini with Memory Support."""
    
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("🔑 Google API Key not found")
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",  # Safe model name
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=2048
            convert_system_message_to_human=True
        )

        system_prompt = """You are a helpful AI assistant. Answer based STRICTLY on the provided context.

**RULES:**
1. If user greets you, respond naturally and politely.
2. For information queries, answer ONLY using the provided context.
3. If answer NOT in context, respond EXACTLY with: "The answer is not available on the provided website."
4. Do not use outside knowledge.

Context: {context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
        
    except Exception as e:
        st.error(f"Error initializing AI: {e}")
        return None

def check_faiss_exists():
    return os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH)

# --- Main App Logic ---

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "current_url" not in st.session_state:
        st.session_state.current_url = ""
    if "processed_chunks" not in st.session_state:
        st.session_state.processed_chunks = 0

    with st.sidebar:
        st.title("🤖 Humanli.ai")
        st.markdown("### Website Intelligence Chatbot")
        st.markdown("---")
        
        url_input = st.text_input("Enter Website URL", placeholder="https://example.com")
        
        with st.expander("⚙️ Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
            with col2:
                chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)

        process_btn = st.button("🚀 Process Website", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        if check_faiss_exists():
            st.markdown("🟢 **Persisted Index:** Available")
        else:
            st.markdown("🔴 **Persisted Index:** None")
            
        if st.session_state.vectorstore:
            st.markdown(f"🟢 **Session:** Active | **Chunks:** {st.session_state.processed_chunks}")
        
        st.markdown("---")
        
        if st.session_state.messages:
            chat_history_text = "\n\n".join([f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}" for m in st.session_state.messages])
            st.download_button("📥 Download Chat", chat_history_text, "chat_history.txt", use_container_width=True)
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        if st.button("🔄 Reset System", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            if os.path.exists(FAISS_INDEX_PATH):
                import shutil
                shutil.rmtree(FAISS_INDEX_PATH)
            st.rerun()

    st.title("💬 Chat with Website")
    
    # Auto-load existing index
    if not st.session_state.vectorstore and not url_input:
        if check_faiss_exists():
            with st.spinner("🔄 Loading previous session..."):
                vs = get_vectorstore(None, reload=True)
                if vs:
                    st.session_state.vectorstore = vs
                    st.info("✅ Loaded previous session")

    # URL Processing
    if process_btn and url_input:
        if not is_valid_url(url_input):
            st.error("Please enter a valid URL")
        else:
            with st.spinner("Processing..."):
                docs = get_content_from_url(url_input)
                
                if docs and len(docs) > 0:
                    st.success(f"✅ Extracted {len(docs)} document(s)")
                    
                    splits = process_content(docs, chunk_size, chunk_overlap)
                    st.session_state.processed_chunks = len(splits)
                    
                    if splits:
                        vectorstore = get_vectorstore(splits)
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.current_url = url_input
                            st.session_state.messages = []
                            st.success("🎯 Ready to chat!")
                            st.balloons()
    
    # Chat Interface
    if not st.session_state.messages and not st.session_state.vectorstore:
        st.info("👈 Enter URL and click 'Process Website'")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"]=="user" else "🤖"):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question...", disabled=not st.session_state.vectorstore):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.vectorstore:
            rag_chain = get_rag_chain(st.session_state.vectorstore)
            
            if rag_chain:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        try:
                            chat_history = []
                            for msg in st.session_state.messages[:-1]:
                                if msg["role"] == "user":
                                    chat_history.append(HumanMessage(content=msg["content"]))
                                else:
                                    chat_history.append(AIMessage(content=msg["content"]))

                            response = rag_chain.invoke({
                                "input": prompt,
                                "chat_history": chat_history
                            })
                            
                            answer = response["answer"]
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error: {e}")

    st.markdown("---")
    st.caption("🚀 Powered by LangChain + Gemini + FAISS")

if __name__ == "__main__":
    main()

