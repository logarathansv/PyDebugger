import os
import streamlit as st
from langchain.schema import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import json
import time
from pptx import Presentation

# Initialize with deepseek-r1:1.5b model
OLLAMA_MODEL = "deepseek-r1:1.5b"

# Initialize LLM and Embeddings
llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

# UI Configuration
st.set_page_config(page_title="Learning Chatbot & Debugging Assistantt", layout="wide")
st.title("💡Learning Chatbot & Debugging Assistant")

# File handling
PDF_STORAGE_PATH = 'document_store/'
CHROMA_DB_PATH = 'chroma_db/'
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Dark Mode CSS
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #3A3A3A !important; }
    .stChatMessage p, .stChatMessage div { color: #FFFFFF !important; }
    .stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; border-radius: 5px; padding: 15px; }
    h1, h2, h3 { color: #00FFAA !important; }
    </style>
    """, unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_text_from_pptx(file_path):
    presentation = Presentation(file_path)
    return "\n".join([shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text")])

def process_document(file_path, file_name):
    file_ext = file_name.split(".")[-1].lower()
    
    if file_ext == "pdf":
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
    elif file_ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        docs = [Document(page_content=text)]
    elif file_ext == "pptx":
        text = extract_text_from_pptx(file_path)
        docs = [Document(page_content=text)]
    else:
        st.error("Unsupported file format")
        return None
    
    # Choose splitter based on file type
    if file_ext in ["md", "rst"]:
        splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=100)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    chunks = splitter.split_documents(docs)
    
    # Create vector store
    collection_name = file_name.replace(" ", "_").lower()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DB_PATH
    )
    
    return vector_store

def extract_youtube_transcript(video_url):
    try:
        video_id = parse_qs(urlparse(video_url).query).get("v", [None])[0]
        if not video_id:
            return None, "Invalid YouTube URL"
            
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = " ".join([entry['text'] for entry in transcript])
        
        # Save transcript
        file_name = f"youtube_{video_id}.txt"
        file_path = os.path.join(PDF_STORAGE_PATH, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        return file_path, None
    except Exception as e:
        return None, str(e)

def find_context(query, selected_docs, mode):
    context_docs = []
    citations = []

    for doc_name in selected_docs:
        vector_store = st.session_state.vector_stores.get(doc_name)
        if not vector_store:
            continue
        
        results = vector_store.similarity_search_with_score(query, k=2)
        
        for doc, score in results:
            if score > 0.7:  # Only include relevant results
                context_docs.append(doc)
                citations.append((doc_name, doc.metadata.get("page", "N/A")))
    
    return context_docs, citations

def generate_follow_up_hint(chat_history, mode):
    conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    if mode == "Programming Tutor":
        prompt = """You are a programming tutor. Provide another hint based on this conversation.
        Keep it concise and helpful.

        Conversation:
        {conversation_context}

        Hint:"""
    else:
        prompt = """You are a debugging assistant. Provide another guiding question.

        Conversation:
        {conversation_context}

        Hint:"""

    return llm(prompt.format(conversation_context=conversation_context))

def generate_answer(query, context_docs, citations, mode):
    context = "\n\n".join([doc.page_content[:500] for doc in context_docs])
    
    if mode == "Programming Tutor":
        prompt = """You are a programming tutor. Explain concepts clearly with examples.
        Use this context:
        {context}
        
        Question: {query}
        Answer:"""
    else:
        prompt = """You are a debugging assistant. Provide step-by-step guidance.
        Context:
        {context}
        
        Problem: {query}
        Guidance:"""

    answer = llm(prompt.format(context=context, query=query))
    
    # Add citations
    if citations:
        sources = "\n".join([f"- {pdf}" for pdf, _ in citations])
        answer += f"\n\nSources:\n{sources}"
    
    return answer

# Initialize session state
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
    
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📂 Document Management")
    
    # File upload
    uploaded_files = st.file_uploader("Upload documents", 
                                    type=["pdf", "txt", "pptx", "md"], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.vector_stores:
                file_path = save_uploaded_file(file)
                vector_store = process_document(file_path, file.name)
                if vector_store:
                    st.session_state.vector_stores[file.name] = vector_store
                    st.success(f"Processed {file.name}")
    
    # YouTube transcript
    st.header("🎬 YouTube Transcript")
    youtube_url = st.text_input("Enter YouTube URL")
    if st.button("Get Transcript") and youtube_url:
        file_path, error = extract_youtube_transcript(youtube_url)
        if error:
            st.error(f"Error: {error}")
        else:
            vector_store = process_document(file_path, f"youtube_{os.path.basename(file_path)}")
            if vector_store:
                st.session_state.vector_stores[f"youtube_{os.path.basename(file_path)}"] = vector_store
                st.success("Transcript processed!")
    
    # Document selection
    st.header("📑 Active Documents")
    selected_docs = []
    for doc_name in st.session_state.vector_stores.keys():
        if st.checkbox(doc_name, value=True):
            selected_docs.append(doc_name)

    # Assistant mode selection
    mode = st.radio("Assistant Mode:", ["Programming Tutor", "Debugging Assistant"])

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                context_docs, citations = find_context(prompt, selected_docs, mode)
                answer = generate_answer(prompt, context_docs, citations, mode)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Follow-up hint button
if mode == "Debugging Assistant" and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    if st.button("💡 Need another hint"):
        with st.spinner("Generating hint..."):
            try:
                hint = generate_follow_up_hint(st.session_state.messages, mode)
                st.session_state.messages.append({"role": "assistant", "content": hint})
                st.rerun()
            except Exception as e:
                st.error(f"Error generating hint: {str(e)}")