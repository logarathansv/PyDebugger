import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import time

load_dotenv()

AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_OPENAI")
AZURE_EMBEDDING_API = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

EMBEDDING_MODEL = AzureOpenAIEmbeddings(
    model=AZURE_EMBEDDING_DEPLOYMENT,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    openai_api_version="2023-05-15",
    api_key=AZURE_EMBEDDING_API,
)

# Azure OpenAI API Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

# Initialize Azure LLM
LANGUAGE_MODEL = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2025-01-01-preview",
    deployment_name=AZURE_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_KEY,
    temperature=0.2
)
# Dark Mode UI
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #3A3A3A !important; }
    .stChatMessage p, .stChatMessage div { color: #FFFFFF !important; }
    .stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; border-radius: 5px; padding: 15px; }
    h1, h2, h3 { color: #00FFAA !important; }
    </style>
    """, unsafe_allow_html=True)

# System Config
PDF_STORAGE_PATH = 'document_store/pdfs/'

# Store PDFs & Chat History Independently
if "pdf_vector_stores" not in st.session_state:
    st.session_state.pdf_vector_stores = {}

if "pdf_list" not in st.session_state:
    st.session_state.pdf_list = []

if "messages" not in st.session_state:
    st.session_state.messages = []  # FIX: Keeps messages even if PDFs are deleted

# Save PDF
def save_uploaded_file(uploaded_file):
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# Load & Chunk Documents
def process_document(file_name, file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    chunker = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = chunker.split_documents(docs)

    vector_store = InMemoryVectorStore(EMBEDDING_MODEL)
    vector_store.add_documents(chunks)
    st.session_state.pdf_vector_stores[file_name] = vector_store

    if file_name not in st.session_state.pdf_list:
        st.session_state.pdf_list.append(file_name)

# Search Relevant Docs
def find_context(query, selected_pdfs):
    context_docs = []
    for pdf in selected_pdfs:
        store = st.session_state.pdf_vector_stores.get(pdf)
        if store:
            context_docs.extend(store.similarity_search(query))
    return context_docs

# Generate AI Answer
def generate_answer(query, context_docs, mode):
    context = "\n\n".join([doc.page_content[:50] for doc in context_docs])  # More context for accuracy

    # Curriculum-based Assistant Prompt
    if mode == "Programming Tutor":
        prompt = """
        You are a programming tutor. Explain concepts clearly with examples and best practices.
        If unsure, say so. Keep explanations concise and beginner-friendly.
        
        Query: {query}
        Context: {context}
        Answer:
        """
    
    # Rubber Duck Debugging Prompt
    elif mode == "Rubber Duck Assistant":
        prompt = """
        You are a rubber duck debugging assistant. Help the user think through their code logically.
        Ask guiding questions rather than directly solving the problem.
        
        Query: {query}
        Context: {context}
        Answer:
        """

    conversation_prompt = ChatPromptTemplate.from_template(prompt)
    response_chain = conversation_prompt | LANGUAGE_MODEL

    time.sleep(20)
    return response_chain.invoke({"query": query, "context": context})

# UI Header
st.title("💡 Curriculum-Based Programming Chatbot & Debugging Assistant")
st.markdown("---")

# Sidebar - Upload PDFs
st.sidebar.header("📤 Upload Programming Resources")
uploaded_pdfs = st.sidebar.file_uploader("Upload PDFs (Textbooks, Docs, etc.)", type="pdf", accept_multiple_files=True)

# Process New PDFs
if uploaded_pdfs:
    for pdf in uploaded_pdfs:
        if pdf.name not in st.session_state.pdf_vector_stores:
            file_path = save_uploaded_file(pdf)
            process_document(pdf.name, file_path)
    st.sidebar.success("✅ Documents uploaded! Select them below.")

# Sidebar - Select PDFs (Checkbox)
st.sidebar.header("📑 Select PDFs")
selected_pdfs = [pdf for pdf in st.session_state.pdf_list if st.sidebar.checkbox(pdf, value=True)]

# Sidebar - Delete PDFs
st.sidebar.header("🗑️ Remove PDFs")
delete_pdf = st.sidebar.selectbox("Select PDF to Remove", ["None"] + st.session_state.pdf_list)

if st.sidebar.button("❌ Delete PDF") and delete_pdf != "None":
    del st.session_state.pdf_vector_stores[delete_pdf]
    st.session_state.pdf_list.remove(delete_pdf)
    st.sidebar.success(f"Deleted {delete_pdf}")

# Chatbot Mode Selection
mode = st.radio("Choose Assistant Mode:", ["Programming Tutor", "Rubber Duck Assistant"])

# Chat Section
if selected_pdfs:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about programming concepts or debugging...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("🔍 Searching resources..."):
            relevant_docs = find_context(user_query, selected_pdfs)
            ai_response = generate_answer(user_query, relevant_docs, mode)

        with st.chat_message("assistant"):
            st.markdown(ai_response.content)
            st.session_state.messages.append({"role": "assistant", "content": ai_response.content})

else:
    st.warning("⚠️ Please upload and select a document to start chatting.")