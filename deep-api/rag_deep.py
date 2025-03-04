# import os
# import streamlit as st
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_ollama import OllamaEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM

# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #0E1117;
#         color: #FFFFFF;
#     }
    
#     /* Chat Input Styling */
#     .stChatInput input {
#         background-color: #1E1E1E !important;
#         color: #FFFFFF !important;
#         border: 1px solid #3A3A3A !important;
#     }
    
#     /* User Message Styling */
#     .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
#         background-color: #1E1E1E !important;
#         border: 1px solid #3A3A3A !important;
#         color: #E0E0E0 !important;
#         border-radius: 10px;
#         padding: 15px;
#         margin: 10px 0;
#     }
    
#     /* Assistant Message Styling */
#     .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
#         background-color: #2A2A2A !important;
#         border: 1px solid #404040 !important;
#         color: #F0F0F0 !important;
#         border-radius: 10px;
#         padding: 15px;
#         margin: 10px 0;
#     }
    
#     /* Avatar Styling */
#     .stChatMessage .avatar {
#         background-color: #00FFAA !important;
#         color: #000000 !important;
#     }
    
#     /* Text Color Fix */
#     .stChatMessage p, .stChatMessage div {
#         color: #FFFFFF !important;
#     }
    
#     .stFileUploader {
#         background-color: #1E1E1E;
#         border: 1px solid #3A3A3A;
#         border-radius: 5px;
#         padding: 15px;
#     }
    
#     h1, h2, h3 {
#         color: #00FFAA !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# PROMPT_TEMPLATE = """
# You are an expert research assistant. Use the provided context to answer the query. 
# If unsure, state that you don't know. Be concise and factual (max 3 sentences).

# Query: {user_query} 
# Context: {document_context} 
# Answer:
# """
# PDF_STORAGE_PATH = 'document_store/pdfs/'
# EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
# DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
# LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")


# def save_uploaded_file(uploaded_file):
#     # Ensure the directory exists
#     os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    
#     file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
#     with open(file_path, "wb") as file:
#         file.write(uploaded_file.getbuffer())
#     return file_path

# def load_pdf_documents(file_path):
#     document_loader = PDFPlumberLoader(file_path)
#     return document_loader.load()

# def chunk_documents(raw_documents):
#     text_processor = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         add_start_index=True
#     )
#     return text_processor.split_documents(raw_documents)

# def index_documents(document_chunks):
#     DOCUMENT_VECTOR_DB.add_documents(document_chunks)

# def find_related_documents(query):
#     return DOCUMENT_VECTOR_DB.similarity_search(query)

# def generate_answer(user_query, context_documents):
#     context_text = "\n\n".join([doc.page_content for doc in context_documents])
#     conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     response_chain = conversation_prompt | LANGUAGE_MODEL
#     return response_chain.invoke({"user_query": user_query, "document_context": context_text})


# # UI Configuration


# st.title("📘 DocuMind AI")
# st.markdown("### Your Intelligent Document Assistant")
# st.markdown("---")

# # File Upload Section
# uploaded_pdf = st.file_uploader(
#     "Upload Research Document (PDF)",
#     type="pdf",
#     help="Select a PDF document for analysis",
#     accept_multiple_files=False

# )

# if uploaded_pdf:
#     saved_path = save_uploaded_file(uploaded_pdf)
#     raw_docs = load_pdf_documents(saved_path)
#     processed_chunks = chunk_documents(raw_docs)
#     index_documents(processed_chunks)
    
#     st.success("✅ Document processed successfully! Ask your questions below.")
    
#     user_input = st.chat_input("Enter your question about the document...")
    
#     if user_input:
#         with st.chat_message("user"):
#             st.write(user_input)
        
#         with st.spinner("Analyzing document..."):
#             relevant_docs = find_related_documents(user_input)
#             ai_response = generate_answer(user_input, relevant_docs)
            
#         with st.chat_message("assistant", avatar="🤖"):
#             st.write(ai_response)

#--------------------------------------------------------------
# import os
# import streamlit as st
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_ollama import OllamaEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM

# # Custom Styling for Dark Theme
# st.markdown("""
#     <style>
#     .stApp { background-color: #0E1117; color: #FFFFFF; }
#     .stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #3A3A3A !important; }
#     .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #1E1E1E !important; border: 1px solid #3A3A3A !important; color: #E0E0E0 !important; border-radius: 10px; padding: 15px; margin: 10px 0; }
#     .stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #2A2A2A !important; border: 1px solid #404040 !important; color: #F0F0F0 !important; border-radius: 10px; padding: 15px; margin: 10px 0; }
#     .stChatMessage p, .stChatMessage div { color: #FFFFFF !important; }
#     .stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; border-radius: 5px; padding: 15px; }
#     h1, h2, h3 { color: #00FFAA !important; }
#     </style>
#     """, unsafe_allow_html=True)

# # Constants
# PROMPT_TEMPLATE = """
# You are an expert research assistant. Use the provided context to answer the query.
# If unsure, state that you don't know. Be concise and factual (max 3 sentences).

# Query: {user_query}
# Context: {document_context}
# Answer:
# """
# PDF_STORAGE_PATH = 'document_store/pdfs/'
# EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
# DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
# LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# # Function to Save PDF
# def save_uploaded_file(uploaded_file):
#     os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
#     file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
#     with open(file_path, "wb") as file:
#         file.write(uploaded_file.getbuffer())
#     return file_path

# # Function to Extract Text from PDF
# def load_pdf_documents(file_path):
#     document_loader = PDFPlumberLoader(file_path)
#     return document_loader.load()

# # Function to Chunk Documents
# def chunk_documents(raw_documents):
#     text_processor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
#     return text_processor.split_documents(raw_documents)

# # Function to Index Documents
# def index_documents(document_chunks):
#     DOCUMENT_VECTOR_DB.add_documents(document_chunks)

# # Function to Search Relevant Documents
# def find_related_documents(query):
#     return DOCUMENT_VECTOR_DB.similarity_search(query)

# # Function to Generate AI Answer
# def generate_answer(user_query, context_documents):
#     context_text = "\n\n".join([doc.page_content for doc in context_documents])
#     conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     response_chain = conversation_prompt | LANGUAGE_MODEL
#     return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# # UI Configuration
# st.title("📘 DocuMind AI - Chat with Your Documents")
# st.markdown("---")

# # Sidebar - File Upload
# st.sidebar.header("📤 Upload Your Source to Get Started")
# uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=False)

# if uploaded_pdf:
#     saved_path = save_uploaded_file(uploaded_pdf)
#     raw_docs = load_pdf_documents(saved_path)
#     processed_chunks = chunk_documents(raw_docs)
#     index_documents(processed_chunks)
#     st.sidebar.success("✅ Document uploaded successfully! Start chatting below.")

#     # Chat UI
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     user_input = st.chat_input("Enter your question about the document...")

#     if user_input:
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)

#         with st.spinner("🔍 Analyzing document..."):
#             relevant_docs = find_related_documents(user_input)
#             ai_response = generate_answer(user_input, relevant_docs)

#         with st.chat_message("assistant"):
#             st.markdown(ai_response)
# else:
#     st.warning("⚠️ Please upload a document to start chatting.")


#--------------------------------------------------------------

import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

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
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

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
    context = "\n\n".join([doc.page_content for doc in context_docs[:5]])  # More context for accuracy

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
            st.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

else:
    st.warning("⚠️ Please upload and select a document to start chatting.")
