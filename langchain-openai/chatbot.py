import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version="2023-05-15",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
# Load environment variables
load_dotenv()

# Azure OpenAI API Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

# Initialize Azure LLM
LANGUAGE_MODEL = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version="2023-05-15",
    deployment_name="gpt-35-turbo",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.2
)

# Constants
PDF_STORAGE_PATH = 'document_store/pdfs/'
PROMPT_TEMPLATE = """
You are an expert programming assistant and rubber duck debugging companion. 
Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Initialize Memory for Debugging Chat
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=LANGUAGE_MODEL, memory=memory)

# Initialize Session State
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(embedding=embeddings)

# Functions
def save_uploaded_file(uploaded_file):
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    return PDFPlumberLoader(file_path).load()

def chunk_documents(raw_documents):
    return RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500
    ).split_documents(raw_documents)

def index_documents(file_name, document_chunks):
    st.session_state.vector_store.add_documents(document_chunks)
    st.session_state.documents[file_name] = document_chunks

def remove_pdf(file_name):
    if file_name in st.session_state.documents:
        del st.session_state.documents[file_name]
        st.session_state.vector_store = InMemoryVectorStore()
        for doc_name, chunks in st.session_state.documents.items():
            st.session_state.vector_store.add_documents(chunks)

def find_related_documents(query, num_docs=5):
    return st.session_state.vector_store.similarity_search(query, k=num_docs)

def merge_context_chunks(chunks):
    return "\n\n".join([doc.page_content for doc in chunks])

def generate_answer(user_query, context_documents):
    context_text = merge_context_chunks(context_documents)
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

def debug_python_code(user_code):
    """Analyzes Python code and provides debugging hints one at a time."""
    prompt = f"""
    You are an expert Python debugger. Analyze the following Python code and provide debugging hints *one at a time*.
    Do NOT give the full solution immediately.
    Each response should include:
    1️⃣ The next step in debugging.
    2️⃣ A short explanation.
    3️⃣ Ask if the user wants another hint.

    *Code:*
    python
    {user_code}
    

    Respond with only one debugging hint at a time.
    """
    response = conversation.predict(input=prompt)
    return response

# UI
st.title("📘 Curriculum-Based Chatbot & Python Debugging Assistant (Azure GPT-4)")
st.markdown("### Chat with your Programming Documents or Debug Python Code")

# File Upload Section
uploaded_pdf = st.file_uploader("Upload Programming Document (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_pdf:
    for pdf in uploaded_pdf:
        if pdf.name not in st.session_state.documents:
            saved_path = save_uploaded_file(pdf)
            raw_docs = load_pdf_documents(saved_path)
            processed_chunks = chunk_documents(raw_docs)
            index_documents(pdf.name, processed_chunks)
    st.success("✅ Documents processed! Select sources to use.")

# Show uploaded PDFs with checkboxes
st.markdown("### Select Sources to Use:")
selected_pdfs = []
for file_name in st.session_state.documents.keys():
    if st.checkbox(file_name, value=True):
        selected_pdfs.append(file_name)

# Allow removing PDFs
st.markdown("### Remove Sources:")
for file_name in list(st.session_state.documents.keys()):
    if st.button(f"🗑 Remove {file_name}"):
        remove_pdf(file_name)
        st.experimental_rerun()

# Chat Interface
user_input = st.chat_input("Ask about programming concepts...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.spinner("Thinking..."):
        selected_chunks = []
        for pdf in selected_pdfs:
            selected_chunks.extend(st.session_state.documents[pdf])
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)
        
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", ai_response))

# Debugging Section
st.markdown("## 🐞 Python Debugging Assistant")
user_code = st.text_area("Paste your Python code here:")

if st.button("Debug Code"):
    if user_code.strip():
        debugging_hint = debug_python_code(user_code)
        st.session_state.chat_history.append(("debugger", debugging_hint))
        with st.chat_message("debugger"):
            st.write(debugging_hint)
    else:
        st.warning("⚠ Please enter some Python code to debug.")

# Display Chat History
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# Follow-up debugging
if "debugger" in [role for role, _ in st.session_state.chat_history]:
    follow_up = st.button("Give me another debugging hint")
    if follow_up:
        more_hints = conversation.predict(input="Yes, I need another hint.")
        st.session_state.chat_history.append(("debugger", more_hints))
        with st.chat_message("debugger"):
            st.write(more_hints)