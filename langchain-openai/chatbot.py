import os
import time
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()

AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_OPENAI")
AZURE_EMBEDDING_API = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

embeddings = AzureOpenAIEmbeddings(
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

if "documents" not in st.session_state:
    st.session_state.documents = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = Chroma(
        collection_name="programming_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

# Functions
def save_uploaded_file(uploaded_file):
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    try:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        valid_documents = []
        for i, doc in enumerate(documents):
            if doc.page_content.strip():  # Only include pages with valid text content
                valid_documents.append(doc)
            else:
                st.warning(f"Skipping Page {i + 1} in {file_path} (no text content).")
        
        return valid_documents
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def chunk_documents(raw_documents):
    return RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500
    ).split_documents(raw_documents)

def index_documents(file_name, document_chunks):
    # Extract text content
    processed_texts = []
    for chunk in document_chunks:
        if hasattr(chunk, "page_content") and chunk.page_content.strip():
            processed_texts.append(chunk.page_content)
        else:
            st.warning(f"Skipping invalid or empty chunk in file: {file_name}")

    if not processed_texts:
        raise ValueError(f"No valid text found in document chunks for file: {file_name}")

    # Convert plain text back into `Document` objects
    document_objects = [Document(page_content=text) for text in processed_texts]

    # Debugging: Print the first few documents to ensure they are correct
    for doc in document_objects[:3]:
        st.write(doc.page_content)

    # Add documents to vector store with delay
    try:
        batch_size = 3  # Adjust batch size based on rate limits
        for i in range(0, len(document_objects), batch_size):
            batch = document_objects[i:i + batch_size]

            # Extract text & metadata
            texts = [doc.page_content for doc in batch]
            metadata = [{"source": file_name} for _ in batch]

            # Call OpenAI embeddings with 10-second delay
            st.session_state.vector_store.add_texts(texts=texts, metadatas=metadata)

            # Wait before next API call (free-tier rate limit)
            time.sleep(10)

        # Store documents in session state
        st.session_state.documents[file_name] = document_objects

    except Exception as e:
        st.error(f"Error adding documents to vector store: {e}")



def remove_pdf(file_name):
    if file_name in st.session_state.documents:
        del st.session_state.documents[file_name]

        # Remove documents from ChromaDB by filtering by metadata (source)
        st.session_state.vector_store.delete(
            where={"source": file_name}  # Deletes all docs with this source
        )

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
    """Analyzes Python code and provides debugging hints based on real issues."""
    prompt = f"""
    You are an expert Python debugger. Analyze the following Python code.
    
    - If there are **syntax errors**, identify them first.
    - If the syntax is correct, look for **logical errors**.
    - Provide **one hint at a time**, guiding the user step-by-step.
    - If the code is correct, explain why it works.
    - Always ask if the user needs another hint.

    Code:
    ```python
    {user_code}
    ```

    Respond with:
    - A **brief debugging hint**.
    - Ask if the user wants another hint.
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
            print("Saved ")
            raw_docs = load_pdf_documents(saved_path)
            print("Loaded")
            processed_chunks = chunk_documents(raw_docs)
            print("Processed")
            index_documents(pdf.name, processed_chunks)
            print("Indexed")
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
        print("Found Related Documents ... ")
        ai_response = generate_answer(user_input, relevant_docs)
        print("Generate answer")
        
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", ai_response))

# Debugging Section
st.markdown("## 🐞 Python Debugging Assistant")
user_code = st.text_area("Paste your Python code here:")

if st.button("Debug Code"):
    if user_code.strip():
        debugging_hint = debug_python_code(user_code)

        # Store in chat history
        st.session_state.chat_history.append(("debugger", debugging_hint))
        
        with st.chat_message("debugger"):
            st.write(debugging_hint)

        # Add a follow-up hint button directly in response
        if st.button("🔄 Need another hint?"):
            more_hints = conversation.predict(input="Yes, I need another hint.")
            st.session_state.chat_history.append(("debugger", more_hints))
            with st.chat_message("debugger"):
                st.write(more_hints)

    else:
        st.warning("⚠ Please enter some Python code to debug.")

# Display Chat History
# for role, message in st.session_state.chat_history:
#     with st.chat_message(role):
#         st.write(message)

# # Follow-up debugging
# if "debugger" in [role for role, _ in st.session_state.chat_history]:
#     follow_up = st.button("Give me another debugging hint")
#     if follow_up:
#         more_hints = conversation.predict(input="Yes, I need another hint.")
#         st.session_state.chat_history.append(("debugger", more_hints))
#         with st.chat_message("debugger"):
#             st.write(more_hints)