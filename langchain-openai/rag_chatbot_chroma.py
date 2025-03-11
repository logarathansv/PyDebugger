from langchain.schema import Document
from openai import BadRequestError
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from pptx import Presentation
from urllib.parse import urlparse, parse_qs
import time
import json
import os

# Load environment variables
load_dotenv()

# Azure OpenAI Embeddings Configuration
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
    temperature=0.3
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
PDF_STORAGE_PATH = 'document_store/files/'
CHROMA_DB_PATH = 'chroma_db/'
PDF_LIST_PATH = os.path.join(CHROMA_DB_PATH, "pdf_list.json")

# Ensure directories exist
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Function to save transcript
def save_transcript(video_name, transcript_text):
    file_path = os.path.join(PDF_STORAGE_PATH, f"{video_name}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(transcript_text)
    return file_path
# Function to extract YouTube Video ID from URL
def extract_video_id(url):
    parsed_url = urlparse(url)
    video_id = None
    
    if parsed_url.netloc in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]

    return video_id

# Function to get only English transcript and align into 30s segments
def get_english_transcript(video_id, video_name, segment_duration=30):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en']).fetch()
        
        # Organize transcript into 30-second segments
        segments = []
        current_segment = []
        start_time = 0

        for entry in transcript:
            if entry["start"] - start_time >= segment_duration:
                segments.append(f"[{format_time(start_time)}] " + " ".join(current_segment))
                start_time = entry["start"]
                current_segment = []
            
            current_segment.append(entry["text"])

        # Save the last segment
        if current_segment:
            segments.append(f"[{format_time(start_time)}] " + " ".join(current_segment))

        # Save to a text file
        file_path = save_transcript(video_name, "\n".join(segments))
        return file_path, "\n".join(segments)

    except Exception as e:
        return None, f"Error: {e}"

def save_pdf_list(pdf_list):
    with open(PDF_LIST_PATH, "w") as f:
        json.dump(pdf_list, f)

def load_pdf_list():
    if os.path.exists(PDF_LIST_PATH):
        with open(PDF_LIST_PATH, "r") as f:
            return json.load(f)
    return []

# Initialize Session State
if "pdf_vector_stores" not in st.session_state:
    st.session_state.pdf_vector_stores = {}

if "pdf_list" not in st.session_state:
    st.session_state.pdf_list = load_pdf_list()  # Load PDF list from disk

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load existing Chroma collections
for pdf_name in st.session_state.pdf_list:
    collection_name = pdf_name.replace(" ", "_").lower()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDING_MODEL,
        persist_directory=CHROMA_DB_PATH
    )
    st.session_state.pdf_vector_stores[pdf_name] = vector_store

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_pptx(file_path):
    presentation = Presentation(file_path)
    return "\n".join([shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text")])

# Save PDF
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# Process PDF
def process_document(file_name, file_path):
    file_ext = file_name.split(".")[-1].lower()
    
    if file_ext == "pdf":
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
    elif file_ext == "txt":
        text = extract_text_from_txt(file_path)
        docs = [Document(page_content=text)] 
    elif file_ext == "pptx":
        text = extract_text_from_pptx(file_path)
        docs = [Document(page_content=text)] 
    else:
        st.error("Unsupported file format.")
        return
    
    chunker = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = chunker.split_documents(docs)

    # Create or load a Chroma collection for the PDF
    collection_name = file_name.replace(" ", "_").lower()  # Use file name as collection name
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDING_MODEL,
        collection_name=collection_name,
        persist_directory=CHROMA_DB_PATH
    )
    st.session_state.pdf_vector_stores[file_name] = vector_store

    if file_name not in st.session_state.pdf_list:
        st.session_state.pdf_list.append(file_name)
        save_pdf_list(st.session_state.pdf_list)  # Save updated PDF list to disk

# Search Relevant Docs
# def find_context(query, selected_pdfs):
#     context_docs = []
#     for pdf in selected_pdfs:
#         store = st.session_state.pdf_vector_stores.get(pdf)
#         if store:
#             context_docs.extend(store.similarity_search(query))
#     return context_docs

def find_context(query, selected_pdfs, threshold=0.7, max_results=5):
    context_docs = []
    citations = []

    for pdf in selected_pdfs:
        store = st.session_state.pdf_vector_stores.get(pdf)
        
        if store:
            results = store.similarity_search_with_score(query)  # Get scores
            
            # Filter out low-confidence matches
            filtered_results = [(doc, score) for doc, score in results if score >= threshold]
            
            # Sort by highest similarity score
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            
            # Select top results
            top_results = filtered_results[:max_results]
            
            # Extract documents & citations
            for doc, score in top_results:
                context_docs.append(doc)
                citations.append((pdf, doc.metadata.get("page", "Unknown"), round(score, 2)))  # Include score

    return context_docs, citations


def generate_follow_up_hint(chat_history, mode):
    # Extract the conversation context
    conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    print(conversation_context)
    # Define the follow-up hint prompt
    if mode == "Programming Tutor":
        prompt = """
        You are a programming tutor. Provide another hint or clarification based on the conversation below.
        Keep your response concise and helpful.

        Conversation:
        {conversation_context}

        Hint:
        """
    elif mode == "Rubber Duck Assistant":
        prompt = """
        You are a rubber duck debugging assistant. Provide another guiding question or hint to help the user think through their problem.

        Conversation:
        {conversation_context}

        Hint:
        """

    # Generate the follow-up hint
    conversation_prompt = ChatPromptTemplate.from_template(prompt)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"conversation_context": conversation_context})

def get_chunk_size(query, mode):
    if mode == "Programming Tutor" and len(query) < 50:
        return 100
    elif mode == "Rubber Duck Assistant":
        return 150
    else:
        return 100
# def generate_answer(query, context_docs, citations, mode):
#     chunk_size = get_chunk_size(query, mode)
#     context = "\n\n".join([doc.page_content[:chunk_size] for doc in context_docs])

#     prompt = """
#     You are an AI assistant that provides answers with reliable sources.
#     Use the context below to generate an accurate response.
    
#     Query: {query}
#     Context: {context}
#     Answer:
#     """
    
#     conversation_prompt = ChatPromptTemplate.from_template(prompt)
#     response_chain = conversation_prompt | LANGUAGE_MODEL
#     answer = response_chain.invoke({"query": query, "context": context})

#     # Append citations
#     if citations:
#         sources = "\n".join([f"- {pdf} (Page {page})" for pdf, page in citations])
#         answer += f"\n\n**Sources:**\n{sources}"

#     return answer

def generate_answer(query, context_docs, citations, mode):
    chunk_size = get_chunk_size(query, mode)
    context = "\n\n".join([doc.page_content[:chunk_size] for doc in context_docs])
    print("Context:", context, "\n")
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
        You are an expert Rubber Duck Python debugging assistant. Analyze the following Python code and provide debugging hints **one at a time**.
        Do NOT give the full solution immediately.
        Each response should include:
        1️- The next step in debugging.
        2️- A short explanation.
        3-if the user wants another hint, give it unless it's obvious.
        4- specify the line number if needed.
        
        Query: {query}
        Context: {context}
        Answer:
        """

    conversation_prompt = ChatPromptTemplate.from_template(prompt)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    answer = response_chain.invoke({"query": query, "context": context})
    cited = ""
    cited_pdfs = set()  # Use a set to store unique citations

    for pdf in citations:
        cited_pdfs.add(f"- {pdf[0]}")  # Add citation as a formatted string

    # Append citations
    if cited_pdfs:
        sources = "\n".join(sorted(cited_pdfs))  # Sort to maintain order if needed
        cited += f"\n\n**Sources:**\n{sources}"

    #time.sleep(20)
    return answer,cited


st.title("Learning Chatbot & Debugging Assistant")
st.markdown("---")
st.sidebar.header("Upload Programming Resources")
uploaded_files = st.sidebar.file_uploader("Upload PDF, TXT, PPTX", type=["pdf", "txt", "pptx"], accept_multiple_files=True)

# Process New PDFs
if uploaded_files:
    for pdf in uploaded_files:
        if pdf.name not in st.session_state.pdf_vector_stores:
            file_path = save_uploaded_file(pdf)
            process_document(pdf.name, file_path)
    st.sidebar.success("Documents uploaded! Select them below.")

st.sidebar.header("YouTube Transcript Extractor")
st.sidebar.caption("Paste a YouTube link to extract and save the transcript.")
video_url = st.sidebar.text_input("🔗 Paste YouTube link here:")

if st.sidebar.button("Get Transcript"):
    if video_url:
        try:
            # Extract Video ID
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please check and try again.")
                st.stop()
            video_name = f"transcript_{video_id}"
            file_path, transcript_text = get_english_transcript(video_id, video_name)
            if file_path:
                # Update PDF list
                pdf_list = load_pdf_list()
                if video_name not in pdf_list:
                    pdf_list.append(video_name)
                    save_pdf_list(pdf_list)
                st.sidebar.success(f"Transcript saved as: {video_name}.txt")
                st.rerun()
            else:
                st.sidebar.error(transcript_text)

        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    else:
        st.warning("⚠ Please enter a valid YouTube link.")

# Sidebar - Select PDFs (Checkbox)
st.sidebar.header("Select Files")
selected_pdfs = [pdf for pdf in st.session_state.pdf_list if st.sidebar.checkbox(pdf, value=True)]

# Sidebar - Delete Files
st.sidebar.header("Remove Files")
delete_pdf = st.sidebar.selectbox("Select PDF to Remove", ["None"] + st.session_state.pdf_list)

if st.sidebar.button("Delete Files") and delete_pdf != "None":
    # Delete the Chroma collection for the PDF
    vector_store = st.session_state.pdf_vector_stores.get(delete_pdf)
    if vector_store:
        vector_store.delete_collection()
    del st.session_state.pdf_vector_stores[delete_pdf]
    st.session_state.pdf_list.remove(delete_pdf)
    save_pdf_list(st.session_state.pdf_list)  # Save updated PDF list to disk
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
        # Add user query to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate AI response
        with st.spinner("🔍 Searching resources..."):
            relevant_docs, citations = find_context(user_query, selected_pdfs)
            try:
                ai_response,cited = generate_answer(user_query, relevant_docs,citations, mode)
            except BadRequestError as e:
                ai_response = "Sorry, I cannot provide a response to that query due to content filtering policies. Please rephrase your question."

        # Add AI response to chat history
        with st.chat_message("assistant"):
            st.markdown(ai_response.content)
            st.markdown(cited)
            st.session_state.messages.append({"role": "assistant", "content": ai_response.content})
            st.session_state.messages.append({"role": "assistant", "content": cited})

    # Add "Need another hint" button
    if mode == "Rubber Duck Assistant" and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        if st.button("Need another hint"):
            # Generate a follow-up hint based on the conversation context
            with st.spinner("🤔 Generating another hint..."):
                try:
                    follow_up_response = generate_follow_up_hint(st.session_state.messages, mode)
                except BadRequestError as e:
                    follow_up_response = "Sorry, I cannot provide a response to that query."
                    st.markdown(follow_up_response.content)
                    st.session_state.messages.append({"role": "assistant", "content": follow_up_response.content})
else:
    st.warning("⚠️ Please upload and select a document to start chatting.")