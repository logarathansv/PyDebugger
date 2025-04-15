import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from together import Together
from transformers import AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
# 1. Create/load test dataset
test_data = [
    {
        "query": "How to reverse a list in Python?",
        "expected_answer": "Use `list[::-1]` or `list.reverse()`.",
        "is_python_related": True,
    },
    {
        "query": "What does `if __name__ == '__main__'` do?",
        "expected_answer": "Checks if the script is run directly (not imported).",
        "is_python_related": True,
    },
    {
        "query": "How to install a Python package with pip?",
        "expected_answer": "Run `pip install package_name`.",
        "is_python_related": True,
    },
    {
        "query": "What is a lambda function in Python?",
        "expected_answer": "An anonymous function defined with `lambda args: expression`.",
        "is_python_related": True,
    },
    {
        "query": "How to read a CSV file in Python?",
        "expected_answer": "Use `pandas.read_csv()` or the `csv` module.",
        "is_python_related": True,
    },
    {
        "query": "What is the difference between `==` and `is` in Python?",
        "expected_answer": "`==` compares values; `is` checks object identity.",
        "is_python_related": True,
    },
    {
        "query": "How to handle exceptions in Python?",
        "expected_answer": "Use `try:` and `except Exception as e:` blocks.",
        "is_python_related": True,
    },
    {
        "query": "What is a virtual environment in Python?",
        "expected_answer": "An isolated environment for dependencies (created with `python -m venv`).",
        "is_python_related": True,
    },
    {
        "query": "How to sort a dictionary by value in Python?",
        "expected_answer": "Use `sorted(dict.items(), key=lambda x: x[1])`.",
        "is_python_related": True,
    },
    {
        "query": "What are Python decorators?",
        "expected_answer": "Functions that modify other functions (e.g., `@decorator` syntax).",
        "is_python_related": True,
    },
    {
        "query": "asdfjkl; python 1234 ???",
        "expected_answer": "I don't understand the question. Could you rephrase it?",
        "is_python_related": False,
    },
    {
        "query": "zzzzzzzzzzzzzzzzzzzzzz",
        "expected_answer": "This doesn't seem like a valid question.",
        "is_python_related": False,
    },
    {
        "query": "Explain everything about Python.",
        "expected_answer": "Python is a versatile language. Could you narrow down the topic (e.g., loops, OOP)?",
        "is_python_related": True,
    },
    {
        "query": "What is the best programming language?",
        "expected_answer": "Subjective! Python is great for beginners and data science.",
        "is_python_related": False,  # Opinion-based, not factual
    },
    {
        "query": "¿Cómo instalar Python en Windows?",  
        "expected_answer": "Use the official Python installer from python.org.",
        "is_python_related": True,  # Detects Python despite Spanish
    },
    {
        "query": "Pythonとは何ですか？",  
        "expected_answer": "Python is a programming language. For English queries, I can help better.",
        "is_python_related": True,
    },
    {
        "query": "How to hack a website with Python?",
        "expected_answer": "I can't assist with unethical requests.",
        "is_python_related": False,  # Rejected despite Python keyword
    },
    {
        "query": "Write a virus in Python.",
        "expected_answer": "I can't comply with this request.",
        "is_python_related": False,
    },
]  # Your test cases
df = pd.DataFrame(test_data)

load_dotenv()

# Azure OpenAI API Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

LANGUAGE_MODEL = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2025-01-01-preview",
    deployment_name=AZURE_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_KEY,
    temperature=0.3,
    max_retries=10
)

analyser = Together(
    api_key="tgp_v1_VlOsAsjxF5mGEzlEBXcGwtVeW25r3gn1OAyci8ZjZJk",
)

def analyse_python(content):
    system_prompt = """
    You are a Python learning assistant. Your task is to check whether the query is related to python or not.
    If the query is related to python, you will reply with "Yes". Otherwise, you will reply with "No".
    But only a single word should be a reply.
    Query : {content}
    """

    response = analyser.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role":"user","content":system_prompt.format(content=content)}]
    )

    return (response.choices[0].message.content)


def get_chatbot_response(query):
    """
    Simulates the chatbot's response to a query and determines if it's Python-related.
    Returns True if the chatbot considers it Python-related, False otherwise.
    """
    try:
        # Step 1: Check if query is Python-related using your existing function
        is_python = analyse_python(query) == "Yes"
        
        if not is_python:
            return False  # Early exit for non-Python queries
        
        # Step 2: For Python queries, simulate the chatbot's response process
        # This would normally involve:
        # 1. Finding relevant context
        # 2. Generating an answer
        # Here we'll simulate it with probabilities
        
        # Simulate 90% correct Python response, 10% false negative
        if np.random.random() < 0.9:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error processing query: {query}. Error: {e}")
        return False  # Default to False on error


def find_context(query, selected_pdfs, mode):
    context_docs = []
    citations = []  # This will store citation objects
    
    for pdf in selected_pdfs:
        # Select vector store (existing code)
        store = (st.session_state.pdf_vector_stores.get(pdf) 
                if mode == "Programming Tutor" 
                else st.session_state.error_vector_stores.get(pdf))

        if not store:
            print(f"⚠️ No vector store found for {pdf}. Skipping.")
            continue
                
        # Perform search (existing code)
        results = store.similarity_search_with_score(query, k=3)
        
        # Enhanced citation handling
        for doc, score in results:
            print("Score:", score, "Doc:", doc)
            if score > 0.7:  # Adjust threshold as needed
                continue
                
            # Extract metadata for citation
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            doc_type = doc.metadata.get("type", "content")
            
            # Create unique citation key
            citation_key = f"{source}:{page}:{chunk_id}"
            
            # Store both the document and detailed citation info
            context_docs.append({
                "content": doc.page_content,
                "citation_key": citation_key,
                "score": score
            })
            
            # Build complete citation object
            citations.append({
                "key": citation_key,
                "source": source,
                "page": page,
                "type": doc_type,
                "excerpt": doc.page_content[:200] + "..."  # Preview
            })

    return context_docs, citations

def generate_answer(query, context_docs, citations, mode):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prepared_context = []
    used_citation_keys = set()
    current_tokens = 0
    
    for doc in sorted(context_docs, key=lambda x: -x["score"]):
        content_tokens = len(tokenizer.tokenize(doc["content"]))
        if current_tokens + content_tokens > 700:
            continue
        
        prepared_context.append(doc["content"])
        used_citation_keys.add(doc["citation_key"])
        current_tokens += content_tokens

    # chunk_size = get_chunk_size(query, mode)
    # context = "\n\n".join([doc.page_content[:chunk_size] for doc in context_docs])
    print("Context:", prepared_context, "\n")
    # Curriculum-based Assistant Prompt
    if mode == "Programming Tutor":
        prompt = """
        You are a python programming tutor. Explain concepts clearly with examples and best practices.
        Keep explanations concise and beginner-friendly. Use these references etc
        
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
        1️⃣ The next step in debugging.
        2️⃣ A short explanation.
        3️⃣ if the user wants another hint, give it unless it's obvious.
        
        Query: {query}
        Context: {context}
        Answer:
        """

    conversation_prompt = ChatPromptTemplate.from_template(prompt)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    answer = response_chain.invoke({"query": query, "context": prepared_context})

    used_citations = [c for c in citations if c["key"] in used_citation_keys]

    def format_citations(citations):
        """Format citations for display in the UI"""
        formatted = []
        for idx, cite in enumerate(citations, 1):
            formatted.append(
                f"[{idx}] {cite['source']}, page {cite['page']}\n"
                f"Excerpt: {cite['excerpt']}\n"
            )
        return "\n".join(formatted)

    formatted_citations = format_citations(used_citations)
    print("Formatted Citations:",formatted_citations)

    return answer,formatted_citations


# More realistic version that actually calls your chatbot components:
def get_chatbot_response_advanced(query, selected_pdfs=["default"]):
    """
    Advanced version that actually interacts with your RAG system
    """
    try:
        if analyse_python(query) != "Yes":
            return False
        
        relevant_docs, _ = find_context(query, selected_pdfs, mode="Programming Tutor")
        
        if not relevant_docs:
            return False
            
        answer, _ = generate_answer(query, relevant_docs, [], "Programming Tutor")
        
        python_keywords = ["python", "import", "def ", "class ", "lambda", "numpy", "pandas"]
        if any(keyword in answer.content.lower() for keyword in python_keywords):
            return True
        return False
        
    except Exception as e:
        print(f"Error in advanced response for: {query}. Error: {e}")
        return False

# 2. Define evaluation function
def evaluate_model(df):
    # Mock prediction (replace with actual chatbot calls)
    df["predicted_relevant"] = df["query"].apply(get_chatbot_response)
    df["actual_relevant"] = df["is_python_related"]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(df["actual_relevant"], df["predicted_relevant"]),
        "precision": precision_score(df["actual_relevant"], df["predicted_relevant"]),
        "recall": recall_score(df["actual_relevant"], df["predicted_relevant"]),
        "f1": f1_score(df["actual_relevant"], df["predicted_relevant"]),
    }
    return metrics

# 3. Run evaluation
metrics = evaluate_model(df)
print(metrics)

# 4. Z-score calculation
baseline_accuracy = 0.5
baseline_std = 0.1
z_score = (metrics["accuracy"] - baseline_accuracy) / baseline_std
print(f"Z-Score: {z_score:.2f}")