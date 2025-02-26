from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version="2023-05-15",
    deployment_name="gpt-35-turbo",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.2
)

def debug_llm(user_code):
    prompt = f"""
        You are an expert Python debugger. Analyze the following Python code and identify errors.
        Provide:
        1️⃣ A clear explanation of the issue.
        2️⃣ A step-by-step debugging guide.
        3️⃣ Best practices to avoid similar errors in the future.

        Return only debugging suggestions, not the full solution.
    """
    system_prompt = (
        "You are an AI debugging assistant. Your job is to analyze the given Python code, "
        "identify logical, syntax, or runtime issues, and suggest debugging steps. "
        "Do NOT provide the entire solution; instead, guide the user towards resolving the issues themselves."
    )
    
    user_prompt = f"""
    Here is the Python code:
    ```python
    {user_code}
    ```
    Analyze the code and provide debugging hints without revealing the full solution.
    """
    
    messages = [SystemMessage(content=prompt), HumanMessage(content=user_prompt)]
    
    response = chat.invoke(messages)
    return response.content

if __name__ == "__main__":
    sample_code = """
    Here is my code:
    ```python
    def divide(a, b):
        return a / b
    
    print(divide(10, 0))```
    """
    
    debugging_tips = debug_llm(sample_code)
    print("Debugging Suggestions:")
    print(debugging_tips)