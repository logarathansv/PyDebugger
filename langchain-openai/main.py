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

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=chat,
    memory=memory
)


def debug_python_code(user_code):
    # """
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
    
    prompt = f"""
    You are an expert Python debugger. Analyze the following Python code and provide debugging hints **one at a time**.
    Do NOT give the full solution immediately.
    Each response should include:
    1️⃣ The next step in debugging.
    2️⃣ A short explanation.
    3️⃣ Ask if the user wants another hint.

    **Code:**
    ```python
    {user_code}
    ```

    Respond with only one debugging hint at a time.
    """

    response = conversation.predict(input=prompt)
    return response

if __name__ == "__main__":
    sample_code = """
    Here is my code:
    ```python
    def divide(a, b):
        return a / b
    
    print(divide(10, 0))```
    """
    
    debugging_hint = debug_python_code(sample_code)
    print(debugging_hint)

    follow_up = "Yes, I need another hint."
    more_hints = conversation.predict(input=follow_up)
    print(more_hints)