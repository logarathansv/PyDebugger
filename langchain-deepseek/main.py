from langchain_deepseek import ChatDeepSeek
import asyncio

API_KEY = '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'

async def main():
    llm = ChatDeepSeek(
        model='deepseek/deepseek-r1:free',
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        timeout=60
    )
    # inp = input("Enter your question: ")

    messages = [
        ("system", "You are a Python Code Debugger and Learning Chatbot. You can only give hints to the code where errors happen. You can only respond with a code snippet."),
        ("human", "What is the meaning of life?"),
    ]

    # response = llm.invoke(messages)
    # print(response.text())


    for chunk in llm.stream(messages):
        print(chunk.text(), end="")

    stream = llm.stream(messages)
    full = next(stream)
    for chunk in stream:
        full += chunk

    print(full.text()) 


if __name__ == "__main__":
    asyncio.run(main())