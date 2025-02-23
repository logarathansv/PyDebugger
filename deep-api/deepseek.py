import os
import requests

API_URL = 'https://openrouter.ai/api/v1/chat/completions'
API_KEY = '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'

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

data = {
    "model": "deepseek/deepseek-chat:free",
    "messages": [("system", "You are a Python Code Debugger and Learning Chatbot. You can only give hints to the code where errors happen. You can only respond with a code snippet."),
        ("human", "What is the use of the __init__ function?")]
}

response = requests.post(API_URL, json=data, headers=headers)

if response.status_code == 200:
    print("API Response:", response.json())
else:
    print("Failed to fetch data from API. Status Code:", response.status_code)