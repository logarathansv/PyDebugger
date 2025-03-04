
import openai

openai.api_base = "https://20225-m7usbn58-swedencentral.cognitiveservices.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2025-01-01-preview"
openai.api_key = "856b8KJuWnNDFXwJOXhZGZwaQQ8n4w2PLJ7uOMZDUbiBxVZIkBPXJQQJ99BCACfhMk5XJ3w3AAAAACOGCpsc"

try:
    response = openai.ChatCompletion.create(
        model="gpt-35-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print("✅ API Connected Successfully")
except Exception as e:
    print("❌ Error:", e)
