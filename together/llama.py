from together import Together

client = Together(
    api_key="tgp_v1_VlOsAsjxF5mGEzlEBXcGwtVeW25r3gn1OAyci8ZjZJk",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[{"role":"user","content":""}]
)
print(response.choices[0].message.content)