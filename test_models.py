from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(token=token)

models = [
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]

for m in models:
    try:
        resp = client.chat_completion(
            messages=[{"role": "user", "content": "say hi"}],
            model=m,
            max_tokens=5,
        )
        print(f"OK: {m} -> {resp.choices[0].message.content}")
    except Exception as e:
        print(f"FAIL: {m} -> {type(e).__name__}: {str(e)[:100]}")
