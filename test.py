import os
from llama_cpp import Llama

# This is an example code that uses Hugging Face's 'repo_id' and a specific model filename
llm = Llama.from_pretrained(
    repo_id="Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M",   # Hugging Face repo_id
    filename="llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf",     # Model file name
    n_gpu_layers=20,    # Number of GPU layers for Apple Silicon MPS
    n_ctx=2048,         # Context window size
    f16_kv=True         # Use half-precision for key/value cache
)

# Korean prompt
prompt = "안녕하세요."

# Make a chat completion request
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

print(response["choices"][0]["message"]["content"])
