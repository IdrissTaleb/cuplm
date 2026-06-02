from llama_cpp import Llama

llm = Llama(
    model_path="./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4
)

output = llm(
    "What is 2+2?",
    max_tokens=20,
    temperature=0.7
)

print(output["choices"][0]["text"])