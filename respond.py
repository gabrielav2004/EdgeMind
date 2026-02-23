from llama_cpp import Llama
import os

MODEL_PATH = "models/tinyllama.gguf"

# load once, keep resident
llm = None

def load_model():
    global llm
    if llm is None:
        print("loading local model...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,        # context window
            n_threads=4,       # cpu threads
            n_gpu_layers=0,    # 0 = pure cpu, no gpu needed
            verbose=False
        )
        print("model loaded")
    return llm

def build_prompt(query, chunks):
    # format retrieved chunks as context
    context = "\n\n".join([
        f"[{i+1}] {chunk}" 
        for i, chunk in enumerate(chunks)
    ])
    
    prompt = f"""<|system|>
You are a helpful assistant. Answer the question using only the provided context. 
Be concise and factual. If the context does not contain the answer, say so. Decide properly on the given context Semantically.</s>
<|user|>
Context:
{context}

Question: {query}</s>
<|assistant|>"""
    
    return prompt

def respond(query, chunks, max_tokens=256):
    model = load_model()
    
    if not chunks:
        return "no relevant context found in knowledge base"
    
    prompt = build_prompt(query, chunks)
    
    print("generating response...")
    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,      # low temp = more factual
        stop=["</s>", "<|user|>", "<|system|>"],
        echo=False
    )
    
    response = output['choices'][0]['text'].strip()
    return response

if __name__ == "__main__":
    # test directly
    test_chunks = [
        "robot motors require precise torque control for smooth operation",
        "calibration is essential for maintaining robot arm accuracy",
        "sensors provide feedback to the robot control system"
    ]
    
    query = "how do robot motors work"
    print(f"query: {query}\n")
    
    answer = respond(query, test_chunks)
    print(f"answer: {answer}")