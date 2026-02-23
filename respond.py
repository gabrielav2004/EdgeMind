from llama_cpp import Llama
import os

MODEL_PATH = "models/tinyllama.gguf"

llm = None

def load_model():
    global llm
    if llm is None:
        print("loading local model...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )
        print("model loaded")
    return llm

def build_prompt(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source {i+1}]:\n{chunk}\n\n"

    prompt = f"""<|system|>
You are a precise question answering assistant.
The answer to the question is contained somewhere in the sources below.
Read ALL sources carefully before answering.
Give a direct, specific answer. Quote numbers and facts exactly as they appear.</s>
<|user|>
{context}
Question: {query}</s>
<|assistant|>
Based on the provided sources, """

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
        temperature=0.1,
        stop=["</s>", "<|user|>", "<|system|>"],
        echo=False
    )

    response = output['choices'][0]['text'].strip()
    return response

if __name__ == "__main__":
    test_chunks = [
        "robot motors require precise torque control for smooth operation",
        "calibration is essential for maintaining robot arm accuracy",
        "sensors provide feedback to the robot control system"
    ]

    query = "how do robot motors work"
    print(f"query: {query}\n")

    answer = respond(query, test_chunks)
    print(f"answer: {answer}")