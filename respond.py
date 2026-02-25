from config import (
    MODE, TEMPERATURE, MAX_TOKENS,
    MODEL_PATH, N_THREADS, N_CTX, N_GPU_LAYERS,
    API_KEY, API_BASE_URL, MODEL_NAME
)

_llm = None

def load_model():
    global _llm
    if MODE != "local":
        print(f"using {MODE} api")
        return None
    if _llm is None:
        print("loading local model...")
        from llama_cpp import Llama
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False
        )
        print("model loaded")
    return _llm

def build_prompt(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source {i+1}]:\n{chunk}\n\n"
    return context, query


def _respond_local(query, chunks):
    model = load_model()
    context, q = build_prompt(query, chunks)

    prompt = f"""<|system|>
You are a strict fact extraction assistant.
RULES:
- ONLY use exact information from the provided sources
- NEVER infer, assume, or add information not explicitly written
- NEVER combine information from unrelated sources
- If a source is clearly unrelated to the question, ignore it completely
- If the answer is not clearly stated in any source, say "not found in knowledge base"
- Quote names, numbers, and facts exactly as written
- Do not explain or elaborate beyond what the source states</s>
<|user|>
{context}
Question: {q}

Important: Only answer if the information is explicitly stated in the sources above. Ignore any source that is not directly relevant to the question.</s>
<|assistant|>
According to the sources, """

    model.reset()

    output = model(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["</s>", "<|user|>", "<|system|>", "\n\n"],
        echo=False
    )
    return output['choices'][0]['text'].strip()

def _respond_cloud(query, chunks):
    from openai import OpenAI
    context, q = build_prompt(query, chunks)

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content":
                "You are a precise question answering assistant. "
                "Read ALL sources carefully. Quote facts exactly."},
            {"role": "user", "content": f"{context}\nQuestion: {q}"}
        ]
    )
    return response.choices[0].message.content.strip()

def _respond_anthropic(query, chunks):
    import anthropic
    context, q = build_prompt(query, chunks)

    client = anthropic.Anthropic(api_key=API_KEY)
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        system=(
            "You are a precise question answering assistant. "
            "Read ALL sources carefully. Quote facts exactly."
        ),
        messages=[
            {"role": "user", "content": f"{context}\nQuestion: {q}"}
        ]
    )
    return message.content[0].text.strip()

def respond(query, chunks):
    if not chunks:
        return "no relevant context found in knowledge base"

    print(f"generating response via {MODE}...")

    if MODE == "local":
        return _respond_local(query, chunks)
    elif MODE == "cloud":
        return _respond_cloud(query, chunks)
    elif MODE == "anthropic":
        return _respond_anthropic(query, chunks)
    else:
        raise ValueError(
            f"unknown MODE: {MODE}. "
            f"choose from: local, cloud, anthropic"
        )

if __name__ == "__main__":
    test_chunks = [
        "robot motors require precise torque control for smooth operation",
        "calibration is essential for maintaining robot arm accuracy",
    ]
    query = "how do robot motors work"
    print(f"mode: {MODE}")
    answer = respond(query, test_chunks)
    print(f"answer: {answer}")