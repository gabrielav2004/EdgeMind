from respond import load_model
from config import MAX_TOKENS

def format_section(text):
    """
    Reformat a section using configured provider (local or cloud).
    """
    from config import MODE
    from respond import _respond_cloud, _respond_anthropic

    format_prompt = """Rewrite the following text into clean structured paragraphs.
Rules:
- Each paragraph must cover exactly one topic or fact
- Each paragraph must be a complete self contained sentence or group of sentences
- Keep all facts, names, numbers exactly as they appear
- Remove broken sentences, formatting artifacts, redundant whitespace
- Do not add, infer, or remove any information

Text to rewrite:
""" + text

    if MODE == "local":
        from respond import load_model
        model = load_model()
        if model is None:
            return text
        model.reset()
        from config import MAX_TOKENS, TEMPERATURE
        output = model(
            f"""<|system|>
You are a document formatter. Follow the rules exactly.</s>
<|user|>
{format_prompt}</s>
<|assistant|>
""",
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["</s>", "<|user|>", "<|system|>"],
            echo=False
        )
        return output['choices'][0]['text'].strip()

    elif MODE == "cloud":
        from openai import OpenAI
        from config import API_KEY, API_BASE_URL, MODEL_NAME, TEMPERATURE, MAX_TOKENS
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "You are a document formatter. Follow the rules exactly."},
                {"role": "user", "content": format_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    elif MODE == "anthropic":
        import anthropic
        from config import API_KEY, MODEL_NAME
        client = anthropic.Anthropic(api_key=API_KEY)
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            system="You are a document formatter. Follow the rules exactly.",
            messages=[{"role": "user", "content": format_prompt}]
        )
        return message.content[0].text.strip()

    return text  # fallback

def format_document(text, use_llm=True):
    """
    Full pipeline: raw text → formatted → clean structured text.
    Falls back to rule based cleaning if use_llm is False.
    """
    if not use_llm:
        return clean_text(text)

    try:
        sections = split_into_sections(text)
        print(f"formatting {len(sections)} sections...")
        formatted_sections = []

        for i, section in enumerate(sections):
            print(f"  formatting section {i+1}/{len(sections)}...")
            formatted = format_section(section)
            formatted_sections.append(formatted)

        return "\n\n".join(formatted_sections)

    except Exception as e:
        print(f"formatter error: {e} — falling back to rule based cleaning")
        return clean_text(text)

def split_into_sections(text, max_chars=800):
    """
    Split raw text into sections small enough for TinyLlama context.
    Always splits at paragraph or sentence boundaries.
    """
    sections = []
    paragraphs = text.split('\n\n')
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + "\n\n"
        else:
            if current.strip():
                sections.append(current.strip())
            current = para + "\n\n"

    if current.strip():
        sections.append(current.strip())

    return sections

def format_document(text):
    """
    Full pipeline: raw text → TinyLlama formatted → clean structured text
    """
    model = load_model()
    sections = split_into_sections(text)
    
    print(f"formatting {len(sections)} sections...")
    formatted_sections = []

    for i, section in enumerate(sections):
        print(f"  formatting section {i+1}/{len(sections)}...")
        formatted = format_section(section, model)
        formatted_sections.append(formatted)

    return "\n\n".join(formatted_sections)