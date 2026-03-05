import os
import re
import json
from edgemind.core.config import CHUNK_SIZE, CHUNK_OVERLAP

try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("pypdf not installed, pdf support disabled")

def clean_text(text):
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' \n', '\n', text)
    text = text.strip()
    return text

def find_split_point(text, start, end):
    segment = text[start:end]
    min_pos = len(segment) // 2

    idx = segment.rfind('\n\n')
    if idx > min_pos:
        return start + idx + 2

    sentence_end = max(
        segment.rfind('. '),
        segment.rfind('! '),
        segment.rfind('? '),
        segment.rfind('.\n'),
        segment.rfind('!\n'),
        segment.rfind('?\n'),
    )
    if sentence_end > min_pos:
        return start + sentence_end + 2

    clause_end = max(
        segment.rfind(', '),
        segment.rfind('; '),
        segment.rfind(': '),
    )
    if clause_end > min_pos:
        return start + clause_end + 2

    word_end = segment.rfind(' ')
    if word_end > 0:
        return start + word_end + 1

    return end

def find_sentence_start(text, pos):
    """
    From overlap position move forward to next sentence start.
    Ensures chunks never start mid-sentence.
    """
    sentence_starters = ['. ', '! ', '? ', '.\n', '!\n', '?\n', '\n\n']

    best = None
    for starter in sentence_starters:
        idx = text.find(starter, pos)
        if idx != -1 and idx < pos + 150:
            candidate = idx + len(starter)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        best = pos
        while best < len(text) and text[best].isalpha():
            best += 1
        if best < len(text) and text[best] == ' ':
            best += 1

    return best

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = clean_text(text)
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        split = find_split_point(text, start, end)
        chunk = text[start:split].strip()
        if chunk:
            chunks.append(chunk)

        next_start = split - overlap
        next_start = find_sentence_start(text, next_start)

        if next_start <= start:
            next_start = split
        start = next_start

    return chunks

def clean_formatter_output(text, prompt):
    # strip entire prompt if echoed back
    text = text.replace(prompt, '').strip()

    # strip common leftover artifacts
    text = re.sub(r'(?i)^output the reformatted text now:\s*', '', text).strip()
    text = re.sub(r'(?i)^---end document---\s*', '', text).strip()
    text = re.sub(r'(?i)^---begin document---\s*', '', text).strip()
    text = re.sub(r'(?i)^text to rewrite:.*?\n', '', text).strip()
    text = re.sub(r'(?i)^here is the rewritten.*?\n', '', text).strip()
    text = re.sub(r'(?i)^rewritten.*?\n', '', text).strip()

    # clean up extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def format_section(text):
    """
    Reformat a section using configured provider (local or cloud).
    """
    from edgemind.core.config import MODE

    format_prompt = """ <START_OF_PROMPT>
    You are a document formatter. Your output must contain ONLY the reformatted text, nothing else.

STRICT RULES:
- Output the reformatted paragraphs ONLY
- NO rules, NO labels, NO headers, NO footers
- NO "Paragraph 1:", NO "Rule:", NO "Text:", NO "Here is..."
- NO markdown, NO bullet points, NO numbering
- NO commentary before or after the text
- Each paragraph covers exactly one person, event, or fact
- Every sentence must be complete and self contained
- Keep all names, dates, numbers, and titles exactly as written
- If you add anything other than the reformatted text, you have failed

Text to format: (UNTIL THIS LINE IS THE PROMPT NOT THE DOCUMENT CONTENT) <END_OF_PROMPT>

"""

    if MODE == "local":
        from edgemind.generation.respond import load_model
        from edgemind.core.config import MAX_TOKENS
        model = load_model()
        if model is None:
            return text
        model.reset()
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
        result = output['choices'][0]['text'].strip()
        result = clean_formatter_output(result, format_prompt)
        return result

    elif MODE == "cloud":
        from openai import OpenAI
        from edgemind.core.config import API_KEY, API_BASE_URL, MODEL_NAME
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": format_prompt},
                {"role": "user", "content": text}
            ]
        )
        result = response.choices[0].message.content.strip()
        result = clean_formatter_output(result, format_prompt)
        return result

    elif MODE == "anthropic":
        import anthropic
        from edgemind.core.config import API_KEY, MODEL_NAME
        client = anthropic.Anthropic(api_key=API_KEY)
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            system=format_prompt,
            messages=[{"role": "user", "content": text}]
        )
        result = message.content[0].text.strip()
        result = clean_formatter_output(result, format_prompt)
        return result

    return text

def split_into_sections(text, max_chars=800):
    """
    Split raw text into sections small enough for model context.
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

def parse_txt(filepath, use_llm=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    text = format_document(text, use_llm=use_llm)
    return chunk_text(text)

def parse_pdf(filepath, use_llm=True):
    if not PDF_SUPPORT:
        print(f"skipping {filepath} — pypdf not installed")
        return []
    text = ""
    with open(filepath, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    text = format_document(text, use_llm=use_llm)
    return chunk_text(text)

def parse_json(filepath, use_llm=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def flatten(obj, prefix=""):
        parts = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                parts.extend(flatten(v, f"{prefix}{k}: "))
        elif isinstance(obj, list):
            for item in obj:
                parts.extend(flatten(item, prefix))
        else:
            parts.append(f"{prefix}{obj}")
        return parts

    text = "\n".join(flatten(data))
    text = format_document(text, use_llm=use_llm)
    return chunk_text(text)

def parse_file(filepath, use_llm=True):
    ext = os.path.splitext(filepath)[1].lower()
    print(f"parsing {filepath}...")

    if ext == '.txt':
        chunks = parse_txt(filepath, use_llm=use_llm)
    elif ext == '.pdf':
        chunks = parse_pdf(filepath, use_llm=use_llm)
    elif ext == '.json':
        chunks = parse_json(filepath, use_llm=use_llm)
    else:
        print(f"unsupported format: {ext}")
        return []

    print(f"extracted {len(chunks)} chunks")
    return chunks

def parse_folder(folder, use_llm=True):
    all_chunks = []
    supported = {'.txt', '.pdf', '.json'}

    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported:
            filepath = os.path.join(folder, filename)
            chunks = parse_file(filepath, use_llm=use_llm)
            all_chunks.extend(chunks)

    print(f"\ntotal chunks extracted: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    os.makedirs("data/docs", exist_ok=True)

    sample = """robots are machines designed to perform tasks automatically.
they are widely used in manufacturing, healthcare, and space exploration.

robot motors require precise torque control for smooth and accurate operation.
without proper torque management, the robot arm will overshoot its target position.
calibration is essential for maintaining robot arm accuracy over time.

sensors provide continuous feedback to the robot control system.
the control system processes sensor data and sends corrective commands to motors.
common sensors include accelerometers, gyroscopes, and force sensors.

battery life is a critical factor in mobile robot design.
most industrial robots run on 24V or 48V DC power systems.
robot joints must be lubricated regularly to prevent mechanical wear.

computer vision allows robots to perceive and interpret their environment.
machine learning enables robots to improve their performance through experience."""

    with open("data/docs/robotics.txt", "w") as f:
        f.write(sample)

    print("=== without LLM formatter ===")
    chunks = parse_file("data/docs/robotics.txt", use_llm=False)
    for i, chunk in enumerate(chunks):
        print(f"\nchunk {i+1} ({len(chunk)} chars)")
        print(f"  starts: [{chunk[:60]}]")
        print(f"  ends:   [{chunk[-60:]}]")

    print("\n=== with LLM formatter ===")
    chunks = parse_file("data/docs/robotics.txt", use_llm=True)
    for i, chunk in enumerate(chunks):
        print(f"\nchunk {i+1} ({len(chunk)} chars)")
        print(f"  starts: [{chunk[:60]}]")
        print(f"  ends:   [{chunk[-60:]}]")
