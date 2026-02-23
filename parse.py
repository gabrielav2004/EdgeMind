import os
import re
import json

try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("pypdf not installed, pdf support disabled")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

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

def find_word_boundary_forward(text, pos):
    while pos < len(text) and text[pos].isalpha():
        pos += 1
    return pos

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
        next_start = find_word_boundary_forward(text, next_start)

        if next_start <= start:
            next_start = split
        start = next_start

    return chunks

def parse_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return chunk_text(text)

def parse_pdf(filepath):
    if not PDF_SUPPORT:
        print(f"skipping {filepath} — pypdf not installed")
        return []
    text = ""
    with open(filepath, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return chunk_text(text)

def parse_json(filepath):
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
    return chunk_text(text)

def parse_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    print(f"parsing {filepath}...")

    if ext == '.txt':
        chunks = parse_txt(filepath)
    elif ext == '.pdf':
        chunks = parse_pdf(filepath)
    elif ext == '.json':
        chunks = parse_json(filepath)
    else:
        print(f"unsupported format: {ext}")
        return []

    print(f"extracted {len(chunks)} chunks")
    return chunks

def parse_folder(folder):
    all_chunks = []
    supported = {'.txt', '.pdf', '.json'}

    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported:
            filepath = os.path.join(folder, filename)
            chunks = parse_file(filepath)
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

    chunks = parse_file("data/docs/robotics.txt")
    print(f"\n--- chunks ---")
    for i, chunk in enumerate(chunks):
        print(f"\nchunk {i+1} ({len(chunk)} chars)")
        print(f"  starts: [{chunk[:50]}]")
        print(f"  ends:   [{chunk[-50:]}]")