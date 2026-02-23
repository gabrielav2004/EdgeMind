import os
import re
import json

try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("pypdf not installed, pdf support disabled")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text = clean_text(text)
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            boundary = max(
                text.rfind('.', start, end),
                text.rfind('?', start, end),
                text.rfind('\n', start, end)
            )
            if boundary > start + (chunk_size // 2):
                end = boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
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
    # create sample robotics doc
    os.makedirs("data/docs", exist_ok=True)
    
    sample = """robots are machines that can perform tasks automatically.
they are used in manufacturing, healthcare, and exploration.
robot motors require precise torque control for smooth operation.
calibration is essential for maintaining robot arm accuracy.
sensors provide feedback to the robot control system.
the control system processes sensor data and sends commands to motors.
battery life is a critical factor in mobile robot design.
robot joints must be lubricated regularly to prevent wear.
computer vision allows robots to perceive their environment.
machine learning enables robots to improve their performance over time."""

    with open("data/docs/robotics.txt", "w") as f:
        f.write(sample)
    
    print("--- parsing single file ---")
    chunks = parse_file("data/docs/robotics.txt")
    
    print("\n--- extracted chunks ---")
    for i, chunk in enumerate(chunks):
        print(f"\nchunk {i+1}: '{chunk[:80]}'")
    
    print("\n--- parsing folder ---")
    all_chunks = parse_folder("data/docs")
    print(f"ready to store {len(all_chunks)} chunks")