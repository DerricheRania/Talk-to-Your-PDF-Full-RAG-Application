import os, re, json, math, urllib.request
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pypdf

# ── Config 
UPLOAD_DIR    = Path("uploads")
INDEX_FILE    = Path("index.json")
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K         = 5
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3.2"

UPLOAD_DIR.mkdir(exist_ok=True)
app = Flask(__name__, static_folder="static")
CORS(app)

INDEX = {"filename": None, "pages": 0, "chunks": [], "vectors": [], "vocab": {}, "idf": {}}



# PERSISTENCE

def save_index():
    INDEX_FILE.write_text(json.dumps({
        "filename": INDEX["filename"],
        "pages":    INDEX["pages"],
        "chunks":   INDEX["chunks"],
        "vectors":  INDEX["vectors"],
        "vocab":    INDEX["vocab"],
        "idf":      INDEX["idf"],
    }))
    print(f"[index] saved → {INDEX_FILE}  ({len(INDEX['chunks'])} chunks)")

def load_index():
    if not INDEX_FILE.exists():
        return
    try:
        INDEX.update(json.loads(INDEX_FILE.read_text()))
        print(f"[index] loaded '{INDEX['filename']}' — {len(INDEX['chunks'])} chunks")
    except Exception as e:
        print(f"[index] could not load: {e}")



# PDF + CHUNKING

def extract_pages(pdf_path):
    pages = []
    for i, page in enumerate(pypdf.PdfReader(str(pdf_path)).pages, 1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages

def chunk_pages(pages):
    chunks, cid = [], 0
    for p in pages:
        start = 0
        while start < len(p["text"]):
            piece = p["text"][start:start + CHUNK_SIZE].strip()
            if len(piece) > 40:
                chunks.append({"chunk_id": cid, "page": p["page"], "text": piece})
                cid += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks



# TF-IDF VECTORS

def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_index(chunks):
    vocab = {}
    for c in chunks:
        for t in _tok(c["text"]):
            if t not in vocab:
                vocab[t] = len(vocab)
    N  = len(chunks)
    df = {}
    for c in chunks:
        for t in set(_tok(c["text"])):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((N + 1) / (v + 1)) + 1 for t, v in df.items()}

    vectors = []
    for c in chunks:
        toks = _tok(c["text"])
        tf   = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        L   = len(toks) or 1
        vec = [0.0] * len(vocab)
        for t, cnt in tf.items():
            if t in vocab:
                vec[vocab[t]] = (cnt / L) * idf.get(t, 1.0)
        vectors.append(vec)

    INDEX.update({"chunks": chunks, "vectors": vectors, "vocab": vocab, "idf": idf})

def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)

def retrieve(query):
    if not INDEX["chunks"]:
        return []
    toks = _tok(query)
    tf   = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1
    L    = len(toks) or 1
    qvec = [0.0] * len(INDEX["vocab"])
    for t, cnt in tf.items():
        if t in INDEX["vocab"]:
            qvec[INDEX["vocab"][t]] = (cnt / L) * INDEX["idf"].get(t, 1.0)
    scored = sorted(
        zip([_cosine(qvec, v) for v in INDEX["vectors"]], INDEX["chunks"]),
        key=lambda x: x[0], reverse=True
    )
    return [c for _, c in scored[:TOP_K]]



# OLLAMA  (local AI — no API key needed)

def generate_answer(question, chunks):
    context = "\n\n---\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks)
    prompt  = f"""You are a precise document assistant. Answer using ONLY the context below.
- Be concise and direct.
- If the answer is not in the context, say "I couldn't find that in the document."
- Reference page numbers when helpful (e.g. "According to page 3...").

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""

    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()



# ROUTES

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "PDF only"}), 400
    path = UPLOAD_DIR / f.filename
    f.save(path)
    pages  = extract_pages(path)
    chunks = chunk_pages(pages)
    build_index(chunks)
    INDEX["filename"] = f.filename
    INDEX["pages"]    = len(pages)
    save_index()
    return jsonify({"filename": f.filename, "pages": len(pages), "chunks": len(chunks)})

@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question"}), 400
    if not INDEX["chunks"]:
        return jsonify({"error": "No document loaded. Upload a PDF first."}), 400
    try:
        hits   = retrieve(question)
        answer = generate_answer(question, hits)
        pages  = sorted(set(c["page"] for c in hits))
        return jsonify({
            "answer":       answer,
            "pages_cited":  pages,
            "chunks_used":  len(hits),
            "total_chunks": len(INDEX["chunks"]),
            "filename":     INDEX["filename"],
        })
    except Exception as e:
        err = str(e)
        if "11434" in err or "Connection refused" in err.lower():
            return jsonify({"error": "Ollama is not running. Open a terminal and run: ollama serve"}), 500
        return jsonify({"error": f"Error: {err}"}), 500

@app.route("/status")
def status():
    return jsonify({
        "loaded":   INDEX["filename"] is not None,
        "filename": INDEX["filename"],
        "pages":    INDEX.get("pages", 0),
        "chunks":   len(INDEX["chunks"]),
    })

# ── Startup 
load_index()

if __name__ == "__main__":
    print("\n🚀  Talk to Your PDF — RAG server (Ollama mode)")
    print(f"   Model: {OLLAMA_MODEL}  |  Ollama: {OLLAMA_URL}")
    if INDEX["filename"]:
        print(f"   ✅ Auto-loaded: '{INDEX['filename']}' ({len(INDEX['chunks'])} chunks)")
    print("   Open http://localhost:5000\n")
    app.run(debug=True, port=5000)