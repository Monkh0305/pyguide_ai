"""Download W3Schools Python lessons and build the local RAG dataset."""
import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from rag_store import CACHE_VERSION

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BASE = "https://www.w3schools.com/python/"
LESSONS = [
    "intro", "getstarted", "syntax", "comments", "variables",
    "variables_names", "variables_multiple", "variables_output", "variables_global",
    "datatypes", "numbers", "casting", "strings", "strings_slicing",
    "strings_modify", "strings_concatenate", "strings_format", "strings_escape",
    "booleans", "operators", "lists", "lists_access", "lists_change",
    "lists_add", "lists_remove", "lists_loop", "lists_comprehension", "lists_sort",
    "lists_copy", "lists_join", "tuples", "sets", "dictionaries", "conditions",
    "while_loops", "for_loops", "functions", "lambda", "arrays", "classes",
    "inheritance", "iterators", "polymorphism", "scope", "modules", "datetime",
    "math", "json", "regex", "pip", "try_except", "user_input",
    "file_handling", "file_open", "file_write", "file_remove",
]


def fetch_lesson(session, url):
    response = session.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    main = soup.select_one("#main")
    if main is None or main.h1 is None:
        raise ValueError(f"Missing lesson content: {url}")
    title = main.h1.get_text(" ", strip=True)
    for node in main.select("script, style, iframe, .w3-clear, .w3-btn, .w3-button, form, #mypagediv, .w3-ad, .ad, #exercisecontainer"):
        node.decompose()
    text = main.get_text("\n", strip=True)
    if len(text) < 100:
        raise ValueError(f"Empty lesson: {url}")
    return {"title": title, "url": url, "text": text}


def chunk_text(text, size=900, overlap=150):
    for start in range(0, len(text), size - overlap):
        part = text[start:start + size].strip()
        if part:
            yield part
        if start + size >= len(text):
            break


def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # Use the same cached model as the application; no API key is needed.
    model = SentenceTransformer(MODEL, local_files_only=True)
    docs = []
    urls = [BASE + "python_" + lesson + ".asp" for lesson in LESSONS]
    with requests.Session() as session:
        session.headers["User-Agent"] = "PyGuideAI/1.0 (Python learning dataset)"
        for i, url in enumerate(urls, 1):
            docs.append(fetch_lesson(session, url))
            print(f"Downloaded {i}/{len(urls)}: {docs[-1]['title']}", flush=True)
            time.sleep(0.2)
    chunks = [
        {"text": text, "source": doc["title"], "url": doc["url"], "chunk_id": i}
        for doc in docs for i, text in enumerate(chunk_text(doc["text"]))
    ]
    vectors = np.asarray(model.encode(
        [c["text"] for c in chunks], normalize_embeddings=True,
        show_progress_bar=True), dtype="float32")
    signature = {"version": CACHE_VERSION, "urls": urls, "model": MODEL,
                 "chunk_size": 900, "overlap": 150}
    metadata = {"signature": signature, "created_at": time.time(),
                "docs": docs, "chunks": chunks, "errors": []}
    info = {"source": "W3Schools Python Tutorial", "documents": len(docs),
            "chunks": len(chunks), "embedding_shape": list(vectors.shape),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "signature": signature, "errors": []}
    data = Path(__file__).parent / "data"
    data.mkdir(exist_ok=True)
    # Finish downloading and encoding before replacing the existing dataset.
    with tempfile.TemporaryDirectory(dir=data) as staging:
        stage = Path(staging)
        np.savez_compressed(stage / "rag.npz", metadata=json.dumps(metadata, ensure_ascii=False), vectors=vectors)
        for name, rows in [("documents", docs), ("chunks", chunks)]:
            (stage / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            with (stage / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (stage / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        (stage / "README.txt").write_text(
            f"PyGuide AI: W3Schools Python Tutorial\nDocuments: {len(docs)}\nChunks: {len(chunks)}\n"
            "Source: https://www.w3schools.com/python/\n"
            "documents.json/csv: title, url, text\nchunks.json/csv: text, source, url, chunk_id\n"
            "rag.npz: metadata and normalized embedding vectors used by app.py\n"
            "dataset_info.json: model, parameters, download time\n"
            "Refresh: .venv\\Scripts\\python.exe download_data.py\n"
            "Content belongs to W3Schools. Terms: https://www.w3schools.com/about/about_copyright.asp\n",
            encoding="utf-8")
        for file in stage.iterdir():
            os.replace(file, data / file.name)
    print(f"Ready: {len(docs)} documents, {len(chunks)} chunks, {vectors.shape}")


if __name__ == "__main__":
    main()
