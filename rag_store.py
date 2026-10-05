"""Disk cache for RAG; imported only when retrieval is needed."""
import json
import os
import tempfile
import time
from zipfile import BadZipFile
from concurrent.futures import ThreadPoolExecutor

CACHE_VERSION = 1
MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def load_or_build(urls, model_name, chunk_size, overlap, loader, chunker, cache_path,
                  model_factory=None):
    import numpy as np
    import faiss
    signature = {"version": CACHE_VERSION, "urls": list(urls), "model": model_name,
                 "chunk_size": chunk_size, "overlap": overlap}
    cached = None
    try:
        with np.load(cache_path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata"].item()))
            vectors = artifact["vectors"]
            if (metadata["signature"] == signature
                    and 0 <= time.time() - metadata["created_at"] < MAX_AGE_SECONDS
                    and vectors.ndim == 2 and vectors.shape[1] > 0
                    and len(vectors) == len(metadata["chunks"]) and len(vectors) > 0
                    and np.isfinite(vectors).all()):
                cached = (metadata, vectors)
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        pass
    if model_factory is None:
        from sentence_transformers import SentenceTransformer
        model_factory = SentenceTransformer
    model = model_factory(model_name)
    if cached is not None:
        metadata, vectors = cached
    else:
        def fetch(url):
            try:
                return loader(url), None
            except Exception as exc:
                return None, f"{url}: {exc}"
        with ThreadPoolExecutor(max_workers=4) as pool:
            fetched = list(pool.map(fetch, urls))
        docs = [doc for doc, error in fetched if doc is not None]
        errors = [error for doc, error in fetched if error is not None]
        chunks = [
            {"text": text, "source": doc["title"], "url": doc["url"], "chunk_id": i}
            for doc in docs for i, text in enumerate(chunker(doc["text"], chunk_size, overlap))
        ]
        if not chunks:
            raise RuntimeError("ไม่สามารถโหลดข้อมูลจาก Python Documentation ได้")
        vectors = np.asarray(model.encode(
            [chunk["text"] for chunk in chunks], normalize_embeddings=True,
            show_progress_bar=False), dtype="float32")
        metadata = {"signature": signature, "created_at": time.time(),
                    "docs": docs, "errors": errors, "chunks": chunks}
        if not errors:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz",
                                                  delete=False) as temp:
                    temp_path = temp.name
                    np.savez_compressed(temp, metadata=json.dumps(metadata, ensure_ascii=False),
                                        vectors=vectors)
                os.replace(temp_path, cache_path)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.asarray(vectors, dtype="float32"))
    return model, index, metadata["chunks"], metadata["docs"], metadata["errors"]


def load_local(data_path, model_name, model_factory=None):
    """Read permanent downloaded data without fetching documentation."""
    import numpy as np
    import faiss

    try:
        with np.load(data_path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact['metadata'].item()))
            vectors = artifact['vectors']
        if (metadata['signature']['model'] != model_name
                or metadata['signature']['version'] != CACHE_VERSION
                or vectors.ndim != 2 or vectors.shape[1] == 0
                or len(vectors) == 0 or len(vectors) != len(metadata['chunks'])
                or not np.isfinite(vectors).all() or metadata.get('errors')):
            raise ValueError('Invalid or incomplete local dataset')
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        raise RuntimeError(f'Cannot load local dataset: {data_path}') from exc

    if model_factory is None:
        from sentence_transformers import SentenceTransformer
        # Cloud instances start without the model cache; download it when needed.
        model = SentenceTransformer(model_name)
    else:
        model = model_factory(model_name)
    if model.get_sentence_embedding_dimension() != vectors.shape[1]:
        raise RuntimeError('Local dataset embedding dimension mismatch')
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.asarray(vectors, dtype='float32'))
    return model, index, metadata['chunks'], metadata['docs'], metadata.get('errors', [])
