"""Code index: chunk a repository, embed chunks into Qdrant, keep a BM25 index, search with RRF fusion."""

from __future__ import annotations

import fnmatch
import re
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .chunker import Chunk, chunk_file, language_of
from .lexical import BM25Index

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", ".pytest_cache", "coverage"}
MAX_FILE_BYTES = 400_000
DEFAULT_MODEL = os.getenv("CODESEARCH_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)


SECONDARY_WEIGHT = 0.85
ASKS_FOR_TESTS_OR_DOCS = re.compile(r"\b(test|tests|spec|readme|docs?|documentation|ci|workflow|pipeline yaml)\b", re.I)
TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]+$|_test\.\w+$|\.(test|spec)\.\w+$")


def is_secondary(c: Chunk) -> bool:
    """Tests, documentation and CI/config files: relevant context, rarely the answer itself."""
    return bool(TEST_PATH.search(c.path)) or c.language in ("markdown", "yaml", "toml")


def embedding_text(c: Chunk) -> str:
    """What the dense model sees: where the code lives, what it is called, its doc, then the code."""
    head = f"{c.kind} {c.symbol} in {c.path}"
    return "\n".join(x for x in (head, c.docstring, c.text) if x)[:2000]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    lexical_rank: int | None
    dense_rank: int | None

    def to_dict(self) -> dict:
        d = self.chunk.to_dict()
        d.update(score=round(self.score, 5), lexical_rank=self.lexical_rank, dense_rank=self.dense_rank)
        return d


def list_files(root: Path, include: list[str] | None = None) -> list[str]:
    """Files git knows about or would add (tracked + untracked, minus .gitignore'd), else a filtered walk.

    The walk fallback also covers a folder nested inside an unrelated parent repository whose
    index knows nothing about it (e.g. a home directory that happens to be a git repo).
    """
    paths: list[str] = []
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             check=True, capture_output=True).stdout
        paths = [p for p in dict.fromkeys(out.decode("utf-8", "replace").split("\0")) if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        paths = []
    if not paths:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                paths.append(str(Path(dirpath, f).relative_to(root)).replace("\\", "/"))
    paths = [p for p in paths if language_of(p) and not any(part in SKIP_DIRS for part in p.split("/"))]
    if include:
        paths = [p for p in paths if any(fnmatch.fnmatch(p, g) for g in include)]
    return sorted(paths)


class CodeIndex:
    def __init__(self, embedder: Embedder | None, storage: str | None = None, collection: str = "code") -> None:
        from qdrant_client import QdrantClient

        self.embedder = embedder
        self.client = QdrantClient(path=storage) if storage else QdrantClient(location=":memory:")
        self.collection = collection
        self.chunks: dict[str, Chunk] = {}
        self.bm25 = BM25Index()
        self.repos: dict[str, str] = {}

    # ---------------------------------------------------------------- indexing
    def index_repo(self, root: str | Path, name: str | None = None, include: list[str] | None = None) -> dict:
        root = Path(root)
        name = name or root.resolve().name
        new_chunks: list[Chunk] = []
        for rel in list_files(root, include):
            p = root / rel
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                source = p.read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for c in chunk_file(f"{name}/{rel}", source):
                new_chunks.append(c)
        # Incremental: chunk ids are content hashes, so unchanged chunks are not re-embedded.
        old_ids = {cid for cid, c in self.chunks.items() if c.path.startswith(f"{name}/")}
        fresh = [c for c in new_chunks if c.id not in self.chunks]
        stale = old_ids - {c.id for c in new_chunks}
        for cid in stale:
            del self.chunks[cid]
        for c in new_chunks:
            self.chunks[c.id] = c
        self._upsert_vectors(fresh, stale)
        self.bm25.build(list(self.chunks.values()))
        self.repos[name] = str(root)
        return {"repo": name, "files": len({c.path for c in new_chunks}), "chunks": len(new_chunks),
                "embedded": len(fresh), "removed": len(stale)}

    def _upsert_vectors(self, chunks: list[Chunk], stale: set[str]) -> None:
        if self.embedder is None:
            return
        from qdrant_client import models

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE))
        if stale:
            self.client.delete(self.collection, points_selector=models.PointIdsList(points=[_point_id(c) for c in stale]))
        for i in range(0, len(chunks), 256):
            batch = chunks[i:i + 256]
            vectors = self.embedder.embed([embedding_text(c) for c in batch])
            self.client.upsert(self.collection, points=[
                models.PointStruct(id=_point_id(c.id), vector=v.tolist(), payload={"cid": c.id, "path": c.path, "language": c.language})
                for c, v in zip(batch, vectors)])

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int = 10, mode: str = "hybrid", language: str | None = None,
               path_glob: str | None = None, rrf_k: int = 60, candidates: int = 50,
               prefer_implementation: bool = True) -> list[Hit]:
        if mode not in ("hybrid", "lexical", "dense"):
            raise ValueError("mode must be hybrid, lexical or dense")
        if mode != "lexical" and self.embedder is None:
            raise ValueError("dense search needs an embedder")

        def allowed(cid: str) -> bool:
            c = self.chunks.get(cid)
            return c is not None and (not language or c.language == language) and (not path_glob or fnmatch.fnmatch(c.path, path_glob))

        lexical = [cid for cid, _ in self.bm25.search(query, candidates * 3) if allowed(cid)][:candidates] if mode != "dense" else []
        dense: list[str] = []
        if mode != "lexical" and self.client.collection_exists(self.collection):
            qv = self.embedder.embed([query])[0]
            res = self.client.query_points(self.collection, query=qv.tolist(), limit=candidates * 3, with_payload=True).points
            dense = [p.payload["cid"] for p in res if allowed(p.payload["cid"])][:candidates]

        lex_rank = {cid: r for r, cid in enumerate(lexical, start=1)}
        den_rank = {cid: r for r, cid in enumerate(dense, start=1)}
        scores: dict[str, float] = {}
        for ranking in (lexical, dense):
            for r, cid in enumerate(ranking, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + r)
        if prefer_implementation and not ASKS_FOR_TESTS_OR_DOCS.search(query):
            # Most code-search questions want the implementation; tests, docs and CI config are
            # useful but should not crowd it out. Only a mild prior: strong matches still win.
            for cid in scores:
                if is_secondary(self.chunks[cid]):
                    scores[cid] *= SECONDARY_WEIGHT
        ordered = sorted(scores, key=lambda cid: -scores[cid])[:k]
        return [Hit(self.chunks[cid], scores[cid], lex_rank.get(cid), den_rank.get(cid)) for cid in ordered]

    def stats(self) -> dict:
        langs: dict[str, int] = {}
        for c in self.chunks.values():
            langs[c.language] = langs.get(c.language, 0) + 1
        return {"repos": self.repos, "chunks": len(self.chunks), "by_language": langs,
                "embedder": getattr(self.embedder, "name", None)}

    def save_chunks(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([c.to_dict() for c in self.chunks.values()]), encoding="utf-8", newline="\n")


def _point_id(cid: str) -> str:
    # Qdrant point ids must be unsigned ints or UUIDs; derive a stable UUID from the chunk id.
    h = hashlib.md5(cid.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
