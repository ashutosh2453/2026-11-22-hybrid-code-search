"""FastAPI service: index repositories once, then search them.

    CODESEARCH_REPOS="api=../api,web=../web" uvicorn codesearch.api:app
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from . import __version__
from .index import CodeIndex, Embedder, SentenceTransformerEmbedder

STATIC = Path(__file__).parent / "static"


def create_app(index: CodeIndex | None = None, repos: dict[str, str] | None = None, embedder: Embedder | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        idx = index
        if idx is None:
            idx = CodeIndex(embedder or SentenceTransformerEmbedder(), storage=os.getenv("CODESEARCH_STORAGE") or None)
            spec = repos or dict(p.split("=", 1) for p in os.getenv("CODESEARCH_REPOS", "").split(",") if "=" in p)
            for name, path in spec.items():
                idx.index_repo(path, name)
        app.state.index = idx
        yield

    app = FastAPI(title="Hybrid Code Search", version=__version__, lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html")

    @app.get("/stats")
    def stats():
        return app.state.index.stats()

    @app.get("/search")
    def search(q: str = Query(..., min_length=2, max_length=500), k: int = Query(10, ge=1, le=50),
               mode: str = Query("hybrid", pattern="^(hybrid|lexical|dense)$"), lang: str | None = None,
               path: str | None = Query(None, max_length=200), include_tests_docs: bool = False):
        try:
            hits = app.state.index.search(q, k=k, mode=mode, language=lang, path_glob=path,
                                          prefer_implementation=not include_tests_docs)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return {"query": q, "mode": mode, "results": [h.to_dict() for h in hits]}

    return app


app = create_app()
