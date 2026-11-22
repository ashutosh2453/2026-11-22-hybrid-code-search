"""Retrieval benchmark: Recall@1, Recall@5 and MRR@10 for lexical, dense and hybrid search."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .index import CodeIndex


def is_relevant(hit_path: str, hit_symbol: str, expected: dict) -> bool:
    if hit_path != expected["path"]:
        return False
    return expected.get("symbol") is None or re.search(expected["symbol"], hit_symbol) is not None


def evaluate(index: CodeIndex, queries: list[dict], modes=("lexical", "dense", "hybrid"), k: int = 10) -> dict:
    results = {}
    per_query = []
    for mode in modes:
        r1 = r5 = rr = 0.0
        for q in queries:
            hits = index.search(q["q"], k=k, mode=mode)
            rank = next((i for i, h in enumerate(hits, start=1) if is_relevant(h.chunk.path, h.chunk.symbol, q)), None)
            r1 += rank == 1
            r5 += rank is not None and rank <= 5
            rr += 1.0 / rank if rank else 0.0
            per_query.append({"mode": mode, "query": q["q"], "rank": rank})
        n = len(queries)
        results[mode] = {"recall@1": round(r1 / n, 3), "recall@5": round(r5 / n, 3), "mrr@10": round(rr / n, 3)}
    return {"queries": len(queries), "metrics": results, "per_query": per_query}


def load_queries(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for q in data:
        if not {"q", "path"} <= set(q):
            raise ValueError(f"query entry needs 'q' and 'path': {q}")
    return data
