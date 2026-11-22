"""codesearch CLI.

    codesearch search ~/code/app "where are jwt tokens verified"            # one-off: index then search
    codesearch search ~/code/app "rate limiting" --mode lexical --lang python
    codesearch eval --repo name=path [--repo ...] --queries eval/queries.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import evaluate, load_queries
from .index import CodeIndex, SentenceTransformerEmbedder


def _index(repos: list[str], storage: str | None, lexical_only: bool) -> CodeIndex:
    idx = CodeIndex(None if lexical_only else SentenceTransformerEmbedder(), storage=storage)
    for spec in repos:
        name, _, path = spec.partition("=") if "=" in spec else (None, "", spec)
        stats = idx.index_repo(path, name or None)
        print(f"indexed {stats['repo']}: {stats['files']} files, {stats['chunks']} chunks ({stats['embedded']} embedded)", file=sys.stderr)
    return idx


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="codesearch", description="Hybrid (BM25 + embeddings) semantic code search")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("repo", nargs="+", help="repository path(s) (optionally name=path); the last positional is the query")
    s.add_argument("--mode", choices=["hybrid", "lexical", "dense"], default="hybrid")
    s.add_argument("--lang")
    s.add_argument("--path", help="glob filter on 'repo/path'")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--json", action="store_true")
    s.add_argument("--include-tests-docs", action="store_true", help="disable the implementation-first prior")
    e = sub.add_parser("eval")
    e.add_argument("--repo", action="append", required=True, help="name=path, repeatable")
    e.add_argument("--queries", type=Path, required=True)
    e.add_argument("--out", type=Path)
    for x in (s, e):
        x.add_argument("--storage", help="persist the Qdrant index in this directory")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        if args.cmd == "search":
            if len(args.repo) < 2:
                p.error("give at least one repository and a query")
            *repos, query = args.repo
            idx = _index(repos, args.storage, args.mode == "lexical")
            hits = idx.search(query, k=args.k, mode=args.mode, language=args.lang, path_glob=args.path,
                              prefer_implementation=not args.include_tests_docs)
            if args.json:
                print(json.dumps([h.to_dict() for h in hits], indent=2))
            for i, h in enumerate(hits, start=1):
                c = h.chunk
                print(f"\n{i}. {c.path}:{c.start_line}-{c.end_line}  {c.kind} {c.symbol}  "
                      f"(bm25 #{h.lexical_rank or '-'}, dense #{h.dense_rank or '-'})")
                print("\n".join("   " + l for l in c.text.split("\n")[:6]))
        else:
            idx = _index(args.repo, args.storage, False)
            res = evaluate(idx, load_queries(args.queries))
            print(json.dumps(res["metrics"], indent=2))
            if args.out:
                args.out.write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
