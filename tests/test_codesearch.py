import hashlib
import subprocess
import textwrap

import numpy as np
import pytest
from fastapi.testclient import TestClient

from codesearch.api import create_app
from codesearch.chunker import chunk_file
from codesearch.evaluate import evaluate, is_relevant
from codesearch.index import CodeIndex, is_secondary
from codesearch.lexical import BM25Index, split_identifier, tokenize


class HashingEmbedder:
    """Deterministic bag-of-words embedder: lets the full pipeline run offline in tests."""
    dim = 256
    name = "hashing-test"

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


PY = textwrap.dedent('''
    """Auth helpers."""
    import jwt

    SECRET = "x"

    def verify_token(token):
        """Check the JWT signature and expiry."""
        return jwt.decode(token, SECRET, algorithms=["HS256"])

    @dataclass
    class RateLimiter:
        """Token bucket limiter."""
        capacity: int = 10
        refill: float = 1.0

        def allow(self, key):
            return True
''')

TS = textwrap.dedent('''
    import { z } from "zod";

    export const POLICIES = `
    CREATE POLICY tenant_isolation ON flags USING (tenant_id = current_setting('app.tenant_id'));
    `;

    export function bucket(flagKey: string, userKey: string): number {
      const s = "{ not a brace }";
      return 1; // }
    }

    export class FlagService {
      constructor(private db: Db) {}

      async updateConfig(key: string, version: number): Promise<Flag> {
        if (version !== 1) {
          throw new Error("version conflict");
        }
        return this.db.get(key);
      }

      get size(): number {
        return 1;
      }
    }

    export const hashKey = (key: string) => sha256(key);
''')


# ---------------------------------------------------------------- chunking

def test_python_chunks_functions_methods_and_classes_with_fields():
    chunks = {c.symbol: c for c in chunk_file("repo/auth.py", PY)}
    assert chunks["verify_token"].kind == "function" and chunks["verify_token"].docstring == "Check the JWT signature and expiry."
    assert chunks["RateLimiter"].kind == "class" and "capacity: int = 10" in chunks["RateLimiter"].text  # fields stay with the class
    assert chunks["RateLimiter.allow"].kind == "method"
    assert any(c.kind == "module" and "SECRET" in c.text for c in chunks.values())


def test_typescript_chunks_declarations_methods_and_top_level_code():
    chunks = {c.symbol: c for c in chunk_file("repo/service.ts", TS)}
    assert chunks["bucket"].kind == "function" and chunks["bucket"].text.rstrip().endswith("}")  # braces in strings/comments ignored
    assert {"FlagService.updateConfig", "FlagService.size", "FlagService.constructor"} <= set(chunks)
    assert "version conflict" in chunks["FlagService.updateConfig"].text
    assert chunks["hashKey"].start_line == chunks["hashKey"].end_line
    assert any("CREATE POLICY" in c.text for c in chunks.values())  # top-level template string is indexed


def test_unknown_files_are_skipped_and_others_windowed():
    assert chunk_file("img.png", "x") == []
    md = "\n".join(f"line {i}" for i in range(100))
    windows = chunk_file("README.md", md)
    assert len(windows) == 3 and windows[1].start_line == 31  # 40-line windows, 10 overlap


def test_python_syntax_error_falls_back_to_windows():
    assert {c.kind for c in chunk_file("bad.py", "def (:\n  pass\n")} == {"window"}


# ---------------------------------------------------------------- lexical

def test_identifier_splitting_and_tokenizing():
    assert split_identifier("verifyJWTToken") == ["verify", "jwt", "token"]
    assert split_identifier("parse_http2_response") == ["parse", "http", "2", "response"]
    assert tokenize("const userRepository = await getUsers()") == ["user", "repository", "user"]


def test_bm25_prefers_specific_matches():
    chunks = chunk_file("r/auth.py", PY) + chunk_file("r/service.ts", TS)
    bm = BM25Index()
    bm.build(chunks)
    top = bm.search("verify jwt signature")[0][0]
    assert next(c for c in chunks if c.id == top).symbol == "verify_token"
    assert bm.search("the of and") == []


# ---------------------------------------------------------------- index & search

@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "auth.py").write_text(PY)
    (tmp_path / "src" / "service.ts").write_text(TS)
    (tmp_path / "tests" / "test_auth.py").write_text("def test_verify_token_signature():\n    assert verify_token('jwt token signature')\n")
    (tmp_path / "README.md").write_text("# Demo\nVerify JWT token signatures here.\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("function verifyToken() {}\n")
    return tmp_path


def test_index_and_hybrid_search(repo):
    idx = CodeIndex(HashingEmbedder())
    stats = idx.index_repo(repo, "demo")
    assert stats["files"] == 4  # node_modules skipped
    hits = idx.search("verify the jwt signature", k=5)
    assert hits[0].chunk.symbol == "verify_token"  # implementation first, despite test and README matches
    assert hits[0].lexical_rank and hits[0].dense_rank
    raw = idx.search("verify the jwt signature", k=5, prefer_implementation=False)
    assert {h.chunk.path for h in raw} >= {"demo/tests/test_auth.py"}
    assert all(h.chunk.language == "typescript" for h in idx.search("version conflict", language="typescript"))
    assert all(h.chunk.path.startswith("demo/src/") for h in idx.search("token", path_glob="demo/src/*"))
    with pytest.raises(ValueError):
        idx.search("x", mode="fuzzy")


def test_incremental_reindex_only_embeds_changed_chunks(repo):
    idx = CodeIndex(HashingEmbedder())
    first = idx.index_repo(repo, "demo")
    assert idx.index_repo(repo, "demo")["embedded"] == 0  # nothing changed
    (repo / "src" / "auth.py").write_text(PY.replace("algorithms=[\"HS256\"]", "algorithms=[\"HS512\"]"))
    again = idx.index_repo(repo, "demo")
    assert again["embedded"] == 1 and again["removed"] == 1
    assert idx.stats()["chunks"] == first["chunks"]


def test_lexical_only_index_without_embedder(repo):
    idx = CodeIndex(None)
    idx.index_repo(repo, "demo")
    assert idx.search("rate limiter token bucket", mode="lexical")[0].chunk.symbol == "RateLimiter"
    with pytest.raises(ValueError):
        idx.search("x", mode="hybrid")


def test_git_repos_respect_gitignore(repo):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("README.md\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    idx = CodeIndex(None)
    assert idx.index_repo(repo, "demo")["files"] == 3


def test_secondary_detection():
    from codesearch.chunker import Chunk
    mk = lambda p, lang="python": Chunk("i", p, lang, "function", "s", 1, 1, "")
    assert is_secondary(mk("r/tests/test_a.py")) and is_secondary(mk("r/src/a.test.ts", "typescript"))
    assert is_secondary(mk("r/README.md", "markdown")) and not is_secondary(mk("r/src/testing_utils.py"))


# ---------------------------------------------------------------- evaluation & API

def test_evaluate_metrics(repo):
    idx = CodeIndex(HashingEmbedder())
    idx.index_repo(repo, "demo")
    queries = [{"q": "verify jwt signature", "path": "demo/src/auth.py", "symbol": "verify_token"},
               {"q": "quantum flux capacitor", "path": "demo/src/nowhere.py", "symbol": None}]
    res = evaluate(idx, queries, modes=("hybrid",))
    assert res["metrics"]["hybrid"] == {"recall@1": 0.5, "recall@5": 0.5, "mrr@10": 0.5}
    assert is_relevant("a/b.py", "anything", {"path": "a/b.py", "symbol": None})
    assert not is_relevant("a/b.py", "foo", {"path": "a/b.py", "symbol": "^bar$"})


def test_api(repo):
    idx = CodeIndex(HashingEmbedder())
    idx.index_repo(repo, "demo")
    with TestClient(create_app(index=idx)) as c:
        assert c.get("/stats").json()["chunks"] > 0
        body = c.get("/search", params={"q": "verify jwt signature", "k": 3}).json()
        assert body["results"][0]["symbol"] == "verify_token"
        assert c.get("/search", params={"q": "x"}).status_code == 422  # too short
        assert c.get("/search", params={"q": "token", "mode": "magic"}).status_code == 422
        assert "Hybrid Code Search" in c.get("/").text


def test_cli_json_output_is_valid_json(repo, capsys):
    import json
    from codesearch.cli import main

    assert main(["search", f"demo={repo}", "rate limiter token bucket", "--mode", "lexical", "--json", "-k", "2"]) == 0
    hits = json.loads(capsys.readouterr().out)  # nothing but JSON on stdout
    assert hits[0]["symbol"] == "RateLimiter"
