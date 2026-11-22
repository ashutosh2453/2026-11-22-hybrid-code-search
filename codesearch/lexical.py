"""Code-aware BM25: identifiers are split (camelCase, snake_case, digits) and paths/symbols are boosted."""

from __future__ import annotations

import math
import re
from collections import Counter

from .chunker import Chunk

STOP = set("""a an the and or of to in on for with from by is are be this that it as at into how where what which
when do does done we i you get set use using code file function method class return returns value values""".split())
CODE_NOISE = set("""self const let var def import export default new true false none null undefined await async
if else elif for while try except catch finally raise throw pass break continue in of public private static void
int str string bool boolean dict list number any""".split())


def split_identifier(token: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", parts)
    return [p.lower() for p in re.split(r"[^A-Za-z0-9]+|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", parts) if p]


def stem(t: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ers", "ies", "ed", "es", "er", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


def tokenize(text: str, drop_code_noise: bool = True) -> list[str]:
    out = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        for p in split_identifier(raw):
            if len(p) < 2 or p in STOP or (drop_code_noise and p in CODE_NOISE):
                continue
            out.append(stem(p))
    return out


def chunk_terms(c: Chunk) -> list[str]:
    # Path and symbol name carry strong intent ("auth", "verifyToken"), so they count three times.
    boost = tokenize(c.path.replace("/", " ") + " " + c.symbol) * 3
    return boost + tokenize(c.docstring) * 2 + tokenize(c.text)


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tfs: list[Counter] = []
        self.lengths: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def build(self, chunks: list[Chunk]) -> None:
        self.ids = [c.id for c in chunks]
        self.tfs = [Counter(chunk_terms(c)) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        n = len(chunks)
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.avgdl = sum(self.lengths) / n if n else 0.0

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q or not self.ids:
            return []
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1)))
            if s > 0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
