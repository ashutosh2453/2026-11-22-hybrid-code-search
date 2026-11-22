"""Split source files into semantically meaningful chunks (functions, classes, methods).

Python is parsed with `ast`. JavaScript/TypeScript use a brace-matching scanner that
understands strings, template literals and comments. Everything else falls back to
overlapping line windows. Each chunk records its symbol, kind and exact line range so a
search result can point to `path:line`.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".md": "markdown", ".sql": "sql", ".sh": "shell", ".yml": "yaml", ".yaml": "yaml", ".toml": "toml",
    ".html": "html", ".css": "css", ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp", ".kt": "kotlin",
}
MAX_CHUNK_LINES = 120
WINDOW, OVERLAP = 40, 10


@dataclass
class Chunk:
    id: str
    path: str
    language: str
    kind: str  # function | method | class | module | window
    symbol: str
    start_line: int
    end_line: int
    text: str
    docstring: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def language_of(path: str) -> str | None:
    m = re.search(r"\.[A-Za-z0-9]+$", path)
    return LANGUAGES.get(m.group(0).lower()) if m else None


def _make(path, language, kind, symbol, start, end, lines, docstring="") -> Chunk:
    end = min(end, start + MAX_CHUNK_LINES - 1)
    text = "\n".join(lines[start - 1:end])
    cid = hashlib.sha1(f"{path}:{start}:{end}:{text}".encode()).hexdigest()[:16]
    return Chunk(cid, path, language, kind, symbol, start, end, text, docstring)


def _windows(path: str, language: str, lines: list[str]) -> list[Chunk]:
    out, start = [], 1
    while start <= len(lines):
        end = min(len(lines), start + WINDOW - 1)
        if any(l.strip() for l in lines[start - 1:end]):
            out.append(_make(path, language, "window", f"{path.rsplit('/', 1)[-1]}:{start}", start, end, lines))
        if end == len(lines):
            break
        start = end - OVERLAP + 1
    return out


def chunk_python(path: str, source: str) -> list[Chunk]:
    lines = source.split("\n")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _windows(path, "python", lines)
    out: list[Chunk] = []
    covered: set[int] = set()

    def visit(node: ast.AST, prefix: str, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                end = child.end_lineno or child.lineno
                name = f"{prefix}{child.name}"
                if isinstance(child, ast.ClassDef):
                    # Class chunk = header, docstring and class-level statements (e.g. dataclass fields)
                    # up to the first method; methods become their own chunks.
                    first_def = next((n for n in child.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), None)
                    if first_def is None:
                        header_end = end
                    else:
                        first_start = min([first_def.lineno] + [d.lineno for d in first_def.decorator_list])
                        header_end = max(start, first_start - 1)
                    doc = ast.get_docstring(child) or ""
                    out.append(_make(path, "python", "class", name, start, header_end, lines, doc))
                    covered.update(range(start, header_end + 1))
                    visit(child, name + ".", True)
                else:
                    out.append(_make(path, "python", "method" if in_class else "function", name, start, end, lines,
                                     ast.get_docstring(child) or ""))
                    covered.update(range(start, end + 1))

    visit(tree, "", False)
    # module-level code (imports, constants, script blocks) not inside any def/class
    rest = [i for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()]
    if rest:
        module_doc = ast.get_docstring(tree) or ""
        block: list[int] = []
        for i in rest:
            if block and i - block[-1] > 3:
                out.append(_make(path, "python", "module", f"{path.rsplit('/', 1)[-1]}:{block[0]}", block[0], block[-1], lines, module_doc))
                block = []
            block.append(i)
        if block:
            out.append(_make(path, "python", "module", f"{path.rsplit('/', 1)[-1]}:{block[0]}", block[0], block[-1], lines, module_doc))
    return sorted(out, key=lambda c: c.start_line)


JS_DECL = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?"
    r"(?:function\s*\*?\s*(?P<fn>[A-Za-z_$][\w$]*)"
    r"|class\s+(?P<cls>[A-Za-z_$][\w$]*)"
    r"|(?:const|let|var)\s+(?P<var>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>"
    r"|(?:const|let|var)\s+(?P<fexpr>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?function\b"
    r"|(?P<route>(?:router|app)\.(?:get|post|put|patch|delete|use)\s*\(\s*['\"`][^'\"`]*['\"`]))"
)
JS_COMMENT_DOC = re.compile(r"/\*\*(.*?)\*/", re.S)
# Class members: `async updateConfig(p: Principal, ...) {`, `static create() {`, `get size() {`
JS_METHOD = re.compile(
    r"^\s+(?:(?:public|private|protected|static|async|readonly|override|get|set)\s+)*"
    r"(?P<name>[A-Za-z_$#][\w$]*)\s*(?:<[^>()]*>)?\s*\(")
JS_NOT_METHOD = {"if", "for", "while", "switch", "catch", "return", "function", "constructor_", "else", "do", "with", "typeof", "new", "await"}


def _block_end(lines: list[str], start: int) -> int:
    """Line (1-based) where the brace block opened on/after `start` closes; skips strings and comments."""
    depth, opened = 0, False
    in_str: str | None = None
    in_block_comment = False
    for li in range(start - 1, len(lines)):
        line = lines[li]
        i = 0
        while i < len(line):
            ch = line[i]
            nxt = line[i + 1] if i + 1 < len(line) else ""
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment, i = False, i + 2
                    continue
            elif in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
            elif ch == "/" and nxt == "/":
                break
            elif ch == "/" and nxt == "*":
                in_block_comment, i = True, i + 2
                continue
            elif ch in "'\"`":
                in_str = ch
            elif ch == "{":
                depth, opened = depth + 1, True
            elif ch == "}":
                depth -= 1
                if opened and depth == 0:
                    return li + 1
            i += 1
        if in_str in ("'", '"'):
            in_str = None  # unterminated single-line string
        if not opened and li - start > 3:
            return start  # declaration without a block (e.g. one-line arrow function)
    return len(lines) if opened else start  # unterminated block runs to EOF; no block at all ends here


def _class_methods(lines: list[str], start: int, end: int, cls: str, path: str, language: str) -> list[Chunk]:
    """Split a JS/TS class body into one chunk per method (methods have no `function` keyword)."""
    out: list[Chunk] = []
    li = start + 1
    while li < end:
        line = lines[li - 1]
        m = JS_METHOD.match(line)
        if m and m.group("name") not in JS_NOT_METHOD and not line.rstrip().endswith(";"):
            m_end = _block_end(lines, li)
            if m_end > li or "{" in line:  # one-line bodies such as `constructor(private db: Db) {}`
                doc = ""
                prev = "\n".join(lines[max(0, li - 8):li - 1])
                docs = JS_COMMENT_DOC.findall(prev)
                if docs and prev.rstrip().endswith("*/"):
                    doc = re.sub(r"\s*\*\s?", " ", docs[-1]).strip()
                out.append(_make(path, language, "method", f"{cls}.{m.group('name')}", li, m_end, lines, doc))
                li = m_end + 1
                continue
        li += 1
    return out


def chunk_js(path: str, source: str, language: str) -> list[Chunk]:
    lines = source.split("\n")
    out: list[Chunk] = []
    i = 1
    while i <= len(lines):
        m = JS_DECL.match(lines[i - 1])
        if not m:
            i += 1
            continue
        name = m.group("fn") or m.group("cls") or m.group("var") or m.group("fexpr") or m.group("route")
        kind = "class" if m.group("cls") else ("route" if m.group("route") else "function")
        end = _block_end(lines, i)
        doc = ""
        if i > 1:
            prev = "\n".join(lines[max(0, i - 8):i - 1])
            docs = JS_COMMENT_DOC.findall(prev)
            if docs and prev.rstrip().endswith("*/"):
                doc = re.sub(r"\s*\*\s?", " ", docs[-1]).strip()
        if kind == "class":
            methods = _class_methods(lines, i, max(end, i), name.strip(), path, language)
            header_end = (methods[0].start_line - 1) if methods else max(end, i)
            out.append(_make(path, language, "class", name.strip(), i, max(i, header_end), lines, doc))
            out.extend(methods)
            i = max(end, i) + 1
            continue
        out.append(_make(path, language, kind, name.strip(), i, max(end, i), lines, doc))
        i = max(end, i) + 1
    if not out:
        return _windows(path, language, lines)
    # Top-level code outside any declaration (constants, SQL/template strings, config objects)
    # would otherwise never be indexed: group uncovered non-blank lines into module chunks.
    covered = {ln for c in out for ln in range(c.start_line, c.end_line + 1)}
    block: list[int] = []

    def flush() -> None:
        if block:
            for s in range(block[0], block[-1] + 1, MAX_CHUNK_LINES):
                out.append(_make(path, language, "module", f"{path.rsplit('/', 1)[-1]}:{s}", s,
                                 min(block[-1], s + MAX_CHUNK_LINES - 1), lines))

    for ln in range(1, len(lines) + 1):
        if ln in covered or not lines[ln - 1].strip():
            continue
        if block and ln - block[-1] > 3:
            flush()
            block = []
        block.append(ln)
    flush()
    return sorted(out, key=lambda c: c.start_line)


def chunk_file(path: str, source: str) -> list[Chunk]:
    language = language_of(path)
    if language is None:
        return []
    source = source.replace("\r\n", "\n")
    if language == "python":
        return chunk_python(path, source)
    if language in ("javascript", "typescript"):
        return chunk_js(path, source, language)
    return _windows(path, language, source.split("\n"))
