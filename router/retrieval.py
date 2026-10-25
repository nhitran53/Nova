"""Offline document retrieval for the grounded-answer path: an in-memory SQLite FTS5 index over a markdown corpus.

FTS5 (porter unicode61) only fetches and ranks candidates. The score handed to TurnHandler is the query's content-term
COVERAGE of a chunk: matched non-stopword query terms / all query content terms, in [0, 1]. It does not depend on corpus
size (BM25 idf collapses on tiny corpora). A chunk matching fewer than two query terms is capped below the 0.5
acceptance threshold, unless the query is a single content term equal to the chunk's title or heading; so a two-term
query must match both terms to be accepted.

make_retriever(dir) -> callable(query) -> [{"text", "score", "id", "source", "title", "sha256"}] or None.
find_section(retriever, query) -> best document {"id", "source", "title", "score"} for display, or None.
"""
import hashlib
import re
import sqlite3
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

MIN_CHUNK_WORDS = 60
MAX_CHUNK_WORDS = 200
MIN_SECTION_SCORE = 0.5
SINGLE_TERM_CAP = 0.49
MAX_RESULTS = 5
MAX_QUERY_TERMS = 12
MAX_TERM_LENGTH = 40
MAX_QUERY_CHARS = 500

# Function words plus voice filler ("hey nova, show me the manual page on ...") and repair-intent words that carry no
# topic ("damaged", "replacement"): the topic is the part left over.
STOPWORDS = frozenset("""
a an the and or not but of to in on at for with by from as into onto up out off over about around
is are was were be been am do does did done have has had having i me my mine you your we us our it its this that these those
there here what whats which who whom whose where when why how can could would should will shall may might must
please nova hey hi hello ok okay thanks thank show tell find open pull bring give get got take go see look looking need
needs want wants like let lets know
manual manuals page pages section sections document documents doc docs guide instructions
any some anything something one ones
much many long far little left right more most very really just also still way thing things kind
damaged broken cracked replacement replace spare
""".split())

# Hand-written demo phrases -> the manual's vocabulary. Longest phrase wins; applied on whole-word boundaries.
ALIASES = {
    "tire": "wheel", "tires": "wheels", "tyre": "wheel", "tyres": "wheels",
    "left tire": "wheel drive module", "right tire": "wheel drive module",
    "left tyre": "wheel drive module", "right tyre": "wheel drive module",
    "left wheel": "wheel drive module", "right wheel": "wheel drive module",
    "front wheel": "wheel drive module", "rear wheel": "wheel drive module",
    "front tire": "wheel drive module", "rear tire": "wheel drive module",
}

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_IMAGE_ONLY = re.compile(r"^\s*(?:!\[[^\]]*\]\([^)]*\)\s*)+$")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_TAG = re.compile(r"</?[A-Za-z][^>]*>")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    title: str
    heading_path: tuple
    text: str
    sha256: str


def _log(message):
    print(f"retrieval: {message}", file=sys.stderr)


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _words(text):
    return len(text.split())


def _clean_line(line):
    line = _IMAGE.sub("", line)
    line = _LINK.sub(r"\1", line)
    line = _URL.sub("", line)
    line = _TAG.sub("", line)
    line = line.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"[ \t]+", " ", line).rstrip()


def _clean_heading(text):
    return _clean_line(text).strip(" *_")


def _sections(markdown, fallback_title):
    """-> (title, [(heading_path, [paragraph, ...])]) for one markdown file; headings in code fences are not headings."""
    stack, sections, paragraphs, current = [], [], [], []
    title = None
    in_fence = False
    path = ()

    def flush():
        if current:
            paragraphs.append("\n".join(current))
            current.clear()
        if paragraphs:
            sections.append((path, list(paragraphs)))
            paragraphs.clear()

    for raw in markdown.splitlines():
        if _FENCE.match(raw):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(raw)
        if heading:
            flush()
            level, text = len(heading.group(1)), _clean_heading(heading.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text or "section"))
            if title is None and level == 1:
                title = stack[0][1]
            path = tuple(name for _, name in stack)
            continue
        if not in_fence and _IMAGE_ONLY.match(raw):
            continue
        line = _clean_line(raw)
        if not in_fence and re.fullmatch(r"[-*_=\s]{3,}", line):
            line = ""
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current.clear()
    flush()
    has_h1 = title is not None
    title = title or fallback_title
    sections = [(path if path and has_h1 else (title,) + path, paras) for path, paras in sections]
    return title, sections


def _split_line(line):
    """A line over MAX_CHUNK_WORDS words -> consecutive runs of at most MAX_CHUNK_WORDS words."""
    words = line.split()
    return [" ".join(words[i:i + MAX_CHUNK_WORDS]) for i in range(0, len(words), MAX_CHUNK_WORDS)]


def _split_block(block):
    if _words(block) <= MAX_CHUNK_WORDS:
        return [block]
    pieces, group = [], []
    lines = [part for line in block.split("\n") for part in (_split_line(line) if _words(line) > MAX_CHUNK_WORDS else [line])]
    for line in lines:
        if group and _words("\n".join(group + [line])) > MAX_CHUNK_WORDS:
            pieces.append("\n".join(group))
            group = []
        group.append(line)
    if group:
        pieces.append("\n".join(group))
    return pieces


def _pieces(sections):
    """Sections -> [(heading_path, body)] with every body <= MAX_CHUNK_WORDS (long ones split on paragraph boundaries)."""
    out = []
    for path, paragraphs in sections:
        group = []
        for block in (b for p in paragraphs for b in _split_block(p)):
            if group and _words("\n\n".join(group + [block])) > MAX_CHUNK_WORDS:
                out.append((path, "\n\n".join(group)))
                group = []
            group.append(block)
        if group:
            out.append((path, "\n\n".join(group)))
    return out


def _joined(prev, nxt):
    """-> the merged (path, body) of two adjacent pieces, labelling the appended body with its heading when it differs."""
    (prev_path, prev_body), (path, body) = prev, nxt
    label = f"{path[-1]}:\n" if path != prev_path else ""
    return prev_path, f"{prev_body}\n\n{label}{body}"


def _merge_small(pieces):
    """Merge pieces under MIN_CHUNK_WORDS into the following one, and a small trailing piece into the previous one,
    only while the result (including an inserted heading label) stays within MAX_CHUNK_WORDS body words."""
    merged = []
    for piece in pieces:
        if merged and _words(merged[-1][1]) < MIN_CHUNK_WORDS and _words(_joined(merged[-1], piece)[1]) <= MAX_CHUNK_WORDS:
            merged[-1] = _joined(merged[-1], piece)
        else:
            merged.append(piece)
    if len(merged) > 1 and _words(merged[-1][1]) < MIN_CHUNK_WORDS and _words(_joined(merged[-2], merged[-1])[1]) <= MAX_CHUNK_WORDS:
        merged[-2:] = [_joined(merged[-2], merged[-1])]
    return merged


def _file_slug(source, stems):
    """-> a document id prefix unique across the corpus: the file stem, else the path slug, else the path slug plus a
    short hash of the relative path (deterministic; rechecked until unclaimed)."""
    stem_slug = _slug(Path(source).stem)
    if stems.get(stem_slug, source) != source:  # same file name in another directory
        stem_slug = _slug(source[:-3] if source.lower().endswith(".md") else source)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    length = 6
    while stems.get(stem_slug, source) != source:
        stem_slug = f"{_slug(source[:-3] if source.lower().endswith('.md') else source)}-{digest[:length]}"
        length += 2
    stems[stem_slug] = source
    return stem_slug


def _file_chunks(path, root, stems):
    source = path.relative_to(root).as_posix()
    stem_slug = _file_slug(source, stems)
    title, sections = _sections(path.read_text(encoding="utf-8"), path.stem)
    chunks = []
    for n, (heading_path, body) in enumerate(_merge_small(_pieces(sections))):
        text = " > ".join(heading_path) + "\n" + body
        chunks.append(Chunk(id=f"{stem_slug}:{_slug(heading_path[-1])}:{n}", source=source, title=title,
                            heading_path=tuple(heading_path), text=text,
                            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()))
    return chunks


def load_corpus(corpus_dir):
    """-> [Chunk] for every *.md under `corpus_dir` (sorted by relative path; unreadable files are skipped with a stderr
    log; provenance files named SOURCE.md are not documents)."""
    root = Path(corpus_dir)
    chunks, stems = [], {}
    if not root.is_dir():
        return chunks
    for path in sorted((p for p in root.rglob("*.md") if p.is_file() and p.name != "SOURCE.md"), key=lambda p: p.relative_to(root).as_posix()):
        try:
            file_chunks = _file_chunks(path, root, stems)
        except (OSError, UnicodeError) as exc:
            _log(f"skipping {path.name}: {type(exc).__name__}")
            continue
        chunks.extend(file_chunks)
    return chunks


def _terms(query):
    """Content terms of a query: lowercase words minus stopwords, aliases applied, deduplicated, bounded."""
    return _content_terms(_WORD.findall(query[:MAX_QUERY_CHARS].lower()))


def _content_terms(words):
    terms = []
    for word in words:
        if word not in STOPWORDS and len(word) <= MAX_TERM_LENGTH and word not in terms:
            terms.append(word)
    return terms[:MAX_QUERY_TERMS]


def _expand_aliases(words):
    out, i = [], 0
    longest = max(len(k.split()) for k in ALIASES)
    while i < len(words):
        for n in range(min(longest, len(words) - i), 0, -1):
            target = ALIASES.get(" ".join(words[i:i + n]))
            if target:
                out.extend(target.split())
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return out


def _fold(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _heading_keys(chunk):
    names = {chunk.title, chunk.heading_path[-1]}
    return {frozenset(_fold(w) for w in _content_terms(_WORD.findall(name.lower()))) for name in names} - {frozenset()}


class _Index:
    def __init__(self, chunks, connection):
        self.chunks = chunks
        self._db = connection
        self._lock = threading.Lock()
        self._heading_keys = [_heading_keys(c) for c in chunks]

    def _matches(self, term, column):
        match = f'"{term}"' if column is None else f'{column} : "{term}"'
        rows = self._db.execute("SELECT rowid FROM docs WHERE docs MATCH ?", (match,)).fetchall()
        return {row[0] for row in rows}

    def _score(self, terms):
        """-> {chunk index: (score, heading_hits, bm25)} for chunks matching at least one term."""
        if not terms:
            return {}
        any_hits = [self._matches(t, None) for t in terms]
        heading_hits = [self._matches(t, "heading") for t in terms]
        wanted = frozenset(_fold(t) for t in terms)
        ranks = {}
        match = " OR ".join(f'"{t}"' for t in terms)
        for rowid, rank in self._db.execute("SELECT rowid, bm25(docs, 4.0, 1.0) FROM docs WHERE docs MATCH ?", (match,)):
            ranks[rowid] = rank
        scored = {}
        for rowid, rank in ranks.items():
            matched = sum(rowid in hits for hits in any_hits)
            coverage = matched / len(terms)
            exact_heading = len(terms) == 1 and wanted in self._heading_keys[rowid]
            if matched < 2 and not exact_heading:
                coverage = min(coverage, SINGLE_TERM_CAP)
            scored[rowid] = (coverage, sum(rowid in hits for hits in heading_hits), rank)
        return scored

    def search(self, query):
        """-> every matching chunk as a passage dict, best first (coverage, then heading hits, then bm25, then id)."""
        words = _WORD.findall(query[:MAX_QUERY_CHARS].lower())
        variants = [_content_terms(words)]
        expanded = _content_terms(_expand_aliases(words))
        if expanded != variants[0]:
            variants.append(expanded)
        with self._lock:
            best = {}
            for terms in variants:
                for rowid, value in self._score(terms).items():
                    if rowid not in best or (-value[0], -value[1], value[2]) < (-best[rowid][0], -best[rowid][1], best[rowid][2]):
                        best[rowid] = value
        ordered = sorted(best, key=lambda r: (-best[r][0], -best[r][1], best[r][2], self.chunks[r].id))
        return [{"text": self.chunks[r].text, "score": best[r][0], "id": self.chunks[r].id,
                 "source": self.chunks[r].source, "title": self.chunks[r].title, "sha256": self.chunks[r].sha256}
                for r in ordered]


def _build_index(chunks):
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.execute("CREATE VIRTUAL TABLE docs USING fts5(heading, body, tokenize='porter unicode61')")
    for rowid, chunk in enumerate(chunks):
        heading, _, body = chunk.text.partition("\n")
        connection.execute("INSERT INTO docs(rowid, heading, body) VALUES (?, ?, ?)", (rowid, heading, body))
    return _Index(chunks, connection)


def make_retriever(corpus_dir):
    """-> retriever(query) -> list of passage dicts (best first, scores in [0, 1], never raises), or None when the
    directory is missing/empty or SQLite lacks FTS5. The callable carries its index as `.index` for find_section."""
    if not corpus_dir:
        return None
    chunks = load_corpus(corpus_dir)
    if not chunks:
        _log(f"no markdown documents under {corpus_dir}; retrieval disabled")
        return None
    try:
        index = _build_index(chunks)
    except sqlite3.Error as exc:
        _log(f"FTS5 index unavailable ({type(exc).__name__}: {exc}); retrieval disabled")
        return None

    def retriever(query):
        try:
            return index.search(query)[:MAX_RESULTS] if isinstance(query, str) else []
        except Exception:
            return []

    retriever.index = index
    return retriever


def find_section(retriever, query):
    """-> {"id", "source", "title", "score"} for the best-matching DOCUMENT (best chunk per source file), or None when
    nothing scores >= MIN_SECTION_SCORE. `retriever` is a make_retriever result. Never raises."""
    try:
        index = getattr(retriever, "index", retriever)
        if not isinstance(index, _Index) or not isinstance(query, str):
            return None
        best = {}
        for passage in index.search(query):
            best.setdefault(passage["source"], passage)
        if not best:
            return None
        top = max(best.values(), key=lambda p: p["score"])  # max keeps the first (best-ranked) on ties
        if top["score"] < MIN_SECTION_SCORE:
            return None
        return {"id": top["id"], "source": top["source"], "title": top["title"], "score": top["score"]}
    except Exception:
        return None
