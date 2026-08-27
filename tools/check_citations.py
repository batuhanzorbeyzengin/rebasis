"""Every arXiv citation names the paper its author believed they were citing.

`docs/cascade-band.md` cited arXiv:2605.24297 for the claim that an off-the-shelf
reranker can make a strong first stage *worse*. That identifier is Yousefiramandi
and Cooney, *Benchmarking Patent Embeddings* — a multi-task evaluation of 22
embedding models on patents, which says nothing about rerankers at all. The paper
meant was arXiv:2411.11767, *Drowning in Documents: Consequences of Scaling
Reranker Inference*. The mistake survived review because a bare identifier beside
a sentence carries nothing a reader can check without opening it, and a reader
who opens one has no reason to think the next is worse.

**A tool cannot judge whether a paper supports a claim.** That is the constraint
this is built around, and a check that pretended otherwise would be either
useless or confidently wrong. What a tool *can* do is make the author write the
title down next to the identifier, and then ask arXiv. The author above knew
perfectly well they meant "Drowning in Documents"; had that title been recorded,
arXiv would have answered "Benchmarking Patent Embeddings" and the contradiction
would have been mechanical rather than a matter of anyone noticing. That is the
whole mechanism, and the only justification for asking anyone to maintain a
manifest at all: it turns a private belief about an identifier into a statement
something outside the repository can falsify.

## The manifest

`docs/citations.toml`, one entry per cited identifier.

`title` is required, and is the mechanism. `authors` is optional and, where
present, verified: every surname listed has to appear among the authors arXiv
returns. It earns a field because this repository cites by name — "Maystre et
al.", "Jacob et al." — so a name is a claim about a paper's identity exactly as a
title is. It is optional rather than required because an author who does not want
to assert it should not be made to; what is written is checked, and what is not
written claims nothing. A field that were recorded but never verified would be
the worst of the three, because it would look like a check.

`why` is required, free text, and deliberately never verified. It is the field
that survives the limitation above. Nothing here can tell whether *Drowning in
Documents* supports the sentence it is attached to — a reviewer can, and only if
the claim is written down beside the identifier. Requiring it costs one line per
paper and is the only thing in the manifest a machine could not have derived.
What it must *not* record is where the paper is cited: this check computes that,
so it cannot go stale, and a hand-written file list would.

## Scope: whatever git says is committed

The file list is `git ls-files --cached --others --exclude-standard` — tracked
files plus new ones not yet added, minus everything `.gitignore` excludes. Using
git rather than walking the tree is the point: `docs/_local/` and `docs/design/`
are the maintainer's private copies of the design document, they are gitignored,
and a hand-maintained exclusion list here would be a second copy of that decision
with nothing keeping the two in agreement.

`.md` and `.py`, because both carry citations here. Not only the prose:
`src/rebasis/core/geometry.py`, `src/rebasis/migrate/health.py` and
`tools/bridge_band.py` cite papers in docstrings and comments, and the citation
in `src/rebasis/observability/events.py` is *generated into*
`docs/reference/events.md` — checking the generated copy and not the generator
would leave the generator free to emit an identifier the check never sees.

`docs/citations.toml` is not scanned, and not merely because it is TOML: a
manifest that counted as a citation of its own entries would make the
dangling-entry check vacuous.

## Two modes

**Offline** (the default, and what runs on every pull request). Every identifier
cited in a committed file has an entry, and every entry is still cited. A
dangling entry is rot: it is a claim about a paper the repository no longer
makes. This mode is deterministic and touches nothing, so it cannot fail because
arXiv is down, which is the only reason it is safe to put on the merge path.

**Online** (`--online`, weekly). Resolves every manifest identifier through
arXiv's API and compares what comes back. Two things about that API were measured
rather than assumed, and both shape what this does:

* **An unknown identifier is dropped, not reported.** Asking for
  `2510.13406,9999.99999` returns one entry and a `totalResults` of 1; there is
  no error entry for the second. Iterating over what came back would therefore
  pass an identifier that does not exist. This iterates over what was *asked
  for*, and a missing answer is a failure.
* **A withdrawal does not change the title.** Withdrawn papers keep the title
  they were announced under and put the withdrawal in `arxiv:comment` ("This
  paper has been withdrawn"). A title comparison alone would pass one, so the
  comment is read too.

Titles are compared after normalising whitespace, case, LaTeX commands and
typographic punctuation, because those differ between two spellings of the same
title without being a disagreement. The failure message prints both titles
*unnormalised*: the normalised pair would hide the very difference the reader has
to act on.

There is no `--json`. Nothing in this repository would read it, and an output
format with no consumer is one more thing that can quietly stop being true.

    python tools/check_citations.py            # the pull-request check
    python tools/check_citations.py --online   # ask arXiv as well
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

#: The repository root. `tools/` sits directly beneath it.
ROOT = Path(__file__).resolve().parents[1]

#: The manifest, and the one file deliberately excluded from the scan.
MANIFEST = ROOT / "docs" / "citations.toml"

#: Suffixes scanned. Both carry citations here; see the module docstring.
SCANNED = (".md", ".py")

#: The fields an entry may carry. An unknown key is an error rather than
#: something ignored: `titel = "..."` would otherwise switch off the only check
#: that matters while looking exactly like a working entry.
FIELDS = frozenset({"title", "authors", "why"})

ARXIV_API = "https://export.arxiv.org/api/query"

#: arXiv's API terms ask for roughly one request every three seconds.
REQUEST_INTERVAL = 3.0

#: Identifiers per request. The API takes a list, and one request for the whole
#: manifest is both faster and politer than one request per paper.
BATCH = 25

#: arXiv asks callers to identify themselves.
USER_AGENT = "rebasis-check-citations (+https://github.com/batuhanzorbeyzengin/rebasis)"

#: Files listed against an identifier before the listing is summarised.
PREVIEW = 2

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

#: An arXiv identifier: the modern form, or the pre-2007 archive form
#: (`math.AG/0601001`). Nothing here uses the old form today, and it is matched
#: anyway because a form the pattern does not know is a citation that goes
#: unchecked in silence, which is the failure mode this exists to remove.
_ID = r"\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Za-z]{2})?/\d{7}"

#: The citation forms this repository actually uses: a link to the abstract, a
#: link to the PDF (which is how the mis-citation was written), and a bare
#: `arXiv:` prefix in running prose. The full-text link is matched too because
#: the design document uses it.
CITATION = re.compile(
    rf"(?:arxiv\.org/(?:abs|pdf|html|ps)/|arxiv:\s*)(?P<id>{_ID})(?:v\d+)?",
    re.IGNORECASE,
)

#: The same thing on its own, for validating a manifest key.
IDENTIFIER = re.compile(_ID)

#: The `<id>` arXiv puts on an entry, from which the identifier is read back.
_ENTRY_ID = re.compile(rf"arxiv\.org/abs/(?P<id>{_ID})(?:v\d+)?$")

_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\s*")

#: Two spellings of one title differ in ways that are not disagreements: braces
#: around a capitalised word, a typographic dash where the other has a hyphen, a
#: non-breaking space. Removing them makes the comparison about the title rather
#: than about typography.
_PUNCTUATION = str.maketrans(
    {
        "{": None,
        "}": None,
        "\\": None,
        "$": None,
        "~": " ",
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    },
)


@dataclass(frozen=True, slots=True)
class Citation:
    """One arXiv identifier, where it was written."""

    arxiv_id: str
    path: str
    line: int


@dataclass(frozen=True, slots=True)
class Entry:
    """One manifest entry: what the author says an identifier is."""

    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    why: str


@dataclass(frozen=True, slots=True)
class Paper:
    """What arXiv says an identifier is."""

    title: str
    authors: tuple[str, ...]
    comment: str


def relative(path: Path) -> str:
    """Render a path as a human can paste it back into a shell."""
    return path.relative_to(ROOT).as_posix()


def normalise(text: str) -> str:
    """Reduce a title to the form two spellings of it have in common."""
    stripped = _LATEX_COMMAND.sub(" ", text)
    return " ".join(stripped.translate(_PUNCTUATION).casefold().split())


def committed_files() -> list[Path]:
    """List the files a clone would carry, as git sees them."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is not on PATH, and the file list comes from git")
    listing = subprocess.run(  # noqa: S603 - a fixed argv, and git comes from shutil.which
        [git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / name for name in listing.split("\0") if name.endswith(SCANNED)]


def read(path: Path) -> str | None:
    """Return a file's text, or None where it cannot be read.

    `--cached` lists what is in the index, which includes a file deleted from the
    working tree. That is the ordinary case for this returning None, and it is
    reported rather than raised so one such file cannot stop the whole check.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return text


def find_citations(paths: Iterable[Path]) -> tuple[list[Citation], list[Path]]:
    """Collect every arXiv identifier cited in the given files."""
    citations: list[Citation] = []
    unreadable: list[Path] = []
    for path in paths:
        if path == MANIFEST:
            continue
        text = read(path)
        if text is None:
            unreadable.append(path)
            continue
        where = relative(path)
        citations.extend(
            Citation(arxiv_id=match["id"].lower(), path=where, line=number)
            for number, line in enumerate(text.splitlines(), 1)
            for match in CITATION.finditer(line)
        )
    return citations, unreadable


def parse_entry(arxiv_id: str, fields: Any) -> tuple[Entry | None, list[str]]:
    """Validate one manifest entry, returning it or what is wrong with it."""
    where = f'{relative(MANIFEST)} [papers."{arxiv_id}"]'
    if not isinstance(fields, dict):
        return None, [f"{where}: not a table"]

    problems = [f"{where}: unknown field {key!r}" for key in sorted(set(fields) - FIELDS)]
    if not IDENTIFIER.fullmatch(arxiv_id):
        problems.append(f"{where}: the key is not an arXiv identifier")

    title = fields.get("title")
    if not isinstance(title, str) or not title.strip():
        problems.append(f"{where}: `title` is required; it is the whole mechanism")
    why = fields.get("why")
    if not isinstance(why, str) or not why.strip():
        problems.append(f"{where}: `why` is required; say what the paper is cited for")
    authors = fields.get("authors", "")
    if not isinstance(authors, str):
        problems.append(f"{where}: `authors` is a comma-separated string of surnames")

    if problems:
        return None, problems
    return (
        Entry(
            arxiv_id=arxiv_id,
            title=str(title),
            authors=tuple(name.strip() for name in str(authors).split(",") if name.strip()),
            why=str(why),
        ),
        [],
    )


def load_manifest() -> tuple[dict[str, Entry], list[str]]:
    """Read the manifest, returning its entries and whatever is wrong with it."""
    try:
        raw = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    except OSError:
        return {}, [f"{relative(MANIFEST)}: missing"]
    except tomllib.TOMLDecodeError as error:
        return {}, [f"{relative(MANIFEST)}: {error}"]

    papers = raw.get("papers")
    if not isinstance(papers, dict):
        return {}, [f"{relative(MANIFEST)}: no [papers] table"]

    entries: dict[str, Entry] = {}
    problems: list[str] = []
    for arxiv_id, fields in sorted(papers.items()):
        entry, trouble = parse_entry(arxiv_id, fields)
        problems.extend(trouble)
        if entry is not None:
            entries[entry.arxiv_id] = entry
    return entries, problems


def check_offline(entries: Mapping[str, Entry], citations: Sequence[Citation]) -> list[str]:
    """Check that the manifest and the documentation name the same papers."""
    cited = {citation.arxiv_id for citation in citations}
    unrecorded = [
        f"{citation.path}:{citation.line}: arXiv:{citation.arxiv_id} is cited with no entry "
        f"in {relative(MANIFEST)}. Add one saying which paper it is."
        for citation in citations
        if citation.arxiv_id not in entries
    ]
    dangling = [
        f"{relative(MANIFEST)}: arXiv:{arxiv_id} has an entry but is cited nowhere. Remove "
        "it, or restore the citation it was written for."
        for arxiv_id in sorted(set(entries) - cited)
    ]
    return unrecorded + dangling


def parse_paper(element: ET.Element) -> tuple[str, Paper] | None:
    """Read one Atom entry into the paper it describes."""
    found = element.find(f"{_ATOM}id")
    match = _ENTRY_ID.search(found.text or "") if found is not None else None
    if match is None:
        return None
    return match["id"].lower(), Paper(
        title=text_of(element.find(f"{_ATOM}title")),
        authors=tuple(text_of(name) for name in element.iterfind(f"{_ATOM}author/{_ATOM}name")),
        comment=text_of(element.find(f"{_ARXIV}comment")),
    )


def text_of(element: ET.Element | None) -> str:
    """Return an element's text with its wrapping collapsed to single spaces."""
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split())


def fetch(arxiv_ids: Sequence[str], timeout: float) -> tuple[dict[str, Paper], str | None]:
    """Resolve one batch of identifiers.

    The error handling lives here rather than in the caller so that a network
    failure names the batch it happened on, and so that the loop over batches has
    no try block in it.
    """
    url = f"{ARXIV_API}?id_list={','.join(arxiv_ids)}&max_results={len(arxiv_ids)}"
    # S310 on both the Request and the urlopen below: one host, one scheme, both
    # constants above, and the only variable part is a list of identifiers the
    # manifest check has already matched against `IDENTIFIER`. There is no path
    # by which a `file:` or custom scheme reaches either call.
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read()
    except OSError as error:
        return {}, f"arXiv did not answer for {', '.join(arxiv_ids)}: {error}"

    # arXiv's own API response, parsed by a job that runs weekly rather than on
    # the merge path. `defusedxml` is not a dependency of this project and adding
    # one to a tools/ script would put it on every developer's install.
    try:
        feed = ET.fromstring(body)  # noqa: S314
    except ET.ParseError as error:
        return {}, f"arXiv returned something that is not the Atom feed it documents: {error}"

    papers: dict[str, Paper] = {}
    for element in feed.iterfind(f"{_ATOM}entry"):
        parsed = parse_paper(element)
        if parsed is not None:
            papers[parsed[0]] = parsed[1]
    return papers, None


def resolve(arxiv_ids: Sequence[str], timeout: float) -> tuple[dict[str, Paper], list[str]]:
    """Resolve every identifier through arXiv, in batches and at their rate."""
    papers: dict[str, Paper] = {}
    problems: list[str] = []
    for start in range(0, len(arxiv_ids), BATCH):
        if start:
            time.sleep(REQUEST_INTERVAL)
        found, failure = fetch(arxiv_ids[start : start + BATCH], timeout)
        papers.update(found)
        if failure is not None:
            problems.append(failure)
    return papers, problems


def missing_authors(expected: Sequence[str], listed: Sequence[str]) -> list[str]:
    """Return the surnames the manifest claims that arXiv does not list.

    Whole-word rather than substring, so that a surname cannot be satisfied by
    happening to sit inside a longer one; and against the joined list rather than
    position by position, so that no correct citation fails over an initial, a
    middle name or the order the authors are given in.
    """
    joined = normalise("; ".join(listed))
    return [
        surname
        for surname in expected
        if not re.search(rf"\b{re.escape(normalise(surname))}\b", joined)
    ]


def check_online(entries: Mapping[str, Entry], papers: Mapping[str, Paper]) -> list[str]:
    """Check every manifest entry against what arXiv returned for it."""
    problems: list[str] = []
    for arxiv_id, entry in sorted(entries.items()):
        paper = papers.get(arxiv_id)
        if paper is None:
            problems.append(
                f"arXiv:{arxiv_id}: arXiv returned nothing for this identifier. One it does "
                "not know is dropped from the feed rather than reported, so this is what a "
                "non-existent or removed paper looks like."
            )
            continue
        if normalise(paper.title) != normalise(entry.title):
            problems.append(
                f"arXiv:{arxiv_id}: this is not the paper the manifest says it is.\n"
                f"      manifest: {entry.title}\n"
                f"      arXiv:    {paper.title}"
            )
        absent = missing_authors(entry.authors, paper.authors)
        if absent:
            problems.append(
                f"arXiv:{arxiv_id}: not among the authors arXiv lists: {', '.join(absent)}\n"
                f"      arXiv: {'; '.join(paper.authors)}"
            )
        if "withdrawn" in paper.comment.casefold():
            problems.append(
                f"arXiv:{arxiv_id}: arXiv's own comment reads {paper.comment!r}. A withdrawal "
                "leaves the title alone, so nothing above would have caught it."
            )
    return problems


def report_citations(entries: Mapping[str, Entry], citations: Sequence[Citation]) -> None:
    """Print which identifier is cited where, and whether it is recorded."""
    by_id: dict[str, list[Citation]] = {}
    for citation in citations:
        by_id.setdefault(citation.arxiv_id, []).append(citation)

    print(
        f"{len(entries)} entries in {relative(MANIFEST)}; "
        f"{len(by_id)} identifiers cited in {len(citations)} places"
    )
    for arxiv_id in sorted(set(by_id) | set(entries)):
        found = by_id.get(arxiv_id, [])
        if not found:
            status = "DANGLING"
        elif arxiv_id not in entries:
            status = "NO ENTRY"
        else:
            status = "ok"
        files = sorted({citation.path for citation in found})
        shown = ", ".join(files[:PREVIEW])
        if len(files) > PREVIEW:
            shown += f" and {len(files) - PREVIEW} more"
        print(f"  {arxiv_id:<14}{len(found):>3}  {status:<10}{shown}")


def report_papers(entries: Mapping[str, Entry], papers: Mapping[str, Paper]) -> None:
    """Print what arXiv answered for each manifest entry."""
    print(f"\nresolved {len(papers)} of {len(entries)} identifiers through {ARXIV_API}")
    for arxiv_id, entry in sorted(entries.items()):
        paper = papers.get(arxiv_id)
        if paper is None:
            print(f"  {arxiv_id:<14}NOT FOUND")
            continue
        agrees = normalise(paper.title) == normalise(entry.title)
        print(f"  {arxiv_id:<14}{'ok' if agrees else 'WRONG':<10}{paper.title}")


def main() -> int:
    """Check every arXiv citation in the committed documentation."""
    parser = argparse.ArgumentParser(
        description="Check that every arXiv citation names the paper it claims to."
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="also resolve every manifest identifier through the arXiv API",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds to wait on arXiv (default: 30)",
    )
    args = parser.parse_args()

    entries, problems = load_manifest()
    if problems:
        # Without a manifest every citation reads as unrecorded, which would bury
        # the one thing that actually needs fixing under a screen of noise.
        print(f"{relative(MANIFEST)} could not be read as a manifest:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    try:
        paths = committed_files()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"the file list comes from git, and git failed: {error}", file=sys.stderr)
        return 2

    citations, unreadable = find_citations(paths)
    report_citations(entries, citations)
    for path in unreadable:
        print(f"  (not read, so not scanned: {relative(path)})")
    problems += check_offline(entries, citations)

    if args.online:
        papers, failures = resolve(sorted(entries), args.timeout)
        report_papers(entries, papers)
        problems += failures
        problems += check_online(entries, papers)

    if problems:
        print("\nwhat is wrong:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print("\nevery citation names its paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
