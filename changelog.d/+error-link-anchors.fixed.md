The link under every error now lands on that error's row.

Each panel prints `…/reference/errors/#rb-e3004` as its subtitle. The page defined anchors for its ten family headings — `#rb-e0xxx` through `#rb-e9xxx` — and none for the rows, which are table cells. So all forty-three fragments resolved to nothing and every link scrolled to the top of a page with forty-three rows on it. The previous change to this subtitle replaced a repository path with a published URL and left the fragment exactly as undefined as it found it.

`report.catalog` now emits `{ #rb-e3004 }` on each code, which `attr_list` renders as `<code id="rb-e3004">`, and the CLI builds the URL through `error_docs_url` rather than an f-string at the call site. A test asserts the two agree for every code, so the fragment the CLI prints and the anchor the page defines cannot drift apart again.
