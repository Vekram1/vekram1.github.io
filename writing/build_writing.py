#!/usr/bin/env python3
from __future__ import annotations

import html
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITING_DIR = ROOT / "writing"
INDEX_HTML = Path("../src/index.html")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "untitled"


def split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    front = parts[1].strip()
    body = parts[2].lstrip()
    data = parse_front_matter(front)
    return data, body


def parse_front_matter(front: str) -> dict:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(front)
        return data or {}
    except ModuleNotFoundError:
        return parse_front_matter_fallback(front)


def clean_value(raw: str):
    value = raw.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip().strip("'\"") for item in value[1:-1].split(",")]
        return [item for item in items if item]
    return value


def parse_front_matter_fallback(front: str) -> dict:
    data: dict = {}
    current_parent = None
    for line in front.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.startswith("  ") and current_parent:
            key, _, value = line.strip().partition(":")
            if key:
                data.setdefault(current_parent, {})[key.strip()] = clean_value(value)
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if value.strip() == "":
            current_parent = key
            data[key] = {}
        else:
            current_parent = None
            data[key] = clean_value(value)
    return data


def parse_date(value: str | None) -> tuple[str, datetime | None]:
    if not value:
        return "", None
    try:
        parsed = datetime.fromisoformat(value)
        return value, parsed
    except ValueError:
        return value, None


def replace_math(text: str) -> str:
    try:
        from latex2mathml.converter import convert  # type: ignore
    except ModuleNotFoundError:
        print("Missing dependency: latex2mathml. Install with:")
        print("  python3 -m pip install latex2mathml")
        raise

    def convert_display(match: re.Match) -> str:
        latex = match.group(1).strip()
        if not latex:
            return match.group(0)
        return f"<div class=\"math-block\">{convert(latex)}</div>"

    def convert_inline(match: re.Match) -> str:
        latex = match.group(1).strip()
        if not latex:
            return match.group(0)
        return f"<span class=\"math-inline\">{convert(latex)}</span>"

    segments = re.split(r"(```.*?```)", text, flags=re.S)
    for idx in range(0, len(segments), 2):
        segment = segments[idx]
        segment = re.sub(r"\$\$(.+?)\$\$", convert_display, segment, flags=re.S)
        segment = re.sub(r"(?<!\\)\$(?!\\$)(.+?)(?<!\\)\$", convert_inline, segment)
        segments[idx] = segment
    return "".join(segments)


def render_paper_html(meta: dict, body_html: str) -> str:
    title = meta.get("title") or "Untitled"
    subtitle = meta.get("subtitle")
    author = meta.get("author")
    date_value, _ = parse_date(str(meta.get("date") or ""))
    summary = meta.get("summary")
    edit = meta.get("editPost") or {}
    edit_url = edit.get("URL") if isinstance(edit, dict) else None
    edit_text = edit.get("Text") if isinstance(edit, dict) else None

    subtitle_html = (
        f"<p class=\"paper-subtitle\">{html.escape(str(subtitle))}</p>"
        if subtitle
        else ""
    )
    summary_html = (
        f"<p class=\"paper-summary\">{html.escape(str(summary))}</p>"
        if summary
        else ""
    )
    edit_html = (
        f"<a class=\"paper-edit\" href=\"{html.escape(edit_url)}\">"
        f"{html.escape(edit_text or 'External Version')}</a>"
        if edit_url
        else ""
    )
    meta_html = ""

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(str(title))} | Vikram Oddiraju</title>
  <meta name=\"description\" content=\"{html.escape(str(summary or ''))}\" />
  <link rel=\"stylesheet\" href=\"../../src/styles.css\" />
</head>
<body class=\"paper-page\">
  <main class=\"paper-container\">
    <header class=\"paper-header\">
      <p class=\"paper-kicker\"><a href=\"../../src/index.html\">Vikram Oddiraju</a> | {date_value} </p>
      <h1>{html.escape(str(title))}</h1>
      {subtitle_html}
      {meta_html}
      {summary_html}
      {edit_html}
    </header>
    <article class=\"paper-body\">
{body_html}
    </article>
  </main>
</body>
</html>
"""


def render_writing_list(entries: list[dict]) -> str:
    items = []
    for entry in entries:
        title = html.escape(entry["title"])
        date_value = html.escape(entry["date"])
        summary = html.escape(entry.get("summary", ""))
        href = html.escape(entry["href"])
        summary_html = f"<p class=\"writing-summary\">{summary}</p>" if summary else ""
        items.append(
            "\n".join(
                [
                    "    <li class=\"writing-item\">",
                    f"        <a href=\"..{href}\">{title}</a>",
                    f"        <span class=\"date\">{date_value}</span>",
                    f"        {summary_html}" if summary_html else "",
                    "    </li>",
                ]
            )
        )
    return "\n".join(items)


def update_index(entries: list[dict]) -> None:
    if not INDEX_HTML.exists():
        print(f"Missing {INDEX_HTML}")
        return
    index_text = INDEX_HTML.read_text(encoding="utf-8")
    list_html = render_writing_list(entries)
    pattern = re.compile(r"(<ul class=\"writing-list\">)(.*?)(</ul>)", re.S)
    if not pattern.search(index_text):
        print("Could not find writing list in index.html")
        return

    def repl(match: re.Match) -> str:
        return f"{match.group(1)}\n{list_html}\n    {match.group(3)}"

    new_text = pattern.sub(repl, index_text)
    INDEX_HTML.write_text(new_text, encoding="utf-8")


def main() -> int:
    try:
        import markdown  # type: ignore
    except ModuleNotFoundError:
        print("Missing dependency: markdown. Install with:")
        print("  python3 -m pip install markdown pyyaml latex2mathml")
        return 1

    entries = []
    md_files = sorted(WRITING_DIR.glob("*_writing/*.md"))
    if not md_files:
        print("No markdown files found.")
        return 0

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        meta, body = split_front_matter(text)
        draft = meta.get("draft")
        if isinstance(draft, str):
            draft = draft.lower() == "true"
        if draft:
            continue

        title = meta.get("title") or md_path.stem
        date_value, parsed_date = parse_date(str(meta.get("date") or ""))
        summary = meta.get("summary") or ""

        folder_name = md_path.parent.name
        slug = folder_name
        # if slug.endswith("_writing"):
        #     slug = slug[: -len("_writing")]
        title_slug = slugify(str(title))

        output_dir = WRITING_DIR / slug
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{title_slug}.html"

        body_with_math = replace_math(body)
        body_html = markdown.markdown(body_with_math, extensions=["fenced_code", "tables"])
        paper_html = render_paper_html(meta, body_html)
        output_file.write_text(paper_html, encoding="utf-8")

        entries.append(
            {
                "title": str(title),
                "date": date_value,
                "summary": str(summary),
                "href": f"/writing/{slug}/{title_slug}.html",
                "sort_key": parsed_date or datetime.min,
            }
        )

    entries.sort(key=lambda item: item["sort_key"], reverse=True)
    update_index(entries)
    print(f"Generated {len(entries)} paper(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
