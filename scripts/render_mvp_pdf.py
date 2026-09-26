"""Render the MVP report and matched dialogue appendix to a portable PDF."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def footer(canvas: object, document: object) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"LANC MVP report | page {document.page}")
    canvas.restoreState()


def markdown_story(path: Path, styles: dict[str, ParagraphStyle]) -> list[object]:
    story: list[object] = []
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        rows = [row for row in table_rows if not all(cell.strip("-: ") == "" for cell in row)]
        if rows:
            table = Table(rows, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7),
                        ("FONT", (0, 1), (-1, -1), "Helvetica", 7),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.extend([table, Spacer(1, 4 * mm)])
        table_rows = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("|") and line.endswith("|"):
            table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        flush_table()
        if not line:
            story.append(Spacer(1, 2 * mm))
        elif line.startswith("### "):
            story.append(Paragraph(escape(line[4:]), styles["Heading3"]))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), styles["Heading2"]))
        elif line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), styles["Heading1"]))
        elif line.startswith("- "):
            story.append(Paragraph(f"&bull; {escape(line[2:])}", styles["BodyText"]))
        elif line.startswith("**") and ":** " in line:
            speaker, text = line.split(":** ", maxsplit=1)
            story.append(
                Paragraph(f"<b>{escape(speaker[2:])}:</b> {escape(text)}", styles["Dialogue"])
            )
        else:
            story.append(Paragraph(escape(line.replace("`", "")), styles["BodyText"]))
    flush_table()
    return story


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite PDF: {args.output}")

    stylesheet = getSampleStyleSheet()
    stylesheet.add(
        ParagraphStyle("Dialogue", parent=stylesheet["BodyText"], fontSize=8, leading=10)
    )
    stylesheet["Title"].alignment = TA_CENTER
    document = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,
        title="LANC MVP Report With Prompt And Dialogue Appendix",
    )
    story = markdown_story(args.report, stylesheet)
    story.extend(
        [
            Spacer(1, 6 * mm),
            Paragraph("Appendix: Matched Prompts And Dialogues", stylesheet["Title"]),
            Spacer(1, 4 * mm),
        ]
    )
    story.extend(markdown_story(args.examples, stylesheet))
    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
