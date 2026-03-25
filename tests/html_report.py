"""HTML report generator for RAG Agent responses."""

import os
import re
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any


def generate_html_report(results: List[Tuple[str, Dict[str, Any]]], output_path: str) -> str:
    """Generate a single HTML report with all queries and answers."""
    
    html_parts = [_get_html_header()]

    for i, (query, result) in enumerate(results, 1):
        html_parts.append(_render_qa_section(i, query, result))

    html_parts.append(_get_html_footer())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    
    return output_path


def _get_html_header() -> str:
    """Return HTML header with styles."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Agent Response</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 24px; }
        h3 { color: #666; margin-top: 16px; }
        .qa-section {
            border-bottom: 1px solid #eee;
            padding-bottom: 24px;
            margin-bottom: 24px;
        }
        .qa-section:last-child { border-bottom: none; margin-bottom: 0; }
        .query-number {
            background: #007bff;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 14px;
            margin-right: 8px;
        }
        .query { 
            background: #e7f3ff; 
            padding: 12px 16px; 
            border-radius: 6px; 
            border-left: 4px solid #007bff;
            margin-bottom: 16px;
            font-weight: 500;
        }
        .answer { line-height: 1.6; color: #333; white-space: pre-wrap; }
        .image-container {
            margin: 16px 0;
            padding: 16px;
            background: #fafafa;
            border-radius: 6px;
            border: 1px solid #eee;
        }
        .image-container img { max-width: 100%; height: auto; border-radius: 4px; }
        .image-details { margin-top: 8px; }
        .image-details summary { font-size: 13px; color: #888; cursor: pointer; user-select: none; }
        .image-details summary:hover { color: #555; }
        .image-caption { margin-top: 6px; font-size: 14px; color: #666; }
        .image-meta { font-size: 12px; color: #999; margin-top: 4px; }
        table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }
        th, td { padding: 10px 12px; text-align: left; border: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: 600; color: #333; }
        tr:nth-child(even) { background: #fafafa; }
        tr:hover { background: #f0f7ff; }
        .table-container {
            margin: 16px 0;
            padding: 16px;
            background: #fafafa;
            border-radius: 6px;
            border: 1px solid #eee;
            overflow-x: auto;
        }
        .table-description {
            margin-bottom: 12px;
            color: #333;
        }
        .table-section-label {
            font-weight: 600;
            color: #555;
            margin: 12px 0 8px 0;
            font-size: 13px;
        }
        .preview-table {
            background: #f9f9f9;
        }
        .preview-table th {
            background: #e9ecef;
        }
        .result-table th {
            background: #d4edda;
        }
        .sql-query {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 10px 14px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            margin: 12px 0;
            overflow-x: auto;
        }
        .sources {
            margin-top: 16px;
            padding: 10px 12px;
            background: #fff3cd;
            border-radius: 6px;
            font-size: 13px;
        }
        .timestamp { text-align: right; color: #999; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG Agent Response</h1>"""


def _get_html_footer() -> str:
    """Return HTML footer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f'<div class="timestamp">Generated: {timestamp}</div>\n</div></body></html>'


def _render_qa_section(index: int, query: str, result: Dict[str, Any]) -> str:
    """Render a single Q&A section with images inline in the answer text."""
    answer_html = result["answer"]
    images = result.get("images", [])

    # Replace [IMAGE:X] placeholders with rendered image HTML
    def _replace_image_placeholder(match):
        img_idx = int(match.group(1))
        if 0 <= img_idx < len(images):
            return _render_image(images[img_idx], img_idx)
        return match.group(0)

    answer_html = re.sub(r'\[IMAGE:\s*(\d+)\s*\]', _replace_image_placeholder, answer_html)

    # Strip any remaining unreplaced [IMAGE:X] placeholders (e.g. LLM hallucinated without calling tool)
    answer_html = re.sub(r'\s*\[IMAGE:\s*\d+\s*\]\s*', ' ', answer_html)

    # Any images that weren't referenced inline get appended at the end
    referenced = set(int(m.group(1)) for m in re.finditer(r'\[IMAGE:\s*(\d+)\s*\]', result["answer"]))
    unreferenced = [(i, img) for i, img in enumerate(images) if i not in referenced]

    parts = [
        f'<div class="qa-section">',
        f'<h2><span class="query-number">Q{index}</span></h2>',
        f'<div class="query">{query}</div>',
        '<h3>Answer</h3>',
        f'<div class="answer">{answer_html}</div>',
    ]

    # Append any unreferenced images as fallback
    if unreferenced:
        parts.append("<h3>Images</h3>")
        for i, img in unreferenced:
            parts.append(_render_image(img, i))

    # Tables
    if result["tables"]:
        parts.append("<h3>Tables</h3>")
        for tbl in result["tables"]:
            parts.append(_render_table(tbl))

    # Sources
    if result["sources"]:
        parts.append(f'<div class="sources"><strong>Sources:</strong> {", ".join(result["sources"])}</div>')

    parts.append("</div>")
    return "\n".join(parts)


def _render_image(img: Dict[str, Any], index: int) -> str:
    """Render an image block."""
    parts = ['<div class="image-container">']
    
    if img.get("base64"):
        img_format = img.get("format", "png")
        parts.append(f'<img src="data:image/{img_format};base64,{img["base64"]}" alt="Image {index+1}">')
    elif img.get("path") and os.path.exists(img["path"]):
        with open(img["path"], "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        ext = Path(img["path"]).suffix.lstrip(".") or "png"
        parts.append(f'<img src="data:image/{ext};base64,{img_data}" alt="Image {index+1}">')
    
    description = img.get("description", "")
    image_type = img.get("image_type", "unknown")
    source = img.get("source", "unknown")
    if description or image_type or source:
        parts.append(f'<details class="image-details"><summary>Image details</summary>')
        if description:
            parts.append(f'<div class="image-caption">{description}</div>')
        parts.append(f'<div class="image-meta">Type: {image_type} | Source: {source}</div>')
        parts.append('</details>')
    parts.append("</div>")
    return "\n".join(parts)


def _render_table(tbl: Dict[str, Any]) -> str:
    """Render a table block with preview and query result."""
    parts = ['<div class="table-container">']
    
    # Table description
    if tbl.get("description"):
        parts.append(f'<div class="table-description"><strong>Table:</strong> {tbl["description"]}</div>')
    
    # Table preview (original data)
    preview_data = tbl.get("preview", [])
    if preview_data:
        total_rows = tbl.get("total_rows", len(preview_data))
        parts.append(f'<div class="table-section-label">Table Preview (showing {len(preview_data)} of {total_rows} rows):</div>')
        headers = list(preview_data[0].keys())
        parts.append('<table class="preview-table"><thead><tr>')
        parts.extend(f"<th>{h}</th>" for h in headers)
        parts.append("</tr></thead><tbody>")
        for row in preview_data:
            parts.append("<tr>")
            parts.extend(f"<td>{row.get(h, '')}</td>" for h in headers)
            parts.append("</tr>")
        parts.append("</tbody></table>")
    
    # SQL query
    parts.append(f'<div class="sql-query">SQL Query: {tbl.get("sql", "")}</div>')
    
    # Query result
    result_data = tbl.get("result", [])
    if result_data:
        parts.append('<div class="table-section-label">Query Result:</div>')
        headers = list(result_data[0].keys())
        parts.append('<table class="result-table"><thead><tr>')
        parts.extend(f"<th>{h}</th>" for h in headers)
        parts.append("</tr></thead><tbody>")
        for row in result_data:
            parts.append("<tr>")
            parts.extend(f"<td>{row.get(h, '')}</td>" for h in headers)
            parts.append("</tr>")
        parts.append("</tbody></table>")
    
    parts.append(f'<div class="image-meta">Source: {tbl.get("source", "unknown")}</div>')
    parts.append("</div>")
    return "\n".join(parts)
