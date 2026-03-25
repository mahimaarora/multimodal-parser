"""Render RAG agent responses with inline images across different mediums."""

import re
import base64
from typing import Dict, Any


def render(result: Dict[str, Any], mode: str = "auto"):
    """Render agent response with inline images.

    Args:
        result: Dict from RAGAgent.query() with keys: answer, images, tables, sources.
        mode: "auto" (detect environment), "jupyter", "terminal", or "html" (returns string).
    """
    if mode == "auto":
        mode = "jupyter" if _in_jupyter() else "terminal"

    if mode == "jupyter":
        _render_jupyter(result)
    elif mode == "html":
        return _render_html(result)
    else:
        _render_terminal(result)


def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:
        return False


def _render_jupyter(result: Dict[str, Any]):
    from IPython.display import display, Image, Markdown

    answer = result["answer"]
    images = result.get("images", [])

    # Split by [IMAGE:X] placeholders — odd indices are image index strings
    parts = re.split(r'\[IMAGE:\s*(\d+)\s*\]', answer)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                display(Markdown(part.strip()))
        else:
            idx = int(part)
            if 0 <= idx < len(images):
                img = images[idx]
                if img.get("base64"):
                    display(Image(data=base64.b64decode(img["base64"])))
                elif img.get("path"):
                    display(Image(filename=img["path"]))

    if result.get("sources"):
        display(Markdown(f"**Sources:** {', '.join(result['sources'])}"))


def _render_terminal(result: Dict[str, Any]):
    answer = result["answer"]
    images = result.get("images", [])

    def _replace(match):
        idx = int(match.group(1))
        if 0 <= idx < len(images):
            desc = images[idx].get("description", "Image")
            path = images[idx].get("path", "")
            label = f"\n  [Image: {desc[:80]}]"
            if path:
                label += f"\n  [Path: {path}]"
            return label + "\n"
        return ""

    text = re.sub(r'\[IMAGE:\s*(\d+)\s*\]', _replace, answer)
    print(text)

    if result.get("sources"):
        print(f"\nSources: {', '.join(result['sources'])}")


def _render_html(result: Dict[str, Any]) -> str:
    answer = result["answer"]
    images = result.get("images", [])

    def _replace(match):
        idx = int(match.group(1))
        if 0 <= idx < len(images):
            img = images[idx]
            parts = ['<div style="margin:12px 0">']
            if img.get("base64"):
                fmt = img.get("format", "png")
                parts.append(f'<img src="data:image/{fmt};base64,{img["base64"]}" style="max-width:100%">')
            elif img.get("path"):
                parts.append(f'<img src="{img["path"]}" style="max-width:100%">')
            parts.append("</div>")
            return "\n".join(parts)
        return ""

    html = re.sub(r'\[IMAGE:\s*(\d+)\s*\]', _replace, answer)
    return html
