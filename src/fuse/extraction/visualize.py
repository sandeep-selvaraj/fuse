"""HTML visualization for spanned extraction results.

Renders source text with highlighted spans — color-coded per field,
solid outlines for explicit extractions, dashed for implicit.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fuse.extraction.spans import SpannedResult

# 10 visually distinct hues, easily extended
_PALETTE = [
    "#4f86f7",  # blue
    "#ff6b6b",  # red
    "#51cf66",  # green
    "#fcc419",  # yellow
    "#cc5de8",  # purple
    "#ff922b",  # orange
    "#22b8cf",  # cyan
    "#f06595",  # pink
    "#20c997",  # teal
    "#adb5bd",  # grey
]


def render_html(source: str, result: SpannedResult) -> str:
    """Render source text with highlighted extraction spans as a standalone HTML page.

    Args:
        source: The original source text.
        result: A SpannedResult containing fields with spans.

    Returns:
        A complete HTML document string.
    """
    # Assign a color to each field
    color_map: dict[str, str] = {}
    for i, field in enumerate(result.fields):
        color_map[field.name] = _PALETTE[i % len(_PALETTE)]

    # Build sorted, non-overlapping insertion points
    # Each entry: (position, is_open, field)
    markers: list[tuple[int, bool, str, str, bool]] = []
    for field in result.fields:
        if field.span is None:
            continue
        markers.append((field.span.start, True, field.name, str(field.value), field.is_explicit))
        markers.append((field.span.end, False, field.name, str(field.value), field.is_explicit))

    # Sort: by position, closes before opens at same position
    markers.sort(key=lambda m: (m[0], m[1]))

    # Build the highlighted text
    parts: list[str] = []
    prev = 0
    for pos, is_open, name, value, is_explicit in markers:
        if pos > prev:
            parts.append(html.escape(source[prev:pos]))
        if is_open:
            color = color_map[name]
            border = "solid" if is_explicit else "dashed"
            tooltip = html.escape(f"{name}: {value}")
            parts.append(
                f'<mark class="span-highlight" '
                f'style="'
                f"background:{color}18;"
                f"border:2px {border} {color};"
                f"border-radius:3px;"
                f"padding:1px 3px;"
                f'" '
                f'title="{tooltip}" '
                f'data-field="{html.escape(name)}">'
            )
        else:
            parts.append("</mark>")
        prev = pos

    if prev < len(source):
        parts.append(html.escape(source[prev:]))

    highlighted_text = "".join(parts)

    # Build legend
    legend_items: list[str] = []
    for field in result.fields:
        color = color_map[field.name]
        border = "solid" if field.is_explicit else "dashed"
        type_label = "explicit" if field.is_explicit else "implicit"
        span_str = f"{field.span.start}:{field.span.end}" if field.span else "—"
        legend_items.append(
            f'<div class="legend-item">'
            f'<span class="legend-swatch" style="'
            f"background:{color}18;"
            f"border:2px {border} {color};"
            f'"></span>'
            f"<strong>{html.escape(field.name)}</strong>"
            f'<span class="legend-value">{html.escape(str(field.value))}</span>'
            f'<span class="legend-type {type_label}">{type_label}</span>'
            f'<span class="legend-span">{span_str}</span>'
            f"</div>"
        )

    legend_html = "\n".join(legend_items)

    return _HTML_TEMPLATE.format(
        highlighted_text=highlighted_text,
        legend=legend_html,
    )


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fuse — Extraction Result</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f1117;
    color: #e1e4e8;
    padding: 2rem;
    line-height: 1.6;
  }}
  h1 {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
  }}
  .source-text {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.5rem;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 0.95rem;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-word;
    margin-bottom: 2rem;
  }}
  mark.span-highlight {{
    color: inherit;
    cursor: default;
    transition: filter 0.15s;
  }}
  mark.span-highlight:hover {{
    filter: brightness(1.5);
  }}
  .legend {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.25rem;
  }}
  .legend h2 {{
    font-size: 0.85rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.35rem 0;
    font-size: 0.9rem;
  }}
  .legend-swatch {{
    display: inline-block;
    width: 18px;
    height: 18px;
    border-radius: 3px;
    flex-shrink: 0;
  }}
  .legend-value {{
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
  }}
  .legend-type {{
    font-size: 0.75rem;
    padding: 1px 6px;
    border-radius: 3px;
    font-weight: 500;
  }}
  .legend-type.explicit {{
    background: #1a3a2a;
    color: #51cf66;
  }}
  .legend-type.implicit {{
    background: #3a2a1a;
    color: #ff922b;
  }}
  .legend-span {{
    color: #484f58;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
  }}
</style>
</head>
<body>
  <h1>Extraction Result</h1>
  <div class="source-text">{highlighted_text}</div>
  <div class="legend">
    <h2>Fields</h2>
    {legend}
  </div>
</body>
</html>
"""
