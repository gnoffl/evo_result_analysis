"""
SVG figure composition utility.

Combines existing SVG files into a single multi-panel figure by placing each
source SVG at an arbitrary position on a new output canvas.

Positions are specified as normalized fractions (0.0–1.0) of the output canvas
dimensions, so layout specs remain independent of absolute canvas size.

Each source SVG is embedded as a nested ``<svg>`` element, preserving its
internal coordinate system (via ``viewBox``) while scaling it to fill the
requested panel rectangle.

Example usage
-------------
from workflows.figure_composition import compose_figures

compose_figures(
    panels=[
        {"path": "/path/to/plot_a.svg", "rect": (0.0, 0.0, 0.5, 0.5)},
        {"path": "/path/to/plot_b.svg", "rect": (0.5, 0.0, 1.0, 0.5)},
        {"path": "/path/to/plot_c.svg", "rect": (0.0, 0.5, 1.0, 1.0)},
    ],
    output_path="figure_1.svg",
    figsize=(8.27, 11.69),  # A4 in inches
    labels=True,
)
"""

from __future__ import annotations

import string
import sys
import xml.etree.ElementTree as ET
from typing import Optional


# Pixels per inch (CSS / SVG standard)
_PX_PER_INCH = 96.0

# SVG XML namespace
_SVG_NS = "http://www.w3.org/2000/svg"

# Register so ElementTree writes the unprefixed SVG namespace
ET.register_namespace("", _SVG_NS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_svg_length(value: str) -> float:
    """Convert an SVG length string to CSS pixels."""
    value = value.strip()
    if value.endswith("px"):
        return float(value[:-2])
    if value.endswith("pt"):
        return float(value[:-2]) * _PX_PER_INCH / 72.0
    if value.endswith("mm"):
        return float(value[:-2]) * _PX_PER_INCH / 25.4
    if value.endswith("cm"):
        return float(value[:-2]) * _PX_PER_INCH / 2.54
    if value.endswith("in"):
        return float(value[:-2]) * _PX_PER_INCH
    # bare number – assume px
    return float(value)


def _svg_viewbox(root: ET.Element) -> str:
    """Return a ``viewBox`` string for the source SVG root element.

    Falls back to ``"0 0 <width> <height>"`` when no ``viewBox`` attribute is
    present.
    """
    vb = root.get("viewBox") or root.get("viewbox")
    if vb:
        return vb
    w_str = root.get("width", "100")
    h_str = root.get("height", "100")
    return f"0 0 {_parse_svg_length(w_str)} {_parse_svg_length(h_str)}"


# ---------------------------------------------------------------------------
# Internal helpers (continued)
# ---------------------------------------------------------------------------

def _create_canvas(figsize: tuple[float, float]) -> tuple[ET.Element, float, float]:
    """Create the output SVG root element for the given *figsize* in inches.

    Returns:
        A tuple of ``(root_element, canvas_width_px, canvas_height_px)``.
    """
    page_w = figsize[0] * _PX_PER_INCH
    page_h = figsize[1] * _PX_PER_INCH
    root = ET.Element(
        f"{{{_SVG_NS}}}svg",
        attrib={
            "width": f"{figsize[0]}in",
            "height": f"{figsize[1]}in",
            "viewBox": f"0 0 {page_w} {page_h}",
        },
    )
    # White canvas background so the output is never transparent.
    ET.SubElement(
        root,
        f"{{{_SVG_NS}}}rect",
        attrib={
            "x": "0",
            "y": "0",
            "width": str(page_w),
            "height": str(page_h),
            "fill": "white",
        },
    )
    return root, page_w, page_h


def _strip_figure_background(root: ET.Element) -> None:
    """Remove matplotlib's figure-background ``<rect>`` from *root*.

    Matplotlib emits a full-canvas ``<rect>`` as the first child of the SVG
    in order to paint the figure facecolor.  Its stroke (figure.linewidth,
    default 0.8 pt) shows as a thin black border when panels are placed edge-
    to-edge.  Removing it prevents that artefact without affecting plot content.
    """
    # The background rect is always the first <rect> child whose x, y, width
    # and height coincide with the figure dimensions (all four can be 0 / the
    # viewBox size).  Matplotlib marks it with id="patch_1" or places it first.
    for child in list(root):
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "rect":
            # Only remove the outermost background rect – it has no clip-path
            # and its fill covers the whole canvas.
            if child.get("clip-path") is None:
                root.remove(child)
                break


def _load_svg(path: str) -> ET.Element:
    """Parse an SVG file, strip its figure background, and return the root.

    Raises:
        FileNotFoundError: If *path* does not exist or cannot be opened.
    """
    try:
        root = ET.parse(path).getroot()
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"Cannot open SVG file: {path}") from exc
    _strip_figure_background(root)
    return root


def _embed_panel(
    canvas: ET.Element,
    src_root: ET.Element,
    x: float,
    y: float,
    width: float,
    height: float,
    panel_scale: float = 1.0,
) -> None:
    """Embed *src_root*'s content as a nested ``<svg>`` inside *canvas*.

    The nested element uses the source's ``viewBox`` so its internal
    coordinate system is preserved while it is scaled to fit the target
    rectangle.

    When *panel_scale* < 1.0 the image is shrunk and anchored to the
    **bottom-right** corner of the panel rectangle, leaving the top-left
    free for panel labels.

    Args:
        canvas: The output SVG root element to append to.
        src_root: Root element of the source SVG.
        x: Left edge of the panel in canvas pixels.
        y: Top edge of the panel in canvas pixels.
        width: Panel width in canvas pixels.
        height: Panel height in canvas pixels.
        panel_scale: Fraction of the panel area used by the image (0–1).
            Defaults to 1.0 (no scaling).
    """
    # White background covering the full panel cell first, so the area
    # outside the scaled image is also white (not transparent).
    ET.SubElement(
        canvas,
        f"{{{_SVG_NS}}}rect",
        attrib={
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": "white",
        },
    )

    scaled_w = width * panel_scale
    scaled_h = height * panel_scale
    # Anchor to bottom-right: push the image right and down by the freed space.
    img_x = x + (width - scaled_w)
    img_y = y + (height - scaled_h)

    vb = _svg_viewbox(src_root)
    nested = ET.SubElement(
        canvas,
        f"{{{_SVG_NS}}}svg",
        attrib={
            "x": str(img_x),
            "y": str(img_y),
            "width": str(scaled_w),
            "height": str(scaled_h),
            "viewBox": vb,
            "preserveAspectRatio": "xMidYMid meet",
            "overflow": "visible",
        },
    )
    # White background inside the nested SVG's own coordinate system.
    vb_parts = vb.split()
    ET.SubElement(
        nested,
        f"{{{_SVG_NS}}}rect",
        attrib={
            "x": vb_parts[0],
            "y": vb_parts[1],
            "width": vb_parts[2],
            "height": vb_parts[3],
            "fill": "white",
        },
    )
    for child in src_root:
        nested.append(child)


def _add_panel_label(
    canvas: ET.Element,
    label: str,
    x: float,
    y: float,
    fontsize: float,
    color: tuple[float, float, float],
) -> None:
    """Overlay a single panel label (e.g. "A") on *canvas*.

    Args:
        canvas: The output SVG root element to append to.
        label: The label string to render (typically one uppercase letter).
        x: Horizontal position in canvas pixels.
        y: Vertical baseline position in canvas pixels.
        fontsize: Font size in pixels.
        color: RGB color as three floats in ``[0, 1]``.
    """
    r, g, b = (int(c * 255) for c in color)
    text_el = ET.SubElement(
        canvas,
        f"{{{_SVG_NS}}}text",
        attrib={
            "x": str(x),
            "y": str(y),
            "font-size": str(fontsize),
            "font-family": "Helvetica, Arial, sans-serif",
            "font-weight": "bold",
            "fill": f"rgb({r},{g},{b})",
        },
    )
    text_el.text = label


def _save_svg(root: ET.Element, output_path: str) -> None:
    """Serialise *root* to *output_path* with a UTF-8 XML declaration.

    Applies pretty-printing when running on Python >= 3.9.
    """
    if sys.version_info >= (3, 9):
        ET.indent(root, space="  ")

    ET.ElementTree(root).write(output_path, encoding="unicode", xml_declaration=False)

    with open(output_path, "r+", encoding="utf-8") as fh:
        body = fh.read()
        fh.seek(0)
        fh.write('<?xml version="1.0" encoding="utf-8"?>\n' + body)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose_figures(
    panels: list[dict],
    output_path: str,
    figsize: tuple[float, float] = (8.27, 11.69),
    labels: bool = False,
    label_fontsize: float = 14.0,
    label_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    margin: float = 0.01,
    panel_scale: float = 0.92,
) -> None:
    """Compose multiple existing SVG files into a single SVG figure.

    Args:
        panels (list[dict]): Each dict describes one panel and must contain:
            ``path`` (str): Absolute or relative path to an SVG file.
            ``rect`` (tuple[float, float, float, float]): ``(x0, y0, x1, y1)``
                as fractions of the output canvas (0.0 = left/top edge,
                1.0 = right/bottom edge). Example: ``(0.0, 0.0, 0.5, 1.0)``
                fills the left half.
        output_path (str): Path where the composed SVG will be saved.
        figsize (tuple[float, float]): Output canvas size in inches as
            ``(width, height)``. Common values: ``(8.27, 11.69)`` (A4),
            ``(7.0, 5.0)`` (landscape half), ``(6.93, 5.0)``
            (two-column journal width).
        labels (bool): If True, overlay uppercase panel labels (A, B, C …)
            at the top-left corner of each panel's bounding box.
        label_fontsize (float): Font size for panel labels in pixels.
            Default is 14.
        label_color (tuple[float, float, float]): RGB color for panel labels,
            each channel in [0, 1]. Default is black.
        margin (float): Fractional inset applied to each panel rect before
            placing the label, so the letter is not clipped at the panel
            edge. Default is 0.01.
        panel_scale (float): Fraction of each panel cell filled by the image
            (0–1). The image is anchored to the bottom-right of the cell so
            the top-left corner is free for the panel label. Default is 0.92.
    """
    canvas, page_w, page_h = _create_canvas(figsize)

    for idx, panel in enumerate(panels):
        x0_frac, y0_frac, x1_frac, y1_frac = panel["rect"]
        x0 = x0_frac * page_w
        y0 = y0_frac * page_h
        panel_w = (x1_frac - x0_frac) * page_w
        panel_h = (y1_frac - y0_frac) * page_h

        src_root = _load_svg(panel["path"])
        _embed_panel(canvas, src_root, x0, y0, panel_w, panel_h, panel_scale=panel_scale)

        if labels and idx < len(string.ascii_uppercase):
            _add_panel_label(
                canvas,
                label=string.ascii_uppercase[idx],
                x=x0 + margin * page_w,
                y=y0 + margin * page_h + label_fontsize,
                fontsize=label_fontsize,
                color=label_color,
            )

    _save_svg(canvas, output_path)
    print(f"Saved composed figure to: {output_path}")
