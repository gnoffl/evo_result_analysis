import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from workflows.figure_composition_svg import compose_figures, _PX_PER_INCH, _SVG_NS

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_svg(path: str, width: float = 200, height: float = 200, text: str = "test") -> None:
    """Create a minimal SVG file at *path*."""
    root = ET.Element(
        "svg",
        attrib={
            "xmlns": _SVG_NS,
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )
    txt = ET.SubElement(root, "text", attrib={"x": "10", "y": "20", "font-size": "12"})
    txt.text = text
    tree = ET.ElementTree(root)
    tree.write(path, encoding="unicode", xml_declaration=False)


def _svg_root(out_path: str) -> ET.Element:
    """Parse the output SVG and return the root element."""
    tree = ET.parse(out_path)
    return tree.getroot()


def _svg_dimensions(out_path: str) -> "tuple[float, float]":
    """Return (width_px, height_px) from the output SVG viewBox."""
    root = _svg_root(out_path)
    vb = root.get("viewBox") or root.get("viewbox", "")
    parts = vb.replace(",", " ").split()
    return float(parts[2]), float(parts[3])


def _all_text_content(out_path: str) -> str:
    """Collect all text present in <text> elements of the output SVG."""
    root = _svg_root(out_path)
    texts = []
    for el in root.iter(f"{{{_SVG_NS}}}text"):
        if el.text:
            texts.append(el.text)
    # Also try without namespace (in case root has no explicit ns)
    for el in root.iter("text"):
        if el.text:
            texts.append(el.text)
    return " ".join(texts)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestComposeFileCreation(unittest.TestCase):
    """Output file is created and is a valid SVG."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.svg_a = os.path.join(self.tmp.name, "a.svg")
        self.svg_b = os.path.join(self.tmp.name, "b.svg")
        _make_svg(self.svg_a, text="Panel A")
        _make_svg(self.svg_b, text="Panel B")

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_file_is_created(self):
        out = os.path.join(self.tmp.name, "out.svg")
        compose_figures(
            panels=[{"path": self.svg_a, "rect": (0.0, 0.0, 1.0, 1.0)}],
            output_path=out,
        )
        self.assertTrue(os.path.exists(out))

    def test_output_is_valid_svg(self):
        out = os.path.join(self.tmp.name, "out.svg")
        compose_figures(
            panels=[{"path": self.svg_a, "rect": (0.0, 0.0, 1.0, 1.0)}],
            output_path=out,
        )
        # Should parse without error and have an <svg> root
        root = _svg_root(out)
        self.assertIn("svg", root.tag.lower())

    def test_two_panels_side_by_side(self):
        out = os.path.join(self.tmp.name, "out.svg")
        compose_figures(
            panels=[
                {"path": self.svg_a, "rect": (0.0, 0.0, 0.5, 1.0)},
                {"path": self.svg_b, "rect": (0.5, 0.0, 1.0, 1.0)},
            ],
            output_path=out,
        )
        self.assertTrue(os.path.exists(out))

    def test_four_panels_grid(self):
        out = os.path.join(self.tmp.name, "out.svg")
        compose_figures(
            panels=[
                {"path": self.svg_a, "rect": (0.0, 0.0, 0.5, 0.5)},
                {"path": self.svg_b, "rect": (0.5, 0.0, 1.0, 0.5)},
                {"path": self.svg_a, "rect": (0.0, 0.5, 0.5, 1.0)},
                {"path": self.svg_b, "rect": (0.5, 0.5, 1.0, 1.0)},
            ],
            output_path=out,
        )
        self.assertTrue(os.path.exists(out))


class TestComposeCanvasSize(unittest.TestCase):
    """Output canvas dimensions match the requested figsize."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.svg_a = os.path.join(self.tmp.name, "a.svg")
        _make_svg(self.svg_a)

    def tearDown(self):
        self.tmp.cleanup()

    def _output_canvas_size(self, figsize):
        out = os.path.join(self.tmp.name, "out.svg")
        compose_figures(
            panels=[{"path": self.svg_a, "rect": (0.0, 0.0, 1.0, 1.0)}],
            output_path=out,
            figsize=figsize,
        )
        return _svg_dimensions(out)

    def test_default_a4_portrait(self):
        w, h = self._output_canvas_size((8.27, 11.69))
        self.assertAlmostEqual(w, 8.27 * _PX_PER_INCH, places=1)
        self.assertAlmostEqual(h, 11.69 * _PX_PER_INCH, places=1)

    def test_a4_landscape(self):
        w, h = self._output_canvas_size((11.69, 8.27))
        self.assertAlmostEqual(w, 11.69 * _PX_PER_INCH, places=1)
        self.assertAlmostEqual(h, 8.27 * _PX_PER_INCH, places=1)

    def test_custom_size(self):
        w, h = self._output_canvas_size((5.0, 4.0))
        self.assertAlmostEqual(w, 5.0 * _PX_PER_INCH, places=1)
        self.assertAlmostEqual(h, 4.0 * _PX_PER_INCH, places=1)


class TestComposeLabels(unittest.TestCase):
    """Panel labels are present / absent according to the *labels* flag."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.svg_a = os.path.join(self.tmp.name, "a.svg")
        self.svg_b = os.path.join(self.tmp.name, "b.svg")
        _make_svg(self.svg_a)
        _make_svg(self.svg_b)

    def tearDown(self):
        self.tmp.cleanup()

    def test_labels_true_adds_A_and_B(self):
        out = os.path.join(self.tmp.name, "out.svg")
        compose_figures(
            panels=[
                {"path": self.svg_a, "rect": (0.0, 0.0, 0.5, 1.0)},
                {"path": self.svg_b, "rect": (0.5, 0.0, 1.0, 1.0)},
            ],
            output_path=out,
            labels=True,
        )
        text = _all_text_content(out)
        self.assertIn("A", text)
        self.assertIn("B", text)

    def test_labels_false_no_standalone_label(self):
        # SVGs contain only numeric text, so no A/B should appear without labels=True
        out = os.path.join(self.tmp.name, "out.svg")
        _make_svg(self.svg_a, text="111")
        _make_svg(self.svg_b, text="222")
        compose_figures(
            panels=[
                {"path": self.svg_a, "rect": (0.0, 0.0, 0.5, 1.0)},
                {"path": self.svg_b, "rect": (0.5, 0.0, 1.0, 1.0)},
            ],
            output_path=out,
            labels=False,
        )
        text = _all_text_content(out)
        self.assertNotIn("A", text)
        self.assertNotIn("B", text)

    def test_labels_up_to_26_panels(self):
        """Labels should be added for up to 26 panels (A–Z)."""
        import string
        n = 8
        _make_svg(self.svg_a, text="111")
        panels = [
            {"path": self.svg_a, "rect": (i / n, 0.0, (i + 1) / n, 1.0)}
            for i in range(n)
        ]
        out = os.path.join(self.tmp.name, "out.svg")
        compose_figures(panels=panels, output_path=out, labels=True)
        text = _all_text_content(out)
        for letter in string.ascii_uppercase[:n]:
            self.assertIn(letter, text)


class TestComposePanelKeyIgnored(unittest.TestCase):
    """The ``page`` key (legacy from PDF mode) is silently ignored."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.svg_a = os.path.join(self.tmp.name, "a.svg")
        _make_svg(self.svg_a, text="hello")

    def tearDown(self):
        self.tmp.cleanup()

    def test_page_key_is_ignored(self):
        """Specifying ``page`` for an SVG panel must not raise an error."""
        out = os.path.join(self.tmp.name, "out.svg")
        compose_figures(
            panels=[{"path": self.svg_a, "rect": (0.0, 0.0, 1.0, 1.0), "page": 0}],
            output_path=out,
        )
        self.assertTrue(os.path.exists(out))


class TestComposeEdgeCases(unittest.TestCase):
    """Edge cases and defensive behaviour."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.svg_a = os.path.join(self.tmp.name, "a.svg")
        _make_svg(self.svg_a)

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_panel_full_canvas(self):
        out = os.path.join(self.tmp.name, "out.svg")
        compose_figures(
            panels=[{"path": self.svg_a, "rect": (0.0, 0.0, 1.0, 1.0)}],
            output_path=out,
        )
        self.assertTrue(os.path.exists(out))

    def test_same_svg_used_twice(self):
        """The same source SVG can appear in multiple panels."""
        out = os.path.join(self.tmp.name, "out.svg")
        compose_figures(
            panels=[
                {"path": self.svg_a, "rect": (0.0, 0.0, 0.5, 1.0)},
                {"path": self.svg_a, "rect": (0.5, 0.0, 1.0, 1.0)},
            ],
            output_path=out,
        )
        self.assertTrue(os.path.exists(out))

    def test_missing_source_file_raises(self):
        out = os.path.join(self.tmp.name, "out.svg")
        with self.assertRaises(Exception):
            compose_figures(
                panels=[{"path": "/nonexistent/path/plot.svg", "rect": (0.0, 0.0, 1.0, 1.0)}],
                output_path=out,
            )


if __name__ == "__main__":
    unittest.main()
