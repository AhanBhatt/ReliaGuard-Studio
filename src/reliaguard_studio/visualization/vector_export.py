from __future__ import annotations

import json
from pathlib import Path

from .figure_canvas import FigureCanvas
from .figure_registry import figure_by_stem


def _svg_to_pdf_png_with_pymupdf(svg_path: Path, pdf_path: Path, png_path: Path, *, zoom: float = 3.2) -> None:
    import fitz

    svg_doc = fitz.open(str(svg_path))
    pdf_bytes = svg_doc.convert_to_pdf()
    pdf_path.write_bytes(pdf_bytes)
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = pdf_doc[0]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pixmap.save(str(png_path))
    pdf_doc.close()
    svg_doc.close()


def _svg_to_pdf_png_with_cairosvg(svg_path: Path, pdf_path: Path, png_path: Path, width: int, height: int, *, png_dpi: int) -> None:
    import cairosvg

    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path), output_width=width, output_height=height)
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=int(width * png_dpi / 96),
        output_height=int(height * png_dpi / 96),
    )


def _write_metadata(stem: str, svg_path: Path, pdf_path: Path, png_path: Path, canvas: FigureCanvas, backend: str) -> None:
    try:
        figure = figure_by_stem(stem)
        metadata_path = figure.metadata_path
        figure_number = figure.number
        title = figure.title
    except KeyError:
        metadata_path = pdf_path.parent / "metadata" / f"{stem}.json"
        figure_number = None
        title = stem
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "figure_number": figure_number,
        "stem": stem,
        "title": title,
        "backend": backend,
        "browser_used": False,
        "svg": str(svg_path.as_posix()),
        "pdf": str(pdf_path.as_posix()),
        "png": str(png_path.as_posix()),
        "width": canvas.width,
        "height": canvas.height,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def export_vector_canvas(canvas: FigureCanvas, stem: str, output_dir: Path, *, png_dpi: int = 360) -> list[Path]:
    """Export SVG, PDF and PNG without browser printing or headed browser windows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    canvas.save(svg_path)

    backend = "pymupdf"
    try:
        _svg_to_pdf_png_with_cairosvg(svg_path, pdf_path, png_path, canvas.width, canvas.height, png_dpi=png_dpi)
        backend = "cairosvg"
    except Exception:
        _svg_to_pdf_png_with_pymupdf(svg_path, pdf_path, png_path, zoom=max(2.0, png_dpi / 120))
    _write_metadata(stem, svg_path, pdf_path, png_path, canvas, backend)
    return [pdf_path, svg_path, png_path]
