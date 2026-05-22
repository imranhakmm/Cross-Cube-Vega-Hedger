from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

README_FIGURE_DIR = Path("outputs/figures/readme")
PDF_PATH = Path("docs/report/report.pdf")
ARCHITECTURE_SVG = Path("docs/report/templates/_architecture.svg")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _cover_with_pdftoppm(destination: Path) -> bool:
    if shutil.which("pdftoppm") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = Path(tmp_dir) / "cover"
        _run(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-f",
                "1",
                "-l",
                "1",
                "-scale-to-x",
                "600",
                "-scale-to-y",
                "-1",
                str(PDF_PATH),
                str(prefix),
            ]
        )
        shutil.copyfile(prefix.with_suffix(".png"), destination)
    return True


def _cover_with_pdf2image(destination: Path) -> bool:
    try:
        from pdf2image import convert_from_path
    except Exception:
        return False
    image = convert_from_path(PDF_PATH, first_page=1, last_page=1, size=(600, None))[0]
    image.save(destination)
    return True


def _cover_with_sips(destination: Path) -> None:
    if shutil.which("sips") is None:
        raise RuntimeError("Need pdftoppm, pdf2image, or macOS sips to render the PDF cover.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw = Path(tmp_dir) / "cover_raw.png"
        _run(["sips", "-s", "format", "png", str(PDF_PATH), "--out", str(raw)])
        _run(["sips", "-z", "776", "600", str(raw), "--out", str(destination)])


def build_cover_thumbnail() -> Path:
    """Render page 1 of the report PDF to a 600px-wide PNG."""

    destination = README_FIGURE_DIR / "cover_thumbnail.png"
    if not _cover_with_pdftoppm(destination) and not _cover_with_pdf2image(destination):
        _cover_with_sips(destination)
    return destination


def build_architecture_png() -> Path:
    """Render the inline architecture SVG to a 1200px-wide PNG."""

    from playwright.sync_api import sync_playwright

    destination = README_FIGURE_DIR / "architecture.png"
    font_url = Path("docs/report/style/fonts/SourceSans3.ttf").resolve().as_uri()
    svg = ARCHITECTURE_SVG.read_text(encoding="utf-8")
    svg_style = (
        "<style>"
        f"@font-face{{font-family:'Source Sans 3';src:url('{font_url}') format('truetype');}}"
        "text{font-family:'Source Sans 3',sans-serif!important;font-weight:600;}"
        "</style>"
    )
    svg = svg.replace(">", f">{svg_style}", 1)
    html = f"""
<!doctype html>
<html>
<head>
  <link rel="stylesheet" href="{Path('docs/report/style/report.css').resolve().as_uri()}">
  <style>
    body {{ margin: 0; background: white; }}
    .frame {{ width: 1200px; padding: 20px 0; }}
    svg {{ width: 1200px; height: auto; display: block; }}
  </style>
</head>
<body><div class="frame">{svg}</div></body>
</html>
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        html_path = Path(tmp_dir) / "architecture.html"
        html_path.write_text(html, encoding="utf-8")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                if Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").exists()
                else None,
            )
            page = browser.new_page(viewport={"width": 1200, "height": 240}, device_scale_factor=1)
            page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            page.evaluate("document.fonts.ready")
            page.locator(".frame").screenshot(path=str(destination))
            browser.close()
    return destination


def main() -> None:
    README_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cover = build_cover_thumbnail()
    architecture = build_architecture_png()
    print(f"wrote {cover}")
    print(f"wrote {architecture}")


if __name__ == "__main__":
    main()
