# Report Style Notes

The preferred LaTeX build uses XeLaTeX with EB Garamond for body text, Source Sans Pro for headings, and JetBrains Mono for code.

The current sandbox does not provide `xelatex` or `dvisvgm`, so `scripts/build_pdf_report.py` writes the LaTeX sources and then renders the delivered PDF through a deterministic Matplotlib fallback. The fallback uses DejaVu Serif and DejaVu Sans, which are bundled with Matplotlib, while preserving the project palette, charcoal cover page, header/footer treatment, and wireframe cube mark.
