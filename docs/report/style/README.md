# Report Style Notes

The report is rendered from `docs/report/templates/report.html.j2` and `docs/report/style/report.css`.

The preferred renderer is WeasyPrint. In this sandbox, WeasyPrint installs but cannot load the native Pango/GObject stack, so the build falls back to Playwright/headless Chrome. The Matplotlib renderer remains available only through `scripts/build_pdf_report.py --fallback-matplotlib`.

Local fonts are bundled under `docs/report/style/fonts/`: EB Garamond for body text, Source Sans 3 for headings and tables, and JetBrains Mono for monospace/equation-adjacent styling. The same stylesheet and font files are copied into `docs/site/assets/` during the static-site build so the PDF and HTML site share typography and palette.
