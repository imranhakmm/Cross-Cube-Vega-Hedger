# Build Notes

## PDF Renderer

`scripts/build_pdf_report.py` renders `docs/report/templates/report.html.j2` with Jinja2 and writes `docs/report/report.pdf` from the generated HTML.

The preferred renderer is WeasyPrint. In the current macOS sandbox, the Python package installs but cannot load the native Pango/GObject library stack (`libgobject-2.0-0`), so the default build falls back automatically to Playwright driving the installed Google Chrome application. This keeps the PDF on an HTML/CSS renderer and avoids the old Matplotlib PDF path.

Use the emergency renderer only when debugging the HTML/CSS stack:

```bash
python scripts/build_pdf_report.py --fallback-matplotlib
```

## Fonts

The report and static site use local font files under `docs/report/style/fonts/` so the build is offline-safe after the first install.

| Family | File | Source |
|---|---|---|
| EB Garamond | `EBGaramond.ttf`, `EBGaramond-Italic.ttf` | `https://github.com/google/fonts/tree/main/ofl/ebgaramond` |
| Source Sans 3 | `SourceSans3.ttf` | `https://github.com/google/fonts/tree/main/ofl/sourcesans3` |
| JetBrains Mono | `JetBrainsMono.ttf` | `https://github.com/google/fonts/tree/main/ofl/jetbrainsmono` |

## Known Sandbox Issues

WeasyPrint requires native text/rendering libraries in addition to the Python wheel. If those are missing, the build logs the WeasyPrint import/rendering failure and uses Playwright with a system Chrome/Chromium executable. If neither WeasyPrint nor a Chromium executable is available, install Chrome or run the explicit Matplotlib fallback above.
