.PHONY: install lint test simulate figures site report all clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev,viz,docs]"

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m mypy src/cxvega

test:
	$(PYTHON) -m pytest

simulate:
	$(PYTHON) scripts/run_mm_simulation.py --config configs/default.yaml

figures:
	$(PYTHON) scripts/generate_figures.py --config configs/default.yaml

site:
	$(PYTHON) scripts/build_html_site.py --config configs/default.yaml

report:
	$(PYTHON) scripts/build_pdf_report.py --config configs/default.yaml

all: clean install lint test simulate figures site report

clean:
	rm -rf outputs/figures outputs/tables outputs/simulations
	rm -rf docs/site/*.html docs/site/assets
	rm -f docs/report/report.pdf docs/report/*.aux docs/report/*.bbl docs/report/*.bcf docs/report/*.blg docs/report/*.fdb_latexmk docs/report/*.fls docs/report/*.log docs/report/*.out docs/report/*.run.xml docs/report/*.toc docs/report/*.xdv
	mkdir -p outputs/figures outputs/tables outputs/simulations docs/report/figures docs/site
