PYTHON ?= .venv/bin/python
# TeX binaries: TinyTeX (user install) or MacTeX/BasicTeX
export PATH := $(HOME)/Library/TinyTeX/bin/universal-darwin:/Library/TeX/texbin:$(PATH)

.PHONY: all setup test data pipeline figures forecast paper clean

all: data pipeline figures forecast

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) scripts/setup_r.py
	$(PYTHON) scripts/setup_latex.py

test:
	$(PYTHON) -m pytest -q tests

data:
	$(PYTHON) -m src.data

pipeline:
	$(PYTHON) -m src.pipeline

figures:
	$(PYTHON) -m src.figures

forecast:
	$(PYTHON) -m src.forecast

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	rm -rf data/processed/*.pkl
	cd paper && latexmk -C
