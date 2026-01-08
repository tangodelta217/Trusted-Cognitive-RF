# Cognitive Trusted RF Receiver
# Makefile for INDRA demo

PYTHON := python
PYTEST := $(PYTHON) -m pytest
PIP := $(PYTHON) -m pip

.PHONY: help setup test demo report wow clean

help:
	@echo "Cognitive Trusted RF Receiver - Makefile"
	@echo "========================================="
	@echo ""
	@echo "Targets:"
	@echo "  make setup   - Install dependencies"
	@echo "  make test    - Run all pytest tests"
	@echo "  make demo    - Run offline demo (single sample)"
	@echo "  make report  - Generate metrics report"
	@echo "  make wow     - Run full WOW verification check"
	@echo "  make clean   - Clean generated artifacts"
	@echo ""
	@echo "Quick start:"
	@echo "  make setup && make test && make wow"

setup:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements_ml.txt
	@echo ""
	@echo "Done! Run 'make test' to verify installation."

test:
	@echo "Running pytest tests..."
	$(PYTEST) tests/ -v --tb=short
	@echo ""
	@echo "Running feature tests..."
	$(PYTHON) -m unittest common.features.tests.test_features_v0 -v

demo:
	@echo "Running INDRA offline demo..."
	$(PYTHON) -m tools.demo
	@echo ""
	@echo "Demo complete! Check reports/demo/ for outputs."

report:
	@echo "Generating INDRA Benchmark Report..."
	$(PYTHON) -m tools.make_report
	@echo ""
	@echo "Report complete! Check docs/indra_pack/INDRA_BenchmarkReport_v1.md"

wow:
	@echo "============================================"
	@echo "Running FULL WOW Check for INDRA Demo"
	@echo "============================================"
	@echo ""
	@echo "[1/5] Running tests..."
	$(PYTEST) tests/ -q --tb=short
	@echo ""
	@echo "[2/5] Running demo..."
	$(PYTHON) -m tools.demo -q
	@echo ""
	@echo "[3/5] Running latency benchmark..."
	$(PYTHON) -m tools.run_bundle --benchmark
	@echo ""
	@echo "[4/5] Generating report..."
	$(PYTHON) -m tools.make_report
	@echo ""
	@echo "[5/5] Verifying WOW criteria..."
	$(PYTHON) -m tools.wow_check --verbose
	@echo ""
	@echo "============================================"
	@echo "WOW check complete!"
	@echo "============================================"
	@echo "Artefactos:"
	@echo "  - docs/indra_pack/INDRA_BenchmarkReport_v1.md"
	@echo "  - reports/demo/waterfall.png"
	@echo "  - reports/figures/*.png"
	@echo ""

clean:
	@echo "Cleaning generated artifacts..."
	-rm -rf runs/wow_check/
	-rm -rf __pycache__/
	-find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	-find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"

# Alias for Windows compatibility
.PHONY: all
all: help
