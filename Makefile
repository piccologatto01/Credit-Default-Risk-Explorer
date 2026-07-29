PYTHON ?= python3

.PHONY: demo analyze dashboard test clean-results

demo:
	$(PYTHON) scripts/generate_demo_data.py
	$(PYTHON) -m src.risk_analytics.pipeline --data-dir data/demo --output-dir data/processed

analyze:
	$(PYTHON) -m src.risk_analytics.pipeline --data-dir data/raw --output-dir data/processed

dashboard:
	$(PYTHON) app.py

test:
	$(PYTHON) -m pytest -q

clean-results:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('data/processed').glob('*') if p.is_file()]"
