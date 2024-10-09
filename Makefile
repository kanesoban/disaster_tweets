install:
	pip install .[dev]

fix_lint:
	black src/ tests/ --line-length 120
	isort src/ tests/
	flake8 src/ --max-line-length 120
	toml-sort pyproject.toml -i

check_lint:
	black --check src/ tests/ --line-length 120
	isort --check src/ tests/
	flake8 src/ --max-line-length 120
	toml-sort --check pyproject.toml
	mypy src --ignore-missing-imports
