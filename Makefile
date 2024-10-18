fix_lint:
	black src/ --line-length 120
	isort src/
	flake8 src/ --max-line-length 120
	toml-sort pyproject.toml -i

check_lint:
	black --check src/ --line-length 120
	isort --check src/
	flake8 src/ --max-line-length 120
	toml-sort --check pyproject.toml
	mypy src --ignore-missing-imports
