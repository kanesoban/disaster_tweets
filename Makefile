install:
	conda create -n disaster_tweets3 python=3.10
	conda activate disaster_tweets
	pip install -e .[dev]
	python -m spacy download en

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
