# Makefile -- Test & Quality Commands
# Usage: make test | make test-fast | make coverage

.PHONY: test test-fast coverage lint

# Full test suite with coverage
test:
	python -m pytest tests/ -v --cov=src --cov=app --cov-report=term-missing --tb=short

# Skip integration tests (fast feedback loop)
test-fast:
	python -m pytest tests/ -v --ignore=tests/test_integration.py --tb=short

# HTML coverage report (open htmlcov/index.html)
coverage:
	python -m pytest tests/ --cov=src --cov=app --cov-report=html --cov-report=term-missing
	@echo "Open htmlcov/index.html for full report"

# Code quality checks
lint:
	python -m flake8 src/ app.py --max-line-length=100 --ignore=E501,W503
