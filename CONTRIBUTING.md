# Contributing

## Development Setup

```bash
# Clone and create virtual environment
git clone <repo-url> && cd sports-betting
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests

```bash
# Full suite with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Fast feedback (skip integration tests)
python -m pytest tests/ --ignore=tests/test_integration.py

# Single module
python -m pytest tests/test_math_engine.py -v
```

## Code Standards

- **Type hints**: All function signatures must have complete type annotations
- **Docstrings**: NumPy-style docstrings on all public methods
- **Tests**: Every new feature must include tests. Minimum bar: positive case, negative case, edge case
- **No emoji**: Keep all user-facing text (README, docstrings, logs) free of emoji characters

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scanner): add three-way market arb detection
fix(ingestion): handle empty bookmaker array gracefully
test: add Kelly criterion edge case tests
perf: vectorize EV calculation loop
docs: update architecture diagram
refactor(math): extract odds conversion to standalone module
```

## Architecture Rules

1. **MathEngine is stateless** -- all methods are `@staticmethod` with `@lru_cache`
2. **Scanner never calls the API** -- it receives pre-fetched event data
3. **Data ingestion is async** -- all fetch methods are coroutines
4. **Mock mode is always available** -- no API key required for development
5. **Dashboard imports from src/** -- no business logic in `app.py`

## Adding a New Market Type

1. Add the market key to `SUPPORTED_MARKETS` in `data_ingestion.py`
2. Add mock data generation in `_build_mock_props()` or a new builder
3. Ensure `Scanner._build_outcome_pairs()` handles the outcome naming pattern
4. Add tests in `test_scanner.py` and `test_data_ingestion.py`
5. Update the README configuration table

## Adding a New Bookmaker

1. Add the bookmaker key to `BOOKMAKERS` in `data_ingestion.py`
2. If it should be treated as sharp, update `SHARP_BOOK` in `scanner.py`
3. If soft, add to `SOFT_BOOKS` in `scanner.py`
4. Add mock data entries in the builders
5. Update tests to verify the new book appears in mock data
