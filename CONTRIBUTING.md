# Contributing

This guide covers everything needed to work on WhoDidThis — dev environment setup,
how to run tests, code style requirements, and the PR checklist.

---

## Dev Environment Setup

### 1. Clone and install in editable mode

```bash
git clone git@github.com:IdoWolf42/WhoDidThis.git
cd WhoDidThis
pip install -e ".[dev]"
```

This installs the package in editable mode plus all dev tools (`pytest`, `ruff`, `mypy`).

### 2. (Optional) Install SynthID dependencies

```bash
pip install -e ".[synthid]"
```

Required only if you are working on `SynthIDSignal` or running SynthID-related tests/examples.

### 3. Verify the setup

```bash
pytest tests/unit/     # should pass with zero failures
python examples/classify_metadata.py   # should print classification results
```

---

## Running Tests

```bash
# Unit tests only (fast, no optional deps required)
pytest tests/unit/

# System tests only (end-to-end with real fixtures)
pytest tests/system/

# All tests
pytest tests/

# With coverage
pytest tests/ --cov=whodidthis --cov-report=term-missing

# Only system tests (using pytest marker)
pytest tests/ -m system

# Only unit tests (using pytest marker)
pytest tests/ -m unit
```

### Test categories

| Command | Speed | Requires optional deps | Purpose |
|---|---|---|---|
| `pytest tests/unit/` | Fast | No | Isolated logic tests |
| `pytest tests/system/` | Moderate | No (SynthID tests auto-skip) | End-to-end pipeline |
| `pytest tests/` | Moderate | No | Full suite |

---

## Code Style

### Linting

```bash
ruff check .
ruff check . --fix    # auto-fix safe issues
```

### Type checking

```bash
mypy whodidthis/
```

### Formatting

Ruff handles formatting. Line length is 100. Target: Python 3.10+.

All new code must:
- Pass `ruff check .` with no errors
- Pass `mypy whodidthis/` with no errors under `strict = true`
- Use `from __future__ import annotations` at the top of every module
- Have docstrings on all public classes and methods

---

## Adding a New Signal

See [ARCHITECTURE.md](ARCHITECTURE.md#how-to-add-a-new-signal) for the complete step-by-step guide.

Summary:
1. Create `whodidthis/signals/my_signal.py`, subclass `Signal`, implement `analyze()`.
2. Register in `_default_signals()` in `classifier.py` (if it should run by default).
3. Add unit tests in `tests/unit/signals/test_my_signal.py`.
4. Add a fixture and system test in `tests/system/`.
5. Update `ARCHITECTURE.md` (Signal Inventory section + table).
6. Update `README.md` (signal overview table).
7. Add optional dep to `pyproject.toml` if needed.

---

## PR Checklist

Before opening a pull request, verify:

- [ ] `pytest tests/unit/` passes with no failures
- [ ] `pytest tests/system/` passes with no failures
- [ ] `ruff check .` reports no errors
- [ ] `mypy whodidthis/` reports no errors
- [ ] New signal? `ARCHITECTURE.md` updated (inventory + table)
- [ ] New signal? `README.md` signal table updated
- [ ] New signal? Unit tests added in `tests/unit/signals/`
- [ ] New signal? System test fixture added in `tests/system/fixtures/`
- [ ] New optional dep? Added to `pyproject.toml` and documented in `ARCHITECTURE.md`
- [ ] No secrets, API keys, or private watermarking keys committed

---

## Repository Layout Quick Reference

```
whodidthis/            # library source
  __init__.py          # public API exports
  classifier.py        # ContentClassifier — touch this to add/remove default signals
  result.py            # data classes — only change if the data model changes
  signals/
    base.py            # Signal ABC — the contract every signal must fulfill
    metadata.py        # MetadataSignal — add model patterns to MODEL_PATTERNS list
    synthid.py         # SynthIDSignal — SynthID watermark detection

tests/
  unit/                # isolated tests, no network, no model loading
  system/              # full pipeline tests using fixture files
    fixtures/          # add realistic sample content here

examples/              # runnable scripts demonstrating the library
ARCHITECTURE.md        # design reference — keep up to date
CONTRIBUTING.md        # this file
README.md              # user-facing quick-start
```
