# System Tests

End-to-end tests for WhoDidThis. These run the full `ContentClassifier` pipeline
with no mocks, using realistic fixture content files.

---

## What are system tests?

Unlike unit tests (which test components in isolation with mocks), system tests:
- Use the real `ContentClassifier` with real signals
- Load real fixture files from `tests/system/fixtures/`
- Assert on the output of the full pipeline

They verify that the system behaves correctly as a whole.

---

## Running

```bash
# System tests only
pytest tests/system/

# With verbose output
pytest tests/system/ -v

# Using the pytest marker
pytest tests/ -m system

# All tests (unit + system)
pytest tests/
```

---

## Fixtures

Fixture files live in `tests/system/fixtures/` and represent realistic content shapes.

| File | Description | Expected outcome |
|---|---|---|
| `gemini_api_response.json` | Realistic Gemini API response JSON | Attribution: "Google Gemini", confidence ≥ 0.9 |
| `openai_api_response.json` | Realistic OpenAI Chat Completions response | Attribution: "OpenAI GPT-4o", confidence ≥ 0.9 |
| `anthropic_api_response.json` | Realistic Anthropic Messages API response | Attribution: "Anthropic Claude 3", confidence ≥ 0.9 |
| `human_written.txt` | Genuine human-written personal text | No attribution, no signals |
| `unknown_ai_text.txt` | Generic AI-sounding text with no model mention | No attribution (no metadata) |
| `mixed_content_with_model_mention.txt` | Human text that mentions GPT-4 by name | Attribution via free-text, medium confidence |

### Adding a new fixture

1. Add the content file to `tests/system/fixtures/`.
2. Add test cases in `tests/system/test_end_to_end.py`.
3. Document the fixture in the table above.

Use realistic content shapes — actual API response structures, not toy examples.

---

## Optional Dependency Tests

Some system tests check `SynthIDSignal` behavior. These tests auto-skip or auto-adjust
based on whether `transformers` is installed:

- If `transformers` is **not** installed: tests verify that `SynthIDSignal.analyze()`
  raises `ImportError` with a helpful message, and that the default classifier silently
  omits SynthID.
- If `transformers` **is** installed: tests verify that the default classifier includes
  `SynthIDSignal`.

No manual configuration needed — the tests detect the environment at runtime.

---

## Classifier Used in System Tests

Most system tests use `ContentClassifier(signals=[MetadataSignal()])` to keep the
test suite fast and dependency-free. This is intentional — metadata-based detection
is the primary signal for system tests.

SynthID end-to-end tests (which would require loading a transformer model) are not
included because they would be slow and require the `[synthid]` extra. They are
covered in `tests/unit/signals/test_synthid.py` via mocking.
