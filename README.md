# WhoDidThis

Content model classifier — determines which AI model (or a human) produced a piece of text,
using a pipeline of detection signals: structured metadata inspection, SynthID watermark
detection, and more.

---

## Quick Start

```bash
pip install -e .
```

```python
from whodidthis import ContentClassifier

classifier = ContentClassifier()
result = classifier.classify(content)

print(result.model_attribution)   # "Google Gemini" | "OpenAI GPT-4" | None
print(result.confidence)          # float in [0.0, 1.0]
print(result.is_ai_generated)     # True | False | None
print(result.signals)             # list of SignalResult, sorted by confidence desc
```

### Example: classifying an OpenAI API response

```python
import json
from whodidthis import ContentClassifier

api_response = json.dumps({
    "model": "gpt-4o",
    "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
})

result = ContentClassifier().classify(api_response)
# result.model_attribution → "OpenAI GPT-4o"
# result.confidence        → 0.95
# result.is_ai_generated   → True
# result.signals[0].evidence → {"source": "json_metadata_field", "key": "model", ...}
```

---

## Signals

Signals are detection strategies run in priority order (cheapest first). Each returns a
`SignalResult` with an attribution, confidence score, and evidence dict, or `None` if
it has no opinion on the content.

| Signal | Priority | What it detects | Optional dep |
|---|---|---|---|
| `MetadataSignal` | 10 | Model name in JSON metadata or free-text | None |
| `SynthIDSignal` | 20 | Google SynthID watermark in text | `whodidthis[synthid]` |

### SynthID (optional)

```bash
pip install "whodidthis[synthid]"   # installs transformers>=4.46.0, torch
```

> **Note**: SynthID detects only watermarks it applied. Without Google's private
> production keys, confidence is limited. It cannot detect Claude, GPT-4, or other
> models that don't embed SynthID watermarks.

---

## Output Structure

```python
@dataclass
class ClassificationResult:
    model_attribution: str | None   # best guess at originating model
    confidence: float               # 0.0–1.0
    is_ai_generated: bool | None    # True / False / None (uncertain)
    signals: list[SignalResult]     # all fired signals, sorted by confidence desc

@dataclass
class SignalResult:
    signal_name: str                # "metadata" | "synthid" | ...
    model_attribution: str | None
    confidence: float               # 0.0–1.0
    evidence: dict                  # raw data that produced this result
```

### Confidence scale

| Range | Meaning |
|---|---|
| 0.9–1.0 | Strong structural evidence (explicit metadata field) |
| 0.7–0.9 | Good evidence (SynthID score above threshold) |
| 0.5–0.7 | Moderate / heuristic evidence (free-text model mention) |
| 0.0–0.5 | Weak evidence |

---

## Customizing Signals

```python
from whodidthis import ContentClassifier
from whodidthis.signals.metadata import MetadataSignal
from whodidthis.signals.synthid import SynthIDSignal  # requires [synthid]

# Use only specific signals
classifier = ContentClassifier(signals=[MetadataSignal()])

# Adjust thresholds
classifier = ContentClassifier(
    ai_confidence_threshold=0.7,    # default 0.5
    stop_on_high_confidence=False,  # default True (stops at confidence >= 0.9)
)
```

---

## Running Tests

```bash
pytest tests/unit/     # fast unit tests, no optional deps required
pytest tests/system/   # end-to-end tests with realistic fixtures
pytest tests/          # all tests
```

---

## Running Examples

```bash
# No optional deps needed
python examples/classify_metadata.py

# Requires [synthid] extra
pip install "whodidthis[synthid]"
python examples/classify_with_synthid.py
```

---

## Project Structure

```
whodidthis/          # library source
tests/unit/          # isolated unit tests
tests/system/        # end-to-end system tests + fixtures
examples/            # runnable classification examples
ARCHITECTURE.md      # design, data flow, how to add signals
CONTRIBUTING.md      # dev setup, code style, PR checklist
```

For a deep dive into the design and how to add new signals, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
