# Examples

Runnable scripts demonstrating the WhoDidThis library.

---

## classify_metadata.py

Classifies content using `MetadataSignal` — inspects JSON metadata fields and
free-text model-name patterns.

**No optional dependencies required.**

```bash
python examples/classify_metadata.py
```

Demonstrates classification of:
- A Gemini API response (confidence: 0.95 via `model` JSON field)
- An OpenAI Chat Completions response (confidence: 0.95 via `model` JSON field)
- An Anthropic Messages API response (confidence: 0.95 via `model` JSON field)
- Human-written text (no signals fired)
- Text that mentions GPT-4 by name (confidence: 0.60 via free-text pattern)

---

## classify_with_synthid.py

Classifies content using `SynthIDSignal` — detects Google DeepMind's SynthID watermark.

**Requires the `[synthid]` optional dependencies:**

```bash
pip install "whodidthis[synthid]"
python examples/classify_with_synthid.py
```

> **Note**: SynthID is a watermarking system, not a general AI-text detector.
> Without Google's private production watermarking keys, confidence is limited.
> Expect low/no confidence for arbitrary text samples.
> See `whodidthis/signals/synthid.py` for full limitations.

---

## Writing your own example

```python
from whodidthis import ContentClassifier

classifier = ContentClassifier()
result = classifier.classify("your content here")

print(f"Model: {result.model_attribution}")
print(f"Confidence: {result.confidence:.2f}")
print(f"AI-generated: {result.is_ai_generated}")
for signal in result.signals:
    print(f"  [{signal.signal_name}] {signal.confidence:.2f} — {signal.evidence}")
```
