# Architecture

This document is the primary reference for anyone (human or agent) working on WhoDidThis.
It covers the system design, data flow, code layout, and how to extend the project.

---

## Overview

WhoDidThis is a **content model classifier**: given a piece of text, it tries to determine
which AI model (or a human) produced it, along with a confidence score and the evidence trail.

The core design principle is a **pluggable signal pipeline**:

```
content (str)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   ContentClassifier                     │
│                                                         │
│  signals sorted by priority (ascending = cheapest first)│
│                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐  │
│  │MetadataSignal│──▶│SynthIDSignal│──▶│  (future...) │  │
│  │ priority=10  │   │ priority=20  │   │              │  │
│  └──────┬──────┘   └──────┬──────┘   └──────┬───────┘  │
│         │                 │                  │          │
│     SignalResult       SignalResult        None         │
│      or None            or None                        │
│                                                         │
│  ───────────────── aggregation ───────────────────────  │
│  • collect non-None results                             │
│  • sort by confidence descending                        │
│  • best result → model_attribution + confidence         │
│  • any result above threshold → is_ai_generated=True    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
ClassificationResult
  .model_attribution   # "Google Gemini" | "OpenAI GPT-4" | None
  .confidence          # 0.0–1.0
  .is_ai_generated     # True | False | None
  .signals             # [SignalResult, ...] sorted by confidence desc
```

---

## File Map

```
WhoDidThis/
├── pyproject.toml                   # package definition, tool config
├── README.md                        # quick-start for humans
├── ARCHITECTURE.md                  # this file — design reference
├── CONTRIBUTING.md                  # dev setup, workflow, PR checklist
│
├── whodidthis/
│   ├── __init__.py                  # public API: ContentClassifier, ClassificationResult, SignalResult
│   ├── classifier.py                # ContentClassifier — orchestrates signals
│   ├── result.py                    # ClassificationResult, SignalResult dataclasses
│   └── signals/
│       ├── __init__.py
│       ├── base.py                  # Signal abstract base class (implement this to add a signal)
│       ├── metadata.py              # MetadataSignal — JSON + free-text model-name detection
│       └── synthid.py              # SynthIDSignal — SynthID watermark detection (optional dep)
│
├── tests/
│   ├── unit/                        # fast, isolated, no external deps
│   │   ├── test_result.py
│   │   ├── test_classifier.py
│   │   └── signals/
│   │       ├── test_metadata.py
│   │       └── test_synthid.py
│   └── system/                      # end-to-end tests with real fixture content
│       ├── README.md
│       ├── fixtures/                # sample content files
│       └── test_end_to_end.py
│
└── examples/
    ├── README.md
    ├── classify_metadata.py         # runnable demo, no optional deps
    └── classify_with_synthid.py     # runnable demo, requires [synthid]
```

---

## Data Models

### `SignalResult` (`whodidthis/result.py`)

Produced by one signal. Represents what a single detection strategy found.

| Field | Type | Description |
|---|---|---|
| `signal_name` | `str` | Stable identifier of the signal. E.g. `"metadata"`, `"synthid"`. |
| `model_attribution` | `str \| None` | Canonical model/provider name, or `None` if the signal cannot attribute. |
| `confidence` | `float` | 0.0–1.0. See confidence semantics below. |
| `evidence` | `dict` | Raw data that produced this result. Keys vary per signal. |

### `ClassificationResult` (`whodidthis/result.py`)

The aggregated output of `ContentClassifier.classify()`.

| Field | Type | Description |
|---|---|---|
| `model_attribution` | `str \| None` | Attribution from the highest-confidence signal. |
| `confidence` | `float` | Confidence of the best signal. 0.0 if no signals fired. |
| `is_ai_generated` | `bool \| None` | `True` if any signal fired above the threshold (default 0.5). `None` if no signals fired. |
| `signals` | `list[SignalResult]` | All fired signals, sorted by confidence descending. |

Convenience property: `result.best_signal` — the first element of `signals`, or `None`.

---

## Confidence Semantics

All confidence values are floats in `[0.0, 1.0]`. The scale is:

| Range | Meaning |
|---|---|
| `0.0` | No evidence at all (signal returned `None`, not a `SignalResult`) |
| `0.3–0.5` | Weak / heuristic evidence |
| `0.5–0.7` | Moderate evidence (e.g. free-text pattern match) |
| `0.7–0.9` | Strong evidence (e.g. SynthID score above threshold) |
| `0.9–1.0` | Very strong / structural evidence (e.g. explicit `model` field in API response JSON) |

**`is_ai_generated` threshold**: by default, `confidence >= 0.5` means the content is
considered AI-generated. This is configurable via `ContentClassifier(ai_confidence_threshold=...)`.

**Short-circuit threshold**: by default, if any signal returns `confidence >= 0.9`, the
classifier stops running further signals (cheap wins immediately). Configurable via
`ContentClassifier(stop_on_high_confidence=False)`.

---

## Priority System

Signal priority determines execution order. **Lower value = runs first.**

Convention:

| Priority range | Signal type |
|---|---|
| 0–9 | Instant / zero-cost (e.g. regex on in-memory string) |
| 10–19 | Cheap, no external I/O (e.g. JSON parsing + pattern matching) |
| 20–49 | Moderate cost (e.g. local model inference) |
| 50+ | Expensive (e.g. remote API call, large model load) |

Current signals:

| Signal | Priority | Cost | Optional dep |
|---|---|---|---|
| `MetadataSignal` | 10 | Negligible (pure Python, no I/O) | None |
| `SynthIDSignal` | 20 | High (loads transformer model on first call) | `whodidthis[synthid]` |

---

## Signal Inventory

### `MetadataSignal` (`whodidthis/signals/metadata.py`)

**What it detects**: Model name from structured metadata or free-text mentions.

**Strategy**:
1. Try to parse content as JSON.
2. If valid JSON dict, inspect known trusted metadata keys (`model`, `engine`, `model_name`, etc.).
3. If a trusted key contains a known model name pattern → confidence `0.95`.
4. If a non-trusted JSON value matches → confidence `0.60`.
5. If not JSON (or no match in JSON), scan the raw string for model-name patterns → confidence `0.60`.

**Model patterns**: Defined in `MODEL_PATTERNS` list in `metadata.py`. Regex patterns matched
case-insensitively. First match wins. Add new patterns there — no other changes needed.

**Limitations**:
- Only detects models by name. Cannot detect AI authorship without a model mention.
- Free-text matches may produce false positives (e.g. a blog post *discussing* GPT-4).
- Pattern list must be manually maintained as new models are released.

---

### `SynthIDSignal` (`whodidthis/signals/synthid.py`)

**What it detects**: Google DeepMind's SynthID watermark in text.

**Strategy**: Uses `transformers >= 4.46.0` `SynthIDTextWatermarkDetector` with a
community pre-trained Bayesian detector. Returns a score in `[0, 1]` mapped to confidence.

**Critical limitation**: SynthID is NOT a general AI-text detector. It only detects
watermarks it embedded during generation. Without Google's private production keys,
this signal has inherently limited confidence. It is included as a starting point and
will improve as better detectors or operator keys become available.

**Models it can (tentatively) detect**: Google Gemini (production outputs may be watermarked).

**Models it cannot detect**: Claude, GPT-4, LLaMA, Mistral, or any model that doesn't
embed SynthID watermarks. For these, `SynthIDSignal` will correctly return `None`.

**Optional dependency**: Requires `pip install "whodidthis[synthid]"`. If not installed,
`analyze()` raises `ImportError` with an actionable message. The default `ContentClassifier`
silently omits `SynthIDSignal` if the dependency is absent.

---

## How to Add a New Signal

Follow these steps to add a new detection strategy:

### 1. Create the signal file

```
whodidthis/signals/my_signal.py
```

```python
from __future__ import annotations
from whodidthis.result import SignalResult
from whodidthis.signals.base import Signal

class MySignal(Signal):
    name = "my_signal"
    priority = 30  # choose appropriate priority

    def analyze(self, content: str) -> SignalResult | None:
        # ... your detection logic ...
        if found:
            return SignalResult(
                signal_name=self.name,
                model_attribution="Some Model",
                confidence=0.7,
                evidence={"source": "my_method", "detail": "..."},
            )
        return None
```

Return `None` when the signal has no opinion. Return a `SignalResult` with
a low confidence if the signal saw something but is uncertain.

### 2. Register in the default classifier (optional)

If this signal should run by default (for all users), add it to `_default_signals()`
in `whodidthis/classifier.py`. If it has optional deps, guard the import:

```python
def _default_signals() -> list[Signal]:
    signals: list[Signal] = [MetadataSignal()]
    try:
        from whodidthis.signals.my_signal import MySignal
        signals.append(MySignal())
    except ImportError:
        pass
    return signals
```

If it should only be used when explicitly requested, skip this step and let users
pass `signals=[MySignal()]` to `ContentClassifier`.

### 3. Handle optional dependencies

If your signal requires heavy dependencies (ML libraries, etc.), guard the import
inside `analyze()` or a `_load()` method, not at module level:

```python
def _load(self) -> None:
    try:
        import some_heavy_lib
    except ImportError as e:
        raise ImportError(
            "MySignal requires: pip install \"whodidthis[my_extra]\""
        ) from e
```

Add the dependency to `pyproject.toml` under `[project.optional-dependencies]`.

### 4. Add unit tests

Create `tests/unit/signals/test_my_signal.py`. Test:
- Signal metadata (`name`, `priority`)
- Detection when evidence is present (various confidence levels)
- Returns `None` when no evidence
- Edge cases (empty content, short content, malformed input)
- `ImportError` path if deps are optional

### 5. Add a system test fixture

If possible, add a realistic sample content file to `tests/system/fixtures/` and
add test cases to `tests/system/test_end_to_end.py`.

### 6. Update documentation

- Add the signal to the **Signal Inventory** table above.
- Add a new subsection in **Signal Inventory** documenting what it detects,
  its strategy, limitations, and optional deps.
- Update `README.md`'s signal overview table.

### 7. Update `pyproject.toml` if you added an optional dep

```toml
[project.optional-dependencies]
my_extra = ["some-heavy-lib>=1.0"]
```

---

## Optional Dependency Pattern

The project core (`pip install whodidthis`) has **zero required dependencies**.
Heavy dependencies (ML frameworks, etc.) are optional extras.

The pattern for signals with optional deps:

1. The signal class can always be **instantiated** without the dep.
2. The dep is imported lazily inside `analyze()` (or a `_load()` helper).
3. If the dep is missing, `analyze()` raises `ImportError` with a clear install hint.
4. `_default_signals()` in `classifier.py` catches `ImportError` and silently omits the signal.
5. The dep is declared in `pyproject.toml` under `[project.optional-dependencies]`.

This means:
- Bare install works: `MetadataSignal` always runs.
- Heavy signals are available via extras: `pip install "whodidthis[synthid]"`.
- Users can always explicitly pass any signal to `ContentClassifier(signals=[...])`.

---

## Testing Strategy

| Layer | Location | Speed | Mocks? | When to run |
|---|---|---|---|---|
| Unit tests | `tests/unit/` | Fast | Yes (heavy deps mocked) | Always — `pytest tests/unit/` |
| System tests | `tests/system/` | Moderate | No (real pipeline, real fixtures) | CI + local — `pytest tests/system/` |
| All tests | `tests/` | Moderate | Mixed | Full check — `pytest tests/` |

**Unit tests** are isolated: signals are tested with mocked detectors; the classifier
is tested with fake signals. No network I/O, no model loading.

**System tests** run the full pipeline end-to-end with realistic fixture content.
`MetadataSignal`-based tests require no optional deps. `SynthIDSignal` tests skip
automatically if `transformers` is not installed.

---

## Design Decisions & Rationale

**Why signals return `None` instead of low-confidence results for "no opinion"?**
A `None` return means "I looked and found nothing." A low-confidence `SignalResult`
means "I found weak evidence." This distinction matters for aggregation: a signal that
found nothing should not influence `is_ai_generated` or `model_attribution`.

**Why is priority an integer rather than an enum?**
Integers allow third-party signals to insert themselves at arbitrary positions without
modifying the core. An enum would require updating the enum for every new signal.

**Why is MetadataSignal's free-text match confidence capped at 0.60?**
A blog post discussing GPT-4 would match the same pattern as GPT-4 output. 0.60 is
above the `is_ai_generated` threshold (0.5) but well below the short-circuit threshold
(0.9), allowing other signals to provide stronger evidence if available.

**Why is SynthIDSignal's max confidence 0.75 (not higher)?**
Without Google's private production watermarking keys, the community detector cannot
be fully trusted. 0.75 reflects meaningful evidence while honestly communicating the
limitation. This can be raised if a higher-quality detector becomes available.
