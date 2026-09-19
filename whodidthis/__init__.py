"""
WhoDidThis — Content model classifier.

Determines which AI model (or a human) produced a piece of content,
using a pipeline of detection signals (metadata inspection, SynthID
watermark detection, and more to come).

Public API:

    from whodidthis import ContentClassifier, ClassificationResult, SignalResult

    classifier = ContentClassifier()
    result = classifier.classify("some text...")

    result.model_attribution   # str | None — e.g. "Google Gemini"
    result.confidence          # float in [0.0, 1.0]
    result.is_ai_generated     # bool | None
    result.signals             # list[SignalResult], sorted by confidence desc
    result.best_signal         # SignalResult | None — the highest-confidence signal

For advanced usage (custom signals, configuration), import directly from submodules:
    from whodidthis.classifier import ContentClassifier
    from whodidthis.signals.base import Signal
    from whodidthis.signals.metadata import MetadataSignal
    from whodidthis.signals.synthid import SynthIDSignal  # requires [synthid] extra
"""

from whodidthis.classifier import ContentClassifier
from whodidthis.result import ClassificationResult, SignalResult

__all__ = [
    "ContentClassifier",
    "ClassificationResult",
    "SignalResult",
]

__version__ = "0.1.0"
