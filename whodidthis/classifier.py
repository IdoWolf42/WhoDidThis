"""
ContentClassifier — the main orchestrator for WhoDidThis.

Usage:
    from whodidthis import ContentClassifier

    classifier = ContentClassifier()
    result = classifier.classify("Some text to analyze...")

    print(result.model_attribution)   # e.g. "Google Gemini"
    print(result.confidence)          # e.g. 0.95
    print(result.is_ai_generated)     # True / False / None
    for signal in result.signals:
        print(signal.signal_name, signal.confidence, signal.evidence)

Customizing signals:
    from whodidthis.signals.metadata import MetadataSignal
    from whodidthis.signals.synthid import SynthIDSignal

    # Use only specific signals
    classifier = ContentClassifier(signals=[MetadataSignal()])

    # Or add your own signal to the default set
    classifier = ContentClassifier(signals=[MetadataSignal(), MyCustomSignal()])
"""

from __future__ import annotations

from whodidthis.result import ClassificationResult, SignalResult
from whodidthis.signals.base import Signal
from whodidthis.signals.metadata import MetadataSignal

# Confidence threshold above which content is considered AI-generated.
_AI_CONFIDENCE_THRESHOLD = 0.5

# Confidence threshold above which we short-circuit (skip remaining signals).
_HIGH_CONFIDENCE_THRESHOLD = 0.9


def _synthid_deps_available() -> bool:
    """Return True if the [synthid] optional dependencies are importable."""
    try:
        import transformers  # type: ignore[import-untyped]  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _default_signals() -> list[Signal]:
    """
    Build the default signal list.

    SynthIDSignal is included only if its optional dependencies (transformers, torch)
    are available. This means a bare `pip install whodidthis` (no extras) will still
    work — it just won't run the SynthID signal.
    """
    signals: list[Signal] = [MetadataSignal()]

    if _synthid_deps_available():
        from whodidthis.signals.synthid import SynthIDSignal  # noqa: PLC0415

        signals.append(SynthIDSignal())

    return signals


class ContentClassifier:
    """
    Orchestrates detection signals to classify the origin model of content.

    The classifier runs signals in priority order (lowest priority value first),
    collects all results, and aggregates them into a single ClassificationResult.

    Args:
        signals:                  List of Signal instances to run. If None, uses
                                  the default set (MetadataSignal + SynthIDSignal
                                  if [synthid] extras are installed).
        stop_on_high_confidence:  If True (default), stop running signals once any
                                  signal returns confidence >= 0.9. This avoids
                                  running expensive signals unnecessarily.
        ai_confidence_threshold:  Minimum confidence to consider content AI-generated.
                                  Default 0.5.
    """

    def __init__(
        self,
        signals: list[Signal] | None = None,
        stop_on_high_confidence: bool = True,
        ai_confidence_threshold: float = _AI_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._signals: list[Signal] = sorted(
            signals if signals is not None else _default_signals(),
            key=lambda s: s.priority,
        )
        self._stop_on_high_confidence = stop_on_high_confidence
        self._ai_confidence_threshold = ai_confidence_threshold

    @property
    def signals(self) -> list[Signal]:
        """The registered signals, sorted by priority ascending."""
        return list(self._signals)

    def classify(self, content: str) -> ClassificationResult:
        """
        Classify the origin model of the given content.

        Runs all registered signals in priority order, collects results, and
        returns a ClassificationResult summarizing the findings.

        Args:
            content: The raw text content to classify.

        Returns:
            ClassificationResult with model attribution, confidence, and
            the list of all SignalResults that fired.
        """
        fired: list[SignalResult] = []

        for signal in self._signals:
            result = signal.analyze(content)
            if result is not None:
                fired.append(result)
                if (
                    self._stop_on_high_confidence
                    and result.confidence >= _HIGH_CONFIDENCE_THRESHOLD
                ):
                    break

        return self._aggregate(fired)

    def _aggregate(self, fired: list[SignalResult]) -> ClassificationResult:
        """
        Merge a list of SignalResults into a single ClassificationResult.

        Strategy:
        - Sort all fired results by confidence descending.
        - The top result provides model_attribution and confidence.
        - is_ai_generated = True if any result has confidence >= threshold.
        - is_ai_generated = None if no signals fired.
        """
        if not fired:
            return ClassificationResult(
                model_attribution=None,
                confidence=0.0,
                is_ai_generated=None,
                signals=[],
            )

        sorted_results = sorted(fired, key=lambda r: r.confidence, reverse=True)
        best = sorted_results[0]

        is_ai_generated = any(r.confidence >= self._ai_confidence_threshold for r in fired)

        return ClassificationResult(
            model_attribution=best.model_attribution,
            confidence=best.confidence,
            is_ai_generated=is_ai_generated,
            signals=sorted_results,
        )
