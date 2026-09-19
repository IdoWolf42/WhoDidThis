"""
Result data structures for WhoDidThis content classification.

ClassificationResult is the top-level output of ContentClassifier.classify().
SignalResult is produced by each individual signal and collected into ClassificationResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SignalResult:
    """
    The output of a single detection signal.

    Attributes:
        signal_name:       Identifier of the signal that produced this result.
                           E.g. "metadata", "synthid".
        model_attribution: Human-readable name of the attributed model/provider,
                           or None if the signal could not make an attribution.
                           E.g. "Google Gemini", "OpenAI GPT-4".
        confidence:        Float in [0.0, 1.0].
                           0.0 = no evidence at all.
                           0.5 = weak / heuristic evidence.
                           0.9+ = strong / structural evidence (e.g. explicit metadata field).
        evidence:          Raw data that led to this result. Keys vary per signal.
                           Useful for debugging and transparency.
                           E.g. {"field": "model", "value": "gpt-4o", "source": "json_metadata"}
    """

    signal_name: str
    model_attribution: str | None
    confidence: float
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"SignalResult.confidence must be in [0.0, 1.0], got {self.confidence}"
            )


@dataclass
class ClassificationResult:
    """
    The aggregated output of ContentClassifier.classify().

    The classifier runs all registered signals in priority order and merges
    their results into this single object. The signal with the highest
    confidence determines the top-level attribution and confidence.

    Attributes:
        model_attribution: Best guess at the originating model/provider, or None
                           if no signal could make an attribution.
        confidence:        Confidence of the top attribution (from the winning signal).
                           0.0 if no signals fired.
        is_ai_generated:   True if any signal identified AI authorship with confidence
                           above the classifier's threshold (default 0.5).
                           False if a signal explicitly determined human authorship.
                           None if no signal could make a determination.
        signals:           All SignalResults that fired (returned non-None), sorted
                           by confidence descending. Empty list if no signal fired.
    """

    model_attribution: str | None
    confidence: float
    is_ai_generated: bool | None
    signals: list[SignalResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ClassificationResult.confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    @property
    def best_signal(self) -> SignalResult | None:
        """Return the highest-confidence signal, or None if no signals fired."""
        return self.signals[0] if self.signals else None
