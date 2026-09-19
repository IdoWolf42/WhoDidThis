"""
Unit tests for ContentClassifier.

Tests cover:
- Default signal loading (MetadataSignal always present; SynthIDSignal conditionally)
- Signal priority ordering (lower priority runs first)
- Aggregation: best signal wins for attribution and confidence
- stop_on_high_confidence behaviour (short-circuits when confidence >= 0.9)
- is_ai_generated threshold logic
- Empty results (no signal fires)
- Custom signal list injection
- classify() returns correct ClassificationResult shape
- best_signal property
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from whodidthis import ClassificationResult, ContentClassifier, SignalResult
from whodidthis.signals.base import Signal
from whodidthis.signals.metadata import MetadataSignal


# ---------------------------------------------------------------------------
# Helpers — fake signals for testing
# ---------------------------------------------------------------------------


def _make_signal(
    name: str,
    priority: int,
    result: SignalResult | None,
) -> Signal:
    """Create a concrete Signal stub that always returns the given result."""

    class _FakeSignal(Signal):
        def analyze(self, content: str) -> SignalResult | None:
            return result

    sig = _FakeSignal()
    sig.name = name  # type: ignore[misc]
    sig.priority = priority  # type: ignore[misc]
    return sig


def _sr(name: str, confidence: float, attribution: str | None = "Model X") -> SignalResult:
    return SignalResult(
        signal_name=name,
        model_attribution=attribution,
        confidence=confidence,
        evidence={"test": True},
    )


# ---------------------------------------------------------------------------
# Default signals
# ---------------------------------------------------------------------------


class TestDefaultSignals:
    def test_metadata_always_present(self) -> None:
        classifier = ContentClassifier()
        names = [s.name for s in classifier.signals]
        assert "metadata" in names

    def test_signals_sorted_by_priority(self) -> None:
        classifier = ContentClassifier()
        priorities = [s.priority for s in classifier.signals]
        assert priorities == sorted(priorities)

    def test_custom_signal_list_replaces_defaults(self) -> None:
        fake = _make_signal("fake", 5, None)
        classifier = ContentClassifier(signals=[fake])
        assert len(classifier.signals) == 1
        assert classifier.signals[0].name == "fake"


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_signals_sorted_ascending_by_priority(self) -> None:
        s1 = _make_signal("slow", 50, _sr("slow", 0.4))
        s2 = _make_signal("fast", 5, _sr("fast", 0.4))
        classifier = ContentClassifier(signals=[s1, s2])
        assert classifier.signals[0].name == "fast"
        assert classifier.signals[1].name == "slow"

    def test_execution_order_respected(self) -> None:
        """Verify execution order via call tracking."""
        call_order: list[str] = []

        class TrackingSignal(Signal):
            def __init__(self, _name: str, _priority: int) -> None:
                self.name = _name  # type: ignore[misc]
                self.priority = _priority  # type: ignore[misc]

            def analyze(self, content: str) -> SignalResult | None:
                call_order.append(self.name)
                return None

        s_high = TrackingSignal("high_priority", 5)
        s_low = TrackingSignal("low_priority", 99)
        classifier = ContentClassifier(signals=[s_low, s_high])
        classifier.classify("some content")
        assert call_order == ["high_priority", "low_priority"]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_highest_confidence_wins_attribution(self) -> None:
        s1 = _make_signal("a", 10, _sr("a", 0.6, "Model A"))
        s2 = _make_signal("b", 20, _sr("b", 0.9, "Model B"))
        classifier = ContentClassifier(signals=[s1, s2], stop_on_high_confidence=False)
        result = classifier.classify("content")
        assert result.model_attribution == "Model B"
        assert result.confidence == 0.9

    def test_signals_sorted_by_confidence_descending(self) -> None:
        s1 = _make_signal("a", 10, _sr("a", 0.4, "Model A"))
        s2 = _make_signal("b", 20, _sr("b", 0.8, "Model B"))
        classifier = ContentClassifier(signals=[s1, s2], stop_on_high_confidence=False)
        result = classifier.classify("content")
        assert result.signals[0].confidence >= result.signals[1].confidence

    def test_no_signals_fired_returns_empty_result(self) -> None:
        s = _make_signal("none", 10, None)
        classifier = ContentClassifier(signals=[s])
        result = classifier.classify("content")
        assert result.model_attribution is None
        assert result.confidence == 0.0
        assert result.is_ai_generated is None
        assert result.signals == []

    def test_all_signals_included_when_multiple_fire(self) -> None:
        s1 = _make_signal("a", 10, _sr("a", 0.6))
        s2 = _make_signal("b", 20, _sr("b", 0.7))
        classifier = ContentClassifier(signals=[s1, s2], stop_on_high_confidence=False)
        result = classifier.classify("content")
        signal_names = {r.signal_name for r in result.signals}
        assert "a" in signal_names
        assert "b" in signal_names

    def test_only_fired_signals_in_result(self) -> None:
        s1 = _make_signal("fires", 10, _sr("fires", 0.8))
        s2 = _make_signal("silent", 20, None)
        classifier = ContentClassifier(signals=[s1, s2], stop_on_high_confidence=False)
        result = classifier.classify("content")
        assert len(result.signals) == 1
        assert result.signals[0].signal_name == "fires"


# ---------------------------------------------------------------------------
# stop_on_high_confidence
# ---------------------------------------------------------------------------


class TestStopOnHighConfidence:
    def test_stops_after_high_confidence_signal(self) -> None:
        second_called = []

        class SecondSignal(Signal):
            name = "second"  # type: ignore[misc]
            priority = 20  # type: ignore[misc]

            def analyze(self, content: str) -> SignalResult | None:
                second_called.append(True)
                return _sr("second", 0.3)

        s1 = _make_signal("first", 10, _sr("first", 0.95, "Model A"))
        classifier = ContentClassifier(
            signals=[s1, SecondSignal()],
            stop_on_high_confidence=True,
        )
        classifier.classify("content")
        assert second_called == []  # second signal was never called

    def test_continues_when_stop_disabled(self) -> None:
        second_called = []

        class SecondSignal(Signal):
            name = "second"  # type: ignore[misc]
            priority = 20  # type: ignore[misc]

            def analyze(self, content: str) -> SignalResult | None:
                second_called.append(True)
                return _sr("second", 0.3)

        s1 = _make_signal("first", 10, _sr("first", 0.95, "Model A"))
        classifier = ContentClassifier(
            signals=[s1, SecondSignal()],
            stop_on_high_confidence=False,
        )
        classifier.classify("content")
        assert second_called == [True]

    def test_does_not_stop_below_threshold(self) -> None:
        """Confidence of 0.89 should NOT trigger the short-circuit."""
        second_called = []

        class SecondSignal(Signal):
            name = "second"  # type: ignore[misc]
            priority = 20  # type: ignore[misc]

            def analyze(self, content: str) -> SignalResult | None:
                second_called.append(True)
                return None

        s1 = _make_signal("first", 10, _sr("first", 0.89))
        classifier = ContentClassifier(
            signals=[s1, SecondSignal()],
            stop_on_high_confidence=True,
        )
        classifier.classify("content")
        assert second_called == [True]


# ---------------------------------------------------------------------------
# is_ai_generated threshold
# ---------------------------------------------------------------------------


class TestIsAiGenerated:
    def test_ai_generated_true_when_above_threshold(self) -> None:
        s = _make_signal("a", 10, _sr("a", 0.6))
        classifier = ContentClassifier(signals=[s], ai_confidence_threshold=0.5)
        result = classifier.classify("content")
        assert result.is_ai_generated is True

    def test_ai_generated_false_when_all_below_threshold(self) -> None:
        s = _make_signal("a", 10, _sr("a", 0.4))
        classifier = ContentClassifier(signals=[s], ai_confidence_threshold=0.5)
        result = classifier.classify("content")
        assert result.is_ai_generated is False

    def test_ai_generated_none_when_no_signals(self) -> None:
        s = _make_signal("a", 10, None)
        classifier = ContentClassifier(signals=[s])
        result = classifier.classify("content")
        assert result.is_ai_generated is None

    def test_custom_threshold_respected(self) -> None:
        s = _make_signal("a", 10, _sr("a", 0.7))
        classifier = ContentClassifier(signals=[s], ai_confidence_threshold=0.8)
        result = classifier.classify("content")
        assert result.is_ai_generated is False


# ---------------------------------------------------------------------------
# ClassificationResult shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_result_is_classification_result(self) -> None:
        classifier = ContentClassifier(signals=[_make_signal("a", 10, None)])
        result = classifier.classify("content")
        assert isinstance(result, ClassificationResult)

    def test_confidence_in_valid_range(self) -> None:
        s = _make_signal("a", 10, _sr("a", 0.72))
        classifier = ContentClassifier(signals=[s])
        result = classifier.classify("content")
        assert 0.0 <= result.confidence <= 1.0

    def test_best_signal_property_returns_top(self) -> None:
        s1 = _make_signal("a", 10, _sr("a", 0.4))
        s2 = _make_signal("b", 20, _sr("b", 0.8))
        classifier = ContentClassifier(signals=[s1, s2], stop_on_high_confidence=False)
        result = classifier.classify("content")
        assert result.best_signal is not None
        assert result.best_signal.signal_name == "b"

    def test_best_signal_none_when_no_results(self) -> None:
        s = _make_signal("a", 10, None)
        classifier = ContentClassifier(signals=[s])
        result = classifier.classify("content")
        assert result.best_signal is None


# ---------------------------------------------------------------------------
# Integration with real MetadataSignal
# ---------------------------------------------------------------------------


class TestRealMetadataIntegration:
    """Quick integration checks using real MetadataSignal (no mocks)."""

    def test_classifies_openai_json(self) -> None:
        import json

        classifier = ContentClassifier(signals=[MetadataSignal()])
        content = json.dumps({"model": "gpt-4o", "choices": [{"text": "Hello"}]})
        result = classifier.classify(content)
        assert result.model_attribution == "OpenAI GPT-4o"
        assert result.confidence == 0.95
        assert result.is_ai_generated is True

    def test_unknown_content_returns_none_attribution(self) -> None:
        classifier = ContentClassifier(signals=[MetadataSignal()])
        result = classifier.classify("The quick brown fox.")
        assert result.model_attribution is None
        assert result.is_ai_generated is None
