"""
Unit tests for ClassificationResult and SignalResult dataclasses.
"""

from __future__ import annotations

import pytest

from whodidthis.result import ClassificationResult, SignalResult


class TestSignalResult:
    def test_basic_construction(self) -> None:
        r = SignalResult(
            signal_name="metadata",
            model_attribution="OpenAI GPT-4",
            confidence=0.95,
            evidence={"key": "model"},
        )
        assert r.signal_name == "metadata"
        assert r.model_attribution == "OpenAI GPT-4"
        assert r.confidence == 0.95
        assert r.evidence == {"key": "model"}

    def test_none_attribution_allowed(self) -> None:
        r = SignalResult(
            signal_name="metadata",
            model_attribution=None,
            confidence=0.3,
        )
        assert r.model_attribution is None

    def test_default_evidence_is_empty_dict(self) -> None:
        r = SignalResult(signal_name="x", model_attribution=None, confidence=0.0)
        assert r.evidence == {}

    def test_confidence_zero_allowed(self) -> None:
        r = SignalResult(signal_name="x", model_attribution=None, confidence=0.0)
        assert r.confidence == 0.0

    def test_confidence_one_allowed(self) -> None:
        r = SignalResult(signal_name="x", model_attribution="M", confidence=1.0)
        assert r.confidence == 1.0

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            SignalResult(signal_name="x", model_attribution=None, confidence=-0.1)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            SignalResult(signal_name="x", model_attribution=None, confidence=1.01)


class TestClassificationResult:
    def _make(self, **kwargs: object) -> ClassificationResult:
        defaults: dict[str, object] = {
            "model_attribution": "Google Gemini",
            "confidence": 0.9,
            "is_ai_generated": True,
            "signals": [],
        }
        defaults.update(kwargs)
        return ClassificationResult(**defaults)  # type: ignore[arg-type]

    def test_basic_construction(self) -> None:
        r = self._make()
        assert r.model_attribution == "Google Gemini"
        assert r.confidence == 0.9
        assert r.is_ai_generated is True
        assert r.signals == []

    def test_none_attribution_allowed(self) -> None:
        r = self._make(model_attribution=None)
        assert r.model_attribution is None

    def test_none_is_ai_generated_allowed(self) -> None:
        r = self._make(is_ai_generated=None)
        assert r.is_ai_generated is None

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            self._make(confidence=-0.01)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            self._make(confidence=1.001)

    def test_best_signal_none_when_no_signals(self) -> None:
        r = self._make(signals=[])
        assert r.best_signal is None

    def test_best_signal_returns_first(self) -> None:
        s1 = SignalResult("a", "M", 0.9)
        s2 = SignalResult("b", "N", 0.5)
        r = self._make(signals=[s1, s2])
        assert r.best_signal is s1

    def test_signals_list_preserved(self) -> None:
        s = SignalResult("metadata", "OpenAI GPT-4", 0.95)
        r = self._make(signals=[s])
        assert len(r.signals) == 1
        assert r.signals[0] is s
