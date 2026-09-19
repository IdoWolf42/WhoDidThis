"""
Unit tests for SynthIDSignal.

Because SynthID requires the [synthid] optional dependencies (transformers, torch),
most tests mock the heavy dependencies so the unit test suite runs without them.

Test categories:
- ImportError raised with a helpful message when deps are missing
- Signal metadata (name, priority)
- Short content is skipped (returns None)
- Detector score below threshold returns None
- Detector score in medium range returns medium confidence
- Detector score in high range returns high confidence
- Evidence fields are correct
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from whodidthis.signals.synthid import SynthIDSignal, _DETECTOR_MODEL_ID, _THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal_with_mock_detector(score: float) -> SynthIDSignal:
    """
    Return a SynthIDSignal with its internal state set to a mock detector
    that returns the given score, bypassing _load().
    """
    signal = SynthIDSignal()

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(input_ids=MagicMock())

    mock_detector = MagicMock()
    mock_detector.return_value = [score]

    signal._tokenizer = mock_tokenizer
    signal._detector = mock_detector

    return signal


# ---------------------------------------------------------------------------
# Signal metadata
# ---------------------------------------------------------------------------


class TestSignalMetadata:
    def test_name(self) -> None:
        signal = SynthIDSignal()
        assert signal.name == "synthid"

    def test_priority(self) -> None:
        signal = SynthIDSignal()
        assert signal.priority == 20

    def test_priority_higher_than_metadata(self) -> None:
        from whodidthis.signals.metadata import MetadataSignal

        assert SynthIDSignal().priority > MetadataSignal().priority

    def test_repr(self) -> None:
        signal = SynthIDSignal()
        r = repr(signal)
        assert "SynthIDSignal" in r
        assert "synthid" in r


# ---------------------------------------------------------------------------
# ImportError when dependencies are missing
# ---------------------------------------------------------------------------


class TestImportError:
    def test_import_error_raised_with_helpful_message(self) -> None:
        """If transformers is not installed, analyze() raises ImportError with install hint."""
        signal = SynthIDSignal()
        # Leave _detector as None so _load() is called, then patch the import.
        with patch.dict(sys.modules, {"transformers": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError) as exc_info:
                signal.analyze("Some content that is long enough to pass the length check.")
        assert "whodidthis[synthid]" in str(exc_info.value)
        assert "pip install" in str(exc_info.value)

    def test_import_error_not_raised_when_detector_already_loaded(self) -> None:
        """If _detector is already loaded, analyze() does not try to re-import."""
        signal = _make_signal_with_mock_detector(score=0.85)
        # Even with transformers blocked, should not raise because _load() is skipped.
        with patch.dict(sys.modules, {"transformers": None}):  # type: ignore[dict-item]
            result = signal.analyze("This is a sufficiently long piece of content for the test.")
        # Result may or may not fire depending on score; just confirm no ImportError.
        # score=0.85 >= 0.8 threshold, so result should be non-None.
        assert result is not None


# ---------------------------------------------------------------------------
# Short content guard
# ---------------------------------------------------------------------------


class TestShortContentGuard:
    def test_empty_string_returns_none(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.99)
        result = signal.analyze("")
        assert result is None

    def test_short_content_returns_none(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.99)
        result = signal.analyze("Too short.")
        assert result is None

    def test_whitespace_only_returns_none(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.99)
        result = signal.analyze("   \n\t  ")
        assert result is None

    def test_exactly_50_chars_is_analyzed(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.85)
        # 50 non-whitespace characters
        content = "a" * 50
        result = signal.analyze(content)
        assert result is not None  # score 0.85 >= 0.8 threshold

    def test_49_chars_returns_none(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.99)
        content = "a" * 49
        result = signal.analyze(content)
        assert result is None


# ---------------------------------------------------------------------------
# Detector score thresholds
# ---------------------------------------------------------------------------


LONG_CONTENT = "a" * 200  # long enough to pass the guard


class TestDetectorScoreThresholds:
    def test_score_below_minimum_returns_none(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.59)
        result = signal.analyze(LONG_CONTENT)
        assert result is None

    def test_score_at_minimum_threshold_returns_none(self) -> None:
        # The threshold list uses >=, so 0.6 should fire medium confidence.
        score = 0.6
        signal = _make_signal_with_mock_detector(score=score)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        # Find the expected confidence for score=0.6.
        expected_confidence = next(
            conf for thresh, conf, _ in _THRESHOLDS if score >= thresh
        )
        assert result.confidence == expected_confidence

    def test_score_medium_range_returns_medium_confidence(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.7)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        assert result.confidence == 0.50

    def test_score_high_range_returns_high_confidence(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.85)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        assert result.confidence == 0.75

    def test_score_exactly_at_high_threshold(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.8)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        assert result.confidence == 0.75

    def test_score_zero_returns_none(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.0)
        result = signal.analyze(LONG_CONTENT)
        assert result is None

    def test_score_one_returns_high_confidence(self) -> None:
        signal = _make_signal_with_mock_detector(score=1.0)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        assert result.confidence == 0.75


# ---------------------------------------------------------------------------
# Attribution and evidence
# ---------------------------------------------------------------------------


class TestAttributionAndEvidence:
    def test_attribution_is_google_gemini(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.85)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        assert result.model_attribution == "Google Gemini"

    def test_signal_name(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.85)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        assert result.signal_name == "synthid"

    def test_evidence_contains_detector_score(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.85)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        assert "detector_score" in result.evidence
        assert abs(float(result.evidence["detector_score"]) - 0.85) < 1e-6  # type: ignore[arg-type]

    def test_evidence_contains_detector_model(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.85)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        assert result.evidence["detector_model"] == _DETECTOR_MODEL_ID

    def test_evidence_contains_note(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.85)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        assert "note" in result.evidence

    def test_confidence_in_valid_range(self) -> None:
        signal = _make_signal_with_mock_detector(score=0.85)
        result = signal.analyze(LONG_CONTENT)
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0
