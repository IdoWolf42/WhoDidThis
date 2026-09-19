"""
System (end-to-end) tests for WhoDidThis.

These tests run the full ContentClassifier pipeline — no mocks — using realistic
fixture content. They verify the system behaves correctly as a whole unit.

Fixture files live in tests/system/fixtures/ and represent real-world content shapes:
  - API response JSON from Gemini, OpenAI, Anthropic
  - Human-written text (no AI signals expected)
  - Unknown AI text (no metadata, signals should be low/absent)
  - Mixed content mentioning a model by name (free-text signal)

Running:
    pytest tests/system/              # all system tests
    pytest tests/system/ -m system   # only marked system tests
    pytest tests/                     # all tests (unit + system)

See tests/system/README.md for more details.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whodidthis import ClassificationResult, ContentClassifier
from whodidthis.signals.metadata import MetadataSignal

# ---------------------------------------------------------------------------
# Fixtures directory
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(filename: str) -> str:
    return (FIXTURES_DIR / filename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Classifier used in all system tests (metadata only — no heavy deps required)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def classifier() -> ContentClassifier:
    """
    ContentClassifier with only MetadataSignal for system tests.

    SynthIDSignal is excluded so system tests run without optional deps.
    SynthID-specific system tests are in a separate class and skip if unavailable.
    """
    return ContentClassifier(signals=[MetadataSignal()])


# ---------------------------------------------------------------------------
# Gemini API response
# ---------------------------------------------------------------------------

@pytest.mark.system
class TestGeminiApiResponse:
    def test_attributes_to_gemini(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("gemini_api_response.json")
        result = classifier.classify(content)
        assert result.model_attribution == "Google Gemini"

    def test_high_confidence(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("gemini_api_response.json")
        result = classifier.classify(content)
        assert result.confidence >= 0.9

    def test_is_ai_generated_true(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("gemini_api_response.json")
        result = classifier.classify(content)
        assert result.is_ai_generated is True

    def test_metadata_signal_fired(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("gemini_api_response.json")
        result = classifier.classify(content)
        signal_names = [s.signal_name for s in result.signals]
        assert "metadata" in signal_names

    def test_evidence_shows_model_field(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("gemini_api_response.json")
        result = classifier.classify(content)
        assert result.best_signal is not None
        assert result.best_signal.evidence.get("key") == "model"


# ---------------------------------------------------------------------------
# OpenAI API response
# ---------------------------------------------------------------------------

@pytest.mark.system
class TestOpenAIApiResponse:
    def test_attributes_to_openai_gpt4o(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("openai_api_response.json")
        result = classifier.classify(content)
        assert result.model_attribution == "OpenAI GPT-4o"

    def test_high_confidence(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("openai_api_response.json")
        result = classifier.classify(content)
        assert result.confidence >= 0.9

    def test_is_ai_generated_true(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("openai_api_response.json")
        result = classifier.classify(content)
        assert result.is_ai_generated is True

    def test_metadata_signal_fired(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("openai_api_response.json")
        result = classifier.classify(content)
        signal_names = [s.signal_name for s in result.signals]
        assert "metadata" in signal_names


# ---------------------------------------------------------------------------
# Anthropic API response
# ---------------------------------------------------------------------------

@pytest.mark.system
class TestAnthropicApiResponse:
    def test_attributes_to_claude(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("anthropic_api_response.json")
        result = classifier.classify(content)
        assert result.model_attribution == "Anthropic Claude 3"

    def test_high_confidence(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("anthropic_api_response.json")
        result = classifier.classify(content)
        assert result.confidence >= 0.9

    def test_is_ai_generated_true(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("anthropic_api_response.json")
        result = classifier.classify(content)
        assert result.is_ai_generated is True


# ---------------------------------------------------------------------------
# Human-written text — no AI signals expected
# ---------------------------------------------------------------------------

@pytest.mark.system
class TestHumanWrittenText:
    def test_no_attribution(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("human_written.txt")
        result = classifier.classify(content)
        assert result.model_attribution is None

    def test_is_ai_generated_none_or_false(self, classifier: ContentClassifier) -> None:
        """Human text has no signals, so is_ai_generated should be None (uncertain)."""
        content = _load_fixture("human_written.txt")
        result = classifier.classify(content)
        assert result.is_ai_generated is None

    def test_no_signals_fired(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("human_written.txt")
        result = classifier.classify(content)
        assert result.signals == []

    def test_confidence_zero(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("human_written.txt")
        result = classifier.classify(content)
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Unknown AI text — no metadata, signals should be absent
# ---------------------------------------------------------------------------

@pytest.mark.system
class TestUnknownAiText:
    def test_no_attribution(self, classifier: ContentClassifier) -> None:
        """Plain AI-sounding text without model mentions should yield no attribution."""
        content = _load_fixture("unknown_ai_text.txt")
        result = classifier.classify(content)
        assert result.model_attribution is None

    def test_result_is_valid(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("unknown_ai_text.txt")
        result = classifier.classify(content)
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Mixed content — free-text model mention
# ---------------------------------------------------------------------------

@pytest.mark.system
class TestMixedContentWithModelMention:
    def test_detects_gpt4_from_free_text(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("mixed_content_with_model_mention.txt")
        result = classifier.classify(content)
        assert result.model_attribution == "OpenAI GPT-4"

    def test_medium_confidence_for_free_text(self, classifier: ContentClassifier) -> None:
        """Free-text matches should have lower confidence than structured metadata."""
        content = _load_fixture("mixed_content_with_model_mention.txt")
        result = classifier.classify(content)
        # Free-text confidence is 0.60 — above the 0.5 AI threshold but clearly lower
        # than metadata confidence (0.95).
        assert 0.5 <= result.confidence < 0.9

    def test_is_ai_generated_true(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("mixed_content_with_model_mention.txt")
        result = classifier.classify(content)
        assert result.is_ai_generated is True

    def test_evidence_source_is_free_text(self, classifier: ContentClassifier) -> None:
        content = _load_fixture("mixed_content_with_model_mention.txt")
        result = classifier.classify(content)
        assert result.best_signal is not None
        assert result.best_signal.evidence.get("source") == "free_text"


# ---------------------------------------------------------------------------
# Signal ordering guarantees
# ---------------------------------------------------------------------------

@pytest.mark.system
class TestSignalOrdering:
    def test_signals_always_sorted_by_confidence_descending(
        self, classifier: ContentClassifier
    ) -> None:
        """Verify the invariant holds regardless of content type."""
        fixtures = [
            "gemini_api_response.json",
            "openai_api_response.json",
            "anthropic_api_response.json",
            "human_written.txt",
            "unknown_ai_text.txt",
            "mixed_content_with_model_mention.txt",
        ]
        for fixture_name in fixtures:
            content = _load_fixture(fixture_name)
            result = classifier.classify(content)
            confidences = [s.confidence for s in result.signals]
            assert confidences == sorted(confidences, reverse=True), (
                f"Signals not sorted for fixture: {fixture_name}"
            )


# ---------------------------------------------------------------------------
# SynthID system tests (skipped if deps not installed)
# ---------------------------------------------------------------------------

@pytest.mark.system
class TestSynthIDOptionalDependency:
    def test_synthid_raises_import_error_without_deps(self) -> None:
        """
        When transformers is not installed, instantiating SynthIDSignal is fine,
        but calling analyze() raises ImportError with an actionable message.
        """
        try:
            import transformers  # noqa: F401
            pytest.skip("transformers is installed; this test requires it to be absent")
        except ImportError:
            pass

        from whodidthis.signals.synthid import SynthIDSignal

        signal = SynthIDSignal()
        with pytest.raises(ImportError) as exc_info:
            signal.analyze("This is a long enough string to pass the length guard " * 3)
        assert "pip install" in str(exc_info.value)
        assert "synthid" in str(exc_info.value)

    def test_synthid_included_in_default_classifier_when_deps_present(self) -> None:
        """
        When transformers IS installed, the default ContentClassifier should
        include SynthIDSignal automatically.
        """
        try:
            import transformers  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed")

        default_classifier = ContentClassifier()
        signal_names = [s.name for s in default_classifier.signals]
        assert "synthid" in signal_names

    def test_synthid_not_in_default_classifier_when_deps_absent(self) -> None:
        """
        When transformers is NOT installed, the default ContentClassifier should
        silently omit SynthIDSignal (no error).
        """
        try:
            import transformers  # noqa: F401
            pytest.skip("transformers is installed; this test requires it to be absent")
        except ImportError:
            pass

        default_classifier = ContentClassifier()
        signal_names = [s.name for s in default_classifier.signals]
        assert "synthid" not in signal_names
        assert "metadata" in signal_names  # metadata is always present
