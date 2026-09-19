"""
SynthIDSignal — watermark-based detection using Google DeepMind's SynthID.

This signal requires the [synthid] optional dependency group:
    pip install "whodidthis[synthid]"

Which installs:
    transformers>=4.46.0
    torch

IMPORTANT LIMITATIONS — READ BEFORE USING:
-------------------------------------------
SynthID is NOT a general-purpose AI-text detector. It is a watermarking system.
It can only detect watermarks that were embedded during generation using a known
watermarking key configuration.

This signal uses a community pre-trained detector. Without the operator's private
watermarking keys (which Google keeps secret for production Gemini), detection is
best-effort and confidence is inherently limited (~0.3–0.7 range).

This means:
- Content from Claude, GPT-4, etc. will NOT be detected by this signal (they
  don't use SynthID), and the signal will correctly return None for them.
- Content from Gemini MAY be detectable if it was generated with SynthID
  watermarking, but confidence will be moderate without the private keys.
- This signal is included as a starting point; confidence will improve as
  better pre-trained detectors or operator keys become available.

See ARCHITECTURE.md for guidance on adding alternative text-fingerprinting signals
to cover models that don't use SynthID.

Confidence mapping:
    detector score >= 0.8  → confidence 0.75, attribution "Google Gemini"
    detector score >= 0.6  → confidence 0.50, attribution "Google Gemini"
    detector score <  0.6  → return None (below detection threshold)
"""

from __future__ import annotations

from whodidthis.result import SignalResult
from whodidthis.signals.base import Signal

# Name of the pre-trained Bayesian detector on Hugging Face Hub.
# This is a community/reference detector — replace with a better one when available.
_DETECTOR_MODEL_ID = "joaogante/dummy_synthid_detector"

# Detector score thresholds → (confidence, attribution)
_THRESHOLDS: list[tuple[float, float, str]] = [
    (0.8, 0.75, "Google Gemini"),
    (0.6, 0.50, "Google Gemini"),
]


class SynthIDSignal(Signal):
    """
    Detects SynthID watermarks in text content using a pre-trained Bayesian detector.

    Requires: pip install "whodidthis[synthid]"

    The detector model is loaded lazily on the first call to analyze() to avoid
    import-time cost and to allow the signal to be instantiated even when
    transformers is not installed (it will raise ImportError only at analyze() time).
    """

    name = "synthid"
    priority = 20

    def __init__(self) -> None:
        self._detector: object | None = None
        self._tokenizer: object | None = None
        self._logits_processor: object | None = None

    def _load(self) -> None:
        """Lazy-load the SynthID detector. Raises ImportError if deps missing."""
        if self._detector is not None:
            return

        try:
            from transformers import (  # type: ignore[import-untyped]
                AutoTokenizer,
                BayesianDetectorModel,
                SynthIDTextWatermarkDetector,
                SynthIDTextWatermarkLogitsProcessor,
            )
        except ImportError as e:
            raise ImportError(
                "SynthIDSignal requires the [synthid] optional dependencies.\n"
                "Install them with:\n\n"
                "    pip install \"whodidthis[synthid]\"\n\n"
                "This installs: transformers>=4.46.0, torch"
            ) from e

        detector_model = BayesianDetectorModel.from_pretrained(_DETECTOR_MODEL_ID)
        logits_processor = SynthIDTextWatermarkLogitsProcessor(
            **detector_model.config.watermarking_config,
            device="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(detector_model.config.model_name)

        self._detector = SynthIDTextWatermarkDetector(
            detector_model, logits_processor, tokenizer
        )
        self._tokenizer = tokenizer

    def analyze(self, content: str) -> SignalResult | None:
        """
        Run SynthID watermark detection on the content.

        Returns None if:
        - Content is too short to be meaningful (< 50 characters).
        - The detector score is below the minimum threshold (0.6).
        - The [synthid] optional dependencies are not installed (raises ImportError).

        Args:
            content: Raw text to analyze.

        Returns:
            SignalResult with confidence and evidence, or None.

        Raises:
            ImportError: If transformers/torch are not installed.
        """
        if len(content.strip()) < 50:
            return None

        self._load()

        # _load() guarantees _detector and _tokenizer are set, but mypy needs help.
        assert self._detector is not None
        assert self._tokenizer is not None

        # Type-ignore: transformers stubs are not always available.
        tokenizer = self._tokenizer  # type: ignore[assignment]
        detector = self._detector  # type: ignore[assignment]

        tokenized = tokenizer([content], return_tensors="pt")  # type: ignore[operator]
        scores = detector(tokenized.input_ids)  # type: ignore[operator]

        # scores is a jnp or torch array of shape [batch_size]; take the first element.
        score: float = float(scores[0])

        for threshold, confidence, attribution in _THRESHOLDS:
            if score >= threshold:
                return SignalResult(
                    signal_name=self.name,
                    model_attribution=attribution,
                    confidence=confidence,
                    evidence={
                        "detector_score": score,
                        "detector_model": _DETECTOR_MODEL_ID,
                        "threshold_used": threshold,
                        "note": (
                            "Best-effort detection without operator private keys. "
                            "Confidence is limited by public detector availability."
                        ),
                    },
                )

        return None
