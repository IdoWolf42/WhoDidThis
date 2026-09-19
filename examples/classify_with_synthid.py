"""
Example: classify_with_synthid.py

Demonstrates ContentClassifier using the SynthIDSignal to detect Google's
SynthID watermark in text content.

Requires the [synthid] optional dependencies:
    pip install "whodidthis[synthid]"

Which installs: transformers>=4.46.0, torch

IMPORTANT: SynthID can only detect watermarks it applied itself.
This example uses a community pre-trained detector — confidence will be
moderate without Google's private production watermarking keys.
See whodidthis/signals/synthid.py for full limitations.

Run:
    pip install "whodidthis[synthid]"
    python examples/classify_with_synthid.py
"""

from __future__ import annotations

import sys

from whodidthis import ClassificationResult, ContentClassifier


# ---------------------------------------------------------------------------
# Check dependencies before doing anything
# ---------------------------------------------------------------------------

def _check_deps() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        print(
            "\nSynthIDSignal requires optional dependencies.\n"
            "Install them with:\n\n"
            '    pip install "whodidthis[synthid]"\n\n'
            "This installs: transformers>=4.46.0, torch\n"
        )
        return False


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

# In a real use case, you would obtain text from a Gemini API response or
# another source that uses SynthID watermarking. Here we use generic samples
# to illustrate the API surface. Without the operator's private keys,
# the detector will return low/no confidence for arbitrary text.

SAMPLES: list[tuple[str, str]] = [
    (
        "Generic AI-sounding text (no known watermark)",
        (
            "Large language models are neural networks trained on vast amounts of text data. "
            "They learn to predict the next token in a sequence, which allows them to generate "
            "coherent and contextually relevant text. Modern LLMs are used for a wide range of "
            "tasks including question answering, summarization, translation, code generation, "
            "and creative writing. Their capabilities have grown significantly with scale, "
            "following empirical scaling laws that relate model performance to parameters, "
            "dataset size, and compute budget."
        ),
    ),
    (
        "Short text (below minimum length — skipped by SynthIDSignal)",
        "This is too short.",
    ),
    (
        "Technical paragraph (possible Gemini output, watermark uncertain)",
        (
            "Transformer architectures rely on the self-attention mechanism to model "
            "relationships between tokens in a sequence. Unlike recurrent networks, "
            "transformers process all tokens in parallel, enabling more efficient training "
            "on modern hardware. The attention mechanism computes a weighted sum of value "
            "vectors, where weights are determined by the similarity between query and key "
            "vectors. This allows the model to dynamically focus on the most relevant parts "
            "of the input when generating each output token."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _print_result(label: str, result: ClassificationResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Sample: {label}")
    print(f"{'=' * 60}")
    print(f"  Model attribution : {result.model_attribution or '(unknown)'}")
    print(f"  Confidence        : {result.confidence:.2f}")
    print(f"  Is AI-generated   : {result.is_ai_generated}")
    print(f"  Signals fired     : {len(result.signals)}")

    for i, signal in enumerate(result.signals, 1):
        print(f"\n  Signal #{i}: {signal.signal_name}")
        print(f"    Attribution : {signal.model_attribution or '(none)'}")
        print(f"    Confidence  : {signal.confidence:.2f}")
        print(f"    Evidence    :")
        for key, value in signal.evidence.items():
            value_str = str(value)
            if len(value_str) > 70:
                value_str = value_str[:67] + "..."
            print(f"      {key}: {value_str}")

    if not result.signals:
        print("\n  (No signals fired — content origin unknown or below threshold)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _check_deps():
        sys.exit(1)

    # Import here so missing deps give a clear error above, not a traceback.
    from whodidthis.signals.synthid import SynthIDSignal

    print("\nWhoDidThis — SynthID Classification Example")
    print("Classifying content using SynthID watermark detection.\n")
    print(
        "Note: This uses a community pre-trained detector.\n"
        "Confidence is limited without Google's private watermarking keys.\n"
        "Expect low/no confidence for arbitrary text.\n"
    )

    # Use only SynthIDSignal to isolate its output.
    classifier = ContentClassifier(signals=[SynthIDSignal()])

    print("Loading SynthID detector model (first run may download from Hugging Face Hub)...")

    for label, content in SAMPLES:
        result = classifier.classify(content)
        _print_result(label, result)

    print(f"\n{'=' * 60}")
    print("Done.")
    print(
        "\nTo see SynthID alongside metadata detection, use the default classifier:\n"
        "    classifier = ContentClassifier()  # uses both MetadataSignal + SynthIDSignal"
    )


if __name__ == "__main__":
    main()
