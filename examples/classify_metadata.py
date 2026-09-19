"""
Example: classify_metadata.py

Demonstrates ContentClassifier using the MetadataSignal to identify content
from various AI providers via structured JSON metadata.

No optional dependencies required — runs with a bare `pip install -e .`.

Run:
    python examples/classify_metadata.py
"""

from __future__ import annotations

import json
import textwrap

from whodidthis import ClassificationResult, ContentClassifier
from whodidthis.signals.metadata import MetadataSignal


# ---------------------------------------------------------------------------
# Sample content — realistic API response shapes
# ---------------------------------------------------------------------------

SAMPLES: list[tuple[str, str]] = [
    (
        "Gemini API response",
        json.dumps({
            "model": "gemini-1.5-pro",
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "The water cycle describes the continuous movement of water within Earth."}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 20},
        }, indent=2),
    ),
    (
        "OpenAI Chat Completions response",
        json.dumps({
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Photosynthesis converts light energy into glucose."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
        }, indent=2),
    ),
    (
        "Anthropic Messages API response",
        json.dumps({
            "id": "msg_01abc",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Quantum entanglement is a fascinating phenomenon."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 9},
        }, indent=2),
    ),
    (
        "Human-written text (no AI metadata)",
        "I've been thinking about the way light falls through my kitchen window in the morning. "
        "There's something about October light that feels different from any other month.",
    ),
    (
        "Text mentioning GPT-4 by name (free-text signal)",
        "Below is a summary I generated using GPT-4 for a presentation on renewable energy. "
        "I edited the tone a bit but kept the structure from the original output.",
    ),
]


# ---------------------------------------------------------------------------
# Pretty-printing helpers
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
            if len(value_str) > 60:
                value_str = value_str[:57] + "..."
            print(f"      {key}: {value_str}")

    if not result.signals:
        print("\n  (No signals fired — content origin unknown)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Use only MetadataSignal for this example (no heavy deps needed)
    classifier = ContentClassifier(signals=[MetadataSignal()])

    print("\nWhoDidThis — Metadata Classification Example")
    print("Classifying content using structured metadata and model-name patterns.\n")

    for label, content in SAMPLES:
        result = classifier.classify(content)
        _print_result(label, result)

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
