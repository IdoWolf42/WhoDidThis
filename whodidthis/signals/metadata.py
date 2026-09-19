"""
MetadataSignal — fast, zero-dependency signal that inspects structured metadata.

This is the cheapest signal (priority=10) and should always run first.

Detection strategy:
  1. If content is valid JSON, inspect known metadata fields (e.g. "model", "engine").
  2. Regardless, scan the content string for known model-name patterns.

Confidence levels:
  0.95  — model name found in a trusted structured metadata field (JSON key)
  0.60  — model name found as a free-text pattern in the content string
  (no result) — no model-identifying pattern found

To add support for a new model/provider:
  Add an entry to MODEL_PATTERNS below. The key is a regex pattern (case-insensitive)
  matched against the field value or content string. The value is the canonical
  human-readable model name returned in SignalResult.model_attribution.

  Example:
      r"llama[-\\s]?3": "Meta LLaMA 3",
"""

from __future__ import annotations

import json
import re

from whodidthis.result import SignalResult
from whodidthis.signals.base import Signal

# ---------------------------------------------------------------------------
# Model pattern registry
# ---------------------------------------------------------------------------
# Maps a case-insensitive regex pattern to a canonical model name.
# Patterns are tried in order; the first match wins.
# Add new models here — no other code changes required.
# ---------------------------------------------------------------------------
MODEL_PATTERNS: list[tuple[str, str]] = [
    # OpenAI
    (r"gpt-?4o", "OpenAI GPT-4o"),
    (r"gpt-?4", "OpenAI GPT-4"),
    (r"gpt-?3\.5", "OpenAI GPT-3.5"),
    (r"o1-?(preview|mini)?", "OpenAI o1"),
    (r"text-davinci", "OpenAI GPT-3"),
    # Anthropic
    (r"claude-?3[-\s]?(opus|sonnet|haiku)", "Anthropic Claude 3"),
    (r"claude-?3", "Anthropic Claude 3"),
    (r"claude-?2", "Anthropic Claude 2"),
    (r"claude", "Anthropic Claude"),
    # Google
    (r"gemini[-\s]?(ultra|pro|flash|nano)?[-\s]?\d*", "Google Gemini"),
    (r"gemini", "Google Gemini"),
    (r"palm[-\s]?2?", "Google PaLM"),
    (r"bard", "Google Bard"),
    # Meta
    (r"llama[-\s]?3", "Meta LLaMA 3"),
    (r"llama[-\s]?2", "Meta LLaMA 2"),
    (r"llama", "Meta LLaMA"),
    # Mistral
    (r"mistral[-\s]?(large|medium|small|7b|8x7b)?", "Mistral AI"),
    (r"mixtral", "Mistral AI Mixtral"),
    # Cohere
    (r"command[-\s]?(r\+?|light)?", "Cohere Command"),
    # Perplexity
    (r"pplx[-\s]?\d+b", "Perplexity"),
    # Falcon
    (r"falcon[-\s]?\d+b", "TII Falcon"),
]

# JSON keys that are trusted sources of model identity (checked first, higher confidence).
TRUSTED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "model_id",
        "model_name",
        "model_version",
        "engine",
        "generator",
        "source_model",
        "created_by",
    }
)

# Confidence assigned when a match is found in a trusted metadata key.
CONFIDENCE_METADATA_FIELD = 0.95
# Confidence assigned when a match is found via free-text scan.
CONFIDENCE_FREE_TEXT = 0.60


def _match_model(text: str) -> str | None:
    """Return canonical model name if text matches any known pattern, else None."""
    for pattern, canonical_name in MODEL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return canonical_name
    return None


class MetadataSignal(Signal):
    """
    Detects model attribution from structured metadata and known model-name patterns.

    No external dependencies. Always safe to instantiate and run.
    Priority 10 — runs before heavier signals.
    """

    name = "metadata"
    priority = 10

    def analyze(self, content: str) -> SignalResult | None:
        """
        Inspect content for model-identifying metadata.

        Checks (in order):
        1. Parse content as JSON; scan trusted metadata keys for model names.
        2. Scan the full content string for known model-name patterns.

        Returns None if no model-identifying information is found.
        """
        # --- Step 1: structured JSON metadata ---
        json_result = self._analyze_json(content)
        if json_result is not None:
            return json_result

        # --- Step 2: free-text pattern scan ---
        return self._analyze_free_text(content)

    def _analyze_json(self, content: str) -> SignalResult | None:
        """Try to parse content as JSON and inspect trusted metadata fields."""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict):
            return None

        # Recursively collect all key-value pairs from top-level and one level deep.
        candidates: list[tuple[str, str]] = []
        for key, value in data.items():
            if isinstance(value, str):
                candidates.append((key.lower(), value))
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        candidates.append((sub_key.lower(), sub_value))

        # Check trusted keys first for highest-confidence match.
        for key, value in candidates:
            if key in TRUSTED_METADATA_KEYS:
                model = _match_model(value)
                if model:
                    return SignalResult(
                        signal_name=self.name,
                        model_attribution=model,
                        confidence=CONFIDENCE_METADATA_FIELD,
                        evidence={"source": "json_metadata_field", "key": key, "value": value},
                    )

        # Fall back to checking all string values in the JSON.
        for key, value in candidates:
            model = _match_model(value)
            if model:
                return SignalResult(
                    signal_name=self.name,
                    model_attribution=model,
                    confidence=CONFIDENCE_FREE_TEXT,
                    evidence={"source": "json_value", "key": key, "value": value},
                )

        return None

    def _analyze_free_text(self, content: str) -> SignalResult | None:
        """Scan the raw content string for known model-name patterns."""
        model = _match_model(content)
        if model:
            # Find which pattern triggered for the evidence dict.
            for pattern, canonical_name in MODEL_PATTERNS:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return SignalResult(
                        signal_name=self.name,
                        model_attribution=canonical_name,
                        confidence=CONFIDENCE_FREE_TEXT,
                        evidence={
                            "source": "free_text",
                            "matched_text": match.group(0),
                            "pattern": pattern,
                        },
                    )
        return None
