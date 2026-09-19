"""
Abstract base class for all detection signals.

To add a new signal:
1. Create a new file in whodidthis/signals/ (e.g. my_signal.py).
2. Subclass Signal and implement analyze().
3. Set a unique name and an appropriate priority (lower = runs first).
4. Register the signal in ContentClassifier's default_signals() in classifier.py.
5. Add unit tests in tests/unit/signals/test_my_signal.py.
6. Add a system test fixture in tests/system/fixtures/ if possible.
7. Document the signal in ARCHITECTURE.md.

See ARCHITECTURE.md for the full signal authoring guide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from whodidthis.result import SignalResult


class Signal(ABC):
    """
    Abstract base class for all detection signals.

    Each signal encapsulates a single detection strategy (e.g. metadata inspection,
    SynthID watermark detection). Signals are stateless by convention — all
    inputs come through analyze() and all outputs are returned as SignalResult.

    Attributes:
        name:     Unique, stable string identifier for this signal.
                  Used in SignalResult.signal_name. Use snake_case.
                  E.g. "metadata", "synthid".
        priority: Integer priority. Lower values run first.
                  Convention:
                    0–9   : instant / zero-cost checks (e.g. in-memory pattern match)
                    10–19 : cheap I/O-free checks (e.g. metadata parsing)
                    20–49 : moderate-cost checks (e.g. local model inference)
                    50+   : expensive checks (e.g. remote API calls, large model inference)
    """

    name: str
    priority: int

    @abstractmethod
    def analyze(self, content: str) -> SignalResult | None:
        """
        Analyze content and return a SignalResult, or None.

        Return None if this signal cannot make any determination about the
        given content (e.g. no relevant metadata present, dependency not
        installed, content too short).

        Returning None is the correct behavior for "I don't know" — it is
        different from returning a low-confidence SignalResult, which means
        "I looked and found weak evidence."

        Args:
            content: The raw text content to analyze.

        Returns:
            A SignalResult with the signal's findings, or None.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, priority={self.priority})"
