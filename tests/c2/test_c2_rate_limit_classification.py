"""A 429 must be classified by its replenishment hint, not by its prose.

C2 finding, observed twice against the live Free Tier on 2026-08-10: Gemini
returns the same "You exceeded your current quota" wording for a short-window
request limit as for daily exhaustion. The only discriminator in the body is the
replenishment hint. Classifying on the wording stopped a pilot that a bounded
backoff completed, and wrote "daily quota exhaustion" into the block artifact for
a limit that cleared in eighteen seconds.

These are the exact bodies the provider returned, kept verbatim as fixtures.
"""
from __future__ import annotations

import pytest

from prism_fas.llm.contracts import ErrorClass
from prism_fas.llm.providers.gemini import RATE_LIMIT_RETRY_CEILING_SECONDS, _classify

#: Verbatim, from reports/c2/C2_PILOT_RAW_ARCHIVE.json (pilot_019).
LIVE_SHORT_WINDOW_429 = (
    "Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check "
    "your plan and billing details. For more information on this error, head to: "
    "https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head "
    "to: https://ai.dev/rate-limit. \\n* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: "
    "gemini-3.6-flash\\nPlease retry in 18.039133522s.', 'code': 'too_many_requests'}}"
)


class RateLimitError(Exception):
    def __init__(self, message: str, status_code: int = 429) -> None:
        self.status_code = status_code
        super().__init__(message)


def test_a_short_window_429_is_a_transient_rate_limit_not_quota_exhaustion():
    error = _classify(RateLimitError(LIVE_SHORT_WINDOW_429))
    assert error.error_class is ErrorClass.RATE_LIMIT
    assert error.retryable is True
    assert error.retry_after_seconds == pytest.approx(18.039133522)


def test_the_second_live_body_classifies_the_same_way():
    body = LIVE_SHORT_WINDOW_429.replace("18.039133522s", "8.405604838s")
    error = _classify(RateLimitError(body))
    assert error.error_class is ErrorClass.RATE_LIMIT
    assert error.retry_after_seconds == pytest.approx(8.405604838)


def test_a_429_with_no_replenishment_hint_still_fails_closed_as_quota():
    body = ("Error code: 429 - You exceeded your current quota, please check your plan and "
            "billing details.")
    error = _classify(RateLimitError(body))
    assert error.error_class is ErrorClass.QUOTA_EXHAUSTED
    assert error.retryable is False


def test_a_per_day_metric_is_quota_exhaustion_whatever_hint_it_carries():
    body = ("Error code: 429 - Quota exceeded for metric: "
            "generate_requests_per_model_per_day, limit: 250. Please retry in 12s.")
    error = _classify(RateLimitError(body))
    assert error.error_class is ErrorClass.QUOTA_EXHAUSTED
    assert error.retryable is False


def test_a_window_longer_than_the_ceiling_is_treated_as_exhaustion():
    body = (f"Error code: 429 - You exceeded your current quota. Please retry in "
            f"{RATE_LIMIT_RETRY_CEILING_SECONDS + 1:.0f}s.")
    error = _classify(RateLimitError(body))
    assert error.error_class is ErrorClass.QUOTA_EXHAUSTED


def test_a_plain_rate_limit_without_quota_wording_remains_retryable():
    error = _classify(RateLimitError("Error code: 429 - too many requests. Please retry in 5s."))
    assert error.error_class is ErrorClass.RATE_LIMIT
    assert error.retry_after_seconds == pytest.approx(5.0)


def test_the_classifier_never_leaks_a_credential_shaped_string():
    body = "Error code: 429 - api_key=AIza0123456789012345678901234567890abcd rate limit"
    error = _classify(RateLimitError(body))
    assert "AIza0123456789012345678901234567890abcd" not in str(error)
    assert "[REDACTED]" in str(error)
