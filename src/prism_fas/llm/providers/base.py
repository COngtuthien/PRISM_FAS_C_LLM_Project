"""The provider-neutral interface.

Every implementation, including the test doubles, runs the target-leakage
firewall before the request leaves the method. A leak therefore cannot hide
behind a mock, and the firewall is exercised by the ordinary unit tests rather
than only on the live path.
"""
from __future__ import annotations

import abc
from typing import Any

from ..contracts import GenerationRequest, ProviderGenerationResult
from ..firewall import assert_request_is_target_free


class RecipeProvider(abc.ABC):
    """Turns a `GenerationRequest` into a `ProviderGenerationResult`.

    Implementations never raise a provider failure: a failure is returned inside
    the result as a classified `ProviderError`, so one attempt always produces
    exactly one provenance record.
    """

    #: Recorded in provenance; must match the configured provider name.
    name: str = "abstract"

    def generate(self, request: GenerationRequest, *, attempt: int = 1) -> ProviderGenerationResult:
        self.check_request(request)
        return self._generate(request, attempt=attempt)

    def check_request(self, request: GenerationRequest) -> None:
        """Fail closed before anything is transmitted."""
        assert_request_is_target_free({
            "system_instruction": request.system_instruction,
            "input_text": request.input_text,
            "response_json_schema": request.response_json_schema,
            "metadata": request.metadata,
        })

    @abc.abstractmethod
    def _generate(self, request: GenerationRequest, *, attempt: int) -> ProviderGenerationResult:
        ...

    def describe(self) -> dict[str, Any]:
        """Provider identity for the audit report. Never includes a credential."""
        return {"provider": self.name}
