"""One diagnostic result, shared by ``kennel doctor`` and provider-specific checks.

It lives here (not in ``kennel.cli.doctor``) so that a :class:`~kennel.providers.registry.ProviderSpec`
can declare its own checks without the providers layer importing the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    hint: str | None = None
