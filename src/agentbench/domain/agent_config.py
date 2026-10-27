"""Versioned configuration of a model-backed agent.

Everything that can change a model agent's behaviour is part of its configuration and
therefore of its `config_hash`: provider, model id, prompt id and version, prompt text,
decoding settings, and API timeout and retry settings. Two runs with the same hash used
the same configuration. That does NOT make them deterministic: a live model can answer
the same request differently.

The tool configuration (which tools, with which schemas) is versioned separately: the
runner records a hash of the exact tool schemas the agent was shown in RUN_STARTED.
"""

import hashlib
import json
from typing import Literal

from pydantic import Field

from agentbench.domain.models import DomainModel, Slug


class DecodingSettings(DomainModel):
    """Sampling settings sent to the model. None means "provider default" and is not
    sent at all, so the request says exactly what was chosen."""

    max_output_tokens: int = Field(default=1024, ge=1, le=64_000)
    temperature: float | None = Field(default=None, ge=0, le=1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop_sequences: tuple[str, ...] = ()


class PromptTemplate(DomainModel):
    id: Slug
    version: str = Field(pattern=r"^[A-Za-z0-9._-]{1,50}$")
    system: str = Field(min_length=1, max_length=20_000)


class ModelAgentConfig(DomainModel):
    label: str = Field(min_length=1, max_length=100)  # display name; not part of the hash
    provider: Literal["anthropic", "scripted"]
    model: str = Field(min_length=1, max_length=100)
    prompt: PromptTemplate
    decoding: DecodingSettings = Field(default_factory=DecodingSettings)
    request_timeout_s: float = Field(default=60.0, gt=0, le=600)
    max_api_retries: int = Field(default=2, ge=0, le=5)
    api_retry_backoff_s: float = Field(default=1.0, ge=0, le=60)

    @property
    def config_hash(self) -> str:
        content = self.model_dump(mode="json", exclude={"label"})
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def version(self) -> str:
        """Human-readable version: prompt version plus a short configuration hash."""
        return f"{self.prompt.id}@{self.prompt.version}+{self.config_hash[:12]}"


def tool_schemas_hash(schemas: list[dict] | tuple[dict, ...]) -> str:
    canonical = json.dumps(list(schemas), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
