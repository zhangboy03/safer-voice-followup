"""[E] Exit Judge LLM runner — Qwen3.6-Flash, strict JSON, 4-intent enum.

Per spec §7.2. Mirrors KnowledgeReviewRunner: LLMBackend in constructor,
last_usage captured. Returns KnowledgeExitDecision (no followup_round_at_request
— that field belongs only to [A] envelopes).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from src.backends.llm_base import LLMBackend
from src.backends.structured_capabilities import build_response_format
from src.core.logging import get_logger
from src.flows.knowledge_review_models import (
    KnowledgeDecisionEnvelope,
    KnowledgeExitDecision,
)

logger = get_logger(__name__)

_MANIFEST = Path("config/prompts/packs/sop-v1/manifest.yaml")
_PACK_ROOT = Path("config/prompts/packs/sop-v1")


class _ExitSchema(BaseModel):
    exit_intent: str
    reason: str


_EXIT_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "exit_intent": {
            "type": "string",
            "enum": ["done", "new_question", "continue", "escalate"],
        },
        "reason": {"type": "string", "maxLength": 80},
    },
    "required": ["exit_intent", "reason"],
    "additionalProperties": False,
}


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class KnowledgeExitRunner:
    backend: LLMBackend
    model: str
    template_text: str
    prompt_hash: str
    temperature: float
    max_tokens: int
    last_raw_text: str = field(default="")
    last_usage: dict[str, Any] | None = field(default=None)

    @classmethod
    def from_manifest(cls, *, backend: LLMBackend) -> "KnowledgeExitRunner":
        m = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
        template_rel = m["nodes"]["knowledge_qa"]["exit_judge"]
        exit_defaults = m["defaults"]["exit_judge"]
        template_path = _PACK_ROOT / template_rel
        template_text = template_path.read_text(encoding="utf-8")
        return cls(
            backend=backend,
            model=exit_defaults["model"],
            template_text=template_text,
            prompt_hash=_hash(template_text),
            temperature=exit_defaults.get("temperature", 0),
            max_tokens=exit_defaults.get("max_tokens", 100),
        )

    async def run(
        self,
        *,
        last_assistant_text: str,
        user_text: str,
        turn_id: str,
        flow_epoch: int,
        session_id: str,
        kb_hash: str,
    ) -> KnowledgeExitDecision:
        messages = [
            {"role": "system", "content": self.template_text},
            {"role": "user", "content": (f"last_assistant_text: {last_assistant_text}\nuser_text: {user_text}")},
        ]
        response_format = build_response_format(
            model=self.model,
            schema_name="knowledge_qa_exit_v1",
            schema=_EXIT_SCHEMA_DICT,
        )
        logger.info(
            "knowledge_exit.call response_format_kind=%s model=%s turn_id=%s session_id=%s",
            response_format.get("type"),
            self.model,
            turn_id,
            session_id,
        )
        result = await self.backend.complete_structured(
            messages,
            response_format=response_format,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            session_id=session_id,
            turn_id=turn_id,
        )
        self.last_raw_text = result.raw_text
        self.last_usage = result.usage
        try:
            if result.parsed_json is None:
                raise RuntimeError("parse_failed")
            parsed = _ExitSchema.model_validate(result.parsed_json)
        except (ValidationError, RuntimeError) as exc:
            raise RuntimeError(f"[E] response schema invalid: {exc}") from exc

        envelope = KnowledgeDecisionEnvelope(
            turn_id=turn_id,
            flow_epoch=flow_epoch,
            session_id=session_id,
            input_hash=_hash(user_text),
            kb_hash=kb_hash,
            prompt_hash=self.prompt_hash,
        )
        return KnowledgeExitDecision(
            exit_intent=parsed.exit_intent,  # type: ignore[arg-type]
            envelope=envelope,
            raw_user_reply=user_text,
        )
