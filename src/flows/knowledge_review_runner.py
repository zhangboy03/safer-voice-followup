"""[A] Review LLM runner — Qwen-Plus, strict JSON schema, identity-stamped envelope.

Per spec §4 + §Data Model. Mirrors src/flows/fast_judge_runner.py:
- Constructed with a concrete LLMBackend (the same protocol FastJudge uses).
- `_call_llm` delegates to backend.complete_structured() (no NotImplementedError).
- Captures last_raw_text and last_usage so callers can attribute cache-hit metrics.

Identity (envelope) is set here, NOT inside the LLM output — the LLM cannot
tamper with identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from src.backends.llm_base import LLMBackend
from src.backends.structured_capabilities import build_response_format
from src.core.logging import get_logger
from src.flows.knowledge_review_models import (
    KnowledgeReviewEnvelope,
    KnowledgeReviewFacts,
)

logger = get_logger(__name__)

_MANIFEST = Path("config/prompts/packs/sop-v1/manifest.yaml")
_PACK_ROOT = Path("config/prompts/packs/sop-v1")


class _ReviewSchema(BaseModel):
    scope: str
    out_of_scope_class: str | None
    missing_prerequisites: list[str]
    answer_readiness: str | None
    reason: str
    confidence: float


_RESPONSE_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "scope": {"type": "string", "enum": ["in_scope", "out_of_scope"]},
        "out_of_scope_class": {
            "type": ["string", "null"],
            "enum": [
                "medical_judgment",
                "kb_out_of_coverage",
                "adversarial_or_pii",
                "safety_risk",
                None,
            ],
        },
        "missing_prerequisites": {
            "type": "array",
            "items": {"type": "string", "maxLength": 12},
            "minItems": 0,
            "maxItems": 3,
        },
        "answer_readiness": {
            "type": ["string", "null"],
            "enum": ["ready", "needs_clarification", None],
        },
        "reason": {"type": "string", "maxLength": 120},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "scope",
        "out_of_scope_class",
        "missing_prerequisites",
        "answer_readiness",
        "reason",
        "confidence",
    ],
    "additionalProperties": False,
}


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class KnowledgeReviewRunner:
    backend: LLMBackend
    model: str
    template_text: str
    prompt_hash: str
    temperature: float
    max_tokens: int
    last_raw_text: str = field(default="")
    last_usage: dict[str, Any] | None = field(default=None)

    @classmethod
    def from_manifest(cls, *, backend: LLMBackend) -> "KnowledgeReviewRunner":
        m = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
        # New manifest schema (Task 4): node lane is a string path; defaults.review carries model config.
        template_rel = m["nodes"]["knowledge_qa"]["review"]
        review_defaults = m["defaults"]["review"]
        template_path = _PACK_ROOT / template_rel
        template_text = template_path.read_text(encoding="utf-8")
        return cls(
            backend=backend,
            model=review_defaults["model"],
            template_text=template_text,
            prompt_hash=_hash(template_text),
            temperature=review_defaults.get("temperature", 0),
            max_tokens=review_defaults.get("max_tokens", 200),
        )

    async def run(
        self,
        *,
        user_text: str,
        recent_dialogue: list[dict],
        session_id: str,
        followup_round: int,
        turn_id: str,
        flow_epoch: int,
        kb_hash: str,
        prompt_hash: str | None = None,
    ) -> KnowledgeReviewFacts:
        messages = [
            {"role": "system", "content": self.template_text},
            {
                "role": "user",
                "content": self._user_block(
                    user_text,
                    recent_dialogue,
                    followup_round,
                ),
            },
        ]
        response_format = build_response_format(
            model=self.model,
            schema_name="knowledge_qa_review_v1",
            schema=_RESPONSE_SCHEMA_DICT,
        )
        logger.info(
            "knowledge_review.call response_format_kind=%s model=%s turn_id=%s session_id=%s",
            response_format.get("type"),
            self.model,
            turn_id,
            session_id,
        )

        result = await self._call_llm(
            messages=messages,
            response_format=response_format,
            session_id=session_id,
            turn_id=turn_id,
        )
        self.last_raw_text = result.raw_text
        self.last_usage = result.usage

        try:
            if result.parsed_json is None:
                raise RuntimeError("parse_failed")
            parsed = _ReviewSchema.model_validate(result.parsed_json)
        except (ValidationError, RuntimeError) as exc:
            raise RuntimeError(f"[A] response schema invalid: {exc}") from exc

        envelope = KnowledgeReviewEnvelope(
            turn_id=turn_id,
            flow_epoch=flow_epoch,
            session_id=session_id,
            followup_round_at_request=followup_round,
            input_hash=_input_hash(user_text, recent_dialogue),
            kb_hash=kb_hash,
            prompt_hash=prompt_hash or self.prompt_hash,
        )
        return KnowledgeReviewFacts(
            scope=parsed.scope,  # type: ignore[arg-type]
            out_of_scope_class=parsed.out_of_scope_class,  # type: ignore[arg-type]
            missing_prerequisites=list(parsed.missing_prerequisites),
            answer_readiness=parsed.answer_readiness,  # type: ignore[arg-type]
            reason=parsed.reason,
            confidence=parsed.confidence,
            envelope=envelope,
        )

    async def _call_llm(self, *, messages, response_format, session_id, turn_id):
        """Delegate to LLMBackend.complete_structured (mirrors FastJudgeRunner)."""
        return await self.backend.complete_structured(
            messages,
            model=self.model,
            response_format=response_format,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            session_id=session_id,
            turn_id=turn_id,
        )

    @staticmethod
    def _user_block(user_text: str, recent_dialogue: list[dict], followup_round: int) -> str:
        return (
            f"user_text: {user_text}\n"
            f"session_state.followup_round: {followup_round}\n"
            f"recent_dialogue: {_canonical_dialogue(recent_dialogue)}"
        )


def _canonical_dialogue(recent_dialogue: list[dict]) -> str:
    """Canonical JSON for hashing AND the user-block. Used by both sides so
    input_hash is reproducible by the envelope guard at receive time."""
    return json.dumps(recent_dialogue, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _input_hash(user_text: str, recent_dialogue: list[dict]) -> str:
    """Single canonical recipe for input_hash. The envelope guard MUST use
    the same function (see knowledge_turn._compute_input_hash) — diverging
    recipes are the bug Codex Gate 3 flagged as a No-Ship test inconsistency."""
    return _hash(user_text + "\n" + _canonical_dialogue(recent_dialogue))
