"""Thesis Chapter 4 ablation switches — single import point.

This module exposes helper functions reading the 4 ablation flags defined on
``Settings`` in :mod:`src.core.config`.  Centralising the reads here keeps
``os.environ`` access out of the flow / pipeline codepaths and lets tests
monkey-patch behaviour via the Settings object.

Defaults MUST keep the production system behaviour 100% unchanged:

- ``is_judge_disabled()``         returns ``False`` unless ``ABLATION_DISABLE_JUDGE=1``
- ``is_state_machine_disabled()`` returns ``False`` unless ``ABLATION_DISABLE_SM=1``
- ``is_verify_disabled()``        returns ``False`` unless ``ABLATION_DISABLE_VERIFY=1``
- ``is_bridge_phrase_enabled()``  returns ``True`` unless ``BRIDGE_PHRASE_ENABLED=0``

Each switch is documented with a grep-able tag so the ablation branches can be
traced throughout the codebase::

    # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6

Pydantic-Settings already reads env at import time via
``BaseSettings(env_file=".env")``, so helpers always re-read the live
``settings`` attribute (no caching) — this keeps tests deterministic when they
flip the env and reload the module, and also keeps runtime cheap because
``settings`` is a plain attribute lookup.
"""

from __future__ import annotations

from src.core.config import settings

# Single tag used across the codebase so reviewers can grep every ablation
# branch in one pass.
ABLATION_COMMENT_TAG = "ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6"


def is_judge_disabled() -> bool:
    """Return True when Judge LLM must be skipped.

    When True, :class:`src.processors.llm_processor.LLMProcessor` must not
    spawn a Judge task; Speech LLM output becomes the only source for field
    extraction / advance decisions.
    """
    # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
    return bool(settings.ablation_disable_judge)


def is_state_machine_disabled() -> bool:
    """Return True when FlowRuntime must skip epoch + 5-level field statuses.

    When True, ``_commit_transition`` must not bump ``flow_epoch`` and fast-judge
    field commits must collapse the 5-level ``FieldStatus`` into a binary
    EMPTY / FILLED model (mapped to ``CONFIRMED`` / absent).
    """
    # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
    return bool(settings.ablation_disable_sm)


def is_verify_disabled() -> bool:
    """Return True when the knowledge_qa three-step verification must be stripped.

    When True, ``pinchi_flow_config.build_knowledge_qa_system_prompt`` drops the
    "三步检查" + "必须拒绝" blocks and keeps only the base persona + KB
    injection.
    """
    # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
    return bool(settings.ablation_disable_verify)


def is_bridge_phrase_enabled() -> bool:
    """Return True when the TTS bridge-phrase (``followup_buffer``) may fire.

    Defaults to True (production behaviour). When the operator sets
    ``BRIDGE_PHRASE_ENABLED=0``, the LLM processor must skip the pre-LLM
    buffer push so TTFA measurements exclude the pre-played audio.
    """
    # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
    return bool(settings.bridge_phrase_enabled)


def is_fast_judge_enabled() -> bool:
    return not is_judge_disabled()


def is_speech_prior_state_snapshot_enabled() -> bool:
    return bool(settings.speech_prior_state_snapshot_enabled)


def is_short_term_memory_analyzer_enabled() -> bool:
    return bool(settings.short_term_memory_analyzer_enabled)


def is_sop_progress_analyzer_enabled() -> bool:
    return bool(settings.sop_progress_analyzer_enabled)


def is_legacy_memory_extractor_enabled() -> bool:
    return bool(settings.legacy_memory_extractor_enabled)


def is_legacy_memory_extractor_shadow_enabled() -> bool:
    return bool(settings.legacy_memory_extractor_shadow_enabled)


def can_legacy_memory_extractor_mutate() -> bool:
    return (
        is_legacy_memory_extractor_enabled()
        and not is_short_term_memory_analyzer_enabled()
        and not is_sop_progress_analyzer_enabled()
    )


def is_stm_for_patient_calls_enabled() -> bool:
    return bool(settings.short_term_memory_analyzer_enabled_for_patient_calls)


def is_note_judge_injection_enabled() -> bool:
    return bool(settings.note_judge_injection_enabled)


def get_note_filter_confidence_threshold() -> float:
    return float(settings.note_filter_confidence_threshold)


def get_note_filter_blacklist() -> list[str]:
    return list(settings.note_filter_blacklist)


__all__ = [
    "ABLATION_COMMENT_TAG",
    "can_legacy_memory_extractor_mutate",
    "get_note_filter_blacklist",
    "get_note_filter_confidence_threshold",
    "is_bridge_phrase_enabled",
    "is_fast_judge_enabled",
    "is_judge_disabled",
    "is_legacy_memory_extractor_enabled",
    "is_legacy_memory_extractor_shadow_enabled",
    "is_note_judge_injection_enabled",
    "is_short_term_memory_analyzer_enabled",
    "is_sop_progress_analyzer_enabled",
    "is_speech_prior_state_snapshot_enabled",
    "is_state_machine_disabled",
    "is_stm_for_patient_calls_enabled",
    "is_verify_disabled",
]
