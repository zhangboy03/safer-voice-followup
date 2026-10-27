from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.core.logging import get_logger

logger = get_logger("prompt_pack")


class PromptPackError(ValueError):
    pass


_VAR_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")
_HASH_RE = re.compile(r"sha256:[0-9a-f]{12,64}")
_GLOBAL_STATIC_MD_REL_PATH = "fast_judge/global_static.md"
_STAGE2_ENTRY_PERSONA_EXCLUDED_LINES = {
    "如果用户本句已经足够完成当前节点，只做简短承接，等待系统进入下一段；不要自己顺手问下一节点。",
}
DEFAULTS = {
    "critical_facts": "无",
    "speaking_frame": "接听人身份尚未确认。用语保持中性，按当前节点的任务输出。",
    "next_speech_strategy": "无上一轮策略。",
    "call_working_memory": "无",
    "rolling_call_summary": "无",
    "recent_verbatim_window": "无",
    "heard_context": "无",
    "knowledge_context": "无",
    "speech_text": "",
    "user_text": "",
}


def _clean_model_visible_text(text: Any, *, limit: int = 1200) -> str:
    if text in (None, "", [], {}):
        return ""
    cleaned = _HASH_RE.sub("<hash>", str(text)).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


@dataclass(frozen=True)
class PromptRef:
    path: Path
    model_config: dict[str, Any]
    schema_version: str


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    hash: str
    ref: PromptRef


@dataclass(frozen=True)
class FastJudgeStaticPromptV2:
    """v2 prompt split: global_part 跨节点共享，node_part 节点专属。"""

    global_part: RenderedPrompt
    node_part: RenderedPrompt


class PromptPack:
    def __init__(self, pack_dir: Path, manifest: dict[str, Any]) -> None:
        self.pack_dir = pack_dir
        self.manifest = manifest
        self.name = manifest["name"]

    def node_ids(self) -> list[str]:
        return sorted(self.manifest.get("nodes", {}).keys())

    def has_ref(self, node_id: str, lane: str) -> bool:
        return lane in self.manifest.get("nodes", {}).get(node_id, {})

    def model_override(self, node_id: str, lane: str) -> str | None:
        """Per-node model override for a lane, or None to use the lane default.

        Reads the optional top-level ``node_overrides`` section:
            node_overrides: {knowledge_qa: {speech: {model: qwen3.7-plus}}}
        Keeps the ``nodes`` template map untouched (lane → path string)."""
        return self.manifest.get("node_overrides", {}).get(node_id, {}).get(lane, {}).get("model")

    def resolve(self, node_id: str, lane: str) -> PromptRef:
        try:
            rel = self.manifest["nodes"][node_id][lane]
        except KeyError as exc:
            raise PromptPackError(f"missing prompt ref: {node_id}/{lane}") from exc
        return self._ref(rel, lane)

    def resolve_global(self, group: str, name: str) -> PromptRef:
        rel = self.manifest[group][name]
        return self._ref(rel, group)

    def _ref(self, rel: str, lane: str) -> PromptRef:
        path = self.pack_dir / rel
        if not path.exists():
            raise PromptPackError(f"prompt file missing: {path}")
        return PromptRef(
            path=path,
            model_config=dict(self.manifest.get("defaults", {}).get(lane, {})),
            schema_version=self.manifest["schema_version"],
        )

    def render(self, node_id: str, lane: str, variables: dict[str, Any]) -> RenderedPrompt:
        ref = self.resolve(node_id, lane)
        return _render_ref(ref, variables)


def _render_ref(ref: PromptRef, variables: dict[str, Any]) -> RenderedPrompt:
    raw = ref.path.read_text(encoding="utf-8")
    values = dict(DEFAULTS)
    values.update({k: v for k, v in variables.items() if v not in (None, "")})

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise PromptPackError(f"missing template variable: {key}")
        return str(values[key])

    text = _VAR_RE.sub(repl, raw)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return RenderedPrompt(text=text, hash=f"sha256:{digest}", ref=ref)


def _stage2_entry_persona_text(persona: str) -> str:
    return "\n".join(line for line in persona.splitlines() if line.strip() not in _STAGE2_ENTRY_PERSONA_EXCLUDED_LINES)


class PromptPackLoader:
    def __init__(self, root: Path = Path("config/prompts")) -> None:
        self.root = root

    def load_active(self) -> PromptPack:
        active = yaml.safe_load((self.root / "active.yaml").read_text(encoding="utf-8"))
        pack_dir = self.root / "packs" / active["active_pack"]
        manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text(encoding="utf-8"))
        pack = PromptPack(pack_dir=pack_dir, manifest=manifest)
        for node_id in pack.node_ids():
            for lane in manifest["nodes"][node_id]:
                pack.resolve(node_id, lane)
        for group in ("persona", "silent_rescue", "strategy", "memory", "eval", "fast_judge"):
            for name in manifest.get(group, {}):
                pack.resolve_global(group, name)
        return pack


class PromptRegistry:
    def __init__(self, pack: PromptPack) -> None:
        self._pack = pack

    def _render_ref(self, ref: PromptRef, variables: dict[str, Any]) -> RenderedPrompt:
        return _render_ref(ref, variables)

    def render_persona(self, variables: dict[str, Any] | None = None) -> RenderedPrompt:
        ref = self._pack.resolve_global("persona", "main")
        return self._render_ref(ref, variables or {})

    def render_node(self, *, node_id: str, lane: str, variables: dict[str, Any]) -> RenderedPrompt:
        return self._pack.render(node_id=node_id, lane=lane, variables=variables)

    def has_node_prompt(self, node_id: str, lane: str) -> bool:
        return self._pack.has_ref(node_id, lane)

    def model_override(self, node_id: str, lane: str) -> str | None:
        """Per-node model override for a lane (None = use lane default)."""
        return self._pack.model_override(node_id, lane)

    def render_stage2_entry_prompt(
        self,
        *,
        node_id: str,
        variables: dict[str, Any] | None = None,
    ) -> RenderedPrompt:
        values = variables or {}
        persona = _stage2_entry_persona_text(self.render_persona(values).text)
        contract = self.render_node(node_id=node_id, lane="stage2_entry", variables=values)
        invariant = "\n".join(
            [
                "# Stage 2 Entry Ask Mode",
                "",
                "你刚说完上一节点的一句话，现在正在接着进入当前节点。",
                "上一句 assistant 是本句的上半句；本句必须像同一个人自然接着说下去。",
                "",
                "最高优先级：问出 runtime 指定的目标字段。",
                "自然衔接是包装，目标字段提问是动作；包装不能吞掉动作。",
                "",
                "输出要求：",
                "- 只输出一句中文。",
                "- 如果 runtime 指定了目标字段，本句必须问出该字段，必须带问号。",
                "- 如果 runtime 明确说当前节点不需要提问，就自然完成该节点，不要硬造问题。",
                "- 不重新打招呼，不重新自我介绍，不重新总结上一节点。",
            ]
        )
        text = "\n\n".join([persona, invariant, contract.text])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return RenderedPrompt(text=text, hash=f"sha256:{digest}", ref=contract.ref)

    def render_speech_prompt(self, *, node_id: str, weighted_context, variables: dict[str, Any]) -> RenderedPrompt:
        persona = self.render_persona(variables).text
        speech_variables = {k: v for k, v in variables.items() if k != "weighted_context_hash"}
        speech_variables.update(
            {
                "weighted_context": "",
                "user_text": "",
            }
        )
        node = self.render_node(
            node_id=node_id,
            lane="speech",
            variables=speech_variables,
        ).text
        text = "\n\n".join([persona, node])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return RenderedPrompt(text=text, hash=f"sha256:{digest}", ref=self._pack.resolve(node_id, "speech"))

    def render_silent_rescue_prompt(self, variables: dict[str, Any] | None = None) -> RenderedPrompt:
        """ADR 0006: silent-rescue uses persona + a dedicated rescue prompt that
        deliberately bypasses node-specific scripts. Node prompts (e.g. greeting's
        "B2 必须包含完整开场白") otherwise dominate and force re-delivery of the
        same intro line. The rescue mode wants natural continuation anchored on
        the last assistant utterance — see ``speech/silent_rescue.md``.
        """
        persona = self.render_persona(variables).text
        ref = self._pack.resolve_global("silent_rescue", "main")
        rescue = _render_ref(ref, variables or {}).text
        text = "\n\n".join([persona, rescue])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return RenderedPrompt(text=text, hash=f"sha256:{digest}", ref=ref)

    def render_speech_runtime_context(self, *, weighted_context, variables: dict[str, Any] | None = None) -> str:
        del variables
        sections: list[str] = []

        speaking_frame = _clean_model_visible_text(getattr(weighted_context, "speaking_frame", ""), limit=500)
        if speaking_frame:
            sections.append(f"通话视角：{speaking_frame}")

        fast_judge_state = getattr(weighted_context, "fast_judge_state", "")
        if fast_judge_state:
            try:
                state = json.loads(fast_judge_state)
            except (TypeError, json.JSONDecodeError):
                state = {}
            current_node = _clean_model_visible_text(getattr(weighted_context, "current_node", ""), limit=80)
            confirmed = []
            for item in state.get("committed_delta") or []:
                field = item.get("field")
                value = item.get("value")
                if field:
                    confirmed.append(f"{field}={value}")
            for item in state.get("lookahead_committed_delta") or []:
                node = item.get("node_id")
                field = item.get("field")
                value = item.get("value")
                if node and field:
                    # Only show lookahead deltas for the current node. Other
                    # nodes' fields would leak cross-node entities (e.g. show
                    # "identity.is_patient" inside greeting Stage 1's prompt).
                    if current_node and node != current_node:
                        continue
                    confirmed.append(f"{field}={value}")
            if confirmed:
                sections.append(
                    "已确认：" + "；".join(_clean_model_visible_text(item, limit=120) for item in confirmed)
                )
            missing = state.get("missing_fields") or []
            if missing:
                sections.append("待补问：" + "、".join(_clean_model_visible_text(item, limit=80) for item in missing))
            node_action = _clean_model_visible_text(state.get("node_action", ""), limit=80)
            target = _clean_model_visible_text(state.get("stage2_target_node", ""), limit=80)
            if node_action or target:
                sections.append(f"流程判断：动作={node_action or '无'}；下一节点={target or '无'}")
            directive = state.get("speech_directive") or {}
            if isinstance(directive, dict):
                directive_text = _clean_model_visible_text(directive.get("text", ""), limit=500)
                directive_code = _clean_model_visible_text(directive.get("code", ""), limit=80)
                if directive_text:
                    prefix = (
                        f"下一步意图（来自 Judge，仅作方向参考；{directive_code}）"
                        if directive_code
                        else "下一步意图（来自 Judge，仅作方向参考）"
                    )
                    sections.append(
                        f"{prefix}：{directive_text}\n"
                        "（注意：上面这条是 Judge 在看到用户上一句话时给出的方向建议；"
                        "如果用户最新一句话和这个方向不一致——追问、反悔、岔开话题、提新问题——"
                        "以**用户最新一句话**为准，先回应用户，再决定要不要落到这个方向。）"
                    )

        short_term_memory = _clean_model_visible_text(getattr(weighted_context, "short_term_memory", ""), limit=800)
        if short_term_memory and short_term_memory != "无":
            sections.append("短期记忆：\n" + short_term_memory)

        sop_candidates = _clean_model_visible_text(getattr(weighted_context, "sop_progress_candidates", ""), limit=800)
        if sop_candidates and sop_candidates != "无":
            sections.append("候选信息：\n" + sop_candidates)

        recent = _clean_model_visible_text(getattr(weighted_context, "recent_verbatim_window", ""), limit=1200)
        if recent and recent != "无":
            sections.append("最近对话：\n" + recent)

        summary = _clean_model_visible_text(getattr(weighted_context, "rolling_call_summary", ""), limit=800)
        if summary and summary != "无":
            sections.append("摘要：\n" + summary)

        if not sections:
            return ""
        return "# 当前通话状态\n" + "\n\n".join(sections)

    def _fast_judge_context_obj(self, context) -> dict[str, Any]:
        raw_context = context.model_dump(
            mode="json",
            exclude={"user_text", "turn_index", "flow_epoch"},
            exclude_none=True,
        )
        stage1_spoken_text = _clean_model_visible_text(raw_context.pop("speech_text", ""), limit=1200)
        required_context_keys = (
            "node_id",
            "speech_asked_this_turn",
            "intended_ask_fields",
            "required_fields",
            "eligible_optional_fields",
            "pending_fields_before_turn",
        )
        optional_context_keys = (
            "successor_path",
            "node_field_catalog",
            "known_fields_before_turn",
            "candidate_fields_before_turn",
            "recent_relevant_context",
            "short_term_memory",
        )
        context_obj = {key: raw_context[key] for key in required_context_keys if key in raw_context}
        if stage1_spoken_text:
            context_obj["stage1_spoken_text"] = stage1_spoken_text
        for key in optional_context_keys:
            value = raw_context.get(key)
            if value not in (None, "", [], {}):
                context_obj[key] = value
        return context_obj

    def render_fast_judge_context_block(self, *, context) -> str:
        context_obj = self._fast_judge_context_obj(context)
        context_json = json.dumps(context_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "\n".join(["<context>", context_json, "</context>"])

    def render_fast_judge_static_prompt(self, *, node_id: str) -> RenderedPrompt:
        if self._pack.has_ref(node_id, "fast_judge"):
            node = self.render_node(
                node_id=node_id,
                lane="fast_judge",
                variables={
                    "context": "",
                    "context_json": "",
                },
            )
        else:
            ref = self._pack.resolve_global("fast_judge", "default")
            node = self._render_ref(
                ref,
                variables={
                    "context": "",
                    "context_json": "",
                },
            )
        text = "\n".join(
            [
                "You are the Fast SOP Judge. Output only fast_judge.v1 JSON.",
                "<task>",
                "Evaluate only this user turn after Speech. Extract explicit SOP answers; do not summarize.",
                "</task>",
                "<output>",
                "Return exactly one minified single-line JSON object and nothing else.",
                "Do not output wrapper keys such as fast_judge, v1, result, data, or response.",
                "Do not output node_action values like complete or next; use only stay, followup, advance, repair, divert, handoff, or terminal.",
                "Allowed top-level keys only: field_delta, missing_fields, node_action, speech_directive, stage1_posture.",
                'Use field_delta as an array. Each item must use key "field", never "field_name". Use [] when no field is confirmed.',
                'Every field_delta item must include "field", "value", "status", "evidence_span", and "confidence"; if value is unknown, omit the item and use missing_fields.',
                'If the user clearly answered a later node, put it in field_delta with "node_id" set to that target node from node_field_catalog.',
                'Use field value types from node_field_catalog: call_accepted="confirmed"/"refused"; is_patient=true/false booleans; control_score=0-10 number, not a string.',
                "missing_fields must be an array of field-name strings only, not objects.",
                "Do not output any other top-level key.",
                "Never output null. Use empty strings for optional text fields, [] for list fields, and JSON objects for nested fields.",
                "speech_directive must be a JSON object, not a string, boolean, or null.",
                'speech_directive must have {"code":short_stable_code,"text":advisory_Chinese_direction_for_next_Speech}. The text describes intent and bans for the next Speech turn; it is advisory, not a script. Never write a complete spoken sentence in text, never quote phrasing like 「好的，那…」. Use conditional form ("先承接，再…" / "若用户已答则…，若用户未答则…"). Speech reads it as direction, then grounds the wording in the user\'s most recent message.',
                "Use field_delta[].evidence_span for field evidence; use speech_directive.code/text to explain transfer, knowledge-question diversion, or next Speech direction.",
                "If the user text is unclear or incompatible with the task, return empty deltas and the relevant missing_fields.",
                "如果 Speech 已经问出下一节点问题，且当前节点已完成：仍然用 node_action=advance；speech_directive.code=already_asked_next_node，并在 speech_directive.text 提醒下一节点 Speech 不要重复刚问过的问题。",
                "stage1_posture 三选一，描述本轮 Speech(Stage 1)最后一句的姿态："
                '"asked"=已问出一个能让用户回应的问题（任何问题都算）；'
                '"ack_only"=只说了承接/铺垫（如「好的」「稍等」），既没问也没收尾；'
                '"closed_unasked"=已收尾（如「我了解了」「那就这样」），既没问问题、也没把还缺的信息问出来。'
                '判定前先在 reason 里逐字引用 Stage 1 最后一句再分类；拿不准时输出 "asked"（绝不臆造收尾）。',
                "Keep evidence_span short.",
                "</output>",
                "<rules>",
                node.text.strip(),
                "</rules>",
            ]
        )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        logger.info(
            "fast_judge.static_prompt_hash",
            node_id=node_id,
            hash=f"sha256:{digest}",
        )
        return RenderedPrompt(text=text, hash=f"sha256:{digest}", ref=node.ref)

    def render_fast_judge_static_prompt_v2(self, *, node_id: str) -> FastJudgeStaticPromptV2:
        """v2 渲染：返回 (global_part, node_part) 两段，分别挂 cache_control 标记。

        global_part 跨节点共享、≥1024 token、命中阿里云显式 cache 标记 #1；
        node_part 节点专属、命中标记 #2；节点切换时仅 #2 失效。

        v1 接口 render_fast_judge_static_prompt 保留不变。

        Note: node_part.text 经过 .strip() 归一化以稳定 hash（trailing newline 不影响 cache）。
        node_part.ref 仍指向原节点 .md 文件路径，不映射 strip 后的 char offset。

        node_part 通过 render_node() 走，模板变量 context/context_json 传空串（v2 用单独的
        context_block 拼装，不走 template substitution）。如果节点 .md 还残留 v1 时代的
        {context} 占位符，会被替换为空串而非保留——这是 Task 7 改写节点的边界条件。
        """
        # global_part: 从 pack_dir 直接读取 global_static.md（不经过 manifest 模板系统）
        global_md_path = self._pack.pack_dir / Path(_GLOBAL_STATIC_MD_REL_PATH)
        if not global_md_path.exists():
            raise PromptPackError(f"v2 prompt requires global_static.md: {global_md_path}")
        global_text = global_md_path.read_text(encoding="utf-8")
        global_digest = hashlib.sha256(global_text.encode("utf-8")).hexdigest()
        global_ref = PromptRef(
            path=global_md_path,
            model_config=dict(self._pack.manifest.get("defaults", {}).get("fast_judge", {})),
            schema_version=self._pack.manifest["schema_version"],
        )
        global_rendered = RenderedPrompt(
            text=global_text,
            hash=f"sha256:{global_digest}",
            ref=global_ref,
        )

        # node_part: 复用现有节点 .md 读取逻辑（只取节点文本，不拼 v1 英文 preamble）
        if self._pack.has_ref(node_id, "fast_judge"):
            node = self.render_node(
                node_id=node_id,
                lane="fast_judge",
                variables={"context": "", "context_json": ""},
            )
        else:
            ref = self._pack.resolve_global("fast_judge", "default")
            node = self._render_ref(ref, variables={"context": "", "context_json": ""})
        node_text = node.text.strip()
        node_digest = hashlib.sha256(node_text.encode("utf-8")).hexdigest()
        node_rendered = RenderedPrompt(
            text=node_text,
            hash=f"sha256:{node_digest}",
            ref=node.ref,
        )

        return FastJudgeStaticPromptV2(global_part=global_rendered, node_part=node_rendered)

    def render_fast_judge_prompt(self, *, node_id: str, context) -> RenderedPrompt:
        static = self.render_fast_judge_static_prompt(node_id=node_id)
        context_block = self.render_fast_judge_context_block(context=context)
        text = "\n".join([static.text, context_block])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return RenderedPrompt(text=text, hash=f"sha256:{digest}", ref=static.ref)

    def render_memory_prompt(
        self, *, weighted_context, transcript_delta: str, variables: dict[str, Any]
    ) -> RenderedPrompt:
        ref = self._pack.resolve_global("memory", "call_memory_extractor")
        node = self._render_ref(
            ref,
            {
                **variables,
                "weighted_context": weighted_context.render_for_prompt(),
                "weighted_context_hash": weighted_context.weighted_context_hash,
                "context_snapshot_id": weighted_context.context_snapshot_id,
                "transcript_delta": transcript_delta,
            },
        )
        text = "\n\n".join(
            [
                "Extract short-lived call working memory. Output only JSON matching call_memory_patch.v1.",
                "Top-level identity fields to include when available:",
                "\n".join(
                    [
                        'schema_version: "call_memory_patch.v1"',
                        f"turn_id: {variables.get('turn_id', '')}",
                        f"turn_index: {variables.get('turn_index', '')}",
                        f"node_id: {variables.get('node_id', '')}",
                        f"flow_epoch: {variables.get('flow_epoch', '')}",
                        'source: "memory_extractor"',
                    ]
                ),
                "Runtime inputs, not output fields:",
                "\n".join(
                    [
                        f"context_snapshot_id: {weighted_context.context_snapshot_id}",
                        f"weighted_context_hash: {weighted_context.weighted_context_hash}",
                    ]
                ),
                f"ContextSnapshot: {weighted_context.context_snapshot_id}",
                "[WeightedContextHash]",
                weighted_context.weighted_context_hash,
                "[WeightedContext]",
                weighted_context.render_for_prompt(),
                "[transcript_delta]",
                transcript_delta,
                "[MemoryExtractionRules]",
                node.text,
            ]
        )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return RenderedPrompt(text=text, hash=f"sha256:{digest}", ref=ref)
