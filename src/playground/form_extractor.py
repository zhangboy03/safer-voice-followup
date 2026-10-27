"""Silent off-path form extractor (blind design — the flash never sees this).
Re-derives the whole 16-field questionnaire from the full transcript each turn
via DashScope JSON-mode structured output. Output: array of FLAT objects (spec §6)."""

from __future__ import annotations

from src.playground.form_schema import FieldSpec, field_catalog

_STATUS = ("pending", "confirmed", "refused")


def _response_format(catalog: list[FieldSpec]) -> dict:
    # DashScope supports response_format {"type": "json_object"} (JSON mode), NOT the
    # newer OpenAI json_schema strict variant. The exact shape (array of flat objects,
    # spec §6) is specified in _system_prompt and enforced by defensive parsing below.
    # DashScope ALSO requires the word "json" to appear in the messages, otherwise it
    # 400s ("'messages' must contain the word 'json'") — see _system_prompt.
    return {"type": "json_object"}


def _system_prompt(catalog: list[FieldSpec]) -> str:
    lines = [
        "你是随访质检助手。下面是一通随访电话的完整记录。请根据记录,判断每个字段当前的填写情况。",
        "只依据记录中【患者明确说出】的信息填写;患者没提到的标 pending;患者明确拒答的标 refused。不要臆测。",
        "记录可能为英语。按语义匹配中文字段定义，保留患者的英语原话作为 evidence；最新明确更正覆盖旧值。枚举 value 归一到定义，不能因语言差异漏记或臆测。",
        "字段清单(name:定义):",
    ]
    for f in catalog:
        lines.append(f"- {f.name}: {f.definition}")
    lines.append(
        '请只输出 JSON：一个 json 对象,形如 {"fields": [{"name": 字段名, '
        '"status": "confirmed"/"refused", "value": 归一到定义的取值, '
        '"evidence": 患者原话的简短片段}, ...]}。只输出有明确答案或明确拒答的字段，'
        '未提及的 pending 字段省略，系统会补齐；没有答案则输出 {"fields": []}。'
    )
    return "\n".join(lines)


async def extract_form(transcript: str, *, backend, catalog: list[FieldSpec] | None = None) -> list[dict]:
    catalog = catalog if catalog is not None else field_catalog()
    messages = [
        {"role": "system", "content": _system_prompt(catalog)},
        {"role": "user", "content": transcript},
    ]
    res = await backend.complete_structured(
        messages, response_format=_response_format(catalog), max_tokens=1200, temperature=0.0
    )
    data = res.parsed_json
    if not isinstance(data, dict) or not isinstance(data.get("fields"), list):
        raise ValueError("Invalid form extraction JSON; keep the previous snapshot")
    got = {it["name"]: it for it in data.get("fields", []) if isinstance(it, dict) and "name" in it}
    out: list[dict] = []
    for f in catalog:
        it = got.get(f.name, {})
        status = it.get("status")
        out.append(
            {
                "name": f.name,
                "group": f.group,
                "required": f.required,
                "status": status if status in _STATUS else "pending",
                "value": it.get("value", "") or "",
                "evidence": it.get("evidence", "") or "",
            }
        )
    return out
