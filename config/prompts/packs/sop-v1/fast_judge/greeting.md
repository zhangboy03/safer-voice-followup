# 节点：greeting

## 节点目标
确认接听人是否愿意继续本次术后随访。这里要的是「现在方便接电话吗」，
不是寒暄、不是问候、不是身份。

## 输出字段
- `call_accepted`：required；字符串枚举 confirmed / refused。
  用户主动表达不方便时写 status=refused，value=refused。

## 节点专属判定
- advance：用户明确说"行 / 方便 / 你说吧 / 嗯可以 / 好的"。
- terminal：用户明确说"不方便 / 别打了 / 改天 / 在忙"，写 call_accepted=refused。
- followup：用户只寒暄（喂、你好）、反问来意（谁啊、什么事）、答非所问。
  这种情况不写 field_delta，把 call_accepted 留在 missing_fields。

## speech_directive 模板
- `ask_call_permission`   意图=还不知道是否方便接听；禁忌=把"喂/你好"当作同意。
- `advance_to_identity`   意图=已同意，下一段进入身份确认；禁忌=以"好的/嗯"单独开头、再追问是否方便。
- `end_call_refused`      意图=已拒接，下一段礼貌结束；禁忌=继续问随访问题。
- `handoff_user_request`  意图=用户要求人工；禁忌=继续本节点。

## 节点专属示例
- 用户：「嗯行，你说吧。」 → field_delta=[{field:"call_accepted", value:"confirmed", status:"confirmed", evidence_span:"嗯行", confidence:0.95}], node_action=advance, speech_directive.code=advance_to_identity, reason=「用户明确同意接听，推进到身份确认」。
- 用户：「现在不方便，别打了。」 → field_delta=[{field:"call_accepted", value:"refused", status:"refused", evidence_span:"不方便，别打了", confidence:0.98}], node_action=terminal, speech_directive.code=end_call_refused, reason=「用户明确拒接，结束通话」。
- 用户：「方便你说吧。」 → field_delta=[{field:"call_accepted", value:"confirmed", status:"confirmed", evidence_span:"方便", confidence:0.95}], node_action=advance, speech_directive.code=advance_to_identity, reason=「用户确认方便，推进到身份确认」。

## 反例（不该写字段）
- 用户：「喂，你好。」 → field_delta=[], missing_fields=["call_accepted"], node_action=followup, speech_directive.code=ask_call_permission, reason=「纯寒暄，没有同意或拒绝的证据」。
- 用户：「谁啊，什么事？」 → field_delta=[], missing_fields=["call_accepted"], node_action=followup, speech_directive.code=ask_call_permission, reason=「反问来意，不构成同意」。

## 边界 corner case
- 用户"嗯"单字：不写字段，需要看上下文语气；无法判断就留 followup。
- 用户"改天再说"：视为拒绝，写 call_accepted=refused，node_action=terminal。
- 用户"你找谁"：反问，仍是 followup；Speech 需要先说明来意再重问是否方便。
