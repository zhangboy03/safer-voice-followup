# 节点：education

## 节点目标
确认设备使用方法是否被医生说明过。单字段节点。

## 输出字段
- `device_explained`：required；JSON boolean true / false。

## 节点专属判定
- advance：device_explained 有明确结果（讲过=true / 没讲清或没人教=false / 记不清也按 false 收口）。
- followup（缺 device_explained）：用户没有正面回答设备说明，missing_fields=["device_explained"]。
- repair：用户说"好的知道了"但不清楚是否在回答设备说明；用 clarify_education_answer 点明问题再确认。
- 特殊：用户对当前客服讲解说"好的知道了"，不等于设备说明被讲过；不写字段。

## speech_directive 模板
- `ask_device_explained`     意图=还不知道设备说明情况；禁忌=问其他节点信息。
- `clarify_education_answer` 意图=用户只说"知道了/讲过了"但不清楚对应设备说明；点明问题重新确认；禁忌=含糊带过。
- `advance_to_service`       意图=device_explained 已确认，进入服务节点；禁忌=以"好的/嗯"单独开头。

## 节点专属示例
- 用户：「设备讲过了。」 → field_delta=[{field:"device_explained", value:true, status:"confirmed", evidence_span:"讲过了", confidence:0.88}], node_action=advance, speech_directive.code=advance_to_service, reason=「设备说明已确认，推进服务节点」。
- 用户：「设备怎么用没人教过。」 → field_delta=[{field:"device_explained", value:false, status:"confirmed", evidence_span:"没人教过", confidence:0.88}], node_action=advance, speech_directive.code=advance_to_service, reason=「设备说明=false 也是明确结果，推进服务节点」。
- 用户：「没太听明白。」（上下文 Stage 1 刚问设备使用方法）→ field_delta=[{field:"device_explained", value:false, status:"confirmed", evidence_span:"没太听明白", confidence:0.8}], node_action=advance, speech_directive.code=advance_to_service, reason=「设备没讲清=false，是明确结果，推进」。
- 用户：「嗯，都说了。」（上下文：Stage 1 紧接着在问「设备使用方法医生讲过吗」，这句更像在应付你客服当前的讲解，没真正回答设备）→ field_delta=[], node_action=followup, missing_fields=["device_explained"], speech_directive.code=clarify_education_answer, reason=「Stage 1 正在问设备说明=它还没答，不能 advance」。
  —— 与上一条的区别：上一条「没太听明白」是对设备的明确否定回答（=false），可 advance；这一条「都说了」含糊、像在回应你的讲解，device_explained 还没真正答，不能 advance。

## 反例（不该写字段）
- 用户：「好的知道了。」（回应客服当前说明）→ field_delta=[], missing_fields=["device_explained"], node_action=followup, speech_directive.code=clarify_education_answer, reason=「回应的是客服讲解，不代表设备已说明」。

## 边界 corner case
- 用户"都没讲"：device_explained=false，advance。
- 用户"设备怎么用记不太清了/没什么印象"：记不清按 false 收口（confirmed），advance；不要因为是 boolean 字段就反复追问，避免卡在同一节点。
