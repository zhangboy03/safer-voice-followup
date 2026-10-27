# 节点：programming

## 节点目标
了解术后做过程控的次数；如果做过至少一次，还要了解对程控效果是否满意。

## 输出字段
- `programming_count`：required；JSON number；明确没做过写 0。
- `programming_satisfaction`：条件必填；字符串枚举"满意 / 不满意"；只有 programming_count > 0 时需要；用户拒绝评价时 status=refused。

## 节点专属判定
- advance（count=0）：用户明确没做过程控，programming_count=0，不再问满意度。
- advance（count>0）：次数和满意度都确认，或满意度被明确拒绝。
- followup（缺次数）：用户只说"做过"但未给次数；missing_fields=["programming_count"]。
- followup（缺满意度）：次数已知且 count>0，但满意度未说；missing_fields=["programming_satisfaction"]。
- followup（次数模糊）：用户说"做过几次"但未给数字；不写字段，追问具体次数。

## speech_directive 模板
- `ask_programming_count`        意图=还不知道程控次数；禁忌=解释程控概念。
- `ask_programming_satisfaction` 意图=已知做过至少一次但缺满意度；禁忌=重复问次数。
- `advance_no_programming`       意图=用户没做过程控，进入宣教节点；禁忌=以"好的/嗯"单独开头。
- `advance_to_education`         意图=次数和满意度都齐，进入宣教节点；禁忌=以"好的/嗯"单独开头。

## 节点专属示例
- 用户：「没做过。」 → field_delta=[{field:"programming_count", value:0, status:"confirmed", evidence_span:"没做过", confidence:0.97}], node_action=advance, speech_directive.code=advance_no_programming, reason=「次数为零，无需问满意度，直接进宣教」。
- 用户：「做过两次，效果还行。」 → field_delta=[{field:"programming_count", value:2, status:"confirmed", evidence_span:"两次", confidence:0.95}, {field:"programming_satisfaction", value:"满意", status:"confirmed", evidence_span:"效果还行", confidence:0.8}], node_action=advance, speech_directive.code=advance_to_education, reason=「次数和满意度均确认，推进宣教节点」。
- 用户：「两次。」 → field_delta=[{field:"programming_count", value:2, status:"confirmed", evidence_span:"两次", confidence:0.97}], node_action=followup, missing_fields=["programming_satisfaction"], speech_directive.code=ask_programming_satisfaction, reason=「次数已知，还缺满意度」。

## 反例（不该写字段）
- 用户：「做过。」 → field_delta=[], missing_fields=["programming_count"], node_action=followup, speech_directive.code=ask_programming_count, reason=「只说做过，次数不明，不能写字段」。
- 用户：「程控？」 → field_delta=[], missing_fields=["programming_count"], node_action=followup, speech_directive.code=ask_programming_count, reason=「用户反问，未给任何信息」。

## 边界 corner case
- 用户"大概两三次"：取区间下限 2，写 programming_count=2，并追问满意度。
- 用户"做过，不记得几次了"：次数不明，不写 programming_count，followup 追问具体次数。
- 用户"不满意，换了参数"：programming_satisfaction=不满意，记录即可，不追问原因。
