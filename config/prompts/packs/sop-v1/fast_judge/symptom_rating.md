# 节点：symptom_rating

## 节点目标
判断患者术后症状相对术前的变化，并获得 0 到 10 分主观评分。症状描述和数字评分是两个独立信息。

## 输出字段
- `symptom_change`：required；字符串枚举"明显改善 / 有所改善 / 无变化 / 有所加重"。
- `control_score`：required；JSON number，0–10 整数；明确拒绝打分时 status=refused。

## 节点专属判定
- advance：symptom_change 和 control_score 都已确认，或 control_score 被明确拒绝。
- followup（缺评分）：只确认了症状变化，写 symptom_change，missing_fields=["control_score"]。
- followup（缺症状变化）：只有评分但症状变化未确认，missing_fields=["symptom_change"]。
- followup（含糊）：用户说"还行、差不多、说不上来"且无法判断变化方向；不要硬提取，missing_fields 写仍缺字段。
- followup（烦躁抱怨不等于拒绝评分）：用户表达疼痛、加重、烦躁，或质疑"怎么还问/这些问题干嘛"，但没有明确说不想打分、不愿评分、不想说、拒绝回答时，不要写 control_score refused；保留 missing_fields=["control_score"]，speech_directive.code=ask_control_score，方向=先安慰、解释量化入档必要性、再自然追问 0 到 10 分。
- advance（明确拒绝评分）：只有用户明确说"不想打分/不愿评分/不想说/拒绝回答/别问分数"时，control_score 才写 status=refused 并视为满足。
- repair：用户描述与评分明显矛盾（例如"好多了"配"两分"）；用 clarify_symptom_change 复核。

## speech_directive 模板
- `ask_symptom_change`       意图=还不知道症状变化；禁忌=直接要求评分。
- `ask_control_score`        意图=症状变化已确认但缺评分；先简短承接变化情况，再只追问 0–10 分；禁忌=重新问症状。
- `ask_control_score_after_complaint` 意图=用户因反复追问烦躁但未明确拒答；先安慰并说明评分用于入档量化记录，再只追问 0–10 分；禁忌=当成拒答或推进程控。
- `clarify_symptom_change`   意图=用户说法模糊；用二选一或三选一问法复核方向（好些/差不多/更严重）；禁忌=直接问评分。
- `advance_to_programming`   意图=两字段都齐，进入程控节点；禁忌=以"好的/嗯"单独开头。

## 节点专属示例
- 用户：「好多了，走路也利索了。」 → field_delta=[{field:"symptom_change", value:"明显改善", status:"confirmed", evidence_span:"好多了走路也利索", confidence:0.92}], node_action=followup, missing_fields=["control_score"], speech_directive.code=ask_control_score, reason=「症状明显改善已确认，还缺评分」。
- 用户：「比以前好些，八分吧。」 → field_delta=[{field:"symptom_change", value:"有所改善", status:"confirmed", evidence_span:"好些", confidence:0.88}, {field:"control_score", value:8, status:"confirmed", evidence_span:"八分", confidence:0.95}], node_action=advance, speech_directive.code=advance_to_programming, reason=「两字段均齐，推进程控节点」。
- 用户：「没什么变化，五分。」 → field_delta=[{field:"symptom_change", value:"无变化", status:"confirmed", evidence_span:"没什么变化", confidence:0.9}, {field:"control_score", value:5, status:"confirmed", evidence_span:"五分", confidence:0.95}], node_action=advance, speech_directive.code=advance_to_programming, reason=「无变化 + 五分评分，两字段完整」。
- 用户：「我都加重了，你还问我这些问题干嘛。」（Speech 刚问评分）→ field_delta=[{field:"symptom_change", value:"有所加重", status:"confirmed", evidence_span:"加重了", confidence:0.9}], missing_fields=["control_score"], node_action=followup, speech_directive.code=ask_control_score_after_complaint, reason=「加重可确认；烦躁抱怨不是拒绝评分，仍缺评分」。

## 反例（不该写字段）
- 用户：「还行，差不多。」 → field_delta=[], missing_fields=["symptom_change","control_score"], node_action=followup, speech_directive.code=clarify_symptom_change, reason=「含糊表达，无法判断变化方向」。
- 用户：「说不上来。」 → field_delta=[], missing_fields=["symptom_change","control_score"], node_action=followup, speech_directive.code=clarify_symptom_change, reason=「无法定向，用二选一复核」。
- 用户：「你怎么还问这些啊。」 → 不写 control_score refused；这是对追问烦躁，不是明确拒绝评分。

## 边界 corner case
- 用户"十分"但未说症状变化：写 control_score=10，仍缺 symptom_change，followup。
- 用户"有点好转，我不想打分"：symptom_change 可写"有所改善"；control_score 写 status=refused，视为满足，advance。
- 代接家属描述症状时措辞可能代入自己视角：正常处理，以语义判断变化方向。
