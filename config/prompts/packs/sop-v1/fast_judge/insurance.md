# 节点：insurance

## 节点目标
了解医保类型和报销后自付金额；用户不愿说时尊重拒答，不要施压追问。

## 输出字段
- `insurance_type`：required；字符串；如"城镇职工医保 / 城乡居民医保 / 新农合 / 自费 / 惠民保"。
- `self_pay`：required；字符串；保留用户口语金额，如"五万 / 两三万"；拒答时 status=refused。
- `insurance_city`、`total_cost`、`huimin_bao`：optional；用户主动说到才写。

## 节点专属判定
- advance：insurance_type 和 self_pay 都有值；或两个必填都已按拒答处理（status=refused）。
- followup（缺类型）：用户只说"有医保"但未说明类型；不写 insurance_type，missing_fields=["insurance_type"]。
- followup（缺金额）：用户说了类型但未说自付金额；missing_fields=["self_pay"]。
- repair：用户给的类型不在枚举里且语义模糊（如"商业的"）；先复核一次再决定写不写。

## speech_directive 模板
- `ask_insurance_type`      意图=还不知道医保类型；语气要轻；禁忌=施压追问、连问自付金额。
- `ask_self_pay`            意图=类型已知但缺自付金额；先简短承接类型，再问报销后大致自付，并明示用户不方便就可以不说；禁忌=重复问类型。
- `accept_refusal_advance`  意图=用户不愿说且必填已按拒答处理，进入收尾节点；先表达接受意图（没关系/不勉强），再做最小过渡；禁忌=以"好的/嗯"单独开头、继续追问。
- `advance_to_closing`      意图=两个必填都齐，进入收尾节点；禁忌=以"好的/嗯"单独开头。

## 节点专属示例
- 用户：「城镇职工的，自己花了大概五万。」 → field_delta=[{field:"insurance_type", value:"城镇职工医保", status:"confirmed", evidence_span:"城镇职工", confidence:0.95}, {field:"self_pay", value:"五万", status:"confirmed", evidence_span:"五万", confidence:0.95}], node_action=advance, speech_directive.code=advance_to_closing, reason=「两必填齐，推进收尾节点」。
- 用户：「用的是惠民保。」 → field_delta=[{field:"insurance_type", value:"惠民保", status:"confirmed", evidence_span:"惠民保", confidence:0.95}], node_action=followup, missing_fields=["self_pay"], speech_directive.code=ask_self_pay, reason=「类型已知，还缺自付金额」。
- 用户：「这个不想说。」（上下文 Speech 正在问 self_pay）→ field_delta=[{field:"self_pay", value:"", status:"refused", evidence_span:"不想说", confidence:0.95}], node_action=followup, missing_fields=["insurance_type"], speech_directive.code=ask_insurance_type, reason=「自付按拒答处理，仍缺类型」。

## 反例（不该写字段）
- 用户：「有医保。」 → field_delta=[], missing_fields=["insurance_type","self_pay"], node_action=followup, speech_directive.code=ask_insurance_type, reason=「类型不明，不能硬写枚举值」。
- 用户：「商业的吧。」 → field_delta=[], missing_fields=["insurance_type","self_pay"], node_action=repair, speech_directive.code=ask_insurance_type, reason=「不在枚举里，需复核」。

## 边界 corner case
- 用户同时拒绝两个字段："这些我都不想说"：两个必填都写 status=refused，node_action=advance，speech_directive.code=accept_refusal_advance。
- 用户只说"自费"：insurance_type=自费，仍要问 self_pay（自费时也有自付金额）。
- 用户给区间金额"三到五万"：取用户口语原文，self_pay="三到五万"，不要截断。
