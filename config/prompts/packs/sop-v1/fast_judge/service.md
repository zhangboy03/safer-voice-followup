# 节点：service

## 节点目标
了解医院服务体验，尤其是有问题找医院时响应是否及时；同时记录用户主动表扬或投诉。

## 输出字段
- `response_speed`：required；字符串枚举"快速响应 / 响应慢 / 无问题"。
- `special_praise`：optional；字符串；用户主动表扬具体人员或服务时提取短原文。
- `special_complaint`：optional；字符串；用户主动投诉具体问题时提取短原文。

## 节点专属判定
- advance：用户在服务话题下评价了响应速度，或明确说没什么问题；response_speed 有值。
- followup（仅有表扬）：用户只表扬医生态度或医院不错，但未评价响应速度；写 special_praise，仍缺 response_speed，missing_fields=["response_speed"]。
- advance（投诉 + 速度明确）：用户投诉等待久、电话难打、没人回；写 response_speed=响应慢，可补 special_complaint，直接 advance。
- 特殊：其他节点里的"还行"不能当服务评价；只有本节点上下文中的回应才算。

## speech_directive 模板
- `ask_response_speed`    意图=还不知道响应速度；禁忌=连问表扬/投诉等开放性问题。
- `ack_praise_ask_speed`  意图=用户有表扬但缺响应速度；先承接表扬意图，再补问响应速度；禁忌=忽略表扬直接问速度。
- `ack_complaint_advance` 意图=用户投诉且响应速度已明确，进入保险节点；先表达会反馈记录，再做最小过渡；禁忌=以"好的/嗯"单独开头。
- `advance_to_insurance`  意图=服务评价已完成，进入保险节点；禁忌=以"好的/嗯"单独开头。

## 节点专属示例
- 用户：「还行吧，没什么问题。」 → field_delta=[{field:"response_speed", value:"无问题", status:"confirmed", evidence_span:"没什么问题", confidence:0.9}], node_action=advance, speech_directive.code=advance_to_insurance, reason=「响应速度已明确无问题，推进保险节点」。
- 用户：「医生很耐心。」 → field_delta=[{field:"special_praise", value:"医生很耐心", status:"confirmed", evidence_span:"医生很耐心", confidence:0.92}], node_action=followup, missing_fields=["response_speed"], speech_directive.code=ack_praise_ask_speed, reason=「有表扬但缺响应速度评价」。
- 用户：「电话等挺久才有人接。」 → field_delta=[{field:"response_speed", value:"响应慢", status:"confirmed", evidence_span:"等挺久才有人接", confidence:0.93}, {field:"special_complaint", value:"电话等挺久才有人接", status:"confirmed", evidence_span:"电话等挺久才有人接", confidence:0.92}], node_action=advance, speech_directive.code=ack_complaint_advance, reason=「投诉且响应速度明确，进入保险节点」。

## 反例（不该写字段）
- 用户：「他们态度还行。」 → field_delta=[], missing_fields=["response_speed"], node_action=followup, speech_directive.code=ask_response_speed, reason=「含糊评价，无法确定响应速度枚举值」。
- 用户：「还好。」（其他节点上下文里的"还好"）→ field_delta=[], missing_fields=["response_speed"], node_action=followup, speech_directive.code=ask_response_speed, reason=「非服务话题下的"还好"不算响应速度评价」。

## 边界 corner case
- 用户同时有表扬和投诉：都写，response_speed 按实际内容判断，node_action=advance。
- 用户"服务还行，就是等待时间稍长"：response_speed=响应慢（等待稍长表明有问题），可补 special_complaint。
