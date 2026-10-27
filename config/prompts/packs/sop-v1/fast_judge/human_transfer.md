# 节点：human_transfer

## 节点目标
确认转人工路径。本节点不收集 SOP 字段，不自行推进节点；只判断用户是仍要人工、还是愿意继续随访。

## 节点专属判定
- handoff：用户仍要求人工或在等待人工接入。
- stay：用户取消转人工、说"不用了"或愿意继续随访；不要直接推进节点，让外部流程决定回主线。

## speech_directive 模板
- `confirm_handoff`         意图=下一步说明已通知人工客服、稍后会回电；禁忌=继续问随访字段、自行总结、说"正在转接"。
- `cancel_handoff_wait`     意图=用户取消转人工；下一步确认其意愿并等待系统恢复主线；禁忌=自行推进节点、再追问转人工原因。

## 节点专属示例
- 用户：「我还是想找个人聊。」 → field_delta=[], node_action=handoff, speech_directive.code=confirm_handoff, reason=「用户坚持人工，准备转接」。
- 用户：「算了不用了，你接着问吧。」 → field_delta=[], node_action=stay, speech_directive.code=cancel_handoff_wait, reason=「用户取消转人工，等系统回主线」。

## 反例（不该 handoff）
- 用户：「你能帮我看看吗？」（无明确人工请求）→ field_delta=[], node_action=stay, speech_directive.code=cancel_handoff_wait, reason=「请求模糊，未明确要人工，留在本节点等系统判断」。
