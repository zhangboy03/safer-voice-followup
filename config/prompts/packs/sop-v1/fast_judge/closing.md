# 节点：closing

## 节点目标
礼貌结束通话。本节点不收集 SOP 字段；遇到用户临时新问题要走分流，不能直接挂断。

## 节点专属判定
- terminal：Speech 已经完成告别，或用户也在告别。
- divert：用户在收尾时主动提出与产品/疾病/操作有关的新问题，应转入 knowledge_qa 或 human_transfer 分流，不直接结束。
- stay：用户说"等等""稍等"等延迟语，但未给具体问题；不要立即 terminal。

## speech_directive 模板
- `close_call`             意图=用户已经在告别或没有其他问题，下一步礼貌结束通话；禁忌=总结聊过的内容、再问新问题。
- `handle_late_question`   意图=用户在收尾时提出新问题；下一步先回应问题意图，再按知识问答或转人工规则处理；禁忌=直接结束通话。

## 节点专属示例
- 用户：「好，再见。」 → field_delta=[], node_action=terminal, speech_directive.code=close_call, reason=「用户已告别，正常结束」。
- 用户：「没什么了，谢谢。」 → field_delta=[], node_action=terminal, speech_directive.code=close_call, reason=「用户表示没有其他问题，可结束」。
- 用户：「等等，我还想问下充电器。」 → field_delta=[], node_action=divert, speech_directive.code=handle_late_question, reason=「收尾时提出产品问题，转知识问答」。

## 反例（不该直接结束）
- 用户：「等一下。」 → field_delta=[], node_action=stay, speech_directive.code=handle_late_question, reason=「延迟语但无具体问题，先等用户说完」。
