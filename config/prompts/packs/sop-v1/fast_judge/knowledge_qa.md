# 节点：knowledge_qa

## 节点目标
判断知识问答插入是否结束，决定是否回到主线随访。本节点不收集 SOP 字段，也不要把用户继续提问当成新的 divert。

## 节点专属判定
- advance：用户明确"没有了 / 没问题 / 继续吧 / 你接着问 / 回到刚才那个问题"。
- stay：用户继续追问产品使用、标准知识库范围内问题，或追问刚才回答。
- handoff：用户提出个体化医疗判断、用药、治疗方案、安全风险等需要人工处理的问题。

## speech_directive 模板
- `already_asked_next_node`     意图=Speech 上一段已经越界问出了下一节点的问题；下一步进入目标节点，但只承接、不重复那个问题；禁忌=自行复述刚问过的问题。
- `continue_knowledge_answer`   意图=用户还在问知识问题；下一步继续按知识库规则回答，结尾询问是否还有其他想了解的；禁忌=自行给医疗判断、强行结束话题。
- `resume_main_sop`             意图=用户表示没有问题或想继续随访，下一步回到主线节点；本段已由 Stage 1 承接，下一段只做最小过渡；禁忌=以"好的/嗯"单独开头。
- `handoff_medical_question`    意图=用户问题需要医生或人工处理；下一步说明建议咨询医生或转人工；禁忌=自行给医疗判断、绕开人工。

## 节点专属示例
- 用户：「没有了，你继续问吧。」 → field_delta=[], node_action=advance, speech_directive.code=resume_main_sop, reason=「知识问答结束，回主线」。
- 用户：「这个充电器怎么开机？」 → field_delta=[], node_action=stay, speech_directive.code=continue_knowledge_answer, reason=「仍在问产品操作，留在知识问答」。
- 用户：「那我这个情况要不要吃药？」 → field_delta=[], node_action=handoff, speech_directive.code=handoff_medical_question, reason=「个体化医疗判断，建议人工或医生」。

## 反例（不该 advance）
- 用户：「嗯。」 → field_delta=[], node_action=stay, speech_directive.code=continue_knowledge_answer, reason=「纯应答词，不能视为知识问答结束」。
