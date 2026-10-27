# [B1] / Escalation 固定话术（共五条，禁止改写）

## Type 1 — LEGITIMATE × medical_judgment（含回电承诺，NORMAL 优先级）
这个问题我会请专人之后给您回个电话，咱们先继续后面的随访。

## Type 2 — kb_out_of_coverage / adversarial_or_pii / COERCED / INFRA_FALLBACK（不承诺回电）
这个问题我这边暂时没办法回答您，咱们先继续把今天的随访做完。

## Type 3 — LEGITIMATE × safety_risk（含 priority=high 回电；rollout-gated 等 HIGH-consumer SLA）
说明情况已经记录，会有专人给你回电。

## Type E1 — EscalationKind.DEPTH_CAP（含回电承诺，NORMAL 优先级）
您这几个问题专人会回个电话跟您详细说一下，咱们先继续后面的随访。

## Type E2 — EscalationKind.EXIT_ESCALATE（含回电承诺，NORMAL 优先级）
这个让医生之后给您回个电话详细聊聊，咱们先继续后面的随访。
