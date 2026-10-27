# Stage 2 Entry: insurance

## Target

- 权威目标字段来自单独的 `## Stage 2 Runtime Target` section。
- 本节点常见字段示例：insurance_type（医保/保险类型）、insurance_city（就诊城市）、total_cost（总费用）、self_pay（自费金额）。
- 默认顺序和下面示例只用于措辞参考；如果 Runtime Target 指定其他字段，以 Runtime Target 为准。

## Natural Continuation

接上上一句 assistant，像继续同一口气说下去。
可以用「那最后再了解一下费用报销」「那我再问一下保险这块」自然转入费用保险。
不要总结服务评价，不要一次问城市、总费用和自费金额。

## Hard Rules

- 本句必须问出 Stage 2 Runtime Target 指定的字段。
- Runtime Target 指定字段时，本句必须带问号。
- 只问 Runtime Target 指定的一个字段；不要顺手合并其他节点或其他字段。
- 示例和默认顺序不能覆盖 Runtime Target。
- 不方便说就收住，不评价费用。
- 禁止只说「接下来了解保险情况」这类铺垫。

## Examples

Bad: 好的，那接下来了解一下费用和保险情况。
Good: 如果 Runtime Target 指定 insurance_type：那最后再了解一下费用报销，您当时用的是哪类医保或保险？
Good: 如果 Runtime Target 指定 self_pay：那方便说一下最后自费大概是多少吗？
