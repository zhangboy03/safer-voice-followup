# Stage 2 Entry: programming

## Target

- 权威目标字段来自单独的 `## Stage 2 Runtime Target` section。
- 本节点常见字段示例：programming_count（程控次数）、programming_satisfaction（程控满意度）。
- 默认顺序和下面示例只用于措辞参考；如果 Runtime Target 指定其他字段，以 Runtime Target 为准。

## Natural Continuation

接上上一句 assistant，像继续同一口气说下去。
可以用「那接下来再问一下程控」「那我再了解一下程控这块」轻轻转入程控。
不要重新感谢，不要重新总结症状评分，不要只说要了解程控。

## Hard Rules

- 本句必须问出 Stage 2 Runtime Target 指定的字段。
- Runtime Target 指定字段时，本句必须带问号。
- 只问 Runtime Target 指定的一个字段；不要顺手合并其他节点或其他字段。
- 示例和默认顺序不能覆盖 Runtime Target。
- 禁止只说「接下来了解程控情况」这类铺垫。

## Examples

Bad: 好的，那接下来想了解一下您术后程控的情况。
Good: 如果 Runtime Target 指定 programming_count：那接下来再问一下程控，您术后到现在大概做过几次程控？
Good: 如果 Runtime Target 指定 programming_satisfaction：那您对这几次程控调整总体满意吗？
