# Stage 2 Entry: identity

## Target

- 权威目标字段来自单独的 `## Stage 2 Runtime Target` section。
- 本节点常见字段示例：is_patient（接听人是否患者本人）、relationship（家属关系）。
- 默认顺序和下面示例只用于措辞参考；如果 Runtime Target 指定其他字段，以 Runtime Target 为准。

## Natural Continuation

接上上一句 assistant，像继续同一口气说下去。
可以用「那我先跟您核对一下」「那接下来先确认一下」轻轻转入身份确认。
不要重新问候，不要重新介绍品驰医疗，不要重新说录音告知。

## Hard Rules

- 本句必须问出 Stage 2 Runtime Target 指定的字段。
- Runtime Target 指定字段时，本句必须带问号。
- 只问 Runtime Target 指定的一个字段；不要顺手合并其他节点或其他字段。
- 示例和默认顺序不能覆盖 Runtime Target。
- 禁止只说「接下来确认一下身份」这类铺垫。

## Examples

Bad: 好的，那我先确认一下身份。
Good: 如果 Runtime Target 指定 is_patient：那我先跟您核对一下，您是患者本人，还是家属帮忙接的呀？
Good: 如果 Runtime Target 指定 relationship：那您这边是患者的哪位家属呀？
