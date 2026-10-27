# Stage 2 Entry: symptom_rating

## Target

- 权威目标字段来自单独的 `## Stage 2 Runtime Target` section。
- 本节点常见字段示例：symptom_change（症状变化）、control_score（0-10 分控制评分）。
- 默认顺序和下面示例只用于措辞参考；如果 Runtime Target 指定其他字段，以 Runtime Target 为准。

## Natural Continuation

接上上一句 assistant，像继续同一口气说下去。
可以用「那我接着了解一下恢复情况」「那再问一下术后感觉」自然转入症状变化。
不要重新感谢，不要总结身份信息，不要脱离 Runtime Target 自行换字段。

## Hard Rules

- 本句必须问出 Stage 2 Runtime Target 指定的字段。
- Runtime Target 指定字段时，本句必须带问号。
- 只问 Runtime Target 指定的一个字段；不要顺手合并其他节点或其他字段。
- 示例和默认顺序不能覆盖 Runtime Target。
- 禁止只说「接下来了解恢复情况」这类铺垫。

## Examples

Bad: 好的，那接下来了解一下您术后恢复情况。
Good: 如果 Runtime Target 指定 symptom_change：那我接着了解一下恢复情况，跟手术前比，现在是好一些、差不多，还是更重了？
Good: 如果 Runtime Target 指定 control_score：那按 0 到 10 分来算，您现在症状控制大概能打几分？
