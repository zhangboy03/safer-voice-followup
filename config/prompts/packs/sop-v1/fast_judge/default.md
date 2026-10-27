# 节点：default（无节点专属规则时的回退）

## 节点目标
当当前节点未提供专属 `.md` 时，按 `<context>` 中的 `required_fields` 通用判定，给 Speech 留一个保守方向。

## 默认成功 criteria
- advance：`required_fields` 全部已被用户明确回答，或被明确拒答（status=refused）。
- followup：仍有必填字段缺失；missing_fields 写明仍缺的字段。
- stay：本句没有为任何字段提供证据，且 Speech 已问出本节点问题，等待用户继续回答。

## 默认 speech_directive
- 用 `code="generic_ask_required"`，`text` 描述意图："还缺 X 字段；下一步只在本节点内追问 X，禁止越界"。
- 字段已齐时用 `code="generic_advance"`，`text` 描述意图："本节点字段已齐，下一段只做最小过渡，禁忌以'好的/嗯'单独开头"。
