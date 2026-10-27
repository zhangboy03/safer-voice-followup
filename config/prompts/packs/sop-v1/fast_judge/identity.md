# 节点：identity

## 节点目标
判断接听人是不是患者本人，并在非本人时记录关系。不要把纯应答词当身份确认。

## 输出字段
- `is_patient`：required；JSON boolean true / false。
- `relationship`：optional；字符串；只有 is_patient=false 时才写，value 用"哥哥、女儿、老伴、护工"等短关系词。

## 节点专属判定
- advance（is_patient=true）：用户明确说自己是本人、就是我、我是患者。
- advance（is_patient=false）：用户明确说自己是家属、哥哥、女儿、老伴、护工或"他/她不方便接"；尽量写 relationship。
- followup：用户只说"嗯、对、好的、在听"，没有身份信息；不写 is_patient，missing_fields=["is_patient"]。
- repair：用户语义矛盾或含糊（"我就是""差不多就是"）；用 reconfirm_identity 再确认一次，不要当作新问题。

## speech_directive 模板
- `ask_identity`          意图=身份仍未确认；禁忌=问症状或其他后续节点信息。
- `advance_patient_self`  意图=已确认本人，进入症状评估；禁忌=以"好的/嗯"单独开头、继续问关系。
- `advance_proxy_family`  意图=已确认代接人，进入症状评估并以代接人视角询问；禁忌=把接听人称作患者本人。
- `reconfirm_identity`    意图=用户给的是纯应答词或语义矛盾，换说法在本节点内再确认一次；禁忌=当作全新问题重复首次提问。

## 节点专属示例
- 用户：「是我。」 → field_delta=[{field:"is_patient", value:true, status:"confirmed", evidence_span:"是我", confidence:0.95}], node_action=advance, speech_directive.code=advance_patient_self, reason=「用户明确本人接听，推进到症状评估」。
- 用户：「我是患者哥哥。」 → field_delta=[{field:"is_patient", value:false, status:"confirmed", evidence_span:"患者哥哥", confidence:0.98}, {field:"relationship", value:"哥哥", status:"confirmed", evidence_span:"哥哥", confidence:0.98}], node_action=advance, speech_directive.code=advance_proxy_family, reason=「代接人身份明确，关系为哥哥」。
- 用户：「他现在不在，我是他女儿。」 → field_delta=[{field:"is_patient", value:false, status:"confirmed", evidence_span:"他不在", confidence:0.95}, {field:"relationship", value:"女儿", status:"confirmed", evidence_span:"女儿", confidence:0.98}], node_action=advance, speech_directive.code=advance_proxy_family, reason=「患者不在场，代接人为女儿」。

## 反例（不该写字段）
- 用户：「嗯，对。」 → field_delta=[], missing_fields=["is_patient"], node_action=followup, speech_directive.code=ask_identity, reason=「纯应答词，不能证明身份」。
- 用户：「在听呢。」 → field_delta=[], missing_fields=["is_patient"], node_action=followup, speech_directive.code=ask_identity, reason=「仅表示在线，无身份信息」。
