const token = document.querySelector('meta[name=pg-token]').content;
const arch0 = document.querySelector('meta[name=pg-arch]').content;
document.getElementById('arch').value = arch0;

const messages = document.getElementById('messages');
const traceBody = document.getElementById('traceBody');
const statusEl = document.getElementById('status');

const CATALOG = window.PG_CATALOG || [];
const formBody = document.getElementById('formBody');
const formCount = document.getElementById('formCount');
const formState = {};
function renderForm() {
  let html = '';
  let group = null;
  for (const f of CATALOG) {
    if (f.group !== group) {
      group = f.group;
      html += `<div class="fgroup">${group}</div>`;
    }
    const s = formState[f.name] || { status: 'pending', value: '' };
    const star = f.required ? '<span class="star">★</span>' : '';
    const label = { pending: '待填', confirmed: '已确认', refused: '已拒答' }[s.status] || '待填';
    html += `<div class="frow">${star}<span>${f.name}</span><span class="pill ${s.status}" title="${(s.evidence || '').replace(/"/g, '')}">${label}</span><span class="fval">${s.value || ''}</span></div>`;
  }
  formBody.innerHTML = html || '（无字段）';
  const req = CATALOG.filter((f) => f.required);
  const done = req.filter((f) => (formState[f.name] || {}).status === 'confirmed').length;
  formCount.textContent = req.length ? `已确认 ${done}/${req.length} 必填` : '';
}
if (CATALOG.length) renderForm();

function addMsg(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = (role === 'you' ? '你：' : '客服：') + text;
  messages.appendChild(d);
  messages.scrollTop = messages.scrollHeight;
}

function bar(label, ms) {
  if (ms == null) return '';
  const w = Math.min(300, Math.round(ms));
  return `<div class="bar"><span>${label} ${Math.round(ms)}ms</span><i style="width:${w}px"></i></div>`;
}

function renderTrace(ev) {
  const tl = ev.timeline || {};
  const fields = (ev.accepted_fields || [])
    .map((f) => `<li><b>${f.field}</b> = ${JSON.stringify(f.value)} <em>(${f.status})</em></li>`)
    .join('');
  traceBody.innerHTML = `
    <div class="flow">${ev.node_before} → <b>${ev.node_after}</b> · 决策 <b>${ev.fast_judge_action}</b></div>
    <h4>工具 / 字段提交</h4><ul>${fields || '<li>（无）</li>'}</ul>
    <h4>延迟</h4>
    ${bar('stage1 TTFT', tl.stage1_ttft_ms)}
    ${bar('stage1 total', tl.stage1_total_ms)}
    ${bar('judge', tl.judge_total_ms)}
    ${bar('stage2 TTFT', tl.stage2_ttft_ms)}
    ${bar('大脑总计', tl.brain_total_ms)}
    <div class="note">+ ~300ms ASR + ~200ms TTS ≈ 估算语音 E2E</div>`;
}

function renderSoloTrace(ev) {
  const hasTool = ev.tool_calls && ev.tool_calls.length;
  const calls = (ev.tool_calls || [])
    .map((c) => `<div class="tcall">🔧 query_knowledge_base("${c.query}") → ${c.result}</div>`)
    .join('');
  traceBody.innerHTML = `
    <div class="flow">单次 flash 调用${hasTool ? '（含知识库工具）' : ''}</div>
    ${calls}
    <h4>延迟${hasTool ? '（含知识库查询往返）' : '（感知 = 仅 speech）'}</h4>
    ${bar('speech TTFT', ev.speech_ttft_ms)}
    ${bar('speech total', ev.speech_total_ms)}
    <div class="note">${
      hasTool
        ? '安抚话术已先行播报（见上一条），患者感知到的等待 = 那句安抚的延迟'
        : '抽取器在后台跑（见问卷面板），不计入感知延迟'
    }</div>`;
}

function renderSoloAck(ev) {
  traceBody.innerHTML = `
    <div class="flow">🔎 触发知识库查询：先安抚，再调用工具</div>
    <h4>延迟（感知 = 仅这句安抚）</h4>
    ${bar('安抚 TTFT', ev.speech_ttft_ms)}
    ${bar('安抚 total', ev.speech_total_ms)}
    <div class="note">结果回来后会再说一段，用自己的话讲；那一段的延迟见下一条</div>`;
}

let ws;
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws?token=${encodeURIComponent(token)}`);
  ws.onopen = () => (statusEl.textContent = '● 已连接');
  ws.onclose = () => (statusEl.textContent = '○ 已断开');
  ws.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'opening') addMsg('bot', ev.text);
    else if (ev.type === 'assistant_turn') {
      if (ev.kind === 'pretool_ack') {
        addMsg('bot', ev.stage1_text);
        renderSoloAck(ev);
      } else {
        const t = [ev.stage1_text, ev.stage2_text].filter(Boolean).join('\n');
        addMsg('bot', t);
        if (ev.node_after !== undefined) renderTrace(ev);
        else renderSoloTrace(ev);
      }
    } else if (ev.type === 'form_update') {
      for (const f of ev.fields) formState[f.name] = f;
      renderForm();
      if (ev.extractor_ms != null) formCount.title = `后台抽取 ${Math.round(ev.extractor_ms)}ms`;
    } else if (ev.type === 'ended') statusEl.textContent = '○ 通话结束';
    else if (ev.type === 'error') addMsg('bot', '[错误] ' + ev.message);
    else if (ev.type === 'session_state') statusEl.textContent = `● ${ev.arch} / ${ev.model}`;
  };
}
connect();

document.getElementById('inputForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const inp = document.getElementById('input');
  const text = inp.value.trim();
  if (!text || ws.readyState !== 1) return;
  addMsg('you', text);
  ws.send(JSON.stringify({ type: 'user_text', text }));
  inp.value = '';
});
document.getElementById('end').addEventListener('click', () =>
  ws.send(JSON.stringify({ type: 'control', action: 'end_call' })),
);
document.getElementById('reset').addEventListener('click', () => {
  messages.innerHTML = '';
  traceBody.textContent = '等待第一轮…';
  Object.keys(formState).forEach((k) => delete formState[k]);
  if (CATALOG.length) renderForm();
  ws.send(
    JSON.stringify({
      type: 'control',
      action: 'reset',
      arch: document.getElementById('arch').value,
      model: document.getElementById('model').value,
    }),
  );
});
