# Brain Playground

Interactive, local, text-only loop over the real study2_v3 brain (`run_journey` /
`run_spa_journey`). You type as the patient; it shows assistant text, field/state
commits, flow transitions, and brain-only latency — no voice pipeline. See the spec:
`docs/superpowers/specs/2026-06-07-brain-playground-design.md`.

## Run

```bash
.venv/bin/python -m src.playground.server                       # ours, qwen3.6-flash
.venv/bin/python -m src.playground.server --arch spa --model qwen3.7-plus
```

Open the printed `http://127.0.0.1:8920/?token=...` (loopback + per-process token only —
needs `DASHSCOPE_API_KEY` in the environment for the brain to actually reply).

Controls: switch arch (ours / spa) + model, reset/new session, end call. The right panel
shows, per turn: SOP node transition, judge decision, field commits ("tool calls"), and a
latency breakdown (stage1 TTFT / judge / stage2 / brain total).

## Fidelity (read this)

Faithful to the brain's decision logic + the **study2 eval loop** — NOT the live
`llm_processor._run_inference_flow` orchestration (judge-timeout fallback, speculative turns,
followup-cap timing, background schedulers). Latency shown is the eval-loop brain core path;
add **~300ms ASR + ~200ms TTS** to imagine voice E2E. Validate finalists on `study2_v3` + a
live ECS call before trusting voice behavior.

## Tests

```bash
.venv/bin/python -m pytest tests/unit/playground/ -q
```
