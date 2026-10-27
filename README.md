# SAFER: Voice-based Postoperative Follow-up

Research code for a voice agent that supports structured postoperative follow-up. The system combines low-latency speech interaction, a separate task-judging process for checklist completion, and knowledge-grounded question answering with clarification and human handoff.

## Repository

- `src/` — conversation flows, speech processing, knowledge retrieval, and task state.
- `server/` — API and realtime session services.
- `config/prompts/` — conversational prompt templates.
- `frontend/` — React/Vite conversation interface and development console.

This is a sanitized research snapshot. Patient records, the company knowledge base, production credentials, deployment configuration, and private development history are excluded. The included knowledge-base files are empty, and embedded proprietary knowledge entries have been removed from the prompts. Configure your own authorized knowledge and integration credentials before running a complete call. The presentation and its media are hosted separately and are not included here.

## Getting started

Python 3.12 and Node.js are required. Copy `.env.example` to `.env` and configure the necessary services. Install Python dependencies with `uv sync` and frontend dependencies with `cd frontend && npm ci`. The interface can be built with `npm run build` from `frontend/`. A functioning voice session additionally requires configured model providers, LiveKit and a running backend. This snapshot has not been validated for clinical use.

## License

Source code is released under the MIT License; see [LICENSE](LICENSE). Third-party dependencies retain their own licenses.
