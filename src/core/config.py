"""配置管理模块"""

from typing import Literal, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """系统配置"""

    dashscope_api_key: str = ""

    # DashScope 网络配置（生产环境可切换为 VPC 内网地址）
    # 公网默认值。VPC 内网: "https://vpc-cn-beijing.dashscope.aliyuncs.com"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com"
    dashscope_http_api_url: str = "https://dashscope.aliyuncs.com/api/v1"
    dashscope_ws_api_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
    dashscope_compatible_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 开发模式配置
    dev_mode: bool = False  # 开发模式允许 Stub 降级和跳过某些验证
    allow_stub_fallback: bool = False  # 显式允许 LLM Stub 降级（无 API Key 时使用固定回复）

    # ASR 配置
    asr_model: str = "fun-asr-realtime-2026-02-28"
    asr_sample_rate: int = 8000
    asr_frame_bytes: int = 320  # 20ms @ 8kHz, 16bit
    asr_language_hints: list[str] = ["zh"]
    asr_semantic_punctuation: bool = False  # [优化] 关闭语义分句，减少延迟
    asr_max_sentence_silence_ms: int = 200  # 必须 >= silence_threshold_ms，避免文本丢失
    asr_vocabulary_id: str = ""  # DashScope 热词列表 ID（通过控制台创建）
    stt_standby_enabled: bool = True  # 会话空闲时预建下一条 ASR stream，降低用户开口后的建连延迟

    # ASR Final 等待（话轮结束时等待 ASR final 的最大时间）
    # 注意：阿里 ASR 的 final 通常需要 200-400ms，设置太短会导致使用不完整的 partial 文本
    asr_final_wait_timeout_ms: int = 350  # 350ms 超时，超时后用 partial

    # TTS 配置
    tts_model: str = "cosyvoice-v3-flash"
    tts_voice: str = "longyingling_v3"
    tts_sample_rate: int = 16000
    tts_format: Literal["pcm"] = "pcm"
    tts_pool_size: int = 2  # Pre-warm N TTS sessions at startup (0=disabled, 2=recommended)
    tts_pool_max_age_s: float = 25.0  # 主动淘汰超龄 session（DashScope duplex task 空闲约30s超时）

    # 轮次检测 - 统一 200ms 静默阈值
    silence_threshold_ms: int = 200  # 静音阈值（话轮结束判定）
    max_turn_duration_ms: int = 30000  # 最大轮次时长（毫秒）

    # 打断检测 + 附和语过滤
    min_interrupt_duration_ms: int = 500  # 用户说话持续 500ms 才视为真正打断
    max_backchannel_suppress_ms: int = 1500  # 附和语最大抑制时长（超时强制打断）

    # 本地 VAD - 可配置参数
    vad_backend: Literal["energy", "pvad"] = "energy"  # 主路径 VAD 后端
    noise_gate: float = 0.05  # 静态噪声门限（仅当 noise_tracking_enabled=False 时使用）
    min_speech_frames: int = 5
    vad_smoothing_alpha: float = 0.3  # EMA 平滑系数
    silero_speech_threshold: float = 0.4  # Silero 判定阈值（restored from 8825d11^）
    silero_speech_hysteresis: float = 0.08  # Silero 滞回（restored from 8825d11^）
    vad_end_silence_frames: int = 10  # 静音帧数阈值（200ms @ 20ms/帧，配合 200ms VAD）

    # ── pVAD shadow integration (spec 2026-05-16) ────────────────────────────
    vad_shadow_backends: list[str] = ["pvad"]  # default ON — flip to [] to disable
    vad_pvad_model_path: str | None = None  # None = HF auto-download
    vad_pvad_threshold: float = 0.7
    vad_pvad_smoothing_alpha: float = 0.8
    vad_pvad_enroll_target_ms: int = 5000
    vad_pvad_enroll_timeout_ms: int = 30000
    vad_pvad_enroll_pre_roll_ms: int = 100
    vad_pvad_encode_timeout_ms: int = 5000
    vad_pvad_max_consecutive_errors: int = 50
    vad_pvad_min_embedding_magnitude: float = 1e-3
    vad_pvad_ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    vad_pvad_preloaded_voiceprint_path: str = ""  # eval-only: .npy target-speaker embedding
    vad_shadow_trace_enabled: bool = True  # dev True, prod False (env override)
    vad_shadow_trace_dir: str = "logs"
    # AND-gate: pvad shadow 进入 READY 后，要求 is_speech=True 才允许
    # emit UserStartedSpeakingFrame；enrollment 期间 / shadow 故障时退回 energy-only
    vad_pvad_gates_primary: bool = True

    # 动态噪声底跟踪
    noise_tracking_enabled: bool = True  # 启用动态噪声底跟踪
    noise_gain_factor: float = 3.0  # 阈值 = noise_floor × gain_factor
    noise_min_threshold: float = 0.02  # 能量 VAD 阈值（保持偏低避免吞首音）
    noise_max_threshold: float = 0.25  # 绝对上限（防止过于迟钝）
    noise_floor_alpha: float = 0.02  # 更新系数（慢更新，50帧≈1秒才显著变化）
    # noise_floor_initial = min_threshold / gain_factor = 0.027（由代码计算）

    # 限流
    asr_rps: int = 20
    tts_rps: int = 20

    # 发送队列 - 需足够大以容纳完整 TTS 响应（每个音频块需要 audio_meta + PCM = 2条消息）
    send_queue_maxsize: int = 500

    # 对话 LLM（主回复 + Function Calling）
    # Qwen3.6 Flash keeps the low-latency lane while gaining 1M context,
    # Function Calling, structured output, and stronger multimodal/reasoning capability.
    conversation_model: str = "qwen3.6-flash"
    solo_language: str = "en"  # Public conference calls speak English.
    agent_brain: str = "brain"  # env: AGENT_BRAIN; "solo" → SoloDriverProcessor, "brain" → LLMProcessor

    same_playout_append_grace_ms: int = 0
    # Fast Judge v1: live critical-path evaluator after Speech text is done.
    # Compatibility flag only; live LLMProcessor uses Fast Judge whenever Judge is not ablated.
    fast_judge_enabled: bool = True
    fast_judge_model: str = "qwen3.6-flash"
    fast_judge_max_tokens: int = 400
    fast_judge_temperature: float = 0.0
    fast_judge_timeout_s: float = 4.0

    # ── ADR 0006 Silent-Rescue ──────────────────────────────────────────────
    # AI playout 完整结束 timeout_ms 内 VAD 未检测到用户开口 → 重跑 Stage 1 prompt。
    # 默认 enabled=True、timeout=5000ms。max_per_call=0 = 无 cap（架构意图）；
    # 生产首次部署用 env 强制 overlay SILENT_RESCUE_MAX_PER_CALL=3 作初始保险，
    # 24h 健康监测达标后 ops 删 overlay 回归无限（spec §9.2）。
    silent_rescue_enabled: bool = True
    silent_rescue_timeout_ms: int = 5000
    silent_rescue_max_per_call: int = 0
    # Extra wait after the AudioSource drains (wait_for_playout) before the
    # silence window opens, covering network + the browser/earpiece jitter
    # buffer that the server cannot directly observe. Keep small.
    silent_rescue_playout_margin_ms: int = 200

    # ── stage1-followup (2026-05-25 spec) ───────────────────────────────────
    # In-turn Judge-driven followup voicing when Stage 1 closed a node the
    # Judge ruled incomplete. Default off; deploy sets it on. (No shadow phase:
    # the stage1_followup_decision debug event provides live posture-vs-
    # deterministic monitoring in production.)
    stage1_followup_enabled: bool = False
    # Max consecutive auto-followups per (node, field) before force-advance.
    stage1_followup_max_per_field: int = 1
    # Fail-closed deterministic guard: when the Judge outputs advance but Stage 1's
    # last line is a question, downgrade to followup and strip the contradicted
    # current-node field before commit. Default off; deploy sets it on.
    judge_advance_question_guard_enabled: bool = False
    # Dedicated Stage 2 prompt mode for node-entry questions. Default off so
    # prompt-pack/runtime plumbing can be deployed and flipped deliberately.
    stage2_entry_prompt_enabled: bool = False

    # Background analyzer backend (memory extraction / short-term memory / sop_progress).
    # Off the critical path; uses Plus for higher-quality summaries.
    analyzer_model: str = "qwen3.7-plus"
    analyzer_max_tokens: int = 300
    analyzer_temperature: float = 0.0

    # Prompt and async analyzer rollout switches.
    speech_prior_state_snapshot_enabled: bool = False
    short_term_memory_analyzer_enabled: bool = False
    short_term_memory_analyzer_enabled_for_patient_calls: bool = True
    note_judge_injection_enabled: bool = True
    note_filter_confidence_threshold: float = 0.5
    note_filter_blacklist: list[str] = [
        "嗯",
        "好的",
        "好",
        "谢谢",
        "明白了",
        "可以的",
        "没问题",
        "我帮你查",
        "稍等",
        "请问还有什么需要",
    ]
    sop_progress_analyzer_enabled: bool = False
    legacy_memory_extractor_enabled: bool = True
    legacy_memory_extractor_shadow_enabled: bool = False
    # Qwen3.6/Qwen3.5 思考模式参数（DashScope OpenAI-compatible 非标准字段，经 extra_body 透传）
    # Speech lane defaults to non-thinking mode to preserve TTFT and streaming UX.
    conversation_enable_thinking: Optional[bool] = False
    conversation_thinking_budget: Optional[int] = None
    # None = 不透传；True/False 时显式控制 tool 并行（仅当请求含 tools）
    conversation_parallel_tool_calls: Optional[bool] = None

    # Context Cache (M07)
    enable_explicit_cache: bool = True  # 可通过环境变量 ENABLE_EXPLICIT_CACHE=false 禁用
    fast_judge_prompt_v2_enabled: bool = True  # v2 prompt 默认开启；如需快速回滚走 FAST_JUDGE_PROMPT_V2_ENABLED=false
    enable_history_cache: bool = True  # 对话历史显式缓存（标记 history 末尾，DashScope 前缀缓存覆盖 system+history）
    history_cache_min_tokens: int = 1024  # 旧历史估算 token >= 此值才启用 marker 2

    # 知识注入模式
    # "fullcontext" = 全量知识写入 prompt（默认，低延迟）
    # "rag" = 动态向量检索
    knowledge_mode: Literal["fullcontext", "rag"] = "fullcontext"

    # Embedding 配置
    # "dashscope" = 默认生产路径，避免本地大模型依赖
    # "local" = 显式 opt-in，本地模型 bge-base-zh-v1.5
    embedding_provider: Literal["dashscope", "local"] = "dashscope"
    embedding_local_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_local_device: str = "auto"  # cpu/cuda/mps/auto
    embedding_dim: int = 1024  # 兼容旧配置；实际维度由 provider 动态决定

    # ── Polaris 改进计划 Feature Flags ──
    # 所有开关默认 off，通过环境变量启用
    polaris_memory_enabled: bool = False  # Part 3: 记忆注入 Flash prompt
    polaris_memory_extraction_enabled: bool = False  # Part 3: 通话结束后提取记忆（显式灰度开启）
    polaris_memory_db_path: str = "data/memory.db"
    polaris_memory_max_items: int = 20
    polaris_memory_max_calls_context: int = 3
    polaris_memory_extraction_model: str = ""  # 空=使用默认模型
    polaris_ttfa_budget_enabled: bool = False  # Part 4: TTFA 预算管理
    ttfa_total_budget_ms: int = 1000  # Part 4A: 总预算
    ttfa_tts_reserve_ms: int = 300  # Part 4A: TTS 预留
    ttfa_llm_max_tokens_degraded: int = 200  # Part 4A: 预算紧张时降级 max_tokens

    # Part 4A: 长对话上下文管理
    flash_max_context_messages: int = 10  # 默认上下文窗口
    flash_context_budget_tokens: int = 3000  # 软上限 token 估算
    flash_context_auto_shrink: bool = True  # 延迟漂移时自动缩小窗口

    # Part 4A: 首句策略优化
    first_sentence_max_chars: int = 30  # 首句最大字符数（超出则在逗号处断句送 TTS）
    first_sentence_timeout_ms: int = 500  # 首句积累超时（强制发送已有内容）

    polaris_asr_audit_log: bool = False  # Part 0: ASR 审计日志（生产默认关）
    polaris_baseline_export: bool = True  # Part 0: 基线导出

    # Turn detection
    smart_turn_model_path: str = "models/smart_turn/smart-turn-v3.2-cpu.onnx"
    aec_warmup_s: float = 0.8  # 仅屏蔽开场前 800ms 回声，避免压掉问句末尾的正常抢答
    min_words_during_bot: int = 2
    final_timeout_ms: int = 500
    asr_final_grace_ms: int = (
        400  # A2: 超时兜底要发 partial 文本前，再等 ASR final 的有界宽限（ms）；0=禁用（回滚开关）
    )
    min_words_to_interrupt: int = 3  # 降低“好的/嗯嗯/明白”等两字短确认触发声学打断
    backchannel_blacklist: list[str] = [
        "嗯",
        "嗯嗯",
        "好的",
        "对",
        "是的",
        "哦",
        "好",
        "嗯哼",
        "啊",
        "哦哦",
    ]

    # System prompt
    system_prompt: str = "你好，我是品驰医疗的售后回访助手。请问您有什么需要帮助的吗？"

    # VPC — 生产 ECS 开 True 走内网；本地开发 False 走公网
    # 通过环境变量 USE_VPC_ENDPOINT=true 切换
    # 前提：
    # 1. 在阿里云 VPC 控制台创建 PrivateLink 终端节点（服务: com.aliyuncs.dashscope）
    # 2. 如需 HTTPS/WSS，配置自定义服务域名或内网 DNS（Private Hosted Zone）
    use_vpc_endpoint: bool = False
    dashscope_vpc_base_url: str = "https://vpc-cn-beijing.dashscope.aliyuncs.com"
    dashscope_vpc_http_api_url: str = "https://vpc-cn-beijing.dashscope.aliyuncs.com/api/v1"
    dashscope_vpc_ws_api_url: str = "wss://vpc-cn-beijing.dashscope.aliyuncs.com/api-ws/v1/inference"
    dashscope_vpc_compatible_base_url: str = "https://vpc-cn-beijing.dashscope.aliyuncs.com/compatible-mode/v1"

    # ── 第四章毕业论文消融开关（默认全部保持生产行为） ──
    # see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
    # 所有默认值必须让系统表现与当前生产完全一致，开启后才进入消融实验分支。
    ablation_disable_judge: bool = False  # 启用后 Judge LLM 不被调用，Speech LLM 单票决
    ablation_disable_sm: bool = False  # 启用后跳过 flow_epoch / 节点推进 / 五级字段状态
    ablation_disable_verify: bool = False  # 启用后从 knowledge_qa prompt 中剥离三步验证段
    bridge_phrase_enabled: bool = True  # 设为 False 时禁用 TTS 桥接语抢跑（followup_buffer）

    # ── §4.3.1 Speech LLM provider switch ──────────────────────────────────────
    # Production default: qwen (DashScope qwen3.6-flash).  Set to "doubao" or "glm"
    # via env for the horizontal Speech-LLM comparison experiment; Judge LLM
    # stays on DashScope qwen regardless (see docs/superpowers/specs/
    # 2026-04-15-thesis-ch4-eval-roadmap.md §4.3.1 — "only swap Speech").
    speech_llm_provider: Literal["qwen", "doubao", "glm"] = "qwen"

    # Doubao (字节火山方舟) — Speech lane only.
    ark_api_key: str = ""
    # Ark OpenAI-compatible base URL (shared by Doubao Speech + §4.6 A4 patient).
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    # Direct model alias first; if the volcengine console requires 接入 to
    # generate an ``ep-xxx`` endpoint id, set DOUBAO_SPEECH_ENDPOINT_ID instead
    # and it takes precedence.
    doubao_speech_model: str = "doubao-seed-1-6-flash-250615"
    doubao_speech_endpoint_id: str = ""

    # ── §4.6 A4: Virtual Caller patient LLM provider switch ──
    # Only affects src/eval/virtual_caller/ evaluation code. Production paths
    # (src/flows/, src/backends/) are untouched.  Quantifies the §11.2
    # "patient & judge in the same family" bias identified in VC spec.
    # - "qwen":      qwen3.6-plus-latest via DashScope (default analyzer family)
    # - "doubao-pro": Doubao-Pro via Volcengine Ark (cross-family control)
    patient_llm_provider: Literal["qwen", "doubao-pro"] = "qwen"
    # VC spec §11 references the "-latest" Plus alias but DashScope only exposes
    # the bare snapshot. Keep to the real snapshot; override via PATIENT_QWEN_MODEL
    # env if a "-latest" alias becomes available.
    patient_qwen_model: str = "qwen3.7-plus"
    # Ark snapshot id (see console → 模型列表). If the model requires 接入 and
    # exposes an ``ep-xxx`` endpoint id, set DOUBAO_PATIENT_ENDPOINT_ID — it
    # takes precedence over the snapshot id below.
    patient_doubao_model: str = "doubao-1-5-pro-32k-250115"
    doubao_patient_endpoint_id: str = ""

    # Zhipu GLM (智谱 bigmodel) — Speech lane only.
    zhipu_api_key: str = ""
    # Smoke-tested 2026-04-17 on the boyu94828 Zhipu account:
    # glm-4-flash works out-of-the-box (free tier, TTFT ≈ 376 ms mean of 3 runs).
    # glm-4.5-airx / glm-4.5-air / glm-4-flashx / glm-4.6-air all return
    # `余额不足或无可用资源包,请充值` (balance low) so they require a paid account.
    # Override via ZHIPU_SPEECH_MODEL=... once you top up.
    zhipu_speech_model: str = "glm-4-flash"

    # Transfer to human
    human_agent_sip_destination: str = ""
    transfer_summary_webhook_url: str = ""
    transfer_summary_timeout_s: float = 3.0
    transfer_webhook_timeout_s: float = 2.0

    # LiveKit
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_room_name: str = "debug-full"
    livekit_agent_identity: str = "agent"

    # ── Live multi-call dispatch (3-way concurrency on 2C4G) ──
    max_live_agents: int = 3
    live_participant_wait_s: int = 600  # eval/standalone default; live spawns set 60 via env
    live_per_child_rss_max_mb: int = 900
    live_max_call_seconds: int = 1800
    live_artifact_dir: str = "data/live_sessions"
    live_reaper_interval_s: float = 5.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def active_dashscope_http_api_url(self) -> str:
        return self.dashscope_vpc_http_api_url if self.use_vpc_endpoint else self.dashscope_http_api_url

    @property
    def active_dashscope_ws_api_url(self) -> str:
        return self.dashscope_vpc_ws_api_url if self.use_vpc_endpoint else self.dashscope_ws_api_url

    @property
    def active_dashscope_compatible_base_url(self) -> str:
        return self.dashscope_vpc_compatible_base_url if self.use_vpc_endpoint else self.dashscope_compatible_base_url


settings = Settings()
