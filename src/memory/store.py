"""跨通话记忆存储（SQLite + aiosqlite）

零配置、Docker volume ./data:/app/data 已存在、单实例部署够用。
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

import aiosqlite

from src.memory.models import (
    CallSummary,
    MemoryCategory,
    MemoryItem,
    PatientMemoryContext,
)

logger = logging.getLogger(__name__)

# 模块级单例
_memory_store: Optional["MemoryStore"] = None


def get_memory_store() -> "MemoryStore":
    """获取全局 MemoryStore 单例"""
    global _memory_store
    if _memory_store is None:
        from src.core.config import settings

        _memory_store = MemoryStore(db_path=settings.polaris_memory_db_path)
    return _memory_store


class MemoryStore:
    """SQLite 记忆存储"""

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self._initialized = False

    async def initialize(self):
        """创建表结构（幂等）"""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    source_call_date TEXT NOT NULL,
                    months_since_surgery INTEGER NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS call_summaries (
                    call_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    call_date TEXT NOT NULL,
                    duration_seconds INTEGER DEFAULT 0,
                    turn_count INTEGER DEFAULT 0,
                    sop_fields_collected TEXT DEFAULT '{}',
                    sop_groups_completed TEXT DEFAULT '[]',
                    outcome TEXT DEFAULT '',
                    transcript_summary TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_memory_patient ON memory_items(patient_id, is_active)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_calls_patient ON call_summaries(patient_id, call_date)")
            await db.commit()
        self._initialized = True
        logger.info(f"MemoryStore initialized: {self.db_path}")

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.initialize()

    async def save_memory(self, item: MemoryItem):
        """保存一条记忆"""
        await self._ensure_initialized()
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO memory_items
                   (memory_id, patient_id, category, content, source_session_id,
                    source_call_date, months_since_surgery, confidence, is_active,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.memory_id,
                    item.patient_id,
                    item.category.value,
                    item.content,
                    item.source_session_id,
                    item.source_call_date,
                    item.months_since_surgery,
                    item.confidence,
                    1 if item.is_active else 0,
                    item.created_at or now,
                    now,
                ),
            )
            await db.commit()

    async def save_call_summary(self, summary: CallSummary):
        """保存通话摘要"""
        await self._ensure_initialized()
        import json

        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO call_summaries
                   (call_id, patient_id, call_date, duration_seconds, turn_count,
                    sop_fields_collected, sop_groups_completed, outcome,
                    transcript_summary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary.call_id,
                    summary.patient_id,
                    summary.call_date,
                    summary.duration_seconds,
                    summary.turn_count,
                    json.dumps(summary.sop_fields_collected, ensure_ascii=False),
                    json.dumps(summary.sop_groups_completed, ensure_ascii=False),
                    summary.outcome,
                    summary.transcript_summary,
                    summary.created_at or now,
                ),
            )
            await db.commit()

    async def get_patient_memories(
        self, patient_id: str, active_only: bool = True, limit: int = 20
    ) -> List[MemoryItem]:
        """获取患者的记忆列表"""
        await self._ensure_initialized()
        query = "SELECT * FROM memory_items WHERE patient_id = ?"
        params: list = [patient_id]
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_memory(row) for row in rows]

    async def get_patient_call_summaries(self, patient_id: str, limit: int = 3) -> List[CallSummary]:
        """获取患者的通话摘要列表"""
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM call_summaries WHERE patient_id = ? ORDER BY call_date DESC LIMIT ?",
                (patient_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_call_summary(row) for row in rows]

    async def deactivate_memory(self, memory_id: str):
        """软删除记忆"""
        await self._ensure_initialized()
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE memory_items SET is_active = 0, updated_at = ? WHERE memory_id = ?",
                (now, memory_id),
            )
            await db.commit()

    async def supersede_memory(self, old_id: str, new_item: MemoryItem):
        """用新记忆替代旧记忆"""
        await self.deactivate_memory(old_id)
        await self.save_memory(new_item)

    async def get_patient_memory_context(self, patient_id: str) -> PatientMemoryContext:
        """装载完整的患者记忆上下文"""
        await self._ensure_initialized()
        from src.core.config import settings

        memories = await self.get_patient_memories(patient_id, limit=settings.polaris_memory_max_items)
        calls = await self.get_patient_call_summaries(patient_id, limit=settings.polaris_memory_max_calls_context)

        last_call_date = calls[0].call_date if calls else None

        # 总通话数（不限制 limit）
        total = 0
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM call_summaries WHERE patient_id = ?",
                (patient_id,),
            ) as cursor:
                row = await cursor.fetchone()
                total = row[0] if row else 0

        return PatientMemoryContext(
            patient_id=patient_id,
            previous_calls=calls,
            active_memories=memories,
            last_call_date=last_call_date,
            total_call_count=total,
        )

    # ---- 内部工具 ----

    @staticmethod
    def _row_to_memory(row) -> MemoryItem:
        return MemoryItem(
            memory_id=row["memory_id"],
            patient_id=row["patient_id"],
            category=MemoryCategory(row["category"]),
            content=row["content"],
            source_session_id=row["source_session_id"],
            source_call_date=row["source_call_date"],
            months_since_surgery=row["months_since_surgery"],
            confidence=row["confidence"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_call_summary(row) -> CallSummary:
        import json

        return CallSummary(
            call_id=row["call_id"],
            patient_id=row["patient_id"],
            call_date=row["call_date"],
            duration_seconds=row["duration_seconds"],
            turn_count=row["turn_count"],
            sop_fields_collected=json.loads(row["sop_fields_collected"]),
            sop_groups_completed=json.loads(row["sop_groups_completed"]),
            outcome=row["outcome"],
            transcript_summary=row["transcript_summary"],
            created_at=row["created_at"],
        )
