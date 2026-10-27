"""Token Bucket 限流器"""

import asyncio
import time


class TokenBucketRateLimiter:
    """Token Bucket 限流：控制每秒请求数"""

    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: 每秒允许的请求数 (tokens/second)
            burst: 最大突发量 (bucket size)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: float = 30.0) -> bool:
        """获取一个 token，返回是否成功"""
        start = time.monotonic()
        while True:
            async with self._lock:
                now = time.monotonic()
                # 补充 tokens
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

            if time.monotonic() - start > timeout:
                return False
            await asyncio.sleep(0.01)

    def try_acquire(self) -> bool:
        """尝试立即获取 token（非阻塞），返回是否成功"""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class RateLimitExceeded(Exception):
    """限流异常"""

    pass


# 全局限流器实例（延迟初始化）
_asr_limiter: TokenBucketRateLimiter | None = None
_tts_limiter: TokenBucketRateLimiter | None = None


def get_asr_limiter() -> TokenBucketRateLimiter:
    """获取 ASR 限流器"""
    global _asr_limiter
    if _asr_limiter is None:
        from src.core.config import settings

        _asr_limiter = TokenBucketRateLimiter(rate=settings.asr_rps, burst=5)
    return _asr_limiter


def get_tts_limiter() -> TokenBucketRateLimiter:
    """获取 TTS 限流器"""
    global _tts_limiter
    if _tts_limiter is None:
        from src.core.config import settings

        _tts_limiter = TokenBucketRateLimiter(rate=settings.tts_rps, burst=5)
    return _tts_limiter
