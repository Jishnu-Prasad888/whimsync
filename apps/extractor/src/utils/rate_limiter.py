import asyncio
import time


class RequestRateLimiter:
    """
    Paces request starts to at most `rpm` per minute (minimum spacing of 60/rpm
    seconds between consecutive acquisitions). `rpm <= 0` disables limiting.
    """

    def __init__(self, rpm: float = 0.0):
        self._lock = asyncio.Lock()
        self._next_slot = 0.0
        self.set_rpm(rpm)

    def set_rpm(self, rpm: float) -> None:
        self.rpm = max(0.0, float(rpm))
        self._interval = 60.0 / self.rpm if self.rpm > 0 else 0.0

    @property
    def enabled(self) -> bool:
        return self._interval > 0

    async def acquire(self) -> float:
        """Waits for the next free slot; returns the seconds spent waiting."""
        if not self.enabled:
            return 0.0
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)
        return wait
