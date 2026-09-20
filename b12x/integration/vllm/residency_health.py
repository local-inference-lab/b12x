"""Opt-in read-only health utility over the maintained single-worker executor."""

import asyncio
from time import perf_counter_ns


class VllmResidencyHealth:
    """No scheduler pause, policy step or fill. Serialize with maintenance externally."""

    def __init__(self, engine, *, record_history=False):
        if type(record_history) is not bool:
            raise TypeError("record_history must be boolean")
        self.engine = engine
        self.record_history = record_history
        self.failed = False
        self._lock = asyncio.Lock()

    async def probe(self):
        async with self._lock:
            if self.failed:
                raise RuntimeError("health outcome is uncertain; reload the lane")

            async def read():
                start = perf_counter_ns()
                replies = await self.engine.collective_rpc(
                    "b12x_residency_health",
                    args=("start", True) if self.record_history else ("start",),
                )
                if len(replies) != 1 or replies[0].get("submitted") is not True:
                    raise RuntimeError("health probe requires one serialized worker")
                worker_submit_ns = replies[0].get("worker_wall_ns")
                history = replies[0].get("history")
                submitted = perf_counter_ns()
                polls = 0
                while True:
                    await asyncio.sleep(0.001)
                    replies = await self.engine.collective_rpc(
                        "b12x_residency_health", args=("poll",)
                    )
                    polls += 1
                    if len(replies) != 1:
                        raise RuntimeError("health probe rank ownership changed")
                    if replies[0] is not None:
                        return dict(
                            summary=replies[0],
                            worker_submit_ns=worker_submit_ns,
                            submit_rpc_ns=submitted - start,
                            total_wall_ns=perf_counter_ns() - start,
                            polls=polls,
                            history=history,
                        )

            task = asyncio.create_task(read())
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # The engine owns the submitted GPU read and pinned result slot.
                # Consume it before another operation can reuse that storage.
                try:
                    await task
                finally:
                    self.failed = True
                raise
            except BaseException:
                self.failed = True
                raise
