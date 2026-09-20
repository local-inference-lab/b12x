"""Explicit application shutdown for the source-matched serving experiment."""

import asyncio


async def close_serving(engine, tasks=(), *, timeout=30.0):
    """Stop producers, await owned controls, then retire engine resources.

    The task set must include every application request/control producer.
    Shutdown timeout is a failure gate, not permission to resume a lane whose
    control outcome is uncertain. Caller cancellation cannot abandon cleanup.
    """
    if timeout <= 0:
        raise ValueError("shutdown timeout must be positive")

    async def close():
        active = tuple(t for t in tasks if t is not None)
        for task in active:
            if not task.done():
                task.cancel()
        try:
            if active:
                _, pending = await asyncio.wait(active, timeout=timeout)
                if pending:
                    raise TimeoutError(
                        "serving producers did not retire before shutdown"
                    )
                for result in await asyncio.gather(*active, return_exceptions=True):
                    if isinstance(result, BaseException) and not isinstance(
                        result, asyncio.CancelledError
                    ):
                        raise result
        finally:
            await engine.shutdown_async(timeout=timeout)

    operation = asyncio.create_task(close())
    cancelled = False
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            cancelled = True
    operation.result()
    if cancelled:
        raise asyncio.CancelledError
