"""Cyclic executable finalizers must not issue CUDA APIs during capture."""
import gc
from contextlib import nullcontext
import threading

import pytest

from b12x.preparation.session import _capture_gc_guard


@pytest.mark.parametrize("fail", [False, True])
def test_cyclic_finalizer_runs_after_capture_scope(fail):
    enabled = gc.isenabled()
    gc.enable()
    events = []
    class Executable:
        def __del__(self):
            events.append("unloaded")
    try:
        with pytest.raises(RuntimeError) if fail else nullcontext():
            with _capture_gc_guard():
                owner = Executable()
                owner.cycle = owner
                del owner
                assert not gc.isenabled() and events == []
                if fail:
                    raise RuntimeError("capture failed")
        assert gc.isenabled() and events == ["unloaded"]
    finally:
        if not enabled:
            gc.disable()


def test_nested_capture_preserves_caller_gc_state():
    enabled = gc.isenabled()
    gc.disable()
    try:
        with _capture_gc_guard():
            with _capture_gc_guard():
                assert not gc.isenabled()
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        if enabled:
            gc.enable()


def test_overlapping_threads_cannot_restore_gc_inside_another_capture():
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    def first():
        with _capture_gc_guard():
            entered.set()
            assert release.wait(5)
            assert not gc.isenabled()
    def other():
        assert entered.wait(5)
        with _capture_gc_guard():
            assert not gc.isenabled()
            second.set()
    a, b = threading.Thread(target=first), threading.Thread(target=other)
    a.start(); b.start()
    assert entered.wait(5)
    assert not second.wait(0.05)
    release.set()
    a.join(5); b.join(5)
    assert not a.is_alive() and not b.is_alive() and second.is_set()
