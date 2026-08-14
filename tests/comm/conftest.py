"""CUTLASS compatibility for CPU-only communication test collection."""

import importlib
import sys
import types

import pytest


CUTLASS_STUBBED = False


def _stub_cutlass():
    global CUTLASS_STUBBED
    try:
        cutlass = importlib.import_module("cutlass")
        cute = importlib.import_module("cutlass.cute")
        llvm = importlib.import_module("cutlass._mlir.dialects.llvm")
        cutlass_dsl = importlib.import_module("cutlass.cutlass_dsl")
    except (ImportError, AttributeError):
        cutlass = None
    if cutlass is not None and all(
        hasattr(cutlass, name)
        for name in (
            "Float32",
            "Int32",
            "Int64",
            "Uint8",
            "Uint16",
            "Uint32",
            "Uint64",
        )
    ) and all(
        (
            hasattr(cute, "jit"),
            hasattr(llvm, "inline_asm"),
            hasattr(cutlass_dsl, "dsl_user_op"),
        )
    ):
        return

    cutlass = types.ModuleType("cutlass")
    for name in (
        "Float32",
        "Int32",
        "Int64",
        "Uint8",
        "Uint16",
        "Uint32",
        "Uint64",
    ):
        setattr(cutlass, name, type(name, (), {}))
    cutlass._mlir = types.ModuleType("cutlass._mlir")
    cutlass._mlir.dialects = types.ModuleType("cutlass._mlir.dialects")
    cutlass._mlir.dialects.llvm = types.ModuleType("cutlass._mlir.dialects.llvm")
    cutlass._mlir.dialects.llvm.inline_asm = lambda *a, **k: None
    cutlass._mlir.dialects.llvm.AsmDialect = types.SimpleNamespace(AD_ATT=0)
    cutlass.cute = types.ModuleType("cutlass.cute")
    cutlass.cute.jit = lambda f: f
    cutlass.cutlass_dsl = types.ModuleType("cutlass.cutlass_dsl")
    cutlass.cutlass_dsl.T = type("T", (), {})
    cutlass.cutlass_dsl.dsl_user_op = lambda f: f
    cutlass.__b12x_test_stub__ = True
    sys.modules["cutlass"] = cutlass
    sys.modules["cutlass.cute"] = cutlass.cute
    sys.modules["cutlass._mlir"] = cutlass._mlir
    sys.modules["cutlass._mlir.dialects"] = cutlass._mlir.dialects
    sys.modules["cutlass._mlir.dialects.llvm"] = cutlass._mlir.dialects.llvm
    sys.modules["cutlass.cutlass_dsl"] = cutlass.cutlass_dsl
    CUTLASS_STUBBED = True


_stub_cutlass()


@pytest.fixture
def require_real_cutlass():
    if CUTLASS_STUBBED:
        pytest.skip("real CUTLASS DSL is required; collection stub is active")
