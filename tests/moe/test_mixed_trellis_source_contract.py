from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method_name:
                    return member
    raise AssertionError(f"missing {class_name}.{method_name} in {path}")


def test_mixed_trellis_call_tracks_shared_moe_body_abi() -> None:
    kernel_path = ROOT / "b12x/moe/_shared/kernels/w4a16/kernel.py"
    mixed_path = ROOT / "b12x/moe/_shared/kernels/w4a16/mixed_trellis.py"
    body = _method(kernel_path, "W4A16FusedMoeKernel", "_moe_body")
    mixed = _method(mixed_path, "W4A16MixedTrellisKernel", "kernel")

    calls = [
        node
        for node in ast.walk(mixed)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_moe_body"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert not call.keywords

    parameters = [argument.arg for argument in body.args.args][1:]
    arguments = [ast.unparse(argument) for argument in call.args]
    assert len(arguments) == len(parameters)
    bound = dict(zip(parameters, arguments, strict=True))

    assert bound["fc1_trellis_lut_addr"] == "cutlass.Int64(0)"
    assert bound["fc2_trellis_lut_addr"] == "cutlass.Int64(0)"
    assert bound["weight_num_experts"] == "total_experts"
    assert bound["route_num_experts"] == "total_experts"
    assert bound["active_m"] == "active_m"
    assert bound["fc1_emit_tile"] == "fc1_emit"
    assert bound["fc2_emit_tile"] == "fc2_emit"


def test_mixed_trellis_dispatch_tracks_tile_abis() -> None:
    kernel_path = ROOT / "b12x/moe/_shared/kernels/w4a16/kernel.py"
    mixed_path = ROOT / "b12x/moe/_shared/kernels/w4a16/mixed_trellis.py"
    dispatch = _method(mixed_path, "W4A16MixedTrellisKernel", "_dispatch_tier_gemm")

    for method_name in ("_run_tile", "_run_tile_m8_pair"):
        target = _method(kernel_path, "W4A16GemmKernel", method_name)
        calls = [
            node
            for node in ast.walk(dispatch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method_name
        ]
        assert len(calls) == 1
        parameters = [argument.arg for argument in target.args.args][1:]
        arguments = [ast.unparse(argument) for argument in calls[0].args]
        assert len(arguments) == len(parameters)
        bound = dict(zip(parameters, arguments, strict=True))
        assert bound["trellis_lut_addr"] == "trellis_lut_addr"
        assert bound["active_size_m"] == "active_size_m"


def test_w4a16_internal_calls_supply_required_arguments() -> None:
    path = ROOT / "b12x/moe/_shared/kernels/w4a16/kernel.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "W4A16GemmKernel"
    )
    methods = {
        node.name: node for node in kernel.body if isinstance(node, ast.FunctionDef)
    }

    mismatches: list[str] = []
    for caller in methods.values():
        for call in ast.walk(caller):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
                and call.func.attr in methods
            ):
                continue
            callee = methods[call.func.attr]
            maximum = len(callee.args.args) - 1
            required = maximum - len(callee.args.defaults)
            supplied = len(call.args) + len(call.keywords)
            if not required <= supplied <= maximum:
                mismatches.append(
                    f"{caller.name}:{call.lineno} -> {callee.name}: "
                    f"supplied={supplied}, required={required}, maximum={maximum}"
                )

    assert not mismatches, "\n".join(mismatches)
