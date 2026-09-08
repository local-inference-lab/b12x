"""Compile the Kimi-K3 four-row verify MoE on a host without CUDA devices.

Status: research-only. This reports compiler resources and static instructions,
not GPU correctness or latency. Use the same source and compiler for both arms.
The source tree, command, toolchain and compiled object are recorded per run.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def _cuda_elf(obj: bytes) -> bytes:
    # The host wrapper holds an ELF64 CUDA image, including its section and
    # program headers. Bound extraction to that image instead of dumping strings.
    start = obj.index(b"\x7fELF\x02\x01\x01\x41")
    data = memoryview(obj)[start:]

    def field(offset: int, size: int) -> int:
        return int.from_bytes(data[offset : offset + size], "little")

    if field(18, 2) != 190:
        raise ValueError("embedded ELF is not CUDA")
    end = max(
        field(32, 8) + field(54, 2) * field(56, 2),
        field(40, 8) + field(58, 2) * field(60, 2),
    )
    for index in range(field(60, 2)):
        header = field(40, 8) + index * field(58, 2)
        if field(header + 4, 4) != 8:  # SHT_NOBITS has no file payload.
            end = max(end, field(header + 24, 8) + field(header + 32, 8))
    if end > len(data):
        raise ValueError("embedded CUDA ELF extends beyond its wrapper")
    return bytes(data[:end])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(256, 384), required=True)
    parser.add_argument("--threads", type=int, choices=(256, 512), required=True)
    parser.add_argument("--pairs", type=int, choices=(0, 1), default=0)
    parser.add_argument("--phase-profile", type=int, choices=(0, 1), default=0)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUTE_DSL_ARCH": "sm_120a",
            "CUTE_DSL_KEEP": "ptx,cubin",
            "CUTE_DSL_DUMP_DIR": str(out / "dsl"),
            "B12X_COMPILE_DISK_CACHE": "0",
            "B12X_COMPILE_MEMORY_CACHE": "0",
            "B12X_SQG_XOR_CHEB_T12_SMEM": "1",
            "B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM": "1",
            "B12X_SQG_XOR_CHEB_T12_DIRECT_PAIRS": str(args.pairs),
            "B12X_SQG_XOR_CHEB_T12_DECODE_CHAIN": "funnel",
            "B12X_W4A16_TOKEN_MAJOR_ROTATION": "1",
            "B12X_W4A16_SMALL_M_SPLITK": "1",
            "B12X_W4A16_CROSS_TILE_PREFETCH": "1",
            "B12X_W4A16_STABLE_ROUTE_PACK": "1",
            "B12X_W4A16_PREFILL_FUSED_SUM": "1",
            "B12X_W4A16_TOPK_SUM_OUTPUT": "bf16",
            "B12X_W4A16_M8_CTA_THREADS": str(args.threads),
            "B12X_W4A16_PHASE_PROFILE": str(args.phase_profile),
        }
    )

    import cuda.bindings.driver as cuda
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("run this offline probe without GPU device access")

    class _Properties:
        multi_processor_count = 188
        shared_memory_per_block_optin = 101376

    # These are compile-time hardware facts, never a device execution route.
    torch.cuda.is_available = lambda: True
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_properties = lambda *_a, **_k: _Properties()
    from b12x.moe._shared.kernels.w4a16 import kernel

    kernel.current_cuda_stream = lambda: cuda.CUstream(0)
    captured = {}
    original_compile = kernel.b12x_compile

    class _Compiled(Exception):
        pass

    def capture(owner, *values, **options):
        captured.update(
            owner=owner, compiled=original_compile(owner, *values, **options)
        )
        # Stop before resource queries or direct-LUT materialization need a GPU.
        raise _Compiled

    kernel.b12x_compile = capture
    kwargs = dict(
        size_m=4,
        hidden_size=3584,
        intermediate_size=args.width,
        num_experts=896,
        top_k=16,
        activation="situ",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=8,
        max_m_blocks=64,
        element_dtype="fp16",
        fast_math=True,
        sms=188,
        max_shared_mem=101376,
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis3_t256_proj",
        trellis_bits=2,
        trellis_codebook="sqg_xor_cheb_t12",
        force_tile_config=(128, 128, 128, 128),
        direct_topk_routes=True,
        use_expert_map=True,
        intermediate_rotation=True,
        full_rotation=True,
        coupled_hadamard=True,
        rotation_input_dtype="bf16",
        broadcast_suh=True,
        cta_threads_multiplier=args.threads // 256,
    )
    try:
        kernel.compile_w4a16_fused_moe(**kwargs)
    except _Compiled:
        pass
    else:
        raise RuntimeError("compile interception did not run")

    owner = captured["owner"]
    obj = captured["compiled"].dump_to_object("k3_verify_moe")
    cubin = _cuda_elf(obj)
    (out / "kernel.o").write_bytes(obj)
    (out / "kernel.cubin").write_bytes(cubin)
    cuobjdump = os.getenv("CUOBJDUMP", "/usr/local/cuda/bin/cuobjdump")
    sass = _run(cuobjdump, "--dump-sass", str(out / "kernel.cubin"))
    resources = _run(cuobjdump, "--dump-resource-usage", str(out / "kernel.cubin"))
    (out / "sass.txt").write_text(sass)
    (out / "resources.txt").write_text(resources)
    instructions = re.findall(
        r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P(?:\d+|T)\s+)?([A-Z][A-Z0-9_.]*)\b[^;\n]*;",
        sass,
        re.MULTILINE,
    )
    source = Path(__file__).resolve().parents[1]
    source_hashes = {
        path.relative_to(source).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source.joinpath("b12x").rglob("*.py")
    }
    record = {
        "status": "research-only; offline compiler evidence",
        "argv": sys.argv,
        "source_revision": args.source_revision,
        "kwargs": kwargs,
        "environment": {
            k: v for k, v in os.environ.items() if k.startswith(("B12X_", "CUTE_"))
        },
        "toolchain": {
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl", "cuda-python")
        },
        "ptxas": _run("/usr/local/cuda/bin/ptxas", "--version"),
        "source_sha256": source_hashes,
        "object_sha256": hashlib.sha256(obj).hexdigest(),
        "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
        "cta_threads": owner.cta_threads,
        "sms": owner.sms,
        "shared_layout_bytes_unrounded": owner.shared_words * 4 + 16,
        "blocks_per_sm": owner.blocks_per_sm,
        "fc1_splitk": owner.fc1.small_m_splitk,
        "fc2_splitk": owner.fc2.small_m_splitk,
        "phase_profile": owner.phase_profile,
        "workspace_elements": owner.workspace_elements,
        "direct_smem": owner.sqg_xor_cheb_t12_direct_smem,
        "token_major_rotation": owner.token_major_rotation,
        "instruction_count": len(instructions),
        "instruction_histogram": dict(collections.Counter(instructions)),
        "byte_shift_imad": len(
            re.findall(r"\bIMAD\s[^;]*, 0x(?:100|10000|1000000),", sass)
        ),
        "resources": resources,
    }
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "status",
                    "cta_threads",
                    "shared_layout_bytes_unrounded",
                    "instruction_count",
                    "byte_shift_imad",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
