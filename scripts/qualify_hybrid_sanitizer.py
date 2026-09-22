"""Run one bounded sanitizer isolation gate with rank-local evidence."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._sm103_source import source_identity
from scripts.qualify_expert_cache import source_files
from b12x.testing.artifacts import sha256


def classify(returncode, timed_out, summaries, completed, ranks, filtered):
    """Completion, zero errors and full rank coverage are independent requirements."""
    if timed_out:
        return "timeout"
    if returncode != 0 or any(
        value != 0 for values in summaries.values() for value in values
    ):
        return "failed"
    if (
        set(summaries) != set(range(ranks))
        or any(not v for v in summaries.values())
        or completed != set(range(ranks))
    ):
        return "incomplete"
    return "component_pass" if filtered else "whole_program_pass"


def diagnostic_inventory(text):
    """Recognize initialization probes without suppressing or passing their errors."""
    counts = {}
    unknown = []
    for block in re.split(r"(?m)^=========\s*$", text):
        lines = [
            line.removeprefix("========= ")
            for line in block.splitlines()
            if line.startswith("========= ")
        ]
        if not lines:
            continue
        # The banner may share the first block with the first diagnostic.
        lines = [line for line in lines if line != "COMPUTE-SANITIZER"]
        if not lines or lines[0].startswith("ERROR SUMMARY:"):
            continue
        heading = lines[0]
        probe = "ncclInitKernelsForDevice" in block and "libnccl.so" in block
        category = None
        if probe:
            if heading.startswith(
                "Program hit cudaErrorNoKernelImageForDevice (error 209)"
            ) and any(
                f"call to {api}." in heading
                for api in ("cudaFuncGetAttributes", "cudaGetLastError")
            ):
                category = "unavailable_nccl_kernel_image"
            elif re.fullmatch(
                r"CUDA API Error: Kernel \(_Z\w*ncclSymkDevKernel\w*f8\w*\) cannot be found in library due to compilation error",
                heading,
            ):
                category = "nccl_fp8_symmetric_kernel_compilation"
            elif (
                heading
                == "CUDA API Error: To get more information, use the CU_JIT_ERROR_LOG_BUFFER and CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES environment variables"
            ):
                category = "nccl_kernel_jit_information"
        if category:
            counts[category] = counts.get(category, 0) + 1
        else:
            unknown.append(heading)
    summaries = [int(n) for n in re.findall(r"ERROR SUMMARY: (\d+) errors?", text)]
    accounted = bool(summaries) and sum(counts.values()) == sum(summaries)
    return dict(
        categories=counts,
        unclassified=unknown,
        summaries=summaries,
        all_errors_accounted=accounted and not unknown,
        initialization_probe_only=bool(counts) and accounted and not unknown,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=(
            "cuda",
            "nccl-init",
            "collective",
            "production",
            "layer",
            "compact",
            "tp-layer",
        ),
        required=True,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ranks", type=int, default=2)
    p.add_argument("--sanitizer", type=Path, required=True)
    p.add_argument("--tool", choices=("memcheck", "synccheck"), default="memcheck")
    p.add_argument("--deadline", type=float, default=300)
    p.add_argument("--kernel-filter")
    p.add_argument("--layers", type=int, nargs="+", default=[0])
    p.add_argument("--oracle-device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--rank-child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.ranks < 1 or args.deadline <= 0:
        p.error("ranks and deadline must be positive")
    if args.stage in ("layer", "compact") and args.ranks != 1:
        p.error("single-rank checkpoint tests require --ranks 1")
    out = args.output.resolve()
    if args.rank_child:
        rank = int(os.environ.get("RANK", 0))
        command = [
            str(args.sanitizer),
            "--tool",
            args.tool,
            "--target-processes",
            "all",
            "--error-exitcode",
            "99",
            "--print-limit",
            "0",
            "--log-file",
            str(out / f"rank-{rank}-sanitizer.log"),
        ]
        if args.kernel_filter:
            command += ["--kernel-name", args.kernel_filter]
        command += [
            sys.executable,
            str(ROOT / "tests/comm/sanitizer_worker.py"),
            "--stage",
            args.stage,
            "--output",
            str(out),
            "--oracle-device",
            args.oracle_device,
            "--layers",
            *(str(layer) for layer in args.layers),
        ]
        (out / f"rank-{rank}-command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        with (out / f"rank-{rank}-application.log").open("w") as stream:
            status = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
        (out / f"rank-{rank}-exit.json").write_text(
            json.dumps({"returncode": status}) + "\n"
        )
        # Let every instrumented rank flush its summary before the launcher
        # observes failure. The parent requires all recorded rank exit codes.
        return 0
    if out.exists():
        p.error("output must be a new directory")
    out.mkdir(parents=True)
    arguments = sys.argv[1:] if argv is None else argv
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.ranks}",
        str(Path(__file__).resolve()),
        *arguments,
        "--rank-child",
    ]
    receipt = dict(
        schema="b12x-hybrid-sanitizer/v1",
        source=source_identity(ROOT),
        source_files=source_files(),
        command=command,
        stage=args.stage,
        layers=args.layers,
        oracle_device=args.oracle_device,
        scope="filtered component" if args.kernel_filter else "whole program",
        kernel_filter=args.kernel_filter,
        deadline_s=args.deadline,
        sanitizer_sha256=sha256(args.sanitizer),
        sanitizer_version=subprocess.check_output(
            [str(args.sanitizer), "--version"], text=True
        ),
        gpu=subprocess.check_output(["nvidia-smi", "-q"], text=True),
        environment={
            k: os.environ.get(k)
            for k in (
                "VLLM_NCCL_SO_PATH",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "CUDA_VISIBLE_DEVICES",
                "B12X_ACCEPTANCE_BUILD_MANIFEST",
            )
        },
        started_ns=time.time_ns(),
        status="running",
    )
    manifest = out / "acceptance.json"
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    begin = time.monotonic()
    timed_out = False
    with (out / "launcher.log").open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=args.deadline)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    summaries = {}
    diagnostics = {}
    for rank in range(args.ranks):
        path = out / f"rank-{rank}-sanitizer.log"
        if not path.is_file():
            continue
        diagnostics[str(rank)] = diagnostic_inventory(path.read_text())
        summaries[rank] = [
            int(n)
            for n in re.findall(r"ERROR SUMMARY: (\d+) errors?", path.read_text())
        ]
    completed = {
        r for r in range(args.ranks) if (out / f"rank-{r}-complete.json").is_file()
    }
    rank_exits = {
        rank: json.loads((out / f"rank-{rank}-exit.json").read_text())["returncode"]
        for rank in range(args.ranks)
        if (out / f"rank-{rank}-exit.json").is_file()
    }
    launcher_code = code
    if not timed_out and (
        set(rank_exits) != set(range(args.ranks)) or any(rank_exits.values())
    ):
        code = next((v for v in rank_exits.values() if v), code or 1)
    receipt.update(
        returncode=code,
        launcher_returncode=launcher_code,
        rank_returncodes=rank_exits,
        timed_out=timed_out,
        elapsed_s=time.monotonic() - begin,
        summaries=summaries,
        diagnostics=diagnostics,
        completed_ranks=sorted(completed),
        status=classify(
            code, timed_out, summaries, completed, args.ranks, bool(args.kernel_filter)
        ),
        logs={
            path.name: sha256(path)
            for path in sorted(out.iterdir())
            if path != manifest and path.is_file()
        },
        last_progress={
            path.name: path.read_text().splitlines()[-1:]
            for path in out.glob("rank-*-progress.jsonl")
        },
    )
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "status",
                    "returncode",
                    "elapsed_s",
                    "summaries",
                    "completed_ranks",
                )
            }
        ),
        flush=True,
    )
    return 0 if receipt["status"] in ("component_pass", "whole_program_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
