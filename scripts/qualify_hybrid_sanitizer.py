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
    if returncode != 0 or any(value != 0 for value in summaries):
        return "failed"
    if len(summaries) < ranks or completed != set(range(ranks)):
        return "incomplete"
    return "component_pass" if filtered else "whole_program_pass"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=("cuda", "nccl-init", "collective", "production"),
        required=True,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ranks", type=int, default=2)
    p.add_argument("--sanitizer", type=Path, required=True)
    p.add_argument("--tool", choices=("memcheck", "synccheck"), default="memcheck")
    p.add_argument("--deadline", type=float, default=300)
    p.add_argument("--kernel-filter")
    p.add_argument("--rank-child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.ranks < 1 or args.deadline <= 0:
        p.error("ranks and deadline must be positive")
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
        ]
        (out / f"rank-{rank}-command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        with (out / f"rank-{rank}-application.log").open("w") as stream:
            status = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
        (out / f"rank-{rank}-exit.json").write_text(
            json.dumps({"returncode": status}) + "\n"
        )
        return status
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
    summaries = []
    for path in sorted(out.glob("rank-*-sanitizer.log")):
        summaries.extend(
            int(n)
            for n in re.findall(r"ERROR SUMMARY: (\d+) errors?", path.read_text())
        )
    completed = {
        r for r in range(args.ranks) if (out / f"rank-{r}-complete.json").is_file()
    }
    receipt.update(
        returncode=code,
        timed_out=timed_out,
        elapsed_s=time.monotonic() - begin,
        summaries=summaries,
        completed_ranks=sorted(completed),
        status=classify(
            code, timed_out, summaries, completed, args.ranks, bool(args.kernel_filter)
        ),
        logs={
            path.name: sha256(path)
            for path in sorted(out.iterdir())
            if path != manifest
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
