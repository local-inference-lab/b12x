"""CUDA and maintained-engine collective controls for sanitizer isolation."""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import runpy

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b12x.testing.artifacts import sha256, verify_from_file


def main():
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
    p.add_argument("--layers", type=int, nargs="+", default=[0])
    p.add_argument("--oracle-device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output / f"rank-{rank}-progress.jsonl"
    libraries = {}

    def mark(location):
        for line in Path("/proc/self/maps").read_text().splitlines():
            path = line.split()[-1]
            if (
                path.startswith("/")
                and any(name in path for name in ("libnccl", "libcuda.", "libcudart"))
                and path not in libraries
            ):
                libraries[path] = sha256(path)
        (args.output / f"rank-{rank}-libraries.json").write_text(
            json.dumps(libraries, indent=2) + "\n"
        )
        row = dict(
            rank=rank,
            location=location,
            monotonic_ns=time.monotonic_ns(),
            time_ns=time.time_ns(),
        )
        with progress.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    mark("cuda-init")
    torch.cuda.set_device(rank)
    x = torch.full((1024,), rank + 1, device="cuda", dtype=torch.bfloat16)
    mark("cuda-ready")
    if args.stage == "cuda":
        result = x + 2
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            result = x + 2
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(result.cpu(), torch.full_like(result.cpu(), rank + 3))
        graph.reset()
    elif args.stage in ("nccl-init", "collective"):
        # CPU rendezvous does not initialize a second CUDA communicator.
        dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=300))
        mark("gloo-ready")
        if args.stage == "nccl-init":
            from vllm.distributed.device_communicators.pynccl_wrapper import (
                NCCLLibrary,
                ncclUniqueId,
            )

            library = NCCLLibrary()
            unique = library.ncclGetUniqueId() if rank == 0 else ncclUniqueId()
            raw = torch.tensor(list(unique.internal), dtype=torch.uint8)
            dist.broadcast(raw, src=0)
            for i, value in enumerate(raw.tolist()):
                unique.internal[i] = value
            mark("nccl-init-enter")
            comm = library.ncclCommInitRank(world, unique, rank)
            mark("nccl-init-return")
            library.ncclCommDestroy(comm)
        else:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

            mark("pynccl-init-enter")
            comm = PyNcclCommunicator(dist.group.WORLD, rank)
            assert comm.available and not comm.disabled
            mark("pynccl-init-return")
            result = comm.all_reduce(x)
            torch.cuda.synchronize()
            assert torch.all(result == world * (world + 1) // 2)
            mark("collective-return")
            comm.destroy()
        dist.destroy_process_group()
    elif args.stage == "production":
        from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
            tensor_model_parallel_all_reduce,
        )
        from vllm.distributed.parallel_state import graph_capture, set_custom_all_reduce

        config = VllmConfig(
            parallel_config=ParallelConfig(
                tensor_parallel_size=world, disable_custom_all_reduce=True
            )
        )
        with set_current_vllm_config(config):
            set_custom_all_reduce(False)
            mark("model-parallel-init-enter")
            init_distributed_environment(
                world_size=world,
                rank=rank,
                local_rank=rank,
                distributed_init_method="env://",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=world, pipeline_model_parallel_size=1
            )
            mark("model-parallel-init-return")
            result = tensor_model_parallel_all_reduce(x)
            torch.cuda.synchronize()
            assert torch.all(result == world * (world + 1) // 2)
            graph = torch.cuda.CUDAGraph()
            with (
                graph_capture(torch.device("cuda", rank)) as capture,
                torch.cuda.graph(graph, stream=capture.stream),
            ):
                result = tensor_model_parallel_all_reduce(x)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.all(result == world * (world + 1) // 2)
            graph.reset()
            mark("production-graph-return")
            destroy_model_parallel()
            destroy_distributed_environment()
    else:
        os.environ["B12X_CHECKPOINT_PROGRESS"] = str(args.output)
        os.environ["B12X_CHECKPOINT_ORACLE_DEVICE"] = args.oracle_device
        if args.stage == "tp-layer":
            for layer in args.layers:
                sys.argv = [
                    "tp_checkpoint_worker.py",
                    "--checkpoint",
                    os.environ["B12X_TEST_NEXT80_CHECKPOINT"],
                    "--layer",
                    str(layer),
                    "--output",
                    str(args.output / f"layer-{layer}"),
                ]
                runpy.run_path(
                    str(Path(__file__).parents[1] / "moe/tp_checkpoint_worker.py"),
                    run_name="__main__",
                )
        else:
            import pytest
            from scripts.qualify_sm103 import junit_counts

            if args.stage == "compact":
                os.environ["B12X_TEST_NEXT80_ROUTE_SUBSET"] = "1"
            tests = [
                str(Path(__file__).parents[1] / "moe/test_next80_checkpoint.py")
                + f"::test_real_next80_routes_graph_and_independent_arithmetic[{layer}]"
                for layer in args.layers
            ]
            junit = args.output / "checkpoint-tests.xml"
            code = pytest.main(
                [
                    *tests,
                    "-q",
                    "-s",
                    "--junitxml",
                    str(junit),
                    "-o",
                    f"cache_dir={args.output / 'pytest-cache'}",
                ]
            )
            counts = junit_counts(junit)
            if code or counts != dict(
                tests=len(args.layers), failures=0, errors=0, skipped=0
            ):
                raise RuntimeError(
                    f"real checkpoint tests incomplete: exit={code}, counts={counts}"
                )
    torch.cuda.synchronize()
    mark("released")
    paths = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(
                name in line
                for name in ("libnccl", "libcuda.", "libcudart", "libtorch_cuda")
            )
            and line.split()[-1].startswith("/")
        }
    )
    receipt = dict(
        stage=args.stage,
        rank=rank,
        world_size=world,
        completed=True,
        oracle_device=args.oracle_device,
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(rank)),
        libraries={path: sha256(path) for path in paths},
    )
    manifest = os.environ.get("B12X_ACCEPTANCE_BUILD_MANIFEST")
    if args.stage != "cuda":
        if not manifest:
            raise ValueError(
                "companion controls require B12X_ACCEPTANCE_BUILD_MANIFEST"
            )
        receipt["companion"] = verify_from_file(manifest, loaded_only=True)
    (args.output / f"rank-{rank}-complete.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    mark("complete")


if __name__ == "__main__":
    main()
