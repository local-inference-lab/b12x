"""Compare TP checkpoint reads and GPU redistribution using actual W2 tensor bytes.

This standalone research app uses the production coalesced GDS executor as its
baseline. It does not construct a model or change checkpoint loader policy.
"""

from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import sysconfig
import time

import torch

from b12x.loader._checkpoint import DirectWeightSession



@dataclass(frozen=True)
class Case:
    name: str
    path: str
    offset: int
    rows: int
    width: int
    dtype: str
    header_sha256: str

    @property
    def nbytes(self):
        return self.rows * self.width


def select_cases(model, experts, tp):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    pattern = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w2\.weight$")
    names = sorted(
        (name for name in index["weight_map"] if pattern.fullmatch(name)),
        key=lambda name: tuple(map(int, pattern.fullmatch(name).groups())),
    )
    if experts < tp or experts % tp or experts > len(names):
        raise ValueError("expert count must be a multiple of TP, at least TP, and within the corpus")
    selected = [names[i * len(names) // experts] for i in range(experts)]
    cases = []
    headers = {}
    for name in selected:
        for key in (name, name.removesuffix("weight") + "scale"):
            path = model / index["weight_map"][key]
            if path not in headers:
                with path.open("rb") as file:
                    length = int.from_bytes(file.read(8), "little")
                    raw = file.read(length)
                headers[path] = json.loads(raw), 8 + length, hashlib.sha256(raw).hexdigest()
            header, start, digest = headers[path]
            entry = header[key]
            rows, width = entry["shape"]
            begin, end = entry["data_offsets"]
            if end - begin != rows * width or width % tp:
                raise ValueError(f"expected byte tensors with TP-divisible columns: {key}")
            cases.append(Case(key, str(path), start + begin, rows, width, entry["dtype"], digest))
    return cases


def mapped_libraries():
    return sorted({
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if any(name in line for name in ("/libcufile.so", "/libcudart.so", "/libcuda.so"))
    })


def build_batch_module(output, cuda):
    source = Path(__file__).with_name("_gds_shard_reads.c")
    target = output / ("_gds_shard_reads" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [
        "cc", "-O2", "-std=c99", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
        f"-I{sysconfig.get_path('include')}", f"-I{cuda / 'include'}", str(source),
        f"-L{cuda / 'lib64'}", f"-Wl,-rpath,{cuda / 'lib64'}", "-lcufile", "-lcudart", "-lcuda",
        "-o", str(target),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    (output / "native_build.json").write_text(json.dumps(command, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("_gds_shard_reads", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu_snapshot(devices):
    return subprocess.check_output([
        "nvidia-smi", "-i", ",".join(map(str, devices)),
        "--query-gpu=index,uuid,name,pstate,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu,clocks_event_reasons.active",
        "--format=csv,noheader,nounits",
    ], text=True).strip()


def block_snapshot():
    path = Path("/sys/block/md0/stat")
    return list(map(int, path.read_text().split())) if path.exists() else None


def delta(after, before):
    return {key: after[key] - before[key] for key in after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--io-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=("coalesced", "batched", "owner_exchange"),
                        default=["coalesced", "batched", "owner_exchange"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    devices, tp = args.devices, len(args.devices)
    if len(set(devices)) != tp or tp < 2 or args.rounds < 1:
        raise ValueError("select at least two distinct CUDA devices and a positive round count")
    args.output.mkdir(parents=True, exist_ok=False)
    for source in (Path(__file__), Path(__file__).with_name("_gds_shard_reads.c")):
        shutil.copyfile(source, args.output / source.name)
    cases = select_cases(args.model.resolve(), args.experts, tp)
    payload = sum(case.nbytes for case in cases)
    print(json.dumps({"scope": "actual W2 byte tensors; standalone transport comparison",
                      "devices": devices, "cases": len(cases), "unique_payload_bytes": payload}), flush=True)
    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "command": sys.argv, "executable": sys.executable, "worktree": str(repo),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "torch": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
        "cases": [asdict(case) for case in cases], "options": vars(args) | {"model": str(args.model), "output": str(args.output)},
        "unique_payload_bytes": payload,
        "semantics": "Wall time includes transfer submission and completion; setup, oracle, poison, and verification excluded. O_DIRECT; no cache dropping. Block counters include any concurrent host I/O.",
        "sources": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [Path(__file__), Path(__file__).with_name("_gds_shard_reads.c"),
                                 *sorted((repo / "b12x/loader").glob("_gds*")),
                                 repo / "b12x/loader/_checkpoint.py", repo / "b12x/loader/_cuda_range.h"] if path.is_file()},
        "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        "gpu_before_setup": gpu_snapshot(devices),
    }
    results = []
    disabled = set()
    with ExitStack() as stack:
        sessions, targets, full, owner_storage = [], [], [], []
        owner_capacity = max(sum(((cases[j].offset % 4096 + cases[j].nbytes + 4095) // 4096) * 4096
                                 for j in (i, i + 1)) for i in range(0, len(cases), 2))
        owner_capacity = (owner_capacity + 65535) // 65536 * 65536
        fds = {}
        for path in sorted({case.path for case in cases}):
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            fds[path] = fd
            stack.callback(os.close, fd)
        for device in devices:
            torch.cuda.set_device(device)
            session = stack.enter_context(DirectWeightSession(device, args.io_threads, read_mode="gds"))
            sessions.append(session)
            targets.append([torch.empty((c.rows, c.width // tp), dtype=torch.uint8, device=device) for c in cases])
            backing = torch.empty(owner_capacity + 65536, dtype=torch.uint8, device=device)
            alignment = (-backing.data_ptr()) % 65536
            owner_storage.append(backing[alignment:alignment + owner_capacity])
            full.append({})
            session._execute(array("Q"))
        gds = sessions[0]._gds
        gds.start_stats(3)
        native = build_batch_module(args.output, Path(os.environ.get("CUDA_HOME", "/opt/cuda")))
        manifest["mapped_libraries"] = mapped_libraries()
        manifest["native_artifacts"] = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [sessions[0].native.__file__, gds.__file__, native.__file__, *mapped_libraries()]
        }
        manifest["gds_version"] = sessions[0].stats()["gds_version"]
        manifest["transport_counter_limitation"] = "cuFile 1.15 reports batch bytes but zero NVFS/P2P/POSIX batch operation counters for unaligned rows. Those counters alone cannot attribute the internal unaligned route. Set CUFILE_ALLOW_COMPAT_MODE=false for strict GDS qualification."
        cudart = ctypes.CDLL(next(path for path in mapped_libraries() if "/libcudart.so" in path))
        cudart.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
        cudart.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
        for source in devices:
            torch.cuda.set_device(source)
            for target in devices:
                if source == target:
                    continue
                if not torch.cuda.can_device_access_peer(source, target):
                    raise RuntimeError(f"direct CUDA peer access unavailable: {source} -> {target}")
                code = cudart.cudaDeviceEnablePeerAccess(target, 0)
                if code not in (0, 704):
                    raise RuntimeError(f"cudaDeviceEnablePeerAccess {source}->{target}: {code}")
        streams = [torch.cuda.Stream(device=device) for device in devices]
        coalesced = []
        owner_readers = []
        owner_waves = [[] for _ in range(args.experts // tp)]
        batches = []
        for rank, device in enumerate(devices):
            rank_fds = {}
            for path in fds:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                rank_fds[path] = fd
                stack.callback(os.close, fd)
            coalesced.append(array("Q"))
            owner_reader = native.create(device, 1)
            owner_handles = {path: native.add_file(owner_reader, path) for path in fds}
            native.register_buffer(owner_reader, owner_storage[rank].data_ptr(), owner_capacity)
            owner_readers.append(owner_reader)
            lanes = []
            for lane in range(args.io_threads):
                capsule = native.create(device, args.batch_size)
                native.share_files(capsule, owner_reader)
                handles = owner_handles
                records = array("Q")
                for i, c in enumerate(cases):
                    begin = c.rows * lane // args.io_threads
                    end = c.rows * (lane + 1) // args.io_threads
                    width = c.width // tp
                    records.extend((handles[c.path], c.offset + begin * c.width + rank * width,
                                    width, targets[rank][i].data_ptr() + begin * width, 0,
                                    end - begin, c.width, width))
                lanes.append((capsule, records))
            batches.append(lanes)
            for i, c in enumerate(cases):
                width = c.width // tp
                coalesced[-1].extend((rank_fds[c.path], c.offset + rank * width, width,
                                     targets[rank][i].data_ptr(), 0, c.rows, c.width, width))
            for wave in range(args.experts // tp):
                instructions = []
                offset = 0
                for i in (2 * (wave * tp + rank), 2 * (wave * tp + rank) + 1):
                    c = cases[i]
                    prefix = c.offset % 4096
                    length = (prefix + c.nbytes + 4095) // 4096 * 4096
                    full[rank][i] = owner_storage[rank][offset + prefix:offset + prefix + c.nbytes].view(c.rows, c.width)
                    chunks = max(1, args.io_threads // 2)
                    chunk = ((length + chunks - 1) // chunks + 4095) // 4096 * 4096
                    for start in range(0, length, chunk):
                        instructions.append((owner_reader, owner_handles[c.path], c.offset - prefix + start,
                                             min(chunk, length - start), offset + start))
                    offset += length
                owner_waves[wave].append(instructions)
        # CPU reads are exclusively the byte oracle, outside every timed arm.
        expected = []
        for c in cases:
            with open(c.path, "rb", buffering=0) as file:
                raw = os.pread(file.fileno(), c.nbytes, c.offset)
            if len(raw) != c.nbytes:
                raise RuntimeError(f"short oracle read: {c.name}")
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(c.rows, c.width)
            if not torch.count_nonzero(tensor):
                raise RuntimeError(f"all-zero oracle: {c.name}")
            expected.append(tensor)
        manifest["oracle_sha256"] = {c.name: hashlib.sha256(t.numpy().tobytes()).hexdigest()
                                     for c, t in zip(cases, expected, strict=True)}
        manifest["application_staging_bytes"] = {
            "coalesced": sum(s.stats()["gpu_scratch_bytes"] for s in sessions),
            "batched": 0,
            "owner_exchange": tp * (owner_capacity + 65536),
        }
        manifest["staging_note"] = "Owner exchange reuses one full expert W2 weight+scale buffer per GPU, plus 4 KiB file and 64 KiB allocation alignment padding. Peer byte-gather kernels write final destinations without a packing buffer. Coalesced executor scratch remains allocated but is unused by the owner arm. cuFile internal buffers are not included."
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=tp * args.io_threads))

        def parallel(function, arguments):
            futures = [executor.submit(function, *argument) for argument in arguments]
            values, errors = [], []
            for future in futures:
                try:
                    values.append(future.result())
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]
            return values

        def sync():
            for device in devices:
                torch.cuda.synchronize(device)

        def poison():
            for rank in range(tp):
                with torch.cuda.device(devices[rank]):
                    for i, target in enumerate(targets[rank]):
                        width = cases[i].width // tp
                        target.copy_(expected[i][:, rank * width:(rank + 1) * width].bitwise_xor(255))
                    owner_storage[rank].fill_(165)
            sync()

        def exchange(wave):
            begin = time.perf_counter()
            for step in range(tp):
                for source in range(tp):
                    target = (source + step) % tp
                    for i in (2 * (wave * tp + source), 2 * (wave * tp + source) + 1):
                        c = cases[i]
                        width = c.width // tp
                        native.peer_gather(devices[source], sessions[source]._copy_programs[0].function,
                                           full[source][i].data_ptr() + target * width, targets[target][i].data_ptr(),
                                           width, c.rows, c.width, width, streams[source].cuda_stream)
                sync()
            return time.perf_counter() - begin

        def run(method, phase, repetition):
            poison()
            before_gpu = gpu_snapshot(devices)
            before_disk = block_snapshot()
            before_sync = gds.synchronous_transport_stats()
            before_batch = gds.transport_stats()
            before_sessions = [session.stats() for session in sessions]
            item = {"method": method, "phase": phase, "repetition": repetition, "gpu_before": before_gpu}
            begin = time.perf_counter()
            try:
                if method == "batched":
                    counters = parallel(native.execute, [lane for lanes in batches for lane in lanes])
                    sync()
                    item["read_seconds"] = time.perf_counter() - begin
                elif method == "owner_exchange":
                    item["read_seconds"] = item["exchange_seconds"] = 0
                    owner_bytes = 0
                    for wave, instructions in enumerate(owner_waves):
                        read_begin = time.perf_counter()
                        owner_bytes += sum(parallel(native.read_whole, [part for rank_parts in instructions for part in rank_parts]))
                        sync()
                        item["read_seconds"] += time.perf_counter() - read_begin
                        item["exchange_seconds"] += exchange(wave)
                    counters = []
                else:
                    parallel(lambda session, record: session._execute(record), zip(sessions, coalesced, strict=True))
                    sync()
                    item["read_seconds"] = time.perf_counter() - begin
                    counters = []
                item["seconds"] = time.perf_counter() - begin
                after_disk = block_snapshot()
                item["transport_sync"] = delta(gds.synchronous_transport_stats(), before_sync)
                item["transport_batch"] = delta(gds.transport_stats(), before_batch)
                item["gpu_after"] = gpu_snapshot(devices)
                item["api_read_bytes"] = (
                    sum(counter["api_read_bytes"] for counter in counters) if method == "batched" else owner_bytes if method == "owner_exchange" else
                    sum(session.stats()["gds_physical_bytes"] - before["gds_physical_bytes"]
                        for session, before in zip(sessions, before_sessions, strict=True))
                )
                item["reads"] = (
                    sum(counter["reads"] for counter in counters) if method == "batched" else
                    sum(len(rank_parts) for wave in owner_waves for rank_parts in wave) if method == "owner_exchange" else
                    sum(session.stats()["reads"] - before["reads"]
                        for session, before in zip(sessions, before_sessions, strict=True))
                )
                item["batch_lanes"] = counters
                item["md0_read_bytes"] = (after_disk[2] - before_disk[2]) * 512 if before_disk else None
                item["peer_bytes"] = payload * (tp - 1) // tp if method == "owner_exchange" else 0
                item["useful_GB_per_second"] = payload / item["seconds"] / 1e9
                item["api_read_amplification"] = item["api_read_bytes"] / payload
                for rank in range(tp):
                    for i, target in enumerate(targets[rank]):
                        width = cases[i].width // tp
                        if not torch.equal(target.cpu(), expected[i][:, rank * width:(rank + 1) * width]):
                            raise RuntimeError(f"byte mismatch: TP rank {rank}, {cases[i].name}")
                if item["transport_sync"]["posix_reads"] or item["transport_batch"]["posix_ops"]:
                    raise RuntimeError("cuFile used POSIX compatibility; strict GDS qualification failed")
                if item["transport_sync"]["read_errors"] or item["transport_batch"]["read_errors"]:
                    raise RuntimeError("cuFile transport reported read errors")
                item["correctness"] = "all destination bytes equal checkpoint TP slice; complemented poison overwritten"
            except Exception as error:
                item["error"] = repr(error)
                disabled.add(method)
            results.append(item)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({key: value for key, value in item.items()
                              if key not in ("gpu_before", "gpu_after", "batch_lanes")}), flush=True)

        for method in args.methods:
            run(method, "qualification", 0)
        # Rotate and reverse arm order to expose cache and temporal asymmetry.
        for repetition in range(args.rounds):
            ordered = args.methods[repetition % len(args.methods):] + args.methods[:repetition % len(args.methods)]
            if repetition % 2:
                ordered = list(reversed(ordered))
            for method in ordered:
                if method not in disabled:
                    run(method, "measurement", repetition)
        summary = {}
        for method in args.methods:
            samples = [r for r in results if r["method"] == method and r["phase"] == "measurement" and "error" not in r]
            if not samples:
                summary[method] = {"qualified": False}
                continue
            median = statistics.median(r["seconds"] for r in samples)
            summary[method] = {"qualified": True, "median_seconds": median,
                               "useful_GB_per_second": payload / median / 1e9,
                               "api_read_amplification": statistics.median(r["api_read_amplification"] for r in samples),
                               "seconds": [r["seconds"] for r in samples]}
        baseline = summary.get("coalesced", {}).get("median_seconds")
        if baseline:
            for item in summary.values():
                if item["qualified"]:
                    item["speedup_baseline_seconds_over_method_seconds"] = baseline / item["median_seconds"]
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        log = Path("cufile.log")
        if log.exists():
            lines = [line for line in log.read_text().splitlines() if f"pid={os.getpid()} " in line]
            (args.output / "cufile.log").write_text("\n".join(lines) + "\n")
        print(json.dumps({"summary": summary, "output": str(args.output)}), flush=True)
        # Release cuFile batch contexts before their GPU tensor destinations.
        batches.clear()
        owner_readers.clear()
        owner_waves.clear()
        instructions.clear()
        owner_reader = None
        lanes.clear()
        capsule = None
    return 1 if disabled else 0


if __name__ == "__main__":
    raise SystemExit(main())
