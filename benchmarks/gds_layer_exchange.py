"""Compare file-ordered complete MoE experts with bounded GPU owner buffers.

The corpus consists of routed-expert W1/W2/W3 weights and scales. Dense,
attention, vision, draft-model, and offloaded embedding tensors are excluded.
This standalone transport experiment does not initialize a serving engine.
"""

from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from b12x.loader._checkpoint import DirectWeightSession

from gds_shard_exchange import block_snapshot, build_batch_module, delta, gpu_snapshot, mapped_libraries


@dataclass
class Tensor:
    name: str
    path: str
    offset: int
    rows: int
    width: int
    axis: int
    dtype: str
    destination_offset: int = 0
    buffer_offset: int = 0

    @property
    def nbytes(self):
        return self.rows * self.width

    def shard(self, rank, tp):
        rows = self.rows // tp if self.axis == 0 else self.rows
        width = self.width // tp if self.axis == 1 else self.width
        source = rank * (rows * width if self.axis == 0 else width)
        return rows, width, source


def inventory(model, selected_layers, tp):
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    pattern = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.weight$")
    headers, experts = {}, []
    for name in index:
        match = pattern.fullmatch(name)
        if not match:
            continue
        layer, expert = map(int, match.groups())
        if selected_layers is not None and layer not in selected_layers:
            continue
        tensors = []
        for kind in ("weight", "scale"):
            for matrix in ("w1", "w2", "w3"):
                key = f"layers.{layer}.ffn.experts.{expert}.{matrix}.{kind}"
                path = str(model / index[key])
                if path not in headers:
                    with open(path, "rb") as file:
                        size = int.from_bytes(file.read(8), "little")
                        raw = file.read(size)
                    headers[path] = (json.loads(raw), 8 + size, hashlib.sha256(raw).hexdigest())
                header, start, _ = headers[path]
                entry = header[key]
                rows, width = entry["shape"]
                begin, end = entry["data_offsets"]
                axis = 1 if matrix == "w2" else 0
                if end - begin != rows * width or (rows if axis == 0 else width) % tp:
                    raise ValueError(f"unsupported expert byte geometry: {key}")
                tensors.append(Tensor(key, path, start + begin, rows, width, axis, entry["dtype"]))
        # Only exactly adjacent tensor payloads share a source span.
        spans = []
        for tensor in sorted(tensors, key=lambda t: (t.path, t.offset)):
            if spans and spans[-1]["path"] == tensor.path and spans[-1]["end"] == tensor.offset:
                spans[-1]["end"] += tensor.nbytes
                spans[-1]["tensors"].append(tensor)
            else:
                spans.append(dict(path=tensor.path, offset=tensor.offset,
                                  end=tensor.offset + tensor.nbytes, tensors=[tensor]))
        buffer_offset = 0
        for span in spans:
            start = span["offset"] // 4096 * 4096
            size = (span["end"] + 4095) // 4096 * 4096 - start
            for tensor in span["tensors"]:
                tensor.buffer_offset = buffer_offset + tensor.offset - start
            span.update(aligned_offset=start, size=size, buffer_offset=buffer_offset)
            del span["tensors"]
            buffer_offset += size
        experts.append(dict(layer=layer, expert=expert, tensors=tensors, spans=spans,
                            capacity=buffer_offset))
    experts.sort(key=lambda e: (e["layer"], e["tensors"][0].path, e["tensors"][0].offset))
    layers = sorted({e["layer"] for e in experts})
    if selected_layers is not None and set(layers) != set(selected_layers):
        raise ValueError("selected layer is absent from the routed-expert corpus")
    if not experts or any(sum(e["layer"] == layer for e in experts) % tp for layer in layers):
        raise ValueError("each layer must contain a positive multiple of TP experts")
    destination = 0
    for expert in experts:
        for tensor in expert["tensors"]:
            tensor.destination_offset = destination
            destination += tensor.nbytes // tp
    return experts, {path: digest for path, (_, _, digest) in headers.items()}, destination


def chunk_fragments(expert, path, start, size, tp):
    fragments = []
    for tensor in expert["tensors"]:
        if tensor.path != path:
            continue
        begin = max(start, tensor.offset) - tensor.offset
        end = min(start + size, tensor.offset + tensor.nbytes) - tensor.offset
        if end <= begin:
            continue
        for rank in range(tp):
            rows, width, shard_offset = tensor.shard(rank, tp)
            if tensor.axis == 0:
                low, high = max(begin, shard_offset), min(end, shard_offset + rows * width)
                if high > low:
                    fragments.append((rank, tensor.offset + low - start,
                                      tensor.destination_offset + low - shard_offset, high - low, 1, 0, 0))
                continue
            first, last = begin // tensor.width, (end - 1) // tensor.width
            pieces = []

            def add(row, low, high, count=1):
                low, high = max(low, shard_offset), min(high, shard_offset + width)
                if high <= low or count <= 0:
                    return
                source = tensor.offset + row * tensor.width + low - start
                destination = tensor.destination_offset + row * width + low - shard_offset
                if (pieces and high - low == width and pieces[-1][3] == width
                        and pieces[-1][1] + pieces[-1][4] * tensor.width == source
                        and pieces[-1][2] + pieces[-1][4] * width == destination):
                    previous = pieces[-1]
                    pieces[-1] = (*previous[:4], previous[4] + count, tensor.width, width)
                else:
                    pieces.append((rank, source, destination, high - low, count, tensor.width, width))

            if first == last:
                add(first, begin - first * tensor.width, end - first * tensor.width)
            else:
                add(first, begin - first * tensor.width, tensor.width)
                add(first + 1, 0, tensor.width, last - first - 1)
                add(last, 0, end - last * tensor.width)
            fragments.extend(pieces)
    return fragments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--all-expert-layers", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--io-threads", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=("coalesced", "owner_exchange", "owner_pipelined", "owner_processes", "shared_native",
                                                        "instanttensor_default", "instanttensor_cufile"),
                        default=["coalesced", "owner_exchange", "owner_pipelined"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.layers is None) == (not args.all_expert_layers):
        raise ValueError("select --layers or --all-expert-layers")
    devices, tp = args.devices, len(args.devices)
    distributed_owner = any(method in args.methods for method in ("owner_processes", "shared_native"))
    if "owner_processes" in args.methods and "shared_native" in args.methods:
        raise ValueError("select one independent-process implementation per run")
    lifecycle = distributed_owner or any(method.startswith("instanttensor_") for method in args.methods)
    if lifecycle and any(method in args.methods for method in ("coalesced", "owner_exchange")):
        raise ValueError("reader-lifetime comparison supports owner_pipelined, owner_processes, and InstantTensor methods")
    if tp < 2 or len(set(devices)) != tp or args.rounds < 1 or not 2 <= args.io_threads <= 16:
        raise ValueError("invalid device, round, or concurrency contract")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), Path(__file__).with_name("gds_shard_exchange.py"),
               Path(__file__).with_name("_gds_shard_reads.c"), Path(__file__).with_name("_instanttensor_compare.py"),
               Path(__file__).with_name("_gds_owner_compare.py"), Path(__file__).with_name("_gds_shared_compare.py")]
    runtime_sources = [p for p in sorted((repo / "b12x/loader").glob("*"))
                       if p.is_file() and p.suffix in (".py", ".c", ".h")]
    for source in sources:
        shutil.copyfile(source, args.output / source.name)
    experts, headers, destination_bytes = inventory(args.model.resolve(), args.layers, tp)
    layers = sorted({e["layer"] for e in experts})
    payload = destination_bytes * tp
    capacity = (max(e["capacity"] for e in experts) + 65535) // 65536 * 65536
    print(json.dumps(dict(stage="inventory", layers=layers, experts=len(experts), tensors=len(experts) * 6,
                          unique_payload_bytes=payload, destination_bytes_per_gpu=destination_bytes,
                          owner_capacity_per_gpu=capacity + 65536)), flush=True)
    manifest = dict(scope=__doc__, command=sys.argv, executable=sys.executable, worktree=str(repo),
                    revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    torch=torch.__version__, devices=devices, pid=os.getpid(), layers=layers,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    experts=[dict(e, tensors=[asdict(t) for t in e["tensors"]]) for e in experts],
                    header_sha256=headers, unique_payload_bytes=payload,
                    destination_bytes_per_gpu=destination_bytes, owner_staging_bytes_per_gpu=capacity + 65536,
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources + runtime_sources},
                    gpu_before=gpu_snapshot(devices),
                    topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
                    semantics="Continuous transfer wall time. Setup, poison, and byte verification excluded. O_DIRECT; no cache dropping. md0 counters include unrelated I/O. One full expert staging buffer per GPU, reused after peer completion. Final TP outputs for all selected layers remain allocated.",
                    options=dict(io_threads=args.io_threads, rounds=args.rounds, methods=args.methods))
    if lifecycle:
        manifest["semantics"] = (
            "Reader initialization through final TP destination completion and reader cleanup. "
            "Final target allocation, CUDA/NCCL initialization, immutable source metadata, kernel compilation, "
            "poison and exact byte verification are excluded. Original checkpoint paths and selected tensor offsets. "
            "InstantTensor uses its unmodified native range API with safe_open's default I/O/buffer policy, "
            "borrowed tensors consumed immediately, and one process per GPU. Buffers are reported per method. "
            "owner_processes uses one rank per GPU owning its final weights; all arms share those weights through CUDA IPC. "
            "O_DIRECT; no cache dropping. md0 counters include unrelated I/O."
        )
    if "shared_native" in args.methods:
        manifest["shared_native"] = dict(
            staging_bytes_per_gpu=18677760, slots_per_gpu=4, slot_bytes=4653056,
            source_partition="Aligned union of routed source envelopes, round-robin bounded chunks.",
            timing="Includes native initialization, collective planning, IPC mapping, completion and cleanup; descriptors and final targets are prepared before timing.",
        )
        manifest["owner_staging_bytes_per_gpu"] = None
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results, oracle_hashes = [], {}
    with ExitStack() as stack:
        sessions, targets, buffers, owner_readers, records = [], [], [], [], []
        for device in devices:
            torch.cuda.set_device(device)
            session = stack.enter_context(DirectWeightSession(device, args.io_threads, read_mode="gds"))
            sessions.append(session)
            if not distributed_owner:
                targets.append(torch.empty(destination_bytes, device=device, dtype=torch.uint8))
            if not lifecycle:
                backing = torch.empty(capacity + 65536, device=device, dtype=torch.uint8)
                start = -backing.data_ptr() % 65536
                buffers.append(backing[start:start + capacity])
            else:
                buffers.append(None)
            session._execute(array("Q"))
        gds = sessions[0]._gds
        gds.start_stats(3)
        native = build_batch_module(args.output, Path(os.environ.get("CUDA_HOME", "/opt/cuda")))
        owner_ranks = None
        if distributed_owner:
            from _gds_owner_compare import OwnerRanks

            worker_options = {}
            if "shared_native" in args.methods:
                from _gds_shared_compare import worker

                worker_options = dict(worker=worker, worker_args=(f"file://{args.output.resolve() / 'shared-read-group'}",))
            owner_ranks = OwnerRanks(devices, experts, destination_bytes, capacity, args.io_threads, native, **worker_options)
            stack.callback(owner_ranks.close)
            targets.extend(owner_ranks.targets)
            manifest["owner_processes"] = owner_ranks.identity
            print(json.dumps(dict(stage="owner_processes_ready", ranks=owner_ranks.identity)), flush=True)
        manifest["gds_version"] = sessions[0].stats()["gds_version"]
        manifest["native_artifacts"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                       for p in [gds.__file__, sessions[0].native.__file__, native.__file__, *mapped_libraries()]}
        manifest["baseline_scratch_bytes"] = sum(s.stats()["gpu_scratch_bytes"] for s in sessions)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        cudart = ctypes.CDLL(next(p for p in mapped_libraries() if "/libcudart.so" in p))
        cudart.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
        for device in devices:
            torch.cuda.set_device(device)
            for other in devices:
                if other != device:
                    if not torch.cuda.can_device_access_peer(device, other):
                        raise RuntimeError(f"peer access unavailable: {device}->{other}")
                    status = cudart.cudaDeviceEnablePeerAccess(other, 0)
                    if status not in (0, 704):
                        raise RuntimeError(f"enable peer access: CUDA {status}")
        paths = sorted(headers)
        destination_pointers = (owner_ranks.destinations if owner_ranks is not None else
                                [[target.data_ptr() for target in targets] for _ in devices])
        owner_handles = []
        cpu_fds = {}
        for path in paths:
            fd = os.open(path, os.O_RDONLY)
            cpu_fds[path] = fd
            stack.callback(os.close, fd)
        for rank, device in enumerate(devices):
            if not lifecycle:
                reader = native.create(device, 1)
                owner_handles.append({path: native.add_file(reader, path) for path in paths})
                native.register_buffer(reader, buffers[rank].data_ptr(), capacity)
            else:
                reader = None
                owner_handles.append({path: index for index, path in enumerate(paths)})
            owner_readers.append(reader)
            fds = {}
            for path in paths:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                fds[path] = fd
                stack.callback(os.close, fd)
            record = array("Q")
            for expert in experts:
                for tensor in expert["tensors"]:
                    rows, width, offset = tensor.shard(rank, tp)
                    pointer = targets[rank].data_ptr() + tensor.destination_offset
                    if tensor.axis == 0:
                        record.extend((fds[tensor.path], tensor.offset + offset, rows * width, pointer, 0, 1, 0, 0))
                    else:
                        record.extend((fds[tensor.path], tensor.offset + offset, width, pointer, 0, rows, tensor.width, width))
            records.append(record)
        waves = []
        for begin in range(0, len(experts), tp):
            group = experts[begin:begin + tp]
            if len({expert["layer"] for expert in group}) != 1:
                raise RuntimeError("owner wave crosses a layer boundary")
            reads = []
            for owner, expert in enumerate(group):
                chunks = max(1, args.io_threads // len(expert["spans"]))
                for span in expert["spans"]:
                    length = ((span["size"] + chunks - 1) // chunks + 4095) // 4096 * 4096
                    for offset in range(0, span["size"], length):
                        reads.append((owner_readers[owner], owner_handles[owner][span["path"]],
                                      span["aligned_offset"] + offset, min(length, span["size"] - offset),
                                      span["buffer_offset"] + offset))
            waves.append((group, reads))
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=tp * args.io_threads))
        streams = [torch.cuda.Stream(device=device) for device in devices]
        slot_bytes = capacity // tp // 65536 * 65536
        pipeline = [[] for _ in devices]
        for index, expert in enumerate(experts):
            owner = index % tp
            for span in expert["spans"]:
                for offset in range(0, span["size"], slot_bytes):
                    start = span["aligned_offset"] + offset
                    size = min(slot_bytes, span["size"] - offset)
                    fragments = chunk_fragments(expert, span["path"], start, size, tp)
                    pipeline[owner].append((owner_handles[owner][span["path"]], start, size, fragments))
        slots = []
        for owner, device in enumerate(devices):
            with torch.cuda.device(device):
                for slot in range(tp):
                    stream = torch.cuda.Stream(device=device)
                    event = torch.cuda.Event()
                    event.record(stream)
                    event.synchronize()
                    slots.append(dict(owner=owner, index=slot, stream=stream, event=event,
                                      state="idle", futures=[], plan=None))
        manifest["pipeline"] = dict(slots_per_gpu=tp, slot_bytes=slot_bytes,
                                    methods=["owner_pipelined", "owner_processes"],
                                    chunk_count=sum(map(len, pipeline)),
                                    maximum_read_tasks_per_gpu=tp * max(1, args.io_threads // tp),
                                    memory="Slots subdivide the same registered owner buffer; no extra payload buffer.")
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        instant = None
        if any(method.startswith("instanttensor_") for method in args.methods):
            from _instanttensor_compare import InstantTensorRanks

            instant = InstantTensorRanks(devices, targets, experts, native, args.output)
            stack.callback(instant.close)
            manifest["instanttensor"] = instant.identity
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(dict(stage="instanttensor_ready", ranks=instant.identity)), flush=True)

        def parallel(function, arguments):
            futures = [executor.submit(function, *a) for a in arguments]
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

        def run_pipeline():
            next_plan = [0] * tp
            completed = transferred = 0
            read_task_seconds = 0
            expected = sum(map(len, pipeline))

            def read_part(*arguments):
                begin = time.perf_counter()
                received = native.read_whole(*arguments)
                return received, time.perf_counter() - begin

            try:
                while completed < expected:
                    progress = False
                    for slot in slots:
                        owner = slot["owner"]
                        if slot["state"] == "reading" and all(f.done() for f in slot["futures"]):
                            for future in slot["futures"]:
                                received, elapsed = future.result()
                                transferred += received
                                read_task_seconds += elapsed
                            slot["futures"] = []
                            for rank, source, destination, width, rows, source_stride, destination_stride in slot["plan"][3]:
                                native.peer_gather(devices[owner], sessions[owner]._copy_programs[0].function,
                                                   buffers[owner].data_ptr() + slot["index"] * slot_bytes + source,
                                                   destination_pointers[owner][rank] + destination, width, rows,
                                                   source_stride, destination_stride, slot["stream"].cuda_stream)
                            slot["event"].record(slot["stream"])
                            slot["state"] = "exchanging"
                            progress = True
                        if slot["state"] == "exchanging" and slot["event"].query():
                            slot["state"] = "idle"
                            slot["plan"] = None
                            completed += 1
                            progress = True
                        if slot["state"] == "idle" and next_plan[owner] < len(pipeline[owner]):
                            plan = pipeline[owner][next_plan[owner]]
                            next_plan[owner] += 1
                            handle, offset, size, _ = plan
                            task_count = max(1, args.io_threads // tp)
                            chunk = ((size + task_count - 1) // task_count + 4095) // 4096 * 4096
                            slot["futures"] = [executor.submit(
                                read_part, owner_readers[owner], handle, offset + part,
                                min(chunk, size - part), slot["index"] * slot_bytes + part,
                            ) for part in range(0, size, chunk)]
                            slot["plan"] = plan
                            slot["state"] = "reading"
                            progress = True
                    if not progress:
                        time.sleep(0.00001)
            finally:
                errors = []
                for slot in slots:
                    for future in slot["futures"]:
                        try:
                            future.result()
                        except Exception as error:
                            errors.append(error)
                sync()
                if errors:
                    raise errors[0]
            return transferred, read_task_seconds

        def verify(method, collect_hashes):
            for layer in layers:
                selected = [expert for expert in experts if expert["layer"] == layer]
                begin = selected[0]["tensors"][0].destination_offset
                last = selected[-1]["tensors"][-1]
                end = last.destination_offset + last.nbytes // tp
                cpu_targets = [target[begin:end].bitwise_xor(0).cpu().numpy() for target in targets]
                for expert in selected:
                    for tensor in expert["tensors"]:
                        raw = os.pread(cpu_fds[tensor.path], tensor.nbytes, tensor.offset)
                        if len(raw) != tensor.nbytes:
                            raise RuntimeError(f"short oracle read: {tensor.name}")
                        expected = np.frombuffer(raw, np.uint8).reshape(tensor.rows, tensor.width)
                        if not expected.any():
                            raise RuntimeError(f"all-zero checkpoint tensor: {tensor.name}")
                        if collect_hashes:
                            oracle_hashes[tensor.name] = hashlib.sha256(raw).hexdigest()
                        for rank in range(tp):
                            rows, width, _ = tensor.shard(rank, tp)
                            offset = tensor.destination_offset - begin
                            actual = cpu_targets[rank][offset:offset + rows * width].reshape(rows, width)
                            reference = (expected[rank * rows:(rank + 1) * rows, :] if tensor.axis == 0
                                         else expected[:, rank * width:(rank + 1) * width])
                            if not np.array_equal(actual, reference):
                                raise RuntimeError(f"byte mismatch: {method}, rank {rank}, {tensor.name}")
                del cpu_targets
                print(json.dumps(dict(stage="verified", method=method, layer=layer)), flush=True)

        def run(method, phase, repetition):
            for rank, target in enumerate(targets):
                with torch.cuda.device(devices[rank]):
                    if results:
                        target.bitwise_not_()
                    else:
                        target.fill_(165)
                    if buffers[rank] is not None:
                        buffers[rank].fill_(90)
            sync()
            before = [session.stats() for session in sessions]
            before_gpu = gpu_snapshot(devices)
            before_disk = block_snapshot()
            before_transport = gds.synchronous_transport_stats()
            read_seconds = exchange_seconds = 0
            transferred = 0
            start = time.perf_counter()
            if method == "coalesced":
                parallel(lambda session, record: session._execute(record), zip(sessions, records, strict=True))
                sync()
                read_seconds = time.perf_counter() - start
            elif method == "owner_pipelined":
                if lifecycle:
                    for owner, device in enumerate(devices):
                        with torch.cuda.device(device):
                            backing = torch.empty(capacity + 65536, device=device, dtype=torch.uint8)
                            aligned = -backing.data_ptr() % 65536
                            buffers[owner] = backing[aligned:aligned + capacity]
                            reader = native.create(device, 1)
                            for path, index in owner_handles[owner].items():
                                if native.add_file(reader, path) != index:
                                    raise RuntimeError("owner file inventory changed")
                            native.register_buffer(reader, buffers[owner].data_ptr(), capacity)
                            owner_readers[owner] = reader
                    backing = reader = None
                owner_ready = time.perf_counter()
                try:
                    transferred, read_task_seconds = run_pipeline()
                    owner_transferred = time.perf_counter()
                finally:
                    if lifecycle:
                        for owner in range(tp):
                            if owner_readers[owner] is not None:
                                native.close(owner_readers[owner])
                                owner_readers[owner] = None
                            buffers[owner] = None
                read_seconds = exchange_seconds = None
            elif method.startswith("instanttensor_"):
                instant_result = instant.run(method)
                read_seconds = exchange_seconds = None
            elif method in ("owner_processes", "shared_native"):
                owner_result = owner_ranks.run()
                read_seconds = exchange_seconds = None
            else:
                for group, reads in waves:
                    tick = time.perf_counter()
                    transferred += sum(parallel(native.read_whole, reads))
                    read_seconds += time.perf_counter() - tick
                    tick = time.perf_counter()
                    for step in range(tp):
                        for owner, expert in enumerate(group):
                            rank = (owner + step) % tp
                            for tensor in expert["tensors"]:
                                rows, width, offset = tensor.shard(rank, tp)
                                native.peer_gather(devices[owner], sessions[owner]._copy_programs[0].function,
                                                   buffers[owner].data_ptr() + tensor.buffer_offset + offset,
                                                   targets[rank].data_ptr() + tensor.destination_offset,
                                                   width, rows, tensor.width, width, streams[owner].cuda_stream)
                        sync()
                    exchange_seconds += time.perf_counter() - tick
            elapsed = time.perf_counter() - start
            if method.startswith("instanttensor_"):
                command_seconds = elapsed
                elapsed = instant_result["seconds"]
            elif method in ("owner_processes", "shared_native"):
                command_seconds = elapsed
                elapsed = owner_result["seconds"]
            after_disk = block_snapshot()
            transport = delta(gds.synchronous_transport_stats(), before_transport)
            if method == "instanttensor_cufile":
                transport = {key: sum(rank["transport"][key] for rank in instant_result["ranks"])
                             for key in transport}
                transferred = transport["read_bytes"]
            elif method == "instanttensor_default":
                transport = None
                transferred = None
            elif method in ("owner_processes", "shared_native"):
                transport = {key: sum(rank["transport"][key] for rank in owner_result["ranks"])
                             for key in transport}
                transferred = sum(rank["api_read_bytes"] for rank in owner_result["ranks"])
            if transport is not None and (transport["posix_reads"] or transport["read_errors"] or not transport["nvfs_reads"] + transport["p2p_reads"]):
                raise RuntimeError(f"strict GPU transport qualification failed: {transport}")
            if method == "coalesced":
                transferred = sum(s.stats()["gds_physical_bytes"] - b["gds_physical_bytes"]
                                  for s, b in zip(sessions, before, strict=True))
            result = dict(method=method, phase=phase, repetition=repetition, seconds=elapsed,
                          read_seconds=read_seconds, exchange_seconds=exchange_seconds,
                          useful_GB_per_second=payload / elapsed / 1e9, api_read_bytes=transferred,
                          md0_read_bytes=(after_disk[2] - before_disk[2]) * 512,
                          peer_bytes=(None if method.startswith("instanttensor_") else
                                      payload * (tp - 1) // tp if method != "coalesced" else 0),
                          transport=transport, gpu_before=before_gpu, gpu_after=gpu_snapshot(devices),
                          poison="bitwise complement of verified outputs" if results else "0xa5 initialization")
            if method == "owner_pipelined":
                result["read_task_seconds_sum"] = read_task_seconds
                result["stage_time_note"] = "Read and exchange overlap; aggregate read task durations are not wall time."
                if lifecycle:
                    result["reader_setup_seconds"] = owner_ready - start
                    result["transfer_seconds"] = owner_transferred - owner_ready
                    result["reader_cleanup_seconds"] = elapsed - (owner_transferred - start)
            if method.startswith("instanttensor_"):
                result["ranks"] = instant_result["ranks"]
                result["command_seconds"] = command_seconds
                result["stage_time_note"] = "open_seconds overlaps library prefetch and is included; never subtracted."
            elif method in ("owner_processes", "shared_native"):
                result["ranks"] = owner_result["ranks"]
                if method == "shared_native":
                    result["peer_bytes"] = sum(rank["counters"]["peer_bytes"] for rank in owner_result["ranks"])
                result["command_seconds"] = command_seconds
                result["stage_time_note"] = "Earliest rank initialization to latest rank cleanup; read and exchange overlap."
            print(json.dumps(dict(stage="transferred", **{k: v for k, v in result.items() if not k.startswith("gpu_")})), flush=True)
            verify(method, not results)
            result["correctness"] = "every W1/W2/W3 weight and scale byte equals its checkpoint TP slice"
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            if len(results) == 1:
                (args.output / "oracle_sha256.json").write_text(json.dumps(oracle_hashes, indent=2) + "\n")

        methods = args.methods
        for method in methods:
            run(method, "qualification", 0)
        for repetition in range(args.rounds):
            for method in methods if repetition % 2 == 0 else methods[::-1]:
                run(method, "measurement", repetition)
        summary = {}
        for method in methods:
            samples = [r for r in results if r["method"] == method and r["phase"] == "measurement"]
            median = statistics.median(r["seconds"] for r in samples)
            summary[method] = dict(median_seconds=median, useful_GB_per_second=payload / median / 1e9,
                                   seconds=[r["seconds"] for r in samples],
                                   median_api_read_bytes=(statistics.median(r["api_read_bytes"] for r in samples)
                                                          if samples[0]["api_read_bytes"] is not None else None),
                                   median_md0_read_bytes=statistics.median(r["md0_read_bytes"] for r in samples))
        if "coalesced" in summary:
            for method in methods:
                summary[method]["speedup_baseline_seconds_over_method_seconds"] = summary["coalesced"]["median_seconds"] / summary[method]["median_seconds"]
        elif "owner_pipelined" in summary:
            for method in methods:
                summary[method]["owner_seconds_over_method_seconds"] = summary["owner_pipelined"]["median_seconds"] / summary[method]["median_seconds"]
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(dict(stage="complete", summary=summary, output=str(args.output))), flush=True)
        waves.clear()
        reads.clear()
        owner_readers.clear()
        reader = None
        targets.clear()
        target = None


if __name__ == "__main__":
    main()
