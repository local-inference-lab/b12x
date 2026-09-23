"""One owner process per GPU for the bounded checkpoint transport benchmark.

Each rank owns its final allocation and exports it to peer ranks and the
benchmark controller. Only control messages cross CPU pipes; payload writes
use CUDA IPC mappings. The controller fences all ranks before consuming output.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

from _instanttensor_compare import _load_native


class _CudaBytes:
    def __init__(self, pointer, size):
        self.__cuda_array_interface__ = dict(
            shape=(size,), strides=None, typestr="|u1", data=(pointer, False), version=3,
        )


def _worker(connection, barrier, rank, devices, native_path, experts, destination_bytes, capacity, io_threads):
    try:
        import torch
        from b12x.loader._gds_kernels import compile_copies
        from b12x.loader._gds_native import load

        from gds_layer_exchange import chunk_fragments

        torch.set_num_threads(1)
        device, tp = devices[rank], len(devices)
        torch.cuda.set_device(device)
        gds = load()
        native = _load_native(native_path)
        program = compile_copies(device)[0]
        cufile_path = next(line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                           if "/libcufile.so" in line)

        class CufileStatus(ctypes.Structure):
            _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]

        cufile = ctypes.CDLL(cufile_path)
        cufile.cuFileDriverOpen.restype = CufileStatus
        status = cufile.cuFileDriverOpen()
        if status.err:
            raise RuntimeError(f"cuFileDriverOpen: {status.err}, CUDA {status.cu_err}")
        gds.start_stats(3)
        paths = sorted({span["path"] for expert in experts for span in expert["spans"]})
        file_ids = {path: index for index, path in enumerate(paths)}
        slot_bytes = capacity // tp // 65536 * 65536
        plans = []
        for index, expert in enumerate(experts):
            if index % tp != rank:
                continue
            for span in expert["spans"]:
                for offset in range(0, span["size"], slot_bytes):
                    start = span["aligned_offset"] + offset
                    size = min(slot_bytes, span["size"] - offset)
                    plans.append((file_ids[span["path"]], start, size,
                                  chunk_fragments(expert, span["path"], start, size, tp)))
        slots = []
        for index in range(tp):
            stream, event = torch.cuda.Stream(device=device), torch.cuda.Event()
            event.record(stream)
            event.synchronize()
            slots.append(dict(index=index, stream=stream, event=event, state="idle", futures=[], plan=None))
        with ThreadPoolExecutor(max_workers=io_threads) as executor:
            target = torch.empty(destination_bytes, device=device, dtype=torch.uint8)
            ipc = native.export_ipc(device, target.data_ptr())
            connection.send(("allocated", ipc))
            command, all_ipc = connection.recv()
            if command != "map":
                raise RuntimeError(f"expected map, received {command}")
            destinations, imports = [], []
            for peer, handle in enumerate(all_ipc):
                if peer == rank:
                    destinations.append(target.data_ptr())
                else:
                    if not torch.cuda.can_device_access_peer(device, devices[peer]):
                        raise RuntimeError(f"peer access unavailable: {device}->{devices[peer]}")
                    base = native.import_ipc(device, handle[0])
                    imports.append(base)
                    destinations.append(base + handle[1])
            torch.cuda.synchronize(device)
            libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                if any(key in line for key in ("libcufile.so", "libcudart.so", "libcuda.so"))})
            identity = dict(pid=os.getpid(), rank=rank, device=device, chunks=len(plans),
                            final_destination_bytes=target.nbytes, final_destination_owner="rank process",
                            slot_bytes=slot_bytes, slots=tp, staging_bytes=capacity + 65536,
                            maximum_read_tasks=tp * max(1, io_threads // tp),
                            hashes={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                    for p in [gds.__file__, native.__file__, *libraries]})
            connection.send(("ready", identity))
            while True:
                command = connection.recv()
                if command == "unmap":
                    torch.cuda.synchronize(device)
                    for base in imports:
                        native.close_ipc(device, base)
                    imports.clear()
                    connection.send(("unmapped", None))
                    if connection.recv() != "close":
                        raise RuntimeError("expected close after unmap")
                    break
                if command != "run":
                    raise RuntimeError(f"unknown owner command: {command}")
                before = gds.synchronous_transport_stats()
                barrier.wait(timeout=300)
                started = time.perf_counter()
                reader = backing = buffer = None
                transferred = completed = next_plan = 0
                read_task_seconds = 0.0

                def read_part(*arguments):
                    begin = time.perf_counter()
                    count = native.read_whole(*arguments)
                    return count, time.perf_counter() - begin

                try:
                    backing = torch.empty(capacity + 65536, device=device, dtype=torch.uint8)
                    aligned = -backing.data_ptr() % 65536
                    buffer = backing[aligned:aligned + capacity]
                    reader = native.create(device, 1)
                    for path, index in file_ids.items():
                        if native.add_file(reader, path) != index:
                            raise RuntimeError("owner file inventory changed")
                    native.register_buffer(reader, buffer.data_ptr(), capacity)
                    ready = time.perf_counter()
                    while completed < len(plans):
                        progress = False
                        for slot in slots:
                            if slot["state"] == "reading" and all(f.done() for f in slot["futures"]):
                                for future in slot["futures"]:
                                    count, elapsed = future.result()
                                    transferred += count
                                    read_task_seconds += elapsed
                                slot["futures"] = []
                                for peer, source, destination, width, rows, source_stride, destination_stride in slot["plan"][3]:
                                    native.peer_gather(device, program.function,
                                                       buffer.data_ptr() + slot["index"] * slot_bytes + source,
                                                       destinations[peer] + destination, width, rows,
                                                       source_stride, destination_stride, slot["stream"].cuda_stream)
                                slot["event"].record(slot["stream"])
                                slot["state"] = "exchanging"
                                progress = True
                            if slot["state"] == "exchanging" and slot["event"].query():
                                slot["state"] = "idle"
                                slot["plan"] = None
                                completed += 1
                                progress = True
                            if slot["state"] == "idle" and next_plan < len(plans):
                                plan = plans[next_plan]
                                next_plan += 1
                                handle, offset, size, _ = plan
                                task_count = max(1, io_threads // tp)
                                chunk = ((size + task_count - 1) // task_count + 4095) // 4096 * 4096
                                slot["futures"] = [executor.submit(
                                    read_part, reader, handle, offset + part, min(chunk, size - part),
                                    slot["index"] * slot_bytes + part,
                                ) for part in range(0, size, chunk)]
                                slot["plan"], slot["state"] = plan, "reading"
                                progress = True
                        if not progress:
                            time.sleep(0.00001)
                    torch.cuda.synchronize(device)
                    destinations_ready = time.perf_counter()
                finally:
                    # Drain every submitted read before any registration or allocation retires.
                    for slot in slots:
                        for future in slot["futures"]:
                            try:
                                future.result()
                            except BaseException:
                                pass
                    torch.cuda.synchronize(device)
                    if reader is not None:
                        native.close(reader)
                    reader = buffer = backing = None
                finished = time.perf_counter()
                transport = {key: value - before[key] for key, value in gds.synchronous_transport_stats().items()}
                if transport["posix_reads"] or transport["read_errors"] or not transport["nvfs_reads"] + transport["p2p_reads"]:
                    raise RuntimeError(f"owner transport qualification failed: {transport}")
                connection.send(("result", dict(started=started, finished=finished, seconds=finished - started,
                                                reader_setup_seconds=ready - started,
                                                transfer_seconds=destinations_ready - ready,
                                                reader_cleanup_seconds=finished - destinations_ready,
                                                read_task_seconds_sum=read_task_seconds,
                                                api_read_bytes=transferred, chunks=completed, transport=transport)))
            del target
        connection.send(("closed", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise


class OwnerRanks:
    def __init__(self, devices, experts, destination_bytes, capacity, io_threads, native,
                 *, worker=None, worker_args=()):
        import torch

        self.devices, self.native = devices, native
        context = mp.get_context("spawn")
        self.barrier = context.Barrier(len(devices) + 1)
        self.connections, self.processes, self.imports, self.targets = [], [], [], []
        self.destinations = []
        self.closed = False
        try:
            for rank in range(len(devices)):
                parent, child = context.Pipe()
                process = context.Process(target=worker or _worker, args=(child, self.barrier, rank, devices, native.__file__,
                                                                experts, destination_bytes, capacity, io_threads, *worker_args))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            handles = self._receive("allocated")
            for connection in self.connections:
                connection.send(("map", handles))
            self.identity = self._receive("ready")
            for rank, device in enumerate(devices):
                with torch.cuda.device(device):
                    pointers = []
                    for handle in handles:
                        base = native.import_ipc(device, handle[0])
                        self.imports.append((device, base))
                        pointers.append(base + handle[1])
                    self.destinations.append(pointers)
                    pointer = pointers[rank]
                    target = torch.as_tensor(_CudaBytes(pointer, destination_bytes), device=f"cuda:{device}")
                    if target.data_ptr() != pointer or target.nbytes != destination_bytes:
                        raise RuntimeError("controller CUDA IPC tensor view copied or changed size")
                    self.targets.append(target)
        except BaseException:
            self.abort()
            raise

    def _receive(self, expected):
        pending = set(range(len(self.connections)))
        results = [None] * len(pending)
        deadline = time.monotonic() + 300
        while pending:
            for rank in tuple(pending):
                connection = self.connections[rank]
                if connection.poll(0.01):
                    kind, value = connection.recv()
                    if kind != expected:
                        raise RuntimeError(f"owner rank {rank}: expected {expected}, received {kind}: {value}")
                    results[rank] = value
                    pending.remove(rank)
                elif not self.processes[rank].is_alive():
                    raise RuntimeError(f"owner rank {rank} exited with {self.processes[rank].exitcode}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"owner ranks timed out waiting for {expected}: {sorted(pending)}")
        return results

    def run(self):
        for connection in self.connections:
            connection.send("run")
        self.barrier.wait(timeout=300)
        ranks = self._receive("result")
        return dict(seconds=max(r["finished"] for r in ranks) - min(r["started"] for r in ranks), ranks=ranks)

    def _unmap(self):
        import torch

        for device, _ in self.imports:
            torch.cuda.synchronize(device)
        self.targets.clear()
        for device, base in self.imports:
            self.native.close_ipc(device, base)
        self.imports.clear()

    def abort(self):
        try:
            self._unmap()
        finally:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join()
            self.closed = True

    def close(self):
        if self.closed:
            return
        try:
            for connection in self.connections:
                connection.send("unmap")
            self._receive("unmapped")
            self._unmap()
            for connection in self.connections:
                connection.send("close")
            self._receive("closed")
            for process in self.processes:
                process.join(10)
        finally:
            self.abort()
