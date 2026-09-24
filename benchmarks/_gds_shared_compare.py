"""Exercise the production collective reader on the expert transport corpus."""

import ctypes
import hashlib
import os
from pathlib import Path
import time
import traceback

from _instanttensor_compare import _load_native


def worker(connection, barrier, rank, devices, native_path, experts,
           destination_bytes, capacity, io_threads, rendezvous):
    try:
        from datetime import timedelta
        import torch
        import torch.distributed as dist
        from b12x.loader._checkpoint import DirectWeightSession
        from b12x.loader._gds_native import load

        from b12x.loader._shared_checkpoint import SharedReadGroup

        torch.set_num_threads(1)
        device, tp = devices[rank], len(devices)
        torch.cuda.set_device(device)
        gds, native = load(), _load_native(native_path)
        cufile_path = next(line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                           if "/libcufile.so" in line)

        class Status(ctypes.Structure):
            _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]

        cufile = ctypes.CDLL(cufile_path)
        cufile.cuFileDriverOpen.restype = Status
        status = cufile.cuFileDriverOpen()
        if status.err:
            raise RuntimeError(f"cuFileDriverOpen: {status.err}, CUDA {status.cu_err}")
        gds.start_stats(3)
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=tp,
                                timeout=timedelta(seconds=180))
        group = SharedReadGroup(dist.group.WORLD, device)
        paths = sorted({tensor.path for expert in experts for tensor in expert["tensors"]})
        target = torch.empty(destination_bytes, dtype=torch.uint8, device=device)
        connection.send(("allocated", native.export_ipc(device, target.data_ptr())))
        command, _ = connection.recv()
        if command != "map":
            raise RuntimeError("expected controller mapping acknowledgement")
        libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if any(key in line for key in ("libcufile.so", "libcudart.so", "libcuda.so"))})
        identity = dict(pid=os.getpid(), rank=rank, device=device,
                        final_destination_bytes=target.nbytes, final_destination_owner="rank process",
                        staging_bytes=18677760, slots=4, slot_bytes=4653056,
                        implementation="DirectWeightSession.finish / SharedReadGroup / native owner executor",
                        hashes={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in [gds.__file__, native.__file__, *libraries]})
        connection.send(("ready", identity))
        while True:
            command = connection.recv()
            if command == "unmap":
                torch.cuda.synchronize(device)
                connection.send(("unmapped", None))
                if connection.recv() != "close":
                    raise RuntimeError("expected close after controller unmap")
                break
            if command != "run":
                raise RuntimeError(f"unknown shared reader command: {command}")
            with DirectWeightSession(device, io_threads=io_threads,
                                     shared_read_group=group) as session:
                files = {path: os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC) for path in paths}
                session.files.extend(files.values())
                session.file_identities.update({fd: session._file_identity(fd) for fd in files.values()})
                for expert in experts:
                    for tensor in expert["tensors"]:
                        rows, width, source = tensor.shard(rank, tp)
                        session.records.extend((files[tensor.path], tensor.offset + source, width,
                                                target.data_ptr() + tensor.destination_offset,
                                                0, rows, tensor.width, width))
                session.destinations.append(target)
                before_stats = dict(group.totals)
                before = gds.synchronous_transport_stats()
                barrier.wait(timeout=180)
                started = time.perf_counter()
                session.finish()
                transferred = time.perf_counter()
                group.close()
                finished = time.perf_counter()
                transport = {key: value - before[key] for key, value in gds.synchronous_transport_stats().items()}
                counters = {key: value - before_stats.get(key, 0) for key, value in group.totals.items()}
                for key in ("staging_bytes", "gds_version"):
                    counters[key] = group.totals[key]
                if transport["posix_reads"] or transport["read_errors"] or not transport["nvfs_reads"] + transport["p2p_reads"]:
                    raise RuntimeError(f"shared reader transport qualification failed: {transport}")
                connection.send(("result", dict(started=started, finished=finished, seconds=finished - started,
                                                api_read_bytes=counters["physical_bytes"], transport=transport,
                                                completion_seconds=transferred - started,
                                                reader_cleanup_seconds=finished - transferred, counters=counters)))
        del target
        dist.destroy_process_group()
        connection.send(("closed", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise
