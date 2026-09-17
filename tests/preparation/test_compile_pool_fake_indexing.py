import os
import subprocess
import sys
import textwrap


def test_fake_cuda_indexing_stays_offline():
    code = textwrap.dedent(
        """
        import multiprocessing
        import torch
        from b12x._lib import compile_pool

        compile_pool._initialize_worker(
            0, (12, 1), "synthetic", "GB10", 48, 232448, 232448,
            multiprocessing.get_context("spawn").Array("q", (0, 0)),
        )
        from torch._subclasses.fake_tensor import FakeTensorMode

        cases = [
            slice(1, None, 2),
            1,
            (slice(None), slice(1, 4)),
            (Ellipsis, None, 2),
            True,
            [1, 3],
        ]
        with FakeTensorMode():
            fake = torch.empty((200,), dtype=torch.float32, device="cuda").narrow(
                0, 3, 100
            ).view(20, 5)
            real = torch.empty((200,), dtype=torch.float32).narrow(0, 3, 100).view(20, 5)
            for index in cases:
                got = fake[index]
                expected = real[index]
                assert (tuple(got.shape), tuple(got.stride()), got.storage_offset()) == (
                    tuple(expected.shape), tuple(expected.stride()), expected.storage_offset()
                )
                assert got.device.type == "cuda"
            fake[:2] = 1.0
            fake[[1, 3]] = 2.0
            assert not torch.cuda.is_initialized()
        """
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
