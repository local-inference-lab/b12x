"""Byte-level coverage for bounded io_uring PLE transactions (no CUDA work)."""

from array import array

import pytest

from b12x.loader._native import load


@pytest.fixture
def reader_factory():
    native = load()

    def create(
        shard_rows,
        padded_rows,
        tp_start,
        tp_end,
        weight_bytes,
        scale_bytes,
        capacity,
    ):
        try:
            return native.ple_reader(
                shard_rows,
                padded_rows,
                tp_start,
                tp_end,
                weight_bytes,
                scale_bytes,
                capacity,
                4,
            )
        except RuntimeError as error:
            message = str(error)
            if (
                "support is unavailable" in message
                or "initialization failed: Operation not permitted" in message
                or "initialization failed: Function not implemented" in message
            ):
                pytest.skip(message)
            raise

    return native, create


def test_cross_block_rows_deduplicate_planes_and_preserve_compact_order(
    tmp_path, reader_factory
):
    native, create = reader_factory
    row_bytes, scale_bytes, rows = 5003, 513, 10
    weights = bytes((i * 17 + i // 257) % 256 for i in range(rows * row_bytes))
    scales = bytes((i * 31 + 5) % 256 for i in range(rows * scale_bytes))
    weight_offset = 4103
    scale_offset = weight_offset + len(weights) + 31
    path = tmp_path / "both_planes"
    path.write_bytes(b"x" * weight_offset + weights + b"x" * 31 + scales)
    reader = create(6, rows, 2, 8, row_bytes, scale_bytes, 10)
    for shard in [1, 0]:
        native.ple_reader_add(
            reader, shard, str(path), weight_offset + shard * 6 * row_bytes, False
        )
        native.ple_reader_add(
            reader, shard, str(path), scale_offset + shard * 6 * scale_bytes, True
        )
    ids = array("q", [2, 2, 5, 6, 7, 0, -1, 8, 10, 100])
    output = bytearray(b"z" * (len(ids) * row_bytes + 7))
    output_scales = bytearray(b"z" * (len(ids) * scale_bytes + 7))
    allocation = native.ple_reader_stats(reader)
    native.ple_reader_run(reader, ids, output, output_scales, len(ids))
    for i, row in enumerate(ids):
        assert output[i * row_bytes : (i + 1) * row_bytes] == (
            weights[row * row_bytes : (row + 1) * row_bytes]
            if 2 <= row < 8
            else bytes(row_bytes)
        )
        assert output_scales[i * scale_bytes : (i + 1) * scale_bytes] == (
            scales[row * scale_bytes : (row + 1) * scale_bytes]
            if 2 <= row < 8
            else bytes(scale_bytes)
        )
    assert output[-7:] == output_scales[-7:] == b"z" * 7
    blocks = set()
    for row in ids:
        if 2 <= row < 8:
            for offset, size in [
                (weight_offset, row_bytes),
                (scale_offset, scale_bytes),
            ]:
                begin = offset + row * size
                blocks.update(range(begin // 4096, (begin + size - 1) // 4096 + 1))
    stats = native.ple_reader_stats(reader)
    assert stats["unique_blocks"] == len(blocks)
    assert stats["requested_bytes"] == 5 * (row_bytes + scale_bytes)
    assert stats["read_bytes"] == sum(
        min(4096, path.stat().st_size - block * 4096) for block in blocks
    )
    assert stats["read_calls"] < stats["unique_blocks"]
    assert stats["coalesced_reads"] > 0
    assert stats["staging_bytes"] == allocation["staging_bytes"]
    assert stats["metadata_bytes"] == allocation["metadata_bytes"]

    # The same allocations support a smaller subsequent batch without stale rows.
    second_ids = array("q", [7, 6, 2])
    native.ple_reader_run(reader, second_ids, output, output_scales, 3)
    assert output[: 3 * row_bytes] == b"".join(
        weights[row * row_bytes : (row + 1) * row_bytes] for row in second_ids
    )
    assert output_scales[: 3 * scale_bytes] == b"".join(
        scales[row * scale_bytes : (row + 1) * scale_bytes] for row in second_ids
    )
    assert native.ple_reader_stats(reader)["lookups"] == 3
    before = bytes(output)
    native.ple_reader_run(reader, array("q"), output, output_scales, 0)
    assert bytes(output) == before
    assert native.ple_reader_stats(reader)["read_calls"] == 0


def test_sparse_requests_never_read_gaps_and_cap_coalescing(tmp_path, reader_factory):
    native, create = reader_factory
    path = tmp_path / "pages"
    payload = b"".join(bytes([row]) * 4096 for row in range(40))
    path.write_bytes(payload)
    reader = create(40, 40, 0, 40, 4096, 0, 24)
    native.ple_reader_add(reader, 0, str(path), 0, False)
    ids = array("q", [*range(20), 39, 39])
    output = bytearray(len(ids) * 4096)
    native.ple_reader_run(reader, ids, output, None, len(ids))
    assert output == b"".join(payload[row * 4096 : (row + 1) * 4096] for row in ids)
    stats = native.ple_reader_stats(reader)
    assert stats["unique_blocks"] == 21
    assert stats["read_bytes"] == 21 * 4096
    assert stats["read_calls"] == 3  # 16 adjacent pages, four adjacent pages, page 39.
    assert stats["coalesced_reads"] == 2


def test_partial_final_block_and_truncation_drain_before_reuse(
    tmp_path, reader_factory
):
    native, create = reader_factory
    path = tmp_path / "partial"
    offset, row_bytes, rows = 107, 509, 80
    payload = bytes((i * 7) % 256 for i in range(rows * row_bytes))
    contents = b"h" * offset + payload
    path.write_bytes(contents)
    reader = create(rows, rows, 0, rows, row_bytes, 0, 8)
    native.ple_reader_add(reader, 0, str(path), offset, False)
    ids = array("q", [79, 0, 33, 51, 17, 79])
    output = bytearray(len(ids) * row_bytes)
    native.ple_reader_run(reader, ids, output, None, len(ids))
    expected = b"".join(payload[row * row_bytes : (row + 1) * row_bytes] for row in ids)
    assert output == expected
    with path.open("r+b") as stream:
        stream.truncate(len(contents) - 1)
    with pytest.raises(RuntimeError, match="short PLE read"):
        native.ple_reader_run(reader, ids, output, None, len(ids))
    # A failed transaction drains all CQEs: restore and reuse its slots.
    path.write_bytes(contents)
    native.ple_reader_run(reader, ids, output, None, len(ids))
    assert output == expected


def test_large_source_offsets_keep_exact_rows(tmp_path, reader_factory):
    native, create = reader_factory
    offset, rows, width = 2**32 + 4103, 8, 96
    payload = bytes((index * 17) % 256 for index in range(rows * width))
    path = tmp_path / "large_offset"
    with path.open("wb") as file:
        file.seek(offset)
        file.write(payload)
    reader = create(rows, rows, 0, rows, width, 0, 4)
    native.ple_reader_add(reader, 0, str(path), offset, False)
    ids = array("q", [7, 0, 7, 3])
    output = bytearray(len(ids) * width)
    native.ple_reader_run(reader, ids, output, None, len(ids))
    assert output == b"".join(payload[row * width : (row + 1) * width] for row in ids)


def test_sources_and_destinations_fail_before_unsafe_reads(tmp_path, reader_factory):
    native, create = reader_factory
    reader = create(4, 4, 0, 4, 16, 0, 4)
    path = tmp_path / "short"
    path.write_bytes(bytes(63))
    with pytest.raises(RuntimeError, match="range exceeds"):
        native.ple_reader_add(reader, 0, str(path), 0, False)
    ids = array("q", [0, 1, 2, 3])
    output = bytearray(64)
    with pytest.raises(RuntimeError, match="missing PLE weight source"):
        native.ple_reader_run(reader, ids, output, None, 4)
    path.write_bytes(bytes(range(64)))
    native.ple_reader_add(reader, 0, str(path), 0, False)
    with pytest.raises(RuntimeError, match="already registered"):
        native.ple_reader_add(reader, 0, str(path), 0, False)
    with pytest.raises(ValueError, match="capacity"):
        native.ple_reader_run(reader, ids, output, None, 5)
    with pytest.raises(ValueError, match="cover count rows"):
        native.ple_reader_run(reader, ids, output[:-1], None, 4)
    with pytest.raises((BufferError, TypeError)):
        native.ple_reader_run(reader, ids, bytes(64), None, 4)
    shared = bytearray(64)
    with pytest.raises(ValueError, match="must not overlap"):
        native.ple_reader_run(reader, memoryview(shared)[:32], shared, None, 4)
    native.ple_reader_run(reader, ids, output, None, 4)
    assert output == bytes(range(64))
