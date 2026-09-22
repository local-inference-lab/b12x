"""Sanitizer completion cannot be inferred from a quiet or partial log."""

from scripts.qualify_hybrid_sanitizer import classify, diagnostic_inventory


def test_requires_completion_from_every_rank_and_zero_summaries():
    assert classify(0, False, [0, 0], {0, 1}, 2, False) == "whole_program_pass"
    assert classify(0, False, [0, 0], {0}, 2, False) == "incomplete"
    assert classify(0, False, [], {0, 1}, 2, False) == "incomplete"
    assert classify(99, False, [120, 120], {0, 1}, 2, False) == "failed"
    assert classify(124, True, [0], {0}, 2, False) == "timeout"


def test_filter_never_qualifies_whole_program():
    assert classify(0, False, [0, 0], {0, 1}, 2, True) == "component_pass"


def test_probe_attribution_requires_stack_and_complete_accounting():
    log = """========= COMPUTE-SANITIZER
========= Program hit cudaErrorNoKernelImageForDevice (error 209) due to "no kernel image is available for execution on the device" on CUDA API call to cudaFuncGetAttributes.
=========     Host Frame: ncclInitKernelsForDevice(int, int, unsigned long*) in enqueue.cc:87 in libnccl.so.2.31.2
=========
========= ERROR SUMMARY: 1 error
"""
    assert diagnostic_inventory(log)["initialization_probe_only"]
    assert not diagnostic_inventory(
        log.replace("ncclInitKernelsForDevice", "application")
    )["initialization_probe_only"]
    assert not diagnostic_inventory(log.replace("SUMMARY: 1", "SUMMARY: 2"))[
        "initialization_probe_only"
    ]
    assert not diagnostic_inventory(
        log.replace(
            "========= ERROR SUMMARY",
            "========= Invalid __global__ read of size 4 bytes\n=========\n========= ERROR SUMMARY",
        )
    )["initialization_probe_only"]
    assert classify(99, False, [1], {0}, 1, False) == "failed"
