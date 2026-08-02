from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PCIE_ONESHOT = ROOT / "sparkinfer" / "comm" / "pcie" / "pcie_oneshot.cu"


def test_oneshot_native_dispose_has_strict_and_best_effort_paths() -> None:
    source = PCIE_ONESHOT.read_text(encoding="utf-8")

    assert "IpcHandleRegistry<IPC_KEY, char*, cudaError_t, cudaSuccess>" in source
    assert "IPCHandles ipc_handles_;" in source
    assert "~PCIeAllreduce() noexcept = default;" in source
    assert "cudaError_t close_ipc_handles_noexcept() noexcept" in source
    assert "void close_ipc_handles_strict()" in source
    assert "&dispose," in source
    assert '"dispose_best_effort"' in source

    dispose = source[source.index("static void dispose(fptr_t _fa)") :]
    dispose = dispose[: dispose.index("static int64_t meta_size()")]
    assert dispose.index("fa->close_ipc_handles_strict();") < dispose.index(
        "delete fa;"
    )
    assert "static bool dispose_best_effort(fptr_t _fa) noexcept" in dispose


def test_native_ipc_registry_destructor_survives_injected_close_failure(
    tmp_path: Path,
) -> None:
    """Compile the production RAII helper and fail a close inside destruction.

    Throwing for this injected status from an implicitly-noexcept destructor,
    as the old PCIeAllreduce destructor did, would invoke the terminate handler
    and make the subprocess exit 86.
    """

    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler available for native teardown regression")

    source = tmp_path / "ipc_registry_failure_injection.cpp"
    binary = tmp_path / "ipc_registry_failure_injection"
    source.write_text(
        textwrap.dedent(
            r"""
            #include "sparkinfer/comm/pcie/ipc_handle_registry.h"

            #include <cstdlib>
            #include <exception>
            #include <type_traits>

            namespace {

            constexpr int kSuccess = 0;
            constexpr int kInjectedFailure = 7;
            int close_calls = 0;
            int failed_handle = 0;
            int failures_remaining = 0;

            int injected_close(int handle) noexcept {
              ++close_calls;
              if (handle == failed_handle && failures_remaining > 0) {
                --failures_remaining;
                return kInjectedFailure;
              }
              return kSuccess;
            }

            using Registry =
                sparkinfer::pcie::IpcHandleRegistry<int, int, int, kSuccess>;

            struct Owner {
              Registry handles{&injected_close};
              ~Owner() noexcept = default;
            };

            static_assert(std::is_nothrow_destructible_v<Registry>);
            static_assert(std::is_nothrow_destructible_v<Owner>);

            }  // namespace

            int main() {
              std::set_terminate([] { std::_Exit(86); });

              failed_handle = 22;
              failures_remaining = 1;
              {
                Owner owner;
                owner.handles.emplace(1, 11);
                owner.handles.emplace(2, 22);
                owner.handles.emplace(3, 33);
              }
              if (close_calls != 3 || failures_remaining != 0) {
                return 10;
              }

              close_calls = 0;
              failures_remaining = 1;
              {
                Owner owner;
                owner.handles.emplace(1, 11);
                owner.handles.emplace(2, 22);
                owner.handles.emplace(3, 33);
                if (owner.handles.close_all_noexcept() != kInjectedFailure) {
                  return 11;
                }
                if (close_calls != 3) {
                  return 12;
                }
                if (owner.handles.find(1) != owner.handles.end() ||
                    owner.handles.find(2) == owner.handles.end() ||
                    owner.handles.find(3) != owner.handles.end()) {
                  return 17;
                }
                if (owner.handles.close_all_noexcept() != kSuccess) {
                  return 13;
                }
                if (close_calls != 4) {
                  return 14;
                }
                if (owner.handles.find(2) != owner.handles.end()) {
                  return 18;
                }
                if (owner.handles.close_all_noexcept() != kSuccess ||
                    close_calls != 4) {
                  return 15;
                }
              }
              return close_calls == 4 ? 0 : 16;
            }
            """
        ),
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compile_result.returncode == 0, (
        "native teardown regression failed to compile\n"
        f"stdout:\n{compile_result.stdout}\n"
        f"stderr:\n{compile_result.stderr}"
    )

    run_result = subprocess.run(
        [str(binary)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run_result.returncode == 0, (
        "injected native close failure terminated or violated idempotency\n"
        f"returncode: {run_result.returncode}\n"
        f"stdout:\n{run_result.stdout}\n"
        f"stderr:\n{run_result.stderr}"
    )
