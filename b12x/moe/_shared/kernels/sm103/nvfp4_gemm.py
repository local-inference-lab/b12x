# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""Selected-expert NVFP4 projection over the shared SM103 compute pipeline."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from b12x.gemm._shared.sm103_blockscaled import BlockscaledGemm


class RoutedNvfp4Gemm(BlockscaledGemm):
    def __init__(self, n: int, k: int, experts: int, capacity: int):
        super().__init__(n, k, experts, recipe="nvfp4", c_dtype=cutlass.BFloat16,
                         routed_capacity=capacity)

    @cute.jit
    def __call__(
        self, a_ptr: cute.Pointer, b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer, sfb_ptr: cute.Pointer, c_ptr: cute.Pointer,
        ids: cute.Pointer, alpha: cute.Pointer, live_routes: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._launch(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, ids, alpha,
                     live_routes, cutlass.Int64(self.k), cutlass.Int64(self.n),
                     cutlass.Int64(1), stream)
