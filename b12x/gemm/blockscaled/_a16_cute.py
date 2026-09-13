"""Admitted SM103 entry points for portable BF16 activation compute.

Separate entry types keep the compiler gate closed for the SM12x block-scaled
variants in the shared dense engine. Core math remains in that shared engine.
"""

from b12x._lib.dense_gemm import _DenseGemmLaunch, _DenseSplitKReduce


class DenseA16Launch(_DenseGemmLaunch):
    def __init__(self, **kwargs):
        if kwargs.get("weight_only") not in ("nvfp4", "mxfp8"):
            raise ValueError("SM103 A16 entry requires inline NVFP4 or MXFP8 weight dequantization")
        super().__init__(**kwargs)


class DenseA16Reduce(_DenseSplitKReduce):
    pass
