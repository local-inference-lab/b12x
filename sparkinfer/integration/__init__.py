"""vLLM / SGLang integration shims for sparkinfer kernels.

The maintainer's private fork carries the full FP4 glue under
``sparkinfer/integration/`` (``tp_moe.py``, ``mla.py``, ...).  FP6 follows the
same pattern: thin framework adapters that call sparkinfer's public
``plan`` / ``bind`` / ``run`` APIs.
"""
