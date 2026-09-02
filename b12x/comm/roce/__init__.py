"""``b12x.comm.roce``: one-shot all-reduce over RoCE for multi-node TP.

Target: clusters of DGX Spark nodes joined by their ConnectX-7 200 GbE ports,
one GPU per node.  The GB10's unified memory lets the NIC RDMA-write straight
into pinned host memory that the GPU kernel then reads in place, so no
GPUDirect RDMA (dmabuf/peermem) support is required.

``AllReduce`` mirrors the ``comm.pcie.AllReduce`` surface (``from_exchange_group``,
``should_allreduce``, ``all_reduce``, ``for_stream``, ``capture``, ``close``) so
integrations can dispatch to it behind the same adapter.  See
``roce_oneshot.py`` for the protocol and constraints.
"""

from .api import *  # noqa: F401,F403
from .api import __all__  # noqa: F401
