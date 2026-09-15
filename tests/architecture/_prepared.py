"""Host-only binding fixtures; these do not qualify CUDA execution."""
from types import SimpleNamespace
from b12x.preparation import DetectedDevice
from b12x.preparation.types import _Prepared


def install_host_state(plan, state, config, *, scratch=()):
    plan._install(_Prepared(state=state, selection=SimpleNamespace(config=config),
                            programs=frozenset(), retained=None, owners=(), closers=(),
                            device=DetectedDevice(None, None), scratch=scratch))
    return plan
