from dataclasses import replace

import pytest

from b12x.gemm.mgroup_fp8_gemm._tuning import (
    MGroupFP8GemmQuery, TUNING, default_config, joint_eligible,
)
from b12x.preparation.types import DeviceIdentity

PRO = DeviceIdentity(vendor='nvidia', product_name='NVIDIA RTX PRO 6000 Blackwell Server Edition',
                     compute_capability=(12, 0), sm_count=188)
BASE = MGroupFP8GemmQuery(mode='contiguous', num_groups=64, n=1024, k=4096,
                         m_capacity=4097, a_sf_gran=32)


@pytest.mark.parametrize('groups,n,k', [(64, 1024, 4096), (64, 4096, 512), (96, 1152, 5120), (96, 5120, 640)])
@pytest.mark.parametrize('capacity', [4097, 16385, 32769, 65536])
def test_joint_extended_domain(groups, n, k, capacity):
    query = replace(BASE, num_groups=groups, n=n, k=k, m_capacity=capacity)
    config = default_config(query, PRO)
    assert joint_eligible(query, PRO) and config.implementation == 'joint_v1'
    choices = [c for _, c in TUNING.choices(query, PRO)]
    assert config in choices
    assert {c.implementation for c in choices} == {'single', 'joint_v1'}
    for candidate in choices:
        TUNING.validate_config(query, candidate, PRO)
        if candidate.implementation == 'single':
            assert candidate.tile_k == 128


@pytest.mark.parametrize('changes', [dict(num_groups=63), dict(num_groups=65),
    dict(n=1023), dict(n=1025), dict(k=3968), dict(k=4224),
    dict(m_capacity=4096), dict(m_capacity=65537),
    dict(mode='masked', a_sf_gran=128)])
def test_joint_extension_exclusions(changes):
    query = replace(BASE, **changes)
    assert not joint_eligible(query, PRO)


@pytest.mark.parametrize('device', [None, replace(PRO, sm_count=187),
    replace(PRO, product_name='NVIDIA RTX 5090'), replace(PRO, compute_capability=(12, 1))])
def test_joint_identity_exclusions(device):
    assert not joint_eligible(BASE, device)


def test_original_joint_domain_and_versions():
    for groups in (1, 63, 65):
        query = replace(BASE, n=4096, k=2048, num_groups=groups, m_capacity=131072)
        assert joint_eligible(query, PRO)
    assert TUNING.candidate_contract_version == 8
    assert TUNING.query_schema_version == 1 and TUNING.config_schema_version == 2


@pytest.mark.parametrize('changes', [dict(num_groups=95), dict(num_groups=97),
    dict(n=1151), dict(n=1153), dict(k=4992), dict(k=5248),
    dict(m_capacity=4096), dict(m_capacity=65537),
    dict(mode='masked', a_sf_gran=128)])
def test_g96_extension_exclusions(changes):
    query = replace(replace(BASE, num_groups=96, n=1152, k=5120), **changes)
    assert not joint_eligible(query, PRO)
    assert all(c.implementation != 'joint_v1' for _, c in TUNING.choices(query, PRO))


@pytest.mark.parametrize('device', [None, replace(PRO, sm_count=187),
    replace(PRO, product_name='NVIDIA RTX 5090'), replace(PRO, compute_capability=(12, 1))])
def test_g96_identity_exclusions(device):
    for n, k in ((1152, 5120), (5120, 640)):
        assert not joint_eligible(replace(BASE, num_groups=96, n=n, k=k), device)


@pytest.mark.parametrize('changes', [dict(num_groups=95), dict(num_groups=97),
    dict(n=5119), dict(n=5121), dict(k=512), dict(k=768),
    dict(m_capacity=4096), dict(m_capacity=65537),
    dict(mode='masked', a_sf_gran=128)])
def test_g96_short_extension_exclusions(changes):
    query = replace(replace(BASE, num_groups=96, n=5120, k=640), **changes)
    assert not joint_eligible(query, PRO)
    assert all(c.implementation != 'joint_v1' for _, c in TUNING.choices(query, PRO))


@pytest.mark.parametrize('capacity', [4097, 65536, 65537, 131072])
def test_g384_extended_domain(capacity):
    test_joint_extended_domain(384, 576, 5120, capacity)


@pytest.mark.parametrize('changes', [dict(num_groups=383), dict(num_groups=385),
    dict(n=575), dict(n=577), dict(k=4992), dict(k=5248),
    dict(m_capacity=4096), dict(m_capacity=131073),
    dict(mode='masked', a_sf_gran=128)])
def test_g384_extension_exclusions(changes):
    query = replace(replace(BASE, num_groups=384, n=576, k=5120), **changes)
    assert not joint_eligible(query, PRO)
    assert all(c.implementation != 'joint_v1' for _, c in TUNING.choices(query, PRO))


@pytest.mark.parametrize('device', [None, replace(PRO, sm_count=187),
    replace(PRO, product_name='NVIDIA RTX 5090'), replace(PRO, compute_capability=(12, 1))])
def test_g384_identity_exclusions(device):
    assert not joint_eligible(replace(BASE, num_groups=384, n=576, k=5120), device)
