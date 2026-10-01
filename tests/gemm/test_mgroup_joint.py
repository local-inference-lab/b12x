import pytest
import torch
from b12x.gemm.mgroup_fp8_gemm._tuning import MGroupFP8GemmQuery, MGroupFP8GemmConfig, TUNING, default_config
from b12x.preparation.types import DeviceIdentity


def test_joint_static_contract_and_device_scope():
    q=MGroupFP8GemmQuery(mode='contiguous',num_groups=64,n=4096,k=2048,m_capacity=8192,a_sf_gran=32)
    pro=DeviceIdentity(vendor='nvidia',product_name='NVIDIA RTX PRO 6000 Blackwell Server Edition',compute_capability=(12,0),sm_count=188)
    cfg=default_config(q,pro)
    assert cfg.implementation=='joint_v1'
    from b12x.gemm.mgroup_fp8_gemm._joint import _JointLaunch
    args = (4096, 2048, 64, 8192, 188)
    base, packed = _JointLaunch(*args), _JointLaunch(*args, packed_sf=True)
    assert base.compile_key() == ('mgroup-joint-v1', 'generic-shared-tma-bidirect-v3', *args)
    assert packed.compile_key() == ('mgroup-joint-packed-sf-v1', 'generic-shared-tma-bidirect-v3', *args)
    assert not base.full.mgroup_packed_sf and not base.narrow.mgroup_packed_sf
    assert packed.full.mgroup_packed_sf and not packed.narrow.mgroup_packed_sf
    for flag in ('sfa_prefetch', 'role_local_scheduler', 'nmajor_mma'):
        with pytest.raises(ValueError):
            _JointLaunch(*args, packed_sf=True, **{flag: True})
    assert TUNING.decode_config(TUNING.config_payload(cfg))==cfg
    assert TUNING.config_schema_version==2 and TUNING.candidate_contract_version==8
    assert TUNING.semantic_version==3
    assert default_config(q,None).implementation=='single'
    spark=DeviceIdentity(vendor='nvidia',product_name='NVIDIA GB10',compute_capability=(12,1),sm_count=48)
    assert default_config(q,spark).implementation=='single'
    with pytest.raises(ValueError):TUNING.validate_config(q,cfg,spark)
    from dataclasses import replace
    assert default_config(q,replace(pro,sm_count=96)).implementation=='single'
    configs=[c for _,c in TUNING.choices(q,pro)]
    assert cfg in configs
    for c in configs:
        TUNING.validate_config(q,c,pro)
        if c.implementation=='joint_v1':assert (c.tile_m,c.tile_n,c.tile_k)==(128,128,64)
    assert len([c for c in configs if c.implementation=='joint_v1'])==1
    masked=replace(q,mode='masked',a_sf_gran=128)
    assert {c.implementation for _,c in TUNING.choices(masked,pro)} == {'single', 'masked_compact'}
    compact=replace(default_config(masked,pro),implementation='masked_compact')
    assert TUNING.decode_config(TUNING.config_payload(compact)) == compact
    with pytest.raises(ValueError):TUNING.validate_config(q,compact,pro)
    with pytest.raises(ValueError):TUNING.validate_config(masked,compact,spark)
    assert default_config(masked,None).implementation == 'single'
    assert default_config(masked,pro).implementation == 'masked_compact'
    assert default_config(masked,spark).implementation == 'single'
    with pytest.raises(ValueError):TUNING.decode_config(dict(backend='cutedsl',tile_m=128,tile_n=128,tile_k=64))


def test_joint_factory_packed_static_contract(monkeypatch):
    from contextlib import nullcontext
    from b12x.gemm.mgroup_fp8_gemm import _joint, _preparation
    q = MGroupFP8GemmQuery(mode='contiguous', num_groups=64, n=4096, k=2048, m_capacity=8192, a_sf_gran=32)
    cfg = MGroupFP8GemmConfig(backend='cutedsl', tile_m=128, tile_n=128, tile_k=64, implementation='joint_v1')
    seen = []
    def compile_joint(*args, **kwargs):
        seen.append(kwargs)
        return _joint._JointLaunch(*args, **kwargs)
    monkeypatch.setattr(torch.cuda, 'device', lambda ordinal: nullcontext())
    monkeypatch.setattr(_joint, 'compile_joint', compile_joint)
    from b12x._lib import compile_plan
    monkeypatch.setattr(compile_plan, 'launch_triton', lambda *args, **kwargs: object())
    launch = _preparation.compile_mgroup_fp8(TUNING.encode_query(q), TUNING.config_payload(cfg), 0, 188)['gemm']
    assert seen == [{'packed_sf': True}]
    assert launch.compile_key()[:2] == ('mgroup-joint-packed-sf-v1', 'generic-shared-tma-bidirect-v3')
    assert launch.full.mgroup_packed_sf and not launch.narrow.mgroup_packed_sf
    base = compile_joint(4096, 2048, 64, 8192, 188, packed_sf=False)
    assert not base.full.mgroup_packed_sf and not base.narrow.mgroup_packed_sf
    assert base.compile_key() != launch.compile_key()


def test_private_workspace_ownership_and_resident_bytes():
    from types import SimpleNamespace
    from b12x.gemm import mgroup_fp8_gemm as mgg
    from b12x.preparation.types import _plan_scope
    q=MGroupFP8GemmQuery(mode='contiguous',num_groups=2,n=264,k=640,m_capacity=257,a_sf_gran=32)
    plans=[mgg.plan(q),mgg.plan(q)]
    cfg=default_config(q,None);dev=SimpleNamespace(ordinal=0)
    memories=[]
    for plan in plans:
        with _plan_scope(plan):memories.append(plan._memory_requirements(cfg,dev))
    own=[next(x for x in m.persistent if x.key[0]=='mgroup.contiguous') for m in memories]
    assert own[0].key!=own[1].key and own[0].required_nbytes==own[1].required_nbytes
    state=SimpleNamespace(sfa_workspace=torch.empty(32,dtype=torch.uint8),sfb_workspace=torch.empty(64,dtype=torch.uint8),selector=torch.empty(1,dtype=torch.int32))
    object.__setattr__(plans[0],'_prepared',SimpleNamespace(state=state,closed=False))
    try:
        with _plan_scope(plans[0]):memory=plans[0]._memory_requirements(cfg,dev)
        assert next(x for x in memory.persistent if x.key[0]=='mgroup.contiguous').resident_nbytes==100
    finally:object.__setattr__(plans[0],'_prepared',None)


def test_joint_sfa_offset_widens_before_multiplication():
    import ast
    from pathlib import Path
    from b12x._lib import dense_gemm
    tree=ast.parse(Path(dense_gemm.__file__).read_text())
    expressions=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sfa_offset' for t in n.targets) and 'Int64(sfa_tile)' in ast.unparse(n.value) and 'l_coord' not in ast.unparse(n.value)]
    assert len(expressions)==1
    code=compile(ast.Expression(expressions[0]),'offset','eval')
    assert eval(code,dict(Int64=int,sfa_tile=1023,sf_k_tiles=8192,sf_k_tile=8191,atom_offset=510))==4294967294


def test_joint_gpu_bodies_live_labels_scales():
    from tests._reference.helpers import require_b12x
    require_b12x()
    from tests.gemm.test_mgroup_fp8_gemm import _contiguous_case, _prepared_mgroup
    from b12x.gemm import mgroup_fp8_gemm as mgg
    lhs,rhs,d,labels,_=_contiguous_case(4,264,256,[128,128,128,65],128,seed=91)
    q=MGroupFP8GemmQuery(mode='contiguous',num_groups=4,n=264,k=256,m_capacity=449,a_sf_gran=32)
    cfg=MGroupFP8GemmConfig(backend='cutedsl',tile_m=128,tile_n=128,tile_k=64,implementation='joint_v1')
    with _prepared_mgroup(q,lhs,rhs,d,labels,override=cfg) as plan:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):mgg.contiguous_mm(lhs,rhs,d,labels,plan=plan)
        observed=[]
        for tile_labels in ([0,1,2,3],[0,1,0,1],[0,0,2,3],[0,0,0,0],[-1,-1,-1,-1],[0,1,2,3]):
            labels.copy_(torch.tensor(tile_labels,device='cuda',dtype=torch.int32)[torch.arange(449,device='cuda')//128])
            graph.replay();torch.cuda.synchronize()
            a=lhs[0].float()*lhs[1].repeat_interleave(32,dim=1)
            b=rhs[0].float()*rhs[1].repeat_interleave(128,dim=2)
            ref=torch.zeros_like(d)
            for g in range(4):
                mask=labels==g
                ref[mask]=(a[mask]@b[g].T).to(torch.bfloat16)
            torch.testing.assert_close(d.float(),ref.float(),rtol=.02,atol=.02)
            observed.append(int(plan._prepared.state.selector[0].item()))
        assert set(observed)=={0,1}
        graph.reset()
