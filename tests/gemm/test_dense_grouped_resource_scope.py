import ast
from pathlib import Path
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import pytest

from b12x._lib import dense_gemm


@pytest.mark.parametrize("labels,masked,expected", [(False, False, (40, 232)), (True, False, (56, 224)), (False, True, (56, 224))])
def test_grouped_register_budget_scope(labels, masked, expected):
    kernel = dense_gemm.DenseGemmKernel(32, (128, 128), (1, 1), mgroup_labels=labels,
                                       mgroup_masked=masked, tile_k=128)
    assert (kernel.load_register_requirement, kernel.mma_register_requirement) == expected


@pytest.mark.parametrize("labels,masked,joint,mask_len,compact,capacity,expected", [
    (False, False, False, 0, False, 0, 0),
    (True, False, False, 0, False, 0, 1),
    (False, True, False, 3, False, 0, 3),
    (False, True, False, 3, True, 0, 4),
    (True, False, True, 0, False, 17, 18),
])
def test_grouped_mask_field_has_zero_nongrouped_storage(labels, masked, joint, mask_len, compact, capacity, expected):
    tree = ast.parse(Path(dense_gemm.__file__).read_text())
    field = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                 and isinstance(n.target, ast.Name) and n.target.id == "mgroup_mask_buf")
    state = SimpleNamespace(mgroup_labels=labels, mgroup_masked=masked, mgroup_joint=joint,
                            mgroup_mask_len=mask_len, mgroup_compact_masked=compact,
                            mgroup_capacity_tiles=capacity)
    prepare = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "prepare_body")
    start = next(i for i, n in enumerate(prepare.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "self.mgroup_mask_storage_len")
    statements = prepare.body[start:start + 3]
    exec(compile(ast.Module(body=statements, type_ignores=[]), "mask-length", "exec"),
         {"cutlass": cutlass, "self": state})
    assert state.mgroup_mask_storage_len == expected
    annotation = eval(compile(ast.Expression(field.annotation), "mask-field", "eval"),
                      {"cute": cute, "cutlass": cutlass, "self": state})
    @cute.struct
    class Storage:
        data: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 256], 1024]
        mask: annotation
    assert Storage.size_in_bytes() == (1024 if expected == 0 else 2048)
    length = field.annotation.slice.elts[0].slice.elts[1]
    assert eval(compile(ast.Expression(length), "mask-length", "eval"), {"self": state}) == expected
