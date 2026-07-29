"""Parser tests for scripts/sass_instruction_mix.py.

CPU-only: these exercise text parsing, not a GPU.

The first run of this tool reported "no loops found" for every FP6 GEMM,
because branch targets were only matched in absolute-hex form while nvdisasm
emits them as symbolic labels. That failure is silent and plausible - a fully
unrolled kernel is a real thing - and it silently substituted a whole-kernel
census for the mainloop census the Phase G analysis depends on. These tests
exist so that specific failure cannot recur unnoticed.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sass_instruction_mix.py"


@pytest.fixture(scope="module")
def mix():
    spec = importlib.util.spec_from_file_location("sass_instruction_mix", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so @dataclass can resolve the defining module.
    sys.modules["sass_instruction_mix"] = module
    spec.loader.exec_module(module)
    return module


_LABEL_FORM = """//--------------------- .text.kernel_demo ------------------------
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   ULDC.64 UR4, c[0x0][0x118] ;
.L_x_0:
        /*0020*/                   LDSM.16.M88.4 R8, [R25+0x800] ;
        /*0030*/                   OMMA.16816.F32.E4M3 R16, R8, R12, R16 ;
        /*0040*/                   IADD3 R25, R25, 0x20, RZ ;
        /*0050*/              @!P0 BRA `(.L_x_0) ;
        /*0060*/                   PLOP3.LUT P0, PT, PT, PT, PT, 0x8, 0x0 ;
        /*0070*/                   EXIT ;
"""

_HEX_FORM = """//--------------------- .text.kernel_demo ------------------------
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   LDSM.16.M88.4 R8, [R25+0x800] ;
        /*0020*/                   OMMA.16816.F32.E4M3 R16, R8, R12, R16 ;
        /*0030*/              @!P0 BRA 0x10 ;
        /*0040*/                   EXIT ;
"""


def test_symbolic_branch_labels_recover_the_loop(mix):
    instructions = mix._instructions(_LABEL_FORM)
    labels = mix._label_offsets(_LABEL_FORM)
    assert labels == {".L_x_0": 0x20}

    loops = mix._find_loops(instructions, labels)
    assert len(loops) == 1
    assert (loops[0].start, loops[0].end) == (0x20, 0x50)
    # LDSM, OMMA, IADD3, BRA - the label target through the branch, inclusive.
    assert loops[0].size == 4


def test_hex_branch_targets_still_recover_the_loop(mix):
    """Some nvdisasm builds/flags emit absolute targets; keep both paths live."""
    instructions = mix._instructions(_HEX_FORM)
    loops = mix._find_loops(instructions, labels={})
    assert len(loops) == 1
    assert (loops[0].start, loops[0].end) == (0x10, 0x30)


def test_forward_branches_do_not_create_loops(mix):
    code = """//--------------------- .text.kernel_demo --------------------
        /*0000*/              @!P0 BRA `(.L_x_1) ;
        /*0010*/                   IADD3 R2, R2, 0x1, RZ ;
.L_x_1:
        /*0020*/                   EXIT ;
"""
    instructions = mix._instructions(code)
    labels = mix._label_offsets(code)
    assert mix._find_loops(instructions, labels) == []


def test_operand_immediates_are_not_mistaken_for_branch_targets(mix):
    """The hex fallback must not fire when a symbolic target already resolved.

    ``BRA `(.L_x_1)`` on a line that also carries a 0x immediate would
    otherwise manufacture a backward edge to that immediate.
    """
    code = """//--------------------- .text.kernel_demo --------------------
.L_x_1:
        /*0100*/                   IADD3 R2, R2, 0x1, RZ ;
        /*0110*/         @!P0 BRA.U `(.L_x_1) ;
        /*0120*/                   EXIT ;
"""
    instructions = mix._instructions(code)
    labels = mix._label_offsets(code)
    targets = mix._branch_targets(instructions[1][2], labels)
    assert targets == [0x100]


_NESTED = """//--------------------- .text.kernel_demo ------------------------
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
.L_tile:
        /*0010*/                   LDGSTS.E R4, [R6] ;
.L_kloop:
        /*0020*/                   LDSM.16.M88.4 R8, [R25+0x800] ;
        /*0030*/                   QMMA.16816.F32 R16, R8, R12, R16 ;
        /*0040*/                   IADD3 R25, R25, 0x20, RZ ;
        /*0050*/              @!P0 BRA `(.L_kloop) ;
        /*0060*/                   F2FP.PACK_AB R20, R16, R17 ;
        /*0070*/                   STG.E [R30], R20 ;
        /*0080*/              @!P1 BRA `(.L_tile) ;
        /*0090*/                   EXIT ;
"""


def test_mainloop_is_the_innermost_mma_loop_not_the_largest(mix):
    """A persistent GEMM nests the k-loop inside a tile-scheduler loop.

    Selecting the largest loop would charge the epilogue's conversion and
    global store to the mainloop's per-MMA budget.
    """
    instructions = mix._instructions(_NESTED)
    labels = mix._label_offsets(_NESTED)
    loops = mix._dedupe_loops(mix._find_loops(instructions, labels))

    outer = max(loops, key=lambda loop: loop.size)
    assert (outer.start, outer.end) == (0x10, 0x80)

    mainloop = mix._select_mainloop(loops)
    assert (mainloop.start, mainloop.end) == (0x20, 0x50)

    counts = mix._census(mainloop.instructions)
    assert counts["mma"] == 1
    # The epilogue's F2FP and STG belong to the tile loop, not the k-loop.
    assert "fp_math" not in counts
    assert "stg" not in counts


def test_loops_sharing_a_body_are_deduplicated(mix):
    """A multi-exit loop emits one backward branch per exit, not one loop."""
    code = """//--------------------- .text.kernel_demo --------------------
.L_a:
        /*0000*/                   QMMA.16816.F32 R16, R8, R12, R16 ;
        /*0010*/              @!P0 BRA `(.L_a) ;
        /*0020*/              @!P1 BRA `(.L_a) ;
        /*0030*/                   EXIT ;
"""
    instructions = mix._instructions(code)
    labels = mix._label_offsets(code)
    raw = mix._find_loops(instructions, labels)
    assert len(raw) == 2
    assert len(mix._dedupe_loops(raw)) == 2  # distinct ends, genuinely two bodies

    same = [mix.Loop(start=0, end=16, instructions=[(0, "QMMA")])] * 2
    assert len(mix._dedupe_loops(same)) == 1


@pytest.mark.parametrize(
    ("opcode", "category"),
    [
        ("OMMA", "mma"),
        ("QGMMA", "mma"),
        ("LDSM", "ldsm"),
        ("LDS", "lds"),
        ("STS", "sts"),
        ("UTMALDG", "tma"),
        ("LDGSTS", "ldgsts"),
        ("LDL", "local"),
        ("STL", "local"),
        ("ULDC", "const_load"),
        ("LDCU", "const_load"),
        ("STSM", "sts"),
        ("F2FP", "fp_math"),
        ("IADD", "int_addr"),
        ("BAR", "barrier"),
        ("SYNCS", "barrier"),
        ("NANOSLEEP", "barrier"),
        ("PLOP3", "pred"),
        ("FFMA", "fp_math"),
        ("IMAD", "int_addr"),
        ("BRA", "control"),
    ],
)
def test_opcode_categories(mix, opcode, category):
    assert mix._categorize(opcode) == category


def test_access_widths_are_recovered_from_modifiers(mix):
    """Byte-wide and 128-bit shared loads must not read as the same thing."""
    code = """//--------------------- .text.kernel_demo --------------------
        /*0000*/                   LDS.U8 R4, [R25] ;
        /*0010*/                   LDS.U8 R5, [R25+0x1] ;
        /*0020*/                   LDS.128 R8, [R26] ;
        /*0030*/                   LDSM.16.M88.4 R12, [R27] ;
"""
    instructions = mix._instructions(code)
    modifiers = mix._modifiers_by_offset(code)
    assert modifiers[0x0] == ".U8"
    assert modifiers[0x20] == ".128"
    assert modifiers[0x30] == ".16.M88.4"

    body = [(offset, opcode) for offset, opcode, _ in instructions]
    counts = Counter(
        f"{opcode}{modifiers[offset]}"
        for offset, opcode in body
        if mix._categorize(opcode) in {"lds", "ldsm"}
    )
    assert counts == {"LDS.U8": 2, "LDS.128": 1, "LDSM.16.M88.4": 1}


def test_modifiers_do_not_leak_into_the_opcode(mix):
    """IMAD.MOV.U32 is an IMAD; a category keyed on the full text would miss it."""
    code = """//--------------------- .text.kernel_demo --------------------
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   LDSM.16.M88.4 R8, [R25+0x800] ;
"""
    assert [opcode for _, opcode, _ in mix._instructions(code)] == ["IMAD", "LDSM"]


def test_uniform_registers_do_not_alias_predicate_opcodes(mix):
    """UPLOP3 is predicate logic, not an unclassified opcode."""
    assert mix._categorize("UPLOP3") == "pred"
    assert mix._categorize("UIMAD") == "int_addr"
