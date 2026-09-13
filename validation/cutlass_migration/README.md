# CUTLASS artifact integrity tooling

Status: implemented; the integrity checks run without a GPU. They establish
manifest consistency and tamper rejection, not GPU correctness or migration
acceptance.

The compiler-cache tests import independent semantic-payload validators from
`evidence/kernel_resources.py` and `acceptance/corpus/ptx_capture.py`. These
modules, `core/comparison_identity.py`, and `paths.py` are retained from
revision `bad4bcb24c21582731666499bea9dac6f25eb115`. The PTX capture hook targets
`b12x._lib.compiler` and accepts both the `_lib` and historical `cute` v3
manifest namespaces. Other manifest fields and object hashes remain strict.

PTX retention controls remain in raw cache and semantic identities. The
comparison module normalizes only its enumerated operational and toolchain
differences in a separate identity; it never changes a raw manifest or hash.

Run the host checks from the repository root:

```bash
python -m pytest tests/_lib/test_compile_cache.py \
  validation/cutlass_migration/integrity_checks -q
python -m validation.cutlass_migration.evidence.kernel_resources --help
```

No temporary `PYTHONPATH`, historical checkout, or GPU is needed for these
checks. Actual object/resource auditing requires compiled cache artifacts and
NVIDIA disassembly tools. PTX capture requires a working CUDA compilation
environment and installation before any compile-cache activity; its module
documents the capture controls. Keep generated evidence outside the source
tree. The retained modules do not constitute a complete migration corpus or
a B300 runtime qualification.
