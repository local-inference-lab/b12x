# Stable W4A16 route packing

Status: research-only. No performance benefit or deterministic MoE output
contract is established for this implementation.

Set `B12X_W4A16_STABLE_ROUTE_PACK=1` before plan declaration to retain
ascending token-major route order within each expert for capacities of at
least 4096 routed rows. Smaller plans and the default configuration use
unordered atomic packing. Expert assignments and router weights are unchanged.

The immutable MoE query retains the setting. Preparation compiles the route
programs for the declared capacity; execution supplies the live route count
without changing the specialization. Environment changes after declaration
do not alter the prepared plan.

Each expert's program scans the complete route capacity and uses prefix sums
to place matching routes in order. Work scales with expert count multiplied
by route capacity. Stable packing does not order floating-point atomics in
downstream GEMMs or output reduction, so it does not make MoE deterministic.
Adoption requires a demonstrated ordering requirement or a measured benefit
against ordinary atomic packing on representative routing distributions.

`tests/moe/test_w4a16_route_pack.py` checks ordering, mapped and disabled
experts, retained capacity, live bounds and CUDA Graph replay. GPU validation
of this prepared implementation remains pending.
