"""Lossless scale compression during ordinary packed-weight preparation."""

from dataclasses import replace

import torch

from b12x._lib.quant.csf_encode import encode_scale_bytes
from b12x.moe._shared.execution import WeightStoragePolicy
from .weights import Mxfp4CsfWeights, Nvfp4CsfWeights, PackedWeights


def prepare_compressed_scales(plan, weights, scratch):
    """Transfer source ownership and retain only CSF scales plus shared scratch.

    The caller owns two full scale buffers shared by serialized layers with
    identical geometry. Native NVFP4 input scales use F8_128x4 order; MXFP4
    scales use logical row-major order. FC1 weights may be normalized in place,
    as in native preparation. No checkpoint or model metadata is consumed.
    """
    from .planning import (
        ActivationMode,
        prepare_weights,
        plan_weights,
        WeightPlanConstraints,
    )
    from .source import W13Layout

    if not isinstance(weights, PackedWeights) or scratch is None or len(scratch) != 2:
        raise ValueError(
            "online CSF preparation requires PackedWeights and two scale buffers"
        )
    e, h, n = (
        plan.geometry.num_experts,
        plan.geometry.hidden_size,
        plan.geometry.intermediate_size,
    )
    nv = plan.source.format.value == "modelopt_nvfp4"
    if not nv:
        factors = [weights.w13_global_scales, weights.w2_global_scales]
        if plan.activation.mode is ActivationMode.A8:
            factors.extend([weights.input_scale, weights.intermediate_scale])
        if any(t is not None and not bool(torch.all(t == 1)) for t in factors):
            raise ValueError(
                "MXFP4-CSF requires unit global weight and activation factors"
            )
    columns = (h // (16 if nv else 32), n // (16 if nv else 32))
    shapes = ((e, 2 * n, columns[0]), (e, h, columns[1]))
    inputs = (weights.w13_block_scales, weights.w2_block_scales)
    source_storage = {
        (t.device, t.untyped_storage().data_ptr())
        for t in vars(weights).values()
        if isinstance(t, torch.Tensor)
    }
    buffer_storage = {(t.device, t.untyped_storage().data_ptr()) for t in scratch}
    if len(buffer_storage) != 2 or source_storage & buffer_storage:
        raise ValueError(
            "CSF buffers must not alias each other or source tensor storage"
        )
    outputs = []
    for buffer, shape in zip(scratch, shapes, strict=True):
        if (
            buffer.device != weights.w13.device
            or not buffer.is_contiguous()
            or buffer.element_size() != 1
            or tuple(buffer.shape) != shape
        ):
            raise ValueError(
                "CSF scratch must be separate contiguous resident byte planes matching source geometry"
            )
        outputs.append(buffer.view(torch.float8_e4m3fn if nv else torch.uint8))
    inner = replace(plan, scale_compression=None)
    planes = []
    if nv:
        from ._impl import _unswizzle_block_scale_bytes

        rotate = plan.source.w13_layout is W13Layout.W31
        for index, (source, shape) in enumerate(zip(inputs, shapes, strict=True)):
            logical = _unswizzle_block_scale_bytes(source, shape[1], shape[2])
            if index == 0 and rotate:
                logical = torch.roll(logical, n, 1)
            planes.append(
                encode_scale_bytes(logical.to(weights.w13.device), format="nvfp4")
            )
        if rotate:
            # Bound temporary storage to one expert half; no process-global
            # normalization registry may keep the original scale plane alive.
            for expert in weights.w13:
                temporary = expert[:n].clone()
                expert[:n].copy_(expert[n:])
                expert[n:].copy_(temporary)
            inner = plan_weights(
                source=replace(inner.source, w13_layout=W13Layout.W13),
                activation=inner.activation,
                geometry=inner.geometry,
                constraints=WeightPlanConstraints(
                    required_packing=inner.prepared_format.packing
                ),
            )
        compressed = Nvfp4CsfWeights(
            packed=replace(
                weights, w13_block_scales=outputs[0], w2_block_scales=outputs[1]
            ),
            w13_scales=planes[0],
            w2_scales=planes[1],
        )
    else:
        for index, source in enumerate(inputs):
            rotation = (
                n if index == 0 and plan.source.w13_layout is W13Layout.W13 else 0
            )
            planes.append(
                encode_scale_bytes(
                    source.view(torch.uint8).to(weights.w13.device),
                    format="mxfp4",
                    exception_row_rotation=rotation,
                )
            )
        if plan.activation.mode is ActivationMode.A16:
            outputs = [
                buffer.view(e, shape[2], shape[1])
                for buffer, shape in zip(outputs, shapes, strict=True)
            ]
        compressed = Mxfp4CsfWeights(
            weights.w13, weights.w2, planes[0], planes[1], *outputs
        )
    prepared = prepare_weights(plan=inner, weights=compressed)
    raw = replace(
        prepared.plan._impl, storage_policy=WeightStoragePolicy.TRANSFER_SOURCE
    )
    return replace(
        prepared,
        plan=replace(prepared.plan, _impl=raw, scale_compression="csf"),
        _impl=replace(prepared._impl, plan=raw),
    )
