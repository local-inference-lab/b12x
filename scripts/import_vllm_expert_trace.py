#!/usr/bin/env python3
"""Convert native vLLM routed-expert exports from serial, nonspeculative requests.

Input manifest binds each response by SHA256 and declares checkpoint, geometry,
workload and held-out split. Prompt routes are excluded. Each processed generated
token is one decode invocation only under the required C1/no-speculation contract.
Rejected verifier tokens, padding and concurrent schedules cannot be recovered
from accepted-token exports; use a worker invocation trace for those experiments.
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def convert(manifest, root):
    if manifest["execution"] != {"concurrency": 1, "speculative": False}:
        raise ValueError(
            "accepted-token exports require serial nonspeculative execution"
        )
    e, h, i, layers = (
        manifest[k] for k in ("experts", "hidden", "intermediate", "layers")
    )
    if any(type(x) is not int or x < 1 for x in (e, h, i, layers)):
        raise ValueError("invalid model geometry")
    result = dict(
        schema="b12x-routing-invocations-v1",
        checkpoint=manifest["checkpoint"],
        truncated=False,
        provenance=manifest,
        layers=[
            dict(
                layer=f"layer.{n}",
                experts=e,
                hidden=h,
                intermediate=i,
                phase="decode",
                segments=[],
            )
            for n in range(layers)
        ],
    )
    for row in manifest["requests"]:
        data = (root / row["response"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("response integrity hash differs")
        response = json.loads(data)
        if len(response["choices"]) != 1:
            raise ValueError("exactly one generated sequence per response is required")
        array = np.load(
            io.BytesIO(
                base64.b64decode(
                    response["choices"][0]["routed_experts"], validate=True
                )
            ),
            allow_pickle=False,
        )
        prompt = response["usage"]["prompt_tokens"]
        generated = response["usage"]["completion_tokens"]
        if (
            array.ndim != 3
            or array.shape[:2] != (prompt + generated - 1, layers)
            or array.dtype.kind not in "iu"
        ):
            raise ValueError(
                "response does not contain every processed token and layer"
            )
        if not array.shape[2] or np.any(array >= e) or np.any(array < 0):
            raise ValueError("invalid canonical expert IDs")
        # Last prompt token predicts the first output; the final output token has
        # not entered the model. Decode invocations are exactly generated-1 rows.
        array = array[prompt:]
        for n, layer in enumerate(result["layers"]):
            layer["segments"].append(
                dict(
                    request=row["request"],
                    workload=row["workload"],
                    split=row["split"],
                    calls=[dict(ids=[ids.tolist()]) for ids in array[:, n, :]],
                )
            )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = convert(json.loads(a.manifest.read_text()), a.manifest.parent)
    with a.output.open("x") as output:
        json.dump(data, output)


if __name__ == "__main__":
    main()
