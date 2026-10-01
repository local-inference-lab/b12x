"""Serving decode A/B client for local-inference-lab/b12x#457 (RoCEnante vs NCCL on a three-Spark ring).

For each concurrency in CONCURRENCY, runs one warmup round and then RUNS rounds. A round launches
`c` identical greedy streaming requests at once (ignore_eos, MAX_TOKENS tokens each) and records, per
request: TTFT, completion tokens (server usage), and decode tok/s = (completion_tokens - 1) /
(last token time - first token time). Writes one JSON with the command, endpoint model list, every
raw sample and per-concurrency medians.
Usage: python decode_ab.py ENDPOINT LABEL OUTPUT_JSON
"""
import concurrent.futures as cf
import json
import statistics
import sys
import time

import httpx

ENDPOINT, LABEL, OUT = sys.argv[1:4]
CONCURRENCY = (1, 2)
RUNS = 5
MAX_TOKENS = 512
PROMPT = ("Write a Go package that implements an LRU cache with generics, a mutex for concurrent "
          "use, and a table-driven test file. Explain each design decision briefly.")


def one(client):
    body = {"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    t0 = time.time(); first = last = None; usage = None
    with client.stream("POST", ENDPOINT + "/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.time(); first = first or now; last = now
    n = (usage or {}).get("completion_tokens", 0)
    return {"ttft_s": round(first - t0, 4), "completion_tokens": n,
            "decode_tok_s": round((n - 1) / (last - first), 3)}


out = {"label": LABEL, "command": " ".join(sys.argv), "endpoint": ENDPOINT, "prompt": PROMPT,
       "max_tokens": MAX_TOKENS, "runs": RUNS, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
with httpx.Client(timeout=900) as c:
    out["models"] = c.get(ENDPOINT + "/v1/models").json()
    out["cells"] = []
    for conc in CONCURRENCY:
        samples = []
        for rnd in range(RUNS + 1):
            with cf.ThreadPoolExecutor(conc) as pool:
                rows = list(pool.map(lambda _: one(c), range(conc)))
            if rnd > 0:
                samples.extend(rows)
        rates = [s["decode_tok_s"] for s in samples]
        cell = {"concurrency": conc, "samples": samples,
                "median_decode_tok_s_per_stream": statistics.median(rates),
                "min": min(rates), "max": max(rates),
                "median_ttft_s": statistics.median(s["ttft_s"] for s in samples)}
        out["cells"].append(cell)
        print(json.dumps({k: v for k, v in cell.items() if k != "samples"} | {"label": LABEL}), flush=True)
out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(out, open(OUT, "w"), indent=1)
