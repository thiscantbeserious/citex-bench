#!/usr/bin/env python3
"""Turn llama-bench rows into the number that actually matters:
wall-clock seconds to extract claims from one document."""
import json, sys, collections

doc, out, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

rows = [json.loads(l) for l in open(path) if l.strip()]
if not rows:
    sys.exit("No results — every benchmark run failed. Check the errors above.")

agg = collections.defaultdict(dict)
meta = {}
for r in rows:
    k = r["_tier"]
    meta[k] = {"membw": r["_membw"], "size_mb": r["_size_mb"], "label": r["_label"]}
    if r.get("n_prompt", 0) > 0:
        agg[k][f"pp{r['n_prompt']}"] = r.get("avg_ts")
    elif r.get("n_gen", 0) > 0:
        agg[k][f"tg{r['n_gen']}"] = r.get("avg_ts")

label = next(iter(meta.values()))["label"]
membw = next(iter(meta.values()))["membw"]

print(f"\n{'='*84}")
print(f"  MACHINE: {label}        measured memory bandwidth: {membw} GB/s")
print(f"  WORKLOAD: {doc}-token document in  ->  {out} tokens of claims out")
print(f"{'='*84}\n")

hdr = (f"{'tier':<14}{'size':>7}{'PP'+str(doc)+' t/s':>13}{'TG'+str(out)+' t/s':>13}"
       f"{'prefill':>10}{'decode':>10}{'TOTAL':>10}{'<30s':>7}")
print(hdr)
print("-" * len(hdr))

for tier, v in agg.items():
    pp, tg = v.get(f"pp{doc}"), v.get(f"tg{out}")
    sz = meta[tier]["size_mb"] / 1024.0
    if not pp or not tg:
        print(f"{tier:<14}{sz:>6.1f}G{'-- incomplete --':>28}")
        continue
    pre, dec = doc / pp, out / tg
    tot = pre + dec
    print(f"{tier:<14}{sz:>6.1f}G{pp:>13.1f}{tg:>13.1f}"
          f"{pre:>9.1f}s{dec:>9.1f}s{tot:>9.1f}s{'  OK' if tot < 30 else '  NO':>7}")

# --- sanity check: does the bandwidth model explain decode? ---
print(f"\n{'-'*84}")
print("Bandwidth model check  (decode is bandwidth-bound: tok/s ~= GB/s / model_GB)")
print(f"{'tier':<14}{'predicted':>12}{'measured':>12}{'efficiency':>12}")
print("-" * 50)
for tier, v in agg.items():
    tg = v.get(f"tg{out}")
    sz = meta[tier]["size_mb"] / 1024.0
    if not tg or sz <= 0:
        continue
    pred = membw / sz
    print(f"{tier:<14}{pred:>11.1f}{tg:>12.1f}{tg/pred*100:>11.0f}%")

print("""
If efficiency is ~50-80%, the bandwidth model holds and you can predict any
machine from its GB/s alone. If it is far lower, this quant has no optimized
CPU kernel on this architecture -- which is itself the finding.

Reading this across machines: memory bandwidth is the explanatory variable for
decode, core count and SIMD width for prefill. Record both per machine and the
tiers stop being a mystery.
""")
