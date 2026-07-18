# How the models were selected

This isn't a generic llama.cpp benchmark, so the model list is not arbitrary. The
tiers exist to answer one question for a specific pipeline: **can a small
quantized model, running CPU-only on a constrained box, extract cited claims
fast enough and accurately enough to be the floor of a fact-checking tool?**

## The pipeline this serves

A document fact-checking system with three stages:

1. **Claim extraction**, read a document, pull every factual claim that carries
 an inline citation marker (`[1]`, `[2]`), ignore uncited opinion, emit
 structured output. *This is what the benchmark's accuracy eval measures.*
2. **Claim verification**, MiniCheck-RoBERTa (already chosen) checks each claim
 against its cited source. Out of scope here.
3. **Domain credibility**, CRED-1 as veto/flag, never a gate. Out of scope.

**Hard constraint:** CPU floor, interactive latency, **under 30s per ~1-2k token
document.** That constraint is what makes model selection non-trivial: a 70B model
would trivially do the extraction well, and trivially fail the budget.

## Why Bonsai (ternary / 1-bit) and not conventional quants

Decode latency is memory-bandwidth-bound, each generated token streams the whole
weight file through the CPU. So `tok/s ~ bandwidth / model_bytes`, which means
**smaller weights = faster decode at the same bandwidth**, with no quality tax
*if* the low-bit representation preserves behavior.

That "if" is the whole bet. Conventional quants (Q4_K_M, Q8_0) shrink weights
but keep mixed precision and have well-tuned CPU kernels. their accuracy is
predictable. Bonsai's ternary (`{-1,0,+1}`, ~1.71 bits/weight) and 1-bit
(`{-1,+1}`, ~1.125 bits/weight) representations are far more aggressive, ~9.4x
and ~14.2x smaller than FP16, and PrismML claims they retain ~95% / ~90% of
FP16 intelligence. The open question is whether that holds on the exact task
structured extraction at small scale, on CPU, with immature kernels. That is what
this benchmark exists to test.

The tradeoff accepted by choosing Bonsai over conventional quants: **footprint
and decode speed in exchange for kernel maturity and instruction-following
that drops hardest exactly on structured-output tasks.** The bandwidth-model
check in the results (7-12% efficiency vs. the 50-80% a tuned kernel gets) is
the empirical measure of that "kernel maturity" cost.

Full background on the representation and the vendor's own claims:
[FINDINGS.md](../FINDINGS.md), the research writeup, especially the model-card
numbers and their caveats.

## How the tier list was derived

The PrismML `prism-ml` HF org publishes the Bonsai family at four sizes -
**1.7B, 4B, 8B, 27B**, each in both ternary (Q2_0) and 1-bit (Q1_0) GGUF.
There is no 9B. sizes are fixed by what the family ships.

Not every published file is usable. The tier resolution in `bench.sh` filters:

- **`PQ2_0`**, a reserved-but-unsupported ggml type id. excluded (won't load).
- **`_g64` ternary**, mainline llama.cpp's group-64 format. The PrismML fork's
 native kernels expect group-128, shipped as plain `*-Q2_0.gguf` (no `_g128`
 in the filename). A naive substring match silently picked `_g64`, which the
 fork's binary cannot load, a real bug that was fixed (see
 `bench.sh` PACK_PREF comment).
- **mmproj / dspark / drafter**, vision tower and speculative-decoding drafter.
 not part of the language model and not needed for text-only extraction.
- **F16 / BF16**, unquantized reference. excluded from the CPU-floor tiers.

### The tiers actually benchmarked

| tier | repo | quant | deployed size |
|---|---|---|---|
| `1.7b` | `Ternary-Bonsai-1.7B-gguf` | Q2_0 (ternary) | ~0.4 GB |
| `1.7b-1bit` | `Bonsai-1.7B-gguf` | Q1_0 (1-bit) | ~0.2 GB |
| `4b` | `Ternary-Bonsai-4B-gguf` | Q2_0 | ~1.0 GB |
| `4b-1bit` | `Bonsai-4B-gguf` | Q1_0 | ~0.5 GB |
| `8b-1bit` | `Bonsai-8B-gguf` | Q1_0 | ~1.1 GB |
| `27b-1bit` | `Bonsai-27B-gguf` | Q1_0 | ~3.6 GB |

**Default accuracy tiers** (`eval.sh`): `1.7b-1bit`, `4b-1bit`, `8b-1bit`, the
1-bit family, smallest to largest, all under the 30s budget on this hardware.

**Why 1-bit over ternary for the accuracy grid:** smaller footprint -> faster
decode on the same bandwidth, and the 1-bit family is what fits a phone-class
budget, which is the deployment ceiling this work aims toward. Ternary is the
quality-oriented rung. it's kept in the speed bench for comparison but not run
through the full accuracy matrix (yet, see
[eval-redesign-plan.md](eval-redesign-plan.md)).

**Why 27B is in the speed bench but not the accuracy grid:** per-case latency at
27B on this CPU is minutes, not seconds, arithmetically excluded from a sub-30s
interactive budget. Measured in the speed bench to *show* the exclusion, not to
propose it as the floor.

## Why CPU-only, and why the fork

Every accelerator backend is compiled OFF (`GGML_METAL/CUDA/VULKAN/BLAS=OFF`).
The point is the *floor*: what can a plain box with no GPU do? PrismML's low-bit
kernels ship in their `prism`-branch llama.cpp fork, with CPU SIMD paths for
both ARM and x86 (verified at build time by the q2_0-kernel assertion in the
Dockerfile). Stock upstream llama.cpp does not carry these kernels, so a plain
`apt install` build would silently fall back to generic quants and the numbers
would describe upstream, not Bonsai. The fork is load-bearing.
