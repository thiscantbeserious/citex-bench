# Credits and third-party notices

## PrismML (current model and runtime provider)

**PrismML** ([huggingface.co/prism-ml](https://huggingface.co/prism-ml)) is the
model and low-bit-kernel provider this benchmark is currently configured
against, via their Bonsai ternary and 1-bit weight representations and the
CPU kernels in their llama.cpp fork. Thanks to PrismML for pushing for
efficiency. The harness is provider-agnostic, see `bench.sh`'s `tier_spec()`,
so this entry is the first of however many providers get tested.

What this repo uses from PrismML, and how, made explicit so the boundary is
clear:

- **Model weights** (Bonsai ternary / 1-bit GGUF, Apache-2.0). Downloaded at
 runtime by `bench.sh` into `./models/` (gitignored, never redistributed from
 this repo). Users fetch them directly from Hugging Face.
- **llama.cpp fork** (`PrismML-Eng/llama.cpp`, `prism` branch). Cloned inside
 the Docker image at build time. Source fetched from GitHub, not copied into
 this repository.
- **Logo**. The PrismML / Bonsai logo in the README is **hotlinked**, not
 copied. It points to the asset as published in `PrismML-Eng/Bonsai-demo` at
 `assets/bonsai-logo.svg`. If that asset moves, the link breaks and should be
 updated to wherever PrismML publishes it. No PrismML asset is stored in this
 repository.

All trademarks, model names, and brand assets remain the property of their
respective owners. Their use here is for attribution and linking only.

## Authoring tools

Research and harness code were produced with the models and harnesses named in
the README header. They are named for credit, not affiliation.
