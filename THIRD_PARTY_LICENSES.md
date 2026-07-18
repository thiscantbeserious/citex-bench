# Third-party licenses and notices

This repository's harness code is MIT-licensed (see [LICENSE](LICENSE)). It
builds on, fetches at runtime, and links to third-party work. None of that
third-party material is stored in or redistributed from this repository.
This file records what is used and under what terms.

## PrismML, Bonsai and the llama.cpp fork

The benchmark is currently configured against models and a runtime built by
PrismML ([huggingface.co/prism-ml](https://huggingface.co/prism-ml)).

- **Model weights**, Bonsai ternary and 1-bit GGUF. Licensed Apache-2.0 by
 PrismML. Downloaded at runtime by `bench.sh` into `./models/` (gitignored)
 from Hugging Face. Not stored in this repository.
- **llama.cpp fork**, `PrismML-Eng/llama.cpp`, `prism` branch. The fork carries
 the low-bit CPU kernels. Cloned inside the Docker image at build time from
 GitHub. Source not copied into this repository. llama.cpp is MIT-licensed.
- **Logo**. The PrismML / Bonsai logo shown in the README is hotlinked, not
 copied. It points to `assets/bonsai-logo.svg` as published in
 `PrismML-Eng/Bonsai-demo`. If that asset moves, the link should be updated to
 wherever PrismML publishes it. No PrismML asset is stored here.

## Authoring tools

Research and harness code were produced with the models and harnesses named in
the README header (GLM 5.2, Ornith 1.0 35B, Claude Code, pi.dev). They are
named for credit, not affiliation.

## Trademarks

All product names, logos, and brands are property of their respective owners.
Use here is for identification and attribution only, and does not imply
endorsement.
