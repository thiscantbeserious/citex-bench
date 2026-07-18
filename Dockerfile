# Bonsai floor benchmark — CPU-only, no accelerators, reproducible across machines.
#
# Every accelerator backend is explicitly OFF. This image measures what a machine
# can do with nothing but its CPU cores and its memory bus. Build it natively on
# each machine you want to compare (arm64 host -> arm64 image, amd64 -> amd64);
# do NOT run under emulation, the numbers become fiction.

FROM debian:bookworm-slim

ARG FORK_URL=https://github.com/PrismML-Eng/llama.cpp
# NOTE: the fork's default branch is `prism`. A stale `master` also exists and is
# plain upstream llama.cpp with NO Prism low-bit kernels — cloning it silently
# produces plausible, meaningless numbers. Never default to master.
#   FORK_REF=prism           -> shipped kernels (default)
#   FORK_REF=pr/q2_0-cpu     -> in-flight CPU kernel work (unmerged, diverged)
#   FORK_REF=pr/q2_0-x86     -> in-flight x86 tuning (unmerged, diverged)
#   FORK_REF=pr/q2_0-vulkan  -> in-flight Vulkan kernels (unmerged; needs VULKAN=ON)
ARG FORK_REF=prism

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
        python3 python3-pip python3-venv libgomp1 procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages "huggingface_hub[cli]"

WORKDIR /opt

# ---- llama.cpp (PrismML fork: carries the Q2_0_g128 / Q1_0 low-bit kernels) ----
RUN git clone --depth 1 --branch "${FORK_REF}" "${FORK_URL}" llama.cpp

# Fail the build loudly if we are not actually on a tree with Prism's Q2_0 CPU
# kernels. Guards against the silent "cloned stale upstream master" disaster.
RUN set -e; \
    for f in ggml/src/ggml-cpu/arch/x86/quants.c ggml/src/ggml-cpu/arch/arm/quants.c; do \
        if [ -f "llama.cpp/$f" ] && grep -qi 'q2_0' "llama.cpp/$f"; then \
            echo "OK: q2_0 CPU kernels present in $f"; found=1; \
        fi; \
    done; \
    if [ -z "${found:-}" ]; then \
        echo "FATAL: no q2_0 CPU kernels found on ref '${FORK_REF}'."; \
        echo "You are probably on upstream llama.cpp. Use FORK_REF=prism."; \
        exit 1; \
    fi

# CPU ONLY. No Metal, no CUDA, no Vulkan, no BLAS, no RPC.
# GGML_NATIVE=ON so each machine compiles for its own SIMD (NEON / AVX2 / AVX-512) —
# that is what a real deployment would do, and it is what we want to measure.
#
# aarch64 exception: under Docker Desktop's Linux VM, gcc's -march=native detection
# does not reliably enable +fp16, even on Apple Silicon hosts that support it — ggml's
# NEON fp16 kernels then fail to compile (inlining error, target mismatch). Pin an
# explicit armv8.2-a+fp16+dotprod baseline there instead; x86 keeps true native detection.
RUN set -e; \
    ARCH="$(uname -m)"; \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        EXTRA_FLAGS="-DGGML_NATIVE=OFF -DCMAKE_C_FLAGS=-march=armv8.2-a+fp16+dotprod -DCMAKE_CXX_FLAGS=-march=armv8.2-a+fp16+dotprod"; \
    else \
        EXTRA_FLAGS="-DGGML_NATIVE=ON"; \
    fi; \
    cmake -S llama.cpp -B /opt/build \
        -DCMAKE_BUILD_TYPE=Release \
        $EXTRA_FLAGS \
        -DGGML_METAL=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_RPC=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
    && cmake --build /opt/build -j"$(nproc)" --target llama-bench llama-completion \
    && find /opt/build \( -name 'llama-bench' -o -name 'llama-completion' \) -type f -exec cp {} /usr/local/bin/ \;

# NOTE: the eval uses llama-completion, NOT llama-cli. In this fork llama-cli is an
# interactive chat REPL: it rejects -no-cnv ("please use llama-completion instead")
# and, with stdin at EOF, spins printing "> " prompts forever — a runaway that filled
# ~1 GB of stdout per case and OOM-killed the container before any result printed.
# llama-completion is the non-interactive one-shot tool and needs no LLAMA_BUILD_SERVER.

# Sanity check only — kept separate so a --help quirk can't mask a real build failure above.
RUN llama-bench --help >/dev/null 2>&1 || true

# ---- memory bandwidth probe ----
COPY membw.c /opt/membw.c
RUN gcc -O3 -fopenmp -o /usr/local/bin/membw /opt/membw.c

COPY bench.sh summarize.py /opt/
COPY eval /opt/eval
RUN chmod +x /opt/bench.sh

VOLUME ["/models", "/results"]
ENTRYPOINT ["/opt/bench.sh"]
