#!/usr/bin/env bash
# eval.sh — accuracy check for tiers that already passed the speed budget.
# Runs eval/cases.jsonl through each tier's already-downloaded GGUF via
# llama-completion, comparing DIRECT ({claim,source_ref}) vs QUOTE
# ({claim,quote} + deterministic citation resolution) architectures.
# Speed passing != accuracy passing; this is the other half of "is this tier usable."
#
#   ./eval.sh                          # default tiers (direct vs quote, both modes)
#   TIERS="1.7b-1bit" ./eval.sh        # one tier
#   MODE=quote ./eval.sh               # quote architecture only
set -euo pipefail

TIERS="${TIERS:-1.7b-1bit 4b-1bit 8b-1bit}"
MODE="${MODE:-both}"
# Bonsai model card: temp 0.5 default, range 0.5-0.7. Sweep the full range plus
# a greedy temp=0 control. Sampling params (top-k 20, top-p 0.9, rep 1.0, and the
# 1.7b presence penalty) are applied by eval.py per the card — temp is the only
# axis swept here. Override with TEMPS="0.5,0".
TEMPS="${TEMPS:-0,0.5,0.7}"
MODELS_DIR="${MODELS_DIR:-$PWD/models}"
# THREADS is intentionally unset by default — the host's core count (e.g. via
# sysctl) is NOT the container's; bench.sh discovered this the hard way (host
# reports far more cores than Docker Desktop's VM actually allocates). Let
# eval.py's os.cpu_count(), evaluated inside the container, pick the real number.
THREADS="${THREADS:-}"
FORK_REF="${FORK_REF:-prism}"
IMAGE="bonsai-floor:$(echo "$FORK_REF" | tr '/' '-')"

command -v docker >/dev/null || {
	echo "docker not found"
	exit 1
}

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
arm64 | aarch64) PLATFORM="linux/arm64" ;;
x86_64 | amd64) PLATFORM="linux/amd64" ;;
*)
	echo "Unknown arch $HOST_ARCH"
	exit 1
	;;
esac

echo "==> Building image (adds llama-cli + eval harness on top of run.sh's image)"
docker build --platform "$PLATFORM" --build-arg FORK_REF="$FORK_REF" -t "$IMAGE" .

# tier -> GGUF filename already resolved by a prior ./run.sh (bench.sh is config.json-driven; eval.sh keeps a tier->file map for the smoke tiers)
# macOS ships bash 3.2 (no associative arrays) — use a case statement, not declare -A.
tier_file() {
	case "$1" in
	1.7b) echo "Ternary-Bonsai-1.7B-Q2_0.gguf" ;;
	1.7b-1bit) echo "Bonsai-1.7B-Q1_0.gguf" ;;
	4b) echo "Ternary-Bonsai-4B-Q2_0.gguf" ;;
	4b-1bit) echo "Bonsai-4B-Q1_0.gguf" ;;
	8b-1bit) echo "Bonsai-8B-Q1_0.gguf" ;;
	27b-1bit) echo "Bonsai-27B-Q1_0.gguf" ;;
	*) echo "" ;;
	esac
}

for tier in $TIERS; do
	f="$(tier_file "$tier")"
	[[ -n "$f" ]] || {
		echo "No known GGUF filename for tier '$tier' — add it to tier_file() in eval.sh"
		exit 1
	}
	[[ -f "$MODELS_DIR/$f" ]] || {
		echo "Model not downloaded yet: $f — run TIERS=\"$tier\" ./run.sh first"
		exit 1
	}

	# Plain string + intentional word-splitting below, not an array: bash 3.2
	# (macOS default) treats expanding an empty array under `set -u` as an
	# unbound-variable error. Safe here since $THREADS is always a bare integer.
	THREAD_FLAG=""
	[[ -n "$THREADS" ]] && THREAD_FLAG="--threads $THREADS"

	docker run --rm --platform "$PLATFORM" \
		-v "$MODELS_DIR:/models" \
		-e PYTHONUNBUFFERED=1 \
		--entrypoint python3 \
		"$IMAGE" -u /opt/eval/eval.py --model "/models/$f" --tier "$tier" --mode "$MODE" --temps "$TEMPS" $THREAD_FLAG
done
