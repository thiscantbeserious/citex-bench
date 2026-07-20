#!/usr/bin/env bash
# eval.sh — accuracy grid for the models in eval/config.json.
#
# Runs eval/cases.jsonl through every (model, mode, param-set, rep) cell in
# eval/config.json via llama-completion, compares the direct ({claim,
# source_ref}) vs quote ({claim, quote} + deterministic citation resolution)
# architectures, writes one JSONL capture per (model, mode, slug) under
# reports/captures/, auto-replays every capture through the scorer, then
# summarizes. Speed passing != accuracy passing; this is the other half of
# "is this tier usable."
#
#   ./eval.sh                              # full grid from eval/config.json
#   TIMEOUT=120 ./eval.sh                  # tighter per-cell timeout (smoke)
#   CONFIG=./eval/config.json ./eval.sh    # point at a different config
#
# The model list, param sets, modes, reps, and threads all come from
# eval/config.json. There is no TIERS / MODE / TEMPS / THREADS env here, the
# config drives all of it. To run a subset, edit a config and pass CONFIG=.
set -euo pipefail

FORK_REF="${FORK_REF:-prism}"
IMAGE="bonsai-floor:$(echo "$FORK_REF" | tr '/' '-')"
MODELS_DIR="${MODELS_DIR:-$PWD/models}"
CONFIG="${CONFIG:-$PWD/eval/config.json}"
CASES="${CASES:-$PWD/eval/cases.jsonl}"
CAPTURE_DIR="${CAPTURE_DIR:-$PWD/reports/captures}"
TIMEOUT="${TIMEOUT:-600}"

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

[[ -f "$CONFIG" ]] || {
	echo "config not found: $CONFIG"
	exit 1
}
[[ -f "$CASES" ]] || {
	echo "cases not found: $CASES"
	exit 1
}
mkdir -p "$CAPTURE_DIR"

echo "==> Building image (adds llama-cli + eval harness on top of run.sh's image)"
docker build --platform "$PLATFORM" --build-arg FORK_REF="$FORK_REF" -t "$IMAGE" .

# Mount eval/ read-only so the local eval.py / verify.py / summarize.py /
# config.json / cases.jsonl run inside the container. The baked-in image copies
# drift from the repo, the mount keeps them in sync without a rebuild for code
# changes (a rebuild is still needed when the Dockerfile or deps change).
echo "==> Running accuracy grid"
echo "    config: $CONFIG"
echo "    cases:  $CASES"
echo "    captures: $CAPTURE_DIR"
echo "    timeout: ${TIMEOUT}s per cell"
echo

docker run --rm --platform "$PLATFORM" \
	-v "$MODELS_DIR:/models:ro" \
	-v "$PWD/eval:/opt/eval:ro" \
	-v "$CAPTURE_DIR:/reports/captures" \
	-e PYTHONUNBUFFERED=1 \
	--entrypoint python3 \
	"$IMAGE" -u /opt/eval/eval.py \
	--config /opt/eval/config.json \
	--models-dir /models \
	--cases /opt/eval/cases.jsonl \
	--capture-dir /reports/captures \
	--timeout "$TIMEOUT"

# eval.py auto-replays every capture at the end of the grid and exits nonzero
# on a mismatch. Summarize the captures into a readable report on stdout.
echo
echo "==> Summary"
docker run --rm --platform "$PLATFORM" \
	-v "$PWD/eval:/opt/eval:ro" \
	-v "$CAPTURE_DIR:/reports/captures:ro" \
	--entrypoint python3 \
	"$IMAGE" -u /opt/eval/summarize.py /reports/captures
