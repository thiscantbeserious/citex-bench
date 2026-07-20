#!/usr/bin/env bash
# run.sh — build + run the floor benchmark on THIS machine.
#
#   ./run.sh                          # default models from eval/config.json
#   LABEL=thinkpad-t14 ./run.sh       # name the machine for cross-machine compare (do NOT use your real hostname)
#   PACK_PREF="_g64 _g128 ANY" ./run.sh   # test the stock-packing pack first
#   DOC_TOKENS=700 OUT_TOKENS=300 REPS=1 ./run.sh   # scoped workload
#
set -euo pipefail

# User's test machine. Stored as a variable so it's easy to swap per host.
# Not auto-detected from the system, deliberately, to avoid leaking real hostnames.
MACHINE="MacBook Pro 14 - M5 Pro"
LABEL="${LABEL:-$MACHINE}"
FORK_REF="${FORK_REF:-prism}"
IMAGE="bonsai-floor:$(echo "$FORK_REF" | tr '/' '-')"
# The model list comes from eval/config.json (read by bench.sh as CONFIG). No
# TIERS var here, the config is the single source of truth for which models run.
DOC_TOKENS="${DOC_TOKENS:-5000}"
OUT_TOKENS="${OUT_TOKENS:-900}"
REPS="${REPS:-2}"
PACK_PREF="${PACK_PREF:-_g128 _g64 ANY}"
MODELS_DIR="${MODELS_DIR:-$PWD/models}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/results}"

command -v docker >/dev/null || {
	echo "docker not found"
	exit 1
}
mkdir -p "$MODELS_DIR" "$RESULTS_DIR"

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
arm64 | aarch64) PLATFORM="linux/arm64" ;;
x86_64 | amd64) PLATFORM="linux/amd64" ;;
*)
	echo "Unknown arch $HOST_ARCH"
	exit 1
	;;
esac

cat <<EOF

  Building natively for $PLATFORM (host is $HOST_ARCH).

  Do NOT pass --platform to cross-build. An emulated container produces
  numbers that measure QEMU, not your CPU. If you want x86 numbers, run
  this script on an x86 machine.

EOF

echo "==> Building image  (CPU-only: Metal/CUDA/Vulkan/BLAS OFF)  ref=$FORK_REF"
docker build --platform "$PLATFORM" --build-arg FORK_REF="$FORK_REF" -t "$IMAGE" .

echo
echo "==> Running. Model cache: $MODELS_DIR   Results: $RESULTS_DIR"
echo "    Docker Desktop users: Settings > Resources — give the VM all cores"
echo "    and >=12 GB RAM, or the 27B tiers will OOM or thrash."
echo

docker run --rm -it \
	--platform "$PLATFORM" \
	-v "$MODELS_DIR:/models" \
	-v "$PWD/eval:/opt/eval:ro" \
	-v "$RESULTS_DIR:/results" \
	-e LABEL="$LABEL" \
	-e DOC_TOKENS="$DOC_TOKENS" \
	-e OUT_TOKENS="$OUT_TOKENS" \
	-e REPS="$REPS" \
	-e PACK_PREF="$PACK_PREF" \
	"$IMAGE"

echo
echo "==> Done. Compare machines:  cat $RESULTS_DIR/summary-*.txt"
