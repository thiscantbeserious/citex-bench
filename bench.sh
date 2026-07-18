#!/usr/bin/env bash
# bench.sh — runs inside the container. CPU-only by construction.
set -euo pipefail

DOC_TOKENS="${DOC_TOKENS:-5000}"
OUT_TOKENS="${OUT_TOKENS:-900}"
REPS="${REPS:-2}"
TIERS="${TIERS:-4b 4b-1bit 8b-1bit 27b-1bit}"
THREADS="${THREADS:-$(nproc)}"
LABEL="${LABEL:-$(uname -m)-$(nproc)core}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

tier_spec() {
  case "$1" in
    1.7b)        echo "prism-ml/Ternary-Bonsai-1.7B-gguf|Q2_0" ;;
    1.7b-1bit)   echo "prism-ml/Bonsai-1.7B-gguf|Q1_0" ;;
    4b)          echo "prism-ml/Ternary-Bonsai-4B-gguf|Q2_0" ;;
    8b)          echo "prism-ml/Ternary-Bonsai-8B-gguf|Q2_0" ;;
    4b-1bit)     echo "prism-ml/Bonsai-4B-gguf|Q1_0" ;;
    8b-1bit)     echo "prism-ml/Bonsai-8B-gguf|Q1_0" ;;
    27b-ternary) echo "prism-ml/Ternary-Bonsai-27B-gguf|Q2_0" ;;
    27b-1bit)    echo "prism-ml/Bonsai-27B-gguf|Q1_0" ;;
    *) echo "" ;;
  esac
}

mkdir -p /models /results

# ---------------------------------------------------------------- host profile
log "Host profile"
CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || true)"
[[ -n "$CPU_MODEL" ]] || CPU_MODEL="$(uname -m) (no model string — likely ARM)"
MEM_TOTAL_GB="$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)"
SIMD="$(grep -m1 '^flags\|^Features' /proc/cpuinfo 2>/dev/null \
        | tr ' ' '\n' | grep -Ei '^(avx512f|avx2|neon|asimd|sve)$' | sort -u | tr '\n' ',' || true)"

echo "  label      : $LABEL"
echo "  arch       : $(uname -m)"
echo "  cpu        : $CPU_MODEL"
echo "  cores      : $(nproc)  (using $THREADS threads)"
echo "  ram        : ${MEM_TOTAL_GB} GB visible to container"
echo "  simd       : ${SIMD:-unknown}"

log "Measuring memory bandwidth (STREAM triad) — the thing decode is bound by"
MEMBW="$(membw)"
echo "  ${MEMBW} GB/s"

# ---------------------------------------------------------------- fetch models
list_ggufs() {
  curl -fsSL "https://huggingface.co/api/models/$1" | python3 -c '
import sys, json
d = json.load(sys.stdin)
for s in d.get("siblings", []):
    f = s["rfilename"]
    if f.endswith(".gguf"):
        print(f)
'
}

MODEL_PATHS=()
MODEL_LABELS=()
DL_PIDS=()

for tier in $TIERS; do
  spec="$(tier_spec "$tier")"
  [[ -n "$spec" ]] || die "Unknown tier: $tier"
  repo="${spec%%|*}"; pat="${spec##*|}"

  log "Resolving $tier  ($repo, /$pat/)"
  all=()
  while IFS= read -r line; do [[ -n "$line" ]] && all+=("$line"); done < <(list_ggufs "$repo" || true)
  [[ ${#all[@]} -gt 0 ]] || die "Could not list GGUFs for $repo (network?)"

  cands=()
  for f in "${all[@]}"; do
    [[ "$f" == *"$pat"* ]] || continue
    # PQ2_0 is a reserved-but-unsupported ggml type id — not loadable anywhere yet.
    case "$f" in *mmproj*|*[Dd]spark*|*[Dd]rafter*|*[Ff]16*|*[Bb][Ff]16*|*PQ2_0*) continue ;; esac
    cands+=("$f")
  done
  if [[ ${#cands[@]} -eq 0 ]]; then
    echo "No /$pat/ match. Available:"; printf '  %s\n' "${all[@]}"; die "Fix tier_spec for $tier"
  fi

  # PACK_PREF: g128 = native pack (needs custom kernels)
  #            g64  = matches stock llama.cpp Q2_0 packing — may hit stock CPU kernels
  # The native g128 pack ships as plain "*-Q2_0.gguf" with no "_g128" in the filename,
  # so "want g128" means "the Q2_0 file that isn't the _g64 variant", not a literal match.
  pick=""
  for want in ${PACK_PREF:-_g128 _g64 ANY}; do
    for f in "${cands[@]}"; do
      if [[ "$want" == "ANY" || "$f" == *"$want"* ]] \
         || [[ "$want" == "_g128" && "$f" != *"_g64"* ]]; then
        pick="$f"; break
      fi
    done
    [[ -n "$pick" ]] && break
  done
  echo "  picked : $pick"
  echo "  others : ${cands[*]}"

  dest="/models/$(basename "$pick")"
  if [[ ! -f "$dest" ]]; then
    echo "  downloading in background..."
    hf download "$repo" "$pick" --local-dir /models >/dev/null &
    DL_PIDS+=("$!")
  else
    echo "  cached"
  fi
  MODEL_PATHS+=("$dest")
  MODEL_LABELS+=("$tier")
done

if [[ ${#DL_PIDS[@]} -gt 0 ]]; then
  log "Waiting for ${#DL_PIDS[@]} background download(s)"
  wait "${DL_PIDS[@]}"
  for i in "${!MODEL_PATHS[@]}"; do
    [[ -f "${MODEL_PATHS[$i]}" ]] && continue
    MODEL_PATHS[$i]="$(find /models -name "$(basename "${MODEL_PATHS[$i]}")" -type f | head -1)"
  done
fi

# ---------------------------------------------------------------- bench
log "Benchmarking — CPU only, no accelerator compiled in"
echo "    pp=512,$DOC_TOKENS   tg=128,$OUT_TOKENS   reps=$REPS   threads=$THREADS"
echo "    The 27B rows are slow. This is the point."

JSON="/results/raw-${LABEL}.json"
: > "$JSON"

for i in "${!MODEL_PATHS[@]}"; do
  m="${MODEL_PATHS[$i]}"
  tier="${MODEL_LABELS[$i]}"
  sz="$(du -m "$m" | cut -f1)"
  log "$tier  (${sz} MB on disk)"
  if llama-bench -m "$m" \
        -p "512,$DOC_TOKENS" -n "128,$OUT_TOKENS" \
        -ngl 0 -t "$THREADS" -r "$REPS" -o json > /tmp/out.json 2>/tmp/err.log; then
    python3 -c "
import sys, json
rows = json.load(open('/tmp/out.json'))
for r in rows:
    r['_tier']  = '$tier'
    r['_label'] = '$LABEL'
    r['_membw'] = float('$MEMBW')
    r['_size_mb'] = int('$sz')
    print(json.dumps(r))
" >> "$JSON"
  else
    echo "  FAILED — see below. Common causes: no CPU kernel for this quant, or OOM."
    tail -5 /tmp/err.log | sed 's/^/    /'
  fi
done

# ---------------------------------------------------------------- summary
python3 /opt/summarize.py "$DOC_TOKENS" "$OUT_TOKENS" "$JSON" \
  | tee "/results/summary-${LABEL}.txt"

log "Raw: $JSON"
log "Summary: /results/summary-${LABEL}.txt"
