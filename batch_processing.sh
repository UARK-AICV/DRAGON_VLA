#!/usr/bin/env bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT="perception_service_batch.py"

VIDEO_ROOT="${VIDEO_ROOT:-assets/grocery_items}"
OUT_ROOT="${OUT_ROOT:-assets/grocery_items}"

# Flags (0/1 env toggles)
OVERWRITE="${OVERWRITE:-1}"          # 1 → pass --overwrite
PER_EPISODE_DIR="${PER_EPISODE_DIR:-1}"  # 1 → pass --per-episode-dir
DRY_RUN="${DRY_RUN:-0}"              # 1 → pass --dry-run

# ── Check arguments ──────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <subject1> [subject2 ...]"
  echo "Example: $0 oliveoil ketchup mayo"
  exit 1
fi

subjects=("$@")

# ── Helper to run Python (uv if available) ────────────────────────────────────
run_py() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$SCRIPT" "$@"
  else
    python "$SCRIPT" "$@"
  fi
}

# ── Loop subjects ─────────────────────────────────────────────────────────────
for subj in "${subjects[@]}"; do
  echo "=== Subject: $subj ==="

  args=(
    --video-root "$VIDEO_ROOT"
    --out-root   "$OUT_ROOT"
    --subject    "$subj"
  )

  [[ "$OVERWRITE" == "1" ]]      && args+=( --overwrite )
  [[ "$PER_EPISODE_DIR" == "1" ]]&& args+=( --per-episode-dir )
  [[ "$DRY_RUN" == "1" ]]        && args+=( --dry-run )

  run_py "${args[@]}"
done

echo "✓ All subjects processed to: $OUT_ROOT"
