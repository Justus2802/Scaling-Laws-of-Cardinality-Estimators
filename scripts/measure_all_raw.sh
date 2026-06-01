#!/usr/bin/env bash
# Measure graph signatures of all raw KGs under graphs/data/*/raw/, skipping the
# motif block (Block E). Yago is skipped (too large for available RAM). The
# extensionless Freebase file graphs/data/raw/59621618 is included via a temp .nt copy.
set -u

cd "$(dirname "$0")/.." || exit 1

PY=.venv/bin/python
BLOCKS="a,b,c,d,f"   # all blocks except 'e' (motifs)

# Temp .nt copy so the loader (which needs .nt/.ttl) accepts the extensionless file.
FREEBASE_SRC="graphs/data/raw/59621618"
FREEBASE_NT="graphs/data/raw/59621618.nt"
TMP_MADE=0
if [[ -f "$FREEBASE_SRC" && ! -f "$FREEBASE_NT" ]]; then
  cp "$FREEBASE_SRC" "$FREEBASE_NT"
  TMP_MADE=1
fi

# Smallest -> largest so quick wins land first.
GRAPHS=(
  "graphs/data/fb237_v4/raw/fb237_v4.nt"
  "$FREEBASE_NT"
  "graphs/data/aids/raw/AIDS.nt"
  "graphs/data/codex_l/raw/codex_l.nt"
  "graphs/data/lubm/raw/59410577.ttl"
  "graphs/data/hetionet/raw/hetionet.nt"
)

declare -a OK FAIL
for g in "${GRAPHS[@]}"; do
  echo "============================================================"
  echo ">>> Measuring: $g"
  echo "============================================================"
  if "$PY" scripts/measure_signature.py "$g" --blocks "$BLOCKS"; then
    OK+=("$g")
  else
    FAIL+=("$g")
    echo "!!! FAILED: $g"
  fi
done

# Clean up the temp copy we created.
if [[ "$TMP_MADE" -eq 1 ]]; then
  rm -f "$FREEBASE_NT"
fi

echo
echo "============================================================"
echo "SUMMARY"
echo "  Succeeded (${#OK[@]}):"
for g in "${OK[@]:-}"; do [[ -n "$g" ]] && echo "    - $g"; done
echo "  Failed (${#FAIL[@]}):"
for g in "${FAIL[@]:-}"; do [[ -n "$g" ]] && echo "    - $g"; done
echo "  Skipped: graphs/data/yago/raw/Yago.{nt,ttl} (too large for RAM)"
echo "============================================================"
