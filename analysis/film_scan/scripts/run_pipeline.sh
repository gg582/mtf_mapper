#!/usr/bin/env bash
# Full film-scan analysis: edge detection -> slanted-edge SFR -> frame selection/aberrations -> per-lens report.
#
# usage: run_pipeline.sh <jobs.txt> <work_dir> <out_dir> [config.json] [parallel_jobs]
#   jobs.txt : one "<roll id>|<image path>" per line (roll ids must match config.json)
#   work_dir : per-image ROI files are written to <work_dir>/<roll>/<image stem>/roi.txt
#   out_dir  : figures, frames.csv, edges.csv, summary.json, lens_summary.json
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
jobs="$1"; work="$2"; out="$3"
cfg="${4:-$here/../config.json}"
par="${5:-8}"
py="${PYTHON:-python3}"

mkdir -p "$work" "$out"
while IFS='|' read -r roll img; do
  [ -n "$roll" ] || continue
  stem="$(basename "$img")"; stem="${stem%.*}"
  mkdir -p "$work/$roll/$stem"
  printf '%s\0%s\0' "$img" "$work/$roll/$stem/roi.txt"
done < "$jobs" | xargs -0 -n 2 -P "$par" "$py" -I "$here/find_edges.py" > "$work/find_edges.log"

"$py" -I "$here/slanted_edge_sfr.py" "$work" "$jobs" roi.txt "$work/edges.pkl"
"$py" -I "$here/analyze.py" "$work/edges.pkl" "$cfg" "$out"
"$py" -I "$here/lens_report.py" "$out" "$cfg"
