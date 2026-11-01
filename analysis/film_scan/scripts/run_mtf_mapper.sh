#!/usr/bin/env bash
# Cross-check: run mtf_mapper on the ROIs found by find_edges.py (edge MTF, v2 output, CA on all edges).
# usage: run_mtf_mapper.sh <mtf_mapper binary> <image> <roi.txt> <out_dir>
set -euo pipefail
mkdir -p "$4"
"$1" --roi-file "$3" -e -q -v 2 --ca --ca-all-edges "$2" "$4" > "$4/mtf.log" 2>&1
