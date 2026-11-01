# Film scan resolution / aberration analysis

Measures resolution (MTF) and aberrations of real film photos, with no test chart, by finding natural slanted edges
in the scans. Results for three rolls (Helios-44M-4 58mm f/2, Meyer-Optik Lydith 30mm f/3.5, Pentacon 50mm f/1.8 auto)
are in [REPORT.md](REPORT.md) (Korean).

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | `scripts/find_edges.py` | LSD line segments tilted 2.5–20° from the axes, with uniform sides and no clipping → ROI file (`x1 y1 x2 y2` per line, same format as `mtf_mapper --roi-file`) |
| 2 | `scripts/slanted_edge_sfr.py` | ISO 12233-style SFR on linearised luma and on R/G/B. Quadratic edge model (curved edges don't smear the ESF), rejects irregular edges, non-flat plateaus and double edges. Also R/B edge position vs G (lateral CA) |
| 3 | `scripts/analyze.py` | dedupe, CNR filter, per-frame estimate (p75 of edge MTF50) with bootstrap CI, selection of comparable frames, field / astigmatism / lateral CA / per-channel MTF, figures |
| 4 | `scripts/lens_report.py` | per-lens summary from the `lenses` rules in `config.json`: top-3 frames, CA, which monitor resolution the images hold up on |
| — | `scripts/run_mtf_mapper.sh` | cross-check: run `mtf_mapper` on the same ROIs (agreement within ±3% on matched edges) |
| — | `scripts/build_mtf_mapper_cli.sh` | build the `mtf_mapper` CLI without root and without Qt (`-DMTFMAPPER_BUILD_GUI=OFF`) |

```sh
pip install -r scripts/requirements.txt
# jobs.txt: one "<roll id>|<image path>" per line; roll ids must match config.json
scripts/run_pipeline.sh jobs.txt work/ out/ config.json 8
```

`config.json` holds the scan width in pixels, the film frame size, the roll labels, the frame-selection thresholds,
the lens assignment rules (roll + frame-name regex) and the monitor list.

## Method notes

- Units are lp/mm on the film: `scan_width_px / frame_width_mm` scan pixels per mm.
- Edges with CNR < 20 are dropped, because grain noise biases MTF50 upward on low-CNR edges.
- Rolls differ a lot in how many edges pass validation (10–32%), so each frame is estimated first, then frames are compared.
  A frame is used for comparison only if it has ≥ 9 valid edges and a 90% CI narrower than 30% of the estimate.
  Selection uses measurement reliability only, not sharpness.
- Monitor criterion: a 3:2 frame shown full-screen on a 16:9 display (height-limited). "Sharp" needs MTF50 ≥ 0.25 cycles per
  display pixel, "fine" needs ≥ 0.20.
- What is measured is the whole system (lens × focus × shake × film × scanner × sharpening × JPEG), not the lens alone.

## Data

`data/` has the per-frame and per-edge tables and the JSON summaries behind the report. `figures/` has the report's figures.
The full rolls are not included. `samples/` has the original scans of the top 3 frames per lens (by frame MTF50),
except frames with people in them:

| lens | folder | frames (MTF50, lp/mm) |
|---|---|---|
| Helios-44M-4 58mm f/2 | `samples/helios-44m-4_58mm/` | 75420005 (29.2), 75450005 (25.6), 75440003 (24.2) |
| Meyer-Optik Lydith 30mm f/3.5 | `samples/lydith_30mm/` | 03000005 (37.7), 03000007 (33.4); 03000008 (31.3) not included |
| Pentacon 50mm f/1.8 auto | `samples/pentacon_50mm/` | 03000027 (38.2), 03000026 (37.7), 03000028 (31.1) |
