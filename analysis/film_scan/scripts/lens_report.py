"""Per-lens summary from analyze.py outputs: resolution, top-3 frames, lateral CA, monitor suitability.

Usage: lens_report.py <analyze_outdir> <config.json>
Lenses are assigned with the "lenses" rules in the config (roll + frame regex). Writes lens_summary.json
next to the inputs and prints Markdown tables.

Monitor criterion: a 3:2 frame shown full-screen on a 16:9 display is height-limited, so the film height
maps to the display's vertical pixels. "sharp" needs MTF50 >= 0.25 cycles/display-pixel, "fine" >= 0.20.
"""
import json
import re
import sys
import numpy as np
import pandas as pd

outdir, cfg_f = sys.argv[1:3]
cfg = json.load(open(cfg_f))
PX = cfg["scan_width_px"] / cfg["frame_width_mm"]
UM = 1000 / PX
H_MM = cfg["frame_height_mm"]

frames = pd.read_csv(f"{outdir}/frames.csv", dtype={"frame": str})
edges = pd.read_csv(f"{outdir}/edges.csv", dtype={"frame": str})


def lens_of(roll, frame):
    for l in cfg["lenses"]:
        if l["roll"] == roll and re.search(l["frame_regex"], frame):
            return l["name"]
    return None


frames["lens"] = [lens_of(r, f) for r, f in zip(frames.roll, frames.frame)]
edges["lens"] = [lens_of(r, f) for r, f in zip(edges.roll, edges.frame)]
el = frames[frames.eligible & frames.lens.notna()]


def monitor_verdict(v):
    sharp = [m["name"] for m in cfg["monitors"] if v >= 0.25 * m["v_px"] / H_MM]
    fine = [m["name"] for m in cfg["monitors"] if v >= 0.20 * m["v_px"] / H_MM]
    return (sharp[-1] if sharp else "-"), (fine[-1] if fine else "-")


out = {}
for l in cfg["lenses"]:
    name = l["name"]
    e = el[el.lens == name].sort_values("p75", ascending=False)
    if e.empty:
        continue
    typ, top3 = float(e.p75.median()), float(e.p75.head(3).mean())
    # lateral CA from edges whose normal is (mostly) radial
    c = edges[(edges.lens == name) & edges.eligible & (edges.n_dot_radial.abs() >= 0.7)].dropna(subset=["pos_R", "pos_G", "pos_B"])
    s = np.sign(c.n_dot_radial)
    dB, dR = (c.pos_B - c.pos_G) * s * UM, (c.pos_R - c.pos_G) * s * UM
    ok = (dB.abs() < 15) & (dR.abs() < 15)
    c, dB, dR = c[ok], dB[ok], dR[ok]
    through0 = lambda y: float((c.r * y).sum() / (c.r ** 2).sum()) if len(c) else float("nan")
    out[name] = dict(
        frames=int(len(e)), median=typ, iqr=[float(e.p75.quantile(.25)), float(e.p75.quantile(.75))],
        top3=[dict(frame=f, mtf50=round(float(v), 1), ci=[round(float(lo), 1), round(float(hi), 1)], edges=int(n))
              for f, v, lo, hi, n in e.head(3)[["frame", "p75", "lo", "hi", "n"]].values],
        top3_mean=top3, ca_corner_um=dict(B=through0(dB), R=through0(dR), n=int(len(c))),
        monitor_typical=monitor_verdict(typ), monitor_top3=monitor_verdict(top3),
    )
json.dump(out, open(f"{outdir}/lens_summary.json", "w"), ensure_ascii=False, indent=1)

print("| 렌즈 | 대상 컷 | 중앙값 | 중간 50% | 상위 3컷 평균 | 청/적 배율 색수차 @모서리 (um) | 모니터: 대표 컷 선명/무난 | 상위 3컷 선명/무난 |")
print("|---|---|---|---|---|---|---|---|")
for n, o in out.items():
    print(f"| {n} | {o['frames']} | {o['median']:.1f} | {o['iqr'][0]:.1f}-{o['iqr'][1]:.1f} | {o['top3_mean']:.1f} | "
          f"{o['ca_corner_um']['B']:+.1f} / {o['ca_corner_um']['R']:+.1f} | "
          f"{o['monitor_typical'][0]} / {o['monitor_typical'][1]} | {o['monitor_top3'][0]} / {o['monitor_top3'][1]} |")
print()
for n, o in out.items():
    print(f"**{n}** 상위 3컷: " + ", ".join(f"`{t['frame']}` {t['mtf50']} ({t['ci'][0]}-{t['ci'][1]}, {t['edges']} edges)" for t in o["top3"]))
