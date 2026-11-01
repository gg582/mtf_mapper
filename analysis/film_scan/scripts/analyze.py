"""Frame-selected comparison + aberrations from slanted_edge_sfr.py output.

Usage: analyze.py <edges.pkl> <config.json> <outdir>
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, cfg_f, outdir = sys.argv[1:4]
os.makedirs(outdir, exist_ok=True)
cfg = json.load(open(cfg_f))
PX = cfg["scan_width_px"] / cfg["frame_width_mm"]     # scan pixels per mm on film
UM = 1000 / PX
ROLLS = {r["id"]: r["label"] for r in cfg["rolls"]}
COL = {r["id"]: r["color"] for r in cfg["rolls"]}
CH = {"R": "#e34948", "G": "#008300", "B": "#2a78d6"}
plt.rcParams.update({"font.family": cfg.get("font", "NanumGothic"), "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6, "figure.dpi": 130,
                     "axes.unicode_minus": False})
sel = cfg.get("selection", {})
MIN_CNR, MIN_EDGES, MAX_CIW = sel.get("min_cnr", 20), sel.get("min_edges", 9), sel.get("max_ci_width", 0.30)

d = pd.read_pickle(src)
cand = d.groupby(["roll", "frame"]).size().rename("n_cand")
a = d[(d.reject == "") & d.m50_Y.notna() & (d.m50_Y < 0.45)].copy()
a = a.sort_values("cnr", ascending=False)
keep = []
for _, g in a.groupby(["roll", "frame"]):
    pts = []
    for i, e in g.iterrows():
        if any(np.hypot(e.x - x, e.y - y) < 12 for x, y in pts):
            continue
        pts.append((e.x, e.y))
        keep.append(i)
a = a.loc[keep]
for pat in cfg.get("exclude_frames", []):     # e.g. edited duplicates of an original scan
    a = a[~a.frame.str.contains(pat)]
a["lp"] = a.m50_Y * PX
n_valid_all = a.groupby("roll").size()
cnr_bias = a.assign(b=pd.cut(a.cnr, [0, 10, 20, 40, 1e9])).groupby(["roll", "b"], observed=True).lp.median().unstack(0)
a = a[a.cnr >= MIN_CNR]

rng = np.random.default_rng(0)
rows = []
for (r, f), g in a.groupby(["roll", "frame"]):
    v = g.lp.values
    bs = [np.quantile(rng.choice(v, len(v)), 0.75) for _ in range(2000)]
    rows.append(dict(roll=r, frame=f, n=len(v), n_cand=int(cand[(r, f)]), p75=np.quantile(v, 0.75),
                     lo=np.quantile(bs, 0.05), hi=np.quantile(bs, 0.95), lum=g.frame_lum.iloc[0]))
fr = pd.DataFrame(rows)
fr["acc"] = fr.n / fr.n_cand
fr["ciw"] = (fr.hi - fr.lo) / fr.p75
fr["eligible"] = (fr.n >= MIN_EDGES) & (fr.ciw <= MAX_CIW)
el = fr[fr.eligible]

# in-focus edges of eligible frames: top half of each frame
a = a.merge(fr[["roll", "frame", "eligible"]], on=["roll", "frame"])
a["frank"] = a.groupby(["roll", "frame"]).lp.rank(pct=True)
foc = a[a.eligible & (a.frank >= 0.5)].copy()

S = {}
for r in ROLLS:
    e = el[el.roll == r].sort_values("p75", ascending=False)
    f_ = foc[foc.roll == r]
    curves = np.stack(f_.mtf_curve.values)
    med = np.median(curves, axis=0)
    fq = np.arange(65) / 128 * PX
    def at(lp):
        return float(np.interp(lp, fq, med))
    b10 = np.where(med < 0.1)[0]
    S[r] = dict(
        frames_total=int(d[d.roll == r].frame.nunique()), frames_measured=int(fr[fr.roll == r].shape[0]),
        frames_eligible=int(len(e)), edges_valid=int(n_valid_all[r]),
        acc_rate=float(fr[fr.roll == r].n.sum() / fr[fr.roll == r].n_cand.sum()),
        eligible_median=float(e.p75.median()), eligible_iqr=(float(e.p75.quantile(.25)), float(e.p75.quantile(.75))),
        top3=float(e.p75.head(3).mean()), top3_frames=e.head(3).frame.tolist(),
        mtf_at={lp: at(lp) for lp in (10, 20, 30, 40)},
        mtf10=float(np.interp(0.1, med[b10[0]:b10[0] - 2:-1], fq[b10[0]:b10[0] - 2:-1])) if len(b10) else np.nan,
    )

# field
foc["zone"] = pd.cut(foc.r, [0, 0.3, 0.6, 1.01], labels=["중앙", "중간", "주변"], right=False)
field = foc.groupby(["roll", "zone"], observed=False).lp.agg(["median", "size"])
# astigmatism (off-axis): normal along radius -> meridional; normal across radius -> sagittal
off = foc[foc.r >= 0.45].copy()
nd = off.n_dot_radial.abs()
off["dir"] = np.where(nd >= 0.8, "meridional", np.where(nd <= 0.35, "sagittal", None))
astig = off.dropna(subset=["dir"]).groupby(["roll", "dir"]).lp.agg(["median", "size"])
# lateral CA: R/B edge position minus G, projected on the radial direction (positive = outward)
ca = a[a.eligible & (a.n_dot_radial.abs() >= 0.7)].copy()
sgn = np.sign(ca.n_dot_radial)
ca["dR"] = (ca.pos_R - ca.pos_G) * sgn * UM
ca["dB"] = (ca.pos_B - ca.pos_G) * sgn * UM
ca = ca[(ca.dR.abs() < 15) & (ca.dB.abs() < 15)].dropna(subset=["dR", "dB"])
CA = {}
for r in ROLLS:
    c = ca[ca.roll == r]
    A = np.vstack([np.ones(len(c)), c.r]).T
    o = {"n": int(len(c))}
    for col in ("dR", "dB"):
        coef, res, *_ = np.linalg.lstsq(A, c[col], rcond=None)
        se = np.sqrt(res[0] / (len(c) - 2) * np.linalg.inv(A.T @ A)[1, 1])
        o[col] = (float(coef[0] + coef[1]), float(se))      # predicted shift at r=1, with slope SE
        o[col + "_outer_med"] = float(c[c.r >= 0.6][col].median())
    CA[r] = o
# channel MTF on in-focus eligible edges
chn = {}
for r in ROLLS:
    f_ = foc[foc.roll == r].dropna(subset=["m50_R", "m50_G", "m50_B"])
    chn[r] = dict(n=int(len(f_)), **{c: float(f_[f"m50_{c}"].median() * PX) for c in "RGB"},
                  RG=float((f_.m50_R / f_.m50_G).median()), BG=float((f_.m50_B / f_.m50_G).median()))

# ---------------- charts
# 1. frame-level estimates with CI
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, (r, lab) in enumerate(ROLLS.items()):
    g = fr[fr.roll == r].sort_values("p75")
    xs = i + np.linspace(-0.35, 0.35, len(g))
    for x, (_, e) in zip(xs, g.iterrows()):
        alpha = 1 if e.eligible else 0.25
        ax.plot([x, x], [e.lo, e.hi], color=COL[r], alpha=alpha, lw=1.5)
        ax.scatter([x], [e.p75], color=COL[r], alpha=alpha, s=22 if e.eligible else 12, zorder=3,
                   edgecolor="white", linewidth=0.8)
    med = el[el.roll == r].p75.median()
    ax.plot([i - 0.4, i + 0.4], [med, med], color="#222", lw=1.2, ls="--")
    ax.text(i + 0.42, med, f"{med:.1f}", va="center", fontsize=8, color="#222")
ax.set_xticks(range(len(ROLLS))); ax.set_xticklabels(ROLLS.values()); ax.grid(axis="x", visible=False)
ax.set_ylabel("프레임 MTF50 (lp/mm, 상위 25% 에지)"); ax.set_ylim(10, 46)
ax.set_title("프레임별 해상력 추정치와 90% 신뢰구간 — 진한 점: 비교 대상 컷, 흐린 점: 제외, 점선: 대상 컷 중앙값", fontsize=9)
fig.tight_layout(); fig.savefig(f"{outdir}/fig1_frames.png"); plt.close(fig)

# 2. MTF curves
fig, ax = plt.subplots(figsize=(7.2, 4.0))
fq = np.arange(65) / 128 * PX
for r, lab in ROLLS.items():
    cv = np.stack(foc[foc.roll == r].mtf_curve.values)
    lo, md, hi = np.percentile(cv, [25, 50, 75], axis=0)
    ax.fill_between(fq, lo, hi, color=COL[r], alpha=0.12, lw=0)
    ax.plot(fq, md, color=COL[r], lw=2, label=lab)
ax.axhline(0.5, color="#888", lw=0.8, ls="--"); ax.axhline(0.1, color="#888", lw=0.8, ls=":")
ax.set_xlim(0, 80); ax.set_ylim(0, 1.1)
ax.set_xlabel("필름면 공간주파수 (lp/mm)"); ax.set_ylabel("MTF")
ax.set_title("비교 대상 컷의 초점 맞은 에지 MTF (중앙값, 음영 IQR)", fontsize=10)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{outdir}/fig2_mtf.png"); plt.close(fig)

# 3. lateral CA
fig, axs = plt.subplots(1, len(ROLLS), figsize=(3 * len(ROLLS), 3.2), sharey=True)
for ax, (r, lab) in zip(axs, ROLLS.items()):
    c = ca[ca.roll == r]
    ax.scatter(c.r, c.dR, s=7, color=CH["R"], alpha=0.45, lw=0, label="R-G")
    ax.scatter(c.r, c.dB, s=7, color=CH["B"], alpha=0.45, lw=0, label="B-G")
    for col, cc in (("dR", CH["R"]), ("dB", CH["B"])):
        if len(c) > 5:
            p = np.polyfit(c.r, c[col], 1)
            xx = np.linspace(0, 1, 10)
            ax.plot(xx, np.polyval(p, xx), color=cc, lw=1.5)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlim(0, 1); ax.set_title(f"{lab} (n={len(c)})", fontsize=9); ax.set_xlabel("중심거리 r")
axs[0].set_ylabel("반경 방향 채널 어긋남 (um)"); axs[0].set_ylim(-8, 8); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("배율 색수차 (+ = 바깥쪽)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{outdir}/fig3_lateral_ca.png"); plt.close(fig)

# 4. channel MTF
fig, ax = plt.subplots(figsize=(7.2, 3.3))
w = 0.25
for i, r in enumerate(ROLLS):
    for j, c in enumerate("RGB"):
        v = chn[r][c]
        ax.bar(i + (j - 1) * w, v, w * 0.92, color=CH[c], label=f"{c}" if i == 0 else None)
        ax.text(i + (j - 1) * w, v + 0.4, f"{v:.1f}", ha="center", fontsize=8, color="#333")
ax.set_xticks(range(len(ROLLS))); ax.set_xticklabels(ROLLS.values()); ax.grid(axis="x", visible=False)
ax.set_ylabel("MTF50 (lp/mm)"); ax.set_ylim(0, max(chn[r][c] for r in ROLLS for c in "RGB") * 1.18)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
ax.set_title("채널별 MTF50 (비교 대상 컷, 같은 에지)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{outdir}/fig4_channels.png"); plt.close(fig)

fr.sort_values(["roll", "p75"], ascending=[True, False]).round(3).to_csv(f"{outdir}/frames.csv", index=False)
a.drop(columns=["mtf_curve"]).round(4).to_csv(f"{outdir}/edges.csv", index=False)
out = dict(summary=S, field=field.round(1).reset_index().to_dict("records"),
           astig=astig.round(1).reset_index().to_dict("records"), lateral_ca=CA, channels=chn,
           cnr_bias=cnr_bias.round(1).reset_index().astype(str).to_dict("records"),
           eligible=el.sort_values(["roll", "p75"], ascending=[True, False]).round(1).to_dict("records"))
json.dump(out, open(f"{outdir}/summary.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: out[k] for k in ("summary", "field", "astig", "lateral_ca", "channels", "cnr_bias")},
                 ensure_ascii=False, indent=1, default=lambda v: round(float(v), 2)))
