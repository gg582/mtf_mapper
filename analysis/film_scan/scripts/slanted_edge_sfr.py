"""Slanted-edge SFR on natural edges with strict QC, per channel, plus lateral CA.

Usage: slanted_edge_sfr.py <work_dir> <jobs.txt> <roi_name> <out.pkl>
jobs.txt lines are "<roll>|<image path>"; ROIs are read from <work_dir>/<roll>/<image stem>/<roi_name>.
Edge model is quadratic (curved edges don't smear the ESF). Rejects: irregular edge, non-flat plateau,
second edge inside the window. For accepted edges: MTF50 on luma and on R/G/B, and the R/B edge position
relative to G along the edge normal (lateral CA).
"""
import multiprocessing as mp
import os
import sys
import numpy as np
import pandas as pd
import cv2

root, jobs_f, roi_name, out_f = sys.argv[1:5]
HALF = 16
OS = 4
W, H = 0, 0          # set per image in work()


def srgb_to_lin(v):
    v = v / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def esf_of(dist, v, bins):
    idx = np.digitize(dist, bins) - 1
    nb = len(bins) - 1
    cnt = np.bincount(idx, minlength=nb + 1)[:nb]
    sm = np.bincount(idx, weights=v, minlength=nb + 1)[:nb]
    if (cnt == 0).mean() > 0.05:
        return None
    esf = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
    ok = ~np.isnan(esf)
    return np.interp(np.arange(nb), np.where(ok)[0], esf[ok])


def mtf_of(esf):
    k = 4 * OS
    lo, hi = esf[:k].mean(), esf[-k:].mean()
    lsf = np.gradient(esf) * (1 if hi > lo else -1)
    c = np.argmax(np.convolve(lsf, np.ones(OS) / OS, mode="same"))
    lsf_c = np.roll(lsf, len(lsf) // 2 - c) * np.hanning(len(lsf))
    mtf = np.abs(np.fft.rfft(lsf_c))
    mtf /= mtf[0]
    f = np.fft.rfftfreq(len(lsf_c), d=1 / OS)
    mtf = mtf / np.maximum(np.sinc(f / OS), 0.1)
    below = np.where(mtf < 0.5)[0]
    if not len(below):
        return np.nan, f, mtf
    i = below[0]
    return f[i - 1] + (mtf[i - 1] - 0.5) / (mtf[i - 1] - mtf[i]) * (f[i] - f[i - 1]), f, mtf


def crossing(esf, bins_c):
    """sub-pixel position where the ESF crosses the midpoint between its plateaus"""
    k = 4 * OS
    lo, hi = esf[:k].mean(), esf[-k:].mean()
    e = (esf - lo) / (hi - lo)
    s = np.convolve(e, np.ones(OS) / OS, mode="same")
    j = np.where(np.diff(np.sign(s - 0.5)) != 0)[0]
    if not len(j):
        return np.nan
    j = j[np.argmin(np.abs(bins_c[j]))]
    return bins_c[j] + (0.5 - s[j]) / (s[j + 1] - s[j]) * (bins_c[j + 1] - bins_c[j])


def measure(ch, x1, y1, x2, y2):
    Y = ch["Y"]
    d = np.array([x2 - x1, y2 - y1], float)
    L = np.hypot(*d)
    t = d / L
    n = np.array([-t[1], t[0]])
    cx = [x1 + n[0] * s for s in (-HALF, HALF)] + [x2 + n[0] * s for s in (-HALF, HALF)]
    cy = [y1 + n[1] * s for s in (-HALF, HALF)] + [y2 + n[1] * s for s in (-HALF, HALF)]
    x0, x1b, y0, y1b = int(min(cx)), int(max(cx)) + 1, int(min(cy)), int(max(cy)) + 1
    if x0 < 0 or y0 < 0 or x1b >= Y.shape[1] or y1b >= Y.shape[0]:
        return dict(reject="border")
    yy, xx = np.mgrid[y0:y1b, x0:x1b]
    px = np.stack([xx.ravel() - x1, yy.ravel() - y1], 1)
    along, across = px @ t, px @ n
    m = (along > 0) & (along < L) & (np.abs(across) <= HALF)
    along, across = along[m], across[m]
    sel = {k: v[y0:y1b, x0:x1b].ravel()[m] for k, v in ch.items()}
    vals = sel["Y"]

    nb = max(int(L // 6), 6)
    ea = np.linspace(0, L, nb + 1)
    grid = np.arange(-HALF, HALF + 0.01, 0.5)
    pos, apos = [], []
    for i in range(nb):
        s = (along >= ea[i]) & (along < ea[i + 1])
        if s.sum() < 20:
            continue
        o = np.argsort(across[s])
        prof = np.interp(grid, across[s][o], vals[s][o])
        g = np.abs(np.gradient(np.convolve(prof, np.ones(3) / 3, mode="same")))
        g[:2] = g[-2:] = 0
        pk = np.argmax(g)
        win = slice(max(pk - 12, 0), pk + 13)
        gw = g[win] - np.median(g)
        gw[gw < 0] = 0
        if gw.sum() > 0:
            pos.append((grid[win] * gw).sum() / gw.sum())
            apos.append((ea[i] + ea[i + 1]) / 2)
    if len(pos) < 6:
        return dict(reject="short")
    pos, apos = np.array(pos), np.array(apos)
    p2 = np.polyfit(apos, pos, 2)
    resid = np.std(pos - np.polyval(p2, apos))
    if resid > 1.0 or abs(p2[0]) * (L / 2) ** 2 > 4:
        return dict(reject="irregular")

    dist = (across - np.polyval(p2, along)) / np.sqrt(1 + np.polyval(np.polyder(p2), along) ** 2)
    keep = np.abs(dist) <= HALF - 2
    bins = np.arange(-(HALF - 2), HALF - 2 + 1e-9, 1 / OS)
    bins_c = (bins[:-1] + bins[1:]) / 2
    esfY = esf_of(dist[keep], vals[keep], bins)
    if esfY is None:
        return dict(reject="sparse")
    k = 4 * OS
    lo, hi = esfY[:k], esfY[-k:]
    contrast = abs(hi.mean() - lo.mean())
    if contrast <= 0:
        return dict(reject="flat")
    flat = max(np.ptp(np.convolve(lo, np.ones(OS) / OS, "valid")), np.ptp(np.convolve(hi, np.ones(OS) / OS, "valid"))) / contrast
    lsf = np.convolve(np.gradient(esfY), np.ones(OS) / OS, mode="same") * (1 if hi.mean() > lo.mean() else -1)
    c = np.argmax(lsf)
    side = np.r_[lsf[: max(c - 5 * OS, 0)], lsf[c + 5 * OS:]]
    second = np.abs(side).max() / lsf[c] if len(side) else 0
    if flat > 0.25 or second > 0.3:
        return dict(reject="plateau/second edge")
    noise = max(lo.std(), hi.std())

    out = dict(reject="", L=L, resid=resid, flat=flat, second=second, contrast=contrast,
               cnr=contrast / max(noise, 1e-6), lum=float(min(lo.mean(), hi.mean()) + contrast / 2))
    for cn in "YRGB":
        e = esfY if cn == "Y" else esf_of(dist[keep], sel[cn][keep], bins)
        if e is None or abs(e[-k:].mean() - e[:k].mean()) < 0.2 * contrast:
            out[f"m50_{cn}"] = np.nan
            out[f"pos_{cn}"] = np.nan
            continue
        m50, f, mtf = mtf_of(e)
        out[f"m50_{cn}"] = m50
        out[f"pos_{cn}"] = crossing(e, bins_c)
        if cn == "Y":
            out["mtf_curve"] = np.interp(np.arange(65) / 128, f, mtf)   # 0..0.5 c/p
    xm = float(x1 + d[0] / 2 + n[0] * np.polyval(p2, L / 2))
    ym = float(y1 + d[1] / 2 + n[1] * np.polyval(p2, L / 2))
    u = np.array([xm - W / 2, ym - H / 2])
    rr = np.hypot(*u)
    u = u / max(rr, 1)
    out.update(x=xm, y=ym, r=rr / np.hypot(W / 2, H / 2), n_dot_radial=float(n @ u))
    return out


def work(job):
    roll, img = job
    name = os.path.splitext(os.path.basename(img))[0]
    roi_f = f"{root}/{roll}/{name}/{roi_name}"
    if not os.path.exists(roi_f):
        return []
    bgr = cv2.imdecode(np.fromfile(img, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    global W, H
    H, W = bgr.shape[:2]
    R, G, B = (srgb_to_lin(bgr[:, :, i].astype(np.float32)) for i in (2, 1, 0))
    ch = dict(Y=0.223 * R + 0.717 * G + 0.061 * B, R=R, G=G, B=B)
    frame_lum = float(bgr[::8, ::8, 1].mean())
    out = []
    for kk, line in enumerate(open(roi_f)):
        x1, y1, x2, y2 = map(float, line.split())
        r = measure(ch, x1, y1, x2, y2)
        r.update(roll=roll, frame=name, roi=kk, frame_lum=frame_lum)
        out.append(r)
    return out


if __name__ == "__main__":
    jobs = [tuple(l.rstrip("\n").split("|", 1)) for l in open(jobs_f)]
    with mp.Pool(8) as p:
        res = [r for rs in p.map(work, jobs, chunksize=1) for r in rs]
    df = pd.DataFrame(res)
    df.to_pickle(out_f)
    print(df.groupby("roll").reject.value_counts())
