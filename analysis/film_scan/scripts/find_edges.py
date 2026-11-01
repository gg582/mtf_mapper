"""Find slanted-edge candidates in natural (non-chart) photos and write an mtf_mapper --roi-file.

Usage: find_edges.py <image> <out_roi_file> [max_edges]
Each output line: x1 y1 x2 y2 (two handles along the edge, full-res pixel coords).
"""
import sys
import numpy as np
import cv2

img_path, out_path = sys.argv[1], sys.argv[2]
max_edges = int(sys.argv[3]) if len(sys.argv) > 3 else 400

bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
g = bgr[:, :, 1].astype(np.float32)  # green channel
H, W = g.shape

# contrast-stretch for detection only (many frames are dark scans)
lo, hi = np.percentile(g, (1, 99.5))
det = np.clip((g - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
blur = cv2.GaussianBlur(det, (0, 0), 1.5)
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
segs = lsd.detect(blur)[0]
segs = np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4)
segs = segs[np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) >= 50]

def strip_stats(x1, y1, x2, y2, off):
    """Sample a strip parallel to the segment at perpendicular offset `off` px."""
    d = np.array([x2 - x1, y2 - y1], float)
    L = np.hypot(*d)
    t = d / L
    n = np.array([-t[1], t[0]])
    s = np.linspace(0.15, 0.85, 40)[:, None]
    pts = np.array([x1, y1]) + s * d + off * n
    xs, ys = pts[:, 0], pts[:, 1]
    if xs.min() < 0 or ys.min() < 0 or xs.max() >= W - 1 or ys.max() >= H - 1:
        return None
    v = cv2.remap(g, xs.astype(np.float32)[None], ys.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
    return v.mean(), v.std()

cands = []
rej = {'slant': 0, 'border': 0, 'contrast': 0, 'uniform': 0}
for x1, y1, x2, y2 in segs:
    L = np.hypot(x2 - x1, y2 - y1)
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
    slant = min(ang % 90, 90 - ang % 90)  # deviation from nearest axis
    if not (2.5 <= slant <= 20):
        rej['slant'] += 1
        continue
    if min(x1, x2) < 40 or min(y1, y2) < 40 or max(x1, x2) > W - 40 or max(y1, y2) > H - 40:
        continue
    sides = []
    ok = True
    for off in (-10, -6, 6, 10):
        st = strip_stats(x1, y1, x2, y2, off)
        if st is None:
            ok = False
            break
        sides.append(st)
    if not ok:
        rej['border'] += 1
        continue
    a = np.mean([sides[0][0], sides[1][0]]); b = np.mean([sides[2][0], sides[3][0]])
    noise = max(s[1] for s in sides)
    contrast = abs(a - b)
    dark, bright = min(a, b), max(a, b)
    if contrast < 10 or contrast / (noise + 0.5) < 4 or dark < 2 or bright > 250:
        rej['contrast'] += 1
        continue
    # side uniformity: inner/outer strip means agree (no texture next to the edge)
    if max(abs(sides[0][0] - sides[1][0]), abs(sides[2][0] - sides[3][0])) > 0.2 * contrast:
        rej['uniform'] += 1
        continue
    score = contrast / (noise + 2) * min(L, 300) ** 0.5
    cands.append((score, x1, y1, x2, y2, L))

# spatial non-max suppression: best edge per 8x6 grid cell, then fill to max_edges
cands.sort(reverse=True)
cell = {}
for c in cands:
    cx, cy = (c[1] + c[3]) / 2, (c[2] + c[4]) / 2
    key = (int(cx / W * 8), int(cy / H * 6))
    cell.setdefault(key, []).append(c)
chosen = cands[:max_edges]

with open(out_path, "w") as f:
    for score, x1, y1, x2, y2, L in chosen:
        # shrink handles toward the centre so the ROI stays on the straight part
        d = np.array([x2 - x1, y2 - y1], float)
        p1 = np.array([x1, y1]) + 0.1 * d
        p2 = np.array([x1, y1]) + 0.9 * d
        f.write(f"{p1[0]:.1f} {p1[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}\n")
print(f"{img_path}: {len(segs)} segs, {len(cands)} candidates, {len(chosen)} written", rej)
