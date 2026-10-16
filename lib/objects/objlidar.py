"""Lidar pass: things standing 0.5-4 m above the model floor that nothing has
inventoried yet.

The 3DEP return cloud is already registered to the photo frame by the combined
build (common.REG_*); this pass only groups it.  Rules:

  * a return counts when its height above the floor (deck, else path, else
    terrain) is 0.5-4.0 m, its classification is not ground (2) or water (9),
    and it is not inside a building mass;
  * returns are binned into 0.5 m cells and 8-connected into clusters;
  * a cluster is dropped when it sits inside a mapped tree crown (the tree
    inventory the build already placed), when it has fewer than MIN_PTS
    returns, or when its footprint is bigger than a street-furniture object (> 8 m);
  * what survives is sized by PCA of its returns: long axis, short axis, p95
    height, and the yaw of the long axis.

The size then names it, which is all lidar alone can do: a 4.5 x 1.9 x 1.5 m
box on a parking aisle is a car, a 2 x 1 x 1 m box on a path is furniture.
The 3DEP flight is not the ortho flight, so a lidar car and an ortho car
in the same bay are two different days; the fuse stage counts that agreement
separately and never merges two transient objects into one.

  python -m objects.objlidar   -> SCR/lidar_clusters.json
"""
import os, sys, json, math, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import common as C
import s80_furniture_inventory as S
from objects.objcommon import SCR

H_LO, H_HI = 0.5, 4.0
CELL = 0.5
MIN_PTS = 8
MAX_FOOT_M = 8.0
MIN_FOOT_M = 0.5


def tree_mask(x, y):
    """True where a mapped tree crown covers the point."""
    try:
        T = json.load(open(C.W + '/tree_inventory.json'))
    except Exception:
        return np.zeros(len(x), bool)
    rows = T['trees'] if isinstance(T, dict) else T
    P = np.array([[t['x_enu_m'], t['y_enu_m'], max(float(t.get('crown_radius_m') or 2.0), 1.0)]
                  for t in rows], 'f8')
    from scipy.spatial import cKDTree
    kd = cKDTree(P[:, :2])
    out = np.zeros(len(x), bool)
    q = kd.query_ball_point(np.stack([x, y], 1), P[:, 2].max())
    for i, idx in enumerate(q):
        if not idx:
            continue
        d = np.hypot(P[idx, 0] - x[i], P[idx, 1] - y[i])
        if (d <= P[idx, 2]).any():
            out[i] = True
    return out


def main():
    t0 = time.time()
    surf = S.Surface()
    reg = np.load(C.FW + '/lidar_reg.npy')
    cls = np.load(C.FW + '/loaded.npz')['lidar_cls']
    if len(cls) != len(reg):
        cls = np.zeros(len(reg), 'i2')
    gx0, gy0, c, nx, ny = surf.grid
    x, y, z = reg[:, 0], reg[:, 1], reg[:, 2]
    keep = ((x > gx0) & (x < gx0 + nx * c) & (y > gy0) & (y < gy0 + ny * c)
            & (cls != 2) & (cls != 9))
    x, y, z, cls = x[keep], y[keep], z[keep], cls[keep]
    i, j = surf._ij(x, y)
    base = np.where(np.isfinite(surf.z_structures[i, j]), surf.z_structures[i, j],
                    np.where(np.isfinite(surf.z_paths[i, j]), surf.z_paths[i, j],
                             surf.z_ground_terrain[i, j]))
    hag = z - base
    m = np.isfinite(hag) & (hag >= H_LO) & (hag <= H_HI)
    m &= ~np.isfinite(surf.z_buildings[i, j])
    x, y, z, hag, cls = x[m], y[m], z[m], hag[m], cls[m]
    print('returns in the 0.5-4 m band, off buildings: %d (%.0fs)'
          % (len(x), time.time() - t0), flush=True)
    tm = tree_mask(x, y)
    x, y, z, hag, cls = x[~tm], y[~tm], z[~tm], hag[~tm], cls[~tm]
    print('after dropping mapped tree crowns: %d' % len(x), flush=True)
    gi = np.floor((x - gx0) / CELL).astype('i4')
    gj = np.floor((y - gy0) / CELL).astype('i4')
    cells = collections.defaultdict(list)
    for n in range(len(x)):
        cells[(gi[n], gj[n])].append(n)
    # 8-connected components over occupied cells
    seen = set(); clusters = []
    for k0 in cells:
        if k0 in seen:
            continue
        stack = [k0]; seen.add(k0); comp = []
        while stack:
            k = stack.pop(); comp.append(k)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    kk = (k[0] + dx, k[1] + dy)
                    if kk in cells and kk not in seen:
                        seen.add(kk); stack.append(kk)
        clusters.append(comp)
    out = []
    for comp in clusters:
        idx = np.concatenate([cells[k] for k in comp])
        if len(idx) < MIN_PTS:
            continue
        px, py, ph = x[idx], y[idx], hag[idx]
        cen = np.array([px.mean(), py.mean()])
        d = np.stack([px - cen[0], py - cen[1]], 1)
        cov = d.T @ d / len(d)
        ev, evec = np.linalg.eigh(cov)
        ax = evec[:, 1]
        lo = float(np.ptp(d @ ax)); sh = float(np.ptp(d @ evec[:, 0]))
        if lo > MAX_FOOT_M or lo < MIN_FOOT_M:
            continue
        h = float(np.percentile(ph, 95))
        zb = float(np.percentile(z[idx], 2))
        out.append(dict(x=float(cen[0]), y=float(cen[1]), z_base=zb,
                        height_m=round(h, 2), long_m=round(lo, 2),
                        short_m=round(sh, 2), points=int(len(idx)),
                        cells=len(comp),
                        yaw_deg=round(float(math.degrees(math.atan2(ax[1], ax[0])) % 180.0), 1),
                        classes={int(k): int(v) for k, v in
                                 collections.Counter(cls[idx].tolist()).most_common(3)}))
    json.dump(dict(clusters=out, band_m=[H_LO, H_HI], cell_m=CELL,
                   min_points=MIN_PTS, returns_used=int(len(x)),
                   seconds=round(time.time() - t0, 1)),
              open(SCR + '/lidar_clusters.json', 'w'), indent=1)
    print('lidar clusters %d, %.0fs' % (len(out), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
