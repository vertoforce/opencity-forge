"""Photo pass: triangulate the new object kinds out of the Mapillary detections.

Same geometry and the same rules as s80_furniture_inventory's photo pass (its
ray builder and its grid-peak clustering are imported, not copied) with three
differences, all forced by the classes being new:

  * rays are built for ~21 kinds at once, so the pair test is vectorised: one
    (na x nb) numpy block per panorama pair instead of a Python double loop;
  * the height window above the model floor is per KIND (a car sits on the
    road, a traffic light hangs 4-6 m up, a boat floats below the quay);
  * a vehicle gets a yaw as well as a position: the long axis of the cloud of
    triangulated points that belong to the same cluster, when that cloud is
    elongated enough to have one.

  python -m objects.objphoto [--limit N]   -> SCR/photo_objects.json
"""
import os, sys, json, math, time, gzip, base64, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import common as C
import s80_furniture_inventory as S
from objects.objcommon import SCR, det_path
from objects import objclasses as K

MAX_PER_PANO = {'car': 24, 'boat': 24, 'traffic_sign_face': 16}
MAX_PER_PANO_DEFAULT = 12
GAP_M = 1.0
MAX_RANGE_M = {'car': 40.0, 'boat': 60.0, 'bus': 45.0, 'truck': 45.0,
               'rail_vehicle': 60.0}
RANGE_DEFAULT = 35.0
MIN_BASE_M = 1.5
PAIR_K = 12
MIN_ANGLE_DEG = 12.0


def rays_for_pano(pid, pose):
    """-> (kinds list[str], dirs (n,3), origin (3,)) for one panorama."""
    import mapbox_vector_tile
    p = det_path(pid)
    if not os.path.exists(p):
        return [], np.zeros((0, 3)), None, []
    try:
        data = json.loads(gzip.open(p, 'rt').read()).get('data', [])
    except Exception:
        return [], np.zeros((0, 3)), None, []
    R = np.array(pose['R'], 'f8')
    Cc = np.array([pose['x'], pose['y'], pose['z']], 'f8')
    per = collections.defaultdict(list)
    for d in data:
        v = d.get('value') or ''
        kk = K.photo_kind(v)
        if kk is None or not d.get('geometry'):
            continue
        kind = kk[0]
        try:
            dec = mapbox_vector_tile.decode(base64.decodebytes(d['geometry'].encode()))
        except Exception:
            continue
        for lay in dec.values():
            for ft in lay.get('features', []):
                rings = []

                def walk(cc):
                    if not len(cc):
                        return
                    if isinstance(cc[0], (int, float)):
                        return
                    if isinstance(cc[0][0], (int, float)):
                        rings.append(np.asarray(cc, 'f8')); return
                    for q in cc:
                        walk(q)
                walk((ft.get('geometry') or {}).get('coordinates', []))
                for r in rings:
                    if len(r) < 3:
                        continue
                    uf = r[:, 0] / 4096.0
                    vf = 1.0 - r[:, 1] / 4096.0
                    area = float((uf.max() - uf.min()) * (vf.max() - vf.min()))
                    base = vf >= np.percentile(vf, 90)
                    u = float(uf[base].mean()); vv = float(vf[base].max())
                    lon = 2 * math.pi * (u - 0.5)
                    lat = math.pi * (0.5 - vv)
                    X = np.array([math.cos(lat) * math.sin(lon), -math.sin(lat),
                                  math.cos(lat) * math.cos(lon)], 'f8')
                    dw = R.T @ X
                    dw /= np.linalg.norm(dw)
                    per[kind].append((area, dw, kk[1]))
    kinds, dirs, subs = [], [], []
    for kind, lst in per.items():
        lst.sort(key=lambda t: -t[0])
        for area, dw, sub in lst[:MAX_PER_PANO.get(kind, MAX_PER_PANO_DEFAULT)]:
            kinds.append(kind); dirs.append(dw); subs.append(sub)
    return kinds, (np.array(dirs, 'f8') if dirs else np.zeros((0, 3))), Cc, subs


def run(surf, limit=None, workers=0):
    t0 = time.time()
    P = json.load(open(C.W + '/pano_poses.json'))['poses']
    ids = [k for k, v in P.items() if v.get('confidence') in ('high', 'usable')
           and os.path.exists(det_path(k))]
    gx0, gy0, c, nx, ny = surf.grid
    ids = [k for k in ids if gx0 - 20 < P[k]['x'] < gx0 + nx * c + 20
           and gy0 - 20 < P[k]['y'] < gy0 + ny * c + 20]
    if limit:
        ids = ids[:limit]
    XY = np.array([[P[k]['x'], P[k]['y']] for k in ids], 'f8')
    from scipy.spatial import cKDTree
    tree = cKDTree(XY)
    print('panoramas %d' % len(ids), flush=True)
    cache = {}

    def rays(i):
        if i not in cache:
            if len(cache) > 500:
                cache.clear()
            cache[i] = rays_for_pano(ids[i], P[ids[i]])
        return cache[i]

    kindlist = sorted(set(list(K.SPEC)))
    kidx = {k: i for i, k in enumerate(kindlist)}
    rng = np.array([MAX_RANGE_M.get(k, RANGE_DEFAULT) for k in kindlist])
    cosmin = math.cos(math.radians(MIN_ANGLE_DEG))
    votes = collections.defaultdict(list)
    nray = 0; pairs = 0
    for ai in range(len(ids)):
        ka, Da, Oa, Sa = rays(ai)
        nray += len(ka)
        if not len(ka):
            continue
        ia = np.array([kidx[k] for k in ka], 'i4')
        nb = tree.query(XY[ai], k=min(PAIR_K + 1, len(ids)))[1]
        for bi in np.atleast_1d(nb):
            bi = int(bi)
            if bi <= ai or np.linalg.norm(XY[ai] - XY[bi]) < MIN_BASE_M:
                continue
            kb, Db, Ob, Sb = rays(bi)
            if not len(kb):
                continue
            pairs += 1
            ib = np.array([kidx[k] for k in kb], 'i4')
            same = ia[:, None] == ib[None, :]
            if not same.any():
                continue
            B = Da @ Db.T
            den = 1.0 - B * B
            ok = same & (np.abs(B) < cosmin) & (np.abs(den) > 1e-9)
            if not ok.any():
                continue
            w = Oa - Ob
            d1 = Da @ w; d2 = Db @ w
            with np.errstate(invalid='ignore', divide='ignore'):
                t1 = (B * d2[None, :] - d1[:, None]) / den
                t2 = (d2[None, :] - B * d1[:, None]) / den
            lim = rng[ia][:, None]
            ok &= (t1 > 0.5) & (t2 > 0.5) & (t1 < lim) & (t2 < lim)
            if not ok.any():
                continue
            I, J = np.nonzero(ok)
            p1 = Oa + t1[I, J][:, None] * Da[I]
            p2 = Ob + t2[I, J][:, None] * Db[J]
            gap = np.linalg.norm(p1 - p2, axis=1)
            m = gap <= GAP_M
            if not m.any():
                continue
            I, J, pm = I[m], J[m], (p1[m] + p2[m]) / 2.0
            inb = ((pm[:, 0] > gx0) & (pm[:, 0] < gx0 + nx * c)
                   & (pm[:, 1] > gy0) & (pm[:, 1] < gy0 + ny * c))
            I, J, pm = I[inb], J[inb], pm[inb]
            for n in range(len(I)):
                kd = ka[I[n]]
                zf = surf.floor_z(pm[n, 0], pm[n, 1])
                if not np.isfinite(zf):
                    continue
                lo, hi = K.SPEC[kd]['dz']
                if not (lo <= pm[n, 2] - zf <= hi):
                    continue
                votes[kd].append((pm[n], ai, bi, Sa[I[n]]))
        if ai % 400 == 0:
            print('  pano %d/%d pairs %d votes %d %.0fs'
                  % (ai, len(ids), pairs, sum(len(v) for v in votes.values()),
                     time.time() - t0), flush=True)
    out = cluster(votes, surf)
    json.dump(dict(objects=out, panoramas=len(ids), rays=nray, pairs=pairs,
                   seconds=round(time.time() - t0, 1)),
              open(SCR + '/photo_objects.json', 'w'), indent=1)
    print('photo objects %d %s  %.0fs' % (len(out),
          dict(collections.Counter(o['kind'] for o in out)), time.time() - t0),
          flush=True)
    return out


def cluster(votes, surf):
    """Grid-peak clustering, the s80 rule: a peak cell, everything within
    GAP_M of it, kept when at least two distinct panoramas voted."""
    from scipy.spatial import cKDTree
    gx0, gy0, c, nx, ny = surf.grid
    out = []
    for kind, vs in votes.items():
        if not vs:
            continue
        pts = np.array([v[0] for v in vs], 'f8')
        pa = np.array([[v[1], v[2]] for v in vs], 'i4')
        subs = [v[3] for v in vs]
        gi = np.floor((pts[:, 0] - gx0) / c).astype('i4')
        gj = np.floor((pts[:, 1] - gy0) / c).astype('i4')
        acc = collections.Counter(zip(gi.tolist(), gj.tolist()))
        kd = cKDTree(pts[:, :2])
        taken = np.zeros(len(pts), bool)
        r = max(GAP_M, K.SPEC[kind]['match_m'] * 0.5)
        for (ci, cj), _n in sorted(acc.items(), key=lambda kv: (-kv[1], kv[0])):
            cx = gx0 + (ci + 0.5) * c; cy = gy0 + (cj + 0.5) * c
            idx = np.array(kd.query_ball_point([cx, cy], r), 'i4')
            if not len(idx):
                continue
            idx = idx[~taken[idx]]
            if not len(idx):
                continue
            panos = set(pa[idx].ravel().tolist())
            if len(panos) < 2:
                continue
            taken[idx] = True
            p = pts[idx].mean(0)
            d = pts[idx][:, :2] - p[:2]
            spread = float(np.hypot(d[:, 0], d[:, 1]).max())
            yaw = None; elong = None
            if len(idx) >= 6:
                cov = d.T @ d / len(idx)
                ev, evec = np.linalg.eigh(cov)
                if ev[1] > 1e-6:
                    elong = float(math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
                    if elong > 1.8:
                        ax = evec[:, 1]
                        yaw = float(math.degrees(math.atan2(ax[1], ax[0])) % 180.0)
            sc = collections.Counter(subs[i] for i in idx)
            out.append(dict(kind=kind, subkind=sc.most_common(1)[0][0],
                            x=float(p[0]), y=float(p[1]), z_est=float(p[2]),
                            n_views=len(panos), n_pairs=int(len(idx)),
                            spread_m=round(spread, 2),
                            yaw_deg=(round(yaw, 1) if yaw is not None else None),
                            elongation=(round(elong, 2) if elong else None),
                            subkinds={k: int(v) for k, v in sc.most_common(6)}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    surf = S.Surface()
    run(surf, limit=a.limit)


if __name__ == '__main__':
    main()
