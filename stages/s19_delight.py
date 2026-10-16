"""Stage 19 -- delighting: per-panorama exposure and white balance, and a
shadow test.

Shadow artefacts in a panorama-projected texture have two separate causes:

  * every panorama was shot on its own day with its own auto-exposure and auto
    white balance, so neighbouring tiles disagree about how bright the same
    concrete is; and
  * each one baked its own cast shadows into the colour.

This stage measures the first and removes it with one linear-RGB gain per
panorama per channel, solved against a common reference (a latent albedo per
probe point) by robust alternating least squares.  It measures the second two
ways -- a geometric cast-shadow test on the lidar DSM driven by the capture
time, and a statistical one -- and reports which of them actually predicts
observed darkness, because Mapillary capture timestamps can carry inconsistent time
zones.

Everything is measured on the same probe points before and after.
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
import common as C
import texlib as T

OUT = T.SCR + '/delight'
os.makedirs(OUT, exist_ok=True)

R_MAX = 45.0
SAMPLE_W = 1024          # the gain solve only needs low frequencies


# ------------------------------------------------------------ probe set --- #
def probe_points(n, seed=5):
    """Face centroids across ground, structures and facades, thinned onto a
    0.4 m grid so the sample is spatial rather than mesh-density weighted."""
    lay = np.load(C.W + '/geom/face_layer.npy')
    CEN = np.load(C.G + '/centroid.npy', mmap_mode='r')
    rng = np.random.default_rng(seed)
    out = []
    for cl, share in ((1, 0.6), (2, 0.2), (3, 0.2)):
        idx = np.nonzero(lay == cl)[0]
        if len(idx) == 0:
            continue
        take = min(len(idx), int(n * share * 4))
        pick = np.sort(rng.choice(idx, take, replace=False))
        P = np.asarray(CEN[pick]).astype('f4')
        key = ((np.floor(P[:, 0] / 0.4).astype(np.int64) << 32) +
               (np.floor(P[:, 1] / 0.4).astype(np.int64) << 8) +
               np.floor(P[:, 2] / 1.5).astype(np.int64))
        _, first = np.unique(key, return_index=True)
        P = P[first]
        if len(P) > int(n * share):
            P = P[rng.choice(len(P), int(n * share), replace=False)]
        out.append((P, np.full(len(P), cl, 'i1')))
    if not out:
        # no photo mesh (no-mesh mode): probe the 3dfier layers instead, ground
        # as class 1 and buildings as class 3, area-weighted then thinned the same way
        for cl, grp, share in ((1, 'ground_terrain', 0.7), (3, 'buildings', 0.3)):
            f = C.W + f'/layers/{grp}.npz'
            if not os.path.exists(f):
                continue
            d = np.load(f); V, F = d['pos'].astype('f8'), d['idx'].astype('i8')
            if len(F) == 0:
                continue
            A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
            k = int(n * share * 4)
            fi = rng.choice(len(F), k, p=A / A.sum())
            u, v = rng.random(k), rng.random(k); sw = u + v > 1; u[sw], v[sw] = 1 - u[sw], 1 - v[sw]
            P = (V[F[fi, 0]] + u[:, None] * (V[F[fi, 1]] - V[F[fi, 0]]) + v[:, None] * (V[F[fi, 2]] - V[F[fi, 0]])).astype('f4')
            key = ((np.floor(P[:, 0] / 0.4).astype(np.int64) << 32) +
                   (np.floor(P[:, 1] / 0.4).astype(np.int64) << 8) + np.floor(P[:, 2] / 1.5).astype(np.int64))
            _, first = np.unique(key, return_index=True)
            P = P[first]
            if len(P) > int(n * share):
                P = P[rng.choice(len(P), int(n * share), replace=False)]
            out.append((P, np.full(len(P), cl, 'i1')))
    P = np.concatenate([o[0] for o in out])
    L = np.concatenate([o[1] for o in out])
    return P, L


# ------------------------------------------------------------ pass one ---- #
_G = {}


def _init(P, poses, order):
    _G['P'] = P; _G['poses'] = poses; _G['order'] = order
    _G['dsm'] = T.DSM()


def _one(j):
    pid = _G['order'][j]
    try:
        s = T.PanoSampler(_G['dsm'], pid, _G['poses'][pid], r_max=R_MAX, width=SAMPLE_W)
        idx, lin, rng = s.sample(_G['P'])
        s.free()
    except Exception as e:
        return j, None, f"{type(e).__name__}: {e}"
    if len(idx) < 50:
        return j, None, None
    return j, (idx.astype('i4'), (lin * 4095).astype('u2'), rng.astype('f2')), None


def collect(P, poses, order, workers):
    import multiprocessing as mp
    rows_i = []; rows_j = []; rows_c = []; rows_r = []
    errs = 0
    t0 = time.time()
    with mp.Pool(workers, initializer=_init, initargs=(P, poses, order),
                 maxtasksperchild=40) as pool:
        for n, (j, r, err) in enumerate(pool.imap_unordered(_one, range(len(order)), chunksize=2)):
            if err:
                errs += 1
                if errs < 4:
                    print("  err", err, flush=True)
                continue
            if r is None:
                continue
            idx, lin, rng = r
            rows_i.append(idx); rows_j.append(np.full(len(idx), j, 'i4'))
            rows_c.append(lin); rows_r.append(rng)
            if (n + 1) % 300 == 0:
                print(f"  {n+1}/{len(order)} panoramas, {time.time()-t0:.0f}s", flush=True)
    if not rows_i:   # fail with the cause, not a numpy ValueError
        sys.exit(f"s19: no panorama gave samples ({errs} of {len(order)} unreadable); if the area's images_pano was "
                 f"deleted for disk, `opencity build` restores the posed panoramas before its stages (restore_panos)")
    return (np.concatenate(rows_i), np.concatenate(rows_j),
            np.concatenate(rows_c).astype('f4') / 4095.0,
            np.concatenate(rows_r).astype('f4'), errs)


# ------------------------------------------------------------- the solve -- #
def solve_gains(pi, pj, lin, npano, npoint, iters=14, floor=1e-4):
    """log(sample) = log(albedo_point) + gain_pano, robust.

    Alternating means with a trimmed residual: samples more than 0.30 log units
    below the current fit are treated as shadow, samples 0.40 above as specular
    or sky bleed, and both are dropped from the next iteration."""
    L = np.log(np.maximum(lin, floor))
    g = np.zeros((npano, 3), 'f4')
    a = np.zeros((npoint, 3), 'f4')
    w = np.ones(len(pi), bool)
    hist = []
    for it in range(iters):
        # albedo given gains
        r = L - g[pj]
        for k in range(3):
            num = np.bincount(pi[w], weights=r[w, k], minlength=npoint)
            den = np.bincount(pi[w], minlength=npoint).astype('f4')
            a[:, k] = np.where(den > 0, num / np.maximum(den, 1), 0.0)
        # gains given albedo
        r = L - a[pi]
        for k in range(3):
            num = np.bincount(pj[w], weights=r[w, k], minlength=npano)
            den = np.bincount(pj[w], minlength=npano).astype('f4')
            g[:, k] = np.where(den > 0, num / np.maximum(den, 1), 0.0)
        g -= np.median(g, axis=0, keepdims=True)         # anchor: median gain 1.0
        res = L - a[pi] - g[pj]
        rl = res @ T.LUMW
        w = (rl > -0.30) & (rl < 0.40)
        hist.append(dict(iter=it, kept=float(w.mean()),
                         rms=float(np.sqrt((rl[w] ** 2).mean()))))
    return g, a, w, hist


# --------------------------------------------------------- shadow tests --- #
def geometric_shadow(P, poses, order, pi, pj, interp):
    """Cast-shadow flag per sample, using the capture time under one of the two
    possible timestamp conventions."""
    dsm = T.DSM()
    off = 0.0                                         # no area: a 'local' stamp is read as UTC
    # an area takes the sun at its own origin and reads a 'local' stamp in its own zone
    # (area.timezone, per capture, so DST is right).
    in_area = hasattr(C, '_areacfg') and C._areacfg.build_area_cfg(C) is not None
    if in_area:
        from zoneinfo import ZoneInfo
        import datetime as _dt
        _tz = ZoneInfo(C._areacfg.build_area_cfg(C)['area'].get('timezone') or 'UTC')
        _ll = dict(lat_deg=float(C.ORIGIN[0]), lon_deg=float(C.ORIGIN[1]))
    buckets = {}
    for j, pid in enumerate(order):
        ca = poses[pid].get('captured_at')
        if not ca:
            buckets[j] = None
            continue
        if in_area:
            o = 0.0 if interp == 'utc' else -_dt.datetime.fromtimestamp(ca / 1000.0, _tz).utcoffset().total_seconds()
            az, el = T.sun_position(ca / 1000.0 + o, **_ll)
            buckets[j] = (round(az / 10.0) * 10, round(el / 5.0) * 5)
            continue
        az, el = T.sun_position(ca / 1000.0 + off, lat_deg=float(C.ORIGIN[0]), lon_deg=float(C.ORIGIN[1]))
        buckets[j] = (round(az / 10.0) * 10, round(el / 5.0) * 5)
    uniq = sorted({b for b in buckets.values() if b is not None and b[1] > 3})
    maps = {}
    for az, el in uniq:
        maps[(az, el)] = T.shadow_map(dsm, az, el)
    flag = np.zeros(len(pi), bool)
    lit_possible = np.zeros(len(pi), bool)
    x0, y0, c = dsm.x0, dsm.y0, dsm.c
    jj = np.clip(((P[:, 0] - x0) / c).astype(np.int32), 0, dsm.nx - 1)
    ii = np.clip(((P[:, 1] - y0) / c).astype(np.int32), 0, dsm.ny - 1)
    for key, m in maps.items():
        sel = np.array([buckets[j] == key for j in range(len(order))])
        take = sel[pj]
        if not take.any():
            continue
        flag[take] = m[ii[pi[take]], jj[pi[take]]]
        lit_possible[take] = True
    return flag, lit_possible, len(uniq)


# ------------------------------------------------------------ seam stats -- #
def pair_stats(P, pi, pj, lin, order, poses, gains=None, min_shared=60, max_sep=14.0):
    """Median luminance difference between panoramas that see the same points.

    'Adjacent' = camera centres within max_sep metres.  This is the number that
    shows up as a visible seam where two panoramas' colour meets on the ground.
    """
    lum = lin @ T.LUMW if gains is None else (lin / np.exp(gains)[pj]) @ T.LUMW
    o = np.lexsort((pj, pi))
    pi_s = pi[o]; pj_s = pj[o]; lum_s = lum[o]
    bounds = np.searchsorted(pi_s, np.arange(len(P) + 1))
    xy = np.array([[poses[p]['x'], poses[p]['y']] for p in order], 'f4')
    npano = len(order)
    rng = np.random.default_rng(1)
    pts = rng.choice(len(P), min(len(P), 30000), replace=False)
    K = []; V = []
    for i in pts:
        a, b = bounds[i], bounds[i + 1]
        n = b - a
        if n < 2:
            continue
        js = pj_s[a:b]; ls = lum_s[a:b]
        if n > 14:
            pick = rng.choice(n, 14, replace=False)
            js = js[pick]; ls = ls[pick]; n = 14
        u, v = np.triu_indices(n, 1)
        d = xy[js]
        near = ((np.abs(d[u, 0] - d[v, 0]) < max_sep) &
                (np.abs(d[u, 1] - d[v, 1]) < max_sep))
        if not near.any():
            continue
        ju = js[u[near]]; jv = js[v[near]]
        lo = np.minimum(ju, jv).astype(np.int64); hi = np.maximum(ju, jv).astype(np.int64)
        K.append(lo * npano + hi)
        V.append(np.abs(ls[u[near]] - ls[v[near]]))
    if not K:
        return dict(pairs=0, median_abs_dL=0.0, p90_abs_dL=0.0)
    K = np.concatenate(K); V = np.concatenate(V)
    o2 = np.argsort(K, kind='stable'); K = K[o2]; V = V[o2]
    uk, st = np.unique(K, return_index=True)
    st = np.append(st, len(K))
    cnt = np.diff(st)
    meds = np.array([np.median(V[st[t]:st[t + 1]]) for t in range(len(uk)) if cnt[t] >= min_shared])
    if len(meds) < 20:
        meds = np.array([np.median(V[st[t]:st[t + 1]]) for t in range(len(uk)) if cnt[t] >= 10])
    if len(meds) == 0:
        return dict(pairs=0, median_abs_dL=0.0, p90_abs_dL=0.0)
    return dict(pairs=int(len(meds)),
                median_abs_dL=round(float(np.median(meds)) * 255, 2),
                p90_abs_dL=round(float(np.percentile(meds, 90)) * 255, 2))


def point_mad(P, pi, lin, pj=None, gains=None):
    lum = lin @ T.LUMW if gains is None else (lin / np.exp(gains)[pj]) @ T.LUMW
    o = np.argsort(pi, kind='stable')
    pis = pi[o]; ls = lum[o]
    bounds = np.searchsorted(pis, np.arange(len(P) + 1))
    m = []
    for i in range(len(P)):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 3:
            continue
        x = ls[a:b]
        m.append(np.median(np.abs(x - np.median(x))))
    m = np.array(m)
    return dict(points=len(m), median=round(float(np.median(m)) * 255, 2),
                p90=round(float(np.percentile(m, 90)) * 255, 2))


# ------------------------------------------------------------------ main -- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', type=int, default=60000)
    ap.add_argument('--spacing', type=float, default=2.0)
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    poses, src = T.load_poses()
    sel = T.thin_poses(poses, a.spacing)
    order = sorted(sel)
    P, cls = probe_points(a.points)
    print(f"{len(P)} probe points, {len(order)} panoramas (from {len(poses)}), "
          f"provenance: {src}", flush=True)

    pi, pj, lin, rng, errs = collect(P, sel, order, a.workers)
    print(f"{len(pi):,} samples, {errs} panorama errors, {time.time()-t0:.0f}s", flush=True)

    before_pair = pair_stats(P, pi, pj, lin, order, sel)
    before_mad = point_mad(P, pi, lin)
    print("before  pair", json.dumps(before_pair), " point MAD", json.dumps(before_mad), flush=True)

    g, alb, keep, hist = solve_gains(pi, pj, lin, len(order), len(P))
    print("solve", json.dumps(hist[-1]), flush=True)
    after_pair = pair_stats(P, pi, pj, lin, order, sel, gains=g)
    after_mad = point_mad(P, pi, lin, pj=pj, gains=g)
    print("after   pair", json.dumps(after_pair), " point MAD", json.dumps(after_mad), flush=True)

    # ---- shadow: which timestamp convention, if either, predicts darkness -- #
    res_l = (np.log(np.maximum(lin, 1e-4)) - alb[pi] - g[pj]) @ T.LUMW
    shadow_report = {}
    for interp in ('utc', 'local'):
        flag, possible, nbuckets = geometric_shadow(P, sel, order, pi, pj, interp)
        m = possible
        if m.sum() < 1000:
            shadow_report[interp] = dict(usable_samples=int(m.sum()))
            continue
        shadow_report[interp] = dict(
            sun_buckets=nbuckets,
            samples_with_sun_above_horizon=int(m.sum()),
            flagged_fraction=round(float(flag[m].mean()), 3),
            mean_log_resid_shadowed=round(float(res_l[m & flag].mean()), 4)
            if (m & flag).any() else None,
            mean_log_resid_lit=round(float(res_l[m & ~flag].mean()), 4),
            separation_log=round(float(res_l[m & ~flag].mean() - res_l[m & flag].mean()), 4)
            if (m & flag).any() else None)
        print(f"  shadow[{interp}] {json.dumps(shadow_report[interp])}", flush=True)
    # statistical shadow: residual below a cut
    for cut in (-0.20, -0.30, -0.45):
        shadow_report[f'statistical_cut{cut}'] = round(float((res_l < cut).mean()), 4)

    gains = {order[j]: [float(x) for x in np.exp(g[j])] for j in range(len(order))}
    np.savez_compressed(OUT + '/gains.npz',
                        pids=np.array(order, object), gain=np.exp(g).astype('f4'))
    json.dump(dict(provenance=src, panoramas=len(order), probe_points=len(P),
                   samples=int(len(pi)),
                   gain_log_std=[round(float(np.std(g[:, k])), 4) for k in range(3)],
                   gain_range=[[round(float(np.exp(g[:, k]).min()), 3),
                                round(float(np.exp(g[:, k]).max()), 3)] for k in range(3)],
                   solve=hist,
                   pair_luminance_before=before_pair, pair_luminance_after=after_pair,
                   point_mad_before=before_mad, point_mad_after=after_mad,
                   shadow=shadow_report),
              open(OUT + '/delight.json', 'w'), indent=1)
    json.dump(gains, open(OUT + '/gains.json', 'w'))
    note = dict(panoramas=len(order), probe_points=len(P), samples=int(len(pi)),
                pair_luminance_before=before_pair, pair_luminance_after=after_pair,
                point_mad_before=before_mad, point_mad_after=after_mad,
                gain_log_std=[round(float(np.std(g[:, k])), 4) for k in range(3)],
                shadow=shadow_report, pose_provenance=src,
                seconds=round(time.time() - t0))
    C.note('delight', note)
    C.mark('delight', note)
    print("done in", round(time.time() - t0), "s")


if __name__ == '__main__':
    main()
