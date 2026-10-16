"""Stage 20 -- project the posed, delighted panoramas onto every layer.

This is the colour pass.  One sample point per face (more on large faces), a
line-of-sight test against the lidar DSM, the panorama's own sky mask, an
incidence and range weighting, the per-panorama gain from stage 19, and then a
robust multi-view statistic per face.

The statistic is deliberately NOT the plain median: a plain median keeps a cast
shadow whenever more than half the views were shot in one.  It is the median of
the upper luminance half of the kept samples, which is the "lit" colour of the
surface as long as at least a third of the views saw it lit.

Work is split into spatial tiles with a halo, so each worker owns its own
accumulator and no shared state is needed.  Output is one npz per layer with a
colour, a view count and a multi-view agreement figure per face.
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
import common as C
import texlib as T

OUT = T.SCR + '/project'
os.makedirs(OUT, exist_ok=True)

R_MAX = 42.0
KSLOTS = 12
SAMPLE_W = 2048
TILE = 80.0
HALO = R_MAX + 2.0
MIN_COS = 0.12
LAYERS = ['photo', 'ground_terrain', 'paths', 'structures', 'buildings', 'water']


# ----------------------------------------------------------- the surfels -- #
def surfels_stale(S):
    """Face-count check of a cached surfel set against the current geometry.

    Every face gets at least one surfel and the face index rises within a
    layer, so the cache matches iff each layer's last face index is its face count - 1.
    Returns a short reason, or '' when it matches."""
    lay_s, fid = S['lay'], S['fid']
    lay = np.load(C.W + '/geom/face_layer.npy')
    keep = np.nonzero((lay >= 1) & (lay <= 3))[0]
    m = lay_s == 0
    if int(m.sum()) != len(keep) or (len(keep) and int(fid[m][-1]) != int(keep[-1])):
        return f'photo faces {int(m.sum())} vs {len(keep)}'
    for li, grp in enumerate(LAYERS[1:], start=1):
        fpn = C.W + f'/layers/{grp}.npz'
        nf = len(np.load(fpn)['idx']) if os.path.exists(fpn) else 0
        m = lay_s == li
        got = int(fid[m].max()) + 1 if m.any() else 0
        if got != nf:
            return f'{grp} faces {got} vs {nf}'
    return ''


def build_surfels(force=False):
    """One sample point per face for every layer, plus normal, area and a face
    index back into that layer.  Cached, because it can be millions of rows."""
    fp = OUT + '/surfels.npz'
    if os.path.exists(fp) and not force:
        d = np.load(fp)
        S = {k: d[k] for k in d}
        bad = surfels_stale(S)
        if not bad:
            return S
        print(f'surfels cache does not match the current geometry ({bad}): rebuilding', flush=True)
    pos_all = []; nrm_all = []; fid_all = []; lay_all = []; area_all = []
    # ---- the photogrammetric mesh: ground, structures, facades ------------ #
    lay = np.load(C.W + '/geom/face_layer.npy')
    keep = np.nonzero((lay >= 1) & (lay <= 3))[0]
    P = np.load(C.G + '/pos.npy', mmap_mode='r')
    F = np.load(C.G + '/faces.npy', mmap_mode='r')
    CEN = np.load(C.G + '/centroid.npy', mmap_mode='r')
    step = 1 << 21
    for a in range(0, len(keep), step):
        sub = keep[a:a + step]
        f = np.asarray(F[sub])
        p0 = np.asarray(P[f[:, 0]]); p1 = np.asarray(P[f[:, 1]]); p2 = np.asarray(P[f[:, 2]])
        n = np.cross(p1 - p0, p2 - p0)
        ar = 0.5 * np.linalg.norm(n, axis=1)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        n[n[:, 2] < 0] *= -1.0                     # the mesh is double sided
        pos_all.append(np.asarray(CEN[sub]).astype('f4') + n.astype('f4') * 0.02)
        nrm_all.append(n.astype('f4')); fid_all.append(sub.astype('i4'))
        lay_all.append(np.zeros(len(sub), 'i1'))   # 0 = photo
        area_all.append(ar.astype('f4'))
        del f, p0, p1, p2, n, ar
    # ---- the vertex-coloured layers, subdivided where faces are large ----- #
    for li, grp in enumerate(LAYERS[1:], start=1):
        fpn = C.W + f'/layers/{grp}.npz'
        if not os.path.exists(fpn):
            continue
        d = np.load(fpn)
        pos = d['pos'].astype('f4'); idx = d['idx']
        t = pos[idx]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        ar = 0.5 * np.linalg.norm(n, axis=1)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        n[n[:, 2] < 0] *= -1.0
        cen = t.mean(1)
        # barycentric jitter: k points on a face of area k*0.5 m2, capped
        k = np.clip(np.ceil(ar / 0.6).astype(int), 1, 12)
        rng = np.random.default_rng(17 + li)
        reps = np.repeat(np.arange(len(idx)), k)
        b = rng.random((len(reps), 2))
        flip = b.sum(1) > 1
        b[flip] = 1 - b[flip]
        w0 = 1 - b[:, 0] - b[:, 1]
        pt = (t[reps, 0] * w0[:, None] + t[reps, 1] * b[:, 0:1] + t[reps, 2] * b[:, 1:2])
        pt = 0.35 * pt + 0.65 * cen[reps]           # pull in, away from the edges
        pos_all.append((pt + n[reps] * 0.03).astype('f4'))
        nrm_all.append(n[reps].astype('f4')); fid_all.append(reps.astype('i4'))
        lay_all.append(np.full(len(reps), li, 'i1'))
        area_all.append((ar[reps] / k[reps]).astype('f4'))
    out = dict(pos=np.concatenate(pos_all), nrm=np.concatenate(nrm_all),
               fid=np.concatenate(fid_all), lay=np.concatenate(lay_all),
               area=np.concatenate(area_all))
    np.savez(fp, **out)
    return out


# ------------------------------------------------------------- the tiles -- #
_G = {}


def _init(S, poses, gains):
    _G['S'] = S; _G['poses'] = poses; _G['gains'] = gains
    _G['dsm'] = T.DSM()


def _tile(job):
    tx, ty = job
    S = _G['S']; pos = S['pos']
    x0, x1 = tx * TILE, (tx + 1) * TILE
    y0, y1 = ty * TILE, (ty + 1) * TILE
    sel = np.nonzero((pos[:, 0] >= x0) & (pos[:, 0] < x1) &
                     (pos[:, 1] >= y0) & (pos[:, 1] < y1))[0]
    if len(sel) == 0:
        return job, None
    Pt = pos[sel]; Nt = S['nrm'][sel]
    n = len(sel)
    Wk = np.zeros((n, KSLOTS), 'f4')
    Ck = np.zeros((n, KSLOTS, 3), 'f4')
    npano = 0
    for pid, p in _G['poses'].items():
        if not (x0 - HALO <= p['x'] <= x1 + HALO and y0 - HALO <= p['y'] <= y1 + HALO):
            continue
        try:
            s = T.PanoSampler(_G['dsm'], pid, p, r_max=R_MAX, width=SAMPLE_W)
            idx, lin, rng = s.sample(Pt)
            s.free()
        except Exception:
            continue
        if len(idx) == 0:
            continue
        npano += 1
        d = Pt[idx] - np.array([p['x'], p['y'], p['z']], 'f4')
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        cosi = np.abs((Nt[idx] * d).sum(1))
        ok = cosi > MIN_COS
        if not ok.any():
            continue
        idx = idx[ok]; lin = lin[ok]; rng = rng[ok]; cosi = cosi[ok]
        g = _G['gains'].get(pid)
        if g is not None:
            lin = lin / g
        tier = 1.0 if p['confidence'] == 'high' else 0.55
        w = (tier * cosi * (8.0 / (8.0 + rng))).astype('f4')
        slot = np.argmin(Wk[idx], axis=1)
        cur = Wk[idx, slot]
        take = w > cur
        if not take.any():
            continue
        ii = idx[take]; ss = slot[take]
        Wk[ii, ss] = w[take]
        Ck[ii, ss] = lin[take]
    return job, (sel.astype('i4'), Wk, Ck, npano)


def lit_colour(Wk, Ck):
    """Robust per-face colour from the K kept samples.

    Median of the upper luminance half, which drops cast shadows as long as a
    third of the views saw the surface lit."""
    n, K = Wk.shape
    valid = Wk > 0
    nv = valid.sum(1)
    lum = Ck @ T.LUMW
    lum = np.where(valid, lum, -1.0)
    order = np.argsort(lum, axis=1)
    rows = np.arange(n)[:, None]
    Cs = Ck[rows, order]
    lo = np.clip((K - nv) + (nv * 0.50).astype(int), 0, K - 1)
    hi = np.clip((K - nv) + np.maximum((nv * 0.90).astype(int), 1), 1, K)
    out = np.zeros((n, 3), 'f4')
    mad = np.zeros(n, 'f4')
    # small K: a python loop over the distinct (lo,hi) pairs, vectorised inside
    key = lo.astype(np.int64) * (K + 1) + hi
    for k in np.unique(key):
        m = key == k
        l0 = int(k // (K + 1)); h0 = int(k % (K + 1))
        if h0 <= l0:
            h0 = min(l0 + 1, K)
        out[m] = np.median(Cs[m, l0:h0], axis=1)
    # agreement: MAD of luminance over all valid samples
    lv = np.where(valid, Ck @ T.LUMW, np.nan)
    with np.errstate(invalid='ignore'):
        med = np.nanmedian(lv, axis=1)
        mad = np.nanmedian(np.abs(lv - med[:, None]), axis=1)
    mad = np.nan_to_num(mad, nan=1.0)
    return out, nv.astype('u1'), mad.astype('f4')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spacing', type=float, default=1.5)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    S = build_surfels(a.force)
    print(f"{len(S['pos']):,} surfels "
          f"({(S['lay']==0).sum():,} photo mesh, {(S['lay']>0).sum():,} OSM layers), "
          f"{time.time()-t0:.0f}s", flush=True)
    poses, src = T.load_poses()
    sel = T.thin_poses(poses, a.spacing)
    gains = {}
    gp = T.SCR + '/delight/gains.npz'
    if os.path.exists(gp):
        d = np.load(gp, allow_pickle=True)
        gains = {str(p): g for p, g in zip(d['pids'], d['gain'])}
    print(f"{len(sel)} panoramas, gains for {len(gains)}", flush=True)

    pos = S['pos']
    txs = np.arange(int(np.floor(pos[:, 0].min() / TILE)), int(np.ceil(pos[:, 0].max() / TILE)) + 1)
    tys = np.arange(int(np.floor(pos[:, 1].min() / TILE)), int(np.ceil(pos[:, 1].max() / TILE)) + 1)
    jobs = [(int(x), int(y)) for x in txs for y in tys]
    print(f"{len(jobs)} tiles", flush=True)

    n = len(pos)
    col = np.zeros((n, 3), 'f4'); nv = np.zeros(n, 'u1'); mad = np.zeros(n, 'f4')
    import multiprocessing as mp
    done = 0; tpan = []
    with mp.Pool(a.workers, initializer=_init, initargs=(S, sel, gains),
                 maxtasksperchild=2) as pool:
        for job, r in pool.imap_unordered(_tile, jobs):
            done += 1
            if r is None:
                continue
            idx, Wk, Ck, npano = r
            c, v, m = lit_colour(Wk, Ck)
            col[idx] = c; nv[idx] = v; mad[idx] = m
            tpan.append(npano)
            print(f"  tile {job} {len(idx):>8,} surfels, {npano:>4} panoramas, "
                  f"{done}/{len(jobs)}, {time.time()-t0:.0f}s", flush=True)
    print(f"projection done {time.time()-t0:.0f}s", flush=True)

    # ---- per-layer face aggregation -------------------------------------- #
    stats = {}
    for li, name in enumerate(LAYERS):
        m = S['lay'] == li
        if not m.any():
            continue
        fid = S['fid'][m]; nf = int(fid.max()) + 1
        w = nv[m].astype('f4')
        acc = np.zeros((nf, 3), 'f8'); den = np.zeros(nf, 'f8')
        for k in range(3):
            acc[:, k] = np.bincount(fid, weights=col[m][:, k] * w, minlength=nf)
        den = np.bincount(fid, weights=w, minlength=nf)
        fcol = np.where(den[:, None] > 0, acc / np.maximum(den[:, None], 1e-9), 0.0).astype('f4')
        fnv = np.bincount(fid, weights=nv[m].astype('f8'), minlength=nf)
        fcnt = np.bincount(fid, minlength=nf).astype('f8')
        fnv = (fnv / np.maximum(fcnt, 1)).astype('f4')
        fmad = np.bincount(fid, weights=mad[m].astype('f8') * w, minlength=nf)
        fmad = (fmad / np.maximum(den, 1e-9)).astype('f4')
        area = np.bincount(fid, weights=S['area'][m].astype('f8'), minlength=nf)
        if name == 'buildings':
            # open-shelter posts and slab underside take one flat
            # colour, not the near-black shadow the panoramas measured on them
            # (see the comment above s10a_layers.SHELTER_MAT for the numbers)
            import s10a_layers as SL
            bz = np.load(C.W + '/layers/buildings.npz')
            bm = json.load(open(C.W + '/layers/buildings.json'))
            pf, uf = SL.shelter_faces(bm, bz['pos'], bz['idx'])
            pf = pf[pf < nf]; uf = uf[uf < nf]
            fcol[pf] = SL.shelter_lin(); fcol[uf] = SL.shelter_lin(SL.SHELTER_UNDER_SHADE)
            # and count as well seen and agreed on, so s22's fill_and_smooth keeps
            # the flat colour (keep 0.85) instead of filling an unseen face
            # from the 3 m material mean around it
            sf = np.concatenate([pf, uf]); fnv[sf] = np.maximum(fnv[sf], 5.0); fmad[sf] = 0.0
            stats['open_shelter_faces_flat'] = dict(posts=int(len(pf)), underside=int(len(uf)))
        np.savez_compressed(OUT + f'/{name}_face.npz', col=fcol, nviews=fnv,
                            mad=fmad, area=area.astype('f4'))
        cov = float((area * (fnv >= 1)).sum() / max(area.sum(), 1e-9))
        cov3 = float((area * (fnv >= 3)).sum() / max(area.sum(), 1e-9))
        stats[name] = dict(faces=nf, area_m2=round(float(area.sum()), 1),
                           photo_coverage_area_frac=round(cov, 4),
                           coverage_ge3_views_frac=round(cov3, 4),
                           median_views=float(np.median(fnv[fnv > 0])) if (fnv > 0).any() else 0.0,
                           median_mad_L=round(float(np.median(fmad[fnv >= 3]) * 255), 2)
                           if (fnv >= 3).any() else None)
        print(f"  {name:<16} {stats[name]}", flush=True)

    note = dict(panoramas=len(sel), pose_provenance=src, surfels=int(n),
                r_max_m=R_MAX, sample_width_px=SAMPLE_W, k_slots=KSLOTS,
                layers=stats, seconds=round(time.time() - t0),
                panoramas_per_tile_median=int(np.median(tpan)) if tpan else 0)
    json.dump(note, open(OUT + '/project.json', 'w'), indent=1)
    C.note('project', note); C.mark('project', note)
    print("done", round(time.time() - t0), "s")


if __name__ == '__main__':
    main()
