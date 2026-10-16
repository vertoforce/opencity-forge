"""Stage 60 step 2 -- rectify every view onto the rect plane, then align.

Two corrections, both measured per rect and both reported:

  1. RELATIVE.  Every view of a rect is correlated against that rect's
     reference view on the rect's own plane (masked normalised cross
     correlation, Padfield, Masked Object Registration in the Fourier Domain), giving a 2D shift in rect metres.  This is
     what makes two neighbouring patches taken from two different panoramas
     line up with each other; without it the ~1 m pose class slides one patch
     against the next and the seam shows as a doubled window.
  2. ABSOLUTE.  The aligned set is pinned to the rect's own geometry by the
     silhouette: the elevation of the rect's top edge against the panorama's
     measured sky boundary gives the vertical shift, and a 1D correlation of
     the predicted silhouette profile (rect top inside the rect's azimuth span,
     lidar DSM skyline outside it) against the measured one gives the
     horizontal shift.  Only columns where the rect really is the top-most
     surface are used.
"""
import os, sys, json, time, math
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import texlib as T
from wallphoto import wpcommon as H
from wallphoto.wpsample import ViewCtx, sample_rect

CHIP_W = 4096            # decode width for the alignment chips
PAIR_MIN_FRAC = 0.02     # a view has to see this much of a rect to be chipped
MAX_VIEWS = 28           # per rect, the best ones by area seen
MIN_OVERLAP = 500        # texels two chips must share for a believable shift
ANCHOR_CLAMP = 2.00      # m; the hard cap on a silhouette shift.  A shift over
TRUST_ANCHOR = 0.70      # TRUST_ANCHOR only survives when several views agree
ANCHOR_SPREAD = 0.35     # on it to within ANCHOR_SPREAD; otherwise it is the
                         # measurement that is wrong, not the wall that is short


# ---------------------------------------------------------------- chips ---- #
_G = {}


# chips and tiles are filed under the rect id, and a rect
# id is the s33 wall id, which is NOT stable (adding geometry renumbers walls).
# A cached chip/tile is trusted only when it carries the current wall's geometric
# signature, the same string stage 34 stores in its projection cache
# (s34_wallproj.wall_sig), not a second scheme.  Files without one are stale.
def rect_sig(rc):
    from s34_wallproj import wall_sig
    return wall_sig(dict(part_name=rc.meta.get('part_name'), origin=rc.origin,
                         u=rc.u, width_m=rc.width, height_m=rc.height))


def rect_sigs(rects):
    return {int(r.id): rect_sig(r) for r in rects}


def rec_matches(rc, r):
    """Stage 35's pack_matches test (part, area 0.5 m2, w/h 0.10 m) between a
    rect and a per-wall record that carries no geometry beyond its size."""
    if r is None or r.get('part_name') != rc.meta.get('part_name'):
        return False
    if abs(float(r.get('area_m2') or 0.0) - float(rc.meta.get('area_m2') or 0.0)) > 0.5:
        return False
    if r.get('width_m') and abs(float(r['width_m']) - rc.width) > 0.10:
        return False
    if r.get('height_m') and abs(float(r['height_m']) - rc.height) > 0.10:
        return False
    return True


def chip_ok(meta, sig, pid):
    """A chip on disk is this rect's chip from this panorama."""
    return (meta.get('sig') == sig and meta.get('pid') == pid
            and bool(meta.get('unvetoed', False)) == _ALIGN_UNVETOED)


def _init(setname, poses, pids, jobs, sigs=None):
    rects, _ = H.load_rects(setname)
    _G['rects'] = {r.id: r for r in rects}
    _G['poses'] = poses; _G['pids'] = pids; _G['jobs'] = jobs
    _G['dsm'] = T.DSM()
    _G['set'] = setname
    _G['sigs'] = sigs if sigs is not None else rect_sigs(rects)


def chip_path(setname, rid, k):
    return f'{H.setdir(setname)}/chips/r{rid:04d}/v{k:04d}.npz'


def _chip_one(k):
    """All the rects one panorama is wanted for, from one decode."""
    setname = _G['set']
    pid = _G['pids'][k]
    rids = [r for r in _G['jobs'].get(k, [])
            if not os.path.exists(chip_path(setname, r, k))]
    if not rids:
        return 0
    try:
        ctx = ViewCtx(pid, _G['poses'][pid], _G['dsm'])
    except Exception:
        return 0
    if _ALIGN_UNVETOED:
        # FACADE_PHOTO_QUALITY.fallback.align_unvetoed: the
        # alignment chips register the panorama, they do not colour the wall, so
        # they keep the not-wall pixels the bake vetoes (with the veto applied,
        # correlation places fewer views, and the next-best views a vetoed texel
        # falls back to are mostly unaligned ones)
        ctx.seg = None
    n = 0
    for rid in rids:
        rc = _G['rects'][rid]
        tx = H.chip_texel(rc)
        uu, vv = H.grid_axes(rc, tx, pad=H.PAD_M)
        res = sample_rect(ctx, rc, uu, vv, width=CHIP_W)
        if res['valid'].sum() < 200:
            continue
        p = chip_path(setname, rid, k)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez_compressed(p + '.tmp.npz', rgb=res['rgb'], valid=res['valid'],
                            rng=res['rng'].astype('f2'),
                            cos=res['cos'].astype('f2'),
                            meta=np.array(json.dumps(dict(
                                pid=pid, view=int(k), rect=int(rid), tx=float(tx),
                                sig=_G['sigs'][int(rid)],
                                nu=len(uu), nv=len(vv), pad=H.PAD_M,
                                masked=int(res['masked']), green=int(res['green']),
                                looked=int(res['looked']),
                                **({'unvetoed': True} if _ALIGN_UNVETOED else {})))))
        os.replace(p + '.tmp.npz', p)
        n += 1
    ctx.free()
    return n


_ALIGN_UNVETOED = False
if H._AP:
    import facade_quality as _FQ
    _ALIGN_UNVETOED = bool((_FQ.fallback() or {}).get('align_unvetoed'))


def chip_jobs(setname, only=None):
    """-> (pids, {view: [rect, ...]}, pairs): the (rect, view) chips a run wants."""
    d = np.load(H.setdir(setname) + '/work/cover.npz', allow_pickle=True)
    rect = d['rect']; view = d['view']; nval = d['nval']; ntot = d['ntot']
    pids = [str(x) for x in d['pids']]
    key = rect.astype('i8') * (len(pids) + 1) + view
    o = np.argsort(key, kind='stable')
    key = key[o]; nv = nval[o]; nt = ntot[o]
    first = np.ones(len(key), bool); first[1:] = key[1:] != key[:-1]
    gid = np.cumsum(first) - 1
    sv = np.bincount(gid, weights=nv); st = np.bincount(gid, weights=nt)
    gk = key[first]
    grect = (gk // (len(pids) + 1)).astype('i4')
    gview = (gk % (len(pids) + 1)).astype('i4')
    frac = sv / np.maximum(st, 1)
    keep = frac >= PAIR_MIN_FRAC
    jobs = {}
    npair = 0
    for rid in np.unique(grect[keep]):
        if only is not None and int(rid) not in only:
            continue
        m = keep & (grect == rid)
        # the views that see the most of this rect
        idx = np.nonzero(m)[0]
        idx = idx[np.argsort(-sv[idx])][:MAX_VIEWS]
        for i in idx:
            jobs.setdefault(int(gview[i]), []).append(int(rid))
            npair += 1
    return pids, jobs, npair


def run_chips(setname, workers=16, only=None):
    t0 = time.time()
    D = H.setdir(setname)
    pids, jobs, npair = chip_jobs(setname, only)
    H.log(f"chips[{setname}]: {npair} (rect, view) chips over {len(jobs)} panoramas")
    rects, _ = H.load_rects(setname)
    sigs = rect_sigs(rects)
    nst = purge_chips(setname, sigs, pids, jobs, only)
    H.log(f"chips[{setname}]: {nst} stale chip files removed (id/geometry signature, "
          f"panorama, or a view this run did not choose)")
    poses, _ = T.load_poses()
    sel = {p: poses[p] for p in pids if p in poses}
    import multiprocessing as mp
    made = 0
    todo = sorted(jobs)
    with mp.Pool(workers, initializer=_init,
                 initargs=(setname, sel, pids, jobs, sigs)) as pool:
        for n, r in enumerate(pool.imap_unordered(_chip_one, todo, chunksize=1)):
            made += r
            if (n + 1) % 100 == 0:
                H.log(f"  {n+1}/{len(todo)} panoramas, {made} chips, "
                      f"{time.time()-t0:.0f}s")
    rep = dict(chips_written=int(made), pairs=int(npair), stale_removed=int(nst),
               panoramas=len(jobs), chip_decode_px=CHIP_W,
               pair_min_frac=PAIR_MIN_FRAC, max_views_per_rect=MAX_VIEWS,
               seconds=round(time.time() - t0))
    json.dump(rep, open(D + '/work/chips.json', 'w'), indent=1)
    H.mark(setname, 'chips', rep)
    H.log(f"chips[{setname}]: {rep}")
    return rep


# ------------------------------------------------------------- the NCC ----- #
def masked_ncc(A, MA, B, MB, max_shift, min_overlap=MIN_OVERLAP):
    """Padfield's masked normalised cross correlation.

    Returns (dr, dc, ncc, overlap) where B[i + dr, j + dc] is the same piece of
    wall as A[i, j], i.e. B's photograph sits (dr, dc) texels further along.
    """
    ny, nx = A.shape
    sy = min(int(max_shift), ny - 1); sx = min(int(max_shift), nx - 1)
    fy = int(2 ** math.ceil(math.log2(ny + sy + 1)))
    fx = int(2 ** math.ceil(math.log2(nx + sx + 1)))
    f1 = np.where(MA, A, 0.0).astype('f8'); m1 = MA.astype('f8')
    f2 = np.where(MB, B, 0.0).astype('f8'); m2 = MB.astype('f8')

    def F(x):
        return np.fft.rfft2(x, s=(fy, fx))

    def corr(Fx, Fy):
        return np.fft.irfft2(np.conj(Fx) * Fy, s=(fy, fx))

    F1, M1 = F(f1), F(m1)
    F2, M2 = F(f2), F(m2)
    F1s, M2s = F(f1 * f1), F(m2)
    M1s, F2s = F(m1), F(f2 * f2)
    n12 = corr(M1, M2)
    with np.errstate(invalid='ignore', divide='ignore'):
        num = corr(F1, F2) - corr(F1, M2) * corr(M1, F2) / np.maximum(n12, 1e-9)
        d1 = corr(F1s, M2s) - corr(F1, M2) ** 2 / np.maximum(n12, 1e-9)
        d2 = corr(M1s, F2s) - corr(M1, F2) ** 2 / np.maximum(n12, 1e-9)
        ncc = num / np.sqrt(np.maximum(d1, 1e-12) * np.maximum(d2, 1e-12))
    ncc = np.nan_to_num(ncc, nan=0.0, posinf=0.0, neginf=0.0)
    ncc[n12 < min_overlap] = 0.0
    # shifts live at [0..sy] and [fy-sy..fy-1]
    rows = np.concatenate([np.arange(0, sy + 1), np.arange(fy - sy, fy)])
    cols = np.concatenate([np.arange(0, sx + 1), np.arange(fx - sx, fx)])
    sub = ncc[np.ix_(rows, cols)]
    k = int(np.argmax(sub))
    ri, ci = divmod(k, sub.shape[1])
    peak = float(sub[ri, ci])
    # parabolic refinement inside the cropped window
    def parab(a, b, c):
        den = (a - 2 * b + c)
        return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (a - c) / den, -1, 1))
    dsr = parab(sub[max(ri - 1, 0), ci], sub[ri, ci], sub[min(ri + 1, sub.shape[0] - 1), ci])
    dsc = parab(sub[ri, max(ci - 1, 0)], sub[ri, ci], sub[ri, min(ci + 1, sub.shape[1] - 1)])
    dr = float(rows[ri]); dc = float(cols[ci])
    if dr > fy / 2:
        dr -= fy
    if dc > fx / 2:
        dc -= fx
    ov = float(n12[int(rows[ri]), int(cols[ci])])
    return dr + dsr, dc + dsc, peak, ov


# ------------------------------------------------------- the silhouette ---- #
def silhouette_anchor(ctx, rect, dsm, ncol_m=0.25, span_m=4.0):
    """Pin one view of one rect to the rect's own outline.

    -> dict(du, dv, n_top, ncc_h, why) in rect metres, or None.
    """
    if ctx.sl is None:
        return None
    NC = len(ctx.sl)
    # columns: sample the rect's top edge every ncol_m, plus span_m of the
    # background either side, on the rect's own u axis
    n = int(max(6, round((rect.width + 2 * span_m) / ncol_m)))
    uu = -span_m + (np.arange(n) + 0.5) * ((rect.width + 2 * span_m) / n)
    inside = (uu >= 0) & (uu <= rect.width)
    Ptop = rect.points(uu, np.full(n, rect.height - 0.02), lift=0.02)
    lon, lat, rng = T.project(Ptop.astype('f4'), ctx.C, ctx.R)
    col = np.clip(((lon / (2 * math.pi) + 0.5) * NC).astype(np.int32), 0, NC - 1)
    # the predicted silhouette: the rect's top where the rect is there, the
    # lidar DSM's own skyline where it is not
    az = np.arctan2(Ptop[:, 1] - ctx.C[1], Ptop[:, 0] - ctx.C[0])
    ai = np.clip(((az - ctx.occ.az0) / ctx.occ.daz).astype(np.int32), 0,
                 ctx.occ.cmax.shape[0] - 1)
    sky_slope = ctx.occ.cmax[ai, -1]
    pred = np.where(inside, lat, np.arctan(sky_slope))
    # a column is usable only where nothing behind the rect rises above it
    rhor = np.hypot(Ptop[:, 0] - ctx.C[0], Ptop[:, 1] - ctx.C[1])
    ri = np.searchsorted(ctx.occ.radii, np.maximum(rhor + 1.5, 0.5)) - 1
    ri = np.clip(ri, 0, len(ctx.occ.radii) - 1)
    behind = ctx.occ.cmax[ai, -1] - ctx.occ.cmax[ai, ri]      # rise after the rect
    top_ok = inside & (behind < 0.02) & (rng < H.rect_rmax(rect) * 1.6) & ctx.sv[col]
    if top_ok.sum() < 6:
        return None
    meas = ctx.sl[col].astype('f8')
    # ---- vertical: the elevation gap on the rect's own columns ------------- #
    resid = meas[top_ok] - lat[top_ok]
    dlat = float(np.median(resid))
    d = float(np.median(rng[top_ok]))
    la = float(np.median(lat[top_ok]))
    dv = dlat * d / max(math.cos(la) ** 2, 1e-3)
    # ---- horizontal: 1D correlation of the two silhouette profiles -------- #
    du = 0.0; ncch = 0.0
    good = ctx.sv[col] & (np.abs(meas) < 1.5)
    if good.sum() >= n * 0.5:
        a = np.where(good, pred, np.nan)
        b = np.where(good, meas, np.nan)
        a = a - np.nanmean(a); b = b - np.nanmean(b)
        a = np.nan_to_num(a); b = np.nan_to_num(b)
        if np.std(a) > 1e-4 and np.std(b) > 1e-4:
            kmax = int(min(n // 3, round(1.5 / ncol_m)))
            best = (-2.0, 0)
            for k in range(-kmax, kmax + 1):
                aa = a[max(0, k):n + min(0, k)]
                bb = b[max(0, -k):n + min(0, -k)]
                g = good[max(0, k):n + min(0, k)] & good[max(0, -k):n + min(0, -k)]
                if g.sum() < 8:
                    continue
                x = aa[g]; y = bb[g]
                s = float(np.dot(x - x.mean(), y - y.mean()) /
                          max(np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean()), 1e-9))
                if s > best[0]:
                    best = (s, k)
            ncch, kb = best
            if ncch > 0.45 and kb != 0:
                # kb columns of the rect's own u axis, in metres
                du = kb * ((rect.width + 2 * span_m) / n)
    why = 'roofline'
    if abs(dv) > ANCHOR_CLAMP:
        dv = math.copysign(ANCHOR_CLAMP, dv); why = 'roofline (clamped)'
    if abs(du) > ANCHOR_CLAMP:
        du = math.copysign(ANCHOR_CLAMP, du)
    return dict(du=float(du), dv=float(dv), n_top=int(top_ok.sum()),
                ncc_h=float(ncch), dlat_deg=round(math.degrees(dlat), 3),
                range_m=round(d, 1), why=why)


def building_mask(pid, cls):
    """-> bool (A.MASK_H, A.MASK_W) equirect mask of Mapillary's own segmentation
    for class `cls` (stage 40's detections cache, s40_masks' vector-tile decode),
    or None when the panorama has no detections."""
    import gzip, base64, cv2, mapbox_vector_tile
    from align import alcommon as A
    import s40_masks as S40
    p = A.det_path(pid)
    if not os.path.exists(p):
        return None
    d = json.load(gzip.open(p))
    dets = d.get('data', d) if isinstance(d, dict) else d
    m = np.zeros((A.MASK_H, A.MASK_W), 'u1'); n = 0
    for det in dets:
        if (det.get('value') or '') != cls or not det.get('geometry'):
            continue
        try:
            dec = mapbox_vector_tile.decode(base64.decodebytes(det['geometry'].encode('utf-8')))
        except Exception:
            continue
        for lay in dec.values():
            for ft in lay.get('features', []):
                for r in S40.rings(ft.get('geometry', {})):
                    if len(r) < 3:
                        continue
                    px = np.empty((len(r), 2), 'i4')
                    px[:, 0] = np.clip(r[:, 0] / 4096.0 * A.MASK_W, 0, A.MASK_W - 1)
                    px[:, 1] = np.clip((1.0 - r[:, 1] / 4096.0) * A.MASK_H, 0, A.MASK_H - 1)
                    cv2.fillPoly(m, [px], 1); n += 1
    return (m > 0) if n else None


def quad_fit(ctx, rect, bmask):
    """Texture2LoD3's per-surface fit on the wall plane: slide the wall box over
    the panorama's building mask resampled on the padded plane.
    -> dict(du, dv, iou, edges) in rect metres (the shift of the PHOTOGRAPH, as
    sample_rect takes it), or None when the fit is not constrained."""
    import common as C
    from align import alcommon as A
    pad = float(C.FACADE_QUAD_PAD_M); cell = float(C.FACADE_QUAD_CELL_M)
    uu, vv = H.grid_axes(rect, cell, pad=pad)
    UU, VV = np.meshgrid(uu, vv)
    P = rect.points(UU.ravel(), VV.ravel())
    # where the ray through each plane point lands in the panorama; occlusion is
    # NOT applied (beyond the wall the ray is meant to see sky, ground or others)
    d = (P - ctx.C[None, :]).astype('f4')
    rng = np.linalg.norm(d, axis=1)
    cos = -((d / np.maximum(rng[:, None], 1e-6)) @ np.asarray(rect.n, 'f4'))
    ok = (cos > H.MIN_COS) & (rng > H.R_MIN) & (rng < H.rect_rmax(rect) + pad + 5.0)
    if ok.sum() < 200:
        return None
    lo, la, _ = T.project(P.astype('f4'), ctx.C, ctx.R)
    u = (lo / (2 * math.pi) + 0.5) * A.MASK_W - 0.5
    v = (0.25 - la / (2 * math.pi)) * A.MASK_W - 0.5
    B = A.mask_lookup(bmask, u, v, A.MASK_W) & ok
    tr = ctx.masked(lo, la) if ctx.mask is not None else np.zeros(len(P), bool)
    ok &= ~tr                                   # a car or a person is no evidence
    NB = ok & ~B
    ny, nx = len(vv), len(uu)
    B = B.reshape(ny, nx).astype('f8'); NB = NB.reshape(ny, nx).astype('f8')
    SB = np.pad(B.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    SN = np.pad(NB.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    def box(S, i0, i1, j0, j1):
        i0 = np.clip(i0, 0, ny); i1 = np.clip(i1, 0, ny); j0 = np.clip(j0, 0, nx); j1 = np.clip(j1, 0, nx)
        return S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]
    p0 = int(round(pad / cell)); bw = int(round(rect.width / cell)); bh = int(round(rect.height / cell))
    sr = int(round(float(C.FACADE_QUAD_SEARCH_M) / cell))
    sh = np.arange(-sr, sr + 1)
    DI, DJ = np.meshgrid(sh, sh, indexing='ij')   # rows = v, cols = u
    i0 = p0 + DI; j0 = p0 + DJ
    bin_ = box(SB, i0, i0 + bh, j0, j0 + bw); nin = box(SN, i0, i0 + bh, j0, j0 + bw)
    tb = B.sum()
    iou = bin_ / np.maximum(tb + nin, 1.0)       # |box & B| / |box | B| over judged cells
    k = np.unravel_index(int(np.argmax(iou)), iou.shape)
    best = float(iou[k]); di, dj = int(DI[k]), int(DJ[k])
    if best < float(C.FACADE_QUAD_IOU_MIN):
        return None
    s = max(2, int(round(0.5 / cell))); a0, b0 = p0 + di, p0 + dj
    def nonb(ii0, ii1, jj0, jj1):
        n = box(SB, ii0, ii1, jj0, jj1) + box(SN, ii0, ii1, jj0, jj1)
        return float(box(SN, ii0, ii1, jj0, jj1) / n) if n >= 10 else 0.0
    ed = dict(top=nonb(a0 + bh, a0 + bh + s, b0, b0 + bw), bottom=nonb(a0 - s, a0, b0, b0 + bw),
              left=nonb(a0, a0 + bh, b0 - s, b0), right=nonb(a0, a0 + bh, b0 + bw, b0 + bw + s))
    em = float(C.FACADE_QUAD_EDGE_MIN)
    if not ((ed['top'] >= em or ed['bottom'] >= em) and (ed['left'] >= em or ed['right'] >= em)):
        return None
    # the mask's wall sits (dj, di) cells from the model's: that is the photograph's shift
    return dict(du=dj * cell, dv=di * cell, iou=round(best, 3), edges={x: round(y, 2) for x, y in ed.items()})


def purge_chips(setname, sigs, pids, jobs, only=None):
    """Remove every chip that a fresh run would not have: a rect id the rect set
    lacks, a chip of another wall geometry (the id moved), a chip whose view
    index now names another panorama, or a view this run did not choose for the
    rect.  With `only`, rects outside it are left alone."""
    import shutil
    D = H.setdir(setname) + '/chips'
    if not os.path.isdir(D):
        return 0
    want = {}
    for k, rs in jobs.items():
        for r in rs:
            want.setdefault(int(r), set()).add(int(k))
    n = 0
    for dn in sorted(os.listdir(D)):
        if not (dn.startswith('r') and dn[1:].isdigit()):
            continue
        rid = int(dn[1:])
        if only is not None and rid not in only:
            continue
        dd = os.path.join(D, dn)
        if rid not in sigs:
            n += sum(1 for f in os.listdir(dd) if f.endswith('.npz'))
            shutil.rmtree(dd); continue
        for f in os.listdir(dd):
            if not (f.startswith('v') and f.endswith('.npz')) or '.tmp' in f:
                if '.tmp' in f:
                    os.remove(os.path.join(dd, f))
                continue
            k = int(f[1:5])
            keep = False
            if k in want.get(rid, ()) and k < len(pids):
                try:
                    m = json.loads(str(np.load(os.path.join(dd, f), allow_pickle=True)['meta']))
                    keep = chip_ok(m, sigs[rid], pids[k])
                except Exception:
                    keep = False
            if not keep:
                os.remove(os.path.join(dd, f)); n += 1
    return n


# ------------------------------------------------------------- the align --- #
def _load_chip(setname, rid, k):
    p = chip_path(setname, rid, k)
    d = np.load(p, allow_pickle=True)
    meta = json.loads(str(d['meta']))
    # a chip of another wall under this id (s33 renumbered), or of another
    # panorama under this view index, is not this rect's chip
    if not chip_ok(meta, _G['sigs'].get(int(rid)), _G['pids'][int(k)]):
        raise ValueError(f'stale chip r{rid:04d}/v{k:04d}')
    return dict(rgb=d['rgb'], valid=d['valid'],
                meta=meta, cos=d['cos'].astype('f4'))


def _align_rect(rid):
    setname = _G['set']
    rc = _G['rects'][rid]
    dd = f'{H.setdir(setname)}/chips/r{rid:04d}'
    if not os.path.isdir(dd):
        return None
    ks = sorted(int(f[1:5]) for f in os.listdir(dd)
                if f.startswith('v') and f.endswith('.npz'))
    ch = {}
    for k in ks:
        try:
            ch[k] = _load_chip(setname, rid, k)
        except Exception:
            continue
    if not ch:
        return None
    tx = ch[min(ch)]['meta']['tx']
    kbox = max(3, int(round(0.5 / tx)) | 1)
    prep = {}
    from scipy.ndimage import uniform_filter
    for k, c in ch.items():
        L = H.luma(c['rgb']).astype('f8')
        V = c['valid']
        num = uniform_filter(np.where(V, L, 0.0), kbox, mode='nearest')
        den = uniform_filter(V.astype('f8'), kbox, mode='nearest')
        hp = np.where(V, L - num / np.maximum(den, 1e-6), 0.0)
        prep[k] = (hp, V, int(V.sum()), float(c['cos'][V].mean()) if V.any() else 0.0)
    order = sorted(prep, key=lambda k: -prep[k][2])

    # ---- the absolute anchor, per view ------------------------------------ #
    anc = {}
    import common as C
    quad = {}
    QA = bool(getattr(C, 'FACADE_QUAD_ALIGN', False))
    for k in (order if QA else order[:12]):
        pid = _G['pids'][k]
        try:
            ctx = ViewCtx(pid, _G['poses'][pid], _G['dsm'], with_mask=QA)
            a = silhouette_anchor(ctx, rc, _G['dsm']) if k in order[:12] else None
            if QA:
                bm = building_mask(pid, C.FACADE_QUAD_CLASS)
                q = quad_fit(ctx, rc, bm) if bm is not None else None
                if q is not None:
                    quad[k] = q
        except Exception:
            a = None
        if a is not None:
            anc[k] = a

    # ---- relative alignment ----------------------------------------------- #
    ref = max(prep, key=lambda k: prep[k][2] * max(prep[k][3], 0.01))
    maxs = max(4, int(round(H.SEARCH_M / tx)))
    off = {ref: (0.0, 0.0)}
    info = {ref: dict(ncc=1.0, overlap=int(prep[ref][2]), via='reference',
                      shift_m=[0.0, 0.0])}
    disag = []
    todo = [k for k in prep if k != ref]
    for rnd in (1, 2):
        nxt = []
        for k in todo:
            parents = ([ref] if rnd == 1 else
                       sorted([p for p in off if p != k],
                              key=lambda p: -prep[p][2])[:3])
            best = None
            for p in parents:
                dr, dc, ncc, ov = masked_ncc(prep[p][0], prep[p][1],
                                             prep[k][0], prep[k][1], maxs)
                if ncc >= H.NCC_MIN and ov >= MIN_OVERLAP and (
                        best is None or ncc > best[2]):
                    best = (dr, dc, ncc, ov, p)
            if best is None:
                nxt.append(k)
                continue
            dr, dc, ncc, ov, p = best
            off[k] = (off[p][0] + dc * tx, off[p][1] + dr * tx)
            info[k] = dict(ncc=round(float(ncc), 3), overlap=int(ov),
                           via=('reference' if p == ref else f'view {p}'),
                           shift_m=[round(dc * tx, 3), round(dr * tx, 3)])
            # how far apart the two photographs were before and after the shift,
            # on the texels they share: the number this stage exists to reduce
            d0 = _disagree(prep[p], prep[k], 0, 0)
            d1 = _disagree(prep[p], prep[k], int(round(dr)), int(round(dc)))
            if d0 is not None and d1 is not None:
                disag.append((d0, d1, math.hypot(dr, dc) * tx))
        todo = nxt
        if not todo:
            break

    # ---- the whole-wall correction ---------------------------------------- #
    # A low rect has no measurable silhouette: the sky boundary over a 0.5 m
    # pier skirt is the far shore, not the skirt's own top edge, so a silhouette
    # measurement there saturates at the trust cap.  Anything under
    # 1.5 m tall is aligned relatively only, and the cap on everything else
    # scales with the rect's own height.
    if rc.height < 1.5:
        anc = {}
    res_dv = [anc[k]['dv'] - off[k][1] for k in anc if k in off]
    res_du = [anc[k]['du'] - off[k][0] for k in anc
              if k in off and anc[k]['ncc_h'] > 0.45 and anc[k]['du'] != 0.0]
    gu = gv = 0.0
    vscale = 1.0
    spread = 0.0
    why = 'relative only: no view of this wall has its outline against the sky'
    if res_dv:
        gv = float(np.median(res_dv))
        spread = float(np.std(res_dv)) if len(res_dv) > 1 else 0.0
        firm = len(res_dv) >= 3 and spread <= ANCHOR_SPREAD
        # a gap that is a big share of the wall's own height is not a wrong
        # storey count, it is a mast, a parapet or an antenna standing over the
        # roofline: the sky boundary there is not this wall's top edge
        big = abs(gv) > min(4.0, 0.30 * rc.height)
        cap = min(TRUST_ANCHOR, 0.25 * rc.height)
        if abs(gv) > cap and (not firm or big):
            gv = math.copysign(cap, gv)
            why = (f'roofline on {len(res_dv)} view(s) held at the {cap:.2f} m '
                   f'trust cap, spread {spread:.2f} m' +
                   (', gap too big a share of the wall height (mast or parapet)'
                    if big else ''))
        elif abs(gv) > 0.35 and firm:
            # the silhouette says the real building is this much taller (or
            # shorter) than the OSM extrusion.  Sliding the whole texture by
            # 2 m would put the photograph's shopfront underground; stretching
            # it from the base keeps both ends of the wall where they belong.
            vscale = float(np.clip((rc.height + gv) / max(rc.height, 0.5), 0.6, 1.6))
            why = (f'roofline on {len(res_dv)} view(s), spread {spread:.2f} m: '
                   f'stretched x{vscale:.3f} from the base, not slid')
            gv = 0.0
        else:
            why = f'roofline on {len(res_dv)} view(s), spread {spread:.2f} m'
        if res_du:
            gu = float(np.median(res_du))
            why += f', corner on {len(res_du)}'
        else:
            why += ', no corner'

    # ---- views the correlation could not place --------------------------- #
    # Texture2LoD3 first: an outline fit, tied to the correlated set's frame by
    # the correlated views' own fits (F = median of their fit minus their shift)
    frame = [(quad[k]['du'] - off[k][0] - gu, quad[k]['dv'] - off[k][1] - gv)
             for k in off if k in quad and k not in todo]
    F = (float(np.median([f[0] for f in frame])), float(np.median([f[1] for f in frame]))) if frame else None
    for k in list(todo):
        if k not in quad:
            continue
        fu, fv = F if F is not None else (0.0, 0.0)
        off[k] = (quad[k]['du'] - fu - gu, quad[k]['dv'] - fv - gv)
        info[k] = dict(ncc=0.0, overlap=0, via='quad' if F is not None else 'quad_abs',
                       shift_m=[round(off[k][0], 3), round(off[k][1], 3)],
                       quad=quad[k], frame_views=len(frame))
        todo.remove(k)
    for k in todo:
        if k in anc:
            off[k] = (anc[k]['du'] - gu, anc[k]['dv'] - gv)
            info[k] = dict(ncc=0.0, overlap=0, via='silhouette',
                           shift_m=[round(off[k][0], 3), round(off[k][1], 3)])
        else:
            off[k] = (0.0, 0.0)
            info[k] = dict(ncc=0.0, overlap=0, via='unaligned',
                           shift_m=[0.0, 0.0])
    for k in info:
        if k in quad:
            info[k].setdefault('quad', quad[k])
    d0 = [d[0] for d in disag]; d1 = [d[1] for d in disag]
    out = dict(rect=int(rid), tx=tx, ref=int(ref), views=len(off),
               aligned=int(sum(1 for k in info
                               if info[k]['via'] != 'unaligned')),
               by_correlation=int(sum(1 for k in info if info[k]['ncc'] > 0)),
               by_silhouette=int(sum(1 for k in info
                                     if info[k]['via'] == 'silhouette')),
               unaligned=int(sum(1 for k in info
                                 if info[k]['via'] == 'unaligned')),
               anchor=dict(du=round(gu, 3), dv=round(gv, 3),
                           vscale=round(vscale, 4), top_gap_m=round(
                               float(np.median(res_dv)) if res_dv else 0.0, 3),
                           why=why, n=len(res_dv), spread_dv_m=round(spread, 3)),
               disagreement_sRGB=dict(
                   before=round(float(np.median(d0)), 2),
                   after=round(float(np.median(d1)), 2), pairs=len(d0))
               if d0 else None,
               shift_m={int(k): [round(off[k][0] + gu, 3),
                                 round(off[k][1] + gv, 3)] for k in off},
               per_view={int(k): info[k] for k in info},
               quad_fits=len(quad), by_quad=int(sum(1 for k in info if info[k]['via'].startswith('quad'))),
               rel_shift_norm_m=round(float(np.median(
                   [math.hypot(*off[k]) for k in off])), 3))
    return out


def _disagree(pa, pb, dr, dc):
    """Median |dL| between two high-passed chips at an integer shift, over the
    texels both of them see.  sRGB levels."""
    A, MA = pa[0], pa[1]
    B, MB = pb[0], pb[1]
    ny, nx = A.shape
    a0 = max(0, -dr); a1 = min(ny, ny - dr)
    b0 = max(0, dr); b1 = min(ny, ny + dr)
    c0 = max(0, -dc); c1 = min(nx, nx - dc)
    e0 = max(0, dc); e1 = min(nx, nx + dc)
    if a1 - a0 < 8 or c1 - c0 < 8:
        return None
    Aa = A[a0:a1, c0:c1]; Ma = MA[a0:a1, c0:c1]
    Bb = B[b0:b1, e0:e1]; Mb = MB[b0:b1, e0:e1]
    m = Ma & Mb
    if m.sum() < MIN_OVERLAP:
        return None
    return float(np.median(np.abs(Aa[m] - Bb[m])))


def run_align(setname, workers=12, only=None):
    t0 = time.time()
    D = H.setdir(setname)
    d = np.load(D + '/work/cover.npz', allow_pickle=True)
    pids = [str(x) for x in d['pids']]
    poses, _ = T.load_poses()
    sel = {p: poses[p] for p in pids if p in poses}
    rects, _ = H.load_rects(setname)
    sigs = rect_sigs(rects)
    rids = sorted(int(x[1:]) for x in os.listdir(D + '/chips')
                  if x.startswith('r') and int(x[1:]) in sigs)
    if only is not None:
        rids = [r for r in rids if r in only]
    H.log(f"align[{setname}]: {len(rids)} rects")
    import multiprocessing as mp
    out = {}
    with mp.Pool(workers, initializer=_init,
                 initargs=(setname, sel, pids, {}, sigs)) as pool:
        for n, r in enumerate(pool.imap_unordered(_align_rect, rids, chunksize=1)):
            if r is not None:
                out[r['rect']] = r
            if (n + 1) % 20 == 0:
                H.log(f"  {n+1}/{len(rids)} rects, {time.time()-t0:.0f}s")
    nv = sum(r['views'] for r in out.values())
    na = sum(r['aligned'] for r in out.values())
    anc = [r for r in out.values() if r['anchor']['n'] > 0]
    allsh = [math.hypot(*v) for r in out.values() for v in r['shift_m'].values()]
    rep = dict(set=setname, rects=len(out), views=int(nv), aligned_views=int(na),
               aligned_pct=round(100 * na / max(nv, 1), 1),
               rects_with_silhouette_anchor=len(anc),
               anchor_du_abs_median_m=round(float(np.median(
                   [abs(r['anchor']['du']) for r in anc])), 3) if anc else None,
               anchor_dv_abs_median_m=round(float(np.median(
                   [abs(r['anchor']['dv']) for r in anc])), 3) if anc else None,
               rects_stretched=int(sum(1 for r in out.values()
                                       if r['anchor']['vscale'] != 1.0)),
               vscale_range=[round(float(min([r['anchor']['vscale'] for r in out.values()])), 3),
                             round(float(max([r['anchor']['vscale'] for r in out.values()])), 3)]
               if out else None,
               top_gap_abs_median_m=round(float(np.median(
                   [abs(r['anchor']['top_gap_m']) for r in anc])), 3) if anc else None,
               disagreement_sRGB_before_median=round(float(np.median(
                   [r['disagreement_sRGB']['before'] for r in out.values()
                    if r.get('disagreement_sRGB')])), 2) if any(
                        r.get('disagreement_sRGB') for r in out.values()) else None,
               disagreement_sRGB_after_median=round(float(np.median(
                   [r['disagreement_sRGB']['after'] for r in out.values()
                    if r.get('disagreement_sRGB')])), 2) if any(
                        r.get('disagreement_sRGB') for r in out.values()) else None,
               anchor_spread_dv_median_m=round(float(np.median(
                   [r['anchor']['spread_dv_m'] for r in anc])), 3) if anc else None,
               applied_shift_m={f'p{p}': round(float(np.percentile(allsh, p)), 3)
                                for p in (50, 75, 90, 100)} if allsh else {},
               search_radius_m=H.SEARCH_M, ncc_min=H.NCC_MIN,
               seconds=round(time.time() - t0), per_rect=out)
    json.dump(rep, open(D + '/work/align.json', 'w'), indent=1)
    H.mark(setname, 'align', {k: v for k, v in rep.items() if k != 'per_rect'})
    H.log(f"align[{setname}]: {na}/{nv} views aligned, "
          f"{len(anc)} rects anchored, {rep['seconds']}s")
    return rep


# ------------------------------------------- per-patch refinement ---------- #
# One translation per view is not enough on a 25 m wall.  Two views that
# correlate at NCC 0.594 over a whole wall can still only reach 0.17-0.45 in the
# 0.3 m strip round a patch boundary, and an artificial 0.5 m error scores
# 0.17-0.70 on the same strips -- i.e. the residual after a global fit is of the
# order of half a metre locally, enough to double a sign.  So each patch of each
# view gets its own small offset, searched around the view's global placement
# against the reference view.
REF_SEARCH_M = 0.40
REF_NCC_MIN = 0.45
REF_MIN_OVERLAP = 300


def _refine_rect(rid):
    setname = _G['set']
    rc = _G['rects'][rid]
    al = _G['align'][str(rid)]
    ref = int(al['ref'])
    pv = {int(k): v for k, v in al['per_view'].items()}
    ok_views = [k for k, v in pv.items()
                if v['via'] == 'reference' or v['via'].startswith('view')]
    if len(ok_views) < 2:
        return []
    off = {int(k): tuple(v) for k, v in al['shift_m'].items()}
    npu, npv = H.patch_shape(rc)
    ch = {}
    for k in ok_views:
        try:
            ch[k] = _load_chip(setname, rid, k)
        except Exception:
            pass
    if ref not in ch:
        return []
    tx = ch[ref]['meta']['tx']
    pad = ch[ref]['meta']['pad']
    from scipy.ndimage import uniform_filter
    kbox = max(3, int(round(0.5 / tx)) | 1)
    prep = {}
    for k, c in ch.items():
        L = H.luma(c['rgb']).astype('f8'); V = c['valid']
        num = uniform_filter(np.where(V, L, 0.0), kbox, mode='nearest')
        den = uniform_filter(V.astype('f8'), kbox, mode='nearest')
        prep[k] = (np.where(V, L - num / np.maximum(den, 1e-6), 0.0), V)
    pw = rc.width / npu; ph = rc.height / npv
    ms = max(3, int(round(REF_SEARCH_M / tx)))
    out = []
    for k in ch:
        if k == ref:
            for p in range(npu * npv):
                out.append((rid, k, p, 0.0, 0.0, 1, 1.0))
            continue
        dku = (off[k][0] - off[ref][0]) / tx
        dkv = (off[k][1] - off[ref][1]) / tx
        for p in range(npu * npv):
            pi, pj = divmod(p, npu)
            # the reference window, in chip texels, padded so the search has room
            u0 = (pj * pw - 0.5 + pad) / tx; u1 = ((pj + 1) * pw + 0.5 + pad) / tx
            v0 = (pi * ph - 0.5 + pad) / tx; v1 = ((pi + 1) * ph + 0.5 + pad) / tx
            j0 = int(max(0, u0)); j1 = int(min(prep[ref][0].shape[1], u1))
            i0 = int(max(0, v0)); i1 = int(min(prep[ref][0].shape[0], v1))
            if j1 - j0 < 12 or i1 - i0 < 12:
                continue
            A = prep[ref][0][i0:i1, j0:j1]; MA = prep[ref][1][i0:i1, j0:j1]
            # the same patch in the other view, already placed by its global shift
            jb0 = int(round(j0 + dku)); ib0 = int(round(i0 + dkv))
            if (jb0 < 0 or ib0 < 0 or jb0 + (j1 - j0) > prep[k][0].shape[1] or
                    ib0 + (i1 - i0) > prep[k][0].shape[0]):
                continue
            B = prep[k][0][ib0:ib0 + (i1 - i0), jb0:jb0 + (j1 - j0)]
            MB = prep[k][1][ib0:ib0 + (i1 - i0), jb0:jb0 + (j1 - j0)]
            if MA.sum() < REF_MIN_OVERLAP or MB.sum() < REF_MIN_OVERLAP:
                continue
            dr, dc, ncc, ov = masked_ncc(A, MA, B, MB, ms,
                                         min_overlap=REF_MIN_OVERLAP)
            good = int(ncc >= REF_NCC_MIN and ov >= REF_MIN_OVERLAP and
                       abs(dr) <= ms and abs(dc) <= ms)
            out.append((rid, k, p, float(dc * tx) * good, float(dr * tx) * good,
                        good, float(ncc)))
    return out


def run_refine(setname, workers=12, only=None):
    """Per (rect, view, patch) offset against the rect's reference view."""
    t0 = time.time()
    D = H.setdir(setname)
    d = np.load(D + '/work/cover.npz', allow_pickle=True)
    pids = [str(x) for x in d['pids']]
    poses, _ = T.load_poses()
    sel = {p: poses[p] for p in pids if p in poses}
    al = json.load(open(D + '/work/align.json'))['per_rect']
    rids = sorted(int(k) for k in al)
    if only is not None:
        rids = [r for r in rids if r in only]
    H.log(f"refine[{setname}]: {len(rids)} rects")
    import multiprocessing as mp
    rows = []

    sigs = rect_sigs(H.load_rects(setname)[0])

    def _ini(s, poses_, pids_, align_, sigs_):
        _init(s, poses_, pids_, {}, sigs_)
        _G['align'] = align_
    with mp.Pool(workers, initializer=_ini,
                 initargs=(setname, sel, pids, al, sigs)) as pool:
        for n, r in enumerate(pool.imap_unordered(_refine_rect, rids, chunksize=1)):
            rows.extend(r)
            if (n + 1) % 20 == 0:
                H.log(f"  {n+1}/{len(rids)} rects, {time.time()-t0:.0f}s")
    if not rows:
        return dict(rows=0)
    A = np.array(rows, dtype='f8')
    np.savez_compressed(D + '/work/refine.npz', rect=A[:, 0].astype('i4'),
                        view=A[:, 1].astype('i4'), patch=A[:, 2].astype('i4'),
                        du=A[:, 3].astype('f4'), dv=A[:, 4].astype('f4'),
                        ok=A[:, 5].astype('u1'), ncc=A[:, 6].astype('f4'))
    nz = A[A[:, 5] > 0]
    rep = dict(set=setname, rows=len(A), accepted=int(A[:, 5].sum()),
               accepted_pct=round(100 * float(A[:, 5].mean()), 1),
               search_m=REF_SEARCH_M, ncc_min=REF_NCC_MIN,
               offset_m={f'p{p}': round(float(np.percentile(
                   np.hypot(nz[:, 3], nz[:, 4]), p)), 3) for p in (50, 75, 90, 100)}
               if len(nz) else {},
               ncc={f'p{p}': round(float(np.percentile(A[:, 6], p)), 3)
                    for p in (10, 50, 90)},
               seconds=round(time.time() - t0))
    json.dump(rep, open(D + '/work/refine.json', 'w'), indent=1)
    H.mark(setname, 'refine', rep)
    H.log(f"refine[{setname}]: {rep['accepted']}/{rep['rows']} patch offsets, "
          f"{rep['seconds']}s")
    return rep
