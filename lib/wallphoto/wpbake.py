"""Stage 60 steps 3-5 -- ownership, the bake, the levelling and the atlas.

own    every 1.5 m patch of every rect is given to ONE panorama: the nearest
       unoccluded front-facing view that is not vetoed by a transient mask,
       with MVS-Texturing's Potts smoothing so neighbouring patches share a
       view wherever that costs little.
bake   each owner paints its own patches at the rect's fine texel size, from
       the panorama's native equirect width, through the 2D correction stage
       60 step 2 measured for it.
level  the owners are brought to a common exposure by a least-squares solve
       over their shared texels, then feathered across the patch boundaries.
atlas  the tiles are packed into 4096 px pages with the same wallmap.npz
       interface stage 22 already reads, so no compositing code changes.
"""
import os, sys, json, time, math
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import texlib as T
from wallphoto import wpcommon as H
from wallphoto.wpsample import ViewCtx, sample_rect
from wallphoto.wpalign import masked_ncc, rect_sigs, rec_matches
# stage_knobs.FACADE_PHOTO_QUALITY.fallback, only when an area is set.
# None (no area set, or the key absent) leaves own and bake on their default behaviour.
_FB = None
if H._AP:
    import facade_quality as _FQ
    _FB = _FQ.fallback()


def _fb_sig():
    return json.dumps(_FB, sort_keys=True) if _FB is not None else None

OWN_MIN_FRAC = 0.50      # a view must see this much of a patch to own it
OWN_LOW_FRAC = 0.25      # ... or this much if nothing else sees it at all
BETA = 0.35              # Potts: bonus per neighbour that agrees, x best score
ICM_ROUNDS = 4
# Only views placed by CORRELATION against the wall's reference view may own a
# patch.  A view placed by its own silhouette, or not placed at all, sits in a
# different frame from its neighbours: two silhouette-placed views of one wall
# can differ by more than a metre along u, which draws a painted sign twice.  A
# material fallback is a worse texture than a photograph, but it cannot draw
# the building twice.
ALIGN_W = dict(reference=1.0, correlation=1.0, silhouette=0.0, unaligned=0.0)
import common as C
# outline-fitted views (wpalign.quad_fit, Texture2LoD3): weights from the area config
ALIGN_W.update(quad=float(getattr(C, 'FACADE_QUAD_W_FRAME', 0.0)),
               quad_abs=float(getattr(C, 'FACADE_QUAD_W_ABS', 0.0)))
PHOTO_MIN_AREA = 0.10
BAND_NCC_REAL = 0.30     # below this the strip has no structure to double
MIN_ISLAND_M2 = 0.40     # a smaller patch of photograph is an occluder's edge



# ------------------------------------------------- patch agreement -------- #
# Two neighbouring patches owned by two panoramas can each be plausible on their
# own and still disagree about WHERE the wall's content is: the relative NCC
# aligns a whole view by one translation, and a view taken from 4 m and a view
# taken from 20 m do not share one translation over a big wall: a painted sign
# can render twice, at two scales, in two adjacent patches.  This pass measures
# every boundary between two owners on the alignment chips and refuses the ones
# that do not agree.
class _Chip:
    """One view's chip, addressed in model (u, v) metres through that view's
    measured shift and the rect's vertical stretch."""

    def __init__(self, path, shift, vscale):
        d = np.load(path, allow_pickle=True)
        self.L = H.luma(d['rgb']).astype('f4')
        self.V = d['valid']
        m = json.loads(str(d['meta']))
        self.tx = m['tx']; self.pad = m['pad']
        self.shift = shift; self.vscale = vscale

    def at(self, uu, vv, extra=(0.0, 0.0)):
        j = np.rint((uu + self.shift[0] + extra[0] + self.pad) / self.tx - 0.5).astype(np.int32)
        i = np.rint((vv * self.vscale + self.shift[1] + extra[1] + self.pad) / self.tx - 0.5).astype(np.int32)
        ok = ((i >= 0) & (i < self.L.shape[0]) & (j >= 0) & (j < self.L.shape[1]))
        ii = np.clip(i, 0, self.L.shape[0] - 1); jj = np.clip(j, 0, self.L.shape[1] - 1)
        return self.L[ii, jj], ok & self.V[ii, jj]


def _band_agreement(ca, cb, rc, p, q, npu, npv, band_m=0.50, step_m=0.05,
                    ea=(0.0, 0.0), eb=(0.0, 0.0), search_m=0.40):
    """How far apart two owners put the wall's content across their shared edge.

    Samples both views on a 2 x band_m strip straddling the boundary and runs
    the same masked NCC the alignment uses.  -> (best_ncc, offset_m, n,
    contrast) or None when the strip is too small, too empty or too flat for a
    correlation to mean anything.  A big offset with a real peak is the doubled
    image the eye sees; a low peak everywhere is a blank wall, where nothing can
    double.
    """
    pw = rc.width / npu; ph = rc.height / npv
    pi, pj = divmod(p, npu); qi, qj = divmod(q, npu)
    if pi == qi:
        u0 = max(pj, qj) * pw
        uu = np.arange(u0 - band_m, u0 + band_m, step_m)
        vv = np.arange(pi * ph, (pi + 1) * ph, step_m)
    else:
        v0 = max(pi, qi) * ph
        vv = np.arange(v0 - band_m, v0 + band_m, step_m)
        uu = np.arange(pj * pw, (pj + 1) * pw, step_m)
    uu = uu[(uu >= -0.05) & (uu <= rc.width + 0.05)]
    vv = vv[(vv >= -0.05) & (vv <= rc.height + 0.05)]
    if len(uu) < 10 or len(vv) < 10:
        return None
    UU, VV = np.meshgrid(uu, vv)
    U = UU.ravel(); V = VV.ravel()
    La, oa = ca.at(U, V, ea); Lb, ob = cb.at(U, V, eb)
    sh = (len(vv), len(uu))
    La = La.reshape(sh).astype('f8'); Lb = Lb.reshape(sh).astype('f8')
    MA = oa.reshape(sh); MB = ob.reshape(sh)
    if MA.sum() < 150 or MB.sum() < 150:
        return None
    contrast = float(min(La[MA].std(), Lb[MB].std()))
    if contrast < 4.0:
        return None
    from scipy.ndimage import uniform_filter
    k = 7
    def hp(L, M):
        num = uniform_filter(np.where(M, L, 0.0), k, mode='nearest')
        den = uniform_filter(M.astype('f8'), k, mode='nearest')
        return np.where(M, L - num / np.maximum(den, 1e-6), 0.0)
    ms = max(2, int(round(search_m / step_m)))
    ms = min(ms, min(sh) // 2 - 1, 12)
    if ms < 2:
        return None
    dr, dc, ncc, ov = masked_ncc(hp(La, MA), MA, hp(Lb, MB), MB, ms,
                                 min_overlap=150)
    return float(ncc), float(np.hypot(dr, dc) * step_m), int(ov), contrast


# ------------------------------------------------------------- ownership --- #
def run_own(setname, beta=BETA, ncc_cut=0.35, only=None, ban=None):
    """only: re-solve ownership for these rects only, merging into own.json.
    ban:  {rid: {view, ...}} -- views this rect may not use.  The exposure
          retry (wpatlas.run_relevel) bans the view that lost a wall to the
          level gate and re-solves, so the wall's second-best panorama gets
          its turn instead of the wall falling back to material."""
    t0 = time.time()
    only = set(int(x) for x in only) if only else None
    ban = {int(k): set(int(x) for x in v) for k, v in (ban or {}).items()}
    D = H.setdir(setname)
    rects, _ = H.load_rects(setname)
    byid = {r.id: r for r in rects}
    d = np.load(D + '/work/cover.npz', allow_pickle=True)
    rect = d['rect']; view = d['view']; patch = d['patch']
    nval = d['nval']; ntot = d['ntot']; cosm = d['cos']; rngm = d['rng']
    al = json.load(open(D + '/work/align.json'))['per_rect']
    REF = {}
    rp = D + '/work/refine.npz'
    if os.path.exists(rp):
        dd = np.load(rp)
        for r_, v_, p_, du_, dv_, ok_ in zip(dd['rect'], dd['view'], dd['patch'],
                                             dd['du'], dd['dv'], dd['ok']):
            if ok_:
                REF[(int(r_), int(v_), int(p_))] = (float(du_), float(dv_))
    out = {}
    nb_before = nb_after = nb_final = 0
    nccs = []
    stats_edges = []
    for rid_s, a in al.items():
        rid = int(rid_s)
        if only is not None and rid not in only:
            continue
        rc = byid[rid]
        npu, npv = H.patch_shape(rc)
        m = rect == rid
        if not m.any():
            continue
        vw = {int(k): v for k, v in a['per_view'].items()}
        vi = view[m]; pa = patch[m]
        frac = nval[m] / np.maximum(ntot[m], 1)
        keep = np.array([v in vw for v in vi])
        if not keep.any():
            continue
        vi = vi[keep]; pa = pa[keep]; frac = frac[keep]
        cs = cosm[m][keep]; rg = rngm[m][keep]
        via = np.array([vw[v]['via'] for v in vi])
        aw = np.array([ALIGN_W['correlation'] if x.startswith(('reference', 'view'))
                       else ALIGN_W.get(x, 0.0) for x in via])
        q = (np.clip(cs, 0, 1) ** 0.7) / (1.0 + (rg / 8.0) ** 2) * aw
        score = np.where(frac >= OWN_MIN_FRAC, frac, 0.0) * q
        low = np.where(frac >= OWN_LOW_FRAC, frac, 0.0) * q * 0.5
        score = np.maximum(score, low * (frac >= OWN_LOW_FRAC))
        bl = ban.get(rid)
        if bl:
            # a banned view scores nothing, so every patch it held falls to the
            # next view that can see it, or to material if there is none
            score = np.where(np.isin(vi, np.array(sorted(bl), 'i8')), 0.0, score)
        npatch = npu * npv
        owner = np.full(npatch, -1, 'i4')
        best = np.zeros(npatch, 'f8')
        srt = np.argsort(pa, kind='stable')
        pa_s = pa[srt]; vi_s = vi[srt]; sc_s = score[srt]
        off = np.searchsorted(pa_s, np.arange(npatch + 1))
        for p in range(npatch):
            s0, s1 = off[p], off[p + 1]
            if s1 <= s0:
                continue
            k = int(np.argmax(sc_s[s0:s1]))
            if sc_s[s0 + k] <= 0:
                continue
            owner[p] = vi_s[s0 + k]; best[p] = sc_s[s0 + k]
        # ---- Potts smoothing ---------------------------------------------- #
        def nbrs(p):
            i, j = divmod(p, npu)
            for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ii, jj = i + di, j + dj
                if 0 <= ii < npv and 0 <= jj < npu:
                    yield ii * npu + jj

        def boundaries(lab):
            c = 0
            for p in range(npatch):
                for q2 in nbrs(p):
                    if q2 > p and lab[p] != lab[q2] and lab[p] >= 0 and lab[q2] >= 0:
                        c += 1
            return c
        nb_before += boundaries(owner)
        for _ in range(ICM_ROUNDS):
            changed = 0
            for p in range(npatch):
                s0, s1 = off[p], off[p + 1]
                if s1 - s0 < 2 or owner[p] < 0:
                    continue
                cand = vi_s[s0:s1]; sc = sc_s[s0:s1].copy()
                nl = [owner[q2] for q2 in nbrs(p) if owner[q2] >= 0]
                if nl:
                    sc = sc + beta * best[p] * np.array(
                        [sum(1 for x in nl if x == c) for c in cand])
                v = cand[int(np.argmax(sc))]
                if v != owner[p] and sc_s[s0:s1][cand == v].max() > 0:
                    owner[p] = v; changed += 1
            if changed == 0:
                break
        nb_after += boundaries(owner)

        # ---- refuse boundaries where the two owners disagree -------------- #
        S = {}
        for p in range(npatch):
            for t in range(off[p], off[p + 1]):
                S[(p, int(vi_s[t]))] = float(sc_s[t])
        dropped = set()     # the patches the agreement gate sent to material
        shifts = {int(k): tuple(v) for k, v in a['shift_m'].items()}
        vscale = a['anchor'].get('vscale', 1.0)
        chips = {}

        def chip(v):
            if v not in chips:
                p = f'{D}/chips/r{rid:04d}/v{int(v):04d}.npz'
                try:
                    chips[v] = _Chip(p, shifts.get(int(v), (0.0, 0.0)), vscale)
                except Exception:
                    chips[v] = None
            return chips[v]

        def edges():
            out = []
            for p in range(npatch):
                for q2 in nbrs(p):
                    if q2 > p and owner[p] >= 0 and owner[q2] >= 0 and owner[p] != owner[q2]:
                        out.append((p, q2))
            return out

        def measure():
            res = []
            for p, q2 in edges():
                ca = chip(owner[p]); cb = chip(owner[q2])
                if ca is None or cb is None:
                    continue
                r = _band_agreement(ca, cb, rc, p, q2, npu, npv,
                                    ea=REF.get((rid, int(owner[p]), p), (0.0, 0.0)),
                                    eb=REF.get((rid, int(owner[q2]), q2), (0.0, 0.0)))
                if r is not None:
                    res.append((p, q2, r[0], r[1], r[2]))   # ncc, offset_m, n
            return res
        meas0 = measure()
        nccs.extend(m[3] for m in meas0)        # the offset in metres
        nfix = ndrop = 0
        for _ in range(3):
            bad = [m for m in meas0 if m[2] >= BAND_NCC_REAL and m[3] >= ncc_cut]
            if not bad:
                break
            for p, q2, nc, offm, ov2 in bad:
                aowner, bowner = owner[p], owner[q2]
                if aowner == bowner or aowner < 0 or bowner < 0:
                    continue
                ca = S.get((p, aowner), 0.0); cb = S.get((p, bowner), 0.0)
                da = S.get((q2, bowner), 0.0); db = S.get((q2, aowner), 0.0)
                optA = (ca - cb, p, bowner) if cb > 0 else None
                optB = (da - db, q2, aowner) if db > 0 else None
                opts = [o for o in (optA, optB) if o is not None]
                if opts:
                    cost, pp, vv = min(opts, key=lambda o: o[0])
                    owner[pp] = vv; nfix += 1
                else:
                    pp = p if S.get((p, aowner), 0) <= S.get((q2, bowner), 0) else q2
                    owner[pp] = -1; ndrop += 1; dropped.add(int(pp))
            meas0 = measure()
        # anything still doubled after three rounds loses its weaker patch: a
        # flat material patch is a worse texture than a photograph and a better
        # one than the building drawn twice
        for p, q2, nc, offm, ov2 in [m for m in meas0
                                     if m[2] >= BAND_NCC_REAL and m[3] >= ncc_cut]:
            if owner[p] < 0 or owner[q2] < 0:
                continue
            pp = p if S.get((p, owner[p]), 0) <= S.get((q2, owner[q2]), 0) else q2
            owner[pp] = -1; ndrop += 1; dropped.add(int(pp))
        meas0 = measure()
        bad_after = sum(1 for m in meas0
                        if m[2] >= BAND_NCC_REAL and m[3] >= ncc_cut)
        nb_final += sum(1 for _ in edges())
        stats_edges.append((len(meas0) + nfix + ndrop, bad_after, nfix, ndrop))

        own_frac = float((owner >= 0).mean())
        out[rid] = dict(npu=npu, npv=npv, owner=[int(x) for x in owner],
                        boundary_offset_max_m=round(float(max(
                            [m[3] for m in meas0])), 3) if meas0 else None,
                        owned_patches=int((owner >= 0).sum()),
                        owners=int(len(set(int(x) for x in owner if x >= 0))),
                        owned_frac=round(own_frac, 4))
        if _FB is not None:
            out[rid]['dropped'] = sorted(p for p in dropped if owner[p] < 0)
            out[rid]['banned'] = sorted(int(x) for x in (bl or ()))
    if only is not None:
        # a partial re-solve keeps every rect it did not touch
        try:
            prev = json.load(open(D + '/work/own.json'))
            merged = dict(prev['per_rect']); merged.update({str(k): v for k, v in out.items()})
            prev['per_rect'] = merged
            prev['reowned_rects'] = sorted(out)
            prev['banned_views'] = {str(k): sorted(v) for k, v in ban.items() if v}
            json.dump(prev, open(D + '/work/own.json', 'w'), indent=1)
            H.log(f'own[{setname}]: re-solved {len(out)} rect(s), merged')
            return prev
        except (OSError, KeyError, ValueError):
            pass
    tot = sum(v['owned_patches'] for v in out.values())
    nb_bad0 = sum(e[1] for e in stats_edges)
    rep = dict(set=setname, rects=len(out), owned_patches=int(tot),
               patch_boundaries_between_views=dict(before_icm=int(nb_before),
                                                   after_icm=int(nb_after),
                                                   after_agreement=int(nb_final)),
               boundary_agreement=dict(
                   offset_cut_m=ncc_cut, measured_boundaries=len(nccs),
                   offset_percentiles_m={f'p{p}': round(float(np.percentile(nccs, p)), 3)
                                         for p in (50, 75, 90, 95, 100)} if nccs else {},
                   below_cut_before=int(sum(1 for x in nccs if x >= ncc_cut)),
                   below_cut_after=int(nb_bad0),
                   patches_reassigned=int(sum(e[2] for e in stats_edges)),
                   patches_dropped_to_material=int(sum(e[3] for e in stats_edges)),
                   why="the distance between where the two owners put the wall's "
                       "content in the 1 m strip round their shared edge, by masked "
                       "NCC on the 4 cm alignment chips.  Counted only where the peak "
                       f"correlation is over {BAND_NCC_REAL} and the quieter view's "
                       "own contrast is over 4 sRGB levels, i.e. where there is "
                       "something that could visibly double"),
               beta=beta, own_min_frac=OWN_MIN_FRAC,
               align_weights=ALIGN_W, seconds=round(time.time() - t0),
               per_rect=out)
    json.dump(rep, open(D + '/work/own.json', 'w'), indent=1)
    H.mark(setname, 'own', {k: v for k, v in rep.items() if k != 'per_rect'})
    H.log(f"own[{setname}]: {len(out)} rects, {tot} owned patches, "
          f"boundaries {nb_before} -> {nb_after}")
    return rep


# ------------------------------------------------------------- the bake ---- #
_B = {}


def offset_field(rid, view, rc, npu, npv, uu, vv, ref):
    """Bilinear (du, dv) over the rect's texels from the per-patch offsets
    stage 60's refine step measured for this view, fading to the view's global
    placement where a patch did not correlate."""
    key = (int(rid), int(view))
    tab = _B['refine'].get(key)
    nu, nv = len(uu), len(vv)
    if tab is None:
        return None
    DU = np.zeros((npv, npu), 'f8'); DV = np.zeros((npv, npu), 'f8')
    OK = np.zeros((npv, npu), 'f8')
    for p, du, dv, ok in tab:
        pi, pj = divmod(int(p), npu)
        if ok:
            DU[pi, pj] = du; DV[pi, pj] = dv; OK[pi, pj] = 1.0
    if OK.sum() == 0:
        return None
    pw = rc.width / npu; ph = rc.height / npv
    fx = np.clip(uu / pw - 0.5, 0, npu - 1.0001)
    fy = np.clip(vv / ph - 0.5, 0, npv - 1.0001)
    j0 = np.floor(fx).astype(int); i0 = np.floor(fy).astype(int)
    tx_ = (fx - j0)[None, :]; ty_ = (fy - i0)[:, None]
    au = np.zeros((nv, nu)); av = np.zeros((nv, nu)); aw = np.zeros((nv, nu))
    for di in (0, 1):
        for dj in (0, 1):
            ii = np.minimum(i0 + di, npv - 1); jj = np.minimum(j0 + dj, npu - 1)
            w = ((ty_ if di else 1 - ty_) * (tx_ if dj else 1 - tx_))
            ww = w * OK[np.ix_(ii, jj)]
            au += ww * DU[np.ix_(ii, jj)]
            av += ww * DV[np.ix_(ii, jj)]
            aw += ww
    g = aw > 1e-6
    return (np.where(g, au / np.maximum(aw, 1e-6), 0.0),
            np.where(g, av / np.maximum(aw, 1e-6), 0.0))


def _init_bake(setname, dq=None, cap=None, sigs=None):
    _B['dq'] = dq or {}
    _B['cap'] = cap or {}
    _B['refine'] = {}
    rp = H.setdir(setname) + '/work/refine.npz'
    if os.path.exists(rp):
        d = np.load(rp)
        r = d['rect']; v = d['view']; p = d['patch']
        du = d['du']; dv = d['dv']; ok = d['ok']
        for i in range(len(r)):
            _B['refine'].setdefault((int(r[i]), int(v[i])), []).append(
                (int(p[i]), float(du[i]), float(dv[i]), int(ok[i])))
    rects, _ = H.load_rects(setname)
    _B['rects'] = {r.id: r for r in rects}
    _B['sigs'] = sigs if sigs is not None else rect_sigs(rects)
    _B['set'] = setname
    D = H.setdir(setname)
    _B['align'] = json.load(open(D + '/work/align.json'))['per_rect']
    _B['own'] = json.load(open(D + '/work/own.json'))['per_rect']
    d = np.load(D + '/work/cover.npz', allow_pickle=True)
    _B['pids'] = [str(x) for x in d['pids']]
    if _FB is not None:
        # each (rect, view)'s cover score, the same quality run_own ranks
        # owners by (incidence^0.7 / range falloff, times the share it sees)
        fr = d['nval'] / np.maximum(d['ntot'], 1)
        q = fr * (np.clip(d['cos'], 0, 1) ** 0.7) / (1.0 + (d['rng'] / 8.0) ** 2)
        sc = {}
        for r_, v_, q_ in zip(d['rect'], d['view'], q):
            sc[(int(r_), int(v_))] = sc.get((int(r_), int(v_)), 0.0) + float(q_)
        _B['fbscore'] = sc
    poses, _ = T.load_poses()
    _B['poses'] = poses
    _B['dsm'] = T.DSM()
    _B['gains'] = {}
    gp = H.SCR + '/delight/gains.npz'
    if os.path.exists(gp):
        g = np.load(gp, allow_pickle=True)
        _B['gains'] = {str(p): np.asarray(x, 'f4') for p, x in zip(g['pids'], g['gain'])}


def tile_path(setname, rid):
    return f'{H.setdir(setname)}/tiles/r{rid:04d}.npz'


# tiles are filed under the s33 wall id, which moves when
# geometry is added.  A tile on disk is this wall's tile only
# if it carries the wall's s34 signature (wpalign.rect_sig) AND was baked from
# the panoramas that own the wall now (view indices are positions in this
# run's cover.npz, so they are compared as panorama ids).
def owner_views(ow):
    return sorted(set(int(x) for x in ow['owner'] if x >= 0))


def tile_ok(meta, sig, view_pids):
    return meta.get('sig') == sig and meta.get('view_pids') == view_pids


def purge_tiles(setname, sigs, own, pids, rids, only=None):
    """Remove every tile a fresh bake would not leave behind: an id the rect
    set lacks or that owns no patch, another wall's geometry, or
    other owner panoramas.  With `only`, tiles outside it are left alone
    (wpatlas.run_relevel re-bakes one wall at a time)."""
    D = H.setdir(setname) + '/tiles'
    if not os.path.isdir(D):
        return 0
    rids = set(rids)
    n = 0
    for f in sorted(os.listdir(D)):
        if not (f.startswith('r') and f.endswith('.npz') and f[1:5].isdigit()) \
                or len(f) != 9:
            continue                               # .retrybak, .tmp.npz: not a tile
        rid = int(f[1:5])
        if only is not None and rid not in only:
            continue
        keep = False
        if rid in sigs and rid in rids:
            try:
                m = json.loads(str(np.load(os.path.join(D, f), allow_pickle=True)['meta']))
                keep = tile_ok(m, sigs[rid], [pids[v] for v in owner_views(own[str(rid)])])
            except Exception:
                keep = False
        if not keep:
            os.remove(os.path.join(D, f)); n += 1
    return n


def _solve_levels(pairs, views, lam=0.02):
    """Least squares per-view log gain from the pairwise median log differences.

    pairs: {(i, j): (d_rgb (3,), n)}.  -> {view: gain (3,)}"""
    idx = {v: i for i, v in enumerate(views)}
    n = len(views)
    if n < 2 or not pairs:
        return {v: np.ones(3, 'f4') for v in views}
    rows = []; rhs = []; wts = []
    for (i, j), (dv, cnt) in pairs.items():
        r = np.zeros(n); r[idx[i]] = 1.0; r[idx[j]] = -1.0
        rows.append(r); rhs.append(dv); wts.append(math.sqrt(max(cnt, 1)))
    A = np.array(rows) * np.array(wts)[:, None]
    B = np.array(rhs) * np.array(wts)[:, None]
    # gauge: pull the mean gain to zero
    A = np.vstack([A, np.full((1, n), lam * np.sqrt(sum(wts)))])
    B = np.vstack([B, np.zeros((1, 3))])
    g, *_ = np.linalg.lstsq(A, B, rcond=None)
    return {v: np.exp(np.clip(g[idx[v]], -0.6, 0.6)).astype('f4') for v in views}


def _bake_rect(rid):
    setname = _B['set']
    ow = _B['own'][str(rid)]
    view_pids = [_B['pids'][v] for v in owner_views(ow)]
    rc = _B['rects'][rid]
    # the texel is an input of the tile too.  The second s35 -> s60 bake pass in
    # pipeline/area_cli.py re-reads the material texel cap from the new
    # _pre_wallphoto/facadetex.json;
    # a tile keyed only on its signature and owner panoramas would be returned as it
    # stood, so a changed cap would never reach it.  A tile is reused only when it was
    # baked at the texel this run asks for.
    tx = H.fine_texel(rc, _B['dq'].get(rid, 12.0), cap_m=_B.get('cap', {}).get(rid))
    if os.path.exists(tile_path(setname, rid)):
        try:
            d = np.load(tile_path(setname, rid), allow_pickle=True)
            meta = json.loads(str(d['meta']))
            if tile_ok(meta, _B['sigs'][rid], view_pids) \
                    and abs(float(meta.get('tx', -1.0)) - round(tx, 4)) < 5e-5 \
                    and (_FB is None or meta.get('fb') == _fb_sig()):
                return meta
        except Exception:
            pass
    al = _B['align'][str(rid)]
    owner = np.array(ow['owner'], 'i4')
    npu, npv = ow['npu'], ow['npv']
    if (owner >= 0).sum() == 0:
        return None
    shifts = {int(k): v for k, v in al['shift_m'].items()}
    vscale = al['anchor'].get('vscale', 1.0)
    views = sorted(set(int(x) for x in owner if x >= 0))
    # ---- the fine grid ----------------------------------------------------- #
    uu, vv = H.grid_axes(rc, tx)
    nu, nv = len(uu), len(vv)
    UU, VV = np.meshgrid(uu, vv)
    pid_of = H.patch_of_uv(rc, UU.ravel(), VV.ravel(), npu, npv).reshape(nv, nu)
    own_tex = owner[pid_of]
    # ---- sample every owner ------------------------------------------------ #
    lin = {}; val = {}; cosv = {}; segv = {}; rngv = {}
    for v in views:
        pid = _B['pids'][v]
        try:
            ctx = ViewCtx(pid, _B['poses'][pid], _B['dsm'])
            wpx = ctx.native_width
            g = shifts.get(v, (0.0, 0.0))
            fld = offset_field(rid, v, rc, npu, npv, uu, vv, None)
            offs = ((g[0] + fld[0], g[1] + fld[1]) if fld is not None else g)
            res = sample_rect(ctx, rc, uu, vv, width=wpx,
                              offset=offs, vscale=vscale)
            ctx.free()
        except Exception:
            continue
        cosv[v] = res['cos']
        if 'seg' in res:
            segv[v] = res['seg']
        if _FB is not None:
            rngv[v] = res['rng']
        L = H.to_linear(res['rgb'])
        g = _B['gains'].get(pid)
        if g is not None:
            L = L / np.asarray(g, 'f4')[None, None, :]
        lin[v] = L.astype('f4'); val[v] = res['valid']
    views = [v for v in views if v in lin]
    if not views:
        return None
    extras = []
    if _FB is not None:
        # the next-best panoramas of this wall -- correlation-aligned to
        # the reference like every owner (an unaligned view sits a pose error
        # away and would smear), not banned by the exposure retry, best cover
        # score first -- sampled through the same veto as the owners
        ban = set(int(x) for x in ow.get('banned', []))
        cand = [int(k) for k, pv in al['per_view'].items()
                if (pv.get('via') == 'reference' or str(pv.get('via', '')).startswith('view'))
                and int(k) not in views and int(k) not in ban]
        cand.sort(key=lambda k: -_B['fbscore'].get((int(rid), k), 0.0))
        for v in cand[:int(_FB.get('k_views', 4))]:
            pid = _B['pids'][v]
            try:
                ctx = ViewCtx(pid, _B['poses'][pid], _B['dsm'])
                g = shifts.get(v, (0.0, 0.0))
                fld = offset_field(rid, v, rc, npu, npv, uu, vv, None)
                offs = ((g[0] + fld[0], g[1] + fld[1]) if fld is not None else g)
                res = sample_rect(ctx, rc, uu, vv, width=ctx.native_width,
                                  offset=offs, vscale=vscale)
                ctx.free()
            except Exception:
                continue
            if not res['valid'].any():
                continue
            cosv[v] = res['cos']; rngv[v] = res['rng']
            if 'seg' in res:
                segv[v] = res['seg']
            L = H.to_linear(res['rgb'])
            g = _B['gains'].get(pid)
            if g is not None:
                L = L / np.asarray(g, 'f4')[None, None, :]
            lin[v] = L.astype('f4'); val[v] = res['valid']
            extras.append(v)
    # ---- levelling ---------------------------------------------------------- #
    # Lempitsky and Ivanov's globally optimal luminance correction, fitted where
    # it is needed and measured where it shows: the band around the boundaries
    # between two owners.  Fitting over every shared texel makes the seam step
    # worse (29.2 -> 33.0 sRGB levels),
    # because two views of the same wall disagree most where one of them is
    # looking through a tree, and that dominates a whole-overlap fit.
    from scipy.ndimage import uniform_filter
    own_f = {}
    bw = max(3, int(round(0.35 / tx)) | 1)
    for v in views:
        own_f[v] = uniform_filter((own_tex == v).astype('f4'), bw, mode='nearest') > 0.02
    pairs = {}
    band = {}
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            a, b = views[i], views[j]
            m = own_f[a] & own_f[b] & val[a] & val[b]
            n = int(m.sum())
            if n < 200:
                continue
            ii = np.nonzero(m.ravel())[0]
            if len(ii) > 120000:
                ii = ii[:: len(ii) // 120000 + 1]
            la = np.log(np.maximum(lin[a].reshape(-1, 3)[ii], 1e-4))
            lb = np.log(np.maximum(lin[b].reshape(-1, 3)[ii], 1e-4))
            # the solve wants g_a - g_b = log b - log a, so that a's corrected
            # colour meets b's: the other sign doubles the step instead of
            # removing it (measured: 26.8 -> 40.7 sRGB levels)
            pairs[(a, b)] = (np.median(lb - la, axis=0), n)
            band[(a, b)] = ii
    gain = _solve_levels(pairs, views)
    # ---- the seam measurement, before and after ---------------------------- #
    seam = []
    for (a, b), ii in band.items():
        ca = lin[a].reshape(-1, 3)[ii]; cb = lin[b].reshape(-1, 3)[ii]
        f0 = np.abs(H.to_srgb(ca @ H.LUMW).astype('f4') -
                    H.to_srgb(cb @ H.LUMW).astype('f4'))
        f1 = np.abs(H.to_srgb((ca * gain[a][None, :]) @ H.LUMW).astype('f4') -
                    H.to_srgb((cb * gain[b][None, :]) @ H.LUMW).astype('f4'))
        seam.append((float(np.median(f0)), float(np.median(f1)), int(len(ii))))
    # ---- feathered composite ----------------------------------------------- #
    fw = max(3, int(round(2 * H.FEATHER_M / tx)) | 1)
    acc = np.zeros((nv, nu, 3), 'f4'); wsum = np.zeros((nv, nu), 'f4')
    for v in views:
        wm = uniform_filter((own_tex == v).astype('f4'), fw, mode='nearest')
        w = wm * val[v].astype('f4')
        if not w.any():
            continue
        acc += (lin[v] * gain[v][None, None, :]) * w[:, :, None]
        wsum += w
    got = wsum > 1e-4
    # A texel inside an owned patch that its owner cannot see -- a lamp post, a
    # bollard, somebody's head -- takes the next owner that can.  Patches with
    # no owner are NOT filled: those are the ones the agreement gate dropped,
    # and filling them would put back the doubled image it just removed.
    fillable = own_tex >= 0
    srcv = None
    if _FB is not None:
        # a texel no owner can carry takes the best view of ALL the
        # candidates (owners and the aligned extras) that passes the veto, the
        # incidence gate and the occluder test, ranked per texel by the same
        # quality ownership uses; an extra is first brought to the owners'
        # exposure over the texels both already show
        if _FB.get('unowned', True):
            drop = np.array(sorted(set(int(x) for x in ow.get('dropped', []))), 'i8')
            fillable = fillable | ~np.isin(pid_of, drop)
        cur = np.zeros((nv, nu, 3), 'f4')
        cur[got] = acc[got] / wsum[got][:, None]
        for v in extras:
            m = got & val[v]
            if m.sum() >= 200:
                lg = np.median(np.log(np.maximum(cur[m], 1e-4)) -
                               np.log(np.maximum(lin[v][m], 1e-4)), axis=0)
                gain[v] = np.exp(np.clip(lg, -0.6, 0.6)).astype('f4')
            else:
                gain[v] = np.ones(3, 'f4')
        need = fillable & ~got
        bq = np.zeros((nv, nu), 'f4'); srcv = np.full((nv, nu), -1, 'i4')
        for v in views + extras:
            q = (np.clip(cosv[v], 0, 1) ** 0.7) / (1.0 + (rngv[v] / 8.0) ** 2)
            m = need & val[v] & (cosv[v] > float(_FB.get('min_cos', 0.45))) & (q > bq)
            bq[m] = q[m]; srcv[m] = v
        for v in views + extras:
            m = srcv == v
            if m.any():
                acc[m] = lin[v][m] * gain[v][None, :]
                wsum[m] = 1.0
        got = wsum > 1e-4
    elif (fillable & ~got).any():
        for v in views:
            # not at a grazing angle: a view that sees this texel edge-on
            # smears a metre of wall into it
            m = fillable & (~got) & val[v] & (cosv[v] > 0.45)
            if m.any():
                acc[m] = lin[v][m] * gain[v][None, :]
                wsum[m] = 1.0
                got = wsum > 1e-4
    # Islands: a few hundred stray texels of photograph in a field of material
    # read as dirt, and they are usually the ragged edge of an occluder rather
    # than wall.  Anything smaller than MIN_ISLAND_M2 goes back to material.
    from scipy.ndimage import label, binary_closing
    lab, nlab = label(got)
    if nlab:
        sz = np.bincount(lab.ravel())
        small = np.nonzero(sz * tx * tx < MIN_ISLAND_M2)[0]
        small = small[small > 0]
        if len(small):
            got &= ~np.isin(lab, small)
    k = max(3, int(round(0.10 / tx)) | 1)
    got = binary_closing(got, np.ones((k, k), bool))
    got &= (wsum > 1e-4)
    photo = np.zeros((nv, nu, 3), 'f4')
    photo[got] = acc[got] / wsum[got][:, None]
    used_x = [v for v in extras if srcv is not None and (got & (srcv == v)).any()]
    meta = dict(rect=int(rid), tx=round(tx, 4), nu=nu, nv=nv,
                views=[int(v) for v in views + used_x],
                sig=_B['sigs'][rid], view_pids=view_pids,
                owned_patches=int((owner >= 0).sum()),
                patches=int(npu * npv),
                photo_frac=round(float(got.mean()), 4),
                vscale=round(vscale, 4),
                anchor=al['anchor'],
                gains={int(v): [round(float(x), 3) for x in gain[v]] for v in views + used_x},
                seam_before=round(float(np.average([s[0] for s in seam],
                                                   weights=[s[2] for s in seam])), 2)
                if seam else None,
                seam_after=round(float(np.average([s[1] for s in seam],
                                                  weights=[s[2] for s in seam])), 2)
                if seam else None,
                seam_texels=int(sum(s[2] for s in seam)),
                range_m=round(float(_B['dq'].get(rid, 0.0)), 1))
    if segv:
        # FACADE_PHOTO_QUALITY measure: the final photo texels
        # by their owner view's segmentation label and incidence
        nw = gz = 0
        for v in views:
            m = got & (own_tex == v) & val[v]
            if srcv is not None:
                m &= srcv < 0
            if v in segv:
                nw += int((m & segv[v]).sum())
            gz += int((m & (cosv[v] < 0.5)).sum())
        if srcv is not None:
            for v in views + extras:      # the filled texels, by their source
                m = got & (srcv == v)
                if v in segv:
                    nw += int((m & segv[v]).sum())
                gz += int((m & (cosv[v] < 0.5)).sum())
        meta['qstats'] = dict(texels=int(got.sum()), nonwall=nw, graze_lt05=gz,
                              m2=round(float(got.sum()) * tx * tx, 2))
    if srcv is not None:
        fo = got & (srcv >= 0) & np.isin(srcv, np.array(views, 'i4'))
        fx = got & (srcv >= 0) & ~np.isin(srcv, np.array(views, 'i4'))
        meta['fb'] = _fb_sig()
        meta['fallback'] = dict(extras_sampled=[int(v) for v in extras],
                                extras_used=[int(v) for v in used_x],
                                m2_from_owners=round(float(fo.sum()) * tx * tx, 2),
                                m2_from_extras=round(float(fx.sum()) * tx * tx, 2),
                                m2_unowned=round(float((got & (own_tex < 0)).sum()) * tx * tx, 2))
    np.savez_compressed(tile_path(setname, rid) + '.tmp.npz',
                        photo=H.to_srgb(photo), got=got,
                        meta=np.array(json.dumps(meta)))
    os.replace(tile_path(setname, rid) + '.tmp.npz', tile_path(setname, rid))
    return meta


def run_bake(setname, workers=8, only=None):
    t0 = time.time()
    D = H.setdir(setname)
    own = json.load(open(D + '/work/own.json'))['per_rect']
    d = np.load(D + '/work/cover.npz', allow_pickle=True)
    # the range each rect was actually photographed from: the 25th percentile
    # of the owners' own ranges, which is what sets the texel size
    rect = d['rect']; view = d['view']; rng = d['rng']
    dq = {}
    for rid_s, o in own.items():
        rid = int(rid_s)
        vs = set(int(x) for x in o['owner'] if x >= 0)
        m = (rect == rid) & np.isin(view, list(vs)) if vs else np.zeros(len(rect), bool)
        dq[rid] = float(np.percentile(rng[m], 25)) if m.any() else 15.0
    # the material tile each wall already has: the photograph may not be coarser.
    # The facadetex record is found by id only if it is still this wall (stage
    # 35's part/area/size test); an id that moved is looked up among the
    # records of the same part, and used only when exactly one matches.
    rects, _ = H.load_rects(setname)
    sigs = rect_sigs(rects)
    cap = {}
    ncap_moved = ncap_none = 0
    try:
        import json as _j
        _ref = H.FAC + '/_pre_wallphoto/facadetex.json'
        if not os.path.exists(_ref):
            _ref = H.FAC + '/facadetex.json'
        _recs = [_r for _r in _j.load(open(_ref))['per_wall'] if _r.get('cm_per_texel')]
        _byid = {int(_r['id']): _r for _r in _recs}
        for rc in rects:
            _r = _byid.get(int(rc.id))
            if not rec_matches(rc, _r):
                _c = [x for x in _recs if rec_matches(rc, x)]
                _r = _c[0] if len(_c) == 1 else None
                if _r is None:
                    ncap_none += 1
                    continue
                ncap_moved += 1
            cap[int(rc.id)] = float(_r['cm_per_texel']) / 100.0
        H.log(f"bake: material texel cap for {len(cap)} rects "
              f"({ncap_moved} found under another id, {ncap_none} without a record)")
    except Exception as e:
        H.log(f"bake: no material texel cap ({e})")
    rids = sorted(int(k) for k in own if own[k]['owned_patches'] > 0 and int(k) in sigs)
    pids = [str(x) for x in d['pids']]
    nst = purge_tiles(setname, sigs, own, pids, rids, only)
    H.log(f"bake[{setname}]: {nst} stale tile files removed (id/geometry signature, "
          f"owner panoramas, or no owned patches)")
    if only is not None:
        rids = [r for r in rids if r in only]
    H.log(f"bake[{setname}]: {len(rids)} rects")
    import multiprocessing as mp
    recs = []
    with mp.Pool(workers, initializer=_init_bake,
                 initargs=(setname, dq, cap, sigs)) as pool:
        for n, r in enumerate(pool.imap_unordered(_bake_rect, rids, chunksize=1)):
            if r is not None:
                recs.append(r)
            if (n + 1) % 10 == 0:
                H.log(f"  {n+1}/{len(rids)} rects, {time.time()-t0:.0f}s")
    sb = [r['seam_before'] for r in recs if r.get('seam_before') is not None]
    sa = [r['seam_after'] for r in recs if r.get('seam_after') is not None]
    rep = dict(set=setname, rects=len(recs),
               texel_cm={f'p{p}': round(float(np.percentile(
                   [r['tx'] * 100 for r in recs], p)), 2) for p in (10, 50, 90)}
               if recs else {},
               photo_frac_median=round(float(np.median(
                   [r['photo_frac'] for r in recs])), 3) if recs else None,
               seam_step_sRGB=dict(
                   before=round(float(np.median(sb)), 2) if sb else None,
                   after=round(float(np.median(sa)), 2) if sa else None,
                   rects=len(sb)),
               feather_m=H.FEATHER_M,
               seconds=round(time.time() - t0), per_rect=recs)
    json.dump(rep, open(D + '/work/bake.json', 'w'), indent=1)
    H.mark(setname, 'bake', {k: v for k, v in rep.items() if k != 'per_rect'})
    H.log(f"bake[{setname}]: {len(recs)} rects, seam {rep['seam_step_sRGB']}, "
          f"{rep['seconds']}s")
    return rep
