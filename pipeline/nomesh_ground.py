"""Poses-mode (no photo mesh) ground colour and class clean-up, for any area.

A clean ground needs the panorama colour turned into a world-XY tint raster (low
frequency only, levelled to the ortho) that s22 samples per vertex on a refined
mesh.  Without a photo mesh the alternative is fanning one projected colour per
3dfier face across the TIN, and 3dfier TINs are sliver-heavy (about half of the
up faces can have aspect > 10, with longest edges of 10 m and more), so every
colour step becomes a streak, and an unfiltered projection (no skyline mask in
poses mode, loose poses, grazing views) leaves a 20 % bright tail that pushes the
class tiles to the 0.88 cap (an over-bright floor).

This file writes the same rasters from data poses mode does have, so the stock
s50 and s22 code paths (world tint, LEPP refine, class-confined tint, tint relief
clamped to the ortho, edge and canopy fill) run unchanged:

  tint     after s20, before s21:
           ortho -> ENU 0.25 m (linear), masked off lidar canopy, buildings and
           cast shadow; the panoramas re-projected with an incidence and range
           gate (low-frequency colour only), levelled to the ortho; writes
           floor/tint_map.npz, floor/tint_map_blend.npz, ortho/rasters.npz and
           replaces project/ground_terrain_face.npz colours with the raster's
           (the s20 file is kept as ground_terrain_face_s20.npz), so s21's class
           votes and s22's tile means read the gated colour.
  classes  after s21, before s50: photo-decided class islands smaller than
           MIN_ISLAND_M2 take their neighbourhood's class (removes small jagged class patches),
           then polish/class_map.npz for s50 / s22.

Usage (OPENCITY_AREA set, run from the area work dir):  nomesh_ground.py tint|classes
Only areas with paths.scratch_dir (OPENCITY_AREA set) run this file.
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
from scipy import ndimage
import common as C
import texlib as T

SCR = T.SCR
FLOOR, ORTHO, PJ, MT, POL = (SCR + '/floor', SCR + '/ortho', SCR + '/project',
                             SCR + '/material', SCR + '/polish')
CELL = 0.25
# panorama gate for ground colour.  Without it about a third of the kept ground
# samples have cos(incidence) < 0.2.  A ground point seen from a 2.26 m camera at
# 15 m has cos 0.15, so the range gate is the tighter one; together they keep the
# steep, near views.
PANO_COS_MIN = 0.30
PANO_R_MAX = 15.0
PANO_BRIGHT_MAX = 0.60      # linear luma: sky / glare / white object, not ground
PANO_SIGMA_M = 1.0          # low-frequency colour only
PANO_MIN_WEIGHT = 0.15      # normalised-convolution support needed to trust a cell
ORTHO_OWN = 0.80            # ortho share of the blend where the ortho is usable
CANOPY_PAD_M = 1.5          # crown lean + blur in the ortho
BUILDING_PAD_M = 1.0
SHADOW_RATIO = 0.55         # ortho luma under this x its 6 m local level = cast shadow
CAL_CHROMA_TOL = 0.10      # per-channel ortho gain may differ from the luma gain by this much
MIN_ISLAND_M2 = 8.0         # a photo-decided class island smaller than this is noise
ISLAND_SRC = (1, 2, 4)      # s21 src codes it may reassign: photo_only, photo_override, fallback
# per-class colour transfer, the classes step.  After canopy, building pad and cast shadow
# the ortho sees only 22-34 % of the ground cells (Lombard 0.219, Tucson 0.303, Boston 0.335).
# Without the transfer every other cell takes a cross-class cascade fill of grey pano /
# neighbour colour, and the pano calibration's +-10 % per-channel term cools the ortho
# (Tucson gain 0.947 / 0.998 / 1.126 turns lit concrete sat 0.147 into 0.081).
# The transfer: the ortho's own hue (dark-object haze removed, luma levelled back to the
# tint step's level); cast-shadow cells de-shadowed with a local in-class per-channel gain;
# every other cell of a class filled from that class's seen cells only.  All numbers are
# derived per area from the ortho; area.yaml stage_knobs.GROUND_COLOUR_TRANSFER overrides
# any key; {enabled: false} selects the clamp_pano_cells path.
GROUND_COLOUR_TRANSFER = dict(enabled=True, haze_pct=0.5, haze_keep_min=0.5, shadow_sigma_m=6.0,
                              shadow_gain_max=16.0, min_support=0.05, luma_pct=(10, 95),
                              skip_classes=('water',))


def _stage_knob(name, default):
    """area.yaml / derived.yaml stage_knobs.<name>.value; a dict value updates the default dict."""
    v = ((((getattr(C, '_AREA_CFG', None) or {}).get('stage_knobs') or {}).get(name)) or {}).get('value')
    if v is None:
        return default
    return dict(default, **v) if isinstance(default, dict) and isinstance(v, dict) else v


def _grid():
    z = np.load(C.W + '/layers/ground_terrain.npz')['pos']
    x0 = math.floor(float(z[:, 0].min())) - 2.0; y0 = math.floor(float(z[:, 1].min())) - 2.0
    nx = int(math.ceil((float(z[:, 0].max()) + 2.0 - x0) / CELL)); ny = int(math.ceil((float(z[:, 1].max()) + 2.0 - y0) / CELL))
    return x0, y0, CELL, nx, ny


def mgauss(A, m, s_px):
    """Normalised convolution inside mask m -> (value, support)."""
    mf = ndimage.gaussian_filter(m.astype('f4'), s_px, mode='nearest')
    if A.ndim == 2:
        v = ndimage.gaussian_filter(np.where(m, A, 0).astype('f4'), s_px, mode='nearest')
        return v / np.maximum(mf, 1e-6), mf
    v = np.stack([ndimage.gaussian_filter(np.where(m, A[..., k], 0).astype('f4'), s_px, mode='nearest')
                  for k in range(A.shape[2])], -1)
    return v / np.maximum(mf, 1e-6)[..., None], mf


def cascade_fill(A, m):
    """Fill cells outside m smoothly from inside (the s50 rule: no Voronoi steps)."""
    out = np.where(m[..., None], A, 0).astype('f4'); filled = m.copy()
    for sig in (2.0, 6.0, 18.0, 54.0, 162.0):
        if filled.all():
            break
        g, mf = mgauss(A, m, sig)
        ok = (~filled) & (mf > 1e-5); out[ok] = g[ok]; filled |= ok
    if not filled.all():
        ix = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
        out = out[tuple(ix)]
    return out


def _resample_mask(src, sgrid, grid):
    hx0, hy0, hc, hnx, hny = [float(v) for v in sgrid]; x0, y0, c, nx, ny = grid
    jx = np.clip(((x0 + (np.arange(nx) + 0.5) * c - hx0) / hc).astype(int), 0, int(hnx) - 1)
    iy = np.clip(((y0 + (np.arange(ny) + 0.5) * c - hy0) / hc).astype(int), 0, int(hny) - 1)
    return src[np.ix_(iy, jx)]


def _tile_sample(r, lon, lat, coarse_m=None):
    """Sample ortho tile r (any CRS) at lon/lat: uint8-scale float RGB (n, 3) and a hit mask.

    Nearest pixel for a tile as fine as coarse_m or finer (a local web-mercator
    ortho).  When the tile's ground pixel is coarser than coarse_m (NAIP, 0.3-1 m)
    the sample is bilinear, so a 1 m pixel does not print as a 1 m block on a 0.25 m grid."""
    import pyproj
    key = r.crs.to_string()
    tr = _TR.get(key) or _TR.setdefault(key, pyproj.Transformer.from_crs(4326, r.crs, always_xy=True))
    mx, my = tr.transform(lon, lat)
    col, row = ~r.transform * (mx, my)
    img = r.read([1, 2, 3]).astype('f4')
    gm = abs(r.transform.a) * (1.0 if r.crs.is_projected and r.crs.to_epsg() != 3857 else
                               math.cos(math.radians(float(np.mean(lat)))))
    if coarse_m is None or gm <= coarse_m:
        ci = np.floor(col).astype(int); ri = np.floor(row).astype(int)
        ok = (ci >= 0) & (ci < r.width) & (ri >= 0) & (ri < r.height)
        v = np.zeros((len(ci), 3), 'f4'); v[ok] = img[:, ri[ok], ci[ok]].T
        return v, ok
    cx, cy = col - 0.5, row - 0.5
    c0 = np.floor(cx).astype(int); r0 = np.floor(cy).astype(int); fx = (cx - c0).astype('f4'); fy = (cy - r0).astype('f4')
    ok = (c0 >= 0) & (c0 + 1 < r.width) & (r0 >= 0) & (r0 + 1 < r.height)
    v = np.zeros((len(c0), 3), 'f4')
    a, b, f, g = c0[ok], r0[ok], fx[ok, None], fy[ok, None]
    q00, q10, q01, q11 = (img[:, b, a].T, img[:, b, a + 1].T, img[:, b + 1, a].T, img[:, b + 1, a + 1].T)
    zero = (q00.max(1) == 0) | (q10.max(1) == 0) | (q01.max(1) == 0) | (q11.max(1) == 0)
    v[ok] = (q00 * (1 - f) + q10 * f) * (1 - g) + (q01 * (1 - f) + q11 * f) * g
    v[np.nonzero(ok)[0][zero]] = 0.0                     # touching nodata -> nodata
    return v, ok


_TR = {}


def ortho_raster(grid, rep):
    """Ortho tiles in <scratch>/ortho (NAIP in the area's UTM zone from
    pipeline/fetch_ortho.py, or a local ortho; any CRS) -> linear RGB on the ENU grid."""
    import rasterio
    x0, y0, c, nx, ny = grid
    od = os.path.dirname(SCR) + '/ortho'        # <scratch>/ortho, written by pipeline/fetch_ortho.py
    tiles = sorted(p for p in os.listdir(od) if p.endswith('.tif'))
    acc = np.zeros((3, ny * nx), 'f8'); cnt = np.zeros(ny * nx, 'f8')
    SUB = 4                                   # 4x4 sub-samples per 0.25 m cell (resolves a fine local ortho)
    for k in range(SUB * SUB):
        ox = ((k % SUB) + 0.5) / SUB; oy = ((k // SUB) + 0.5) / SUB
        X, Y = np.meshgrid(x0 + (np.arange(nx) + ox) * c, y0 + (np.arange(ny) + oy) * c)
        lat, lon, _ = C.enu_to_lla(X.ravel(), Y.ravel(), 0.0, o=C.ORIGIN)
        for tp in tiles:
            with rasterio.open(os.path.join(od, tp)) as r:
                vv, ok = _tile_sample(r, lon, lat, coarse_m=c)
                if not ok.any():
                    continue
                v = vv[ok]
                good = v.max(1) > 0                # nodata = black
                flat = np.nonzero(ok)[0][good]
                lin = np.power(v[good] / 255.0, 2.2)
                for q in range(3):
                    acc[q, flat] += lin[:, q]
                cnt[flat] += 1.0
    cov = (cnt >= SUB * SUB * 0.5).reshape(ny, nx)
    Mo = (acc / np.maximum(cnt, 1)[None]).T.reshape(ny, nx, 3).astype('f4')
    rep['ortho_tiles'] = len(tiles); rep['ortho_coverage'] = round(float(cov.mean()), 4)
    return Mo, cov


def gated_projection(rep):
    """s20's per-surfel projection on the ground layer with the incidence/range/brightness gate.
    -> (surfel xy, linear colour, weight) for surfels that kept >= 1 sample."""
    import s20_project as S20
    S = S20.build_surfels(False)
    m = S['lay'] == 1
    P = S['pos'][m]; N = S['nrm'][m]
    poses, _ = T.load_poses(); sel = T.thin_poses(poses, 1.5)
    gains = {}
    gp = SCR + '/delight/gains.npz'
    if os.path.exists(gp):
        d = np.load(gp, allow_pickle=True); gains = {str(p): g for p, g in zip(d['pids'], d['gain'])}
    dsm = T.DSM()
    acc = np.zeros((len(P), 3), 'f8'); wsum = np.zeros(len(P), 'f8'); nkept = 0; nseen = 0
    for pid, p in sel.items():
        try:
            s = T.PanoSampler(dsm, pid, p, r_max=PANO_R_MAX, width=S20.SAMPLE_W)
            idx, lin, rng = s.sample(P); s.free()
        except Exception:
            continue
        if len(idx) == 0:
            continue
        dv = P[idx] - np.array([p['x'], p['y'], p['z']], 'f4')
        dv /= np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-6)
        cosi = np.abs((N[idx] * dv).sum(1))
        raw_l = lin @ T.LUMW
        nseen += len(idx)
        ok = (cosi >= PANO_COS_MIN) & (raw_l <= PANO_BRIGHT_MAX)
        idx, lin, rng, cosi = idx[ok], lin[ok], rng[ok], cosi[ok]
        g = gains.get(pid)
        # the per-pano gain is applied in the log domain clamped to [1/2, 2]: the no-mesh
        # delight solve is loose (gain log sd up to about 0.6)
        if g is not None:
            lin = lin / np.clip(np.asarray(g, 'f4'), 0.5, 2.0)
        w = cosi * (8.0 / (8.0 + rng))
        np.add.at(acc, idx, lin * w[:, None]); np.add.at(wsum, idx, w); nkept += len(idx)
    have = wsum > 0
    rep['pano_gate'] = dict(cos_min=PANO_COS_MIN, r_max_m=PANO_R_MAX, bright_max=PANO_BRIGHT_MAX,
                            samples_admissible=int(nseen), samples_kept=int(nkept),
                            kept_frac=round(nkept / max(nseen, 1), 3),
                            surfels_coloured=round(float(have.mean()), 3))
    return P[have, :2], (acc[have] / wsum[have, None]).astype('f4'), wsum[have]


def cmd_tint():
    t0 = time.time(); rep = {}
    grid = _grid(); x0, y0, c, nx, ny = grid
    rep['grid'] = dict(origin=[x0, y0], cell_m=c, px=[nx, ny])
    for d in (FLOOR, ORTHO, POL):
        os.makedirs(d, exist_ok=True)
    LW = T.LUMW.astype('f4')
    # ---- ortho, masked to open ground ------------------------------------ #
    Mo, cov = ortho_raster(grid, rep)
    ch = np.load(C.W + '/geom/chm.npz')
    can = _resample_mask(ch['canopy'].astype(bool), ch['grid'], grid)
    bld = _resample_mask(ch['bmask'].astype(bool), ch['grid'], grid)
    wm = _resample_mask(ch['wmask'].astype(bool), ch['grid'], grid) if 'wmask' in ch.files else np.zeros((ny, nx), bool)
    can_d = ndimage.distance_transform_edt(~can) * c <= CANOPY_PAD_M
    bld_d = ndimage.distance_transform_edt(~bld) * c <= BUILDING_PAD_M
    open_ = cov & ~can_d & ~bld_d
    Lo = np.log(np.maximum(Mo @ LW, 1e-5))
    loc, _ = mgauss(Lo, open_, 6.0 / c)
    shadow = open_ & (Lo < loc + math.log(SHADOW_RATIO))
    ovalid = open_ & ~shadow
    rep['ortho_masks'] = dict(canopy_frac=round(float(can_d.mean()), 3), building_frac=round(float(bld_d.mean()), 3),
                              shadow_frac_of_open=round(float(shadow.sum() / max(open_.sum(), 1)), 3),
                              valid_frac=round(float(ovalid.mean()), 3))
    Mo_lf, _ = mgauss(Mo, ovalid, 0.5 / c)          # the ortho is handed over at 0.5 m
    # ---- panoramas, gated, low frequency --------------------------------- #
    xy, col, w = gated_projection(rep)
    j = np.clip(((xy[:, 0] - x0) / c).astype(int), 0, nx - 1); i = np.clip(((xy[:, 1] - y0) / c).astype(int), 0, ny - 1)
    flat = i * nx + j
    A = np.zeros((ny * nx, 3), 'f8'); Wc = np.bincount(flat, weights=w, minlength=ny * nx)
    for q in range(3):
        A[:, q] = np.bincount(flat, weights=col[:, q] * w, minlength=ny * nx)
    pm = (Wc > 0).reshape(ny, nx)
    Ap = (A / np.maximum(Wc, 1e-9)[:, None]).reshape(ny, nx, 3).astype('f4')
    Mp, sup = mgauss(Ap, pm, PANO_SIGMA_M / c)
    pvalid = sup > PANO_MIN_WEIGHT * ndimage.gaussian_filter(np.ones_like(sup), PANO_SIGMA_M / c)
    # calibrate the ortho to the panoramas where both measure open ground.  Gain only, no
    # haze offset: a 20/80-percentile offset fit can drive a channel to zero on a large
    # share of the open lawn cells (near-black and off-hue tint, a purple curb fringe and
    # path noise).  The luma level comes from the panoramas (a raw ortho can sit well
    # above them), the hue stays the ortho's within +-CAL_CHROMA_TOL per channel.
    both = pvalid & ovalid
    gcal = np.ones(3, 'f4'); bcal = np.zeros(3, 'f4')
    if both.sum() > 2000:
        gl = float(np.median(Mp[both] @ LW) / max(np.median(Mo_lf[both] @ LW), 1e-6))
        gch = np.median(Mp[both], 0) / np.maximum(np.median(Mo_lf[both], 0), 1e-6)
        gcal = (gl * np.clip(gch / gl, 1 - CAL_CHROMA_TOL, 1 + CAL_CHROMA_TOL)).astype('f4')
    Mo_lf = (Mo_lf * gcal).astype('f4')
    rep['ortho_calibration'] = dict(gain_rgb=[round(float(v), 4) for v in gcal], haze_offset=[round(float(v), 4) for v in bcal],
                                    cells_both=int(both.sum()), pano_valid_frac=round(float(pvalid.mean()), 3),
                                    median_rel_luma_diff=round(float(np.median(((Mo_lf[both] - Mp[both]) @ LW) /
                                                                                np.maximum(Mp[both] @ LW, 1e-4))), 4) if both.any() else None)
    # ---- tint_map (panorama, filled), blend, rasters --------------------- #
    seed = pvalid | ovalid
    base = np.where(pvalid[..., None], Mp, Mo_lf)
    Mp_full = cascade_fill(base, seed)
    Mo_full = np.where(ovalid[..., None], Mo_lf, Mp_full).astype('f4')
    own = ORTHO_OWN * ndimage.gaussian_filter(ovalid.astype('f4'), 0.5 / c)
    own = np.where(pvalid, own, np.where(ovalid, 1.0, 0.0)).astype('f4')
    Mb = ((1 - own)[..., None] * Mp_full + own[..., None] * Mo_full).astype('f4')
    garr = np.array([x0, y0, c, nx, ny], 'f8')
    np.savez_compressed(FLOOR + '/tint_map.npz', M=Mp_full.astype('f4'), grid=garr, aligned=np.array(True))
    np.savez_compressed(FLOOR + '/tint_map_blend.npz', M=Mb, grid=garr)
    # okU / okS: the lit and cast-shadow ortho cells; s22 shade_fill
    # (GROUND_SHADE_FILL lit_ref) reads okU.  Here lit = open and not cast shadow (= ovalid).
    np.savez_compressed(ORTHO + '/rasters.npz', Mo=Mo_full, ovalid=ovalid, okU=ovalid, okS=open_ & shadow,
                        own=own, wmask=wm, grid=garr)
    for fp in _SNAP_FILES:                             # a fresh tint: the classes step re-snapshots it
        if os.path.exists(_pre(fp)):
            os.remove(_pre(fp))
    # ---- face colours for s21 / s22 -------------------------------------- #
    z = np.load(C.W + '/layers/ground_terrain.npz'); t = z['pos'].astype('f8')[z['idx']]
    bary = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    fc = np.zeros((len(t), 3), 'f8'); fv = np.zeros(len(t), 'f8')
    for bw in bary:
        p = (t * bw[None, :, None]).sum(1)
        jj = np.clip(((p[:, 0] - x0) / c).astype(int), 0, nx - 1); ii = np.clip(((p[:, 1] - y0) / c).astype(int), 0, ny - 1)
        fc += Mb[ii, jj]; fv += seed[ii, jj]
    fc /= len(bary); seen = fv / len(bary) >= 0.5
    pf = PJ + '/ground_terrain_face.npz'; keep = PJ + '/ground_terrain_face_s20.npz'
    if 'nomesh_tint' not in np.load(pf).files:      # a fresh s20 output: keep it as the source
        os.replace(pf, keep)
    d = np.load(keep)
    nv = np.where(seen, np.maximum(d['nviews'], 3.0), 0.0).astype('f4')
    np.savez_compressed(pf, col=fc.astype('f4'), nviews=nv, mad=np.zeros(len(t), 'f4'), area=d['area'],
                        nomesh_tint=np.array(True))
    s20c = d['col'][d['nviews'] > 0] @ LW
    rep['face_colour'] = dict(faces=len(t), seen_frac=round(float(seen.mean()), 3),
                              luma_p50_p97_s20=[round(float(np.percentile(s20c, q)), 3) for q in (50, 97)],
                              luma_p50_p97_new=[round(float(np.percentile(fc[seen] @ LW, q)), 3) for q in (50, 97)])
    rep['seconds'] = round(time.time() - t0)
    json.dump(rep, open(FLOOR + '/nomesh_tint.json', 'w'), indent=1)
    C.note('nomesh_tint', rep)
    print(json.dumps(rep, indent=1))


def cmd_classes():
    """Class islands + class_map.  Rewrites material/ground_terrain_mat.npz 'mat' in place
    (the s21 original is kept as ground_terrain_mat_s21.npz)."""
    import polish_lib as PL
    from s21_material import MATERIALS
    t0 = time.time(); rep = {}
    mp = MT + '/ground_terrain_mat.npz'; keep = MT + '/ground_terrain_mat_s21.npz'
    if 'nomesh_classes' not in np.load(mp).files:   # a fresh s21 output: keep it as the source
        import shutil; shutil.copy2(mp, keep)
    m = dict(np.load(keep)); mat = m['mat'].astype('i1').copy(); src = m['src']
    z = np.load(C.W + '/layers/ground_terrain.npz'); P = z['pos'].astype('f8'); F = z['idx'].astype('i8')
    t = P[F]; ar = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    # face adjacency over shared edges
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e.sort(1)
    fid = np.tile(np.arange(len(F)), 3)
    key = e[:, 0] * (len(P) + 1) + e[:, 1]; o = np.argsort(key); ks = key[o]; fo = fid[o]
    same = ks[1:] == ks[:-1]; a_, b_ = fo[:-1][same], fo[1:][same]
    import scipy.sparse as sp
    import scipy.sparse.csgraph  # noqa: F401
    adj = sp.coo_matrix((np.ones(len(a_) * 2), (np.r_[a_, b_], np.r_[b_, a_])), shape=(len(F), len(F))).tocsr()
    changed = 0; rounds = 0
    for rounds in range(1, 6):
        # connected same-class components
        rr, cc = adj.nonzero(); keepe = mat[rr] == mat[cc]
        g = sp.coo_matrix((np.ones(keepe.sum()), (rr[keepe], cc[keepe])), shape=adj.shape)
        ncomp, lab = sp.csgraph.connected_components(g, directed=False)
        carea = np.bincount(lab, weights=ar, minlength=ncomp)
        # a component may be reassigned only if every face in it was decided by photo/fallback
        movable = np.bincount(lab, weights=np.isin(src, ISLAND_SRC).astype('f8'), minlength=ncomp) == np.bincount(lab, minlength=ncomp)
        small = (carea < MIN_ISLAND_M2) & movable
        cand = np.nonzero(small[lab])[0]
        if not len(cand):
            break
        # neighbour class vote, area-weighted, from faces outside the component
        rr, cc = adj[cand].nonzero(); fr = cand[rr]
        outside = lab[cc] != lab[fr]
        vote = np.zeros((ncomp, len(MATERIALS)), 'f8')
        np.add.at(vote, (lab[fr[outside]], mat[cc[outside]]), ar[cc[outside]])
        best = vote.argmax(1); hasv = vote.sum(1) > 0
        mv = cand[hasv[lab[cand]]]
        newc = best[lab[mv]].astype('i1'); ch_ = newc != mat[mv]
        if not ch_.any():
            break
        mat[mv[ch_]] = newc[ch_]; changed += int(ch_.sum())
    before = np.bincount(m['mat'].astype(int), weights=ar, minlength=len(MATERIALS))
    after = np.bincount(mat.astype(int), weights=ar, minlength=len(MATERIALS))
    rep['islands'] = dict(min_island_m2=MIN_ISLAND_M2, reassignable_src=list(ISLAND_SRC), faces_changed=changed,
                          area_changed_m2=round(float(ar[mat != m['mat']].sum()), 1), rounds=rounds,
                          class_area_before={MATERIALS[k]: round(float(before[k]), 1) for k in range(len(MATERIALS)) if before[k] > 0},
                          class_area_after={MATERIALS[k]: round(float(after[k]), 1) for k in range(len(MATERIALS)) if after[k] > 0})
    m['mat'] = mat; m['nomesh_classes'] = np.array(True)
    np.savez_compressed(mp, **m)
    # ---- class map: top surface class per 0.25 m cell (water over terrain) --- #
    d = np.load(FLOOR + '/tint_map_blend.npz'); x0, y0, c, nx, ny = [float(v) for v in d['grid']]; nx, ny = int(nx), int(ny)
    cls = np.full((ny, nx), -1, 'i1'); srcm = np.full((ny, nx), -1, 'i1'); K = len(MATERIALS)
    for name, sid in (('ground_terrain', 0), ('water', 3)):
        fp = MT + f'/{name}_mat.npz'
        if not os.path.exists(fp):
            continue
        mm = np.load(fp) if name != 'ground_terrain' else m
        zz = np.load(C.W + f'/layers/{name}.npz'); tt = zz['pos'].astype('f8')[zz['idx']]
        s = (~mm['kill']) & (mm['up'] > 0.70); ix = np.nonzero(s)[0]
        vals = np.zeros((len(ix), K), 'f8'); vals[np.arange(len(ix)), np.clip(mm['mat'][ix], 0, K - 1)] = 1.0
        acc, den = PL.rasterise_tris(tt[ix], vals, nx, ny, x0, y0, c)
        have = den > 1e-9
        cls[have] = acc[have].argmax(1); srcm[have] = sid
    np.savez_compressed(POL + '/class_map.npz', cls=cls, src=srcm)
    _snapshot_or_restore()                         # both paths start from the tint step's own rasters
    ct = _stage_knob('GROUND_COLOUR_TRANSFER', GROUND_COLOUR_TRANSFER)
    if ct and ct.get('enabled', True):
        rep['colour_transfer'] = colour_transfer(cls, MATERIALS, ct)
    else:
        rep['pano_cells_clamped_to_ortho'] = clamp_pano_cells(cls, MATERIALS)
    rep['class_map_cells'] = int((cls >= 0).sum()); rep['seconds'] = round(time.time() - t0)
    json.dump(rep, open(POL + '/nomesh_classes.json', 'w'), indent=1)
    C.note('nomesh_classes', rep)
    print(json.dumps(rep, indent=1))


def clamp_pano_cells(cls, MATERIALS):
    """Cells the ortho could not own (canopy, building pad, cast shadow) carry the gated
    panorama colour, which in poses mode is still shade- and cast-prone: such cells come out
    near-black or off-hue several times as often as the ortho-owned cells.  Per class, hold those cells to the class's own ortho
    statistics: luma inside the ortho's [p10, p95], each channel's RGB/luma ratio inside the
    ortho's [p5, p95].  Rewrites tint_map, tint_map_blend and rasters Mo in place (before s50)."""
    LW = T.LUMW.astype('f4'); r = dict(np.load(ORTHO + '/rasters.npz')); ov = r['ovalid'] > 0
    files = {'Mo': (ORTHO + '/rasters.npz', r)}
    for key, fp in (('tm', FLOOR + '/tint_map.npz'), ('tb', FLOOR + '/tint_map_blend.npz')):
        files[key] = (fp, dict(np.load(fp)))
    out = {}
    for k, mname in enumerate(MATERIALS):
        own = (cls == k) & ov; tgt = (cls == k) & ~ov
        if own.sum() < 400 or not tgt.any():
            continue
        Ro = r['Mo'][own]; Lo = np.maximum(Ro @ LW, 1e-5)
        l_lo, l_hi = np.percentile(Lo, [10, 95]); rat = Ro / Lo[:, None]
        r_lo, r_hi = np.percentile(rat, 5, axis=0), np.percentile(rat, 95, axis=0)
        n_l = n_c = 0
        for key, (fp, d) in files.items():
            M = d['M' if key != 'Mo' else 'Mo'].astype('f4'); A = M[tgt]
            L = np.maximum(A @ LW, 1e-5); L2 = np.clip(L, l_lo, l_hi)
            q = np.clip(A / L[:, None], r_lo, r_hi); B = q * L2[:, None]
            B *= (L2 / np.maximum(B @ LW, 1e-5))[:, None]       # keep the clamped luma after the hue clamp
            if key == 'tm':
                n_l = int((np.abs(L2 - L) > 1e-6).sum()); n_c = int((np.abs(q - A / L[:, None]) > 1e-6).any(1).sum())
            M[tgt] = B; d['M' if key != 'Mo' else 'Mo'] = M
        out[mname] = dict(cells=int(tgt.sum()), luma_range=[round(float(l_lo), 4), round(float(l_hi), 4)],
                          luma_clamped=n_l, hue_clamped=n_c)
    for key, (fp, d) in files.items():
        np.savez_compressed(fp, **d)
    return out


_PRE_FILES = (ORTHO + '/rasters.npz', FLOOR + '/tint_map.npz', FLOOR + '/tint_map_blend.npz')
_SNAP_FILES = _PRE_FILES + (PJ + '/ground_terrain_face.npz',)


def _pre(fp):
    return fp[:-4] + '_pre_classes.npz'


def _snapshot_or_restore():
    """The classes step rewrites the tint rasters in place.  Keep the tint step's output as
    *_pre_classes.npz the first time and restore it on every rerun, so a rerun (or a switch
    of GROUND_COLOUR_TRANSFER) never compounds on its own output.  cmd_tint drops the copies."""
    import shutil
    for fp in _SNAP_FILES:
        if os.path.exists(_pre(fp)):
            shutil.copy2(_pre(fp), fp)
        else:
            shutil.copy2(fp, _pre(fp))


def _sat(lin):
    s = np.power(np.clip(lin, 0, 1), 1 / 2.2); mx = s.max(-1)
    return (mx - s.min(-1)) / np.maximum(mx, 1e-3)


def colour_transfer(cls, MATERIALS, cfg):
    """give every cell of a ground class its class's photographed colour.

    1. hue from the ortho itself: dark-object haze removal (per channel, the haze_pct
       percentile of the covered tile, no channel below haze_keep_min of its value), then
       one luma gain onto the tint step's ortho level (median over ovalid), so the
       panorama calibration sets brightness only;
    2. cast-shadow cells (okS) de-shadowed per class: their luma times the ratio of the
       class's lit to shadow log-luma mean within shadow_sigma_m (class-wide where the local
       support is under min_support, capped at shadow_gain_max), with the hue of the
       class's lit cells (cascade fill of their chromaticity), since shadow hue is skylight;
    3. lit cells keep each raster's own luma and take the ortho's chromaticity; shadow
       cells take the de-shadowed colour; every other cell of the class (canopy, building
       pad, no ortho) takes a cascade fill of the class's seen cells only, luma held to
       the lit cells' [luma_pct] range.
    Writes the same class colour into rasters Mo, tint_map and tint_map_blend (before s50)
    and resamples the ground faces' colour (s22 tile means) from the new blend."""
    LW = T.LUMW.astype('f4')
    files = {k: (fp, dict(np.load(fp))) for k, fp in zip(('Mo', 'M', 'Mb'), _PRE_FILES)}
    r = files['Mo'][1]; ov = r['ovalid'] > 0; okS = r['okS'] > 0
    x0, y0, c, nx, ny = [float(v) for v in r['grid']]; nx, ny = int(nx), int(ny)
    rep = dict(cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()})
    Mo_raw, cov = ortho_raster((x0, y0, c, nx, ny), rep)
    d = np.percentile(Mo_raw[cov], cfg['haze_pct'], axis=0).astype('f4')
    Mo_dh = np.maximum(Mo_raw - d, cfg['haze_keep_min'] * Mo_raw)
    lit_lf, _ = mgauss(Mo_dh, ov, 0.5 / c)
    sh_lf, _ = mgauss(Mo_dh, okS, 0.5 / c)
    O = np.where(ov[..., None], lit_lf, sh_lf).astype('f4')
    Mo_pre = r['Mo'].astype('f4')
    g = float(np.median(Mo_pre[ov] @ LW) / max(np.median(O[ov] @ LW), 1e-6)) if ov.sum() > 400 else 1.0
    O *= g
    rep['haze_dark_level_lin'] = [round(float(v), 4) for v in d]
    rep['luma_gain'] = round(g, 4)
    out = {}
    new = {k: v[1]['M' if k != 'Mo' else 'Mo'].astype('f4').copy() for k, v in files.items()}
    sat_before = {}
    s_sig = cfg['shadow_sigma_m'] / c
    for k, mname in enumerate(MATERIALS):
        if mname in cfg['skip_classes']:
            continue
        cm = cls == k; own = cm & ov; sh = cm & okS & ~ov
        if own.sum() < 400:
            continue
        sat_before[mname] = round(float(np.median(_sat(files['Mb'][1]['M'][cm].astype('f4')))), 3)
        Lown = np.maximum(O[own] @ LW, 1e-5); l_lo, l_hi = np.percentile(Lown, cfg['luma_pct'])
        # -- de-shadow ------------------------------------------------------- #
        S = np.zeros((ny, nx, 3), 'f4'); seen = own.copy(); gmed = None
        if sh.sum() >= 50:
            # the shadow's brightness pattern, levelled by the local in-class lit/shadow
            # log-luma ratio; its hue is skylight (Tucson concrete: R 0.026 vs G 0.044 in
            # shadow, 0.274 / 0.257 lit), so the hue is the class's lit chromaticity
            lo = np.log(np.maximum(O @ LW, 1e-5))
            ll, sl_ = mgauss(lo, own, s_sig); ls, ss_ = mgauss(lo, sh, s_sig)
            gglob = float(np.median(lo[own]) - np.median(lo[sh]))
            okl = (sl_ > cfg['min_support']) & (ss_ > cfg['min_support'])
            gl = np.clip(np.where(okl, ll - ls, gglob), 0.0, math.log(cfg['shadow_gain_max']))
            L = np.clip(np.exp(lo[sh] + gl[sh]), l_lo, l_hi)
            Q = cascade_fill(O / np.maximum(O @ LW, 1e-5)[..., None], own)
            D = Q[sh] * L[:, None]
            D *= (L / np.maximum(D @ LW, 1e-5))[:, None]
            S[sh] = D; seen |= sh; gmed = round(float(np.exp(np.median(gl[sh]))), 3)
        # -- lit: each raster keeps its luma, takes the ortho's chromaticity -- #
        q = O[own] / np.maximum(O[own] @ LW, 1e-5)[:, None]
        S[own] = O[own]
        # -- unseen cells of the class: fill from the class's seen cells ------- #
        un = cm & ~seen
        if un.any():
            F = cascade_fill(S, seen)[un]
            L = np.maximum(F @ LW, 1e-5); F *= (np.clip(L, l_lo, l_hi) / L)[:, None]
        for key in new:
            A = new[key]
            A[own] = q * np.maximum(A[own] @ LW, 1e-5)[:, None]
            if sh.any() and gmed is not None:
                A[sh] = S[sh]
            if un.any():
                A[un] = F
        out[mname] = dict(cells=int(cm.sum()), lit=int(own.sum()), shadow=int(sh.sum()), filled=int(un.sum()),
                          shadow_gain_median=gmed, luma_range=[round(float(l_lo), 4), round(float(l_hi), 4)],
                          sat_median_before=sat_before[mname],
                          sat_median_after=round(float(np.median(_sat(new['Mb'][cm]))), 3))
    for key, (fp, dd) in files.items():
        dd['M' if key != 'Mo' else 'Mo'] = new[key].astype('f4')
        np.savez_compressed(fp, **dd)
    rep['classes'] = out
    rep['faces_resampled'] = _resample_faces(new['Mb'], (x0, y0, c, nx, ny))
    rep['ground_refs'] = _ground_refs(cls, MATERIALS, O, Mo_raw, ov, c)
    return rep


def _ground_refs(cls, MATERIALS, O, Mo_raw, ov, c):
    """per-class references s22 reads when the area sets the stage knobs
    GROUND_SAT_CAP / GROUND_TILE_CONTRAST (s22 ignores the files otherwise):
      polish/nomesh_sat_cap.json       p95 linear saturation (max-min)/max of the lit cells (O)
      polish/nomesh_tile_contrast.json the ortho's finest-band log-luma sd on the lit cells
                                       (DoG lo_m..4 lo_m, lo_m = the ortho pixel, >= the cell),
                                       eroded 6 cells off the class edge"""
    import rasterio
    od = os.path.dirname(SCR) + '/ortho'
    with rasterio.open(os.path.join(od, sorted(p for p in os.listdir(od) if p.endswith('.tif'))[0])) as r:
        res = abs(r.transform.a) * (math.cos(math.radians(float(C.ORIGIN[0]))) if r.crs.to_epsg() == 3857 else 1.0)
    lo_m = max(float(res), c); hi_m = 4 * lo_m
    L = np.log(np.maximum(Mo_raw @ T.LUMW.astype('f4'), 1e-4))
    dog = ndimage.gaussian_filter(L, lo_m / c / 2) - ndimage.gaussian_filter(L, hi_m / c / 2)
    sat_cap, tcon = {}, {}
    for k, m in enumerate(MATERIALS):
        own = (cls == k) & ov
        if m == 'water' or own.sum() < 400:
            continue
        A = O[own]; mx = A.max(1)
        sat_cap[m] = round(float(np.percentile((mx - A.min(1)) / np.maximum(mx, 1e-6), 95)), 4)
        e = ndimage.binary_erosion(own, iterations=6)
        if e.sum() >= 500:
            tcon[m] = dict(ortho_band_sd=round(float(dog[e].std()), 4), lo_m=round(lo_m, 3), hi_m=round(hi_m, 3), cells=int(e.sum()))
    json.dump(sat_cap, open(POL + '/nomesh_sat_cap.json', 'w'), indent=1)
    json.dump(tcon, open(POL + '/nomesh_tile_contrast.json', 'w'), indent=1)
    return dict(sat_cap_p95_lin=sat_cap, tile_contrast_ref=tcon)


def _resample_faces(Mb, grid):
    """cmd_tint's face colour, read again off the transferred blend (col only; nviews kept)."""
    x0, y0, c, nx, ny = grid
    pf = PJ + '/ground_terrain_face.npz'; d = dict(np.load(pf))
    z = np.load(C.W + '/layers/ground_terrain.npz'); t = z['pos'].astype('f8')[z['idx']]
    fc = np.zeros((len(t), 3), 'f8')
    for bw in np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]):
        p = (t * bw[None, :, None]).sum(1)
        jj = np.clip(((p[:, 0] - x0) / c).astype(int), 0, nx - 1); ii = np.clip(((p[:, 1] - y0) / c).astype(int), 0, ny - 1)
        fc += Mb[ii, jj]
    d['col'] = (fc / 4).astype('f4')
    np.savez_compressed(pf, **d)
    return len(t)


def cmd_ortho_enu(gsd=0.0762):
    """W/ortho/ortho_enu.tif + .json, the ENU-gridded ortho that objects/objaerial (s84's
    aerial car/boat pass) reads.  RGBA uint8, alpha 0 = no tile; nearest sample of the
    ortho tiles at each pixel centre, at a 7.62 cm GSD over the area bbox in ENU
    (areacfg.enu_ext).  The tif has no CRS: the ENU affine lives in the json, which is all
    objaerial reads."""
    import rasterio
    x0, y0, x1, y1 = C._areacfg.enu_ext(C)
    nx = int(math.ceil((x1 - x0) / gsd)); ny = int(math.ceil((y1 - y0) / gsd))
    od = os.path.dirname(SCR) + '/ortho'
    tiles = sorted(p for p in os.listdir(od) if p.endswith('.tif'))
    out = np.zeros((4, ny, nx), 'u1')
    for r0 in range(0, ny, 512):                        # row bands keep memory small
        rows = np.arange(r0, min(ny, r0 + 512))
        X, Y = np.meshgrid(x0 + (np.arange(nx) + 0.5) * gsd, y1 - (rows + 0.5) * gsd)
        lat, lon, _ = C.enu_to_lla(X.ravel(), Y.ravel(), 0.0, o=C.ORIGIN)
        band = np.zeros((4, X.size), 'u1')
        for tp in tiles:
            with rasterio.open(os.path.join(od, tp)) as r:
                vv, ok = _tile_sample(r, lon, lat, coarse_m=gsd)
                ok &= (band[3] == 0)
                if not ok.any():
                    continue
                v = np.clip(np.round(vv[ok].T), 0, 255).astype('u1')
                good = v.max(0) > 0
                idx = np.nonzero(ok)[0][good]
                band[:3, idx] = v[:, good]; band[3, idx] = 255
        out[:, rows, :] = band.reshape(4, len(rows), nx)
    os.makedirs(C.W + '/ortho', exist_ok=True)
    tif, js = C.W + '/ortho/ortho_enu.tif', C.W + '/ortho/ortho_enu.json'
    from affine import Affine
    with rasterio.open(tif, 'w', driver='GTiff', width=nx, height=ny, count=4, dtype='uint8',
                       transform=Affine(gsd, 0, x0, 0, -gsd, y1), compress='deflate') as w:
        w.write(out)
    json.dump(dict(crs_pixels_are='model ENU tangent plane at common.ORIGIN %s' % list(C.ORIGIN),
                   gsd_m=gsd, width=nx, height=ny, enu_affine=[gsd, 0.0, x0, 0.0, -gsd, y1],
                   enu_extent=dict(xmin=x0, xmax=x0 + nx * gsd, ymin=y1 - ny * gsd, ymax=y1),
                   model_extent=dict(xmin=x0, xmax=x1, ymin=y0, ymax=y1),
                   source='pipeline/nomesh_ground.py ortho-enu from ' + od,
                   coverage=round(float((out[3] > 0).mean()), 4)), open(js, 'w'), indent=1)
    print(f'ortho-enu: {nx} x {ny} px at {gsd} m, coverage {(out[3] > 0).mean():.3f} -> {tif}')


if __name__ == '__main__':
    {'tint': cmd_tint, 'classes': cmd_classes, 'ortho-enu': cmd_ortho_enu}[sys.argv[1]]()
