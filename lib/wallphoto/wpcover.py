"""Stage 60 step 1 -- which panorama can see which patch of which rect.

Geometry and the transient masks only: no panorama is decoded here.  For every
(rect, panorama) pair it walks a 0.3 m lattice over the rect and records, per
1.5 m patch, how much of it that panorama sees unoccluded, front-facing, inside
the range gate, outside its own sky boundary and outside every transient
detection.  That table is the input both to the alignment (which views are
worth correlating) and to the ownership (which view wins each patch).
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import texlib as T
from wallphoto import wpcommon as H
from wallphoto.wpsample import ViewCtx, sample_rect, candidates

COVER_CELL = 0.30

_G = {}


def _init(setname, poses, pids, bypano):
    rects, _ = H.load_rects(setname)
    _G['rects'] = {r.id: r for r in rects}
    _G['poses'] = poses
    _G['pids'] = pids
    _G['bypano'] = bypano
    _G['dsm'] = T.DSM()


def _one(k):
    pid = _G['pids'][k]
    rids = _G['bypano'].get(k, [])
    if not rids:
        return None
    try:
        ctx = ViewCtx(pid, _G['poses'][pid], _G['dsm'])
    except Exception:
        return None
    R = []; V = []; PA = []; NV = []; NT = []; CS = []; RG = []
    for rid in rids:
        rc = _G['rects'][rid]
        cell = min(COVER_CELL, max(rc.width, rc.height) / 3.0)
        uu, vv = H.grid_axes(rc, cell)
        res = sample_rect(ctx, rc, uu, vv, rgb=False, veto_green=False)
        val = res['valid'].ravel()
        if not val.any():
            continue
        npu, npv = H.patch_shape(rc)
        UU, VV = np.meshgrid(uu, vv)
        pidx = H.patch_of_uv(rc, UU.ravel(), VV.ravel(), npu, npv)
        npatch = npu * npv
        nt = np.bincount(pidx, minlength=npatch)
        nv = np.bincount(pidx[val], minlength=npatch)
        cs = np.bincount(pidx[val], weights=res['cos'].ravel()[val], minlength=npatch)
        rg = np.bincount(pidx[val], weights=res['rng'].ravel()[val], minlength=npatch)
        keep = np.nonzero(nv > 0)[0]
        R.append(np.full(len(keep), rid, 'i4'))
        V.append(np.full(len(keep), k, 'i4'))
        PA.append(keep.astype('i4'))
        NV.append(nv[keep].astype('i4'))
        NT.append(nt[keep].astype('i4'))
        CS.append((cs[keep] / nv[keep]).astype('f4'))
        RG.append((rg[keep] / nv[keep]).astype('f4'))
    if not R:
        return None
    return (np.concatenate(R), np.concatenate(V), np.concatenate(PA),
            np.concatenate(NV), np.concatenate(NT), np.concatenate(CS),
            np.concatenate(RG))


def run(setname, workers=20, spacing=1.0, only=None):
    t0 = time.time()
    D = H.setdir(setname)
    rects, doc = H.load_rects(setname)
    if only is not None:
        rects = [r for r in rects if r.id in only]
    poses, src = T.load_poses()
    sel = T.thin_poses(poses, spacing)
    pids = sorted(sel)
    H.log(f"cover[{setname}]: {len(rects)} rects, {len(pids)} panoramas "
          f"thinned to {spacing} m from {len(poses)} ({src[:40]})")
    cand = candidates(rects, sel, pids)
    bypano = {}
    npair = 0
    for rid, (vs, rr) in cand.items():
        for v in vs:
            bypano.setdefault(int(v), []).append(rid)
            npair += 1
    H.log(f"cover[{setname}]: {npair} (rect, panorama) pairs over "
          f"{len(bypano)} panoramas")
    import multiprocessing as mp
    acc = []
    todo = sorted(bypano)
    with mp.Pool(workers, initializer=_init,
                 initargs=(setname, sel, pids, bypano)) as pool:
        for n, r in enumerate(pool.imap_unordered(_one, todo, chunksize=4)):
            if r is not None:
                acc.append(r)
            if (n + 1) % 200 == 0:
                H.log(f"  {n+1}/{len(todo)} panoramas, {time.time()-t0:.0f}s")
    cols = [np.concatenate([a[i] for a in acc]) for i in range(7)]
    rect, view, patch, nval, ntot, cosm, rngm = cols
    np.savez_compressed(D + '/work/cover.npz', rect=rect, view=view, patch=patch,
                        nval=nval, ntot=ntot, cos=cosm, rng=rngm,
                        pids=np.array(pids, dtype='U24'))
    # ---- report ----------------------------------------------------------- #
    byid = {r.id: r for r in rects}
    per = {}
    for rid in np.unique(rect):
        m = rect == rid
        rc = byid[int(rid)]
        npu, npv = H.patch_shape(rc)
        vseen = np.unique(view[m])
        pseen = np.unique(patch[m])
        # patch coverage: a patch counts as covered when some view sees half of it
        frac = nval[m] / np.maximum(ntot[m], 1)
        good = patch[m][frac >= 0.5]
        per[int(rid)] = dict(views=int(len(vseen)),
                             patches=int(npu * npv),
                             patches_seen=int(len(pseen)),
                             patches_half_covered=int(len(np.unique(good))),
                             best_range_m=round(float(rngm[m].min()), 2))
    A = np.array([byid[r].width * byid[r].height for r in per])
    rep = dict(set=setname, rects=len(rects), rects_with_a_view=len(per),
               pairs=int(npair), rows=int(len(rect)),
               patch_m=H.PATCH_M, cover_cell_m=COVER_CELL,
               r_max_m=H.R_MAX, r_max_far_m=H.R_MAX_FAR,
               far_gate_rects=int(sum(1 for r in rects if H.rect_rmax(r) > H.R_MAX)),
               min_cos=H.MIN_COS, spacing_m=spacing,
               rect_area_with_a_view_m2=round(float(A.sum()), 1),
               patches_half_covered=int(sum(v['patches_half_covered'] for v in per.values())),
               patches_total=int(sum(H.patch_shape(r)[0] * H.patch_shape(r)[1]
                                     for r in rects)),
               seconds=round(time.time() - t0), per_rect=per)
    json.dump(rep, open(D + '/work/cover.json', 'w'), indent=1)
    H.mark(setname, 'cover', {k: v for k, v in rep.items() if k != 'per_rect'})
    H.log(f"cover[{setname}]: {rep['rects_with_a_view']} rects, "
          f"{rep['patches_half_covered']}/{rep['patches_total']} patches half "
          f"covered, {rep['seconds']}s")
    return rep
