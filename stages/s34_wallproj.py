"""Stage 34 -- project the posed panoramas onto every building wall plane.

A wall is a plane, and a plane is where the ~1 m panorama pose class costs the
least: a metre of translation error along the line of sight slides the texture
sideways by a metre at most, and the multi-view median removes whatever does
not agree.  The same machinery the ground pass uses (texlib.PanoSampler's range
gate, DSM line-of-sight and sky mask, stage 19's per-panorama delighting gain,
stage 20's lit-colour statistic) is applied per texel on the wall plane instead
of per face.

Per wall it writes the median colour, the number of views, the multi-view
agreement (MAD of luminance) and a coverage mask, at 2-5 cm per texel.
"""
import os, sys, json, math, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
import common as C
import texlib as T
from s20_project import lit_colour
_FQ = None
if hasattr(C, '_areacfg') and C._areacfg.build_area_cfg(C) is not None:     # only when an area is set
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pipeline'))
    import facade_quality as _FQ
    if not (_FQ.knob() or {}):
        _FQ = None

FAC = T.SCR + '/facades'
PROJ = FAC + '/proj'
PROJ_FAR = PROJ + '/far'      # the far colour/rhythm grid
os.makedirs(PROJ, exist_ok=True)

R_MAX = 40.0            # panoramas within 40 m
KSLOTS = 6
SAMPLE_W = 4096         # source panoramas are 5760 px wide
MAX_DIM = 2048
MIN_COS = 0.20          # front-facing enough to carry texture
BACK_OFF = 1.25         # metres to step back off the wall before the DSM test.
                        # a larger step-back (3 m) clears trees next to a
                        # building and paints foliage onto walls


def texel_size(wall, dq):
    """Texel size from the distance the wall was actually photographed from.

    dq is the 25th percentile of the candidate camera ranges, not the median: a
    wall one camera stood 8 m from carries 2 cm of real information, and the
    source panoramas resolve 1.5 mrad, which is 1.2 cm at 8 m and 3 cm at 20 m.
    Walls more than 120 m from the middle of the area are coarsened, because
    nobody walks up to them, and a tile is capped at 2048 px."""
    t = float(np.clip(0.0025 * dq, 0.02, 0.05))
    r = float(np.hypot(wall['centre'][0], wall['centre'][1]))
    t *= 1.0 + max(0.0, (r - 120.0) / 120.0)
    return max(t, max(wall['width_m'], wall['height_m']) / MAX_DIM)


_G = {}


def _init(poses, gains):
    _G['poses'] = poses; _G['gains'] = gains
    _G['dsm'] = T.DSM()


# the per-wall cache is named by wall id, and wall ids are NOT stable -- s33
# numbers walls in building order, so adding a building renumbers every later
# wall.  A cache file is trusted only if it carries this wall's geometric
# signature; anything else is stale.
def wall_sig(w):
    return '%s|%s|%s|%.2f|%.2f' % (
        w['part_name'], ','.join('%.2f' % x for x in w['origin']),
        ','.join('%.4f' % x for x in w['u']), w['width_m'], w['height_m'])


def cache_ok(w):
    fp = f"{PROJ}/wall_{w['id']:04d}.npz"
    if not os.path.exists(fp):
        return False
    try:
        m = json.loads(str(np.load(fp, allow_pickle=True)['meta']))
    except Exception:
        return False
    if _FQ is not None and m.get('fq') != _fq_sig():
        return False                  # made under another quality knob
    return m.get('sig') == wall_sig(w)


def _fq_sig():
    return json.dumps(_FQ.knob(), sort_keys=True) if _FQ is not None else None


def purge_stale(walls, all_ids):
    n = 0
    for w in walls:
        fp = f"{PROJ}/wall_{w['id']:04d}.npz"
        if os.path.exists(fp) and not cache_ok(w):
            os.remove(fp); n += 1
    for f in os.listdir(PROJ):
        if f.startswith('wall_') and f.endswith('.npz') and int(f[5:9]) not in all_ids:
            os.remove(os.path.join(PROJ, f)); n += 1
    return n


def wall_grid(w, tx):
    nu_ = int(max(2, round(w['width_m'] / tx)))
    nv_ = int(max(2, round(w['height_m'] / tx)))
    O = np.array(w['origin'], 'f8'); U = np.array(w['u'], 'f8')
    N = np.array(w['normal'], 'f8'); V = np.array([0., 0., 1.])
    uu = (np.arange(nu_) + 0.5) * (w['width_m'] / nu_)
    vv = (np.arange(nv_) + 0.5) * (w['height_m'] / nv_)
    UU, VV = np.meshgrid(uu, vv)                      # row 0 = bottom of wall
    P = (O[None, None, :] + U[None, None, :] * UU[:, :, None] +
         V[None, None, :] * VV[:, :, None] + N[None, None, :] * 0.06)
    return P.reshape(-1, 3).astype('f4'), nu_, nv_, N, U


def _part(job):
    pidx, walls, cands = job
    walls = [w for w in walls
             if not os.path.exists(f"{PROJ}/wall_{w['id']:04d}.npz")]
    if not walls:
        return pidx, []
    poses = _G['poses']; gains = _G['gains']; dsm = _G['dsm']
    out = []
    grids = {}
    for w in walls:
        dq = (float(np.percentile(cands[w['id']][1], 25))
              if len(cands[w['id']][0]) else 20.0)
        tx = texel_size(w, dq)
        P, nu_, nv_, N, U = wall_grid(w, tx)
        grids[w['id']] = dict(P=P, nu=nu_, nv=nv_, N=N, tx=tx,
                              Wk=np.zeros((len(P), KSLOTS), 'f4'),
                              Ck=np.zeros((len(P), KSLOTS, 3), 'f4'),
                              cnt=np.zeros(len(P), 'i4'),
                              rmin=np.full(len(P), 1e6, 'f4'), nlook=0, nseg=0)
    # invert: pano -> walls that want it
    bypano = {}
    for w in walls:
        for pid in cands[w['id']][0]:
            bypano.setdefault(pid, []).append(w['id'])
    for pid, wids in bypano.items():
        p = poses[pid]
        Cc = np.array([p['x'], p['y'], p['z']], 'f8')
        try:
            occ = T.Occluder(dsm, Cc, r_max=R_MAX + 15.0, back_off_m=BACK_OFF)
            img = T.pano_rgb(pid, SAMPLE_W)
        except Exception:
            continue
        sl, sv = T.skyline(pid)
        seg = _FQ.load(pid) if _FQ is not None else None     # not-wall mask
        g = gains.get(pid)
        tier = 1.0 if p['confidence'] == 'high' else 0.55
        for wid in wids:
            G = grids[wid]
            P = G['P']; N = G['N']
            d = P - Cc[None, :].astype('f4')
            r = np.linalg.norm(d, axis=1)
            dn = d / np.maximum(r[:, None], 1e-6)
            cosi = -(dn @ N.astype('f4'))            # camera in front of the wall
            ok = (cosi > (MIN_COS if _FQ is None else _FQ.min_cos(MIN_COS, 's34_min_cos'))) & (r > 1.5) & (r < R_MAX)
            if not ok.any():
                continue
            sel = np.nonzero(ok)[0]
            vis = occ.visible(P[sel])
            sel = sel[vis]
            if len(sel) == 0:
                continue
            lon, lat, rng = T.project(P[sel], Cc, p['R'])
            if sl is not None:
                col = np.clip(((lon / (2 * math.pi) + 0.5) * len(sl)).astype(np.int32),
                              0, len(sl) - 1)
                keep = ~(sv[col] & (lat > sl[col] - 0.02))
                sel = sel[keep]; lon = lon[keep]; lat = lat[keep]; rng = rng[keep]
            if seg is not None and len(sel):
                nw = _FQ.lookup(seg, lon, lat)
                G['nlook'] += int(len(sel)); G['nseg'] += int(nw.sum())
                if _FQ.vetoes():
                    keep = ~nw
                    sel = sel[keep]; lon = lon[keep]; lat = lat[keep]; rng = rng[keep]
            if len(sel) == 0:
                continue
            rgb = T.sample_equirect(img, lon, lat)
            lin = np.power(np.clip(rgb / 255.0, 0, 1), 2.2).astype('f4')
            if g is not None:
                lin = lin / g
            G['cnt'][sel] += 1
            np.minimum.at(G['rmin'], sel, rng.astype('f4'))
            wgt = (tier * cosi[sel] * (10.0 / (10.0 + rng))).astype('f4')
            Wk = G['Wk']; Ck = G['Ck']
            slot = np.argmin(Wk[sel], axis=1)
            cur = Wk[sel, slot]
            take = wgt > cur
            if not take.any():
                continue
            ii = sel[take]; ss = slot[take]
            Wk[ii, ss] = wgt[take]
            Ck[ii, ss] = lin[take]
        del img, occ
    for w in walls:
        G = grids[w['id']]
        col, nv, mad = lit_colour(G['Wk'], G['Ck'])
        # The median over six views is robust but soft: a 1 m pose error at
        # 15 m moves the projected point by up to a metre, so averaging views
        # smears a facade horizontally (it shows as vertical streaking).  Keep the single best-weighted view and the mean
        # of the best two as well, so the bake can trade sharpness against
        # robustness with the same projection.
        Wk = G['Wk']; Ck = G['Ck']
        order = np.argsort(-Wk, axis=1)
        rows = np.arange(len(Wk))[:, None]
        Ws = Wk[rows, order]; Cs = Ck[rows, order]
        best = Cs[:, 0]
        w2 = Ws[:, :2]
        top2 = ((Cs[:, :2] * w2[:, :, None]).sum(1) /
                np.maximum(w2.sum(1), 1e-9)[:, None])
        top2 = np.where((Ws[:, 1:2] > 0), top2, best)
        nu_, nv_ = G['nu'], G['nv']
        rec = dict(id=w['id'], part=pidx, tx=round(G['tx'], 4), nu=nu_, nv=nv_,
                   panos=int(len(cands[w['id']][0])),
                   sig=wall_sig(w))
        if _FQ is not None:
            rec['fq'] = _fq_sig()
        cnt = G['cnt']
        cov1 = float((cnt >= 1).mean()); cov3 = float((cnt >= 3).mean())
        m3 = cnt >= 3
        rec['coverage_any'] = round(cov1, 4)
        rec['coverage_ge3'] = round(cov3, 4)
        rec['views_median'] = float(np.median(cnt[cnt > 0])) if (cnt > 0).any() else 0.0
        rec['mad_L'] = round(float(np.median(mad[m3]) * 255), 2) if m3.any() else None
        rec['mad_L_p75'] = round(float(np.percentile(mad[m3], 75) * 255), 2) if m3.any() else None
        rec['mean_rgb'] = [round(float(x), 4) for x in col[m3].mean(0)] if m3.any() else None
        if G['nlook']:
            rec['nonwall_frac_looked'] = round(G['nseg'] / G['nlook'], 4)
        np.savez_compressed(f"{PROJ}/wall_{w['id']:04d}.npz",
                            col=col.reshape(nv_, nu_, 3).astype('f4'),
                            cnt=cnt.reshape(nv_, nu_).astype('u1'),
                            mad=mad.reshape(nv_, nu_).astype('f4'),
                            rmin=G['rmin'].reshape(nv_, nu_).astype('f4'),
                            best=best.reshape(nv_, nu_, 3).astype('f4'),
                            top2=top2.reshape(nv_, nu_, 3).astype('f4'),
                            meta=np.array(json.dumps(rec)))
        out.append(rec)
    return pidx, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--spacing', type=float, default=2.0)
    ap.add_argument('--only', default=None, help='comma-separated part indices')
    a = ap.parse_args()
    t0 = time.time()
    doc = json.load(open(FAC + '/walls.json'))
    walls = doc['walls']
    if a.only:
        keep = set(int(x) for x in a.only.split(','))
        walls = [w for w in walls if w['part'] in keep]
    nst = purge_stale(walls, set(int(w['id']) for w in doc['walls']))
    print(f"{nst} stale per-wall projection cache files removed (id/geometry signature)",
          flush=True)
    poses, src = T.load_poses()
    sel = T.thin_poses(poses, a.spacing)
    gains = {}
    gp = T.SCR + '/delight/gains.npz'
    if os.path.exists(gp):
        d = np.load(gp, allow_pickle=True)
        gains = {str(p): g for p, g in zip(d['pids'], d['gain'])}
    pid_list = list(sel.keys())
    Cx = np.array([[sel[p]['x'], sel[p]['y'], sel[p]['z']] for p in pid_list])
    print(f"{len(walls)} walls, {len(sel)} panoramas ({src[:60]}), gains {len(gains)}",
          flush=True)

    cands = {}
    for w in walls:
        if w.get('buried'):
            cands[w['id']] = ([], np.zeros(0))
            continue
        ctr = np.array(w['centre']); U = np.array(w['u']); N = np.array(w['normal'])
        d = Cx - ctr
        du = d @ U; dv = d[:, 2]; dn = d @ N
        cu = np.clip(du, -w['width_m'] / 2, w['width_m'] / 2)
        cv = np.clip(dv, -w['height_m'] / 2, w['height_m'] / 2)
        r = np.sqrt((du - cu) ** 2 + (dv - cv) ** 2 + dn ** 2)
        ok = (dn > 1.0) & (r < R_MAX)
        cands[w['id']] = ([pid_list[i] for i in np.nonzero(ok)[0]], r[ok])
    nc = np.array([len(cands[w['id']][0]) for w in walls])
    print(f"{int((nc>0).sum())} walls have a candidate view, "
          f"{int(nc.sum())} (wall, panorama) pairs", flush=True)

    jobs = {}
    for w in walls:
        if len(cands[w['id']][0]) == 0:
            continue
        jobs.setdefault(w['part'], []).append(w)
    joblist = [(k, v, {w['id']: cands[w['id']] for w in v})
               for k, v in sorted(jobs.items(), key=lambda kv: -sum(
                   len(cands[w['id']][0]) for w in kv[1]))]
    print(f"{len(joblist)} building parts to project", flush=True)

    import multiprocessing as mp
    recs = []
    with mp.Pool(a.workers, initializer=_init, initargs=(sel, gains),
                 maxtasksperchild=2) as pool:
        for pidx, out in pool.imap_unordered(_part, joblist):
            recs.extend(out)
            print(f"  part {pidx:2d} {len(out):3d} walls, {time.time()-t0:.0f}s",
                  flush=True)
    for w in walls:
        fp = f"{PROJ}/wall_{w['id']:04d}.npz"
        if os.path.exists(fp) and not any(r['id'] == w['id'] for r in recs):
            recs.append(json.loads(str(np.load(fp, allow_pickle=True)['meta'])))
    byid = {r['id']: r for r in recs}
    A = np.array([w['area_m2'] for w in walls])
    cov = np.array([byid.get(w['id'], {}).get('coverage_ge3', 0.0) for w in walls])
    mad = np.array([byid.get(w['id'], {}).get('mad_L') or 99.0 for w in walls])
    note = dict(walls=len(walls), projected=len(recs),
                panoramas=len(sel), pose_provenance=src,
                r_max_m=R_MAX, sample_width_px=SAMPLE_W, k_slots=KSLOTS,
                wall_area_m2=round(float(A.sum()), 1),
                area_with_cov3_over_50pct=round(float(A[cov > 0.5].sum()), 1),
                area_with_cov3_over_25pct=round(float(A[cov > 0.25].sum()), 1),
                mad_L_distribution={f'p{p}': round(float(np.percentile(mad[cov > 0.05], p)), 2)
                                    for p in (10, 25, 50, 75, 90)} if (cov > 0.05).any() else {},
                seconds=round(time.time() - t0), per_wall=recs)
    json.dump(note, open(FAC + '/wallproj.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in note.items() if k != 'per_wall'}, indent=1))
    far_pass(doc['walls'] if not a.only else walls, sel, gains, pid_list, Cx, a.workers)


def far_pass(walls, sel, gains, pid_list, Cx, workers):
    """The far grid: the same projection out to C.FACADE_FAR_R_M on a coarse
    C.FACADE_FAR_TEXEL_M grid, into proj/far/.  Stage 35 reads colour and storey/bay
    rhythm off it for walls the near projection leaves without a colour of their own;
    nothing pastes these texels."""
    global PROJ, R_MAX, texel_size
    rf = float(getattr(C, 'FACADE_FAR_R_M', 0) or 0)
    if rf <= 0:
        return
    keep = (PROJ, R_MAX, texel_size)
    tf = float(C.FACADE_FAR_TEXEL_M)
    PROJ = PROJ_FAR; R_MAX = rf
    texel_size = lambda w, dq: max(tf, max(w['width_m'], w['height_m']) / 512)
    try:
        os.makedirs(PROJ, exist_ok=True)
        nst = purge_stale(walls, set(int(w['id']) for w in walls))
        cands = {}
        for w in walls:
            if w.get('buried'):
                cands[w['id']] = ([], np.zeros(0)); continue
            ctr = np.array(w['centre']); U = np.array(w['u']); N = np.array(w['normal'])
            d = Cx - ctr
            du = d @ U; dv = d[:, 2]; dn = d @ N
            cu = np.clip(du, -w['width_m'] / 2, w['width_m'] / 2)
            cv = np.clip(dv, -w['height_m'] / 2, w['height_m'] / 2)
            r = np.sqrt((du - cu) ** 2 + (dv - cv) ** 2 + dn ** 2)
            ok = (dn > 1.0) & (r < rf)
            cands[w['id']] = ([pid_list[i] for i in np.nonzero(ok)[0]], r[ok])
        jobs = {}
        for w in walls:
            if cands[w['id']][0]:
                jobs.setdefault(w['part'], []).append(w)
        joblist = [(k, v, {w['id']: cands[w['id']] for w in v}) for k, v in jobs.items()]
        import multiprocessing as mp
        t0 = time.time(); n = 0
        with mp.Pool(workers, initializer=_init, initargs=(sel, gains),
                     maxtasksperchild=2) as pool:
            for pidx, out in pool.imap_unordered(_part, joblist):
                n += len(out)
        print(f"far pass: {nst} stale removed, {sum(len(v) for v in jobs.values())} walls "
              f"within {rf:.0f} m, {n} projected, {time.time()-t0:.0f}s", flush=True)
    finally:
        PROJ, R_MAX, texel_size = keep


if __name__ == '__main__':
    main()
