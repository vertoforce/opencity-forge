"""Facade completion (stage 60 atlas): repeat the photographed module across the unseen part of a wall.

module_fill(lin, got, tx) -> (fill, fmask, info)
  lin  (ny, nx, 3) linear photo, row 0 = wall foot
  got  (ny, nx) bool, texels the photograph owns
  tx   metres per texel
Horizontal: shift by k * P, P the bay period measured on the photograph; no period, no completion.
Vertical: rows no horizontal shift reaches take a copy k storeys lower (never the ground storey)
when a storey period is measured.  Pure numpy; no wall ids.
"""
import numpy as np

LUMW = np.array([0.2126, 0.7152, 0.0722], 'f4')


def vegetation(col):
    """s35_facadetex.vegetation: saturated green texels are foliage, not wall."""
    s = np.power(np.clip(col, 0, 1), 1 / 2.2)
    mx = s.max(-1); mn = s.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    g = s[..., 1]
    return (g >= s[..., 0] - 0.005) & (g >= s[..., 2] - 0.005) & (sat > 0.18)


def period(prof, ok, lo, hi, ac_min):
    """Autocorrelation period of a 1-D profile over its trusted samples, or None."""
    if ok.sum() < max(8, 2 * lo) or hi <= lo + 2:
        return None, 0.0
    idx = np.nonzero(ok)[0]
    p = np.interp(np.arange(len(prof)), idx, prof[idx])[idx[0]:idx[-1] + 1]
    p = p - p.mean()
    if p.std() < 1e-4 or len(p) < 2 * lo:
        return None, 0.0
    ac = np.correlate(p, p, 'full')[len(p) - 1:]
    ac = ac / max(ac[0], 1e-9)
    # unbiased: the overlap shrinks with the lag
    ac = ac * len(p) / np.maximum(len(p) - np.arange(len(ac)), 1)
    hi = min(hi, len(p) // 2)
    if hi <= lo + 2:
        return None, 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if lag <= lo or lag >= hi - 1:         # a range edge is not a period
        return None, float(ac[lag])
    return (lag, float(ac[lag])) if ac[lag] >= ac_min else (None, float(ac[lag]))


def _row_fill(lin, src, rest, win, gap, band_mad=None, band_step=None):
    """Bounded along-row completion: a run of hole texels in one row, with the
    photographed wall on both ends and no longer than gap texels, takes the linear
    blend from the median of the seen texels within win before it to the median
    within win after it.  Horizontal bands (plinth, sill, string course, cornice)
    continue through the hole, nothing is copied, and an open-ended run (the unseen
    end of a wall, a notch open to the roofline) stays material.  With band_mad the
    run must also sit in a band: the sRGB luma on each side varies by at most
    band_mad (median absolute deviation) and the two sides differ by at most
    band_step, so a row through windows is never smeared into stripes."""
    Y = np.power(np.clip(lin @ LUMW, 0, 1), 1 / 2.2)
    ny, nx = src.shape
    out = np.full((ny, nx, 3), np.nan, 'f4')
    for r in np.nonzero(rest.any(1))[0]:
        t = rest[r]
        d = np.diff(np.r_[0, t.astype(np.int8), 0]); starts = np.nonzero(d == 1)[0]; ends = np.nonzero(d == -1)[0]
        for a, b in zip(starts, ends):
            if a == 0 or b >= nx or b - a > gap:
                continue
            la = src[r, max(0, a - win):a]; rb = src[r, b:b + win]
            if la.sum() < 8 or rb.sum() < 8:
                continue
            if band_mad is not None:
                yl = Y[r, max(0, a - win):a][la]; yr = Y[r, b:b + win][rb]
                ml, mr = np.median(yl), np.median(yr)
                if (np.median(np.abs(yl - ml)) > band_mad or np.median(np.abs(yr - mr)) > band_mad
                        or abs(ml - mr) > band_step):
                    continue
            lv = np.median(lin[r, max(0, a - win):a][la], axis=0)
            rv = np.median(lin[r, b:b + win][rb], axis=0)
            f = ((np.arange(a, b) - a + 0.5) / (b - a))[:, None]
            out[r, a:b] = lv * (1 - f) + rv * f
    return out


def _median_blocks(lin, L, got, src_ok, row_src, hole, rows_ok, cblocks, P, pac, block, max_k, fill, fm, info):
    """block mode 'median': verified lattice-median completion.
    Only texels in rows the wall is photographed in are holes; a band no photograph
    reaches anywhere stays material, so it does not decide which shift a block
    takes.  A candidate shift k*P is kept only when the source block agrees with the
    target block on the texels both see (NCC of sRGB luma >= ncc_min over >= ov_min
    of the block's seen rows): a far wing whose module differs is refused.  A block
    with no co-seen texels takes unverified shifts only when the wall's bay
    autocorrelation is >= blind_ac.  A hole texel covered by >= n_min kept sources
    is their per-channel median (a unique source bay -- a sign, the photo's edge
    bay -- is voted out, not repeated) when the copies agree (median absolute
    deviation of sRGB luma <= mad_max, None = off); the rest, with row_fill, take the bounded
    along-row blend of the wall on either side (_row_fill; its source is every seen
    non-foliage texel: the median itself votes a sign out, and the accent veto would
    also drop a long coloured band such as a plinth cut into pieces by the hole);
    else they stay material."""
    ny, nx = got.shape
    Ls = np.power(np.clip(L, 0, 1), 1 / 2.2)
    ncc_min = float(block.get('ncc_min', 0.5)); ov_min = float(block.get('ov_min', 0.15))
    n_med = int(block.get('n_med', 5)); n_min = int(block.get('n_min', 1)); kmax = int(block.get('max_k', max_k))
    blind_ok = pac >= float(block.get('blind_ac', 1.0))
    mad_max = block.get('mad_max')
    tx = float(block.get('_tx', 0.02))
    row_win = int(round(float(block.get('row_win_m', 1.0)) / tx))
    nrow = max(int(rows_ok.sum()), 1)
    # with a vote (n_min >= 3) the median rejects a unique sign itself, so the accent
    # veto, which also drops every window pane (small, dark, off-hue), is not applied
    src = row_src if n_min >= 3 else src_ok
    st = dict(blocks=0, verified=0, blind=0, refused=0, lattice_px=0, row_px=0)
    for c0, c1 in cblocks:
        tgt = hole[:, c0:c1]
        if not tgt.any():
            continue
        st['blocks'] += 1
        w = c1 - c0; gt = got[:, c0:c1] & rows_ok[:, None]
        ver, unv, seen_any = [], [], False
        for k in sorted(range(-kmax, kmax + 1), key=abs):
            s0, s1 = c0 - k * P, c1 - k * P
            if k == 0 or s0 < 0 or s1 > nx:
                continue
            so = src[:, s0:s1]
            if not (tgt & so).any():
                continue
            co = gt & so
            if co.sum() >= ov_min * nrow * w:
                seen_any = True
                a = Ls[:, c0:c1][co]; b = Ls[:, s0:s1][co]
                a = a - a.mean(); b = b - b.mean()
                ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
                if ncc >= ncc_min:
                    ver.append((-ncc, s0))
            else:
                unv.append((0.0, s0))
        use = sorted(ver)[:n_med] if seen_any else (unv[:n_med] if blind_ok else [])
        st['verified' if seen_any and use else 'blind' if use else 'refused'] += 1
        done = np.zeros_like(tgt)
        if use:
            stack = np.full((len(use), ny, w, 3), np.nan, 'f4')
            for i, (_, s0) in enumerate(use):
                m = tgt & src[:, s0:s0 + w]
                stack[i][m] = lin[:, s0:s0 + w][m]
            cov = np.isfinite(stack[..., 0]).sum(0) >= n_min
            if cov.any():
                med = np.nanmedian(stack[:, cov], axis=0)
                if mad_max is not None:
                    # consensus: where the copies disagree there is no clean module here
                    sl = np.power(np.clip(stack[:, cov] @ LUMW, 0, 1), 1 / 2.2)
                    ml = np.power(np.clip(med @ LUMW, 0, 1), 1 / 2.2)
                    agree = np.nanmedian(np.abs(sl - ml[None]), axis=0) <= mad_max
                    cv = np.zeros_like(cov); cv[cov] = agree; med = med[agree]; cov = cv
                fill[:, c0:c1][cov] = med
                done |= cov; st['lattice_px'] += int(cov.sum())
        fm[:, c0:c1] |= done
    if block.get('row_fill'):
        rest = hole & ~fm
        hmax = block.get('row_hmax_m')
        if hmax is not None:
            # a row blend is a band continuation: only a hole at most row_hmax_m tall in
            # its column (plinth, sill, cornice line, a sliver); a tall 2D hole would
            # smear into horizontal streaks and stays material
            hm = int(round(float(hmax) / tx)); tall = np.zeros_like(rest)
            for c in np.nonzero(rest.any(0))[0]:
                d = np.diff(np.r_[0, rest[:, c].astype(np.int8), 0]); a_, b_ = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
                for a0, b0 in zip(a_, b_):
                    if b0 - a0 > hm:
                        tall[a0:b0, c] = True
            rest_row = rest & ~tall
        else:
            rest_row = rest
        base = np.where(fm[..., None], fill, lin)       # a lattice-filled texel bounds a run too
        rf = _row_fill(base, row_src | fm, rest_row, row_win, int(round(float(block.get('row_gap_m', 12.0)) / tx)),
                       block.get('band_mad'), block.get('band_step', 0.06))
        okr = rest_row & np.isfinite(rf[..., 0])
        cmin = float(block.get('row_cover_min', 0.0))
        if cmin > 0 and okr.any():
            # where the band rule fills only a few rows of a hole the result reads as
            # stripes over material: keep a row-filled texel only where >= cmin of the
            # hole texels within +-row_win above and below in its column are row-filled too
            from scipy.ndimage import uniform_filter as _uf
            k = 2 * row_win + 1
            dens = _uf(okr.astype('f4'), (k, 1)) / np.maximum(_uf(rest.astype('f4'), (k, 1)), 1e-6)
            n0 = int(okr.sum()); okr &= dens >= cmin; st['row_px_dropped'] = n0 - int(okr.sum())
        fill[okr] = rf[okr]; fm |= okr; st['row_px'] = int(okr.sum())
    info['mode'] = 'bay_median'; info['median_blocks'] = st


def module_fill(lin, got, tx, bay_m=(2.0, 12.0), storey_m=(2.6, 5.5), ac_min=0.25,
                max_k=40, vertical=True, accent_sat=0.45, accent_hue=0.06, accent_max_m2=4.0, accent_grow_m=0.6,
                max_ratio=None, block=None):
    """block: copy whole modules, not texels.
    The texel mode picks the nearest shift with a valid source for every texel on
    its own, so one hole is stitched from several shifts, which draws staircase cuts,
    arrow-shaped pieces and pilaster bases repeated into the plinth.  In block mode the wall is cut into bay-wide column blocks
    (edges where the column profile is flattest) and storey-high row blocks, and
    each block takes ONE shift, the nearest whose source covers at least
    block['min_src'] of the block's unseen texels; the rest stays material."""
    ny, nx = got.shape
    info = dict(bay_px=None, bay_ac=0.0, storey_px=None, storey_ac=0.0, filled_frac=0.0, mode=None)
    fill = np.zeros_like(lin); fm = np.zeros((ny, nx), bool)
    if not got.any():
        return fill, fm, info
    L = lin @ LUMW
    # foliage and whatever it hides is not the building's module: never a copy source
    from scipy.ndimage import binary_dilation
    veg = vegetation(lin) & got
    # a small saturated blob (a sign's letters, a logo, a red car) is unique, not module
    from scipy.ndimage import label
    sr = np.power(np.clip(lin, 0, 1), 1 / 2.2)
    sat = (sr.max(-1) - sr.min(-1)) / np.maximum(sr.max(-1), 1e-6)
    # ... and so is anything whose hue stands out from the wall's own: sign
    # panels, logos, awnings.  A texel is an accent when it is saturated or its
    # chromaticity sits off the wall's median chromaticity; accent blobs under
    # accent_max_m2 are vetoed with an accent_grow_m margin, so no fragment of one
    # is ever repeated
    ch = lin / np.maximum(lin.sum(-1, keepdims=True), 1e-4)
    med = np.median(ch[got], axis=0)
    hue_off = np.abs(ch - med).max(-1) > accent_hue
    lab, n = label(((sat > accent_sat) | (hue_off & (sat > 0.25))) & got)
    acc = np.zeros((ny, nx), bool)
    if n:
        sz = np.bincount(lab.ravel())
        small = sz * tx * tx < accent_max_m2; small[0] = False
        acc = binary_dilation(small[lab], iterations=max(1, int(round(accent_grow_m / tx))))
    src_ok = got & ~veg & ~acc
    colcov = got.mean(0)
    cols = colcov > 0.3 * max(colcov.max(), 1e-6)
    # column profile over the rows the owned columns share
    rows_ok = got[:, cols].mean(1) > 0.5 if cols.any() else np.zeros(ny, bool)
    # median mode: a hole row is any row the owned columns see at row_min (the plinth band counts)
    hole_rows = (got[:, cols].mean(1) > float(block.get('row_min', 0.5))) if (block and cols.any()) else rows_ok
    g2 = got & rows_ok[:, None]
    cp = (L * g2).sum(0) / np.maximum(g2.sum(0), 1)
    P, pac = period(cp, g2.sum(0) > 0.5 * max(rows_ok.sum(), 1),
                    int(bay_m[0] / tx), int(bay_m[1] / tx), ac_min)
    info['bay_ac'] = round(pac, 3)
    if P is not None:
        info['mode'] = 'bay'
    info['bay_px'] = P
    uu = np.arange(nx)
    need = ~got
    # one shift per target COLUMN (and per target row below), the smallest that finds
    # a mostly valid source line: per-texel choices break a window into pieces
    min_src = float(block['min_src']) if block else None
    cblocks = [(0, nx)]
    if P and block:
        gcp = np.abs(np.gradient(cp))
        ph = int(np.argmin([gcp[f::P].mean() for f in range(P)]))
        cuts = [0] + [c for c in range(ph, nx, P) if 0 < c < nx] + [nx]
        cblocks = list(zip(cuts[:-1], cuts[1:]))
        info['mode'] = 'bay_block'
    if P and block and block.get('mode') == 'median':
        _median_blocks(lin, L, got, src_ok, got & ~veg, need & hole_rows[:, None], rows_ok, cblocks, P, pac,
                       dict(block, _tx=tx), max_k, fill, fm, info)
    elif P and block:
        for c0, c1 in cblocks:
            tgt = need[:, c0:c1] & ~fm[:, c0:c1]
            nt = int(tgt.sum())
            if nt == 0:
                continue
            for k in sorted(range(-max_k, max_k + 1), key=abs):
                s0, s1 = c0 - k * P, c1 - k * P
                if k == 0 or s0 < 0 or s1 > nx:
                    continue
                ok = tgt & src_ok[:, s0:s1]
                if ok.sum() >= min_src * nt:
                    fill[:, c0:c1][ok] = lin[:, s0:s1][ok]
                    fm[:, c0:c1] |= ok
                    break
    elif P:
        for k in sorted(range(-max_k, max_k + 1), key=abs):
            if k == 0 or abs(k) * P >= nx:
                continue
            src = uu - k * P
            okc = (src >= 0) & (src < nx)
            sc = np.clip(src, 0, nx - 1)
            hit = need & ~fm & okc[None, :] & src_ok[:, sc]
            if hit.any():
                vv, cc = np.nonzero(hit)
                fill[vv, cc] = lin[vv, sc[cc]]; fm[vv, cc] = True
    if vertical:
        have = src_ok | fm
        rowcov = have.mean(1)
        rp = (L * have).sum(1) / np.maximum(have.sum(1), 1)
        S, sac = period(rp, rowcov > 0.3, int(storey_m[0] / tx), int(storey_m[1] / tx), ac_min)
        info['storey_px'] = S; info['storey_ac'] = round(sac, 3)
        if S and block:
            base = lin.copy(); base[fm] = fill[fm]
            for r0 in range(0, ny, S):
                r1 = min(ny, r0 + S)
                for c0, c1 in cblocks:
                    tgt = need[r0:r1, c0:c1] & ~fm[r0:r1, c0:c1]
                    nt = int(tgt.sum())
                    if nt == 0:
                        continue
                    for k in range(1, max_k + 1):
                        s0 = r0 - k * S
                        if s0 < S:              # never copy the ground storey upward
                            break
                        ok = tgt & have[s0:s0 + (r1 - r0), c0:c1]
                        if ok.sum() >= min_src * nt:
                            fill[r0:r1, c0:c1][ok] = base[s0:s0 + (r1 - r0), c0:c1][ok]
                            fm[r0:r1, c0:c1] |= ok
                            break
        elif S:
            base = lin.copy(); base[fm] = fill[fm]
            vv = np.arange(ny)
            for k in range(1, max_k + 1):
                src = vv - k * S
                okr = src >= S                   # never copy the ground storey upward
                if not okr.any():
                    break
                sr = np.clip(src, 0, ny - 1)
                hit = need & ~fm & okr[:, None] & have[sr, :]
                if hit.any():
                    r, c = np.nonzero(hit)
                    fill[r, c] = base[sr[r], c]; fm[r, c] = True
    if max_ratio is not None and fm.sum() > max_ratio * got.sum():
        # keep the completion nearest the photograph: rank filled texels by distance
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~got)
        cut = np.sort(dist[fm])[int(max_ratio * got.sum()) - 1]
        fm &= dist <= cut
    info['filled_frac'] = round(float(fm.mean()), 4)
    return fill, fm, info
