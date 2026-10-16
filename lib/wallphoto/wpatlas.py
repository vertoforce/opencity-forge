"""Stage 60 step 6 -- pack the baked wall tiles into atlas pages.

The output is exactly the interface stage 22 already reads: atlas_NN.jpg plus
wallmap.npz (face_atlas, uv, fnrm, wall_id) plus facadetex.json.  Walls this
pass did not photograph keep the tile stage 35 baked for them, copied pixel for
pixel out of stage 35's own atlas, so material walls cannot regress.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import common as C
from wallphoto import wpcommon as H
from wallphoto import wpbake

import s35_facadetex as S35

ATLAS_PX = S35.ATLAS_PX
PAD = S35.PAD
CORE = (-110.0, 110.0, -130.0, 90.0)     # the bbox s22 --core keeps
# ---- the far-gate acceptance test ----------------------------------------- #
# A wall no panorama passes inside 25 m is photographed under R_MAX_FAR (40 m).
# A far-gate photograph is worth having only when it covers most of the wall:
# below about half, what ships is a photo island in a grey wash where stage 35
# has a coherent window grid, and it reads worse than the material tile (walls
# covered at 0.13-0.34 do).  Two more far-gate failures are green foliage the
# 0.5 m lidar DSM does not occlude at 30-40 m (64-84 % green texels) and frames
# shot at dusk (over 40 % of the texels under 0.10 linear).  Near walls are held
# only to PHOTO_MIN_AREA.
# A lower floor (0.12) does not help: walls covered at 0.13-0.34 ship a
# photographed band over a featureless wash (a glass corner on grey where stage 35
# has a window grid, sky past the building's own edge, a narrow strip), while the
# walls the far gate keeps sit at 0.48-0.80 coverage in a single blob.  Coverage,
# not the shape of the covered part, is what separates them: the "fill" of the
# photo's own bounding box does not.  The way to photograph more of these walls is
# more photographs (the relevel pass), not a lower bar.
FAR_PHOTO_MIN = 0.45     # share of the wall the photograph has to cover
FAR_GREEN_MAX = 0.08     # share of photo texels that may be foliage-green
FAR_DARK_MAX = 0.25      # share under 0.10 linear luminance

# How far a wall's one photograph may be corrected before it is thrown away.
# A shaded facade needs 2-3x to sit at the multi-view consensus and still looks
# right; a night frame can need several hundred x and does not.  The luminance
# floor is the real test: below 0.008 linear (about 25 sRGB levels) the
# photograph holds only quantisation noise.
LEVEL_MIN, LEVEL_MAX = 0.35, 6.00
LEVEL_FLOOR = 0.008
GAIN_CLIP = 3.2
# ---- the bounded rescue for a near-black photograph ----------------------- #
# LEVEL_FLOOR is an ABSOLUTE test on the photograph's own 60th percentile, and it
# is the most common reason the level gate drops a wall.  An absolute floor also
# drops dark but correctly exposed walls: a rusted corten wall in shade can
# measure 0.0077 linear and need only a 0.91x correction, and its photograph
# carries legible siding, a roofline and a service box where the stage 35 tile is
# a flat brown blur.
# So a wall that fails ONLY the floor takes this path instead: the correction
# it needs is applied up to LEVEL_DARK_GAIN, the NORMAL level gate is re-read
# on the gained photograph (so what is bounded is the amplification, not the
# darkness), and the result has to survive a noise test.  The failure this
# guards against is amplified sensor noise on a near-black panorama crop.
LEVEL_DARK_GAIN = 8.0
# 8x is the ceiling on the APPLIED gain; a wall needing more ships under-
# exposed by the remainder, and the remainder is then what LEVEL_MAX judges
# (a frame shot in the dark can need 662x, a residual of 82.7, and LEVEL_MAX
# 6.00 on the residual turns it down).
DARK_SD_MAX = 22.0
# The noise test the bounded gain needs: mean local standard deviation of the
# gained tile's luminance in a 3x3 window over the photographed texels, in sRGB
# levels.  Over walls that clear the level gate normally it measures min 2.52,
# median 9.05, p95 18.33, max 22.08; the bar is that max.  The test guards
# against amplified sensor noise: a 5760 px panorama crop lifted 8x can come up
# as a grey smear.
DARK_BLOWN_MAX = 0.04
# A near-black 60th percentile on a photograph that also contains sky fails the
# other way: a blown white strip over a black soffit void, lifted 8x, clips
# 13.3 % of its texels above 0.98 luminance and reads far worse than the clean
# brick strip stage 35 has.  Rescued photographs with a blown share of 0.000-0.004
# read well and 0.049-0.133 do not; the bar sits near the p95 (0.030) of the walls
# that clear the gate normally.  It is tighter than the 0.18-0.27 blown share
# some normally-gated walls ship at, because a rescued tile has had no check
# beyond these gates.
DARK_GREEN_MAX = 0.35
# And foliage.  A wall can be covered at 0.94 and lift cleanly, and what comes up
# is a tree and a bright path -- the 0.5 m lidar DSM did not occlude the canopy,
# so 73 % of its texels are foliage-green.  FAR_GREEN_MAX's 0.08 is too tight to
# reuse here (a green roof or a bush in an otherwise good photograph measures
# 0.09-0.13), so the bar is bracketed at 0.156 kept, 0.734 turned down.
# ---- the level reference must be stage 35 BEFORE its tone match ----------- #
# Stage 35 applies a tone match: a wall with no photograph of its own takes the
# linear luminance of the photographed walls near it.  The refreshed
# _pre_wallphoto reference therefore does not carry stage 35's own measured
# level, and levelling this pass's photograph against it is circular -- it
# compares the photograph with a neighbour's exposure (walls brightened about
# 3-4x by the tone match drop against it).
# So the level target is divided back by the wall's own tone-match ratio,
# read from the reference facadetex.json.  The ratio is the flat wall colour's,
# not the tile's, so the reconstruction is not exact: against a kept pre-match
# s35 reference atlas it lands within 3-12 % of the real pre-match 60th
# percentile.  LEVEL_UNDO_TOL is that slack, and only walls whose target was
# reconstructed get it -- at 1.00 a 5 % reconstruction error (real gl 5.89,
# reconstructed 6.20) would throw away a full photographed tile over a rounding.
LEVEL_UNDO_TOL = 1.15

# ---- one building, one level ---------------------------------------------- #
# The undo above is right for a wall stage 35 PHOTOGRAPHED: that tile's
# pre-tone-match level is a measurement of the wall.  It is wrong for a wall
# stage 35 could not photograph, because there is no measurement in that tile
# at all -- it is the OSM building:colour swatch -- and the tone match the undo
# divides out is the only measured thing stage 35 put in it (the level of the
# photographed walls of the SAME building).
# Example: on one building the walls stage 35 photographed ship at 0.162-0.216
# linear luminance, a 1.33x spread over all four compass faces, so the building
# has no real per-face brightness spread.  The walls it could not photograph
# level to 0.717-0.725, 3.8x above that, entirely because of the undo -- a west
# wall at 0.717 against another west wall of the same building at 0.187.
# The source photographs are not the evidence and cannot be used as it: their
# own 60th percentiles run 0.180-0.464 (2.6x) over the same walls and do not
# predict the levels above (the brightest photograph has a middling level).
# That is panorama exposure, not scene illumination.
# So the building's own photographed walls set the level, and an unphotographed
# wall is held within CONSIST_RATIO of it.  A building whose photographed walls
# ARE genuinely bright gets a high level and keeps it; a building already
# inside the band, and every wall of a building with no photographed wall at
# all, is untouched; and a photographed wall is the evidence and never moves.
# The rule is a CEILING, not a band.  A symmetric band, lifting the walls that
# sit BELOW their building's level, damages them, because what is under-exposed
# about those tiles is not a levelling error: their blown share rises (0.000 to
# 0.095) and their local sd passes the DARK_SD_MAX bar.  Walls the ceiling cuts
# improve on both counts (blown share 0.18-0.27 drops to at most 0.003).  So
# only the direction the undo is known to fail in is corrected.
CONSIST_RATIO = 1.6


def building_levels(walls, s35rec, s35_tile):
    """-> {building: linear luminance the walls stage 35 photographed ship at}.

    Area-weighted over those walls' stage 35 tiles, which is the level they
    ship at whatever this pass does: a photographed wall either keeps that tile
    or has its photograph levelled onto it.  Buildings with no photographed
    wall get no entry, and their walls are levelled without the
    building-consistency bound."""
    acc = {}
    for w in walls:
        rec = s35rec.get(w['id'])
        if not rec or 'rect_px' not in rec:
            continue
        lref = rec.get('level_reference')
        sc = 1.0
        if isinstance(lref, dict) and lref.get('scale'):
            # the far-colour wall counts as the colour it replaced (see level_gain)
            rec = dict(rec, tone_match=lref.get('tone_match'))
            sc = np.asarray(lref['scale'], 'f8')
        if tone_undo(rec) != 1.0:
            continue
        bid = rec.get('part_name') or w.get('part')
        a = float(w.get('area_m2') or 0.0)
        if bid is None or a <= 0:
            continue
        try:
            tl = float((np.percentile(H.to_linear(s35_tile(rec)).reshape(-1, 3),
                                      60, axis=0) * sc) @ H.LUMW)
        except Exception:
            continue
        if tl <= 1e-6:
            continue
        s, n = acc.get(bid, (0.0, 0.0))
        acc[bid] = (s + tl * a, n + a)
    return {k: v[0] / v[1] for k, v in acc.items() if v[1] > 0}


def consistency_scale(und, tgt_lum, cur_lum, level, tol=1.0):
    """-> the factor on this wall's levelling target that holds one building's
    walls on one level, 1.0 when nothing needs moving.

    level is what the walls of this building stage 35 photographed ship at, or
    None when it has none.  A photographed wall (und == 1.0) is that evidence
    and is never moved."""
    if level is None or und == 1.0 or tgt_lum <= 1e-6 or not (level > 1e-6):
        return 1.0
    hi = level * CONSIST_RATIO
    if tgt_lum <= hi:
        return 1.0                      # this building already agrees
    s = float(hi / tgt_lum)
    # Never push a wall THROUGH the level gate in either direction: this rule
    # evens out brightness, it does not decide which walls ship.  A wall whose
    # correction is inside the band keeps a correction inside the band; a wall
    # already outside it is left where it was, for run_relevel to retry.
    if cur_lum > 1e-6:
        g0 = tgt_lum / cur_lum
        if LEVEL_MIN / tol <= g0 <= LEVEL_MAX * tol:
            s = min(max(s, (LEVEL_MIN / tol) / g0), (LEVEL_MAX * tol) / g0)
        else:
            return 1.0
    return s


def tone_undo(rec):
    """Factor putting a stage 35 tile back on its pre-tone-match luminance.

    1.0 for a wall stage 35 photographed itself, or a reference with no
    tone_match record: those tiles already carry their own measured level."""
    tm = rec.get('tone_match') if isinstance(rec, dict) else None
    if not isinstance(tm, dict):
        return 1.0
    lb, la = tm.get('luminance_before'), tm.get('luminance_after')
    try:
        lb = float(lb); la = float(la)
    except (TypeError, ValueError):
        return 1.0
    if not (np.isfinite(lb) and np.isfinite(la)) or la <= 1e-6 or lb <= 1e-6:
        return 1.0
    return lb / la


# ---- the exposure retry --------------------------------------------------- #
# The level gate and the far gate both judge ONE composite: the tile the
# ownership stage built out of the views it liked best on geometry alone
# (cosine, range, alignment quality).  Exposure is never in that score, so a
# wall standing next to a well-lit panorama can still lose to a night frame
# that happened to face it more squarely, and then ship as material.  A dropped
# wall with a second correlated panorama can ship once that view gets its turn
# (a winning view under the luminance floor, or one 3.7x too bright, is replaced
# by a view that covers the wall at about 0.5).
# Note what this does NOT predict: a candidate's own chip luminance is not the
# tile's.  Views that measure 0.0152 and 0.0057 linear on their chips can look
# like a certain rescue, and the tile they actually bake still lands under the
# floor once the levelling solve has harmonised them.
# The only test that counts is the re-baked tile, which is why this pass bakes
# before it judges instead of ranking candidates by chip.
# So a dropped wall gets its losing view set BANNED and the ownership re-solved
# without it, up to LEVEL_RETRY_VIEWS times.  Nothing is relaxed: the retried
# tile has to clear the same gates, and a wall that never clears them is put
# back exactly as it was, owner array and tile alike.
LEVEL_RETRY_VIEWS = 3
# What the retry replaces is not a blank wall, it is a stage 35 tile that
# already carries content, so a retried photograph has to cover as much of the
# wall as the far gate demands -- PHOTO_MIN_AREA's 0.10, which is the floor for
# a wall's FIRST photograph, is too low a bar to overwrite one.  A retried
# photograph at 0.51-0.57 coverage is a clear gain (a legible doorway at 1.6 cm
# per texel where stage 35 has a blur at 5, or a coherent glass facade), and one
# at 0.155 is the failure the far gate exists for -- a bright strip in a flat
# wash.  The floor is therefore the far gate's own 0.45.
RETRY_PHOTO_MIN = FAR_PHOTO_MIN


def dark_noise(lin, got, g):
    """-> (local sd in sRGB levels, blown share, foliage-green share).

    Measured on the photograph AFTER the gain `g`, over the photographed texels
    only: the three statistics the bounded rescue is judged on."""
    from scipy.ndimage import uniform_filter
    srgb = np.asarray(H.to_srgb(np.clip(lin * np.asarray(g, 'f4')[None, None, :],
                                        0, 1)), 'f4')
    if srgb.max() > 1.001:
        srgb = srgb / 255.0
    y = srgb @ H.LUMW
    m = np.asarray(got, bool)
    mf = m.astype('f4')
    n = uniform_filter(mf, 3, mode='nearest')
    mu = uniform_filter(y * mf, 3, mode='nearest') / np.maximum(n, 1e-6)
    mu2 = uniform_filter(y * y * mf, 3, mode='nearest') / np.maximum(n, 1e-6)
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0.0)) * 255.0
    inner = m & (n > 0.5)
    px = srgb[m]
    return (float(sd[inner].mean()) if inner.any() else 0.0,
            float((y[m] > 0.98).mean()),
            float(((px[:, 1] - np.maximum(px[:, 0], px[:, 2])) > 0.02).mean()))


def level_gain(rec, lin, got, tile35, photo_frac=None, consist=None):
    """The exposure correction this photograph needs to sit on stage 35's level.

    -> (gl, gc, und, cur_lum, why, clip) with why None when the gate passes, and
    clip the ceiling the caller must apply the per-channel gain under.  This is
    the whole level test, shared by run_atlas and the retry so the two cannot
    drift apart.  photo_frac opens the bounded rescue for a wall that fails only
    the luminance floor; without it the floor is absolute.  consist is this
    building's measured level, see consistency_scale; without it the undo is
    unbounded."""
    ot = H.to_linear(tile35).reshape(-1, 3)
    tgt = np.percentile(ot, 60, axis=0)
    lref = rec.get('level_reference') if isinstance(rec, dict) else None
    if isinstance(lref, dict) and lref.get('scale'):
        # Far grid: stage 35 drew this wall in its far photograph
        # colour; the gate judges against the colour it replaced, so a far
        # colour never lets through a photograph the replaced tile turned down
        tgt = tgt * np.asarray(lref['scale'], 'f8')
        rec = dict(rec, tone_match=lref.get('tone_match'))
    und = tone_undo(rec)
    if und != 1.0:
        # back off stage 35's tone match: level against what this wall's own
        # tile measured, not against its neighbours'
        tgt = tgt * und
    tol = LEVEL_UNDO_TOL if und != 1.0 else 1.0
    cur = np.percentile(lin[got], 60, axis=0)
    lc = float(np.dot(cur, H.LUMW))
    # ...but an unphotographed wall has no own measurement to back off TO, so
    # hold it within CONSIST_RATIO of its building's photographed walls
    cs = consistency_scale(und, float(np.dot(tgt, H.LUMW)), lc, consist, tol)
    if cs != 1.0:
        tgt = tgt * cs
    gl = float(np.dot(tgt, H.LUMW) / max(lc, 1e-4))
    gc = tgt / np.maximum(cur, 1e-4)
    why = None
    clip = GAIN_CLIP
    dark_path = (lc < LEVEL_FLOOR and photo_frac is not None
                 and gl >= LEVEL_MIN / tol)
    if dark_path:
        # bound the amplification, then re-read the normal gate on the result
        clip = LEVEL_DARK_GAIN
        k = min(LEVEL_DARK_GAIN, gl)
        res = gl / max(k, 1e-4)
        g = np.clip(gl * np.sqrt(np.maximum(gc, 1e-3) / max(gl, 1e-4)),
                    0.45, clip)
        if res > LEVEL_MAX * tol:
            why = ('the photograph is %.1fx darker than the wall even after a '
                   '%.0fx lift' % (res, k))
        elif photo_frac < RETRY_PHOTO_MIN:
            # this rescue overwrites an existing stage 35 tile, so it owes the
            # same coverage a retried photograph owes
            why = ('a lifted photograph covers %.2f of the wall' % photo_frac)
        else:
            sd, blown, green = dark_noise(lin, got, g)
            if sd > DARK_SD_MAX:
                why = 'the lifted photograph is noise (local sd %.1f)' % sd
            elif blown > DARK_BLOWN_MAX:
                why = '%.2f of the lifted photograph is blown out' % blown
            elif green > DARK_GREEN_MAX:
                why = '%.2f of the lifted photograph is foliage' % green
    elif gl > LEVEL_MAX * tol or gl < LEVEL_MIN / tol or lc < LEVEL_FLOOR:
        why = ('the photograph itself is under the luminance floor'
               if lc < LEVEL_FLOOR else
               ('the photograph is %.1fx darker than the wall' % gl
                if gl > LEVEL_MAX * tol else
                'the photograph is %.1fx brighter than the wall'
                % (1.0 / max(gl, 1e-4))))
    return gl, gc, und, lc, why, clip


def _fallback(w, rec, nx, ny, bases, rows_part, tint, no_windows, pbr_out=None):
    cls = rec['cls']
    base, tile_m = bases[cls]
    near_core = float(np.hypot(*w['centre'][:2])) < 130.0   # within 130 m of the area origin
    oo = {}
    rows = rows_part.get(str(w['part']))
    # the storey evidence stage 35 used: without it synth_facade takes the
    # blind branch and a partially photographed wall loses its window grid
    syn, ginfo = S35.synth_facade(w, cls, nx, ny, base, tile_m, rows,
                                  seed=1000 + w['id'], near_core=near_core,
                                  no_windows=no_windows,
                                  storey_src=S35.storey_evidence(w, rows),
                                  openings_out=oo, pbr_out=pbr_out)
    if tint.shape[:2] != (ny, nx):
        from PIL import Image
        ti = Image.fromarray(np.clip(tint * 255, 0, 255).astype('u1'))
        tint = np.asarray(ti.resize((nx, ny), Image.BILINEAR), 'f4') / 255.0
    # this is the SECOND place a synthetic wall is multiplied
    # by its tint, so it has the same black-window failure stage 35 has -- a
    # wall whose tint sits at TONE_LUM_FLOOR drives its openings under 0.02
    # linear and they render as holes.  The floor is absolute and only lifts
    # texels that fall under it, so a bright wall is untouched.
    return S35.apply_modules(S35.apply_opening_floor(np.clip(syn * tint, 0, 1),
                                                     oo.get('mask')), pbr_out), ginfo


def far_reject(photo, got, meta):
    """-> None if this far-gate tile is worth shipping, else the reason."""
    if meta['photo_frac'] < FAR_PHOTO_MIN:
        return f"photograph covers {meta['photo_frac']:.2f} of the wall"
    px = np.asarray(photo)[np.asarray(got, bool)]
    if px.size == 0:
        return 'no photo texels'
    f = (px.astype('f4') / 255.0) if px.dtype == np.uint8 else np.asarray(px, 'f4')
    green = float(((f[:, 1] - np.maximum(f[:, 0], f[:, 2])) > 0.02).mean())
    dark = float(((f @ np.array([.2126, .7152, .0722], 'f4')) < 0.10).mean())
    if green > FAR_GREEN_MAX:
        return f'{green:.2f} of the photo texels are foliage'
    if dark > FAR_DARK_MAX:
        return f'{dark:.2f} of the photo texels are under 0.10 linear'
    return None


def reference_pack():
    """(walls, the stage 35 note, its per-wall records, a tile reader).

    The reference is always stage 35's own output: once s62 has swapped this
    pass's atlases into facades/, the stage 35 files live in the backup, and
    reading facades/ instead would compare this pass against itself."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    walls = json.load(open(H.FAC + '/walls.json'))['walls']
    REF = H.FAC + '/_pre_wallphoto'
    if not os.path.exists(REF + '/facadetex.json'):
        REF = H.FAC
    s35pack = json.load(open(REF + '/facadetex.json'))
    s35rec = {r['id']: r for r in s35pack['per_wall']}
    s35atlas = {}

    def s35_tile(rec):
        pg = rec['atlas']
        if pg not in s35atlas:
            s35atlas[pg] = np.asarray(Image.open(f'{REF}/atlas_{pg:02d}.jpg'), 'u1')
        x, y, wd, ht = rec['rect_px']
        return s35atlas[pg][y:y + ht, x:x + wd].copy()

    def s35_aux(rec, kind):
        """stage 35's normal ('n') / roughness ('r') tile, or None."""
        pg = rec.get('atlas'); key = (pg, kind)
        if pg is None or 'rect_px' not in rec:
            return None
        if key not in s35atlas:
            fp = f'{REF}/atlas_{pg:02d}_{kind}.jpg'
            s35atlas[key] = (np.asarray(Image.open(fp).convert('RGB'), 'u1')
                             if os.path.exists(fp) else None)
        if s35atlas[key] is None:
            return None
        x, y, wd, ht = rec['rect_px']
        return s35atlas[key][y:y + ht, x:x + wd].copy()
    s35_tile.aux = s35_aux
    return walls, s35pack, s35rec, s35_tile, REF


def photo_floor_ok(meta, w, floor):
    """True when the baked photograph covers enough of the wall to ship.

    `floor` is a share of the wall.  At the base floor (PHOTO_MIN_AREA) a
    photograph under that share still ships when it covers
    C.FACADE_PARTIAL_PHOTO_MIN_M2 square metres: a large wall seen well along
    one end is partially owned, photo there and material elsewhere, instead of
    losing a full-height strip of photograph to a fraction test.  Stricter
    floors (the far gate, the retry) are unchanged."""
    if meta['photo_frac'] >= floor:
        return True
    if w is None or floor > wpbake.PHOTO_MIN_AREA:
        return False
    return (meta['photo_frac'] * w['width_m'] * w['height_m']
            >= C.FACADE_PARTIAL_PHOTO_MIN_M2)


def run_atlas(setname='walls', quality=93, photo_min=wpbake.PHOTO_MIN_AREA,
              outdir=None):
    t0 = time.time()
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    D = H.setdir(setname)
    OUT = outdir or (D + '/out')
    os.makedirs(OUT, exist_ok=True)
    walls, s35pack, s35rec, s35_tile, REF = reference_pack()
    blevel = building_levels(walls, s35rec, s35_tile)
    consisted = []
    rows_part = {}
    for k, v in (s35pack.get('storey_heights_measured') or {}).items():
        rows_part[str(k)] = dict(storey_m=float(v), phase=0.0)
    # stage 35's own storey phase is not in the note; recover it where the
    # per-wall record kept it
    idx = json.load(open(S35.CC0 + '/index.json'))
    sizes_m = S35.tile_sizes(idx)
    bases = {}
    for cls, bm in S35.BASE_MAT.items():
        bases[cls] = (S35.load_base(idx, bm, sizes_m, S35.KEEP_CHROMA[cls])
                      if bm else (None, None))
    # which walls were photographed under the long gate
    try:
        _rk, _ = H.load_rects(setname)
        far_ids = {r.id for r in _rk if H.rect_rmax(r) > H.R_MAX}
    except Exception:
        far_ids = set()
    far_rejected = {}
    far_kept = []
    tiles = []; recs = []; auxs = []

    def _aux35(rec):
        n = s35_tile.aux(rec, 'n'); r = s35_tile.aux(rec, 'r')
        return None if n is None or r is None else (n, r)
    npho = 0
    nglass = [0]

    def glass_reference(w, rec):
        """Area-weighted linear mean of stage 35's tiles over this building's
        glass walls that stage 35 drew as material, or None."""
        acc = np.zeros(3); aw = 0.0
        for r in s35rec.values():
            if (r.get('part_name') != rec.get('part_name') or r.get('cls') != 'glass'
                    or r.get('texture') != 'material' or r.get('buried')
                    or 'rect_px' not in r or r['id'] == w['id']):
                continue
            t = s35_tile(r)
            if t is None or t.size == 0:
                continue
            acc += H.to_linear(t).reshape(-1, 3).mean(0) * r['area_m2']; aw += r['area_m2']
        return (acc / aw).astype('f4') if aw > 0 else None
    dropped_level = []
    dropped_why = []
    dark_ok = []                      # walls the bounded dark rescue brought back
    undone = []
    nship = [0]
    for w in walls:
        rec = dict(s35rec.get(w['id'], {}))
        tp = wpbake.tile_path(setname, w['id'])
        photo = None
        # A ship is not a building.  The lidar DSM the occlusion test runs on
        # does not carry moored hulls where OSM puts them, so a panorama
        # standing on a quay beside one sees straight THROUGH the hull plane and the
        # projection paints whatever is behind it (interiors and panels of the
        # buildings beyond the hull).  Every ship wall keeps stage 35's hull material and
        # its porthole row.
        if w['osm'].get('building') == 'ship':
            nship[0] += 1
            t = s35_tile(rec) if 'rect_px' in rec else None
            if t is not None:
                rec2 = dict(rec)
                rec2['texture_pass'] = ('stage 35 tile, unchanged: a ship hull, '
                                        'which the DSM cannot occlude')
                tiles.append(t); recs.append(rec2); auxs.append(_aux35(rec))
                continue
        if os.path.exists(tp):
            d = np.load(tp, allow_pickle=True)
            meta = json.loads(str(d['meta']))
            if photo_floor_ok(meta, w, photo_min):
                photo = (d['photo'], d['got'], meta)
            if photo is not None and w['id'] in far_ids:
                why = far_reject(d['photo'], d['got'], meta)
                if why:
                    far_rejected[w['id']] = why
                    photo = None
                else:
                    far_kept.append(w['id'])
        if photo is None:
            t = s35_tile(rec) if 'rect_px' in rec else None
            if t is None:
                continue
            rec2 = dict(rec)
            rec2['texture_pass'] = 'stage 35 tile, unchanged'
            # NOT rebuilt without openings: the dark rectangles on a hull are
            # stage 35's PORTHOLE row (its ship branch draws one band of
            # portholes at 0.76 of the hull height), not a storey window grid,
            # and rebuilding the tile would only flatten the colour.  A hull
            # reads flat when no camera passes inside 25 m of it, not because
            # of the openings.
            tiles.append(t); recs.append(rec2); auxs.append(_aux35(rec))
            continue
        srgb, got, meta = photo
        ny, nx = srgb.shape[:2]
        lin = H.to_linear(srgb)
        # ---- level the wall against the multi-view consensus -------------- #
        # One panorama per patch buys sharpness and loses the average: the
        # single view that wins a wall can be a dusk frame or a deep shadow,
        # so the one-view tile can sit well below the multi-view tile's level
        # (around 10 sRGB levels at the median, over 100 at the worst).  So the
        # LEVEL comes from the tile stage 35 wrote -- which is the median of up to six views, the
        # OSM colour and stage 20's measured face colour -- and only the
        # DETAIL comes from the one photograph.  The 60th percentile is the
        # statistic, because the darkest quarter of stage 35's tile is its
        # generated windows.
        gain_note = None
        if got.any() and 'rect_px' in rec:
            try:
                cons = blevel.get(rec.get('part_name') or w.get('part'))
                gl, gc, und, lc, why, gclip = level_gain(
                    rec, lin, got, s35_tile(rec), meta['photo_frac'], cons)
                if und != 1.0:
                    undone.append((w['id'], round(und, 3)))
                    t35 = float(np.percentile(
                        H.to_linear(s35_tile(rec)).reshape(-1, 3), 60,
                        axis=0) @ H.LUMW) * und
                    cs = consistency_scale(und, t35, lc, cons, LEVEL_UNDO_TOL)
                    if cs != 1.0:
                        consisted.append(dict(
                            id=int(w['id']), building_level=round(cons, 4),
                            undone_target=round(t35, 4),
                            held_to=round(t35 * cs, 4), factor=round(cs, 3)))
                if why is not None:
                    # the photographs that won this wall are night frames or
                    # blown ones and the retry found nothing better: no
                    # exposure correction rescues them, so the wall keeps
                    # stage 35's tile
                    dropped_level.append((w['id'], round(gl, 2)))
                    dropped_why.append(dict(
                        id=int(w['id']), gain=round(gl, 2),
                        photo_luminance_linear=round(lc, 4),
                        tone_undo=round(und, 3), reason=why))
                    t = s35_tile(rec)
                    rec2 = dict(rec)
                    rec2['texture_pass'] = ('stage 35 tile, unchanged: the best '
                                            f'photograph needed a {gl:.1f}x exposure '
                                            'correction to match the consensus')
                    tiles.append(t); recs.append(rec2); auxs.append(_aux35(rec))
                    continue
                if lc < LEVEL_FLOOR:
                    dark_ok.append(int(w['id']))
                g = np.clip(gl * np.sqrt(np.maximum(gc, 1e-3) / max(gl, 1e-4)),
                            0.45, gclip).astype('f4')
                lin = np.clip(lin * g[None, None, :], 0, 1)
                gain_note = [round(float(x), 3) for x in g]
            except Exception:
                gain_note = None
        # the wall's own colour, measured off this pass's photograph, is the
        # tint the material fallback is laid down in
        cell = max(3, int(round(1.2 / max(meta['tx'], 0.01))))
        step = max(1, int(max(nx, ny) / 320))
        cs = lin[::step, ::step]; ts = got[::step, ::step]
        if ts.any():
            tint = S35.smooth_fill(cs, ts, max(3, cell // step))
        else:
            tint = np.tile(np.array(S35.CLASS_DEFAULT_RGB[rec['cls']], 'f4')
                           [None, None, :], (8, 8, 1))
        # A glass wall is drawn, never photographed
        # (common.FACADE_GLASS_PHOTO).  The photograph of a curtain wall is the
        # street behind the camera reflected in it plus the trees in front of
        # it, so the photo island is replaced by the class grid in the
        # building's own glass colour (glass_reference) at the same texel size.
        # A wall the exposure level above dropped already went back to stage
        # 35's tile (same size, so the pack does not move).
        glass_syn = (rec.get('cls') == 'glass'
                     and getattr(C, 'FACADE_GLASS_PHOTO', True) is False)
        glass_ref = None
        if glass_syn:
            glass_ref = glass_reference(w, rec)
            src = (glass_ref if glass_ref is not None else
                   (np.median(lin[got], axis=0) if got.any() else
                    np.array(S35.CLASS_DEFAULT_RGB['glass'], 'f4')))
            tint = np.tile(np.asarray(src, 'f4')[None, None, :], (8, 8, 1))
            got = np.zeros_like(got)
            nglass[0] += 1
        # A synthesised window grid next to a photographed one reads as a fake:
        # the generated openings sit at a different pitch from the real ones a
        # metre away.  So any wall with a
        # real photograph over a quarter of it drops the generated windows and
        # falls back to plain tinted material.
        no_win = meta['photo_frac'] > 0.25 and not glass_syn
        pbf = {} if getattr(C, 'FACADE_PBR', False) else None
        mat, ginfo = _fallback(w, rec, nx, ny, bases, rows_part, tint, no_win, pbf)
        fa_all = np.zeros((ny, nx), 'f4')
        from scipy.ndimage import uniform_filter
        fill_info = None
        if getattr(C, 'FACADE_FILL', False) and not glass_syn:
            # Facade completion: the unseen part repeats the photographed module
            # (common.py FACADE_FILL_*), feathered into the material like the photo
            from wallphoto import wpfill
            fl, fm, fill_info = wpfill.module_fill(
                lin, got, meta['tx'], bay_m=C.FACADE_FILL_BAY_M,
                storey_m=C.FACADE_FILL_STOREY_M, ac_min=C.FACADE_FILL_AC_MIN,
                max_ratio=C.FACADE_FILL_MAX_RATIO,
                block=getattr(C, 'FACADE_FILL_BLOCK', None))
            if fm.any():
                if getattr(C, 'FACADE_BLEND_UNSEEN', None) == 'material':
                    fl = np.where(fm[:, :, None], fl, mat)   # unfilled = material, not black
                fw_ = max(3, int(round(0.45 / meta['tx'])) | 1)
                fa = np.where(fm, np.maximum(uniform_filter(fm.astype('f4'), fw_), 0.75),
                              uniform_filter(fm.astype('f4'), fw_) * 0.9)
                fa = np.where(got, 0.0, fa)
                mat = fl * fa[:, :, None] + mat * (1 - fa[:, :, None])
                fa_all = fa.astype('f4')
            fill_info['filled_m2'] = round(float(fm.mean()) * float(w['area_m2']), 1)
        fw = max(3, int(round(0.45 / meta['tx'])) | 1)
        al = uniform_filter(got.astype('f4'), fw, mode='nearest')
        al = np.where(got, np.maximum(al, 0.75), al * 0.9)
        if getattr(C, 'FACADE_BLEND_UNSEEN', None) == 'material':
            # an unseen photo texel is black; feather photo -> material only
            lin = np.where(got[:, :, None], lin, mat)
        img = lin * al[:, :, None] + mat * (1 - al[:, :, None])
        if glass_syn and glass_ref is not None:
            # level onto the building's drawn glass (packed tile mean, linear)
            img = img * (glass_ref / np.maximum(img.reshape(-1, 3).mean(0), 1e-4))[None, None, :]
        t = H.to_srgb(np.clip(img, 0, 1))[::-1]           # row 0 = top of wall
        rec2 = dict(rec)
        rec2.update(texture='photograph over material',
                    texture_pass='stage 60 best view per patch',
                    px=[nx, ny], cm_per_texel=round(100 * meta['tx'], 2),
                    photo_frac_final=round(float(al.mean()), 4),
                    photo_texel_frac=round(meta['photo_frac'], 4),
                    views_used=len(meta['views']), level_gain=gain_note,
                    owned_patches=meta['owned_patches'], patches=meta['patches'],
                    vscale=meta['vscale'], anchor=meta['anchor'],
                    seam_before=meta['seam_before'], seam_after=meta['seam_after'],
                    range_m=meta['range_m'], completion=fill_info)
        if glass_syn:
            rec2.update(texture='material', photo_frac_final=0.0,
                        texture_pass=('stage 60 synthetic glass (FACADE_GLASS_PHOTO '
                                      'False): the photograph is a reflection; '
                                      + ('levelled onto the building\'s material glass walls'
                                         if glass_ref is not None else
                                         'tinted by the photograph median')))
        tiles.append(t); recs.append(rec2)
        if pbf is not None:
            _n, _r = S35.pbr_pages(pbf, meta['tx'], ny, nx,
                                   photo_alpha=1.0 - (1.0 - al) * (1.0 - fa_all))
            auxs.append((_n[::-1], _r[::-1]))
        else:
            auxs.append(None)
        npho += 1
    # ---- pack: the core walls first, so --core loads the fewest pages ------ #
    incore = []
    for r in recs:
        w = next(x for x in walls if x['id'] == r['id'])
        c = w['centre']
        incore.append(CORE[0] <= c[0] <= CORE[1] and CORE[2] <= c[1] <= CORE[3])
    order = sorted(range(len(tiles)), key=lambda i: (not incore[i], -tiles[i].shape[0]))
    sizes = [(tiles[i].shape[1], tiles[i].shape[0]) for i in order]
    place_o, npages = S35.shelf_pack(sizes)
    place = [None] * len(tiles)
    for k, i in enumerate(order):
        place[i] = place_o[k]
    pages = [np.zeros((ATLAS_PX, ATLAS_PX, 3), 'u1') for _ in range(npages)]
    for t, (pg, x, y), r in zip(tiles, place, recs):
        h, wd = t.shape[:2]
        pages[pg][y:y + h, x:x + wd] = t
        pages[pg][y - PAD:y, x:x + wd] = t[0][None, :, :]
        pages[pg][y + h:y + h + PAD, x:x + wd] = t[-1][None, :, :]
        pages[pg][y - PAD:y + h + PAD, x - PAD:x] = pages[pg][y - PAD:y + h + PAD, x:x + 1]
        pages[pg][y - PAD:y + h + PAD, x + wd:x + wd + PAD] = \
            pages[pg][y - PAD:y + h + PAD, x + wd - 1:x + wd]
        r['atlas'] = int(pg); r['rect_px'] = [int(x), int(y), int(wd), int(h)]
        # ---- record the level the packed tile actually renders at ---- #
        # The median of a photographed wall's TRUSTED TEXELS is not the tile
        # that ships: the stage 60 bake levels the photograph, blends it over
        # the material fallback and re-tints it, and the two disagree.  A wall's
        # trusted-texel median can sit well above the packed tile's mean (for
        # example 0.18 vs 0.13 linear, a 1.37x gap), and that gap shows as a
        # visible step between tone-matched neighbours.  So the pack records the
        # statistic OF THE PACKED TILE, in linear light, measured on `t`: the
        # exact pixels the packing loop above writes into the page, before
        # JPEG.  The next prep's stage 35 reads it back
        # and matches against what the eye will see instead of against texels.
        # Nothing in this pass reads these fields; building_levels() and
        # consistency_scale() do not depend on them.
        _tl = H.to_linear(t).reshape(-1, 3)
        _tmean = _tl.mean(0); _tmed = np.median(_tl, 0)
        r['tile_mean_linear'] = [round(float(v), 5) for v in _tmean]
        r['tile_median_linear'] = [round(float(v), 5) for v in _tmed]
        r['tile_mean_linear_luminance'] = round(float(_tmean @ H.LUMW), 5)
        r['tile_median_linear_luminance'] = round(float(_tmed @ H.LUMW), 5)
        del _tl
    files = []
    for i, p in enumerate(pages):
        fp = f'{OUT}/atlas_{i:02d}.jpg'
        Image.fromarray(p).save(fp, 'JPEG', quality=quality, optimize=True)
        files.append(fp)
    for f in os.listdir(OUT):
        if f.startswith('atlas_') and (f.endswith('_n.jpg') or f.endswith('_r.jpg')):
            os.unlink(OUT + '/' + f)
    if getattr(C, 'FACADE_PBR', False):
        # normal + roughness pages in the colour pages' layout; a tile
        # with no stage 35 aux (normal/roughness) tile is flat
        for kind, k, fill in (('n', 0, S35.NRM_FLAT), ('r', 1, S35.MR_FLAT)):
            at = []
            for t, a in zip(tiles, auxs):
                if a is None or a[k].shape[:2] != t.shape[:2]:
                    at.append(np.broadcast_to(np.asarray(fill, 'u1'), t.shape[:2] + (3,)))
                else:
                    at.append(a[k])
            for i, p in enumerate(S35.pack_aux(at, place, npages, fill)):
                Image.fromarray(p).save(f'{OUT}/atlas_{i:02d}_{kind}.jpg', 'JPEG',
                                        quality=95 if kind == 'n' else 90, optimize=True)
    abytes = int(sum(os.path.getsize(f) for f in files))
    # ---- the map stage 22 reads ------------------------------------------- #
    z = np.load(C.W + '/layers/buildings.npz')
    pos = z['pos'].astype('f8'); bidx = z['idx']
    face_atlas = np.full(len(bidx), -1, 'i1')
    uv = np.zeros((len(bidx), 3, 2), 'f4')
    fnrm = np.zeros((len(bidx), 3), 'f4')
    wall_id = np.full(len(bidx), -1, 'i4')
    bywall = {w['id']: w for w in walls}
    for r in recs:
        w = bywall[r['id']]
        x, y, wd, ht = r['rect_px']
        O = np.array(w['origin']); U = np.array(w['u'])
        for f in w['faces']:
            p = pos[bidx[f]]
            du = np.clip((p - O) @ U / w['width_m'], 0, 1)
            dv = np.clip((p[:, 2] - O[2]) / w['height_m'], 0, 1)
            uv[f, :, 0] = (x + du * wd) / ATLAS_PX
            uv[f, :, 1] = (y + (1 - dv) * ht) / ATLAS_PX
            face_atlas[f] = r['atlas']
            fnrm[f] = w['normal']
            wall_id[f] = w['id']
    np.savez_compressed(OUT + '/wallmap.npz', face_atlas=face_atlas, uv=uv,
                        fnrm=fnrm, wall_id=wall_id,
                        atlas_files=np.array([os.path.basename(f) for f in files],
                                             object))
    # ---- the note --------------------------------------------------------- #
    A = np.array([bywall[r['id']]['area_m2'] for r in recs])
    ph = np.array([r.get('texture_pass', '').startswith('stage 60') for r in recs])
    pf = np.array([r.get('photo_texel_frac', 0.0) for r in recs])
    seam_b = [r['seam_before'] for r in recs if r.get('seam_before')]
    seam_a = [r['seam_after'] for r in recs if r.get('seam_after')]
    note = dict(s35pack, **dict(
        stage35_reference=REF,
        walls=len(recs), atlases=npages, atlas_px=ATLAS_PX, atlas_bytes=abytes,
        pass_name='stage 60 best view per patch (wallphoto)',
        photo_walls=int(ph.sum()),
        photo_area_m2=round(float((A * np.where(ph, pf, 0)).sum()), 1),
        photo_walls_area_m2=round(float(A[ph].sum()), 1),
        fallback_walls=int((~ph).sum()),
        fallback_area_m2=round(float(A[~ph].sum()), 1),
        area_total_m2=round(float(A.sum()), 1),
        photo_texel_frac_median=round(float(np.median(pf[ph])), 3) if ph.any() else 0,
        walls_dropped_for_exposure=len(dropped_level),
        ship_walls_kept_as_material=nship[0],
        glass_walls_drawn_not_photographed=nglass[0],
        walls_dropped_for_exposure_detail=dropped_level[:12],
        walls_dropped_for_exposure_why=dropped_why,
        level_reference='stage 35, with its tone match divided back out',
        level_tone_undo_walls=len(undone),
        level_tone_undo_factor_median=round(float(np.median(
            [f for _, f in undone])), 3) if undone else None,
        level_tone_undo_tolerance=LEVEL_UNDO_TOL,
        # one building, one level: how far an unphotographed wall may sit from
        # the wall of its building stage 35 measured, once its own photograph's
        # exposure has been carried over
        level_building_consistency=dict(
            ratio=CONSIST_RATIO, buildings_measured=len(blevel),
            walls_held=len(consisted), detail=consisted[:12]),
        level_gain_limits=[LEVEL_MIN, LEVEL_MAX],
        level_luminance_floor_linear=LEVEL_FLOOR,
        # the bounded rescue for a photograph under that floor
        level_dark_gain_max=LEVEL_DARK_GAIN,
        level_dark_recovered=sorted(dark_ok),
        level_dark_checks=dict(local_sd_max=DARK_SD_MAX,
                               blown_max=DARK_BLOWN_MAX,
                               green_max=DARK_GREEN_MAX,
                               coverage_min=RETRY_PHOTO_MIN),
        level_gain_median=round(float(np.median(
            [np.mean(r['level_gain']) for r in recs if r.get('level_gain')])), 3)
        if any(r.get('level_gain') for r in recs) else None,
        cm_per_texel=dict(
            photo_median=round(float(np.median(
                [r['cm_per_texel'] for r in recs if r.get('texture_pass', '').startswith('stage 60')])), 2)
            if ph.any() else None,
            photo_p10=round(float(np.percentile(
                [r['cm_per_texel'] for r in recs if r.get('texture_pass', '').startswith('stage 60')], 10)), 2)
            if ph.any() else None,
            photo_p90=round(float(np.percentile(
                [r['cm_per_texel'] for r in recs if r.get('texture_pass', '').startswith('stage 60')], 90)), 2)
            if ph.any() else None,
            fallback_median=round(float(np.median(
                [r['cm_per_texel'] for r in recs if not r.get('texture_pass', '').startswith('stage 60')])), 2)),
        seam_step_sRGB=dict(before=round(float(np.median(seam_b)), 2) if seam_b else None,
                            after=round(float(np.median(seam_a)), 2) if seam_a else None,
                            walls=len(seam_b)),
        licence=dict(photo="Mapillary imagery CC BY-SA 4.0 (colour only)",
                     material="ambientCG CC0 1.0",
                     windows="window and door grid generated in s35_facadetex.py"),
        far_gate=dict(r_max_far_m=H.R_MAX_FAR, rects_under_the_far_gate=len(far_ids),
                      photo_min=FAR_PHOTO_MIN, green_max=FAR_GREEN_MAX,
                      dark_max=FAR_DARK_MAX,
                      kept=sorted(far_kept),
                      rejected={str(k): v for k, v in sorted(far_rejected.items())}),
        # Two-pass tone reference: every per_wall record
        # carries tile_mean_linear / tile_median_linear (and their
        # luminances), measured on the tile as packed into the page.  Stage 35
        # reads tile_mean_linear back on the NEXT run.
        tile_statistic=dict(
            fields=['tile_mean_linear', 'tile_median_linear',
                    'tile_mean_linear_luminance',
                    'tile_median_linear_luminance'],
            space='linear, wpcommon.to_linear of the packed sRGB tile',
            measured_on='the tile written into the atlas page, before JPEG',
            read_by='s35_facadetex.packed_tile_reference() on the next prep'),
        seconds=round(time.time() - t0)))
    note['per_wall'] = recs
    json.dump(note, open(OUT + '/facadetex.json', 'w'), indent=1)
    if outdir is None:      # a scratch pack is not this set's shipped state
        H.mark(setname, 'atlas',
               {k: v for k, v in note.items() if k != 'per_wall'})
    H.log(f"atlas[{setname}]: {npages} pages, {abytes/1e6:.1f} MB, "
          f"{int(ph.sum())} photographed walls, {round(time.time()-t0)}s")
    return {k: v for k, v in note.items() if k != 'per_wall'}


# --------------------------------------------------------- the retry pass --- #
def run_relevel(setname='walls', tries=LEVEL_RETRY_VIEWS, workers=4):
    """Give every gate-rejected wall its next panoramas, and keep the result
    only if it clears the same gates.

    Runs between bake and atlas.  For each wall the level gate or the far gate
    turned down, the views that built the failing tile are banned, ownership is
    re-solved for that rect alone and the tile re-baked; the new tile is put
    through both gates unchanged.  Up to `tries` rounds, so a wall gets at most
    its next three panorama sets.  A wall that never clears the gates is
    restored byte for byte -- its owner entry and its tile npz both go back --
    so this pass can only add photographed walls, never lose one."""
    t0 = time.time()
    D = H.setdir(setname)
    walls, s35pack, s35rec, s35_tile, REF = reference_pack()
    wbyid = {w['id']: w for w in walls}
    # the same building-consistency anchors run_atlas will level against, so
    # the gate the retry runs and the gate that ships cannot drift apart
    blevel = building_levels(walls, s35rec, s35_tile)
    try:
        _rk, _ = H.load_rects(setname)
        far_ids = {r.id for r in _rk if H.rect_rmax(r) > H.R_MAX}
    except Exception:
        far_ids = set()
    ownf = D + '/work/own.json'

    def gates(rid, floor=wpbake.PHOTO_MIN_AREA):
        """-> (why, meta) for the tile on disk; why is None when it ships.

        floor is PHOTO_MIN_AREA when asking whether the BAKED tile ships, and
        RETRY_PHOTO_MIN when asking whether a retried one may overwrite the
        stage 35 tile the wall already has."""
        tp = wpbake.tile_path(setname, rid)
        if not os.path.exists(tp):
            return 'no tile', None
        d = np.load(tp, allow_pickle=True)
        meta = json.loads(str(d['meta']))
        if not photo_floor_ok(meta, wbyid.get(rid), floor):
            return f"photograph covers {meta['photo_frac']:.2f} of the wall", meta
        if rid in far_ids:
            w = far_reject(d['photo'], d['got'], meta)
            if w:
                return w, meta
        rec = s35rec.get(rid, {})
        if 'rect_px' not in rec:
            return None, meta
        got = np.asarray(d['got'], bool)
        if not got.any():
            return 'no photo texels', meta
        _, _, _, _, why, _ = level_gain(
            rec, H.to_linear(d['photo']), got, s35_tile(rec),
            meta['photo_frac'],
            blevel.get(rec.get('part_name') or wbyid[rid].get('part')))
        return why, meta

    todo = []
    for w in walls:
        rid = w['id']
        if w['osm'].get('building') == 'ship':
            continue                      # a hull the DSM cannot occlude: no retry
        why, meta = gates(rid)
        if why is not None and meta is not None:
            todo.append((rid, why, list(meta.get('views') or [])))
    H.log(f'relevel[{setname}]: {len(todo)} rejected wall(s) to retry')
    own0 = json.load(open(ownf))
    recovered = []; failed = []
    for rid, why0, views0 in todo:
        keep0 = own0['per_rect'].get(str(rid))
        tp = wpbake.tile_path(setname, rid)
        bak = tp + '.retrybak'
        if os.path.exists(tp):
            os.replace(tp, bak)
        banned = set(int(v) for v in views0)
        won = None
        for k in range(tries):
            if not banned:
                break
            wpbake.run_own(setname, only={rid}, ban={rid: banned})
            if os.path.exists(tp):
                os.remove(tp)
            wpbake.run_bake(setname, workers=1, only={rid})
            why, meta = gates(rid, floor=RETRY_PHOTO_MIN)
            if meta is None or not meta.get('views'):
                break                     # nothing left that can own this wall
            if why is None:
                won = dict(round=k + 1, banned=sorted(banned),
                           views=[int(v) for v in meta['views']],
                           photo_frac=round(meta['photo_frac'], 4), was=why0)
                break
            banned |= set(int(v) for v in meta['views'])
        if won:
            recovered.append(dict(id=int(rid), **won))
            if os.path.exists(bak):
                os.remove(bak)
        else:
            # put the wall back exactly as the bake left it
            if keep0 is not None:
                cur = json.load(open(ownf))
                cur['per_rect'][str(rid)] = keep0
                json.dump(cur, open(ownf, 'w'), indent=1)
            if os.path.exists(tp):
                os.remove(tp)
            if os.path.exists(bak):
                os.replace(bak, tp)
            failed.append(dict(id=int(rid), reason=why0,
                               views_tried=sorted(banned)))
    rep = dict(set=setname, rejected=len(todo), recovered=len(recovered),
               tries=tries, retry_photo_min=RETRY_PHOTO_MIN,
               recovered_detail=recovered, still_rejected=failed,
               why=('views that lost a wall to the level gate or the far gate '
                    'are banned and the rect re-owned from its next panoramas; '
                    'the new tile faces the same gates, and a wall that still '
                    'fails is restored'),
               seconds=round(time.time() - t0))
    json.dump(rep, open(D + '/work/relevel.json', 'w'), indent=1)
    H.mark(setname, 'relevel', {k: v for k, v in rep.items()
                                if k not in ('recovered_detail', 'still_rejected')})
    H.log(f'relevel[{setname}]: {len(recovered)} of {len(todo)} recovered, '
          f'{rep["seconds"]}s')
    return rep
