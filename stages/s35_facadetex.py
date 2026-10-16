"""Stage 35 -- one texture per building wall: the photograph where it is good,
a real-scale material with a window grid everywhere else.

Stage 34 measured, per wall and per texel, the multi-view colour, how many
panoramas saw that texel and how far apart they were (MAD of luminance).  This
stage turns that into what the model actually shows:

  * a texel keeps its photograph when three or more panoramas saw it and they
    agreed to within the MAD cut taken from the pooled distribution;
  * every other texel gets a seamless CC0 material laid down at its real size,
    measured off the tile where the tile has a period and area-averaged down to
    the wall's own texel size before it is sampled, so what lands in the atlas
    is the material and not its aliasing;
  * a window and door grid is drawn only where the storey height is evidence:
    a building:levels count on the way, or a storey height measured off a
    photographed wall of the same building part.  A wall with neither gets no
    grid at all, and the panel joints of its class instead;
  * the material is tinted by the best colour evidence there is: the
    photograph's own local colour, else OSM building:colour, else the colour
    stage 20 measured on that wall's faces, else the class default.

Walls are packed into 4096 px atlases; stage 22 emits each wall as its own
quad with UVs into its atlas tile.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
import common as C
import texlib as T

FAC = T.SCR + '/facades'
PROJ = FAC + '/proj'
CC0 = T.SCR + '/cc0'
ATLAS_PX = 4096
PAD = 16                # texels of edge-replicated gutter round every tile; a
                        # 4-texel gutter lets the renderer's mips of the 4096 px
                        # atlas pull dark flecks off neighbouring tiles along
                        # every tile edge
# a synthesised wall is cheap to draw finely, so the walls a viewer can walk up
# to (within NEAR_R) get FALLBACK_TEXEL_NEAR texels and the ones farther out get
# common.FACADE_TEXEL_FAR (FALLBACK_TEXEL_FAR when common does not set it)
FALLBACK_TEXEL_NEAR = 0.04
FALLBACK_TEXEL_FAR = 0.08
FALLBACK_MAX_NEAR = 1024
FALLBACK_MAX_FAR = 512
NEAR_R = 120.0                 # m from the area origin (ENU 0,0)

CLASSES = ['brick', 'concrete', 'metal_panel', 'glass', 'wood']
# the CC0 tile each class lays down; the physical size of one tile is measured
# in tile_sizes().  The brick tile is square-unit masonry and the
# wood tile a 150 mm board; the concrete and metal tiles have no period at all,
# so their repeat is grain size, not a pattern, and their panel joints are drawn.
BASE_MAT = dict(brick='brick', concrete='concrete', metal_panel='metal',
                glass=None, wood='wood_deck')
CLASS_DEFAULT_RGB = dict(brick=(0.19, 0.10, 0.075), concrete=(0.30, 0.295, 0.285),
                         metal_panel=(0.34, 0.345, 0.35), glass=(0.10, 0.125, 0.15),
                         wood=(0.20, 0.145, 0.095))
PHOTO_TINT_COV = 0.15   # a wall this photographed can set its own colour
# ---- tone match ------------------------------------------------
# A wall the cameras never reached is tinted by OSM building:colour, by the
# colour stage 20 measured on its faces, or by the class default, and none of
# those three is an exposure.  An OSM colour such as #e8ead4 becomes linear
# (0.83, 0.84, 0.68), clamped to 0.62 luminance by plausible_wall_colour(): several
# times the albedo of a photographed wall it abuts, and yellow-green against
# neutral.  The photograph is the only exposure this pipeline has.  So a wall with
# no photograph of its own is put on the exposure of the photographed walls of its
# own building, failing that on the photographed walls within TONE_MATCH_R of it:
# reference luminance outright (the two surfaces stand in the same sunlight) and
# most of the way to the reference chromaticity, keeping TONE_KEEP_CHROMA of its
# own hue so red brick beside grey concrete stays red.
TONE_MATCH_R = 70.0          # m a wall may reach for a photographed neighbour
TONE_KEEP_CHROMA = 0.35      # share of its own chromaticity a wall keeps
TONE_REF_COV = 0.15          # trusted fraction a wall needs to be a reference
TONE_REF_MIN_TEXELS = 200    # ... and trusted texels, same bar as the tint
# ... and a width.  A corner sliver is not a sample of its building's light: a
# tower whose only photographed wall is a 0.8 m sliver in shade would otherwise put
# every other wall of the tower on that sliver and draw a flat dark facade.
TONE_REF_MIN_WIDTH_M = 1.5
TONE_LUM_FLOOR = 0.035       # a matched wall is never darker than this,
TONE_LUM_CEIL = 0.45         # ... nor brighter: a reference can be in shadow
# ... and never far from ITS OWN measured luminance either.  Taking the reference
# outright throws the wall's own evidence away: a wall at the 0.62 ceiling matched
# to a shaded facade at 0.05 is darkened 12x.  A shaded wall beside a sunlit one is
# a real 2x, so the cap is set outside that and only stops the multiple-fold moves.
TONE_MAX_DARKEN = 0.45       # a matched wall keeps at least this much of its
TONE_MAX_BRIGHTEN = 1.6      # own luminance, and gains at most this much
# ... unless the reference is not a neighbourhood average at all but the one
# photographed wall this one STANDS ON.  An unphotographed upper block above a
# photographed lower band reads as one building to the eye; a 70 m blend of many
# photographed walls, held by TONE_MAX_DARKEN, draws a visible step across that
# line.  A wall a few metres in front, facing the same way, is not a neighbour: it
# is the same surface in the same light, so it is taken before any
# distance-weighted average and the darken cap is relaxed to let it be taken nearly
# outright.
TONE_NEAR_R = 15.0           # m to the nearest point of the reference wall, NOT
                             # to its centre: on a wide wall the centre can be far
                             # while the plane is close, and the plane distance is
                             # the one that means something
TONE_NEAR_FACING = 0.866     # cos 30 deg between the two outward normals: two
                             # faces further apart than that can hold different
                             # sun, and a 45 deg chamfer is a different exposure
TONE_NEAR_MIN_AREA = 20.0    # m2 the same-facing evidence must add up to: a small
                             # photographed return in a facade can sit far darker
                             # than the facade it belongs to, and a sliver in a
                             # return is not its facade's exposure
TONE_MAX_DARKEN_NEAR = 0.30  # the darken cap when the reference came from that
                             # branch: low enough that a near reference is taken
                             # essentially outright, while multiple-fold moves
                             # are still stopped
# ... and the same relaxed cap applies when the synthetic mass SITS ON a
# photographed building.  OSM often maps a building's upper storeys as a second
# way whose footprint falls inside the photographed lower way: it is not a
# neighbour, it is the same building.  Walls of that upper way that face a
# direction nothing photographed faces within TONE_NEAR_R take the lower
# building's exposure over ALL its faces (the mass rule).  The wall planes stage 33
# merges are NOT a usable footprint (an even-odd test against merged wall segments
# can put a building's own centroid outside itself), so the OSM way geometry is
# what this reads.
TONE_MASS_INSIDE = 0.9       # share of this part's footprint vertices that has
                             # to fall inside the other's before the two are
                             # called one building mass
# Last of all, a wall with NOTHING photographed within TONE_MATCH_R keeps whatever
# the class default gives it and never goes through tone_match at all, so it
# carries no exposure.  The nearest photographed walls of the SAME CLASS are weak
# evidence (they are not in this wall's light and may not face its way), but a
# concrete wall's albedo is a property of concrete, and weak evidence beats a number
# from nowhere.  Facing is deliberately ignored: a wall with no photograph within
# 70 m has no same-facing neighbour either.  The reach must exceed TONE_MATCH_R, or
# the branch fires on nothing; a wall whose nearest same-class photograph is beyond
# TONE_CLASS_R stays unmatched.
TONE_CLASS_R = 100.0         # m, plane to centre, the distance wall_gap means
TONE_CLASS_N = 3             # nearest same-class photographed walls to take
# the two-pass tone reference -------------------------- #
# A photographed wall's TRUSTED-TEXEL MEDIAN is not what the wall renders at.  The
# stage 60 wallphoto pass levels the winning photograph against the multi-view
# consensus, blends it over the material fallback with a coverage alpha and
# re-tints the uncovered part, so the tile that ends up in the atlas is a different
# number (the texel statistic can sit 1.3-1.4x high), and the gap between reference
# and tile IS the step between matched and photographed walls.  So stage 35
# prefers, per reference wall, the FINAL PACKED TILE MEAN in linear light out of the
# PREVIOUS prep's pack, and only falls back to the trusted-texel median where the
# pack has no such record (a wall the pack dropped, or a first run with no pack).  It
# converges: each prep matches against the tile the last prep actually shipped, and
# pipeline/area_cli.py runs s35 -> s60 bake -> relevel -> atlas twice, so the
# convergence lands inside one build.
PACK_REF = (os.environ.get('S35_PACK_REF')
            or T.SCR + '/wallphoto/walls/out/facadetex.json')
TONE_TILE_STAT = 'tile_mean_linear'   # the MEAN, not the median: the median of
                             # a tile with a generated window grid in it is the
                             # wall between the windows, while what the eye
                             # reads across a facade at 30 m is the mean (a tile
                             # with a bright roofline can differ by half)
TONE_TILE_MIN_LUM = TONE_LUM_FLOOR   # a packed tile this dark is a failed bake,
                             # not an exposure: fall back to the texel median
                             # rather than hand it the relaxed cap below
# ... and the cap has to move with it, or the new reference changes nothing.  A
# wall entering tone_match() at the plausible_wall_colour ceiling (0.62) is floored
# by TONE_MAX_DARKEN_NEAR at 0.186, which can sit above the packed tile it should
# match; TONE_MAX_DARKEN_TILE = 0.20 floors it at 0.124 instead.
# This cap is NOT a general loosening.  It applies only where BOTH hold:
#   * the branch is one of the close-evidence branches that already carry
#     TONE_MAX_DARKEN_NEAR (same3 / near / mass) -- the neighbourhood averages
#     `area` and `class`, and a lone `same`, keep TONE_MAX_DARKEN = 0.45; and
#   * every reference that went into the colour carried a MEASURED PACKED TILE
#     MEAN (TONE_TILE_MIN_FRAC of the branch's own weight), so the number the
#     cap is being relaxed against is the rendered value and not a texel
#     statistic that is known to sit high.
# The black-wall protections still apply: the reference is clipped to
# TONE_LUM_FLOOR = 0.035 in tone_match(), and both near_facing_reference() and
# mass_reference() refuse a reference at or below that floor outright.  Floor of
# this path in the worst case: a wall entering at the 0.045 plausible_wall_colour
# floor cannot be darkened below 0.20 x 0.045 = 0.009, and tone_match's own clip to
# TONE_LUM_FLOOR means the TARGET never goes under 0.035 anyway.
TONE_MAX_DARKEN_TILE = 0.20
TONE_TILE_MIN_FRAC = 0.999   # share of the branch's weight that must be
                             # tile-backed before the relaxed cap is used
# The synthetic wall is not flat-lit either: it gets a small ambient-occlusion
# gradient, dark where it meets the ground and along its own vertical edges.
# Mean 1, so it costs the tone match nothing.
AO_FOOT = 0.16
AO_FOOT_M = 2.2
AO_EDGE = 0.07
AO_EDGE_M = 1.2
# Metre-scale structure for the classes whose CC0 tile has none, from the same
# table stage 22 uses on the ground (s22_composite.MACRO): form-panel joints,
# a pour/weathering blotch and rain streaks.  A wall that takes this does NOT
# also get panel_field()'s joints -- one joint grid per wall, at the physical
# pitch MACRO states.
MACRO_CLASS = dict(concrete='concrete', metal_panel='metal')
# A window is not a hole in the wall.  synth_facade() hands back a MODULATION
# field -- mean luminance exactly 1, clipped to [0.02, 6] -- and the caller
# multiplies it by the wall's tint, so an opening's final linear colour is
# tint x field.  On a wall the tone match put at TONE_LUM_FLOOR that product is
# tiny: on a wall tinted at the floor (luminance 0.035) the window texels come
# out at mean 0.0216 linear, 37 % of them under 0.02 and the darkest at 0.0061
# -- black rectangles.
# Glass has a colour of its own, and it is ABSOLUTE, not a fraction of the wall
# it sits in: a dark blue-grey pane plus the ambient light the room behind it
# sends back.  So that base goes under the opening texels as an absolute
# linear FLOOR after the tint multiply -- shaped by the opening's own
# structure, so the pane-to-pane brightness, the sky gradient up the pane, the
# reveal and the frame all survive the floor instead of being flattened by it
# -- and a hard absolute minimum goes under that, so no pane can reach black.
# A floor and not a second multiply, and not an addition either: a wall bright
# enough to draw its windows properly already is left exactly as it is
# (on a wall at tint luminance 0.279 the opening mean luminance is 0.16893
# with and without the floor; no texel moves).
# Non-metallic and low roughness is the other half of a dark-glass read; the
# wall atlas carries base colour only, with no per-wall MR map to put it in, so
# it is not expressible here (stage 22's material() sets the wall material's
# metallic and roughness for every wall at once).
GLASS_BASE_RGB = (0.050, 0.060, 0.080)   # linear, the target dark-glass colour
GLASS_AMBIENT_RGB = (0.018, 0.021, 0.028)  # ... and the interior ambient floor
RANGE_CUT = 30.0       # m; beyond this a 1 m pose error is under 2 deg and the
                        # multi-view median stops separating a tree in front of a
                        # wall from the wall
# how much of the CC0 tile's own colour survives; the rest comes from the tint,
# which is the measured evidence.  A brick keeps its hue, a painted plank does not.
KEEP_CHROMA = dict(brick=0.70, concrete=0.25, metal_panel=0.20, glass=0.35, wood=0.45)
OSM_MATERIAL = {'brick': 'brick', 'bricks': 'brick', 'concrete': 'concrete',
                'cement': 'concrete', 'stone': 'concrete', 'masonry': 'brick',
                'metal': 'metal_panel', 'steel': 'metal_panel',
                'aluminium': 'metal_panel', 'glass': 'glass', 'wood': 'wood',
                'timber': 'wood', 'timber_framing': 'wood', 'plaster': 'concrete'}


# ------------------------------------------------------------ tile scale -- #
def measure_period(path, axis, lag_min=12):
    """Fundamental period of a CC0 tile along one axis, as a fraction of the tile.

    The mean profile across the other axis is autocorrelated at the tile's own
    resolution.  A lag only counts as the fundamental when its second harmonic
    is there too (ac[2*lag] at least half of ac[lag]): a material with no period
    still has a fat autocorrelation bump at half the image from its overall
    shading, and that bump has no harmonic behind it.  Returns (None, r) when
    the tile has no period, which is the answer for concrete and for sheet metal.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(path).convert('L'), 'f4')
    p = a.mean(axis=axis)
    n = len(p)
    p = p - p.mean()
    ac = np.correlate(p, p, 'full')[n - 1:]
    ac = ac / max(ac[0], 1e-9)
    cand = []
    for lag in range(lag_min, n // 4):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.25:
            if ac[2 * lag] > 0.5 * ac[lag]:            # the harmonic confirms it
                cand.append((lag, float(ac[lag])))
    if not cand:
        pk = [float(ac[l]) for l in range(lag_min, n // 4)
              if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1]]
        return None, max(pk) if pk else 0.0
    best = max(cand, key=lambda t: t[1])
    # prefer the smallest lag that is itself a sub-multiple of the best one and
    # nearly as strong, so a pattern is not read at twice its true period
    for lag, q in sorted(cand):
        if best[0] % lag <= 2 and q > 0.8 * best[1]:
            best = (lag, q)
            break
    return best[0] / float(n), best[1]


# what one repeat of a material is a picture of, in metres.  The period is
# measured off the tile; these are the real objects that period can be.
UNIT_M = dict(brick_course=0.065, brick_block=0.20, board=0.15,
              form_panel=2.4, cladding_panel=1.0)


def tile_sizes(idx):
    """Physical size of one repeat of each base material.

    Measured where the material has a period, chosen where it has none.  Both
    axes are measured, because what the period *is* depends on whether the two
    agree: a running-bond brick has a short vertical period (the course) and a
    long horizontal one (the brick), whereas a tile whose two periods are equal
    is a square unit and reading its vertical period as a 65 mm course would
    draw the whole wall as a 65 mm mosaic."""
    out = {}
    for name, kind in (('brick', 'masonry'), ('wood_deck', 'board'),
                       ('concrete', 'panel'), ('metal', 'panel')):
        f = idx['materials'][name]['files']['color']
        pv, qv = measure_period(f, 1)                  # rows    -> vertical
        ph, qh = measure_period(f, 0)                  # columns -> horizontal
        seen = (f"vertical period {('%.4f' % pv) if pv else 'none'} (r={qv:.2f}), "
                f"horizontal {('%.4f' % ph) if ph else 'none'} (r={qh:.2f}) "
                f"of the tile")
        if kind == 'masonry' and pv and ph and abs(pv - ph) < 0.12 * pv:
            out[name] = dict(tile_m=round(UNIT_M['brick_block'] / pv, 3),
                             period_v=round(pv, 4), period_h=round(ph, 4),
                             measured=True,
                             why=f"{seen}: the unit cell is square, so this is "
                                 f"square masonry, not a running-bond course; "
                                 f"one unit is a 200 mm block")
        elif kind == 'masonry' and pv:
            out[name] = dict(tile_m=round(UNIT_M['brick_course'] / pv, 3),
                             period_v=round(pv, 4), period_h=ph, measured=True,
                             why=f"{seen}: one vertical repeat is a 65 mm course")
        elif kind == 'board' and pv:
            out[name] = dict(tile_m=round(UNIT_M['board'] / pv, 3),
                             period_v=round(pv, 4), period_h=ph, measured=True,
                             why=f"{seen}: one vertical repeat is a 150 mm board")
        else:
            u = UNIT_M['form_panel'] if name == 'concrete' else UNIT_M['cladding_panel']
            out[name] = dict(tile_m=u, period_v=pv, period_h=ph, measured=False,
                             why=f"{seen}: no intrinsic period, so the tile is "
                                 f"grain, not a pattern.  One repeat is set to "
                                 f"{u} m, which is what the grain size then is; "
                                 f"the panel joints themselves are drawn, not "
                                 f"taken from the tile")
    return out


def load_base(idx, name, sizes, keep_chroma=0.3):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(idx['materials'][name]['files']['color']).convert('RGB')
    if max(im.size) > 1024:
        im = im.resize((1024, int(round(im.size[1] * 1024 / im.size[0]))), Image.LANCZOS)
    a = np.asarray(im, 'f4') / 255.0
    lin = np.power(a, 2.2)
    g = lin @ T.LUMW
    lin = keep_chroma * lin + (1 - keep_chroma) * g[:, :, None]
    return lin.astype('f4'), sizes[name]['tile_m']


_MIP = {}


def mip(base, tw, th):
    """Area-average a base tile down to tw x th, cached."""
    from PIL import Image
    k = (id(base), tw, th)
    if k in _MIP:
        return _MIP[k]
    H, W = base.shape[:2]
    if tw >= W and th >= H:
        _MIP[k] = base
        return base
    ch = [np.asarray(Image.fromarray(base[:, :, c]).resize((tw, th), Image.BOX), 'f4')
          for c in range(3)]
    out = np.stack(ch, -1)
    _MIP[k] = out
    return out


def tiled(base, tile_m, w_m, h_m, nx, ny):
    """Sample a seamless tile over a wall at its true physical size.

    The tile is area-averaged down to the wall's own texel size first.  A 1024 px
    tile laid down at 2.4 m on a wall drawn at 8 cm per texel is otherwise point
    sampled once every 34 source pixels, and what lands in the atlas is the
    tile's per-pixel noise rather than the tile: measured on one wall, point
    sampling gives the concrete 3.8 sRGB levels of texel-to-texel noise (11 cm
    texels, 47 source pixels each) where the area-averaged tile has 1.4, and
    what is left of the tile at that point is sand, not concrete.  Averaging
    before sampling is what a mip level does."""
    H, W = base.shape[:2]
    # source pixels that fall inside one output texel, per axis
    fx = W * (w_m / max(nx, 1)) / tile_m
    fy = H * (h_m / max(ny, 1)) / tile_m
    if fx > 1.3 or fy > 1.3:
        base = mip(base, int(np.clip(round(W / max(fx, 1.0)), 4, W)),
                   int(np.clip(round(H / max(fy, 1.0)), 4, H)))
        H, W = base.shape[:2]
    u = (np.arange(nx) + 0.5) / nx * (w_m / tile_m) * W
    v = (np.arange(ny) + 0.5) / ny * (h_m / tile_m) * H
    ui = np.mod(u.astype(np.int64), W)
    vi = np.mod(v.astype(np.int64), H)
    return base[vi[:, None], ui[None, :]]


# -------------------------------------------------------- window geometry -- #
def levels_of(w):
    lv = w['osm'].get('levels')
    try:
        lv = int(float(lv))
    except (TypeError, ValueError):
        lv = None
    h = w['height_m']
    if lv is None or lv < 1:
        lv = max(1, int(round(h / 3.6)))
    return lv


def window_rows_from_photo(col, trust, h_m):
    """Storey height and sill phase read off a photographed wall.

    The rows of a facade are the darkest rows of the image: glass is darker
    than the wall around it.  Autocorrelating the row profile gives the storey
    height directly, which is what a blind wall of the same building needs."""
    if trust.mean() < 0.25 or col.shape[0] < 24 or h_m < 4.0:
        return None
    L = (col @ T.LUMW)
    wgt = trust.astype('f4')
    prof = (L * wgt).sum(1) / np.maximum(wgt.sum(1), 1e-6)
    ok = wgt.sum(1) > 0.35 * col.shape[1]
    if ok.sum() < 20:
        return None
    prof = np.where(ok, prof, np.nan)
    prof = np.interp(np.arange(len(prof)), np.nonzero(ok)[0], prof[ok])
    prof = prof - prof.mean()
    if prof.std() < 1e-4:
        return None
    ac = np.correlate(prof, prof, 'full')[len(prof) - 1:]
    ac /= max(ac[0], 1e-9)
    px_m = col.shape[0] / h_m
    lo = int(2.4 * px_m); hi = int(min(5.5 * px_m, len(ac) - 2))
    if hi <= lo + 2:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < 0.18:
        return None
    hs = lag / px_m
    # phase: where the dark band (the window) sits inside the storey
    k = np.arange(len(prof)) % lag
    dark = np.bincount(k, weights=prof, minlength=lag) / np.maximum(np.bincount(k, minlength=lag), 1)
    phase = float(np.argmin(dark)) / lag
    return dict(storey_m=round(float(hs), 3), phase=round(phase, 3),
                strength=round(float(ac[lag]), 3))


# A blind wall is still a built surface.  Joint and panel contrasts are a
# fraction of the wall's own sRGB value, so they scale with how light it is.
JOINT_SRGB = 0.035      # the joint line itself, 3.5 % -> ~5 levels on a mid wall
PANEL_SRGB = 0.012      # panel-to-panel difference of a class with no entry
                        # in PANEL_LOOK below; the classes that have one use
                        # their own figure
LIFT_SRGB = 0.014       # horizontal pour/lift line, weaker than the vertical one

# A joint grid drawn on a flat plane is a line, not an edge between
# two pieces of material.  With stage 22's MACRO field alone a metal_panel wall
# carries only a 2-texel joint about every metre and a 1.2 m weathering blotch;
# nothing says the squares between the joints are separate panels.  The three terms below say so, each as a
# fraction of the wall's own sRGB value so they scale with how light it is:
#   tone  largest panel-to-panel departure from the wall mean.  A hung metal
#         panel differs from its neighbour by its coil batch and by how it
#         faces the sky.  Cast concrete varies less between pours of the
#         same wall, glass spandrel less again.
#   bow   peak-to-peak bow across one panel's face.  Sheet cladding oil-cans;
#         a cast panel does not, and glass does not at all.  Chosen, not
#         measured.
#   band  darkening at a storey line, drawn ONLY where the OSM way carries
#         building:levels -- on a wall that takes the blind branch the level
#         count is the only evidence left that says where a floor is.
#   joint_mm  width of the joint panel_field() draws, for the classes that do
#         not take stage 22's MACRO field.  Same numbers as s22_composite.MACRO
#         for the two classes that do.
# tone is a few percent of the wall's own sRGB value.  A dark metal_panel wall
# sits near sRGB 58, where 3 % is 1.7 levels and the panels are not visible;
# above about 10 % the wall reads patchy rather than panelled, so metal_panel
# uses 5.5 %.  Concrete and wood sit at sRGB 130-140, where the same fraction
# is 2.6-2.8 levels, so they need less.
PANEL_LOOK = {
    'metal_panel': dict(tone=0.055, bow=0.028, band=0.030, joint_mm=10.0),
    'concrete':    dict(tone=0.026, bow=0.008, band=0.022, joint_mm=16.0),
    'glass':       dict(tone=0.016, bow=0.000, band=0.030, joint_mm=20.0),
    'wood':        dict(tone=0.022, bow=0.006, band=0.020, joint_mm=6.0),
}


def blind_pitch(cls, sizes_m):
    """Width of one panel of this class, metres, or None when the class's own
    material already carries its joints."""
    if cls == 'concrete':
        return sizes_m['concrete']['tile_m']        # one concrete form panel
    if cls == 'metal_panel':
        return sizes_m['metal']['tile_m']           # one cladding panel
    if cls == 'glass':
        return 1.5                                  # a curtain-wall mullion bay
    return None                    # brick and boarding have joints of their own


def panel_field(xx, yy, pitch, rng, lift=True, cls=None, band_m=None):
    """Panel joints on BOTH axes at `pitch`, as a multiplicative linear-RGB
    field of mean exactly 1.

    A blind wall with nothing on it reads as a painted plane, because nothing
    in it has a size.  Real cast concrete is poured against panels and real
    cladding is hung in panels, so the wall gets a joint line at the panel
    pitch, a panel-to-panel brightness step, a shallow bow across each panel's
    face and a faint horizontal lift line.  Everything here is a few sRGB
    levels: the point is that the eye finds a scale on the wall, not that it
    sees stripes.

    Three properties matter.  The joint grid runs on both axes, not just
    the vertical one: a wall of 1 m panels with only vertical joints is a wall
    of storey-high strips, which is not what cladding is.  The tone step is per
    PANEL (a 2-D index), not per column, so two panels one above the other are
    not forced to the same tone.  And the field is normalised to mean exactly
    1, like wall_ao() and s22_composite.macro_detail(), so it cannot move the
    tile's mean linear luminance away from what the tone match set.

    `band_m` draws a darker band at each storey line and is passed ONLY for a
    wall whose OSM way carries building:levels; with no level count the storey
    height would be invented and no band is drawn.  Mean-1 normalisation makes
    the band cost the rest of the wall a little brightness rather than darken
    the wall as a whole.

    Mean 1 also means this function's mean is unchanged by any of its terms, so
    the classes that take it can be retuned without touching the tone match.
    """
    nx = len(xx); ny = len(yy)
    look = PANEL_LOOK.get(cls) or dict(tone=2 * PANEL_SRGB, bow=PANEL_SRGB,
                                       band=0.0, joint_mm=12.0)
    tex_x = float(xx[1] - xx[0]) if nx > 1 else 0.05
    tex_y = float(yy[1] - yy[0]) if ny > 1 else 0.05
    px = xx / pitch
    py = yy / pitch
    # a joint is the wider of its real width and ~1.5 texels: below that it
    # aliases to nothing at this wall's texel size and the pitch is lost
    wx = max(0.75 * tex_x, 0.5 * look['joint_mm'] * 1e-3) / pitch
    wy = max(0.75 * tex_y, 0.5 * look['joint_mm'] * 1e-3) / pitch
    lx = np.exp(-0.5 * (np.abs(px - np.round(px)) / wx) ** 2)
    ly = np.exp(-0.5 * (np.abs(py - np.round(py)) / wy) ** 2)
    line = np.maximum(lx[None, :], ly[:, None])      # one grid, not two sums
    ix = np.floor(px).astype(int); ix = ix - ix.min()
    iy = np.floor(py).astype(int); iy = iy - iy.min()
    jit = rng.standard_normal((int(iy.max()) + 2, int(ix.max()) + 2))
    step = 0.5 * look['tone'] * np.clip(jit, -2, 2)[iy][:, ix]
    # a panel is very slightly brighter down its middle than at its edges
    bow = 0.5 * look['bow'] * (np.cos(2 * np.pi * (px - np.floor(px)))[None, :]
                               + np.cos(2 * np.pi * (py - np.floor(py)))[:, None])
    s = JOINT_SRGB * line + step + bow
    if lift:
        wl = max(0.75 * tex_y / pitch, 0.004)
        s = s + LIFT_SRGB * np.exp(
            -0.5 * (np.abs(py - np.round(py)) / wl) ** 2)[:, None]
    if band_m and band_m > 0.6:
        pb = yy / float(band_m)
        wb = max(2.5 * tex_y, 0.09)
        s = s + look['band'] * np.exp(
            -0.5 * (np.abs(pb - np.round(pb)) * float(band_m) / wb) ** 2)[:, None]
    f = np.power(np.clip(1.0 - s, 0.4, 1.6), 2.2)
    return (f / max(float(f.mean()), 1e-6)).astype('f4')[:, :, None]


def far_sig(w):
    """stage 34's per-wall cache signature (s34_wallproj.wall_sig)."""
    return '%s|%s|%s|%.2f|%.2f' % (
        w['part_name'], ','.join('%.2f' % x for x in w['origin']),
        ','.join('%.4f' % x for x in w['u']), w['width_m'], w['height_m'])


def storey_evidence(w, rows):
    """'building:levels', 'photograph' or None: what licenses a window grid.
    Stage 35 and the stage 60 material fallback both ask this, so a wall does
    not lose its windows by being partially photographed."""
    if w['osm'].get('levels'):
        return 'building:levels'
    return 'photograph' if rows else None


# ------------------------------------------------ facade modules -- #
# common.py FACADE_PBR / FACADE_MOD.  A generated wall needs, in order of visual
# weight: openings with a frame, reveal, mullion and sill (a flat tinted
# rectangle reads as a sticker); per-building window and glass variation; a
# ground storey (an entrance, or a shopfront where the OSM tags say so);
# something at the foot and top of the wall (plinth, dirt, string course,
# parapet); and relief on blind walls beyond a joint grid.  The functions below
# build the wall from these modules and return, beside the colour field, a
# height field (metres) and a roughness.
SHOP_AMENITY = ('restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'ice_cream',
                'bank', 'pharmacy')


def building_style(w, cls):
    """Per-building module choices seeded by the OSM id, never the wall id:
    every wall of a building shares its window, two buildings differ."""
    M = C.FACADE_MOD
    oid = int(w.get('osm_id') or 0)
    r = np.random.default_rng((oid * 2654435761 + 35) % (2 ** 32))
    names = [m[0] for m in M['mull']]
    p = np.array([m[1] for m in M['mull']], 'f8'); p /= p.sum()
    return dict(
        win_w=float(r.uniform(*M['win_w_frac'])), win_h=float(r.uniform(*M['win_h_frac'])),
        mull=str(r.choice(names, p=p)),
        frame_i=int(r.integers(len(M['frame_rgb']))), glass_i=int(r.integers(len(M['glass_rgb']))),
        band=bool(r.random() < M['band_p']), blind_p=float(r.uniform(*M['blind_p'])),
        dirt=float(r.uniform(*M['dirt_amt'])), streak=float(r.uniform(*M['streak_amt'])),
        curtain_band=bool(r.random() < M['curtain_band_p']), seed=int(r.integers(1 << 30)))


def frontage(w):
    """What the OSM way says the ground storey is: 'shop', 'entrance' or None."""
    M = C.FACADE_MOD; o = w['osm']; b = o.get('building'); am = o.get('amenity')
    am = None if am in (None, 'shelter') else am
    if b in M['shop_building'] or (M['shop_amenity_any'] and am in SHOP_AMENITY):
        return 'shop'
    if b in M['entrance_building'] or (M['entrance_amenity_any'] and am):
        return 'entrance'
    return None


def _boxf(a, k):
    from scipy.ndimage import uniform_filter
    return uniform_filter(a, max(1, int(k)), mode='wrap')


def tile_relief(img, tex_m):
    """Micro relief of the material itself (mortar, board grooves): the high
    pass of the CC0 tile's log luminance, darker = deeper."""
    L = np.log(np.maximum(np.asarray(img, 'f4') @ T.LUMW, 1e-3))
    hp = L - _boxf(L, max(3, round(0.25 / max(tex_m, 1e-3))))
    return (C.FACADE_MOD['tile_relief_m'] * np.clip(hp / 0.35, -1.5, 1.5)).astype('f4')


def dirt_field(yy, h_m, nx, S, rng, tex_m):
    """(ny, nx, 3) multiplier, darker and browner toward the foot, with rain
    runs from the parapet down; plus the roughness it adds."""
    M = C.FACADE_MOD
    e = np.exp(-yy / M['dirt_m'])[:, None]
    k = max(2.0, 0.35 / max(tex_m, 1e-3))
    n0 = rng.standard_normal(int(nx / k) + 2)
    col = np.clip(np.interp(np.arange(nx) / k, np.arange(len(n0)), n0), -2, 2)[None, :]
    top = np.exp(-(h_m - yy) / 3.0)[:, None]
    lum = 1.0 - S['dirt'] * e - 0.5 * S['streak'] * top * np.maximum(col, 0)
    dr = np.asarray(M['dirt_rgb'], 'f4'); dr = dr / (dr @ T.LUMW)
    tint = 1.0 + (dr[None, None, :] - 1.0) * (0.6 * e)[:, :, None]
    return (lum[:, :, None] * tint).astype('f4'), (M['rough_dirt'] * e * np.ones((1, nx))).astype('f4')


def module_facade(w, cls, img, xx, yy, hs, nst, bw, nbay, phase, front, rng,
                  pbr_out, openings_out):
    """The window branch of synth_facade() built from modules.  Returns the
    colour modulation field (mean luminance 1 over the WALL texels, so the tone
    match holds) and fills pbr_out with height (m), rough, abs_rgb, abs_a: the
    absolute linear colour of frames, panes and doors, composited over the
    tinted wall by apply_modules()."""
    M = C.FACADE_MOD; S = building_style(w, cls)
    ny, nx = img.shape[:2]; h_m = w['height_m']; tex = w['width_m'] / max(nx, 1)
    prng = np.random.default_rng(S['seed'] + 7 * int(round(w['width_m'] * 10)))
    stor = np.clip(np.floor(yy / hs).astype(int), 0, nst - 1)
    bay = np.clip(np.floor(xx / bw).astype(int), 0, nbay - 1)
    YI = (yy - stor * hs)[:, None]; XI = (xx - bay * bw)[None, :]
    Y = yy[:, None] * np.ones((1, nx)); ST = stor[:, None]; BY = bay[None, :]
    H = np.zeros((ny, nx), 'f4'); A = np.ones((ny, nx), 'f4')
    R = np.full((ny, nx), M['rough'].get(cls, 0.85), 'f4')
    frame_rgb = np.asarray(M['frame_rgb'][S['frame_i']], 'f4')
    glass_rgb = np.asarray(M['glass_rgb'][S['glass_i']], 'f4')
    npane = (nst + 1) * (nbay + 1)
    ptype = prng.random(npane); pv = prng.uniform(0.8, 1.25, npane)
    pblind = prng.uniform(0.15, 0.85, npane)
    pid = (ST * (nbay + 1) + BY)
    mid = nbay // 2
    # what a facade of glass reflects varies at the scale of the sky and of the
    # buildings opposite, not per pane: a 4-12 m field plus a gain up the wall
    ky = max(2, int(ny / max(h_m / 6.0, 1))); kx = max(2, int(nx / max(w['width_m'] / 8.0, 1)))
    lf = prng.standard_normal((ny // ky + 3, nx // kx + 3)).astype('f4')
    from scipy.ndimage import zoom as _zoom
    lf = _zoom(lf, (ky, kx), order=3)[:ny, :nx]
    if lf.shape != (ny, nx):
        lf = np.pad(lf, ((0, ny - lf.shape[0]), (0, nx - lf.shape[1])), mode='edge')
    big = np.clip(1.0 + 0.22 * lf + 0.25 * (yy / max(h_m, 1.0))[:, None] - 0.12, 0.55, 1.6)
    info = dict(module_style=dict(mull=S['mull'], band=S['band'], frame=S['frame_i'],
                                  glass=S['glass_i'], curtain_band=S['curtain_band']),
                frontage=front)
    if cls == 'glass':
        # curtain wall: vision glass between mullions, spandrel bands or full glazing
        if S['curtain_band']:
            vis = (YI > 0.16 * hs) & (YI < 0.86 * hs)
            v = np.clip((YI - 0.16 * hs) / (0.70 * hs), 0, 1)
        else:
            vis = (YI > 0.05) & (YI < hs - 0.05)
            v = np.clip(YI / hs, 0, 1)
        vis = vis & np.ones((1, nx), bool)
        p = bw / max(1, round(bw / M['curtain_mull_m']))
        xm = np.abs(((xx + 0.5 * p) % p) - 0.5 * p)[None, :]
        hm = max(0.5 * M['mull_m'], 0.5 * tex)
        mull = (xm < hm) | (np.abs(YI - 0.16 * hs) < hm) | (np.abs(YI - 0.86 * hs) < hm)
        mull = mull & np.ones((ny, 1), bool)
        cell = ST * 10007 + np.floor(xx / p).astype(int)[None, :]
        cv = np.random.default_rng(S['seed']).uniform(0.92, 1.08, int(cell.max()) + 2)[cell]
        refl = 0.8 * (0.62 + 0.6 * v ** 1.5) * cv * big
        bl = (np.random.default_rng(S['seed'] + 1).random(int(cell.max()) + 2)[cell] < S['blind_p'])
        refl = np.where(bl & (v > 0.55), M['curtain_blind'], refl)
        A = np.where(vis, refl, A * (1.0 + 0.03 * (cv - 1.0)))
        H = np.where(vis, -0.05, 0.0).astype('f4')
        H = np.where(mull, 0.035, H)
        R = np.where(vis, M['rough_curtain'], np.float32(0.35)).astype('f4')
        R = np.where(mull, M['rough_frame'], R)
        absa = mull.astype('f4'); absrgb = np.broadcast_to(frame_rgb, (ny, nx, 3)).copy()
        inside = vis
        wall = ~vis & ~mull
        info['shopfront'] = False
    else:
        wh = float(np.clip(S['win_h'] * hs, 1.0, 2.6))
        sill = float(np.clip(phase * hs - wh / 2, 0.6, hs - wh - 0.3))
        hw = 0.5 * float(np.clip(S['win_w'] * bw, 0.8, 2.8))
        X0 = np.full((ny, nx), hw, 'f4'); Y0 = np.full((ny, nx), sill, 'f4'); Y1 = np.full((ny, nx), sill + wh, 'f4')
        g = (ST == 0) & np.ones((1, nx), bool)
        shop = front == 'shop' and hs > 2.7
        door = np.zeros((ny, nx), bool)
        if shop:
            gh = float(np.clip(M['shop_h_frac'] * hs, 2.2, 3.6))
            X0 = np.where(g, 0.46 * bw, X0); Y0 = np.where(g, M['bulkhead_m'], Y0)
            Y1 = np.where(g, M['bulkhead_m'] + gh, Y1)
        if front in ('shop', 'entrance') and w['width_m'] > 5.0:
            door = g & (BY == mid)
            X0 = np.where(door, 0.5 * M['door_w_m'], X0); Y0 = np.where(door, 0.0, Y0)
            Y1 = np.where(door, M['door_h_m'], Y1)
        dx = np.abs(XI - 0.5 * bw) - X0
        dy = np.maximum(Y0 - YI, YI - Y1)
        d = np.maximum(dx, dy)
        inside = d < 0
        glass = d <= -max(M['frame_m'], 0.9 * tex)
        frame = inside & ~glass
        v = np.clip((YI - Y0) / np.maximum(Y1 - Y0, 1e-3), 0, 1)
        half = max(0.5 * M['mull_m'], 0.5 * tex)
        mv = np.abs(XI - 0.5 * bw) < half
        if shop:
            # shopfront: a mullion every ~1.2 m and a transom at door head height
            q = 2 * X0 / np.maximum(np.round(2 * X0 / 1.2), 1)
            uq = np.abs(((XI - 0.5 * bw + X0) % q) - 0.5 * q) > 0.5 * q - half
            mshop = g & (uq | (np.abs(YI - (M['bulkhead_m'] + 0.78 * (Y1 - Y0))) < half))
        else:
            mshop = np.zeros((ny, nx), bool)
        if S['mull'] == 'v':
            mu = mv
        elif S['mull'] == 'cross':
            mu = mv | (np.abs(v - 0.5) * (Y1 - Y0) < half)
        elif S['mull'] == 't':
            mu = mv & (v < 0.72) | (np.abs(v - 0.72) * (Y1 - Y0) < half)
        else:
            mu = np.zeros((ny, nx), bool)
        mu = np.where(g & shop, mshop, mu)
        mu = np.where(door, (np.abs(XI - 0.5 * bw) < half) | (np.abs(YI - (M['door_h_m'] - 0.45)) < half), mu)
        mull = glass & mu
        glass = glass & ~mull
        # sill under every window that is not a shopfront or a door
        sl = (~(g & shop)) & ~door & (YI < Y0) & (YI > Y0 - M['sill_h_m']) \
            & (np.abs(XI - 0.5 * bw) < X0 + M['sill_over_m'])
        run = (~(g & shop)) & ~door & (YI < Y0 - M['sill_h_m']) \
            & (YI > Y0 - M['sill_h_m'] - M['streak_m']) & (np.abs(XI - 0.5 * bw) < X0)
        k = max(1.0, 0.08 / max(tex, 1e-3))
        n0 = np.abs(prng.standard_normal(int(nx / k) + 2))
        colz = np.interp(np.arange(nx) / k, np.arange(len(n0)), n0)[None, :]
        A = A * np.where(run, 1.0 - S['streak'] * colz
                         * np.exp(-(Y0 - M['sill_h_m'] - YI) / (0.5 * M['streak_m'])), 1.0)
        # storey string course, parapet cap, plinth
        if S['band'] and nst > 1:
            dl = np.minimum(YI, hs - YI)
            band = (dl < 0.5 * M['band_h_m']) & (Y > 0.5 * hs) & (Y < h_m - M['parapet_m']) & ~inside & ~sl
            A = np.where(band, A * M['band_tone'], A); H = np.where(band, M['band_proj_m'], H)
        if shop:
            fas = g & (YI > Y1) & (YI < Y1 + M['fascia_m']) & ~door
            A = np.where(fas, A * 0.82, A); H = np.where(fas, 0.06, H)
        par = Y > h_m - M['parapet_m']
        A = np.where(par, A * 1.05, A); H = np.where(par, H + M['parapet_proj_m'], H)
        pl = (Y < M['plinth_m']) & ~inside
        A = np.where(pl, A * 0.92, A); H = np.where(pl, H + M['plinth_proj_m'], H)
        A = np.where(sl, A * 1.12, A); H = np.where(sl, M['sill_proj_m'], H)
        R = np.where(sl, M['rough_sill'], R)
        H = np.where(frame, -M['frame_depth_m'], H)
        H = np.where(glass, -M['recess_m'], H)
        H = np.where(mull, -M['recess_m'] + 0.035, H)
        R = np.where(frame | mull, M['rough_frame'], R); R = np.where(glass, M['rough_glass'], R)
        # the panes: a sky-to-ground reflection, blinds, a dark room
        g0, g1 = M['sky_gain']
        refl = glass_rgb[None, None, :] * ((g0 + (g1 - g0) * v ** 1.4) * pv[pid] * big)[:, :, None]
        isb = (ptype[pid] < S['blind_p']) & (v > 1.0 - pblind[pid])
        blind = np.asarray(M['blind_rgb'], 'f4')[None, None, :] * (0.8 + 0.3 * pv[pid])[:, :, None]
        slat = 1.0 - 0.10 * (np.sin(2 * np.pi * YI / 0.05) > 0.6)
        pane = np.where(isb[:, :, None], blind * slat[:, :, None], refl)
        dark = (ptype[pid] > 1.0 - M['dark_p'])[:, :, None]
        pane = np.where(dark & ~isb[:, :, None], refl * 0.55, pane)
        absrgb = np.where(glass[:, :, None], pane, frame_rgb[None, None, :]).astype('f4')
        absa = inside.astype('f4')
        wall = ~inside
        info['shopfront'] = bool(shop)
    # material relief and dirt on the wall
    H = H + np.where(wall, tile_relief(img, tex), 0.0)
    dm, dr = dirt_field(yy, h_m, nx, S, prng, tex)
    R = np.clip(R + np.where(wall, dr, 0.0), 0.02, 1.0)
    out = img * A[:, :, None] * dm
    # mean 1 over the WALL texels for a punched facade (the panes carry their own
    # absolute colour); over the WHOLE tile for a curtain wall, whose measured
    # colour is its glass.  Normalising a curtain wall on its spandrels only puts
    # the glass at ~0.65x the measured colour, and stage 60's level gate then
    # rejects the wall's photograph (gain under the floor).
    wm = (np.ones((ny, nx), 'f4') if cls == 'glass' else wall.astype('f4'))
    lum = (out @ T.LUMW)
    out = out / max(float((lum * wm).sum() / max(wm.sum(), 1.0)), 1e-4)
    if openings_out is not None:
        openings_out['mask'] = inside & ~(absa > 0.5) if cls != 'glass' else inside.copy()
    if pbr_out is not None:
        pbr_out.update(height=H.astype('f4'), rough=R.astype('f4'),
                       abs_rgb=absrgb.astype('f4'), abs_a=absa.astype('f4'))
    return np.clip(out, 0.02, 6.0).astype('f4'), info


def blind_pbr(w, cls, img, out, xx, yy, pitch, rng, pbr_out, dirt=True):
    """Height / roughness of a wall with no window grid (blind, hull, shelter):
    material relief, panel joints recessed, plinth and parapet; dirt on buildings.
    Returns the (possibly dirtier) field, renormalised to mean luminance 1."""
    M = C.FACADE_MOD; S = building_style(w, cls)
    ny, nx = out.shape[:2]; h_m = w['height_m']; tex = w['width_m'] / max(nx, 1)
    H = tile_relief(img, tex) if img is not None else np.zeros((ny, nx), 'f4')
    if pitch:
        px = xx / pitch; py = yy / pitch
        wj = max(0.75 * tex, 0.006) / pitch
        line = np.maximum(np.exp(-0.5 * (np.abs(px - np.round(px)) / wj) ** 2)[None, :],
                          np.exp(-0.5 * (np.abs(py - np.round(py)) / wj) ** 2)[:, None])
        H = H - 0.012 * line
    Y = yy[:, None] * np.ones((1, nx))
    H = H + np.where(Y < M['plinth_m'], M['plinth_proj_m'], 0.0) \
        + np.where(Y > h_m - M['parapet_m'], M['parapet_proj_m'], 0.0)
    R = np.full((ny, nx), M['rough'].get(cls, 0.85), 'f4')
    if dirt:
        dm, dr = dirt_field(yy, h_m, nx, S, np.random.default_rng(S['seed'] + 3), tex)
        out = out * dm; R = R + dr
        out = out / max(float((out @ T.LUMW).mean()), 1e-4)
    if pbr_out is not None:
        pbr_out.update(height=H.astype('f4'), rough=np.clip(R, 0.02, 1).astype('f4'),
                       abs_rgb=None, abs_a=None)
    return out


def apply_modules(img, pbr):
    """Composite the absolute colour of frames, panes and doors over the tinted
    wall (after the tint multiply and the opening floor)."""
    if not pbr or pbr.get('abs_a') is None or pbr['abs_a'].shape != img.shape[:2]:
        return img
    a = pbr['abs_a'][:, :, None]
    return (img * (1 - a) + pbr['abs_rgb'] * a).astype('f4')


def pbr_pages(pbr, tex_m, ny, nx, photo_alpha=None):
    """(normal RGB u1, roughness/metal RGB u1), row 0 = wall FOOT like the
    synthesis arrays (the caller flips).  glTF: normal +X = +u, +Y = up the
    image; MR: G = roughness, B = metallic (x metallicFactor 0)."""
    if pbr is None or pbr.get('height') is None or pbr['height'].shape != (ny, nx):
        Hh = np.zeros((ny, nx), 'f4'); Rr = np.full((ny, nx), 0.85, 'f4')
    else:
        Hh = pbr['height']; Rr = pbr['rough']
    if photo_alpha is not None:
        Hh = Hh * (1 - photo_alpha); Rr = Rr * (1 - photo_alpha) + 0.85 * photo_alpha
    from scipy.ndimage import uniform_filter
    Hs = uniform_filter(Hh.astype('f4'), 2, mode='nearest')
    gy, gx = np.gradient(Hs, max(tex_m, 1e-3))      # axis 0 runs UP the wall here
    s = C.FACADE_NORMAL_SCALE
    n = np.stack([-gx * s, -gy * s, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = np.clip((n * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype('u1')
    mr = np.empty((ny, nx, 3), 'u1'); mr[..., 0] = 255
    mr[..., 1] = np.clip(Rr * 255 + 0.5, 0, 255).astype('u1'); mr[..., 2] = 255
    return nrm, mr


def pack_aux(tiles, place, npages, fill):
    """Pack aux tiles exactly as the colour pages are packed (same gutters)."""
    pages = [np.empty((ATLAS_PX, ATLAS_PX, 3), 'u1') for _ in range(npages)]
    for p in pages:
        p[...] = np.asarray(fill, 'u1')
    for t, (pg, x, y) in zip(tiles, place):
        h, wd = t.shape[:2]
        P = pages[pg]
        P[y:y + h, x:x + wd] = t
        P[y - PAD:y, x:x + wd] = t[0][None, :, :]
        P[y + h:y + h + PAD, x:x + wd] = t[-1][None, :, :]
        P[y - PAD:y + h + PAD, x - PAD:x] = P[y - PAD:y + h + PAD, x:x + 1]
        P[y - PAD:y + h + PAD, x + wd:x + wd + PAD] = P[y - PAD:y + h + PAD, x + wd - 1:x + wd]
    return pages


NRM_FLAT = (128, 128, 255)
MR_FLAT = (255, 217, 255)       # roughness 0.85, the material-wide default


def synth_facade(w, cls, nx, ny, base, tile_m, rows, seed, near_core,
                 no_windows=False, storey_src=None, pitch=None,
                 openings_out=None, bay_m=None, pbr_out=None):
    """A wall of `cls` with a plausible window and door grid, in linear RGB.

    `storey_src` is the evidence for the storey height: 'building:levels' when
    the OSM way carries a level count, 'photograph' when a camera saw another
    wall of the same building part and the storey height was measured off it.
    With neither, no window grid is drawn at all -- an invented grid on a wall
    nobody measured is a guess the model shows as fact -- and the wall gets the
    panel joints of `panel_field` instead.

    Pass a dict as `openings_out` and it comes back with 'mask': the (ny, nx)
    boolean of the window and door texels, for apply_opening_floor().  The
    return value is unchanged either way -- the field carries structure only,
    and a window's colour is absolute, so the floor cannot live in it."""
    h_m = w['height_m']; w_m = w['width_m']
    if openings_out is not None:
        openings_out['mask'] = np.zeros((ny, nx), bool)
    rng = np.random.default_rng(seed)
    bld = w['osm'].get('building')
    if base is not None and cls == 'wood' and bld != 'ship' \
            and C.FACADE_WOOD_BOARDS == 'vertical':
        # the wood tile is a horizontal plank; a clad building wall is laid
        # with it turned 90 degrees (vertical board siding, area config)
        img = tiled(base, tile_m, h_m, w_m, ny, nx).transpose(1, 0, 2).copy()
        img /= max(float(img.reshape(-1, 3).mean(0) @ T.LUMW), 1e-4)
    elif base is not None:
        img = tiled(base, tile_m, w_m, h_m, nx, ny).copy()
        img /= max(float(img.reshape(-1, 3).mean(0) @ T.LUMW), 1e-4)
    else:                                            # glass: no CC0 tile
        img = np.ones((ny, nx, 3), 'f4')
    lv = levels_of(w)
    # OSM building:levels is a mapper's count and beats anything read off a
    # photograph; the measured storey height is for the buildings that have no
    # level count at all.  With neither, nothing below is drawn: the wall takes
    # the blind branch and this height is never used.
    if storey_src == 'building:levels':
        hs = h_m / lv
    elif rows:
        hs = rows['storey_m']
    else:
        hs = h_m / max(lv, 1)
    hs = float(np.clip(hs, 2.7, 5.2))
    nst = max(1, int(round(h_m / hs)))
    hs = h_m / nst
    phase = rows['phase'] if rows else 0.42
    target_bay = bay_m or (5.0 if (w['osm'].get('building') in ('civic', 'industrial', 'warehouse')
                                   or cls == 'wood') else 3.4)
    nbay = max(1, int(round(w_m / target_bay)))
    bw = w_m / nbay
    yy = (np.arange(ny) + 0.5) / ny * h_m          # metres from the wall foot
    xx = (np.arange(nx) + 0.5) / nx * w_m
    stor = np.floor(yy / hs).astype(int)
    yin = yy - stor * hs
    bay = np.floor(xx / bw).astype(int)
    xin = xx - bay * bw
    shop = near_core and (w['osm'].get('building') in
                            ('retail', 'commercial', 'civic', 'yes', None)
                            or w['osm'].get('amenity') or w['osm'].get('tourism'))
    # --- opening rectangles -------------------------------------------------
    if bld in ('ship', 'roof') or no_windows or storey_src is None:
        # a hull, an open shelter, or a wall with no storey evidence at all.
        # A ship gets a dark boot-top, a sheer line and a row of deckhouse
        # lights; a shelter gets a fascia band; a blind wall gets its plinth,
        # its parapet and the panel joints of its class.
        out = img
        if no_windows and bld not in ('ship', 'roof'):
            # the photograph carries this wall's real openings; a second,
            # invented grid painted into the gaps reads as stickers
            out = out * (1.0 - 0.08 * (yy < 0.55)[:, None])[:, :, None]
            out = out * (1.0 + 0.03 * rng.standard_normal((1, nx, 1)))
            out = out / max(float(out.reshape(-1, 3).mean(0) @ T.LUMW), 1e-4)
            if getattr(C, 'FACADE_PBR', False) and pbr_out is not None:
                out = blind_pbr(w, cls, img, out, xx, yy, None, rng, pbr_out)
            return np.clip(out, 0.02, 6.0).astype('f4'), dict(
                levels=lv, storeys=0, storey_m=0.0, bays=0, bay_m=0.0,
                shopfront=False, panel_pitch_m=0.0,
                window_rows_source='none: the photograph carries the openings')
        if bld == 'ship':
            boot = (yy < 0.30 * h_m)[:, None]
            sheer = (np.abs(yy - 0.52 * h_m) < 0.09 * h_m)[:, None]
            out = out * (1.0 - 0.42 * boot)[:, :, None]
            out = out * (1.0 + 0.16 * sheer)[:, :, None]
            port = ((np.abs(yy - 0.76 * h_m) < 0.055 * h_m)[:, None] &
                    (np.abs((xx % 2.4) - 1.2) < 0.26)[None, :])
            out = np.where(port[:, :, None], np.array([0.10, 0.11, 0.12], 'f4'), out)
            why = 'no window grid: OSM building=ship'
        else:
            if bld == 'roof':
                out = out * (1.0 - 0.22 * (yy > h_m - 0.45)[:, None])[:, :, None]
                why = 'no window grid: OSM building=roof'
            else:
                out = out * (1.0 - 0.10 * (yy < 0.55)[:, None])[:, :, None]
                out = out * (1.0 + 0.07 * (yy > h_m - 0.55)[:, None])[:, :, None]
                why = ('no window grid: no building:levels on the way and no '
                       'storey height measured on a photographed wall of this '
                       'building')
            if pitch:
                out = out * panel_field(xx, yy, pitch, rng)
        if bld in ('ship', 'roof'):
            out = out * (1.0 + 0.04 * rng.standard_normal((1, nx, 1)))
        else:
            # a concrete wall does streak vertically, but at the scale of a
            # weathering run, not of a texel: a per-column random number is
            # corduroy: 3.2 sRGB levels of noise finer than 0.3 m, measured on a
            # render with the joints switched off.  Generated at 0.3 m and
            # interpolated.
            k = max(2.0, 0.30 / max(w_m / nx, 1e-3))
            n0 = rng.standard_normal(int(nx / k) + 2)
            col = np.interp(np.arange(nx) / k, np.arange(len(n0)), n0)
            out = out * (1.0 + 0.018 * col)[None, :, None]
        out = out / max(float(out.reshape(-1, 3).mean(0) @ T.LUMW), 1e-4)
        if getattr(C, 'FACADE_PBR', False) and pbr_out is not None:
            out = blind_pbr(w, cls, img, out, xx, yy,
                            pitch if bld != 'ship' else None, rng, pbr_out,
                            dirt=bld not in ('ship', 'roof'))
        return np.clip(out, 0.02, 6.0).astype('f4'), dict(
            levels=lv, storeys=0, storey_m=0.0, bays=0, bay_m=0.0,
            shopfront=False,
            panel_pitch_m=(round(float(pitch), 2) if (pitch and bld != 'ship')
                           else 0.0),
            window_rows_source=why)
    if getattr(C, 'FACADE_PBR', False) and pbr_out is not None:
        field, minfo = module_facade(w, cls, img, xx, yy, hs, nst, bw, nbay, phase,
                                     frontage(w), rng, pbr_out, openings_out)
        return field, dict(levels=lv, storeys=nst, storey_m=round(hs, 2), bays=nbay,
                           bay_m=round(bw, 2), shopfront=minfo.pop('shopfront'),
                           panel_pitch_m=0.0, window_rows_source=storey_src, **minfo)
    if cls == 'glass':
        # curtain wall: a spandrel band per storey, glazing between, mullions
        win_y = (yin > 0.16 * hs) & (yin < 0.86 * hs)
        win_x = np.ones(nx, bool)
        mull = (np.abs(xin - bw / 2) < 0.055) | (xin < 0.055) | (xin > bw - 0.055)
    else:
        wh = np.clip(0.52 * hs, 1.1, 2.4)
        sill = np.clip(phase * hs - wh / 2, 0.55, hs - wh - 0.35)
        ww = np.clip(0.50 * bw, 0.9, 2.6)
        win_y = (yin > sill) & (yin < sill + wh)
        win_x = np.abs(xin - bw / 2) < ww / 2
        mull = np.zeros(nx, bool)
    win = win_y[:, None] & win_x[None, :]
    if cls != 'glass' and shop:
        gh = np.clip(0.66 * hs, 1.8, 3.4)
        gwin = (stor == 0)[:, None] & (yy > 0.45)[:, None] & (yy < 0.45 + gh)[:, None] \
               & (np.abs(xin - bw / 2) < 0.38 * bw)[None, :]
        win = win | gwin
        # one door, in the bay nearest the middle of the wall
        db = nbay // 2
        door = ((bay == db)[None, :] & (np.abs(xin - bw / 2) < 0.55)[None, :]
                & (yy < 2.25)[:, None] & (yy > 0.02)[:, None])
    else:
        door = np.zeros((ny, nx), bool)
    plinth = (yy < 0.55)[:, None] & np.ones((1, nx), bool)
    parapet = (yy > h_m - 0.55)[:, None] & np.ones((1, nx), bool)
    # --- shading ------------------------------------------------------------
    out = img
    # per bay and per storey brightness jitter so a repeat does not read as a stamp
    jb = 1.0 + 0.045 * rng.standard_normal(nbay + 1)[np.clip(bay, 0, nbay)]
    js = 1.0 + 0.030 * rng.standard_normal(nst + 1)[np.clip(stor, 0, nst)]
    out = out * jb[None, :, None] * js[:, None, None]
    out = out * (1.0 - 0.10 * plinth[:, :, None])
    out = out * (1.0 + 0.07 * parapet[:, :, None])
    # glass: dark, blue, one brightness per window pane
    pane_id = (np.clip(stor, 0, nst)[:, None] * (nbay + 1)
               + np.clip(bay, 0, nbay)[None, :])
    pane_v = (0.65 + 0.55 * rng.random(int(pane_id.max()) + 2))[pane_id]
    glass_rgb = np.array([0.36, 0.46, 0.56], 'f4')
    g = glass_rgb[None, None, :] * pane_v[:, :, None]
    if cls == 'glass':
        g = g * 1.25
    # panes brighten toward the top, where a window reflects the sky
    sky = 1.0 + 0.55 * np.clip((yin / max(hs, 1e-3)), 0, 1)[:, None, None]
    out = np.where(win[:, :, None] | door[:, :, None], g * sky, out)
    # window reveal: a one-texel dark line round each opening
    edge = np.zeros((ny, nx), bool)
    e = win | door
    if openings_out is not None:
        openings_out['mask'] = e.copy()
    edge[1:, :] |= e[1:, :] & ~e[:-1, :]
    edge[:-1, :] |= e[:-1, :] & ~e[1:, :]
    edge[:, 1:] |= e[:, 1:] & ~e[:, :-1]
    edge[:, :-1] |= e[:, :-1] & ~e[:, 1:]
    out = np.where(edge[:, :, None], out * 0.45, out)
    # and a light frame just outside that reveal, which is what actually makes
    # an opening read as a window rather than a painted rectangle
    frame = np.zeros((ny, nx), bool)
    frame[2:, :] |= e[:-2, :] & ~e[2:, :]
    frame[:-2, :] |= e[2:, :] & ~e[:-2, :]
    frame[:, 2:] |= e[:, :-2] & ~e[:, 2:]
    frame[:, :-2] |= e[:, 2:] & ~e[:, :-2]
    frame &= ~e
    out = np.where(frame[:, :, None], out * 1.35, out)
    if mull.any():
        out = np.where((mull[None, :] & win)[:, :, None], np.array([0.5, 0.52, 0.55], 'f4'), out)
    # the tint is what sets the colour, so hand back a modulation field: mean
    # luminance exactly 1, carrying only the structure
    out = out / max(float(out.reshape(-1, 3).mean(0) @ T.LUMW), 1e-4)
    return np.clip(out, 0.02, 6.0).astype('f4'), dict(
        levels=lv, storeys=nst, storey_m=round(hs, 2), bays=nbay,
        bay_m=round(bw, 2), shopfront=bool(shop and cls != 'glass'),
        panel_pitch_m=0.0, window_rows_source=storey_src)


def apply_opening_floor(img, openings, base=GLASS_BASE_RGB,
                        ambient=GLASS_AMBIENT_RGB):
    """Give the window and door texels of a baked wall an absolute glass colour.

    `img` is linear RGB AFTER the tint multiply, `openings` the mask
    synth_facade() handed back.  The opening's own structure -- pane-to-pane
    brightness, the sky gradient, the reveal, the frame -- is carried as a
    relative luminance field of mean 1, and it shapes the floor, so nothing
    about the way an opening is drawn is flattened; only its absolute level
    moves, and only where the wall's own product falls below the floor.  `ambient` is then a hard minimum per channel: the light the
    room behind the glass sends back, which is what stops a pane going black.
    Returns a new array; the wall outside the openings is untouched."""
    m = np.asarray(openings, bool) if openings is not None else None
    if m is None or m.ndim != 2 or m.shape != img.shape[:2] or not m.any():
        return img
    out = np.array(img, 'f4')
    o = out[m]
    lum = o @ T.LUMW
    s = lum / max(float(lum.mean()), 1e-6)
    o = np.maximum(o, np.asarray(base, 'f4')[None, :] * s[:, None])
    out[m] = np.maximum(o, np.asarray(ambient, 'f4')[None, :])
    return out


# ---------------------------------------------------------------- colour -- #
def wall_ao(ny, nx, h_m, w_m):
    """Ambient occlusion on a synthetic wall: multiplicative, mean exactly 1.

    Row 0 here is the foot of the wall (the tile is flipped on its way into the
    atlas), so the darkening sits at low y.  Mean 1 means this changes the
    gradient of the wall and not the tone match applied with it."""
    y = (np.arange(ny) + 0.5) / max(ny, 1) * max(h_m, 1e-3)
    x = (np.arange(nx) + 0.5) / max(nx, 1) * max(w_m, 1e-3)
    f = 1.0 - AO_FOOT * np.exp(-y / AO_FOOT_M)
    e = 1.0 - AO_EDGE * (np.exp(-x / AO_EDGE_M) +
                         np.exp(-np.maximum(w_m - x, 0.0) / AO_EDGE_M))
    a = f[:, None] * e[None, :]
    return (a / max(float(a.mean()), 1e-6)).astype('f4')


def macro_for_wall(cls, ny, nx, texel_m, seed=35, band_m=None, tone_seed=0):
    """stage 22's metre-scale field for this class, cropped to the wall, times
    the panel tone stage 22 does not carry.

    macro_detail() is periodic over a square tile, and wall texels are square,
    so the field is generated on a square of the larger side at the wall's own
    texel size and cropped: the joints then land at their stated pitch in
    metres on both axes.  None for a class whose tile has structure already.

    macro_detail() draws three measured things -- the joint grid at
    MACRO[mat]['panel_m'], the weathering blotch and the rain streak -- but
    every cell between its joints carries the same tone, so a blind wall reads
    as one plane with lines drawn on it (a blind metal-panel wall measures sRGB
    sd 3.21 over the wall, 1.93 in a 25 cm window).  The
    second factor below is built on macro_detail's OWN cell geometry -- the
    same square of side n, the same texel size, the same
    nu = round(tile_m / panel_m) -- so its tone steps land INSIDE the cells the
    joints bound instead of beside them, and the wall still has exactly one
    joint grid.  It is normalised to mean 1 by itself, so the tile's mean
    linear luminance is whatever stage 22's field already made it and the tone
    match is untouched.

    `band_m` is the storey height and draws a darker band at each storey line.
    It is passed only for a wall whose OSM way carries building:levels and
    whose window grid is not drawn; with no level count no band is drawn,
    because a storey height read off nothing is an invention.

    `seed` is 35 for every wall, so stage 22's own field is reproduced bit
    for bit; `tone_seed` is the wall id, so two walls of one building
    do not get the same panel laid out the same way."""
    mat = MACRO_CLASS.get(cls)
    if mat is None or min(ny, nx) < 4 or texel_m <= 0:
        return None
    try:
        from s22_composite import macro_detail, MACRO
    except Exception:
        return None
    n = int(max(ny, nx))
    tile_m = n * float(texel_m)
    d = macro_detail(mat, n, n, tile_m, seed=seed)
    if d is None:
        return None
    d = d[:ny, :nx]
    look = PANEL_LOOK.get(cls)
    if look is None:
        return d
    # macro_detail's own cell: nu whole panels across the square it built
    nu = max(1, int(round(tile_m / MACRO[mat]['panel_m'])))
    px = (np.arange(nx) + 0.5) * nu / n              # cell units, same origin
    py = (np.arange(ny) + 0.5) * nu / n
    ix = np.floor(px).astype(int); iy = np.floor(py).astype(int)
    rng = np.random.default_rng(int(tone_seed) * 7919 + int(seed))
    jit = rng.standard_normal((int(iy.max()) + 2, int(ix.max()) + 2))
    s = 0.5 * look['tone'] * np.clip(jit, -2, 2)[iy][:, ix]
    s = s + 0.5 * look['bow'] * (
        np.cos(2 * np.pi * (px - np.floor(px)))[None, :]
        + np.cos(2 * np.pi * (py - np.floor(py)))[:, None])
    if band_m and band_m > 0.6:
        yy = (np.arange(ny) + 0.5) * float(texel_m)  # metres from the wall foot
        pb = yy / float(band_m)
        wb = max(2.5 * float(texel_m), 0.09)
        s = s + look['band'] * np.exp(
            -0.5 * (np.abs(pb - np.round(pb)) * float(band_m) / wb) ** 2)[:, None]
    g = np.power(np.clip(1.0 - s, 0.4, 1.6), 2.2)
    g = g / max(float(g.mean()), 1e-6)
    return (d * g).astype('f4')


def wall_gap(cen, r):
    """Metres from a point to the nearest point of a photographed wall.

    A wall is a segment, not a point.  On a 45 m wide wall a neighbour can be
    13.8 m from the CENTRE and only 9.2 m from the PLANE, and the plane is what
    the two surfaces share."""
    dv = np.asarray(cen, 'f8') - r['centre']
    t = float(np.clip(dv @ r['u'], -r['half'], r['half']))
    return float(np.hypot(*(dv - t * r['u'])))


def ring_area(P):
    """Shoelace area of a closed ring, m2."""
    x = P[:, 0]; y = P[:, 1]
    return 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))


def point_in_ring(q, P):
    """Even-odd crossing count of a point against a closed ring."""
    cr = 0
    n = len(P)
    for i in range(n):
        p0 = P[i]; p1 = P[(i + 1) % n]
        if (p0[1] > q[1]) != (p1[1] > q[1]):
            xx = p0[0] + (q[1] - p0[1]) * (p1[0] - p0[0]) / (p1[1] - p0[1] + 1e-12)
            if xx > q[0]:
                cr += 1
    return cr % 2 == 1


def part_footprints(walls):
    """The OSM way footprint of every wall part, in photo-ENU metres.

    Two parts of one OSM way (building_<id> and building_<id>_2) share its footprint;
    that is what the mass rule wants, because either part of the mass sits on
    the same host."""
    want = set()
    for w in walls:
        if w.get('osm_id'):
            want.add(int(w['osm_id']))
    try:
        _, ways = C.osm_geoms()
    except Exception as e:                                   # pragma: no cover
        print('mass rule off: no OSM way geometry (%s)' % e, flush=True)
        return {}
    xy = {}
    for way in ways:
        if int(way['id']) not in want:
            continue
        g = C.way_xy(way)
        if g is None or len(g) < 4:
            continue
        g = np.asarray(g, 'f8')
        if np.allclose(g[0], g[-1]):
            g = g[:-1]
        if len(g) >= 3:
            xy[int(way['id'])] = g
    out = {}
    for w in walls:
        g = xy.get(int(w.get('osm_id') or -1))
        if g is not None:
            out[w['part']] = g
    return out


# the pack is keyed by wall id and s33 ids are not stable (adding a
# building renumbers every later wall), so a pack record is only this wall's
# tile if its part_name and area agree with the current wall.
PACK_META = {}
PACK_NRM = {}


def pack_matches(w, pm):
    """True when pack record pm (PACK_META entry for w's id) is this wall."""
    if pm is None or pm[0] != w['part_name'] or abs(pm[1] - float(w['area_m2'])) > 0.5:
        return False
    if pm[2] and abs(pm[2] - float(w['width_m'])) > 0.10:
        return False
    if pm[3] and abs(pm[3] - float(w['height_m'])) > 0.10:
        return False
    pn = PACK_NRM.get(int(w['id']))
    if pn is not None:
        wn = np.asarray(w['normal'], 'f8'); wn = wn / max(float(np.linalg.norm(wn)), 1e-9)
        if float(pn @ wn) < 0.98:
            return False
    return True


def packed_tile_reference(path=PACK_REF, stat=TONE_TILE_STAT):
    """{wall id: linear RGB of the tile the PREVIOUS prep actually packed}.

    Read from the wallphoto walls pack, which is the set s62 swaps over
    facades/ and therefore the set that ships.  A record with no tile statistic
    (a wall the pack dropped) is absent, and that wall keeps its trusted-texel
    median, so a missing or unreadable pack
    degrades to the texel-median reference.  Returns ({}, why) when there is nothing to read."""
    if not os.path.exists(path):
        return {}, 'no pack at ' + path
    try:
        d = json.load(open(path))
    except Exception as e:
        return {}, 'pack unreadable (%s)' % e
    out = {}
    dark = []
    for r in d.get('per_wall') or []:
        v = r.get(stat)
        if not v or len(v) != 3:
            continue
        c = np.asarray(v, 'f4')
        if not np.isfinite(c).all():
            continue
        # a packed tile at or under the floor is a failed bake, not an
        # exposure, and this value is about to earn a relaxed darken cap
        if float(c @ T.LUMW) <= TONE_TILE_MIN_LUM:
            dark.append(int(r['id']))
            continue
        out[int(r['id'])] = c
        PACK_META[int(r['id'])] = (r.get('part_name'), float(r.get('area_m2') or 0.0),
                                   float(r.get('width_m') or 0.0),
                                   float(r.get('height_m') or 0.0))
    # part_name + area alone confuse equal-size siblings
    # (e.g. two opposite hull sides of equal area).  The pack carries no
    # wall centre, but its wallmap.npz carries every face's normal by wall id,
    # so the pack record must also face the same way as the current wall.
    PACK_NRM.clear()
    try:
        z = np.load(os.path.join(os.path.dirname(path), 'wallmap.npz'))
        wid = z['wall_id']; fn = z['fnrm'].astype('f8')
        for i in np.unique(wid):
            m = fn[wid == i].sum(0)
            L = float(np.linalg.norm(m))
            if L > 1e-6:
                PACK_NRM[int(i)] = m / L
    except Exception:
        pass
    why = ('%d walls carry %s in %s' % (len(out), stat, path))
    if dark:
        why += '; %d rejected under the %.3f floor' % (len(dark), TONE_TILE_MIN_LUM)
    return out, why


def mass_reference(w, refs, cls, foot):
    """The photographed building this synthetic mass stands on or inside.

    A wall whose whole OSM footprint lies inside a photographed building's
    footprint is not next to that building, it is part of it, so it takes that
    building's exposure across ALL its photographed faces -- not just the ones
    facing the same way, which is the difference from near_facing_reference.
    The smallest qualifying host wins, so a mass inside a mass inside a block
    takes the one it actually sits on.

    The host's photographed walls have to be of the SAME CLASS.  Without that
    rule, concrete walls can take a host whose only photographed walls are
    glass (linear 0.04-0.08) and be darkened about 5x onto the 0.30 cap.  A
    building's glass is not the exposure of its concrete."""
    mine = foot.get(w['part'])
    if mine is None or len(mine) < 3:
        return None, None, 0.0
    hosts = {}
    for r in refs:
        if r['part'] == w['part'] or (cls is not None and r['cls'] != cls):
            continue
        hosts.setdefault(r['part'], []).append(r)
    best = None
    for pt, rr in hosts.items():
        P = foot.get(pt)
        if P is None or len(P) < 3:
            continue
        if float(np.mean([point_in_ring(q, P) for q in mine])) < TONE_MASS_INSIDE:
            continue
        a = ring_area(P)
        if best is None or a < best[0]:
            best = (a, pt, rr)
    if best is None:
        return None, None, 0.0
    _, pt, rr = best
    c = np.median(np.stack([r['c'] for r in rr]), axis=0).astype('f4')
    # same guard near_facing_reference carries: a reference the floor has to
    # rescue is a black photograph, not an exposure, and this branch is about
    # to hand it a relaxed darken cap
    if float(np.asarray(c, 'f8') @ T.LUMW) <= TONE_LUM_FLOOR:
        return None, None, 0.0
    # this branch takes an unweighted median, so the share that came
    # from a measured packed tile is counted per wall, not per m2
    frac = float(np.mean([1.0 if r.get('c_from_tile') else 0.0 for r in rr]))
    return c, ("the %d photographed %s wall(s) of %s, the building whose "
               "footprint this mass stands inside"
               % (len(rr), cls, rr[0]['part_name'])), frac


def class_prior_reference(w, cls, refs):
    """The exposure the same material carries nearby, for a wall with nothing.

    Only reached when no photograph at all is within TONE_MATCH_R, so there is
    no better evidence to lose: the alternative is a class default with no
    exposure behind it."""
    cen = np.asarray(w['centre'][:2], 'f8')
    cand = []
    for r in refs:
        if r['id'] == w['id'] or r['cls'] != cls:
            continue
        if float(np.asarray(r['c'], 'f8') @ T.LUMW) <= TONE_LUM_FLOOR:
            continue
        d = wall_gap(cen, r)
        if d <= TONE_CLASS_R:
            cand.append((d, r))
    if not cand:
        return None, None, 0.0
    cand.sort(key=lambda t: t[0])
    cand = cand[:TONE_CLASS_N]
    c = np.median(np.stack([r['c'] for _, r in cand]), axis=0).astype('f4')
    frac = float(np.mean([1.0 if r.get('c_from_tile') else 0.0 for _, r in cand]))
    return c, ("the %d nearest photographed %s wall(s) within %.0f m (%s)"
               % (len(cand), cls, TONE_CLASS_R,
                  ', '.join('wall %d at %.0f m' % (r['id'], d) for d, r in cand))), frac


def near_facing_reference(w, refs):
    """The one photographed wall a synthetic mass stands on, if there is one.

    Same outward direction to within TONE_NEAR_FACING and within TONE_NEAR_R of
    this wall, measured plane to centre.  Several qualifying walls are blended
    by facing agreement x area / (2 + d)^2, so the big head-on face beside the
    wall wins over a small one at the edge of the reach; the set has to carry
    TONE_NEAR_MIN_AREA of photograph before it is allowed to speak at all."""
    n = np.asarray(w['normal'][:2], 'f8')
    nn = float(np.hypot(*n))
    if nn < 1e-9:
        return None, None, 0.0
    n = n / nn
    cen = np.asarray(w['centre'][:2], 'f8')
    cand = []
    for r in refs:
        if r.get('u') is None or r['id'] == w['id']:
            continue
        ag = float(n @ r['normal'])
        if ag < TONE_NEAR_FACING:
            continue
        d = wall_gap(cen, r)
        if d <= TONE_NEAR_R:
            cand.append((d, ag, r))
    if not cand or sum(r['area'] for _, _, r in cand) < TONE_NEAR_MIN_AREA:
        return None, None, 0.0
    aw = np.array([ag * r['area'] / (2.0 + d) ** 2 for d, ag, r in cand], 'f8')
    c = (np.stack([r['c'] for _, _, r in cand]) * aw[:, None]).sum(0) / max(aw.sum(), 1e-9)
    # a reference TONE_LUM_FLOOR has to rescue is not an exposure, it is a black
    # photograph, and this branch is about to hand it a relaxed darken cap.
    # Example: a tone reference on 21 % trusted texels with median linear
    # 0.0013 (sRGB 3), 2 m from a same-facing wall, would pull that wall from
    # 0.1329 to 0.0886.
    if float(np.asarray(c, 'f8') @ T.LUMW) <= TONE_LUM_FLOOR:
        return None, None, 0.0
    best = cand[int(np.argmax(aw))]
    # the share of THIS blend's weight that came from a measured
    # packed tile mean rather than a trusted-texel median
    tw = np.array([1.0 if r.get('c_from_tile') else 0.0 for _, _, r in cand], 'f8')
    frac = float((aw * tw).sum() / max(aw.sum(), 1e-9))
    return c.astype('f4'), ("photographed wall %d, %.0f m away and facing the "
                            "same way (%d wall(s) within %.0f m at cos >= %.2f)"
                            % (best[2]['id'], best[0], len(cand), TONE_NEAR_R,
                               TONE_NEAR_FACING)), frac


# per-branch darken cap and the name the summary counts it under
# a building with several photographed faces has MEASURED its own
# light from more than one direction, so the 0.45 floor under `same` is not
# the weak evidence it is for a lone reference.  Example: a synthetic
# west wall on branch `same` is held by the 0.45 cap at sRGB 179 while the
# photographed walls of its building render at 104-130, a bright outlier.
# With >= TONE_SAME_MIN_REFS photographed faces the cap drops to the `near`
# value, which is the same argument the near-facing branch already makes.
TONE_SAME_MIN_REFS = 3

TONE_BRANCH = {
    'same':  (TONE_MAX_DARKEN,
              'the photographed walls of this building'),
    'same3': (TONE_MAX_DARKEN_NEAR,
              'the photographed walls of this building (%d or more of them)'
              % TONE_SAME_MIN_REFS),
    'near':  (TONE_MAX_DARKEN_NEAR,
              'a photographed wall within %.0f m facing the same way' % TONE_NEAR_R),
    'mass':  (TONE_MAX_DARKEN_NEAR,
              'the photographed building this mass stands inside'),
    'area':  (TONE_MAX_DARKEN,
              'the photographed walls within %.0f m' % TONE_MATCH_R),
    'class': (TONE_MAX_DARKEN,
              'the nearest photographed walls of the same class'),
}


def tone_reference(w, refs, cls=None, foot=None):
    """Linear colour of the photographed walls that speak for this one.

    In order of how close the evidence is to being this wall's own light:

      same   its own building -- two walls of one building are one paint job;
      near   the one photographed wall it stands on, same facing, within
             TONE_NEAR_R;
      mass   the photographed building whose footprint it sits inside, across
             all that building's faces of this wall's material;
      area   the photographed walls within TONE_MATCH_R, area over distance
             squared;
      class  nothing photographed is in reach at all: the nearest photographed
             walls of the same material.

    `near` is tried before `mass` on purpose: one wall 9 m away facing the way
    this one faces is the same surface in the same sun, while the host mass
    average takes in faces in other light.  Where `near` fires it keeps every
    value it produced; `mass` only fills in behind it.

    Each reference wall's r['c'] is its FINAL PACKED TILE MEAN when the
    previous prep's pack recorded one, and its trusted-texel median otherwise
    (see packed_tile_reference()).
    The fourth return value is the share of THIS branch's own weight that came
    from a measured packed tile, which is what licenses TONE_MAX_DARKEN_TILE.

    Returns (colour, why, branch, tile_frac); (None, None, None, 0.0) when
    nothing speaks for the wall and it keeps the colour it had."""
    if not refs:
        return None, None, None, 0.0
    same = [r for r in refs if r['part'] == w['part']]
    if same:
        aw = np.array([r['area'] for r in same], 'f8')
        c = (np.stack([r['c'] for r in same]) * aw[:, None]).sum(0) / max(aw.sum(), 1e-9)
        tw = np.array([1.0 if r.get('c_from_tile') else 0.0 for r in same], 'f8')
        return c.astype('f4'), ("the %d photographed wall(s) of this building"
                                % len(same)), ('same3' if len(same) >= TONE_SAME_MIN_REFS
                                               else 'same'), \
            float((aw * tw).sum() / max(aw.sum(), 1e-9))
    nc, nwhy, nfrac = near_facing_reference(w, refs)
    if nc is not None:
        return nc, nwhy, 'near', nfrac
    if foot:
        mc, mwhy, mfrac = mass_reference(w, refs, cls, foot)
        if mc is not None:
            return mc, mwhy, 'mass', mfrac
    cen = np.asarray(w['centre'][:2], 'f8')
    near = [(float(np.hypot(*(r['centre'] - cen))), r) for r in refs]
    near = [(dd, r) for dd, r in near if dd <= TONE_MATCH_R]
    if near:
        aw = np.array([r['area'] / (20.0 + dd) ** 2 for dd, r in near], 'f8')
        c = (np.stack([r['c'] for _, r in near]) * aw[:, None]).sum(0) / max(aw.sum(), 1e-9)
        tw = np.array([1.0 if r.get('c_from_tile') else 0.0 for _, r in near], 'f8')
        return c.astype('f4'), ("the %d photographed wall(s) within %.0f m"
                                % (len(near), TONE_MATCH_R)), 'area', \
            float((aw * tw).sum() / max(aw.sum(), 1e-9))
    if cls is not None:
        pc, pwhy, pfrac = class_prior_reference(w, cls, refs)
        if pc is not None:
            return pc, pwhy, 'class', pfrac
    return None, None, None, 0.0


# which branches may take the relaxed tile cap: exactly the three that already
# carry TONE_MAX_DARKEN_NEAR because their evidence is this wall's own surface.
# `area` and `class` are neighbourhood averages and a lone `same` is one
# photograph of one other face, so they keep TONE_MAX_DARKEN = 0.45 whatever
# statistic they are built from.
TONE_TILE_BRANCHES = ('same3', 'near', 'mass')


def darken_cap(branch, tile_frac):
    """The darken floor this reference has earned, and the name for it.

    TONE_MAX_DARKEN_TILE only where the branch is close evidence AND the whole
    of that evidence is a measured packed tile mean -- the value the wall below
    actually renders at, not the trusted-texel median, which can sit well above
    it (1.37x on one measured wall)."""
    if branch in TONE_TILE_BRANCHES and tile_frac >= TONE_TILE_MIN_FRAC:
        return TONE_MAX_DARKEN_TILE, 'packed tile mean'
    return TONE_BRANCH[branch][0], 'trusted-texel median'


def tone_match(c, ref, keep=TONE_KEEP_CHROMA, max_darken=TONE_MAX_DARKEN):
    """Put a synthetic wall's colour on a photographed wall's exposure.

    Luminance comes from the reference outright: the two surfaces stand in the
    same sunlight, so a synthetic wall four times the albedo of the photograph
    beside it is not a paint choice, it is an unexposed guess.  Chromaticity is
    a blend, because OSM building:colour and the class default ARE evidence of
    hue even when they say nothing about exposure."""
    c = np.asarray(c, 'f4'); ref = np.asarray(ref, 'f4')
    gl = float(c @ T.LUMW); gr = float(ref @ T.LUMW)
    if not np.isfinite(gr) or not np.isfinite(gl) or gr <= 1e-4 or gl <= 1e-4:
        return np.asarray(c, 'f4'), None
    chroma = keep * (c / gl) + (1.0 - keep) * (ref / gr)
    raw = float(np.clip(gr, TONE_LUM_FLOOR, TONE_LUM_CEIL))
    # the wall's own luminance is weak evidence of exposure but it is not no
    # evidence: cap the move at TONE_MAX_DARKEN..TONE_MAX_BRIGHTEN of it
    tgt = float(np.clip(raw, max_darken * gl, TONE_MAX_BRIGHTEN * gl))
    capped = None
    if raw < max_darken * gl - 1e-6:
        capped = 'darken'
    elif raw > TONE_MAX_BRIGHTEN * gl + 1e-6:
        capped = 'brighten'
    out = chroma * tgt / max(float(chroma @ T.LUMW), 1e-5)
    out = np.clip(out, 0.01, 0.95).astype('f4')
    return out, dict(before_linear=[round(float(x), 4) for x in c],
                     after_linear=[round(float(x), 4) for x in out],
                     reference_linear=[round(float(x), 4) for x in ref],
                     luminance_before=round(gl, 4), luminance_after=round(tgt, 4),
                     luminance_target_uncapped=round(raw, 4), capped=capped,
                     max_darken=round(float(max_darken), 3))


def plausible_wall_colour(c, cls):
    """Keep a fallback wall colour inside what a building can actually be.

    Clamp the luminance away from black and from white, and cap the saturation:
    a wall is never neon, and a tint taken off a handful of sky or shadow texels
    is exactly how a concrete wall ends up dark blue."""
    c = np.clip(np.asarray(c, 'f4'), 1e-4, 1.0)
    g = float(c @ T.LUMW)
    tgt = float(np.clip(g, 0.045, 0.62))
    c = c * (tgt / max(g, 1e-5))
    mx = float(c.max()); mn = float(c.min())
    sat = (mx - mn) / max(mx, 1e-6)
    cap = 0.55 if cls in ('brick', 'wood') else 0.28
    if sat > cap:
        gg = float(c @ T.LUMW)
        c = gg + (c - gg) * (cap / sat)
    return np.clip(c, 0.01, 0.95).astype('f4')


def vegetation(col):
    """Texels whose colour is foliage, not wall.

    A tree standing a metre off a wall has almost no parallax across a set of
    panoramas taken a metre apart, so the multi-view median keeps it and the
    agreement figure stays low.  Colour is what separates them: a building
    surface is almost never a saturated green."""
    s = np.power(np.clip(col, 0, 1), 1 / 2.2)
    mx = s.max(-1); mn = s.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    g = s[..., 1]
    return (g >= s[..., 0] - 0.005) & (g >= s[..., 2] - 0.005) & (sat > 0.18)


def srgb_hue_sat(rgb):
    s = np.clip(np.power(np.clip(rgb, 0, 1), 1 / 2.2), 0, 1)
    mx = s.max(); mn = s.min()
    sat = (mx - mn) / max(mx, 1e-6)
    r, g, b = s
    if mx - mn < 1e-6:
        hue = 0.0
    elif mx == r:
        hue = 60 * (((g - b) / (mx - mn)) % 6)
    elif mx == g:
        hue = 60 * ((b - r) / (mx - mn) + 2)
    else:
        hue = 60 * ((r - g) / (mx - mn) + 4)
    return float(hue), float(sat), float(mx)


def classify(w, col, trust, sib=None):
    """-> (class, evidence).  OSM first, then the photograph, then the rest of
    the building, then the building's own type."""
    mat = (w['osm'].get('material') or '').lower()
    if mat in OSM_MATERIAL:
        return OSM_MATERIAL[mat], f"OSM building:material={mat}"
    if w['osm'].get('building') == 'ship':
        return 'metal_panel', "OSM building=ship"
    if w['osm'].get('building') == 'roof':
        return 'metal_panel', "OSM building=roof: an open shelter, not a wall"
    if trust is not None and trust.mean() > 0.30:
        m = col[trust]
        rgb = np.median(m, axis=0)
        hue, sat, v = srgb_hue_sat(rgb)
        L = (m @ T.LUMW)
        # storey banding: how much of the luminance variance is horizontal bands
        rows = (col * trust[:, :, None]).sum(1) / np.maximum(trust.sum(1), 1e-6)[:, None]
        rl = rows @ T.LUMW
        band = float(np.std(rl) / max(np.std(L), 1e-6))
        ev = f"photo hue {hue:.0f} deg, sat {sat:.2f}, L {float(L.mean()):.3f}, band {band:.2f}"
        if sat > 0.30 and 2 < hue < 42:
            return 'brick', ev + " -> red masonry"
        if sat > 0.22 and 20 <= hue < 55 and v < 0.55:
            return 'wood', ev + " -> warm timber"
        if float(L.mean()) < getattr(C, 'FACADE_GLASS_L_MAX', 0.085) and rgb[2] >= rgb[0] and band > 0.42:
            return 'glass', ev + " -> dark, blue, strongly banded"
        if sat < 0.11 and float(L.mean()) > 0.20:
            return 'metal_panel', ev + " -> bright and neutral"
        return 'concrete', ev + " -> neutral mid tone"
    if sib:
        return sib, "the classified walls of the same building"
    b = w['osm'].get('building')
    lv = levels_of(w)
    if b in ('commercial', 'office', 'retail') and lv >= 4:
        return 'glass', f"OSM building={b} with {lv} levels"
    if b in ('apartments', 'residential') or lv >= 3:
        return 'concrete', f"OSM building={b}, {lv} levels"
    return 'concrete', "no evidence; the neutral default"


def hexrgb(s):
    s = (s or '').strip()
    if s.startswith('#') and len(s) == 7:
        try:
            v = np.array([int(s[i:i + 2], 16) for i in (1, 3, 5)], 'f4') / 255.0
            return np.power(v, 2.2).astype('f4')
        except ValueError:
            return None
    named = dict(white=(0.75, 0.75, 0.74), red=(0.22, 0.05, 0.04),
                 grey=(0.25, 0.25, 0.25), gray=(0.25, 0.25, 0.25),
                 black=(0.05, 0.05, 0.05), brown=(0.13, 0.08, 0.05),
                 green=(0.09, 0.14, 0.08), blue=(0.07, 0.10, 0.20),
                 yellow=(0.45, 0.38, 0.15), beige=(0.48, 0.44, 0.35))
    if s.lower() in named:
        return np.array(named[s.lower()], 'f4')
    return None


def smooth_fill(col, trust, cell_px):
    """Local colour of a wall: the trusted photo colour, spread into the holes.

    A box mean over a widening window, so a wall with a tree in front of it
    still hands its own colour to the synthetic material behind the tree."""
    ny, nx = trust.shape
    acc = col * trust[:, :, None]
    den = trust.astype('f4').copy()
    k = max(2, int(cell_px))
    for _ in range(7):
        acc = _box(acc, k); den = _box(den, k)
        if (den > 0.02).all():
            break
        k *= 2
        if k > max(ny, nx):
            break
    out = acc / np.maximum(den[:, :, None], 1e-6)
    bad = den < 1e-6
    if bad.any():
        m = col[trust].mean(0) if trust.any() else np.array([0.25, 0.25, 0.25], 'f4')
        out[bad] = m
    return out.astype('f4')


def _box(a, k):
    """Separable box blur with edge clamping; a is HxW or HxWx3."""
    single = a.ndim == 2
    x = a[:, :, None] if single else a
    for ax in (0, 1):
        n = x.shape[ax]
        kk = min(k, max(1, n))
        c = np.cumsum(np.concatenate(
            [np.zeros((1,) + x.shape[1:], 'f4') if ax == 0 else
             np.zeros((x.shape[0], 1) + x.shape[2:], 'f4'), x], axis=ax).astype('f4'), axis=ax)
        i = np.arange(n)
        lo = np.clip(i - kk // 2, 0, n); hi = np.clip(i + kk // 2 + 1, 0, n)
        if ax == 0:
            x = (c[hi] - c[lo]) / np.maximum((hi - lo)[:, None, None], 1)
        else:
            x = (c[:, hi] - c[:, lo]) / np.maximum((hi - lo)[None, :, None], 1)
    return x[:, :, 0] if single else x


# ----------------------------------------------------------------- main --- #
def shelf_pack(sizes, page=ATLAS_PX, pad=PAD):
    """Sort by height, fill shelves.  Returns [(page, x, y)] and page count."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    place = [None] * len(sizes)
    pg = 0; x = pad; y = pad; sh = 0
    for i in order:
        w, h = sizes[i]
        w += 2 * pad; h += 2 * pad
        if w > page or h > page:
            raise ValueError(f"tile {w}x{h} larger than the page")
        if x + w > page:
            x = pad; y += sh; sh = 0
        if y + h > page:
            pg += 1; x = pad; y = pad; sh = 0
        place[i] = (pg, x + pad, y + pad)
        x += w; sh = max(sh, h)
    return place, pg + 1


def _data_colour_knob():
    """stage_knobs.FACADE_DATA_COLOUR.value for the active area ({enabled, k_parts, r_m}),
    else None (an area that does not set the knob keeps the class default)."""
    if not hasattr(C, '_areacfg') or C._areacfg.build_area_cfg(C) is None:
        return None
    v = ((((getattr(C, '_AREA_CFG', None) or {}).get('stage_knobs') or {})
          .get('FACADE_DATA_COLOUR')) or {}).get('value')
    return v if isinstance(v, dict) and v.get('enabled') else None


def main():
    global JOINT_SRGB, PANEL_SRGB, LIFT_SRGB
    ap = argparse.ArgumentParser()
    ap.add_argument('--mad-cut', type=float, default=0.055,
                    help='linear luminance MAD a texel may have and still be '
                         'trusted; 0.055 is the figure the ortho blend already '
                         'uses, 14 sRGB levels on a mid grey')
    ap.add_argument('--cov-cut', type=float, default=0.25)
    ap.add_argument('--stat', default='top2', choices=('median', 'top2', 'best'),
                    help='which per-texel statistic the photograph layer shows; '
                         'the median of six views is robust but smears a facade '
                         'horizontally by the pose error, best is the single '
                         'closest head-on view, top2 the mean of the best two')
    ap.add_argument('--joint-srgb', type=float, default=JOINT_SRGB,
                    help='blind-wall panel joint contrast, as a fraction of the '
                         'wall sRGB value (0.018 is ~4 levels on a light wall)')
    ap.add_argument('--no-mass-rule', action='store_true',
                    help='turn off the building-mass tone reference (the '
                         'rule that puts an upper-storey OSM way on the '
                         'exposure of the photographed building it sits on)')
    ap.add_argument('--out', default=FAC,
                    help='where the atlases, facadetex.json and wallmap.npz are '
                         'written; the inputs are always read from ' + FAC + '. '
                         'A run written anywhere else does not touch status.json')
    a = ap.parse_args()
    if a.joint_srgb != JOINT_SRGB:
        k = a.joint_srgb / JOINT_SRGB
        JOINT_SRGB = a.joint_srgb; PANEL_SRGB *= k; LIFT_SRGB *= k
        # the per-class panel figures ride the same knob, so --joint-srgb
        # still scales every contrast panel_field()/macro_for_wall() draw
        for _c in PANEL_LOOK.values():
            for _f in ('tone', 'bow', 'band'):
                _c[_f] *= k
    OUT = os.path.abspath(a.out)
    os.makedirs(OUT, exist_ok=True)
    is_default = OUT == os.path.abspath(FAC)
    t0 = time.time()
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    doc = json.load(open(FAC + '/walls.json'))
    walls = doc['walls']
    idx = json.load(open(CC0 + '/index.json'))
    sizes_m = tile_sizes(idx)
    bases = {}
    for cls, bm in BASE_MAT.items():
        bases[cls] = (load_base(idx, bm, sizes_m, KEEP_CHROMA[cls])
                      if bm else (None, None))
    print("material tile sizes:", json.dumps(sizes_m), flush=True)

    # per-face measured colour, for walls with no usable texel at all
    pjb = np.load(T.SCR + '/project/buildings_face.npz')
    wnpz = np.load(FAC + '/walls.npz')
    wof = wnpz['wall_of_face']

    # ---- load the projections, pool the MAD distribution ------------------ #
    P = {}
    pool = []
    for w in walls:
        fp = f"{PROJ}/wall_{w['id']:04d}.npz"
        if not os.path.exists(fp):
            continue
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d['meta']))
        extra = {k: d[k] for k in ('rmin', 'best', 'top2') if k in d.files}
        P[w['id']] = dict(col=d['col'], cnt=d['cnt'], mad=d['mad'], **extra, **meta)
        m3 = d['cnt'] >= 3
        if m3.any():
            pool.append(d['mad'][m3][::7])
    pool = np.concatenate(pool) if pool else np.zeros(1, 'f4')
    MAD_CUT = float(a.mad_cut)
    dist = {f'p{p}': round(float(np.percentile(pool, p)) * 255, 2)
            for p in (10, 25, 50, 60, 75, 90, 95)}
    mad_pct = float((pool <= MAD_CUT).mean() * 100)
    print(f"pooled texel MAD distribution (sRGB levels) {dist}; cut {MAD_CUT*255:.2f} "
          f"levels keeps {mad_pct:.1f}% of the texels three or more views saw",
          flush=True)

    # ---- trust masks, window rows per building ---------------------------- #
    rows_by_part = {}
    for w in walls:
        d = P.get(w['id'])
        if d is None:
            continue
        veg = vegetation(d['col'])
        near = d['rmin'] <= RANGE_CUT if 'rmin' in d else np.ones(veg.shape, bool)
        tr = (d['cnt'] >= 3) & (d['mad'] <= MAD_CUT) & near & ~veg
        d['trust'] = tr
        d['veg_frac'] = float((veg & (d['cnt'] >= 3)).mean())
        d['far_frac'] = float(((~near) & (d['cnt'] >= 3)).mean())
        r = window_rows_from_photo(d['col'], tr, w['height_m'])
        if r:
            rows_by_part.setdefault(w['part'], []).append(r)
    rows_part = {k: dict(storey_m=float(np.median([r['storey_m'] for r in v])),
                         phase=float(np.median([r['phase'] for r in v])),
                         n=len(v))
                 for k, v in rows_by_part.items()}
    print(f"storey height measured off a photograph for {len(rows_part)} buildings",
          flush=True)

    # ---- colour and rhythm off the far grid ------------ #
    # stage 34's far pass (common.py FACADE_FAR_*): every panorama within
    # FACADE_FAR_R_M, on a coarse grid.  Too soft to paste; its median colour, its
    # storey rows and its bay pitch are still the building's own.
    from wallphoto.wpfill import period as _period
    FAR = {}; bay_of = {}; far_rows = {}
    for w in walls:
        fp = f"{PROJ}/far/wall_{w['id']:04d}.npz"
        if not os.path.exists(fp):
            continue
        z = np.load(fp, allow_pickle=True)
        m = json.loads(str(z['meta']))
        if m.get('sig') != far_sig(w):
            continue
        fc = z['col']; ok = (z['cnt'] >= 1) & ~vegetation(fc) & (fc.max(-1) > 1e-3)
        tx_ = float(m['tx'])
        FAR[w['id']] = dict(col=fc, ok=ok, tx=tx_, m2=float(ok.sum()) * tx_ * tx_)
        if not ok.any():
            continue
        r = window_rows_from_photo(fc, ok, w['height_m'])
        if r:
            far_rows.setdefault(w['part'], []).append(r)
        L = fc @ T.LUMW
        cp = (L * ok).sum(0) / np.maximum(ok.sum(0), 1)
        P_, _ = _period(cp, ok.mean(0) > 0.3, int(C.FACADE_FILL_BAY_M[0] / tx_),
                        int(C.FACADE_FILL_BAY_M[1] / tx_), C.FACADE_FILL_AC_MIN)
        if P_:
            bay_of[w['id']] = P_ * tx_
    # one far colour per building part: the median over every far texel of its
    # walls, so a building's faces cannot disagree at a corner
    part_far = {}
    for w in walls:
        f = FAR.get(w['id'])
        if f is not None and f['ok'].any():
            part_far.setdefault(w['part'], []).append((f['col'][f['ok']], f['m2']))
    part_far = {k: (np.median(np.concatenate([c for c, _ in v]), axis=0), sum(m for _, m in v))
                for k, v in part_far.items()}
    # stage_knobs.FACADE_DATA_COLOUR (opt-in per area): the
    # colour of a wall nothing photographed comes from the area's own buildings,
    # not CLASS_DEFAULT_RGB.  Per building part: the median of its far-grid texels,
    # else of every near-projected texel (cnt >= 1, not vegetation); a wall then
    # takes the median of the k nearest parts within r_m, else the area median.
    # Run with FACADE_PHOTO_QUALITY.seg_veto so sky / road / trees are not in it.
    _dc = _data_colour_knob()
    palette = {}
    if _dc:
        pc = {}
        for w in walls:
            pc.setdefault(w['part'], []).append(np.asarray(w['centre'][:2], 'f8'))
        near_pool = {}
        for w in walls:
            d = P.get(w['id'])
            if d is None:
                continue
            m = (d['cnt'] >= 1) & ~vegetation(d['col']) & (d['col'].max(-1) > 1e-3)
            if m.any():
                near_pool.setdefault(w['part'], []).append(d['col'][m][::3])
        far_pool = {}
        for w in walls:
            f = FAR.get(w['id'])
            if f is not None and f['ok'].any():
                far_pool.setdefault(w['part'], []).append(f['col'][f['ok']])

        def lit_median(chunks):
            # the sunlit half: a palette colour is an albedo stand-in, and the
            # renderer shades the wall itself (Tucson far grids: shaded north
            # faces at linear L 0.02-0.05 next to lit stucco at 0.38).  Hue from
            # the median of every texel (the lit half alone leans to the
            # sky-blue glints in windows: superior court 0.53/0.48/0.50 vs the
            # full median's tan 0.46/0.37/0.35), luminance from the lit half.
            x = np.concatenate(chunks); L = x @ T.LUMW
            m_all = np.median(x, axis=0)
            l_all = max(float(m_all @ T.LUMW), 1e-4)
            l_lit = float(np.median(L[L >= np.median(L)]))
            return m_all * (l_lit / l_all)
        for k, cen in pc.items():
            if k in far_pool and sum(len(x) for x in far_pool[k]) >= 50:
                col_k = lit_median(far_pool[k]); why = 'far grid'
            elif k in near_pool and sum(len(x) for x in near_pool[k]) >= 200:
                col_k = lit_median(near_pool[k]); why = 'near texels'
            else:
                continue
            palette[k] = dict(c=col_k, xy=np.mean(cen, axis=0), why=why)
        for k_, v_ in palette.items():
            print(f"  palette part {k_}: {np.round(v_['c'], 3).tolist()} ({v_['why']})", flush=True)
        area_med = (np.median(np.array([v['c'] for v in palette.values()]), axis=0)
                    if palette else None)
        print(f"data colour palette: {len(palette)} building parts "
              f"({sum(v['why'] == 'far grid' for v in palette.values())} far grid), "
              f"area median {None if area_med is None else np.round(area_med, 3).tolist()}",
              flush=True)
    data_colour_stats = dict(walls=0, area_m2=0.0, nearest=0, area_median=0)
    n_far_rows = 0
    for k, v in far_rows.items():
        if k not in rows_part:
            rows_part[k] = dict(storey_m=float(np.median([r['storey_m'] for r in v])),
                                phase=float(np.median([r['phase'] for r in v])),
                                n=len(v), far=True)
            n_far_rows += 1
    print(f"far grid: {len(FAR)} walls, {sum(1 for f in FAR.values() if f['m2'] >= C.FACADE_FAR_MIN_M2)} "
          f"over {C.FACADE_FAR_MIN_M2} m2; storey height for {n_far_rows} more buildings; "
          f"bay pitch on {len(bay_of)} walls", flush=True)

    # ---- classify --------------------------------------------------------- #
    cls_of = {}; ev_of = {}
    for w in walls:
        d = P.get(w['id'])
        c, ev = classify(w, d['col'] if d else None, d.get('trust') if d else None)
        cls_of[w['id']] = c; ev_of[w['id']] = ev
    # siblings: a wall with no evidence takes the majority class of the walls of
    # its building that the photographs did decide
    from collections import Counter
    bypart = {}
    for w in walls:
        if ev_of[w['id']].startswith('photo') or ev_of[w['id']].startswith('OSM building:m'):
            if (cls_of[w['id']] == 'glass' and ev_of[w['id']].startswith('photo')
                    and not getattr(C, 'FACADE_GLASS_SIBLINGS', True)):
                continue          # a photographed glass bay stays its own (common.FACADE_GLASS_SIBLINGS)
            bypart.setdefault(w['part'], Counter())[cls_of[w['id']]] += w['area_m2']
    for w in walls:
        if ev_of[w['id']].startswith(('photo', 'OSM building:m', 'OSM building=ship')):
            continue
        c = bypart.get(w['part'])
        if c:
            cls_of[w['id']] = c.most_common(1)[0][0]
            ev_of[w['id']] = "the photographed walls of the same building"

    # coherence: a single small wall classified differently from the rest of its
    # building is the classifier being fooled by one dark photograph, not a
    # change of material.  Otherwise one narrow wall among cream concrete can come
    # out 'glass' and read as a blue slab.
    area_by_part = {}
    for w in walls:
        area_by_part.setdefault(w['part'], 0.0)
        area_by_part[w['part']] += w['area_m2']
    dom = {}
    for k, c in bypart.items():
        dom[k] = c.most_common(1)[0][0]
    n_coh = 0
    for w in walls:
        d0 = dom.get(w['part'])
        if (d0 and cls_of[w['id']] != d0 and
                w['area_m2'] < 0.25 * area_by_part[w['part']]):
            cls_of[w['id']] = d0
            ev_of[w['id']] = ("the dominant material of this building; this wall "
                              "is too small a share of it to differ")
            n_coh += 1
    print(f"{n_coh} small walls took their building's dominant material", flush=True)

    # ---- tone references: the exposure a photographed wall measured -------- #
    # One entry per wall the cameras covered well enough for its own tint, with
    # the median linear colour of its TRUSTED texels only, so a tree or a patch
    # of sky in front of the wall is not part of the exposure.
    # prefer the tile the LAST prep actually packed for this
    # wall over the median of its trusted texels.  See PACK_REF above.
    packtile, packwhy = packed_tile_reference()
    print("tone reference statistic: " + packwhy, flush=True)
    refs = []
    n_pack_stale = 0
    byid_w = {int(w['id']): w for w in walls}
    twins = {}
    for i, wa in enumerate(walls):
        for wb in walls[i + 1:]:
            if (wa['part_name'] == wb['part_name']
                    and abs(float(wa['area_m2']) - float(wb['area_m2'])) <= 0.5
                    and float(np.dot(wa['normal'], wb['normal'])) > 0.98):
                twins.setdefault(int(wa['id']), []).append(int(wb['id']))
                twins.setdefault(int(wb['id']), []).append(int(wa['id']))
    for w in walls:
        d = P.get(w['id'])
        if d is None or 'trust' not in d:
            continue
        tr = d['trust']
        if int(tr.sum()) < TONE_REF_MIN_TEXELS or float(tr.mean()) < TONE_REF_COV:
            continue
        if float(w['width_m']) < TONE_REF_MIN_WIDTH_M:
            continue
        rn = np.asarray(w['normal'][:2], 'f8')
        ru = np.asarray(w['u'][:2], 'f8')
        texmed = np.median(d['col'][tr], axis=0).astype('f4')
        tilec = packtile.get(int(w['id']))
        pm = PACK_META.get(int(w['id']))
        ok = pack_matches(w, pm)
        # a coplanar twin of the same size (e.g. two walls with adjacent ids)
        # still passes the test above under a shift equal to their id gap;
        # then the twin's own pack record names a different wall, so require
        # the twin to match as well
        if ok and tilec is not None:
            for t in twins.get(int(w['id']), ()):
                if not pack_matches(byid_w[t], PACK_META.get(t)):
                    ok = False
                    break
        if tilec is not None and not ok:
            tilec = None
            n_pack_stale += 1
        refs.append(dict(part=w['part'], part_name=w['part_name'],
                         cls=cls_of[w['id']],
                         id=w['id'], area=float(w['area_m2']),
                         c=(tilec if tilec is not None else texmed),
                         c_from_tile=tilec is not None,
                         c_texel_median=texmed,
                         centre=np.asarray(w['centre'][:2], 'f8'),
                         normal=rn / max(float(np.hypot(*rn)), 1e-9),
                         u=ru / max(float(np.hypot(*ru)), 1e-9),
                         half=0.5 * float(w['width_m'])))
    n_tile = sum(1 for r in refs if r['c_from_tile'])
    print(f"{n_pack_stale} pack tile records rejected: id now names a different wall", flush=True)
    print(f"{len(refs)} photographed walls are tone references "
          f"({n_tile} on the previous pack's packed tile mean, "
          f"{len(refs) - n_tile} on their trusted-texel median)", flush=True)
    foot = {} if a.no_mass_rule else part_footprints(walls)
    print(f"{len(foot)} of {len(set(w['part'] for w in walls))} wall parts have "
          f"an OSM footprint for the building-mass rule", flush=True)
    tone_stats = dict(walls_matched=0, walls_unmatched=0, by_source={},
                      by_statistic={},
                      reference_walls_on_packed_tile=n_tile,
                      reference_walls_on_texel_median=len(refs) - n_tile,
                      reference_pack=PACK_REF, reference_pack_read=packwhy,
                      median_luminance_before=[], median_luminance_after=[],
                      capped=[], grazed_walls_on_far_median=0,
                      grazed_walls_kept_tone_match=0)

    # ---- bake ------------------------------------------------------------- #
    tiles = []; recs = []; aux_n = []; aux_r = []
    glass_walls = 0; glass_texels = 0
    for w in walls:
        d = P.get(w['id'])
        cls = cls_of[w['id']]
        trust_frac = float(d['trust'].mean()) if d else 0.0
        use_photo = (d is not None) and trust_frac >= a.cov_cut
        if d is not None and use_photo:
            ny, nx = d['col'].shape[:2]
        else:
            rr = float(np.hypot(w['centre'][0], w['centre'][1]))
            ft = FALLBACK_TEXEL_NEAR if rr < NEAR_R else getattr(
                C, 'FACADE_TEXEL_FAR', FALLBACK_TEXEL_FAR)
            fm = FALLBACK_MAX_NEAR if rr < NEAR_R else getattr(
                C, 'FACADE_MAX_FAR', FALLBACK_MAX_FAR)
            # buried means no camera can ever see it, so a wall three
            # panoramas did see is not buried whatever the footprints say
            if w['buried'] and (d is None or float((d['cnt'] >= 3).mean()) < 0.05):
                ft, fm = 0.25, 128
            nx = int(np.clip(round(w['width_m'] / ft), 8, fm))
            ny = int(np.clip(round(w['height_m'] / ft), 8, fm))
        # colour evidence
        # Colour evidence, best first.  A wall three panoramas barely grazed is
        # NOT colour evidence: at ~7 % coverage what it saw is mostly sky and shadow,
        # which gives a dark blue tint, so the photograph only
        # wins when it covers enough of the wall to mean something, and OSM's
        # building:colour -- a mapper's statement about the building -- beats a
        # sliver of photograph.
        src = None
        tone_det = None
        level_ref = None
        osm_c = hexrgb(w['osm'].get('colour'))
        if (d is not None and trust_frac >= PHOTO_TINT_COV
                and int(d['trust'].sum()) >= 200
                and (use_photo or osm_c is None)):
            # the tint is a 1.2 m field, so it is measured on a downsampled copy:
            # same answer, and it keeps a 2048 px wall out of a gigabyte of
            # intermediate box blurs
            cn, cm_ = d['col'].shape[1], d['col'].shape[0]
            step = max(1, int(max(cn, cm_) / 256))
            cs = d['col'][::step, ::step]; ts = d['trust'][::step, ::step]
            if not ts.any():
                cs = d['col']; ts = d['trust']; step = 1
            cell = max(3, int(1.2 / max(float(d.get('tx', 0.05)) * step, 0.02)))
            tint = smooth_fill(cs, ts, cell)
            src = 'photograph'
        else:
            c = osm_c
            if c is not None:
                src = 'OSM building:colour'
            if c is None and d is not None and trust_frac >= 0.03:
                c = np.median(d['col'][d['trust']], axis=0)
                src = 'the median of what little of this wall was photographed'
            if c is None:
                fm = (wof == w['id'])
                if fm.any() and pjb['nviews'][fm].max() >= 1:
                    aw = pjb['area'][fm] * (pjb['nviews'][fm] > 0)
                    c = (pjb['col'][fm] * aw[:, None]).sum(0) / max(aw.sum(), 1e-6)
                    src = 'the colour stage 20 measured on this wall'
            data_pal = False
            if _dc and palette and (c is None or not np.isfinite(c).all() or c.max() < 1e-3):
                xy = np.asarray(w['centre'][:2], 'f8')
                near = sorted((float(np.hypot(*(v['xy'] - xy))), k) for k, v in palette.items()
                              if k != w['part'])
                near = [k for r_, k in near if r_ <= float(_dc.get('r_m', 150.0))][:int(_dc.get('k_parts', 5))]
                if near:
                    c = np.median(np.array([palette[k]['c'] for k in near]), axis=0)
                    src = f'the area palette: median of the {len(near)} nearest photographed buildings'
                    data_colour_stats['nearest'] += 1
                else:
                    c = np.asarray(area_med, 'f4')
                    src = 'the area palette: median of every photographed building'
                    data_colour_stats['area_median'] += 1
                data_colour_stats['walls'] += 1
                data_colour_stats['area_m2'] += float(w['area_m2'])
                data_pal = True
            if c is None or not np.isfinite(c).all() or c.max() < 1e-3:
                c = np.array(CLASS_DEFAULT_RGB[cls], 'f4'); src = 'the class default'
            c = plausible_wall_colour(np.asarray(c, 'f4'), cls)
            # neither OSM's hex, the class default nor the stage 20 face colour
            # is an exposure -- put this wall on the exposure of the photographs
            # that reach it.
            # a far photograph median is an exposure already (the same delit
            # panoramas the near tint comes from): no tone match onto a neighbour
            rc, rwhy, rbranch, rtile = ((None, None, None, None) if data_pal
                                        else tone_reference(w, refs, cls, foot))
            if rc is not None:
                mdark, rstat = darken_cap(rbranch, rtile)
                c2, tinfo = tone_match(c, rc, max_darken=mdark)
                if tinfo is not None:
                    tone_stats['walls_matched'] += 1
                    # the per-wall evidence names the wall and the distance; the
                    # summary counts branches, not walls
                    sk = TONE_BRANCH[rbranch][1]
                    tone_stats['by_source'][sk] = \
                        tone_stats['by_source'].get(sk, 0) + 1
                    tone_stats['median_luminance_before'].append(
                        tinfo['luminance_before'])
                    tone_stats['median_luminance_after'].append(
                        tinfo['luminance_after'])
                    if tinfo.get('capped'):
                        tone_stats['capped'].append(dict(
                            id=int(w['id']), part=str(w['part']),
                            branch=rbranch, statistic=rstat,
                            luminance_before=tinfo['luminance_before'],
                            luminance_target_uncapped=tinfo['luminance_target_uncapped'],
                            luminance_after=tinfo['luminance_after'],
                            direction=tinfo['capped']))
                    src = (src or 'a colour with no exposure') + \
                        ', tone-matched in linear light to ' + rwhy + \
                        ' (' + rstat + ')'
                    tinfo['branch'] = rbranch
                    tinfo['reference_statistic'] = rstat
                    tinfo['reference_tile_frac'] = round(float(rtile), 3)
                    tone_stats['by_statistic'][rstat] = \
                        tone_stats['by_statistic'].get(rstat, 0) + 1
                    tone_det = tinfo
                    c = c2
            else:
                tone_stats['walls_unmatched'] += 1
            # Far grid: a building the far grid saw takes its far
            # photograph median instead -- an exposure already (the same delit
            # panoramas as the near tint), so no tone match.  The colour it
            # replaces is kept as level_reference: stage 60's exposure gate
            # judges a photograph against that replaced colour.
            # One building, one colour: a near projection that only grazed the
            # wall is not colour evidence, so it does not keep the wall off its
            # building's far median (common.FACADE_FAR_GRAZE_TRUST).
            fr = part_far.get(w['part'])
            gt = getattr(C, 'FACADE_FAR_GRAZE_TRUST', 0.03)
            grazed = d is not None and gt is not None and trust_frac >= gt
            if fr is not None and fr[1] >= C.FACADE_FAR_MIN_M2 and not grazed:
                if d is not None and trust_frac >= 0.03:
                    tone_stats['grazed_walls_on_far_median'] += 1
                cf = plausible_wall_colour(np.asarray(fr[0], 'f4'), cls)
                level_ref = dict(scale=[round(float(x), 5) for x in
                                        np.asarray(c, 'f4') / np.maximum(cf, 1e-5)],
                                 tone_match=tone_det, colour_evidence=src)
                c = cf; tone_det = None
                src = (f"the far photograph median of this building ({fr[1]:.0f} m2 "
                       f"seen within {C.FACADE_FAR_R_M:.0f} m)")
            elif fr is not None and fr[1] >= C.FACADE_FAR_MIN_M2:
                tone_stats['grazed_walls_kept_tone_match'] += 1
            tint = (np.tile(c[None, None, :], (ny, nx, 1))
                    * wall_ao(ny, nx, w['height_m'], w['width_m'])[:, :, None])
        base, tile_m = bases[cls]
        rows = rows_part.get(w['part'])
        near_core = float(np.hypot(*w['centre'][:2])) < 130.0   # within 130 m of the area origin
        # A window grid is drawn only where the storey height is evidence:
        # a building:levels count on the way, or a storey height measured off a
        # photographed wall of the same building part.  Nothing else.
        storey_src = storey_evidence(w, rows)
        # one joint grid per wall: where stage 22's MACRO field draws the form
        # panels, panel_field() does not draw a second set at the CC0 tile's
        # pitch on top of them
        # A storey band is drawn only on a wall that has a level count AND no
        # window grid to carry it: on an open shelter (building=roof) the level
        # count is the only thing that says where its floors are, and on a
        # hull there are no floors to draw.  A wall the photograph covers keeps
        # the photograph's own storeys.
        band_m = None
        if (storey_src == 'building:levels' and not use_photo
                and w['osm'].get('building') == 'roof'):
            band_m = w['height_m'] / max(levels_of(w), 1)
        macro = macro_for_wall(cls, ny, nx, w['width_m'] / max(nx, 1),
                               band_m=band_m, tone_seed=w['id'])
        oo = {}; pb = {} if getattr(C, 'FACADE_PBR', False) else None
        syn, ginfo = synth_facade(w, cls, nx, ny, base, tile_m, rows,
                                  seed=1000 + w['id'], near_core=near_core,
                                  no_windows=use_photo, storey_src=storey_src,
                                  pitch=(None if macro is not None
                                         else blind_pitch(cls, sizes_m)),
                                  openings_out=oo, bay_m=bay_of.get(w['id']),
                                  pbr_out=pb)
        if pb is not None and macro is not None and pb.get('height') is not None:
            # the MACRO field's joints and pour lines, recessed
            lm = np.log(np.maximum(macro, 1e-3))
            pb['height'] = pb['height'] + C.FACADE_MOD['macro_relief_m'] * (
                lm - _boxf(lm, max(3, round(0.4 * nx / max(w['width_m'], 1e-3)))))
        if tint.shape[:2] != (ny, nx):
            ti = Image.fromarray(np.clip(tint * 255, 0, 255).astype('u1'))
            tint = np.asarray(ti.resize((nx, ny), Image.BILINEAR), 'f4') / 255.0
        img = syn * tint
        if macro is not None:
            # metre-scale structure, in linear light, on the synthetic wall
            # only: the photograph below carries its own and is blended over
            # this a few lines down.  Without it a synthetic wall tile measures
            # sd 5.3 sRGB levels against 38.9 on the photographed wall below it.
            img = img * macro[:, :, None]
        # the openings are the last thing the tint touches: their colour is the
        # glass's own, not the wall's (see GLASS_BASE_RGB)
        omask = oo.get('mask')
        img = apply_opening_floor(img, omask)
        img = apply_modules(img, pb)
        glass_texels += int(omask.sum()) if omask is not None else 0
        glass_walls += 1 if (omask is not None and omask.any()) else 0
        photo_frac = 0.0
        if use_photo:
            sharp = d.get(a.stat)
            if sharp is None or sharp.shape != d['col'].shape:
                sharp = d['col']
            tr = d['trust'].astype('f4')
            trs = np.clip(_box(tr, 5), 0, 1)
            trs = np.where(tr > 0.5, 1.0, trs * 0.9)
            img = sharp * trs[:, :, None] + img * (1 - trs[:, :, None])
            photo_frac = float(trs.mean())
        if pb is not None:
            _n, _r = pbr_pages(pb, w['width_m'] / max(nx, 1), ny, nx,
                               photo_alpha=(trs if use_photo else None))
            aux_n.append(_n[::-1]); aux_r.append(_r[::-1])
        img = np.clip(img, 0.0, 1.0)
        srgb = np.clip(np.power(img, 1 / 2.2) * 255, 0, 255).astype('u1')
        tiles.append(srgb[::-1])                        # row 0 of the tile is the top
        recs.append(dict(
            id=w['id'], part_name=w['part_name'], osm_id=w['osm_id'],
            name=w['osm'].get('name'), width_m=w['width_m'], height_m=w['height_m'],
            area_m2=w['area_m2'], buried=w['buried'],
            cls=cls, class_evidence=ev_of[w['id']],
            texture='photograph over material' if use_photo else 'material',
            colour_evidence=src,
            px=[nx, ny], cm_per_texel=round(100 * w['width_m'] / nx, 2),
            panos=(d or {}).get('panos') if d else 0,
            views_median=float((d or {}).get('views_median', 0)) if d else 0.0,
            coverage_any=round(float((d['cnt'] >= 1).mean()), 4) if d else 0.0,
            coverage_ge3=round(float((d['cnt'] >= 3).mean()), 4) if d else 0.0,
            trusted_frac=round(trust_frac, 4),
            vegetation_frac=round(float((d or {}).get('veg_frac', 0.0)), 4),
            beyond_range_frac=round(float((d or {}).get('far_frac', 0.0)), 4),
            photo_frac_final=round(photo_frac, 4),
            tone_match=tone_det,
            macro_detail=(MACRO_CLASS.get(cls) if macro is not None else None),
            far_seen_m2=round(FAR[w['id']]['m2'], 1) if w['id'] in FAR else 0.0,
            far_bay_m=(round(bay_of[w['id']], 2) if w['id'] in bay_of else None),
            far_rows=bool(rows and rows.get('far')),
            level_reference=level_ref,
            mad_L=round(float(np.median(d['mad'][d['cnt'] >= 3]) * 255), 2)
            if d is not None and (d['cnt'] >= 3).any() else None,
            **ginfo))
    # ---- pack ------------------------------------------------------------- #
    place, npages = shelf_pack([(t.shape[1], t.shape[0]) for t in tiles])
    pages = [np.zeros((ATLAS_PX, ATLAS_PX, 3), 'u1') for _ in range(npages)]
    for t, (pg, x, y), r in zip(tiles, place, recs):
        h, wd = t.shape[:2]
        pages[pg][y:y + h, x:x + wd] = t
        # bleed the border into the padding so a mip never samples a neighbour
        pages[pg][y - PAD:y, x:x + wd] = t[0][None, :, :]
        pages[pg][y + h:y + h + PAD, x:x + wd] = t[-1][None, :, :]
        pages[pg][y - PAD:y + h + PAD, x - PAD:x] = pages[pg][y - PAD:y + h + PAD, x:x + 1]
        pages[pg][y - PAD:y + h + PAD, x + wd:x + wd + PAD] = pages[pg][y - PAD:y + h + PAD, x + wd - 1:x + wd]
        r['atlas'] = pg
        r['rect_px'] = [int(x), int(y), int(wd), int(h)]
    files = []
    for i, p in enumerate(pages):
        fp = f"{OUT}/atlas_{i:02d}.jpg"
        Image.fromarray(p).save(fp, 'JPEG', quality=92, optimize=True)
        files.append(fp)
    for f in os.listdir(OUT):
        if f.startswith('atlas_') and (f.endswith('_n.jpg') or f.endswith('_r.jpg')):
            os.unlink(OUT + '/' + f)
    if getattr(C, 'FACADE_PBR', False) and len(aux_n) == len(tiles):
        # normal + roughness pages, same layout as the colour pages
        for kind, aux, fill in (('n', aux_n, NRM_FLAT), ('r', aux_r, MR_FLAT)):
            for i, p in enumerate(pack_aux(aux, place, npages, fill)):
                Image.fromarray(p).save(f"{OUT}/atlas_{i:02d}_{kind}.jpg", 'JPEG',
                                        quality=95 if kind == 'n' else 90, optimize=True)
    print(f"{len(tiles)} wall tiles packed into {npages} x {ATLAS_PX} px atlases, "
          f"{sum(os.path.getsize(f) for f in files)/1e6:.1f} MB", flush=True)

    # ---- the map stage 22 needs ------------------------------------------- #
    z = np.load(C.W + '/layers/buildings.npz')
    pos = z['pos'].astype('f8'); bidx = z['idx']
    nfb = len(bidx)
    face_atlas = np.full(nfb, -1, 'i1')
    uv = np.zeros((nfb, 3, 2), 'f4')
    fnrm = np.zeros((nfb, 3), 'f4')
    wall_id = np.full(nfb, -1, 'i4')
    for w, r in zip(walls, recs):
        x, y, wd, h = r['rect_px']
        O = np.array(w['origin']); U = np.array(w['u'])
        for f in w['faces']:
            p = pos[bidx[f]]
            du = (p - O) @ U / w['width_m']
            dv = (p[:, 2] - O[2]) / w['height_m']
            du = np.clip(du, 0, 1); dv = np.clip(dv, 0, 1)
            uv[f, :, 0] = (x + du * wd) / ATLAS_PX
            uv[f, :, 1] = (y + (1 - dv) * h) / ATLAS_PX
            face_atlas[f] = r['atlas']
            fnrm[f] = w['normal']
            wall_id[f] = w['id']
    np.savez_compressed(OUT + '/wallmap.npz', face_atlas=face_atlas, uv=uv,
                        fnrm=fnrm, wall_id=wall_id,
                        atlas_files=np.array([os.path.basename(f) for f in files], object))

    A = np.array([r['area_m2'] for r in recs])
    ph = np.array([r['texture'] != 'material' for r in recs])
    bu = np.array([r['buried'] for r in recs])
    from collections import Counter
    note = dict(
        walls=len(recs), atlases=npages, atlas_px=ATLAS_PX,
        atlas_bytes=int(sum(os.path.getsize(f) for f in files)),
        mad_cut_sRGB_levels=round(MAD_CUT * 255, 2),
        mad_cut_keeps_pct_of_seen_texels=round(mad_pct, 1),
        mad_cut_why="the same view-agreement cut the ortho blend uses",
        range_cut_m=RANGE_CUT, photo_statistic=a.stat,
        vegetation_veto_area_m2=round(float(sum(
            r['area_m2'] * r['vegetation_frac'] for r in recs)), 1),
        beyond_range_area_m2=round(float(sum(
            r['area_m2'] * r['beyond_range_frac'] for r in recs)), 1),
        keep_chroma=KEEP_CHROMA,
        texel_mad_distribution_sRGB=dist, coverage_cut=a.cov_cut,
        trusted_frac_distribution={
            f'p{q}': round(float(np.percentile(
                [r['trusted_frac'] for r in recs if r['coverage_ge3'] > 0.05], q)), 3)
            for q in (10, 25, 50, 75, 90)} if any(
                r['coverage_ge3'] > 0.05 for r in recs) else {},
        photo_walls=int(ph.sum()), photo_area_m2=round(float(A[ph].sum()), 1),
        fallback_walls=int((~ph).sum()), fallback_area_m2=round(float(A[~ph].sum()), 1),
        buried_walls=int(bu.sum()), buried_area_m2=round(float(A[bu].sum()), 1),
        area_total_m2=round(float(A.sum()), 1),
        class_area_m2={k: round(float(A[[r['cls'] == k for r in recs]].sum()), 1)
                       for k in CLASSES},
        class_walls=dict(Counter(r['cls'] for r in recs)),
        class_evidence=dict(Counter(r['class_evidence'].split(';')[0].split(' ->')[0][:34]
                                    for r in recs)),
        colour_evidence=dict(Counter(str(r['colour_evidence']) for r in recs)),
        **({'data_colour': dict(data_colour_stats, area_m2=round(data_colour_stats['area_m2'], 1))} if _dc else {}),
        tone_match=dict(
            walls_matched=tone_stats['walls_matched'],
            walls_with_no_photograph_in_reach=tone_stats['walls_unmatched'],
            # one building, one colour: hit counts of the
            # graze rule, see common.FACADE_FAR_GRAZE_TRUST
            far_median_graze_trust=getattr(C, 'FACADE_FAR_GRAZE_TRUST', 0.03),
            grazed_walls_on_far_median=tone_stats['grazed_walls_on_far_median'],
            grazed_walls_kept_tone_match=tone_stats['grazed_walls_kept_tone_match'],
            reference=tone_stats['by_source'],
            # which statistic each matched wall's reference was
            reference_statistic=tone_stats['by_statistic'],
            packed_tile_reference=dict(
                pack=tone_stats['reference_pack'],
                read=tone_stats['reference_pack_read'],
                field=TONE_TILE_STAT,
                reference_walls_on_packed_tile=tone_stats['reference_walls_on_packed_tile'],
                reference_walls_on_texel_median=tone_stats['reference_walls_on_texel_median'],
                max_darken=TONE_MAX_DARKEN_TILE,
                branches=list(TONE_TILE_BRANCHES),
                min_weight_share=TONE_TILE_MIN_FRAC,
                tile_luminance_floor=TONE_TILE_MIN_LUM,
                why="the trusted-texel median of a photographed wall is not "
                    "what that wall renders at: the stage 60 bake levels, "
                    "blends and re-tints it, so a wall's texel median can be "
                    "0.1801 while its packed tile is 0.1310, and matching a "
                    "neighbour onto the texel median leaves a visible step.  The "
                    "relaxed darken cap applies only to a close-evidence "
                    "branch whose whole weight is a measured packed tile"),
            keep_chroma=TONE_KEEP_CHROMA, reach_m=TONE_MATCH_R,
            luminance_change_limits=[TONE_MAX_DARKEN, TONE_MAX_BRIGHTEN],
            near_reference=dict(reach_m=TONE_NEAR_R, facing_cos=TONE_NEAR_FACING,
                                min_area_m2=TONE_NEAR_MIN_AREA,
                                max_darken=TONE_MAX_DARKEN_NEAR),
            building_mass_reference=dict(
                vertices_inside=TONE_MASS_INSIDE,
                max_darken=TONE_MAX_DARKEN_NEAR,
                parts_with_a_footprint=len(foot),
                same_class_only=True,
                why="a synthetic mass whose OSM footprint lies inside a "
                    "photographed building's footprint is that building's "
                    "upper storeys, not its neighbour, so it takes that "
                    "building's exposure across all its faces; tried after "
                    "the same-facing rule, which keeps every wall it already "
                    "had"),
            same_building_reference=dict(
                min_photographed_walls_for_tight_cap=TONE_SAME_MIN_REFS,
                max_darken_tight=TONE_MAX_DARKEN_NEAR,
                max_darken_default=TONE_MAX_DARKEN,
                why="a building with several photographed faces has measured "
                    "its own light from more than one direction, so a "
                    "synthetic face of it is held to the same cap as the "
                    "same-facing rule instead of the weaker 0.45 floor"),
            class_prior_reference=dict(
                reach_m=TONE_CLASS_R, walls_taken=TONE_CLASS_N,
                max_darken=TONE_MAX_DARKEN,
                why="a wall with no photograph within %.0f m takes the nearest "
                    "photographed walls of its own material, facing ignored, "
                    "instead of a class default with no exposure behind it"
                    % TONE_MATCH_R),
            walls_capped=len(tone_stats['capped']),
            walls_capped_darken=sum(1 for r in tone_stats['capped']
                                    if r['direction'] == 'darken'),
            walls_capped_brighten=sum(1 for r in tone_stats['capped']
                                      if r['direction'] == 'brighten'),
            capped_walls=sorted(tone_stats['capped'],
                                key=lambda r: -abs(r['luminance_target_uncapped']
                                                   - r['luminance_before'])),
            median_linear_luminance_before=round(float(np.median(
                tone_stats['median_luminance_before'])), 4)
            if tone_stats['median_luminance_before'] else None,
            median_linear_luminance_after=round(float(np.median(
                tone_stats['median_luminance_after'])), 4)
            if tone_stats['median_luminance_after'] else None,
            ambient_occlusion=dict(foot=AO_FOOT, foot_m=AO_FOOT_M,
                                   edge=AO_EDGE, edge_m=AO_EDGE_M),
            rule='a wall with no photograph of its own takes the linear '
                 'luminance of the photographed walls of its building, or of '
                 'the photographed walls within reach, and %.2f of its own '
                 'chromaticity' % TONE_KEEP_CHROMA),
        macro_detail_walls=dict(Counter(
            str(r.get('macro_detail')) for r in recs if r.get('macro_detail'))),
        window_grid_walls=dict(
            drawn=int(sum(1 for r in recs if r['storeys'] > 0)),
            drawn_fallback=int(sum(1 for r in recs
                                   if r['storeys'] > 0 and r['texture'] == 'material')),
            evidence=dict(Counter(r['window_rows_source'] for r in recs
                                  if r['storeys'] > 0)),
            blind_fallback=int(sum(1 for r in recs
                                   if r['storeys'] == 0 and r['texture'] == 'material')),
            blind_reason=dict(Counter(r['window_rows_source'] for r in recs
                                      if r['storeys'] == 0 and r['texture'] == 'material'))),
        window_glass=dict(
            base_linear=list(GLASS_BASE_RGB),
            ambient_floor_linear=list(GLASS_AMBIENT_RGB),
            walls_with_openings=int(glass_walls),
            opening_texels=int(glass_texels),
            why="glass has an absolute colour: the base above is an absolute "
                "linear floor under the opening texels after the tint "
                "multiply, shaped by the opening's own structure so the pane "
                "variation and the reveal survive, and the ambient floor is a "
                "hard per-channel minimum under that, so a wall tinted at the "
                "luminance floor does not draw black windows (a tint x mean-1 "
                "modulation field alone would)"),
        blind_wall_panel=dict(
            joint_sRGB_fraction=JOINT_SRGB, panel_sRGB_fraction=PANEL_SRGB,
            lift_sRGB_fraction=LIFT_SRGB,
            panel_look={c: dict(v) for c, v in PANEL_LOOK.items()},
            pitch_m={c: blind_pitch(c, sizes_m) for c in CLASSES},
            why="a blind wall with nothing on it reads as a painted plane; the "
                "joints give it a size without being visible as stripes"),
        window_rows_from_photograph=len(rows_part),
        storey_heights_measured={str(k): round(v['storey_m'], 2) for k, v in rows_part.items()},
        material_tile_sizes=sizes_m,
        cm_per_texel=dict(
            photo_median=round(float(np.median([r['cm_per_texel'] for r in recs if r['texture'] != 'material'])), 2) if ph.any() else None,
            fallback_median=round(float(np.median([r['cm_per_texel'] for r in recs if r['texture'] == 'material'])), 2)),
        licence=dict(photo="Mapillary imagery CC BY-SA 4.0 (colour only)",
                     material="ambientCG CC0 1.0",
                     windows="window and door grid generated in s35_facadetex.py"),
        seconds=round(time.time() - t0), per_wall=recs)
    json.dump(note, open(OUT + '/facadetex.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in note.items()
                      if k not in ('per_wall', 'storey_heights_measured')}, indent=1))
    if is_default:
        C.note('facades', {k: v for k, v in note.items() if k != 'per_wall'})
    else:
        print(f"written to {OUT}; status.json not touched", flush=True)


if __name__ == '__main__':
    main()
