"""Stage 85 -- put the s84 object inventory into a GLB.

The same machinery and the same shape as s82_prop_swap
(the GLB reader, Appender and compact_write come from s38_tree_models, and the
(x, z, -y) convention is the model's), one scene root:

  LAYER_objects
    obj_0001   a CC0 model instance: translation, yaw about +Y, scale from the
               measured footprint, extras with the whole inventory tag
    obj_0002   an EMPTY node for a kind with no model yet: same translation and
               extras, no mesh, extras.placeholder_box_m says how big the thing
               is so a downstream tool can substitute an asset later
    ...

Two things s82 does not do:

  variants   a car is not one model: the library has sedan, suv, van, taxi, and
             which one an instance gets is a hash of its id, so the same
             inventory always builds the same map.  Boats pick by measured
             length, because a 4 m and a 12 m hull are different boats.
  colour     the measured colour of an object (ortho median over its footprint)
             is baked into a per-colour copy of the mesh, snapped to a 10-entry
             palette so the copies stay countable.  The blend is weighted by
             vertex brightness, so tyres and windows stay dark.

A node with no model draws in one of two forms, unless it is a position
marker (cars, trucks, buses, motorcycles, bicycles, rail vehicles and boats
under way: no mesh, extras only) or its kind is hidden by
stage_knobs.OBJECT_PLACEHOLDER_HIDE:

  cage   the twelve edges of the placeholder box as 0.06 m bars, faces open,
         for anything 0.8 m tall or taller, so a placeholder cannot stand in
         front of the view as a solid wall
  box    a solid box, for the short ground clutter (manholes, cones, cameras)

  place  s85_object_place.py place IN OUT [--draco]
  check  s85_object_place.py check FILE
"""
import os, sys, json, math, time, argparse, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
import common as C
from s38_tree_models import (GLB, Appender, compact_write, instanced_triangles,
                             tri_of_mesh, gt)
import s80_furniture_inventory as S
from objects.objcommon import SCR, hash_unit
from objects import objclasses as K

AD = C.W + '/assets/objects'
INV = C.W + '/object_inventory.json'
LAYER = 'LAYER_objects'
SCRATCH = SCR + '/work'

# kind -> the asset classes that may serve it; a hash of the object id picks one
VARIANTS = {
    'car': ['car_sedan', 'car_suv', 'car_van', 'car_taxi'],
    'truck': ['truck', 'delivery'],
    'bus': ['delivery'],
    'boat': ['boat_row', 'boat_speed', 'boat_sail', 'boat_tow'],
    'utility_pole': ['utility_pole'],
    'traffic_light': ['traffic_light'],
    'sign_board': ['sign_board'],
    'traffic_cone': ['traffic_cone'],
    'jersey_barrier': ['jersey_barrier'],
    'utility_cabinet': ['utility_cabinet'],
    'planter': ['planter'],
}
# a boat this long or shorter gets this hull
BOAT_BY_LENGTH = [(5.5, 'boat_row'), (8.0, 'boat_speed'), (11.0, 'boat_sail'),
                  (1e9, 'boat_tow')]
# vehicles and boats are stretched to the measured footprint; poles are not
FIT_FOOTPRINT = ('car', 'truck', 'bus', 'boat')
FIT_HEIGHT = ('utility_pole', 'traffic_light', 'sign_board', 'jersey_barrier',
              'utility_cabinet', 'planter')
JITTER = 0.03
PALETTE = [(240, 240, 240), (200, 200, 200), (150, 150, 150), (90, 90, 95),
           (35, 35, 38), (150, 60, 55), (55, 75, 135), (45, 90, 70),
           (170, 150, 110), (120, 95, 70)]
BLEND = 0.55
# hulls whose kit paint is replaced by the measured colour instead of tinted
REPAINT_ASSETS = ('car_sedan', 'car_suv', 'car_van', 'car_taxi', 'truck',
                  'delivery', 'boat_row', 'boat_speed', 'boat_sail', 'boat_tow')
BODY_SAT = 0.35          # kit chroma/max at or above this is a painted panel
NEUTRAL_PAL = 2          # the mid grey, for a hull with no measured colour
# How far a single repainted vertex may sit off the class/measured colour.
#
# repaint() scales the target by L / median(L), and the Kenney kits' own shading
# spans a factor of nearly two (a utility pole's kit body runs 1.85x from median to
# max, a traffic light's 1.70x).  Unbounded, a pole asked for weathered wood
# (120, 95, 70) comes out with a mesh mean about 26 % bright and peach-coloured
# highlights.  The palette texture is not the cause: sampled at the kit UVs it is
# ~250/255 white, and the shared material carries no baseColorFactor, so COLOR_0
# reaches the renderer unmultiplied.
#
# So the relative shading is CLAMPED to +/- SHADE_SPAN, which puts the pole's modal
# vertex exactly on the target and its brightest vertex about 30 % above it.
SHADE_SPAN = 0.30
# Which vertices of a kit asset the CLASS colour replaces.
#   kit_neutral   the desaturated structural mass only (the default)
#   all           every vertex: a creosoted pole is wood from butt to top, and
#                 the kit's saturated cap (233,145,103) and crossarm
#                 (182,102,70) are 12.5 % of it and would stay peach
#   lens_keep     everything except a real signal lens.  Measured on the baked
#                 asset, 21.7 % of the traffic light's vertices are saturated
#                 and ALL of them sit in y 2.83-3.95 m, the head: the Kenney
#                 housing is amber-yellow kit paint (255,160,57 / 255,204,93),
#                 not a lens, so kit_neutral would leave the whole signal head
#                 amber and turn only the mast grey.  Only the green (83,189,132 /
#                 32,137,107) and red (245,104,67) texels are really lenses.
CLASS_REPAINT = {'utility_pole': 'all', 'traffic_light': 'lens_keep'}
# Kit paint is not a measurement.  The Kenney city-kit models take their whole
# structural surface from one blue-violet slot of the kit's colormap: measured
# on the baked assets, 59 % of the utility pole's vertices are #666b80 and the
# model's mean is #857f91, so kit paint would render an unmeasured pole mauve.
# The sign board and the jersey barrier are that slot end to end.  A row with no
# measured colour takes its CLASS default instead, and the default goes
# through repaint(), not tinted(), so the violet is replaced outright rather
# than blended 55 % of the way out of.
#
# Only the classes whose kit body IS that violet slot are listed.  traffic_cone
# is not: its kit colour is a real safety orange (#ff9044, chroma 0.73) with a
# white base stripe, and repainting that would be a regression.
CLASS_RGB = {
    'utility_pole':   (120, 95, 70),     # weathered creosoted wood
    'traffic_light':  (90, 90, 95),      # dark grey signal pole and head
    'sign_board':     (150, 150, 150),   # galvanised post, blank panel
    'jersey_barrier': (200, 200, 200),   # unpainted precast concrete
}
# kinds that never get geometry: a vehicle's position is a typical one, not a
# fact about the map.  Boats are not here: a moored hull is usually at a fixed berth.
TRANSIENT_KINDS = ('car', 'truck', 'bus', 'motorcycle', 'bicycle')
# A fixed-kind node without a model draws its own placeholder_box_m, coloured
# by class (an edge cage at CAGE_MIN_H_M and taller, a box below, in a BLEND
# material), unless the area hides that kind; position markers (MARKER_KINDS) draw
# nothing.  A non-static row is fainter than a static one -- its box marks a
# typical position, a kiosk box is a real object nobody has modelled yet.  One material and one mesh per class keeps the cost
# to a handful of triangles.
# No entry here may be mint (green the max channel, chroma over 0.10, green
# over 90): on the ground it reads as a mis-tinted signal, so the cycle and
# motorcycle colours and the planter box are slate and earth.
BOX_COLOUR = {
    'car': (110, 130, 160), 'truck': (110, 130, 160), 'bus': (110, 130, 160),
    'motorcycle': (120, 140, 150), 'bicycle': (120, 140, 150),
    'boat': (90, 140, 175), 'kiosk': (230, 150, 60), 'shelter': (230, 150, 60),
    'playground': (235, 175, 70), 'artwork': (215, 120, 180),
    'planter': (150, 135, 110), 'unknown_cluster': (145, 145, 148),
    'ground_fixture': (150, 150, 155), 'banner': (210, 205, 125),
    'traffic_sign_face': (210, 205, 125), 'sign_board': (210, 205, 125),
    'traffic_light': (200, 170, 90), 'utility_cabinet': (140, 150, 140),
    'utility_pole': (160, 140, 120), 'camera': (170, 170, 190),
    'traffic_cone': (235, 120, 60),
}
BOX_RGB_DEFAULT = (180, 180, 185)
# -- A PLACEHOLDER IS NO BRIGHTER THAN ITS GROUND.
# Untoned, placeholder cages and boxes render far brighter than the photo ground
# and terrain they stand on (banner cages about 1.7x).  A placeholder writes its BOX_COLOUR into
# COLOR_0 AND baseColorFactor, glTF multiplies the two, so what reaches the
# ground's base->pixel curve (GROUND_RENDER_K/E) is rgb*rgb/255: banner
# (210,205,125) is base L 159, against a typical plaza ground of ~70.
# Each kind's colour is scaled uniformly by sqrt(min(1, G / L(rgb*rgb/255))),
# G the median Rec.709 luminance of GroundColour's sample under that kind's own
# placeholder rows (annulus GROUND_SAMPLE_GAP_M..+RING_M, same base space), so
# hue and the class colour code survive (kiosk still orange, cone still safety
# orange, banner still yellow) and only brightness moves.  Kinds already under
# their ground are untouched.  Kit-asset repaint (vehicles, boats, poles) and
# the position-marker records are NOT touched by this: it acts in box_style() only.
BOX_TONE = {}            # kind -> factor, filled by do_place(); empty = no-op
W709 = np.array([0.2126, 0.7152, 0.0722])


def box_tone_factors(objs, src, js):
    """-> ({kind: factor}, report).  See the 'A PLACEHOLDER IS NO BRIGHTER THAN ITS
    GROUND' block above."""
    t0 = time.time()
    grd = GroundColour(src, js)
    rin = GROUND_SAMPLE_GAP_M
    by = collections.defaultdict(list)
    for o in objs:
        g = grd.at(float(o['x']), float(o['y']), rin, rin + GROUND_SAMPLE_RING_M)
        if g is not None:
            by[o['kind']].append(float(np.asarray(g, 'f8') @ W709))
    allg = [v for vs in by.values() for v in vs]
    fac, rep = {}, dict(ground_prims=grd.prims, kinds={})
    if allg:
        for kind in sorted({o['kind'] for o in objs}):
            rgb = np.asarray(BOX_COLOUR.get(kind, BOX_RGB_DEFAULT), 'f8')
            eff = float((rgb * rgb / 255.0) @ W709)
            gs = by.get(kind) or allg
            tgt = float(np.median(gs))
            f = math.sqrt(min(1.0, tgt / max(eff, 1e-6)))
            if f < 1.0:
                fac[kind] = f
            rep['kinds'][kind] = dict(ground_L=round(tgt, 1),
                                      base_L=round(eff, 1), factor=round(f, 3),
                                      ground_samples=len(by.get(kind) or []))
    else:
        rep['skipped'] = 'no textured ground in the input; BOX_COLOUR kept'
    rep['seconds'] = round(time.time() - t0, 1)
    return fac, rep
BOX_ALPHA_TRANSIENT = 0.28
BOX_ALPHA_FIXED = 0.45
# unknown_cluster placeholders drawn magenta (200,80,200) are the loudest thing in
# a top-down view.  They keep their row in object_inventory.json,
# the CSV and the manifest; only what is DRAWN changes: a muted grey at 30 %
# alpha, declared with alphaMode BLEND so a renderer that honours alpha shows
# them as a hint rather than as an object.  (A renderer that ignores alphaMode,
# such as pyrender with default materials, still draws them solid; grey and
# open-caged rather than magenta and solid is the part that reaches it.)
FAINT_KINDS = ('unknown_cluster',)
FAINT_ALPHA = 0.30
# A 0.25 m floor on EVERY axis of a placeholder would turn the class table's
# thin panels into slabs: a banner is 1.0 x 0.10 x 2.0 m in objclasses.SPEC and
# would be drawn 0.25 m thick, a manhole 0.05 m proud of the ground drawn 0.25 m
# proud.  So the floor only stops a zero-sized box.
BOX_MIN_M = 0.05
# A size clamp, on the number it can actually act on.
#
# The PLACEHOLDER BOX is not clamped against the class median box:
# raw_box() deliberately returns the class constant (see its docstring -- sensor
# blob extents make placeholders worse, so they are not substituted), so every
# class has exactly ONE distinct box and max/median is 1.000.
#
# The measurement that DOES drive geometry is footprint_m on the rows whose
# footprint_from == 'measured': instance_scale() stretches a boat or a vehicle
# asset to it.  Against a 3x class median a few measured rows are blown-up
# clusters (a boat cluster can be several times its class median), and instance_scale's
# [0.55, 2.5] clip only limits how giant.  So the clamp bounds the measured
# footprint, before anything reads it.
SIZE_CLAMP_FACTOR = 3.0
# MARKER_KINDS: vehicles and boats under way are POSITION MARKERS, a typical
# position rather than a map feature, recorded as nodes with extras and no mesh
# (see the block below).  A moored hull with a measured footprint still gets the
# Kenney watercraft model scaled to its own length and beam (pick_asset and
# instance_scale size it from footprint_m).  A boat caught under way has no
# footprint; an 8.0 x 2.8 x 2.5 m solid box 6 m from an eye-level camera fills
# about a quarter of the view.  A boat under way is a typical position exactly
# like a parked car, so it is the same no-mesh position marker.
MARKER_KINDS = ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'rail_vehicle',
             'boat')
# The ground's base-to-pixel render curve, used by the placeholder and prop tone
# caps (box_tone_factors() here, s82 tone_to_ground()):
#
#   ground  rendered = 5.0356 * base**0.7163   (base = COLOR_0 x atlas texel,
#           glTF-linear, what GroundColour reads)
#
# fitted over 65 rendered footprints (per-object pixels and the ground ring
# outside them, from an id buffer) in a pyrender scene with a directional light
# of intensity 3.5 and an ambient light of 0.55; other lighting needs a re-fit.
GROUND_RENDER_K, GROUND_RENDER_E = 5.0356, 0.7163
# Where the local ground colour comes from, and how finely it is kept.
GROUND_PREFIX = ('ground_', 'path_', 'stitch_', 'photo_ground')
GROUND_SAMPLE_CELL_M = 2.0   # grid cell the input GLB's ground colour is binned to
GROUND_SAMPLE_GAP_M = 0.8    # skip this much around the footprint: the ortho tint
                          # has the vehicle itself baked into the ground there
GROUND_SAMPLE_RING_M = 6.0   # and average the ground out to here
GROUND_SAMPLE_MIN_CELLS = 2
# ---------------------------------------------------------------------------
# -- A POSITION MARKER DRAWS NOTHING AT ALL.
#
# A full-height translucent box is wrong twice over.  A 1.5 m box stands at eye
# level and hides the ground behind it, and hundreds of them speckle the top-down
# view.  And the translucency is not guaranteed: a pyrender material built without
# alphaMode defaults to OPAQUE, so a BLEND material with alpha 0.28 renders as a
# solid box.
#
# A flat footprint patch is no better: a class colour draws a lattice over
# plazas and roads, and a patch tinted to the ground still shows, because a
# 2 m footprint frame on a textured ground is never the ground.  Each patch
# would also cost a primitive and a material whose only job is to tint ground
# that is already drawn underneath it.
#
# So a vehicle position marker draws no mesh (the thin_sign_props() treatment).
# It is a STRUCTURED RECORD of a typical position -- kind, subkind, yaw,
# footprint, extras.model_swap -- and a record has no pixels.  The node, its
# name, its tag and its whole extras block stay exactly where they are (NODE
# PARITY: the s85 check walks the layer by node name); what goes is the mesh.
# A consumer that wants to put a car there reads extras.model_swap and
# extras.placeholder_box_m.
# ---------------------------------------------------------------------------
# -- the upright sign and pole props, thinned to one per 8 m; see
# thin_sign_props for the measured spacings behind them.
SIGN_KINDS = ('banner', 'sign_board', 'traffic_sign_face', 'traffic_light',
              'utility_pole')
SIGN_SPACING_M = 8.0
# The same argument one level up.  A full-height SOLID placeholder box is a
# wall: a 1.0 x 2.0 m banner box a metre from an eye-level camera fills a third
# of the frame as one blob of the box colour, because a renderer that drops
# alphaMode draws every BLEND material opaque.  So a placeholder that reaches
# above knee height is drawn as the twelve EDGES of its box, solid bars with
# the faces left open: the size and the class colour still read, the ground
# behind does not disappear.  Face-on, a 1.0 x 2.0 m
# banner cage covers 0.35 m2 of bar against 2.0 m2 of slab, 17 % of the area.
# Short placeholders (manholes, cones, cameras) stay solid; a cage at that size
# is invisible and a 5 cm box blocks nothing.
CAGE_MIN_H_M = 0.8

# stage_knobs.OBJECT_PLACEHOLDER_HIDE in an area's area.yaml.
# None (no area set, or knob absent) = every placeholder drawn; [] = hide every
# kind with no CC0 model; a list = hide those kinds only.
def _placeholder_hide():
    v = ((((getattr(C, '_AREA_CFG', None) or {}).get('stage_knobs') or {})
          .get('OBJECT_PLACEHOLDER_HIDE')) or {}).get('value')
    if v is None or v is False:
        return None
    return set() if v is True else set(v)


_PLACEHOLDER_HIDE = _placeholder_hide()
CAGE_BAR_M = 0.06


def load_assets():
    man = json.load(open(AD + '/manifest.json'))
    z = np.load(AD + '/object_assets.npz')
    A = {}
    for cls in man['classes']:
        pos = z['pos_' + cls]
        ext = pos.max(0) - pos.min(0)
        A[cls] = dict(pos=pos, nrm=z['nrm_' + cls], uv=z['uv_' + cls],
                      col=z['col_' + cls], ext=ext,
                      run=('x' if ext[0] >= ext[2] else 'z'), m=man['assets'][cls])
    return A, man


def y_quat(deg):
    t = math.radians(deg) / 2.0
    return [0.0, math.sin(t), 0.0, math.cos(t)]


def snap_colour(rgb):
    if not rgb:
        return None
    c = np.array(rgb, 'f4')
    d = ((np.array(PALETTE, 'f4') - c) ** 2).sum(1)
    return int(np.argmin(d))


def tinted(col, pal):
    """Per-vertex COLOR_0 blended toward a palette entry, weighted by how light
    the baked vertex already is: dark parts (tyres, windows) keep their own."""
    c = col.astype('f4').copy()
    w = (c[:, :3].mean(1) / 255.0)[:, None] ** 1.5
    tgt = np.array(PALETTE[pal], 'f4')[None, :]
    c[:, :3] = c[:, :3] * (1 - BLEND * w) + tgt * (BLEND * w)
    out = col.copy()
    out[:, :3] = np.clip(c[:, :3], 0, 255).astype('u1')
    return out


def _is_lens(rgb):
    """Which vertices of a signal head are a real lens.

    Green if green is the top channel and bright, red if red is the top channel
    and green is under half of it.  The Kenney head's amber housing --
    (255,160,57), (255,204,93), (255,207,96), (255,156,53), 21 % of the model's
    vertices -- fails both tests, which is the point: it is kit paint."""
    mx = rgb.max(1)
    green = (rgb[:, 1] >= mx - 1) & (rgb[:, 1] > 90)
    red = (rgb[:, 0] >= mx - 1) & (rgb[:, 1] < 0.5 * rgb[:, 0])
    return green | red


def _srgb_to_linear(rgb):
    """An sRGB colour as the LINEAR value a glTF COLOR_0 has to carry to be
    DISPLAYED as that colour.

    Measured on utility poles cut out into their own GLB and rendered alone in
    pyrender, so no tree or neighbouring placeholder could contaminate the pixels:

        COLOR_0 = (120, 95, 70)  -- the class target written as-is
                 -> rendered (156.6, 136.5, 127.8)
        COLOR_0 = ( 48,  29, 16) -- the same target, sRGB-decoded
                 -> rendered (105.5,  79.6,  68.5)

    COLOR_0 is LINEAR by the glTF spec and the
    renderer gamma-encodes it on the way out, so a target written straight into
    COLOR_0 comes back roughly its own square root: brighter, and with its hue
    flattened, because the gamma curve compresses the gap between the channels.
    A pole asked for weathered wood renders as a near-neutral (155, 149, 133)
    without decoding, and no change to SHADE_SPAN can fix that -- the clamp
    bounds the luminance, and the luminance is not the problem.

    Decoding first makes the target land: (105.5, 79.6, 68.5) is a brown with
    the right channel spread.  It reads ~12 % dark against the nominal (120, 95,
    70) because the render's own lights are under 1.0, not because the colour is
    wrong, and a spec-conformant glTF viewer shows the asked-for colour
    exactly.
    """
    c = np.asarray(rgb, 'f4') / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin * 255.0


def _shade(L):
    """Relative shading for a repaint, bounded by SHADE_SPAN.

    Normalising on the median keeps the modal vertex -- the flat trunk or mast
    that most of the rendered pixels come from -- at exactly the colour that was
    asked for.  The clamp bounds it: without it one bright kit texel carries
    the class colour 85 % of the way to white (a pole asked for (120, 95, 70)
    gets vertices at (233, 145, 103))."""
    ref = max(float(np.median(L)), 1e-6)
    return np.clip(L / ref, 1.0 - SHADE_SPAN, 1.0 + SHADE_SPAN)


def repaint(col, pal, body='saturated', linear=False, rgb=None):
    """The painted panels of a vehicle hull take the measured colour outright.

    The Kenney kits bake one arbitrary body colour per model -- car_suv and
    truck are mint green (67,172,124), the taxi orange, the sedan red, the van
    blue -- and pulling those hues only 18 % of the way toward the measured
    colour (tinted()) leaves a mint-green kit body mint whatever the panoramas
    say.  Here the painted vertices are repainted outright and keep only their
    own relative shading.

    A painted vertex is one whose kit colour is saturated, chroma/max >= 0.35.
    Measured on the baked assets that separates the body panels (0.53-0.72:
    the green 0.61, the taxi 0.72, the sedan red 0.70, the van blue 0.53) from
    the neutral tyres, trim and shadow (under 0.1) and from the faintly blue
    window glass (0.20), which keeps its own colour.

    `body='kit_neutral'` inverts the test, for the street-furniture kits, where
    the split runs the other way: the structure is the desaturated violet mass
    (chroma 0.20) and the saturated vertices are the things that really are
    coloured -- a signal's lenses, the pole's brown crossarm.

    `body='all'` repaints every vertex, and `body='lens_keep'` every vertex that
    is not a signal lens; see CLASS_REPAINT for why those two exist.

    In every mode the relative shading goes through _shade(), so the modal
    repainted vertex is exactly the requested colour and none is more than
    SHADE_SPAN off it.

    `linear=True` sRGB-decodes the palette entry first, so the RENDERED colour
    is the one that was asked for rather than its gamma-brightened square root;
    see _srgb_to_linear() for the measurement.  It is on for the class-table
    repaints (the street furniture, whose colour is a stated intent) and off for
    a measured vehicle hull, whose palette entry came from a photograph and
    already carries whatever encoding the panorama had.

    `rgb` paints that exact sRGB colour instead of PALETTE[pal];
    the street poles use it for their measured STREET_POLE_RGB."""
    want = PALETTE[pal] if rgb is None else tuple(rgb)
    rgb = col[:, :3].astype('f4')
    mx = rgb.max(1); mn = rgb.min(1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    if body == 'kit_neutral':
        body = sat < BODY_SAT
    elif body == 'all':
        body = np.ones(len(col), bool)
    elif body == 'lens_keep':
        body = ~_is_lens(rgb)
    else:
        body = sat >= BODY_SAT
    out = col.copy()
    if not body.any():
        return out
    tgt = np.array(want, 'f4')[None, :]
    if linear:
        tgt = _srgb_to_linear(want)[None, :]
    out[body, :3] = np.clip(tgt * _shade(rgb[body].mean(1))[:, None],
                            0, 255).astype('u1')
    return out


def pal_of(o, asset):
    """-> (palette entry, from_class_table), or (None, False) for the asset's
    own baked colour.

    A measurement wins.  With no measurement the row takes its CLASS default if
    the table has one, because the kit's own paint is an arbitrary palette slot
    and not evidence about this area; a vehicle hull with neither falls back to
    the neutral grey."""
    if o.get('_street'):
        # nearest NEUTRAL palette entry: plain RGB distance snaps the measured
        # (57, 57, 60) to the green (45, 90, 70), 1333 vs 1452 for (35, 35, 38)
        c = np.array(o['_street']['rgb'], 'f4')
        neu = [i for i, q in enumerate(PALETTE) if max(q) - min(q) <= 10]
        return min(neu, key=lambda i: float(((np.array(PALETTE[i], 'f4') - c) ** 2).sum())), True
    pal = snap_colour(o.get('rgb')) if o.get('rgb') else None
    if pal is not None:
        return pal, False
    d = CLASS_RGB.get(o['kind'])
    if d is not None:
        return snap_colour(d), True
    if asset in REPAINT_ASSETS:
        return NEUTRAL_PAL, False
    return None, False


def paint(col, pal, asset, from_class=False, kind=None, rgb=None):
    if from_class:
        return repaint(col, pal, body=CLASS_REPAINT.get(kind, 'kit_neutral'),
                       linear=True, rgb=rgb)
    return repaint(col, pal) if asset in REPAINT_ASSETS else tinted(col, pal)


def unit_box():
    """A unit box: x and z in [-0.5, 0.5], y in [0, 1], triangle soup with
    flat normals, so one node scale carries the placeholder's measured size."""
    v = np.array([[-.5, 0, -.5], [.5, 0, -.5], [.5, 1, -.5], [-.5, 1, -.5],
                  [-.5, 0, .5], [.5, 0, .5], [.5, 1, .5], [-.5, 1, .5]], 'f4')
    q = [(0, 1, 2, 3), (5, 4, 7, 6), (4, 0, 3, 7), (1, 5, 6, 2),
         (3, 2, 6, 7), (4, 5, 1, 0)]
    P = []
    for a_, b_, c_, d_ in q:
        P += [v[a_], v[b_], v[c_], v[a_], v[c_], v[d_]]
    pos = np.array(P, 'f4')
    tri = pos.reshape(-1, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    nrm = np.repeat(n, 3, 0).astype('f4')
    return pos, nrm, np.zeros((len(pos), 2), 'f4')



def unit_cage(fx, fy, fz):
    """The twelve EDGES of the unit box as solid bars, the six faces open.

    Same unit space as unit_box (x, z in [-0.5, 0.5], y in [0, 1]), so one node
    scale still carries the placeholder's measured size.  fx / fy / fz are the
    bar thicknesses as a FRACTION of each axis, so a bar comes out a constant
    width in metres whatever the box."""
    bp, bn, _ = unit_box()
    fx = float(np.clip(fx, 0.02, 0.45))
    fy = float(np.clip(fy, 0.02, 0.45))
    fz = float(np.clip(fz, 0.02, 0.45))
    bars = []
    for cx in (-0.5 + fx / 2, 0.5 - fx / 2):            # 4 uprights
        for cz in (-0.5 + fz / 2, 0.5 - fz / 2):
            bars.append((cx, 0.5, cz, fx, 1.0, fz))
    for cy in (fy / 2, 1.0 - fy / 2):                   # 8 rails, top and bottom
        for cz in (-0.5 + fz / 2, 0.5 - fz / 2):
            bars.append((0.0, cy, cz, 1.0, fy, fz))
        for cx in (-0.5 + fx / 2, 0.5 - fx / 2):
            bars.append((cx, cy, 0.0, fx, fy, 1.0))
    P, N = [], []
    for cx, cy, cz, sx, sy, sz in bars:
        q = bp.copy()
        q[:, 0] = q[:, 0] * sx + cx
        q[:, 1] = (q[:, 1] - 0.5) * sy + cy
        q[:, 2] = q[:, 2] * sz + cz
        P.append(q); N.append(bn.copy())
    pos = np.concatenate(P).astype('f4')
    nrm = np.concatenate(N).astype('f4')
    return pos, nrm, np.zeros((len(pos), 2), 'f4')


def cage_fracs(bx):
    """Bar thicknesses as a fraction of each box axis, so CAGE_BAR_M of bar is
    the same width on a shelter as on a planter.  bx is [length, width, height]
    and the node's axes are (length, height, width)."""
    return (CAGE_BAR_M / max(float(bx[0]), 1e-6),
            CAGE_BAR_M / max(float(bx[2]), 1e-6),
            CAGE_BAR_M / max(float(bx[1]), 1e-6))


def raw_box(o):
    """The placeholder box a row asks for, [length, width, height] in metres.

    The class table's box_m is the class's ordinary real-world size and stays
    the cap: a measured footprint is NOT substituted here, because the sensor
    footprints are blob extents and substituting them makes the placeholders
    worse, not better: unknown_cluster blobs typically measure a median of
    about 26 m3 against the 1.2 m3 class box, and a playground_item blob can
    measure over 500 m3 against a 22.5 m3 class box."""
    bx = list(o.get('box_m') or K.SPEC.get(o['kind'], {}).get('box_m')
              or [1.0, 1.0, 1.0])
    bx = [float(v) for v in (list(bx) + [1.0, 1.0, 1.0])[:3]]
    return bx


def box_size_stats(by_kind):
    """kind -> n, median / p90 / max volume and longest side, for the report."""
    out = {}
    for k, B in sorted(by_kind.items()):
        A = np.array(B, 'f8')
        V = A.prod(1); M = A.max(1)
        out[k] = dict(n=len(A),
                      median_box_m=[round(float(v), 3) for v in np.median(A, 0)],
                      median_volume_m3=round(float(np.median(V)), 3),
                      p90_volume_m3=round(float(np.percentile(V, 90)), 3),
                      max_volume_m3=round(float(V.max()), 3),
                      median_max_dim_m=round(float(np.median(M)), 3),
                      p90_max_dim_m=round(float(np.percentile(M, 90)), 3),
                      max_max_dim_m=round(float(M.max()), 3))
    return out


def clamp_measured_footprints(objs):
    """Bound each measured footprint by its class's median measured footprint.

    -> the ids that were clamped.  Mutates footprint_m in place and records the
    original on the row as '_footprint_clamped_from_m', so instance_scale(),
    pick_asset() (the boat length -> hull table) and the node extras all see the
    same, bounded, number.

    A measured footprint is a detection-cluster extent, and a cluster that
    swallowed its neighbours reads as one enormous object.  Rows with
    footprint_from != 'measured' are untouched: their footprint is the class
    constant, and clamping a constant against its own median is a no-op."""
    by = collections.defaultdict(list)
    for o in objs:
        if o.get('footprint_from') == 'measured' and o.get('footprint_m'):
            by[o['kind']].append([float(v) for v in o['footprint_m'][:2]])
    med = {}
    for k, v in by.items():
        A = np.array(v, 'f8')
        med[k] = ([float(x) for x in np.median(A, 0)],
                  float(np.median(A[:, 0] * A[:, 1])),
                  float(np.median(A.max(1))))
    hit = []
    for o in objs:
        if o.get('footprint_from') != 'measured' or not o.get('footprint_m'):
            continue
        m = med.get(o['kind'])
        if m is None:
            continue
        mfp, marea, mmax = m
        fp = [float(v) for v in o['footprint_m'][:2]]
        if fp[0] * fp[1] > SIZE_CLAMP_FACTOR * max(marea, 1e-9) or \
           max(fp) > SIZE_CLAMP_FACTOR * max(mmax, 1e-9):
            o['_footprint_clamped_from_m'] = [round(v, 2) for v in fp]
            o['footprint_m'] = [round(v, 2) for v in mfp]
            hit.append(o['id'])
    return hit


class GroundColour:
    """The colour the MODEL's own ground is painted, binned to a metre grid.

    Used by box_tone_factors() here and by s82 tone_to_ground() to cap
    placeholder and prop luminance at the ground's.

    The ortho is the wrong raster for this: it is not what the ground renders
    as.  Over 40 footprints the ortho ground is (167, 166, 142) and the
    rendered ground (106, 106, 99), a 0.645 gain with an 18.2 mean residual,
    so no constant rescale of the ortho lands on the rendered ground.

    What the ground actually renders as is its own vertex colour times the
    texel the atlas hands that vertex, both read out of THE SAME GLB s85 is
    given, so the sample cannot go stale when a texture stage changes.  Over
    the same 40 footprints: rendered = 197 * sqrt(base/255), log-log slope 0.500.

    Cells are GROUND_SAMPLE_CELL_M metres of ENU, keyed (ix, iy); a caller reads
    the annulus GROUND_SAMPLE_GAP_M .. GROUND_SAMPLE_RING_M outside an object's own
    footprint, because the ortho tint has a parked vehicle itself baked into
    the ground colour directly under it.
    """

    def __init__(self, g, js, cell=GROUND_SAMPLE_CELL_M):
        import cv2
        self.cell = float(cell)
        self.cv2 = cv2
        self._tex = {}
        acc = collections.defaultdict(lambda: np.zeros(4))
        self.prims = self.verts = 0
        for name, mi, tr in self._walk(js):
            if not str(name or '').startswith(GROUND_PREFIX):
                continue
            for pr in js['meshes'][mi]['primitives']:
                at = pr['attributes']
                if 'COLOR_0' not in at or 'POSITION' not in at:
                    continue
                P = np.asarray(g.acc(at['POSITION']), 'f8') + tr
                c = g.acc(at['COLOR_0'])
                col = np.asarray(c[:, :3], 'f8')
                if c.dtype == np.uint8:
                    col /= 255.0
                elif c.dtype == np.uint16:
                    col /= 65535.0
                eff = col * self._texel(g, js, pr, at, len(P))
                ix = np.floor(P[:, 0] / self.cell).astype('i8')
                iy = np.floor(-P[:, 2] / self.cell).astype('i8')
                key = ix * 1000000 + iy
                o = np.argsort(key, kind='stable')
                k, e = key[o], eff[o]
                u, st = np.unique(k, return_index=True)
                sums = np.add.reduceat(e, st)
                cnts = np.diff(np.append(st, len(k)))
                for kk, s_, n_ in zip(u.tolist(), sums, cnts.tolist()):
                    a = acc[kk]
                    a[:3] += s_
                    a[3] += n_
                self.prims += 1
                self.verts += len(P)
        self._tex.clear()
        if acc:
            self.key = np.array(sorted(acc.keys()), 'i8')
            v = np.array([acc[int(k)] for k in self.key.tolist()])
            self.col = v[:, :3] / np.maximum(v[:, 3:4], 1.0)
        else:
            self.key = np.zeros(0, 'i8')
            self.col = np.zeros((0, 3))

    @staticmethod
    def _walk(js):
        out = []

        def rec(ni, tr):
            n = js['nodes'][ni]
            t = np.array(n.get('translation', [0.0, 0.0, 0.0]), 'f8') + tr
            if 'mesh' in n:
                out.append((n.get('name', ''), n['mesh'], t))
            for c in n.get('children', []):
                rec(c, t)
        for s in js['scenes'][0].get('nodes', []):
            rec(s, np.zeros(3))
        return out

    def _image(self, g, js, mi):
        """(atlas as HxWx3 float RGB, baseColorFactor rgb) for a material."""
        if mi in self._tex:
            return self._tex[mi]
        m = js['materials'][mi]
        pbr = m.get('pbrMetallicRoughness', {})
        f = np.array((pbr.get('baseColorFactor') or [1, 1, 1, 1])[:3], 'f8')
        t = pbr.get('baseColorTexture')
        a = None
        if t is not None:
            im = js['images'][js['textures'][t['index']]['source']]
            if 'bufferView' in im:
                b = np.frombuffer(g.bv_bytes(im['bufferView']), 'u1')
                a = self.cv2.imdecode(b, self.cv2.IMREAD_COLOR)
            if a is not None:
                if max(a.shape[:2]) > 1024:
                    s = 1024.0 / max(a.shape[:2])
                    a = self.cv2.resize(a, (max(1, int(a.shape[1] * s)),
                                            max(1, int(a.shape[0] * s))),
                                        interpolation=self.cv2.INTER_AREA)
                a = a[:, :, ::-1].astype('f4')
        self._tex[mi] = (a, f)
        return self._tex[mi]

    def _texel(self, g, js, pr, at, n):
        if 'material' not in pr:
            return np.full((n, 3), 255.0)
        a, f = self._image(g, js, pr['material'])
        if a is None:
            return np.tile(f * 255.0, (n, 1))
        if 'TEXCOORD_0' not in at:
            return np.tile(a.reshape(-1, 3).mean(0) * f, (n, 1))
        uv = np.asarray(g.acc(at['TEXCOORD_0']), 'f8')
        h, w = a.shape[:2]
        u = (np.mod(uv[:, 0], 1.0) * (w - 1)).astype('i4')
        v = (np.mod(uv[:, 1], 1.0) * (h - 1)).astype('i4')
        return a[v, u] * f

    def at(self, x, y, rin, rout):
        """Mean ground colour in an ENU annulus, or None if no ground there."""
        if not len(self.key):
            return None
        c = self.cell
        i0, i1 = int(math.floor((x - rout) / c)), int(math.floor((x + rout) / c))
        j0, j1 = int(math.floor((y - rout) / c)), int(math.floor((y + rout) / c))
        got = []
        for i in range(i0, i1 + 1):
            cx = (i + 0.5) * c
            for j in range(j0, j1 + 1):
                d = math.hypot(cx - x, (j + 0.5) * c - y)
                if d < rin or d > rout:
                    continue
                k = i * 1000000 + j
                p = int(np.searchsorted(self.key, k))
                if p < len(self.key) and self.key[p] == k:
                    got.append(self.col[p])
        if len(got) < GROUND_SAMPLE_MIN_CELLS:
            return None
        return np.mean(got, 0)



# -- AN AERIAL "BOAT" THAT IS GREEN IS A TREE.
# objaerial.detect_boats() takes any blob brighter than the surrounding water inside the
# water raster, and shoreline canopy overhanging the water is exactly that.
# An aerial-only boat whose ortho colour chroma (max-min)/max is at or above
# FOLIAGE_BOAT_CHROMA with blue the lowest channel (olive) is foliage over the
# water, not a hull; hull paint is close to grey.  A row with any second
# sensor is never retired.
# Retired rows are thinned exactly like thin_sign_props(): node, position and
# extras stay, no geometry, extras.thinned says why.
FOLIAGE_BOAT_CHROMA = 0.225


def foliage_boats(objs):
    """Retire aerial-only boat rows whose ortho colour is foliage. -> ids"""
    out = []
    for o in objs:
        if o.get('kind') != 'boat' or o.get('_thinned'):
            continue
        if set(o.get('sources') or ()) != {'aerial'}:
            continue
        c = o.get('rgb')
        if not c or len(c) < 3:
            continue
        r, g, b = (float(v) for v in c[:3])
        ch = (max(r, g, b) - min(r, g, b)) / max(r, g, b, 1.0)
        if ch < FOLIAGE_BOAT_CHROMA or b > min(r, g):
            continue
        o['static'] = False
        o['role'] = 'retired'
        o['_thinned'] = dict(
            rule='aerial-only boat blob with foliage colour (ortho chroma '
                 '%.3f >= %.3f, blue lowest): tree canopy over the water '
                 'raster, not a hull' % (ch, FOLIAGE_BOAT_CHROMA),
            kept=None, metres=None)
        out.append(o['id'])
    return sorted(out)


# -- the evidence gate for ortho-only boats lives where the
# rows are built, s84_object_inventory.aerial_boat_evidence() (the measured
# rules and thresholds are documented there: pier finger, shoreline, unseen by
# >= 20 clear panoramas).  s84 records it as row['evidence']['aerial_gate'];
# a row without the record (an inventory fused without it) is judged here
# on the full inventory with the footprint as fused, before the size clamp.
# foliage_boats() above stays as the second net for canopy over the water.
def aerial_gate_boats(objs, inv_path=INV):
    """Retire boat rows the s84 aerial gate rejects. -> ids"""
    todo = [o for o in objs if o.get('kind') == 'boat' and not o.get('_thinned')]
    if not todo:
        return []
    ev = {}
    if any('aerial_gate' not in (o.get('evidence') or {}) for o in todo):
        import copy
        import s84_object_inventory as OI
        rows = copy.deepcopy([o for o in json.load(open(inv_path))['objects']
                              if o.get('kind') == 'boat'])
        ev = OI.aerial_boat_evidence(rows)
    out = []
    for o in todo:
        g = (o.get('evidence') or {}).get('aerial_gate') or ev.get(o['id']) or {}
        if not g.get('retire') or set(o.get('sources') or ()) != {'aerial'}:
            continue
        o['static'] = False
        o['role'] = 'retired'
        o['_thinned'] = dict(rule='aerial-only boat rejected by the s84 evidence '
                                  'gate: ' + g['retire'], kept=None, metres=None)
        out.append(o['id'])
    return sorted(out)


def thin_sign_props(objs):
    """at most one upright sign/pole prop per SIGN_SPACING_M of path.

    MEASURED on object_inventory.json, nearest neighbour of the same kind,
    median metres and mean count within 10 m: banner 4.1 m / 1.56, utility_pole
    6.3 m / 1.33, traffic_sign_face 6.7 m / 0.75, traffic_light 9.0 m / 0.88,
    sign_board 11.5 m / 0.38.  Those are the props that fill an eye-level frame.
    A repeated detection of one real object is the cause: the aerial, photo and
    lidar passes each put a row down, and two rows 4 m apart along the same
    kerb are one sign seen twice.

    NOTHING IS DELETED.  A thinned row keeps its node, its position and its
    whole extras block; what changes is that it draws no geometry and is
    recorded as a typical position (static false, role typical_position), the
    same treatment a transient candidate already gets.

    Two rows that DISAGREE are both kept: different kind, or the same kind with
    a different subkind, is two different things standing together, not one
    thing detected twice.  Order is deterministic -- best evidence first, id as
    the tie-break -- so the same inventory always thins to the same rows.
    """
    kept = collections.defaultdict(list)
    n = 0
    order = sorted((o for o in objs if o.get('kind') in SIGN_KINDS),
                   key=lambda o: (-(o.get('confidence') or 0.0),
                                  -(o.get('n_detections') or 0), o['id']))
    for o in order:
        key = (o['kind'], o.get('subkind'))
        x, y = float(o['x']), float(o['y'])
        near = None
        for kx, ky, kid in kept[key]:
            d = math.hypot(kx - x, ky - y)
            if d <= SIGN_SPACING_M and (near is None or d < near[0]):
                near = (d, kid)
        if near is None:
            kept[key].append((x, y, o['id']))
            continue
        o['static'] = False
        o['role'] = 'typical_position'
        o['_thinned'] = dict(rule='one %s per %.0f m' % (o['kind'],
                                                         SIGN_SPACING_M),
                             kept=near[1], metres=round(near[0], 2))
        n += 1
    return n


# -- A UTILITY POLE STANDS ON A VERGE, NOT ON A PATH.
# Photo/lidar utility poles with no OSM id can land in the middle of plazas and
# paths, and s80's surface tag cannot tell a concrete plaza from a lawn because the whole
# 3dfier terrain is one layer.  The ground class that the model actually
# draws is in the GLB's ground node names (ground_terrain_3dfier_<material>,
# photo_ground_t*_<material>, path_*), so the test is run on THE INPUT GLB:
# every ground triangle is sampled on a <= HARDSCAPE_CELL_M lattice into a
# grid; a cell is hardscape if any path_* triangle or any concrete/asphalt/
# brick/paving/wood_deck/metal/gravel triangle covers it, verge if only lawn
# (or sand) does.  A pole whose disc (HARDSCAPE_POLE_R_M) touches hardscape is
# moved to the nearest verge cell within HARDSCAPE_SNAP_M whose own disc is
# clear of hardscape, buildings and water, with z re-read off s80's floor
# raster; with no such cell in reach it draws nothing and becomes a
# typical-position record, like a thinned sign.  The node records the move in
# extras.hardscape_snap (from, to, metres) and check() verifies it.
HARDSCAPE_KINDS = ('utility_pole',)
HARD_MATERIALS = ('concrete', 'asphalt', 'brick', 'paving', 'wood_deck', 'metal',
                  'gravel')
VERGE_MATERIALS = ('lawn', 'sand')
HARDSCAPE_CELL_M = 0.5
HARDSCAPE_POLE_R_M = 0.5
# -- A SNAP MAY NOT LEAVE THE POLE'S OWN PHOTO CLUSTER.
# A utility_pole photo cluster (lib/objects/objphoto.py, n_views >= 7) spreads
# 2.79 m at the median, 3.36 m at p90, 3.96 m at most, so a snap longer than
# 4.0 m is a position no detection supports.  Short snaps (under 3 m) are stable
# when the ground is re-meshed; the nearest verge of a long snap moves by metres.
# A lidar-support gate does not work here: photo utility_pole rows carry
# points=0, and most street poles have no 3DEP return 3-11 m above the floor
# within 3 m.
HARDSCAPE_SNAP_M = 4.0


class GroundKind:
    """hard / verge masks of the input GLB's drawn ground, HARDSCAPE_CELL_M."""

    def __init__(self, g, js, cell=HARDSCAPE_CELL_M):
        self.cell = c = float(cell)
        pts = {'hard': [], 'verge': []}
        for name, mi, tr in GroundColour._walk(js):
            nm = str(name or '')
            if not nm.startswith(GROUND_PREFIX):
                continue
            mat = next((m for m in HARD_MATERIALS + VERGE_MATERIALS
                        if '_' + m in nm), None)
            if '_water' in nm:
                continue
            if nm.startswith('path_'):
                kind = 'hard'
            elif mat in HARD_MATERIALS:
                kind = 'hard'
            elif mat in VERGE_MATERIALS:
                kind = 'verge'
            else:
                continue
            for pr in js['meshes'][mi]['primitives']:
                at = pr['attributes']
                if 'POSITION' not in at:
                    continue
                V = np.asarray(g.acc(at['POSITION']), 'f8') + tr
                V = np.stack([V[:, 0], -V[:, 2]], 1)
                F = (np.asarray(g.acc(pr['indices']), 'i8').reshape(-1, 3)
                     if 'indices' in pr else np.arange(len(V)).reshape(-1, 3))
                a, b, cc = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
                e = np.max(np.stack([np.hypot(*(a - b).T), np.hypot(*(b - cc).T),
                                     np.hypot(*(cc - a).T)], 1), 1)
                n = np.clip(np.ceil(e / (0.7 * c)), 1, None).astype('i8')
                for k in np.unique(n[n <= 8]):
                    # small triangles: a barycentric lattice finer than a cell
                    sel = n == k
                    ii, jj = np.meshgrid(np.arange(k + 1), np.arange(k + 1))
                    m = (ii + jj) <= k
                    u = (ii[m] / k)[None, :, None]
                    v = (jj[m] / k)[None, :, None]
                    A_, B_, C_ = a[sel][:, None], b[sel][:, None], cc[sel][:, None]
                    P = (A_ + u * (B_ - A_) + v * (C_ - A_)).reshape(-1, 2)
                    pts[kind].append(np.floor(P / c).astype('i8'))
                for t in np.nonzero(n > 8)[0]:
                    # large triangles: cell centres of the bbox inside the tri
                    T = np.stack([a[t], b[t], cc[t]])
                    lo, hi = np.floor(T.min(0) / c), np.floor(T.max(0) / c)
                    I, J = np.meshgrid(np.arange(lo[0], hi[0] + 1),
                                       np.arange(lo[1], hi[1] + 1))
                    X, Y = (I.ravel() + 0.5) * c, (J.ravel() + 0.5) * c
                    v0, v1 = T[1] - T[0], T[2] - T[0]
                    den = v0[0] * v1[1] - v1[0] * v0[1]
                    if abs(den) < 1e-12:
                        continue
                    px, py = X - T[0, 0], Y - T[0, 1]
                    uu = (px * v1[1] - v1[0] * py) / den
                    vv = (v0[0] * py - px * v0[1]) / den
                    m = (uu >= -0.05) & (vv >= -0.05) & (uu + vv <= 1.05)
                    pts[kind].append(np.stack([I.ravel()[m], J.ravel()[m]],
                                              1).astype('i8'))
        self.grid = {}
        allp = [q for L in pts.values() for q in L]
        if allp:
            q = np.concatenate(allp)
            self.o = q.min(0) - 2
            self.shape = tuple(q.max(0) - self.o + 3)
        for kind in ('hard', 'verge'):
            G = np.zeros(self.shape if allp else (1, 1), bool)
            if pts[kind]:
                q = np.concatenate(pts[kind]) - self.o
                G[q[:, 0], q[:, 1]] = True
            self.grid[kind] = G
        self.n = {k: int(G.sum()) for k, G in self.grid.items()}

    def _get(self, kind, k):
        G = self.grid[kind]
        i, j = k[0] - self.o[0], k[1] - self.o[1]
        return 0 <= i < G.shape[0] and 0 <= j < G.shape[1] and bool(G[i, j])

    def _disc(self, x, y, r):
        c = self.cell
        i0, j0 = int(math.floor(x / c)), int(math.floor(y / c))
        n = int(math.ceil(r / c))
        for di in range(-n, n + 1):
            for dj in range(-n, n + 1):
                if math.hypot(di * c, dj * c) <= r:
                    yield (i0 + di, j0 + dj)

    def hard_at(self, x, y, r):
        return any(self._get('hard', k) for k in self._disc(x, y, r))

    def verge_ok(self, x, y, r):
        return all(self._get('verge', k) and not self._get('hard', k)
                   for k in self._disc(x, y, r))


# -- A STREET POLE STANDS ON THE SIDEWALK; LEAVE IT THERE.
# poles_off_hardscape() is right for a pole in a plaza and wrong for one on a
# street kerb or beside a tram track, which is supposed to stand on concrete.
# Distance to the street (s80 street_lines(): OSM street carriageway EDGE =
# centreline - lanes x 3.2 m / 2, and any tram track, railway=tram,
# electrified=contact_line) separates the two: kerb poles sit within a few
# metres of a street edge or a track and interior poles far from both, so
# STREET_POLE_M = 5 sits in that gap.  A row whose recorded position is not at
# least STREET_POLE_KERB_M outside the estimated carriageway, or is ON a track
# (< STREET_POLE_TRACK_M), keeps the treatment: a pole at edge -0.0 m stands in
# the road.  An exempt row keeps its measured position (neither moved to a
# verge nor retired) and is painted STREET_POLE_RGB with the class repaint
# (e.g. a pole near a track but several metres from a street edge is exempt;
# the hardscape move would otherwise put it on a lawn cell).
# STREET_POLE_RGB is measured: the median of the pixels inside Mapillary
# object--support--utility-pole polygons that contain the projected pole axis
# (1-4 m above its base) in high/usable panoramas 5-30 m away is a neutral dark
# grey (57, 57, 60), not creosote brown.  pal_of() snaps it to the nearest
# NEUTRAL palette entry, (35, 35, 38), for the mesh key and name only: the
# paint is STREET_POLE_RGB itself (street_rgb()), because painting the snapped
# (35, 35, 38) puts COLOR_0 at linear 4/255 and the poles render near-black.
STREET_POLE_M = 5.0
STREET_POLE_KERB_M = 0.5
STREET_POLE_TRACK_M = 1.0
STREET_POLE_RGB = (57, 57, 60)


def street_rgb(o):
    """The exact measured paint of a street-kept pole, or None."""
    return tuple(int(v) for v in o['_street']['rgb']) if o.get('_street') else None


def street_poles(objs):
    """Tag HARDSCAPE_KINDS rows that stand at a street edge or a tram track
    (exempt from the hardscape move, street-pole colour). -> report"""
    (SA, SB, SH), (TA, TB) = S.street_lines()
    rep = dict(rule='%s within %.0f m of a street edge or a tram track, '
                    'at least %.1f m outside the carriageway and not on the track '
                    '(< %.1f m): kept where measured'
                    % ('/'.join(HARDSCAPE_KINDS), STREET_POLE_M,
                       STREET_POLE_KERB_M, STREET_POLE_TRACK_M),
               kept=[], not_kept=[])
    for o in objs:
        if o.get('kind') not in HARDSCAPE_KINDS:
            continue
        x, y = float(o['x']), float(o['y'])
        de = (float((S._seg_dist(SA, SB, x, y) - SH).min())
              if len(SA) else 9e9)
        dt = float(S._seg_dist(TA, TB, x, y).min()) if len(TA) else 9e9
        if min(de, dt) > STREET_POLE_M:
            continue
        row = [o['id'], round(de, 1), round(dt, 1)]
        if de < STREET_POLE_KERB_M or dt < STREET_POLE_TRACK_M:
            rep['not_kept'].append(row)
            continue
        o['_street'] = dict(rule='street pole: %.1f m from a street edge, %.1f m '
                                 'from a tram track; kept where measured'
                                 % (de, dt),
                            rgb=list(STREET_POLE_RGB))
        rep['kept'].append(row)
    return rep


def poles_off_hardscape(objs, src, js):
    """Move / retire HARDSCAPE_KINDS rows standing on hardscape. -> report."""
    t0 = time.time()
    gk = GroundKind(src, js)
    rep = dict(hard_cells=gk.n['hard'], verge_cells=gk.n['verge'],
               checked=0, on_hardscape=0, snapped=0, retired=0, snap_m=[])
    if not gk.n['hard']:
        rep['skipped'] = 'no classified ground in the input'
        return rep
    surf = S.Surface()
    c, r = gk.cell, HARDSCAPE_POLE_R_M
    n = int(math.ceil(HARDSCAPE_SNAP_M / c))
    ring = sorted(((math.hypot(di * c, dj * c), di, dj)
                   for di in range(-n, n + 1) for dj in range(-n, n + 1)
                   if 0 < math.hypot(di * c, dj * c) <= HARDSCAPE_SNAP_M))
    for o in objs:
        if o.get('kind') not in HARDSCAPE_KINDS or o.get('_thinned'):
            continue
        if o.get('_street'):
            rep['street_kept'] = rep.get('street_kept', 0) + 1
            continue
        rep['checked'] += 1
        x, y = float(o['x']), float(o['y'])
        if not gk.hard_at(x, y, r):
            continue
        rep['on_hardscape'] += 1
        to = None
        for d, di, dj in ring:
            # candidate at a cell centre on the lattice around the pole
            cx = (math.floor(x / c) + di + 0.5) * c
            cy = (math.floor(y / c) + dj + 0.5) * c
            if not gk.verge_ok(cx, cy, r) or surf.in_building(cx, cy):
                continue
            z, sn = surf.at(cx, cy, r=0.5)
            if sn != 'terrain' or not np.isfinite(z):
                continue
            to = (cx, cy, round(float(z), 3), math.hypot(cx - x, cy - y))
            break
        if to is None:
            o['static'] = False
            o['role'] = 'typical_position'
            o['_thinned'] = dict(rule='%s on hardscape, no verge within %.0f m'
                                 % (o['kind'], HARDSCAPE_SNAP_M),
                                 kept=None, metres=None)
            rep['retired'] += 1
            continue
        o['_snap'] = dict(rule='%s moved off hardscape to the nearest verge'
                          % o['kind'],
                          **{'from': [round(x, 4), round(y, 4), o['z']]},
                          to=[round(to[0], 4), round(to[1], 4), to[2]],
                          metres=round(to[3], 2))
        o['x'], o['y'], o['z'] = to[0], to[1], to[2]
        rep['snapped'] += 1
        rep['snap_m'].append(round(to[3], 2))
    rep['seconds'] = round(time.time() - t0, 1)
    return rep


def box_style(o):
    rgb = BOX_COLOUR.get(o['kind'], BOX_RGB_DEFAULT)
    f = BOX_TONE.get(o['kind'])
    if f is not None:
        rgb = tuple(int(round(c * f)) for c in rgb)
    if o['kind'] in FAINT_KINDS:
        return rgb, FAINT_ALPHA
    a_ = BOX_ALPHA_TRANSIENT if (o['kind'] in TRANSIENT_KINDS or
                                 not o.get('static')) else BOX_ALPHA_FIXED
    return rgb, a_


def pick_asset(o):
    kind = o['kind']
    vs = VARIANTS.get(kind)
    if not vs:
        return None
    if kind == 'boat':
        L = float(o['footprint_m'][0])
        return next(a for lim, a in BOAT_BY_LENGTH if L <= lim)
    if len(vs) == 1:
        return vs[0]
    return vs[int(hash_unit(o['id'], 'variant') * len(vs)) % len(vs)]


def instance_scale(o, a):
    ext = a['ext']
    long_ext = float(max(ext[0], ext[2]))
    short_ext = float(min(ext[0], ext[2]))
    j = 1.0 + JITTER * (hash_unit(o['id'], 'size') * 2 - 1)
    if o['kind'] in FIT_FOOTPRINT and o.get('footprint_from') == 'measured':
        L, W = float(o['footprint_m'][0]), float(o['footprint_m'][1])
        k = float(np.clip(L / max(long_ext, 1e-6), 0.55, 2.5))
        kw = float(np.clip(W / max(short_ext, 1e-6), 0.6, 1.8))
        s = [k, (k + kw) / 2.0, kw] if a['run'] == 'x' else [kw, (k + kw) / 2.0, k]
        return [v * j for v in s]
    if o['kind'] in FIT_HEIGHT and o.get('height_from') == 'lidar':
        k = float(np.clip(float(o['height_m']) / max(float(ext[1]), 1e-6), 0.5, 2.0))
        return [k * j] * 3
    return [j, j, j]


def node_rotation(a, yaw_enu_deg):
    """ENU bearing -> Y rotation, the s82 rule: an asset that runs along +Z
    needs 90 + bearing, one that runs along +X needs the bearing."""
    return (yaw_enu_deg if a['run'] == 'x' else 90.0 + yaw_enu_deg)


def extras_of(o, asset, a):
    return dict(kind=o['kind'], subkind=o.get('subkind'), asset=asset,
                sources=o['sources'], evidence=o.get('evidence'),
                confidence=o.get('confidence'), static=o['static'],
                role=o['role'], surface=o.get('surface'),
                yaw_deg=o.get('yaw_deg'), yaw_from=o.get('yaw_from'),
                footprint_m=o.get('footprint_m'),
                footprint_from=o.get('footprint_from'),
                footprint_clamped_from_m=o.get('_footprint_clamped_from_m'),
                height_m=o.get('height_m'), height_from=o.get('height_from'),
                colour_hex=o.get('colour_hex'), colour_from=o.get('colour_from'),
                osm_id=o.get('osm_id'), osm_type=o.get('osm_type'),
                osm_tag=o.get('osm_tag'), name=o.get('name'),
                placeholder_box_m=(None if a else o.get('box_m')),
                typical_position=(o['kind'] in TRANSIENT_KINDS),
                model_swap=dict(kind=o['kind'], subkind=o.get('subkind'),
                               size_m=(o.get('footprint_m', [None, None]) +
                                       [o.get('height_m')]),
                               transient=not o['static']),
                licence=(a['m'].get('licence') if a else None),
                asset_url=(a['m'].get('page_url') if a else None))


def placeable(o, place_traffic, place_transient=False):
    """Which rows get geometry.

    A vehicle is a moment, not a map feature: the panoramas caught it where it
    happened to be on one pass, and a parked car is no more permanent than a
    moving one.  So every transient KIND -- cars, trucks, buses, motorcycles,
    bicycles -- stays a tagged placeholder node with no mesh whatever its
    subkind says, and a downstream tool treats it as a typical position that
    may or may not be occupied.  A moored boat stays at its berth and keeps its
    model.  `--place-transient` puts the vehicle models back.

    Everything else: scenery and anything parked or moored
    gets its model, traffic does not unless --place-traffic says so."""
    if o['kind'] in TRANSIENT_KINDS and not place_transient:
        return False
    if o['static'] or o.get('subkind') in ('parked', 'moored'):
        return True
    return bool(place_traffic)


CROP_MARGIN = 12.0       # m of slop around the ground crop


def ground_bbox_enu(js):
    """ENU xy bbox of LAYER_ground.  A cropped build (s22 --core) covers only
    the CORE box around the area origin, so placing the full inventory into it scatters cars and
    boats outside the terrain and blows up the model's stated extents."""
    byname = {n.get('name'): i for i, n in enumerate(js['nodes'])}
    if 'LAYER_ground' not in byname:
        return None
    lo = np.array([np.inf, np.inf]); hi = -lo
    for ci in js['nodes'][byname['LAYER_ground']].get('children', []):
        nd = js['nodes'][ci]
        if 'mesh' not in nd:
            continue
        for pr in js['meshes'][nd['mesh']]['primitives']:
            a = js['accessors'][pr['attributes']['POSITION']]
            if not a.get('min') or not a.get('max'):
                continue
            for v in (a['min'], a['max']):
                # model convention: (x_enu, z_up, -y_enu)
                e = np.array([v[0], -v[2]], 'f8')
                lo = np.minimum(lo, e); hi = np.maximum(hi, e)
    return None if not np.isfinite(lo).all() else (lo - CROP_MARGIN, hi + CROP_MARGIN)


def in_crop(o, bb):
    if bb is None:
        return True
    return (bb[0][0] <= float(o['x']) <= bb[1][0] and
            bb[0][1] <= float(o['y']) <= bb[1][1])


def do_place(inp, outp, inv_path=INV, place_traffic=False, place_transient=False):
    t0 = time.time()
    doc = json.load(open(inv_path))
    objs = doc['objects']
    A, man = load_assets()
    src = GLB(inp)
    js = json.loads(json.dumps(src.js))
    rep = dict(input=inp, output=outp, input_bytes=os.path.getsize(inp),
               triangles_before=instanced_triangles(js), objects=len(objs))
    old = [i for i in js['scenes'][0]['nodes']
           if js['nodes'][i].get('name') == LAYER]
    if old:
        js['scenes'][0]['nodes'] = [i for i in js['scenes'][0]['nodes'] if i not in old]
        rep['replaced_existing_layer'] = True
    bb = ground_bbox_enu(js)
    n0 = len(objs)
    objs = [o for o in objs if in_crop(o, bb)]
    rep['ground_crop_enu'] = None if bb is None else [
        [round(float(v), 1) for v in bb[0]], [round(float(v), 1) for v in bb[1]]]
    rep['skipped_outside_crop'] = n0 - len(objs)
    ap = Appender(js)
    for k in ('images', 'textures', 'samplers', 'materials'):
        js.setdefault(k, [])
    bvi = ap.bv(open(AD + '/object_palette.jpg', 'rb').read())
    js['images'].append({"bufferView": bvi, "mimeType": "image/jpeg",
                         "name": "object_palette"})
    js['samplers'].append({"magFilter": 9729, "minFilter": 9987,
                           "wrapS": 10497, "wrapT": 10497})
    js['textures'].append({"sampler": len(js['samplers']) - 1,
                           "source": len(js['images']) - 1})
    js['materials'].append({
        "name": "object_low_poly",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": len(js['textures']) - 1},
                                 "metallicFactor": 0.0, "roughnessFactor": 0.9},
        "doubleSided": True,
        "extras": dict(licence=man['texture']['licence'],
                       texture_source=man['texture']['origin'],
                       asset_licences={c: A[c]['m'].get('licence') for c in A},
                       asset_urls={c: A[c]['m'].get('page_url') for c in A},
                       material_class="scene_objects")})
    mati = len(js['materials']) - 1
    meshes, tri_per_mesh = {}, {}
    boxmesh, boxmat, boxes, box_tris = {}, {}, 0, 0
    tri_per_box, box_inst_tris = {}, 0
    markers_no_geom = 0          # position markers that emit no geometry
    used = collections.Counter(); empties = collections.Counter()
    hidden_placeholders = collections.Counter()
    kids = []
    # class medians over the rows that will be drawn as a placeholder, so the
    # size clamp is measured against this inventory and not against a constant
    # before anything reads footprint_m: pick_asset's boat-length table and
    # instance_scale's footprint stretch both do
    fp_clamped_ids = clamp_measured_footprints(objs)
    placeholder_rows = [o for o in objs
                        if not placeable(o, place_traffic, place_transient)
                        or pick_asset(o) is None]
    BOX_TONE.clear()
    _bt, rep['placeholder_tone_to_ground'] = box_tone_factors(
        [o for o in placeholder_rows if o['kind'] not in MARKER_KINDS], src, js)
    BOX_TONE.update(_bt)
    box_sizes = collections.defaultdict(list)
    rep['aerial_gate_boats_retired'] = aerial_gate_boats(objs, inv_path)
    rep['foliage_boats_retired'] = foliage_boats(objs)
    rep['sign_props_thinned'] = thin_sign_props(objs)
    rep['street_poles'] = street_poles(objs)
    rep['poles_off_hardscape'] = poles_off_hardscape(objs, src, js)
    rep['sign_props_thinned_by_kind'] = dict(collections.Counter(
        o['kind'] for o in objs if o.get('_thinned')))
    rep['sign_prop_rule'] = ('one per %.0f m of the same kind and subkind; a '
                             'thinned row keeps its node, its extras and its '
                             'position and draws nothing'
                             % SIGN_SPACING_M)
    for o in objs:
        thinned = bool(o.get('_thinned'))
        asset = (None if thinned else
                 pick_asset(o) if placeable(o, place_traffic, place_transient)
                 else None)
        a = A[asset] if asset else None
        nd = {"name": o['id'],
              "translation": [round(float(o['x']), 4), round(float(o['z']), 4),
                              round(-float(o['y']), 4)],
              "rotation": y_quat(node_rotation(a, float(o['yaw_deg'])) if a
                                 else float(o['yaw_deg'])),
              "extras": extras_of(o, asset, a)}
        if o.get('_snap'):
            nd['extras']['hardscape_snap'] = dict(o['_snap'])
        if o.get('_street'):
            nd['extras']['street_pole'] = dict(o['_street'])
        if thinned:
            # no geometry at all: the row is a typical-position record now
            empties[o['kind'] + ':thinned'] += 1
            nd['extras']['thinned'] = dict(o['_thinned'])
            nd['extras']['placeholder_box_m'] = o.get('box_m')
            kids.append(ap.node(nd))
            continue
        elif a is None and o['kind'] in MARKER_KINDS:
            # a vehicle position marker carries data, not pixels.
            # Exactly the thin_sign_props() treatment -- the obj_* node, its
            # name, its tag and its extras stay, the mesh goes.  These rows are
            # already static false / role transient_candidate in the inventory, so
            # nothing about the record changes; it just stops being painted.
            empties[o['kind'] + ':position_marker'] += 1
            markers_no_geom += 1
            nd['extras']['placeholder_box_m'] = o.get('box_m')
            nd['extras']['position_marker'] = dict(
                geometry=None,
                rule='a typical position is a record, not a prop: no mesh, no '
                     'material; read extras.model_swap and '
                     'extras.placeholder_box_m to put a vehicle here',
                measured='a typical position is not a map feature, so nothing '
                         'is drawn')
            kids.append(ap.node(nd))
            continue
        elif a is None and _PLACEHOLDER_HIDE is not None and (
                not _PLACEHOLDER_HIDE or o['kind'] in _PLACEHOLDER_HIDE):
            # stage_knobs.OBJECT_PLACEHOLDER_HIDE (only with an area set): a row
            # with no CC0 model keeps its node, its extras and
            # extras.placeholder_box_m and draws nothing -- the position-marker
            # rule applied to the rest.  A dense downtown area can otherwise draw
            # hundreds of cages that read as debug wireframes.
            empties[o['kind'] + ':hidden'] += 1
            hidden_placeholders[o['kind']] += 1
            nd['extras']['placeholder_box_m'] = o.get('box_m')
            nd['extras']['placeholder_hidden'] = dict(
                geometry=None, rule='stage_knobs.OBJECT_PLACEHOLDER_HIDE: no CC0 '
                'model for this kind, so the row is a record, not a drawn box')
            kids.append(ap.node(nd))
            continue
        elif a is None:
            empties[o['kind'] + ('' if o['static'] else ':transient')] += 1
            bx = raw_box(o)
            box_sizes[o['kind']].append(bx)
            rgb, alpha = box_style(o)
            # a MARKER_KINDS row is caught by the no-geometry branch above and
            # never arrives here
            assert o['kind'] not in MARKER_KINDS
            cage = float(bx[2]) >= CAGE_MIN_H_M
            fr = cage_fracs(bx) if cage else None
            form = 'cage' if cage else 'box'
            bkey = (o['kind'], rgb, round(alpha, 2), form, None,
                    None if fr is None else tuple(round(v, 4) for v in fr))
            if bkey not in boxmesh:
                if bkey[:1] + (rgb, form) not in boxmat:
                    js['materials'].append({
                        "name": "placeholder_%s" % o['kind'],
                        "pbrMetallicRoughness": {
                            "baseColorFactor": [rgb[0] / 255.0, rgb[1] / 255.0,
                                                rgb[2] / 255.0, alpha],
                            "metallicFactor": 0.0, "roughnessFactor": 0.9},
                        "alphaMode": "BLEND",
                        "doubleSided": True,
                        "extras": dict(
                            material_class="placeholder_box",
                            kind=o['kind'], alpha=alpha,
                            form=("open edge cage, %.2f m bars" % CAGE_BAR_M)
                                 if cage else "full-height box",
                            licence=("no third-party asset; the marker is "
                                     "generated in s85_object_place.py"))})
                    boxmat[bkey[:1] + (rgb, form)] = len(js['materials']) - 1
                bp, bn, bu = unit_cage(*fr) if cage else unit_box()
                # COLOR_0 alpha stays 255 and the transparency lives only in the
                # material's baseColorFactor.  glTF multiplies the two, so
                # writing alpha into both would square it (0.45 -> 0.20,
                # 0.30 -> 0.09), so a compliant viewer would not show the
                # declared alpha.
                bc = np.tile(np.array(list(rgb) + [255], 'u1'), (len(bp), 1))
                boxmesh[bkey] = ap.mesh(('cage_%s' if cage else 'box_%s') % o['kind'],
                                        bp, bn, bu, bc,
                                        boxmat[bkey[:1] + (rgb, form)])
                tri_per_box[bkey] = len(bp) // 3
                box_tris += len(bp) // 3
            nd["mesh"] = boxmesh[bkey]
            nd["scale"] = [round(max(float(bx[0]), BOX_MIN_M), 4),
                           round(max(float(bx[2]), BOX_MIN_M), 4),
                           round(max(float(bx[1]), BOX_MIN_M), 4)]
            nd['extras']['placeholder_box_m'] = [round(v, 3) for v in bx]
            nd['extras']['placeholder_box_rendered'] = dict(
                rgb=list(rgb), alpha=alpha,
                form=('edge_cage' if cage else 'box'),
                height_m=round(float(bx[2]), 3),
                rim_m=(CAGE_BAR_M if cage else None),
                note=(("the twelve edges of the row's own placeholder_box_m "
                       "as %.2f m bars, faces open, so the box reads as a "
                       "volume without standing in front of the view"
                       % CAGE_BAR_M) if cage else
                      ("a class-coloured box of the row's own "
                       "placeholder_box_m; nothing is modelled here")))
            nd['extras']['licence'] = ("no third-party asset; the placeholder "
                                       "marker is generated in s85_object_place.py")
            box_inst_tris += tri_per_box[bkey]
            boxes += 1
            kids.append(ap.node(nd))
            continue
        pal, from_class = pal_of(o, asset)
        srgb = street_rgb(o)
        key = (asset, pal, from_class, o['kind'] if from_class else None, srgb)
        if key not in meshes:
            col = (paint(a['col'], pal, asset, from_class, o['kind'], rgb=srgb)
                   if pal is not None else a['col'])
            meshes[key] = ap.mesh('obj_%s_%s' % (asset, ('p%d%s%s' % (pal, 'c' if from_class
                                                                     else '',
                                                                     's' if srgb else ''))
                                                 if pal is not None else 'base'),
                                  a['pos'], a['nrm'], a['uv'], col, mati)
            tri_per_mesh[key] = len(a['pos']) // 3
        sx, sy, sz = instance_scale(o, a)
        nd["mesh"] = meshes[key]
        nd["scale"] = [round(sx, 5), round(sy, 5), round(sz, 5)]
        kids.append(ap.node(nd))
        used[asset] += 1
    root = ap.node({"name": LAYER, "children": kids, "extras": dict(
        layer_role=('every recognised object the s80 furniture inventory does not '
                    'place: one node per row of object_inventory.json.  A node '
                    'with a mesh is a CC0 model instance; a node without one is a '
                    'tagged placeholder with extras.placeholder_box_m'),
        inventory='object_inventory.json',
        counts=doc['report']['counts_by_kind'],
        static=doc['report']['static'],
        transient_candidates=doc['report']['transient_candidates'],
        frame='translation is (x_enu, z_up, -y_enu) metres, the model convention')})
    js['scenes'][0]['nodes'].append(root)
    def _mkey(o):
        asset = pick_asset(o)
        pal, fc = pal_of(o, asset)
        return (asset, pal, fc, o['kind'] if fc else None, street_rgb(o))
    rep.update(placeholder_boxes=boxes, placeholder_box_meshes=len(boxmesh),
               placeholder_box_mesh_triangles=box_tris,
               placeholder_box_instanced_triangles=box_inst_tris,
               position_markers_without_geometry=markers_no_geom,
               position_marker_rule='every %s row that gets no CC0 model keeps its '
                              'node, its name, its tag and its extras and '
                              'draws nothing' % '/'.join(MARKER_KINDS),
               placeholder_cage_form='open edge cage, %.2f m bars, on every '
                                     'placeholder %.2f m tall or taller'
                                     % (CAGE_BAR_M, CAGE_MIN_H_M),
               **({} if _PLACEHOLDER_HIDE is None else dict(placeholder_hidden=dict(hidden_placeholders))),
               measured_footprints_size_clamped=len(fp_clamped_ids),
               measured_footprints_size_clamped_examples=fp_clamped_ids[:10],
               size_clamp_rule='a MEASURED footprint over %.1fx its class '
                               'median measured footprint, by area or longest '
                               'side, is put back to that median; the '
                               'placeholder box is a class constant and is '
                               'not clamped' % SIZE_CLAMP_FACTOR,
               placeholder_box_sizes=box_size_stats(box_sizes),
               nodes_added=len(kids) + 1, instances_per_asset=dict(used),
               empty_placeholder_nodes=dict(empties),
               mesh_copies=len(meshes),
               mesh_data_triangles_added=int(sum(tri_per_mesh.values())),
               instanced_triangles_added=int(sum(
                   tri_per_mesh[_mkey(o)] for o in objs
                   if pick_asset(o) and not o.get('_thinned')
                   and placeable(o, place_traffic, place_transient))))
    rep['bytes'] = compact_write(src, js, ap.blobs, outp)
    rep['triangles_after'] = instanced_triangles(js)
    rep['seconds'] = round(time.time() - t0, 1)
    src.close()
    return rep


def do_check(path, inv_path=INV, surface=True):
    doc = json.load(open(inv_path))
    objs = {o['id']: o for o in doc['objects']}
    A, man = load_assets()
    g = GLB(path); js = g.js
    nodes = js['nodes']
    byname = {n.get('name'): n for n in nodes}
    names = [n.get('name') for n in nodes]
    layer = byname.get(LAYER)
    kids = [nodes[i] for i in (layer or {}).get('children', [])]
    dmax, missing, badrot, noextras = 0.0, [], 0, []
    for nd in kids:
        o = objs.get(nd.get('name'))
        if o is None:
            missing.append(nd.get('name')); continue
        tr = nd['translation']
        ex = nd.get('extras') or {}
        sn_ = ex.get('hardscape_snap')
        if sn_:
            # a pole moved off hardscape is checked against its
            # recorded move, and the move against HARDSCAPE_SNAP_M
            f_, t_ = sn_['from'], sn_['to']
            ok_ = (max(abs(f_[0] - o['x']), abs(f_[1] - o['y'])) < 1e-3
                   and sn_['metres'] <= HARDSCAPE_SNAP_M + 1e-6)
            o = dict(o, x=t_[0], y=t_[1], z=t_[2]) if ok_ else o
        dmax = max(dmax, abs(tr[0] - o['x']), abs(tr[1] - o['z']),
                   abs(tr[2] + o['y']))
        for k in ('kind', 'sources', 'confidence', 'static', 'role'):
            if k not in ex:
                noextras.append('%s:%s' % (nd.get('name'), k)); break
        if 'mesh' in nd and not ex.get('licence'):
            noextras.append('%s:licence' % nd.get('name'))
        q = nd.get('rotation') or [0, 0, 0, 1]
        if abs(q[0]) > 1e-9 or abs(q[2]) > 1e-9:
            badrot += 1
    miss = {'NORMAL': 0, 'TEXCOORD_0': 0, 'COLOR_0': 0}
    for me in js['meshes']:
        for pr in me['primitives']:
            att = (pr['extensions']['KHR_draco_mesh_compression']['attributes']
                   if 'KHR_draco_mesh_compression' in (pr.get('extensions') or {})
                   else pr['attributes'])
            for k in miss:
                if k not in att and k not in pr['attributes']:
                    miss[k] += 1
    nolic = [m.get('name') for m in js.get('materials', [])
             if 'baseColorTexture' in m.get('pbrMetallicRoughness', {})
             and not (m.get('extras') or {}).get('licence')]
    samp = js.get('samplers', []); tiling = set()
    for m in js.get('materials', []):
        t = m.get('pbrMetallicRoughness', {}).get('baseColorTexture')
        if t is not None:
            tiling.add(js['textures'][t['index']].get('sampler', 0))
    obj_tris = sum(tri_of_mesh(js, nd['mesh']) for nd in kids if 'mesh' in nd)
    out = dict(file=path, bytes=os.path.getsize(path),
               layer_present=bool(layer), object_nodes=len(kids),
               inventory_rows=len(objs),
               one_node_per_inventory_row=(len(kids) == len(objs)),
               nodes_with_mesh=sum(1 for nd in kids if 'mesh' in nd),
               placeholder_nodes=sum(1 for nd in kids if 'mesh' not in nd),
               node_names_unique=(len(names) == len(set(names))),
               max_placement_delta_m=round(float(dmax), 6),
               unknown_nodes=missing[:10], nodes_missing_extras=noextras[:10],
               nodes_with_non_yaw_rotation=badrot,
               primitives_missing_attributes=miss,
               textured_materials_without_licence=nolic,
               texture_sampler_wrapS=[samp[i].get('wrapS') for i in sorted(tiling)
                                      if i < len(samp)],
               object_instanced_triangles=obj_tris,
               total_instanced_triangles=instanced_triangles(js))
    g.close()
    if surface:
        surf = S.Surface()
        inb, inw, off = [], [], []
        for o in doc['objects']:
            if surf.in_building(o['x'], o['y']):
                inb.append(o['id'])
            z, sn = surf.at(o['x'], o['y'], r=0.5)
            if sn == 'water' and not K.SPEC[o['kind']]['water_ok']:
                inw.append(o['id'])
            if not np.isfinite(z) or abs(z - o['z']) > 0.05:
                off.append(o['id'])
        out.update(n_inside_building=len(inb), n_in_water=len(inw),
                   n_off_floor=len(off), examples_off_floor=off[:10])
    out['pass'] = bool(out['layer_present'] and out['one_node_per_inventory_row']
                       and out['node_names_unique']
                       and out['max_placement_delta_m'] < 1e-3
                       and not out['unknown_nodes']
                       and not out['nodes_missing_extras']
                       and out['nodes_with_non_yaw_rotation'] == 0
                       and all(v == 0 for v in miss.values()) and not nolic
                       and all(w == 10497 for w in out['texture_sampler_wrapS'])
                       and out.get('n_inside_building', 0) == 0
                       and out.get('n_in_water', 0) == 0
                       and out.get('n_off_floor', 0) == 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['place', 'check'])
    ap.add_argument('src', nargs='?'); ap.add_argument('dst', nargs='?')
    ap.add_argument('--inventory', default=INV)
    ap.add_argument('--draco', action='store_true')
    ap.add_argument('--place-traffic', action='store_true',
                    help='also give a model to vehicles caught in traffic; by '
                         'default those are marker nodes with no geometry')
    ap.add_argument('--place-transient', action='store_true',
                    help='also give models to the transient kinds (%s); by '
                         'default every one of those is a tagged placeholder '
                         'node with no geometry' % ', '.join(TRANSIENT_KINDS))
    ap.add_argument('--report', default=None)
    ap.add_argument('--scratch', default=SCRATCH)
    a = ap.parse_args()
    if a.mode == 'check':
        r = do_check(a.src, a.inventory)
        print(json.dumps(r, indent=1))
        if a.report:
            json.dump(r, open(a.report, 'w'), indent=1)
        sys.exit(0 if r['pass'] else 2)
    os.makedirs(a.scratch, exist_ok=True)
    if a.draco:
        base = os.path.basename(a.dst).replace('.glb', '')
        pin = '%s/%s_in_plain.glb' % (a.scratch, base)
        pout = '%s/%s_out_plain.glb' % (a.scratch, base)
        gt(['copy', a.src, pin])
        r = do_place(pin, pout, a.inventory, a.place_traffic, a.place_transient)
        gt(['draco', pout, a.dst])
        r['draco'] = dict(input_bytes=os.path.getsize(a.src),
                          output_bytes=os.path.getsize(a.dst))
        r['bytes'] = os.path.getsize(a.dst)
        for f in (pin, pout):
            if os.path.exists(f):
                os.unlink(f)
    else:
        r = do_place(a.src, a.dst, a.inventory, a.place_traffic, a.place_transient)
    print(json.dumps(r, indent=1))
    if a.report:
        json.dump(r, open(a.report, 'w'), indent=1)


if __name__ == '__main__':
    main()
