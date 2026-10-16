"""Stage 22 -- composite the model: tileable CC0 materials tinted by
the projected photo colour.

Painting every surface with the panorama pixels themselves gives at best 2.2 cm
per texel where they reach at all, and the ground comes out a blur with
passers-by's shadows baked into it.

Here each face gets:

  * a clean, seamless CC0 material for its class (stage 21), projected in world
    space at a fixed metres-per-tile, so the detail density is set by the
    material and not by how close a camera happened to get; and
  * a per-vertex tint (glTF COLOR_0, which multiplies the base colour texture)
    carrying the delighted multi-view colour stage 20 measured, low-pass
    filtered to the scale the panorama poses can actually support.

Sharp boundaries -- lawn to path, path to deck -- come from the material
classes, which are crisp.  Smooth variation inside a class comes from the tint.
That split is what makes it read as a modelled scene rather than a photograph
draped over a mesh.
"""
import os, sys, json, math, time, argparse, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import opencity_paths as OCP                    # noqa: E402  host roots, env-overridable
import numpy as np
import common as C
import texlib as T
import memguard
from glblib import StreamGLBWriter
from s21_material import MATERIALS, MI

PJ = T.SCR + '/project'
MT = T.SCR + '/material'
MT_ORTHO = T.SCR + '/material_ortho'
ORTHO = T.SCR + '/ortho'
CC0 = T.SCR + '/cc0'
TEXOUT = T.SCR + '/prepared'
# credit for the optional photographed ground atlas (PHOTO below), written into
# the glTF extras of the faces it colours
PHOTO_CREDIT = ("Mapillary 360 panoramas, CC BY-SA 4.0: best view per 1.5 m "
                "patch, sampled at the panorama's native equirect width, "
                "exposure-normalised and seam-levelled")
os.makedirs(TEXOUT, exist_ok=True)

# metres per texture tile, chosen so the smallest real feature of each material
# (a blade of grass, an aggregate stone, a plank) lands on a few texels
TILE_M = dict(lawn=2.0, paving=2.4, concrete=3.0, asphalt=4.0, gravel=1.6,
              sand=2.0, wood_deck=2.2, metal=2.0, brick=2.4, water=9.0)
# metallic, roughness.  Water is a DIELECTRIC: what it mirrors is Fresnel off a
# smooth dielectric, not a conductor's tinted reflection, so its metallic has to
# sit at the dielectric floor.  glTF zeroes the diffuse term at metallic 1, which
# makes water read as dark sheet metal, so the metallic-roughness TEXTURE branch
# in material() must not force metallicFactor to 1.0: metallicFactor is the number
# below and the MR texture carries roughness only.  Note roughnessFactor stays 1.0
# whenever an MR texture is present; the 0.22 here is the scalar fallback for a
# plain material.
PBR = dict(lawn=(0.0, 0.95), paving=(0.0, 0.85), concrete=(0.0, 0.88),
           asphalt=(0.0, 0.92), gravel=(0.0, 0.95), sand=(0.0, 0.96),
           wood_deck=(0.0, 0.80), metal=(0.35, 0.40), brick=(0.0, 0.88),
           water=(0.04, 0.22))
def _sky_reflect(m, mname):
    """common.SKY_REFLECT: flag a class material (extras.sky_reflect = envMapIntensity)
    so a viewer can give it the area's sky as its envMap.  Material copies made from
    it carry the same extras, so they inherit the flag."""
    v = (getattr(C, 'SKY_REFLECT', None) or {}).get(mname)
    if v:
        m.setdefault('extras', {})['sky_reflect'] = float(v)


def _floor_lit(m, mname):
    """common.FLOOR_LIT: flag a ground class material (extras.floor_lit) so a viewer can keep an
    up-pointing stored normal on its back-facing fragments (doubleSided would flip it down and draw
    low step faces dark)."""
    fl = getattr(C, 'FLOOR_LIT', None) or {}
    if fl and mname not in tuple(fl.get('skip', ())):
        m.setdefault('extras', {})['floor_lit'] = float(fl['up_min'])


def _sky_glass(m, mclass):
    """common.SKY_GLASS: flag a material whose texture packs glass and opaque texels
    (extras.sky_glass = [gain, r0, r1]) so a viewer can mirror the sky in its specular only
    where the roughness page is glass-smooth; diffuse lighting stays on the studio environment."""
    v = (getattr(C, 'SKY_GLASS', None) or {}).get(mclass)
    if v:
        m.setdefault('extras', {})['sky_glass'] = [float(x) for x in v]


# how much of the material's own chroma to keep; the rest comes from the tint
# lawn: a dry-summer lawn is olive (the panorama tint and the ortho agree on
# mean linear RGB near 0.084/0.080/0.057, G/R 1.02), and a low keep-chroma
# (0.45) flattens the grass tile's own chroma into that olive until the lawn
# reads as grey mud.  Keeping 0.85 of the tile's chroma leaves the MEAN colour
# where the photographs put it -- the tile is renormalised to that mean -- while
# the blade-to-blade green/straw variation survives, which is what makes it read
# as grass instead of noise.
KEEP_CHROMA = dict(lawn=0.85, paving=0.30, concrete=0.25, asphalt=0.30,
                   gravel=0.35, sand=0.40, wood_deck=0.65, metal=0.25,
                   brick=0.70, water=0.60)
# how much of each tile's high-frequency contrast survives (1.0 = untouched)
HF_GAIN = dict(gravel=0.80, sand=0.85)   # lawn: 1.0 (untouched)
# Metre-scale structure for the two CC0 tiles that have none.  Measured
# on the source colour maps, resampled to 512 px (about what is left
# of a 3 m tile on a wall 30 m from the camera): concrete sd/mean 0.028 and
# metal 0.022, against wood_deck 0.137 and asphalt 0.116.  Everything those two
# carry is sub-centimetre grain, the mip chain averages it to a constant, and
# the wall renders as a painted plane (a concrete wall filling 28.7 % of an
# eye-level frame measures local sd < 2.5 and saturation < 0.10).  What cast concrete and sheet metal really have at
# metre scale is a joint: concrete is poured against form panels and metal is
# hung in sheets, and both leave a line every metre or two, plus a slow blotch
# from the pour and from weathering, and concrete streaks vertically under rain.
# panel_m is chosen to divide the material's own TILE_M so the joints tile.
MACRO = dict(concrete=dict(panel_m=1.5, joint_mm=16.0, joint_drop=0.34,
                           blotch_m=0.85, blotch=0.13, streak=0.055),
             metal=dict(panel_m=1.0, joint_mm=10.0, joint_drop=0.26,
                        blotch_m=1.2, blotch=0.08, streak=0.0))
# A photographed ground atlas (the optional PHOTO input) is a real photograph resampled
# to 1.6-6.3 cm texels, so below the texel the panorama resolved it carries no
# structure at all: at a grazing angle a whole screenful of it has zero local
# variance and reads as plastic.  GROUND_DETAIL is the same trick MACRO plays
# on the tiled materials -- an achromatic, mean-1.0 multiplicative field in
# linear light, so it puts aggregate-scale relief back without moving the
# colour the photograph measured.  detail_m is the coarsest octave; finer
# octaves halve it until one would land inside min_sigma_px of the page's own
# texel grid, which is what keeps it from aliasing on the 6.3 cm pages.
GROUND_DETAIL = dict(detail_m=0.5, amp=0.11, slope=0.5, octaves=5,
                     min_sigma_px=1.6, master_px=2048, seed=63)
# GROUND_DETAIL reaches only the photographed ground atlas, which is a minority
# of what a camera sees: the parts named ground_terrain_3dfier_* and path_*
# (plus stitch_* when the optional stitch input is present) are drawn with the tiled CLASS
# materials, and between them they reference every one of MATERIALS.  So the
# same mean-1.0 achromatic multiply is applied to the class colour tiles too.
# It is a SEPARATE config, not a reuse of GROUND_DETAIL, because the two live
# at different scales: a photo page is 32 m across at 1.6-6.3 cm texels, a class
# tile is TILE_M (1.6-4.0 m) across at 1-4 mm texels and repeats in world space
# at that period, so a 0.5 m coarsest octave would put a half-metre blob on a
# two-metre tile and read as tiling.  detail_m 0.25 m down to ~15 mm is the
# band the CC0 tile itself does not carry.
# amp 0.20 is measured, not guessed: over the nine CC0 class tiles at their
# full-res sizes the multiply moves a tile's mean linear RGB by at most 0.00456
# (concrete; median 0.00203) against the 0.01 ceiling, and on concrete -- the
# flattest tile and the one most path_ parts use -- it lifts
# the local sd measured over a 0.5 m window from 5.68 to 11.53 sRGB levels and
# over a 0.15 m window from 4.85 to 7.17.  detail_m 1.0 m at the same amp
# measures 0.01049 on concrete, i.e. over the ceiling, and moves the sd less.
# The generated water tile is deliberately NOT in scope: it is not a CC0
# photograph, it already carries WATER_NOISE_AMP colour noise and WATER_PBR_AMP
# roughness noise, and its mean is what the tint depends on.
CLASS_DETAIL = dict(detail_m=0.25, amp=0.20, slope=0.5, octaves=5,
                    min_sigma_px=1.6, seed=64)
# amp 0.20 is right for a surface the camera looks along and
# wrong for one it looks AT.  A class tile is shared between the two: concrete,
# for example, is drawn on building_*, ground_*, path_* and stitch_* parts, and
# every class except lawn appears on both a horizontal and a vertical part (the
# vertical faces of low floor steps -- 3dfier step walls, path kerbs and, when
# present, stitch skirts -- land on projection axes 1 and 2).  So the amplitude cannot be a property of the class -- it has to be
# a property of the USE, and the tile is baked twice: once for the up-facing
# parts and once for the vertical ones.  On a tall concrete wall the ground
# amplitude reads as blotches, because on a wall the 0.25 m coarsest octave is a 0.25 m
# blob at eye level instead of a 0.25 m blob seen at 10 degrees.  Half the
# amplitude, the same field: the wall keeps the relief and loses the mottling.
# Same seed and same octaves on purpose -- the two tiles differ ONLY in
# contrast, so a wall and the ground it stands on cannot disagree about where
# the structure is.
CLASS_DETAIL_WALL = dict(CLASS_DETAIL, amp=0.10)
# Baking the wall variant for all nine classes adds about 12 MB to model.glb and
# to a --core crop.  Concrete and metal are the only two with a measured
# tile-level gain (concrete local sd at 25/50 cm 9.98/11.69 -> 6.82/7.70, metal
# 10.19 -> 5.79 at 50 cm); for the other seven a second 1-2 MB jpeg and a second
# material buy no measured change.  So the wall variant is baked for these two
# classes only; every other class falls back to its single ground tile and its
# single material through material()'s `use` fallback, which is the same path a
# --no-class-detail run takes.  A class NOT in this set is therefore drawn on a
# vertical part at CLASS_DETAIL's ground amplitude.
CLASS_DETAIL_WALL_CLASSES = ('concrete', 'metal')
# usage -> config.  'ground' is anything whose UVs are projected from above
# (axis_of() axis 0), 'wall' is axes 1 and 2, which is exactly the test that
# already decides how a part is unwrapped.
CLASS_DETAIL_USE = dict(ground=CLASS_DETAIL, wall=CLASS_DETAIL_WALL)
# Floor layers whose near-vertical faces (|n_z| under the
# same 0.45 axis_of() uses) are unwrapped as walls instead of from above.
# Water stays forced flat; its textures are generated for a top-down UV.
FLOOR_WALL_LAYERS = ('ground_terrain', 'paths')
FLOOR_WALL_NZ = 0.45
# class_detail() is periodic at exactly one tile by construction, so every
# component of it coarser than a few blobs is a 2.0 m repeat that reads as a
# lawn grid from above: in a top-down render the 2D FFT peak is the 2.0 m tile
# lattice (39.4 px at 0.050 m/px, harmonics to 10 px), owned by mat_lawn; on the
# prepared lawn tile the class detail
# lifts render-scale (40 px/tile) luminance sd from 4.17 to 10.06 sRGB levels
# and the 1-3 cycles/tile power 53x.  So for the classes listed here the field
# is high-passed: every component with a period longer than hipass_m is removed
# (exact, the field is periodic).  0.33 m keeps the 0.25 m octave's blob grain
# (local sd over 0.15 m 18.98 -> 18.76) and takes the 1-3 cycles/tile power
# back to the no-detail tile's.  Lawn only: the concrete/metal amplitudes were
# tuned on their 0.5 m window sd and a lawn has no joints.
CLASS_DETAIL_HIPASS_M = dict(lawn=0.33)
# Area override (common.GROUND_DETAIL_AMP): the class-detail amplitude
# of a class's GROUND bake, for a CC0 tile that carries its own grain.  Measured on
# Concrete047A in a render: the field draws dark ~1 m blotches (mat_concrete
# log-lum band sd 8-64 px 0.068 -> 0.040 without it; 0.066 with the tint flattened).
GROUND_DETAIL_AMP = dict(getattr(C, 'GROUND_DETAIL_AMP', {}))
# ground (area config, common.py): the photographed-ground hand-over onto the
# class-as-sampled tint raster, and the shading of low floor steps; see common.py
GROUND_PHOTO_MATCH = getattr(C, 'GROUND_PHOTO_MATCH', None)
# Bump GDET_ALGO whenever apply_ground_detail's maths changes, not just its
# config: the cached pages are named after this stamp (and GPM_TAG's), so without
# the bump a run with different maths would reuse pages baked by other maths.
GDET_ALGO = 2
GPM_TAG = ('_pm' + hashlib.md5((repr(sorted(GROUND_PHOTO_MATCH.items())) + f'|algo{GDET_ALGO}'
                                + ('|drawn-floor-0.32-0.18'   # bump if DARK/HARD_TINT_FLOOR change
                                   if (GROUND_PHOTO_MATCH or {}).get('level_to') == 'drawn' else '')).encode()
                               ).hexdigest()[:6]) if GROUND_PHOTO_MATCH else ''
GROUND_STEP_SHADE = getattr(C, 'GROUND_STEP_SHADE', None)
GROUND_TINT_RELIEF = getattr(C, 'GROUND_TINT_RELIEF', None)
GROUND_CANOPY_FILL = getattr(C, 'GROUND_CANOPY_FILL', None)   # level class tint under lidar tree crowns
GROUND_HARD_RECLASS = getattr(C, 'GROUND_HARD_RECLASS', None)   # lawn on measured hard ground -> hard class
GROUND_EDGE_FILL = getattr(C, 'GROUND_EDGE_FILL', None)   # level class tint along its class edge
GROUND_SHADE_FILL = getattr(C, 'GROUND_SHADE_FILL', None)   # level the DRAWN class colour where it is blue-shifted (shade)
# luma-only classes: these tiles keep only their luminance; hue comes from the tint
# (tint*M = measured world colour), so no blade/stone chroma fights the photo colour.
LUMA_ONLY = ('lawn', 'concrete', 'asphalt', 'paving', 'gravel')
# coarse photo pages (band 1 = 1024 px, band 2 = 512 px: 16-32 texels/m)
# are dropped where the class is confident; the tile + class tint draws there.
PHOTO_DROP_BANDS = (1, 2)
PHOTO_DROP_CLASSES = ('lawn', 'concrete', 'asphalt', 'paving', 'gravel')
# area-tuned colour thresholds live in the area config (common.py GROUND_CLASS_COLOUR)
LAWN_GREEN_MIN = C.GROUND_CLASS_COLOUR['lawn_green_min']
HARD_GREEN_MAX = C.GROUND_CLASS_COLOUR['hard_green_max']
GREY_LAWN_GS = C.GROUND_CLASS_COLOUR['grey_lawn_green_max']
GREY_LAWN_SAT = C.GROUND_CLASS_COLOUR['grey_lawn_sat_max']
GREY_LAWN_RECLASS = C.GROUND_CLASS_COLOUR['grey_lawn_reclass']   # False = keep grey lawn as lawn

TEX_PX = dict(lawn=2048, paving=2048, concrete=2048, asphalt=2048,
              gravel=1536, sand=1536, wood_deck=2048, metal=1024,
              brick=1536, water=1024)
NORMAL_PX = 1024
# How far the water tint is lifted over the measured top-down panorama mean.
# In a glTF renderer COLOR_0 is a linear multiplier on the shaded colour and the
# sRGB pixel goes as its 1/2.2 power (a COLOR_0 of 0.60 gives 0.7929 of the pixel
# value against 0.60**(1/2.2) = 0.7928), so an sRGB gain g needs a linear gain
# g**2.2.  1.55 lifts eye-level water to match the surrounding ground with
# margin; the water COLOR_0 stays well under 1 (about 0.08), so nothing clips.
WATER_LIFT = 1.55
# The upper bound of the water tint, as a percentile of the water batch's own
# tint, so WATER_LIFT cannot drive vertices onto the hard 1.0 clip.  See the
# water branch in the world_tint block.
WATER_TINT_CAP_PCT = 99.0
# The water colour ramp in linear RGB, shared by water_textures() and the water
# tint so both are driven by the same numbers.  The generated tile is
# renormalised to the material mean (prep_textures), which throws the ramp's hue
# away, and the world raster's water cells measure a near-neutral olive (stage
# 50's water layer: mean linear RGB about 0.0410/0.0465/0.0439, chromaticity
# 0.312/0.354/0.334), which renders with red and blue equal, i.e. as mottled
# ground, not water.  So the tint takes the ramp's chromaticity.
WATER_TROUGH = np.array([0.026, 0.031, 0.034], 'f4')
WATER_CREST = np.array([0.055, 0.066, 0.080], 'f4')
# The chromaticity (linear RGB normalised to sum 1) every water colour is
# rebased onto: the mean of the ramp above, 0.277 / 0.332 / 0.390 -- red
# clearly lowest, blue a little over green.  Provenance: the ramp's trough is a
# measured top-down open-water colour and its crest is that colour lifted toward
# the reflected sky.  The top-down measurements
# agree on the ordering but carry less blue, because a nadir view of water
# carries no grazing sky reflection: stage 50's water tint layer gives
# chromaticity 0.312/0.354/0.334 at its mean and 0.310/0.333/0.357 over its
# brightest decile, and the ortho's water faces give 0.331/0.322/0.347 at their
# 97th percentile.  The crest end is the eye-level appearance and is the same
# argument that justifies WATER_LIFT, so it is what the tint is rebased onto;
# the exact blue/green split is INFERRED from the ramp, not measured at eye
# level.  Luminance is preserved exactly, so WATER_LIFT is untouched.
WATER_CHROMA = (WATER_TROUGH + WATER_CREST) * 0.5
WATER_CHROMA = (WATER_CHROMA / WATER_CHROMA.sum()).astype('f4')
WATER_CHROMA_MIX = 1.0        # 1.0 = take the chromaticity, keep the luminance
# Bright smears on the water come from COLOR_0 alone: rendering without the
# vertex tint draws water_3dfier_water flat (the tile alone), and removing the sun
# light leaves p99/p50 unchanged, so the specular is not the cause.  The vertex
# tint follows stage 50's water layer closely (r ~0.99), and that layer is the
# panorama projection of the water (project/water_face.npz): its
# bright faces (> 2x the median, 25 % of the faces with views) have a view
# MAD of 0.051 against 0.012 on the rest -- the views DISAGREE there, which is
# what a reflection of a pier, a hull or the sky at grazing incidence does --
# and the nadir ortho under them is no brighter than elsewhere (L 0.027 vs
# 0.029).  So those texels are reflections baked into street-level photos,
# not water colour.  The p99 cap cannot reach them: they are about a quarter of
# the water vertices (> 2x median), so p99 sits inside them.
#   The water ramp already says how bright the water is meant to get:
# WATER_CREST is open water lifted toward its sky reflection.  Luminance above
# the crest (times WATER_LIFT, since the knee runs on the lifted colour) is
# compressed by WATER_KNEE_SLOPE instead of kept, so the ordering and the
# structure survive (a monotone map, not a plateau) and nothing below the crest
# moves (most water vertices are below it).
WATER_KNEE_X = 1.0            # knee at this multiple of lum(WATER_CREST) * WATER_LIFT
WATER_KNEE_SLOPE = 0.25       # gain applied to the luminance above the knee
# How much of the wave field's crest-to-trough swing each map keeps.  The
# COLOUR map keeps less than the roughness map: at the full 0.30 swing the
# water's high-frequency luminance std matches the lawn beside it (about 10
# sRGB levels), which is why it reads as ground.  The roughness map keeps the
# full 0.30 -- its variation is what breaks up the specular, and it carries no
# colour noise.
# The colour swing is 0.22 because the octave weights below move it off the
# coarse octave: the coarse (0.75 m, lawn-sized) component is 0.063 of the field
# (0.133 with equal weights), so lawn-sized mottling stays low at the higher
# total.
WATER_NOISE_AMP = 0.22
WATER_PBR_AMP = 0.30
# The colour map's TOTAL swing is kept (water with lawn-sized mottling reads as
# ground), but its energy is moved off the coarse octave onto the fine ones,
# because what reads as "untextured" is the LOCAL contrast, not the global std: a
# water tile with relative luminance sd ~0.05 (a lawn tile is ~0.19) renders the
# water at a 7 px local sd under the 2.5 that counts as flat.  Weights per colour
# octave against the height field's (1.0, 0.45, 0.20); the field is renormalised
# to unit std afterwards, so the swing is unchanged and only the scale mix moves.
# The HEIGHT field is untouched, so the normal and roughness maps still hold.
WATER_COLOUR_OCT = (0.35, 1.00, 0.60)
# The dark-tint floor.  Up-facing photo-mesh and pier deck primitives under
# LAYER_structures can carry a COLOR_0 tint of 0.07-0.28 where ordinary concrete
# carries 0.44-0.50.  That reads as a flat grey deck (the dark tint crushes the
# tile's own contrast) and as dark flakes along paths.
# Where such an up-facing vertex is inside the ortho, its
# tint is replaced by what the aerial measured on that square metre.
DARK_TINT_FLOOR = 0.32
DARK_LIFT_LAYERS = ('LAYER_structures', 'LAYER_ground', 'LAYER_paths')
# The floor of last resort.  DARK_TINT_FLOOR above only works
# where the ortho covers the vertex AND measured it brighter, so a crushed-black
# photo-mesh face outside the ortho mask would come through untouched: such a
# face can carry a COLOR_0 tint of 7-27/255 (luminance 0.03-0.11) while its baked
# luminance is normal.
# Such a vertex is scaled up to this luminance, which keeps its hue and can
# only lighten it.  Set well under DARK_TINT_FLOOR so it catches near-black
# only and leaves genuinely shadowed undersides alone.
HARD_TINT_FLOOR = 0.18
# A building skirt band (part of the stitch quads: the optional input
# polish/stitch.npz, skipped when absent or with --no-stitch) that takes the ground tint over its whole height
# puts a ruler-straight step where it meets the wall.  The band's COLOUR is
# assigned here (the stitch quads are emitted with world_tint=True), so the
# band gets a vertical gradient: the top SKIRT_BLEND_M metres take the wall's own measured colour, everything at or below the floor
# line beside the wall keeps the ground raster, smoothstep between.  A band
# vertex with no wall within SKIRT_WALL_MAXD is left on the ground tint.
SKIRT_BLEND_M = 0.30      # the band's top metres that carry the wall colour
# How far under the floor the skirt band is carried (its below_floor_m).  That
# part of the band is buried, and the floor line -- where the band stops being
# visible -- is the height the ground colour has to be reached by.
SKIRT_BURIED_M = 1.0
SKIRT_WALL_MAXD = 2.0     # no wall within this of a band vertex: no blend
SKIRT_WALL_STEP = 0.5     # how finely the wall faces' edges are sampled

FLOOR = T.SCR + '/floor'
FAC = T.SCR + '/facades'
FACKILL = FAC + '/kill_facade.npy'
SCRAPS = T.SCR + '/polish/kill_scraps.npy'
KILLW = T.SCR + '/polish/kill_layer_water.npy'
FACMAP = FAC + '/wallmap.npz'
# the glb credits the ortho the area actually used, from <scratch>/ortho/source.json
# (pipeline/fetch_ortho.py, or a local source's own source.json with a `credit`).
ORTHO_CREDIT = "no ortho"
if hasattr(C, '_areacfg') and C._areacfg.build_area_cfg(C) is not None:
    _osj = os.path.join(C._areacfg.build_area_cfg(C)['paths']['scratch_dir'], 'ortho', 'source.json')
    _os = json.load(open(_osj)) if os.path.exists(_osj) else {}
    if _os.get('source') == 'naip':
        ORTHO_CREDIT = ("USDA NAIP %s orthophoto (%.2f m, USDA Farm Production and Conservation), "
                        "public domain, via USGS The National Map" % (_os.get('year'), float(_os.get('gsd_m') or 0)))
    elif _os:
        ORTHO_CREDIT = _os.get('credit') or _os.get('licence') or 'local orthophoto'
# The OSM floors carry their tint in a world-XY raster (floor/tint_map.npz), not in the
# photo colour of each face.  COLOR_0 still delivers it, but only after the
# triangles are bisected wherever the raster changes faster than a linear
# interpolation across the triangle can follow -- which is what a fan is.
FLOOR_LAYERS = ('ground_terrain', 'paths', 'structures', 'water')
# Layers the ortho face-colour override is NOT allowed to paint.  See the use site in the layer loop for why water is on this list.
ORTHO_SKIP_LAYERS = ('water',)
TINT_TOL = 0.12          # relative luminance change a triangle may span
TINT_AMIN = 0.5          # m2; never bisect below this, the raster is 0.25 m
TINT_AMAX = 1.5          # m2; and never leave a floor triangle bigger than this
# the area cap alone leaves the 3dfier slivers long: without an edge cap,
# ground_terrain's area-weighted longest edge is about p50 2.0 m / p90 4.5 m and
# path_* 3.7 / 14 m, so the tint in COLOR_0 is sampled far coarser than the
# 0.25 m raster it is read from, and linear interpolation along a 4 m sliver
# draws streaks.  On the terrain and the path ribbons only (the water
# keeps its tuning) the longest edge is capped too, wherever the raster
# changes by more than TINT_ETOL across the triangle, with a conforming
# longest-edge-first bisection plus same-parent Delaunay flips (refine_lepp).
TINT_EDGE_LAYERS = ('ground_terrain', 'paths')
TINT_EMAX = 1.0          # m, longest edge on those layers (0 = off)
TINT_ETOL = 0.03         # relative luminance change that makes the cap bite
# ... and only on a triangle whose area is over TINT_EGATE * TINT_EMAX^2: a
# needle thinner than that draws a streak narrower than it is long by the same
# factor, and cutting every 3dfier needle to length cost 3x the triangles
# (measured on paths: 504k vs 227k at a 1 m cap, rms tint error 0.032 vs 0.051)
TINT_EGATE = 0.125

GROUP_TO_LAYER = dict(ground_terrain='LAYER_ground', paths='LAYER_paths',
                      structures='LAYER_structures', buildings='LAYER_buildings',
                      water='LAYER_water')
CORE = (-110.0, 110.0, -130.0, 90.0)     # x0,x1,y0,y1 ENU metres
PHOTO_SUB = {1: ('photo_ground', 'LAYER_ground'),
             2: ('photo_structure', 'LAYER_structures'),
             3: ('photo_facade', 'LAYER_buildings')}
LAYER_DESC = {
 'LAYER_ground': "walkable ground: classified surface materials, tinted by the "
                 "delighted multi-view colour measured from the posed panoramas",
 'LAYER_paths': "OSM footways, cycleways and steps as ribbons draped on the lidar terrain",
 'LAYER_structures': "piers, docks and bridges, from photo geometry and lidar deck heights",
 'LAYER_buildings': "building massing from OSM footprints and lidar roof heights, "
                    "plus photographed facades",
 'LAYER_trees': "tree placements; placeholder cylinders scaled to the measured "
                "crown, swap for assets",
 'LAYER_water': "the measured flat water surface"}


# ------------------------------------------------------------- textures --- #
def _periodic_noise(h, w, sigma_px, seed):
    """Zero-mean unit-sd noise that wraps exactly, smoothed to sigma_px.

    The smoothing is a Gaussian in the frequency domain, so the result is
    periodic in both axes by construction: a tile made from it still tiles.
    """
    rng = np.random.default_rng(seed)
    F = np.fft.rfft2(rng.standard_normal((h, w)))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    b = np.fft.irfft2(F * np.exp(-2.0 * (np.pi * sigma_px) ** 2
                                 * (fy ** 2 + fx ** 2)), s=(h, w))
    s = float(b.std())
    return (b / s) if s > 1e-9 else b


def macro_detail(mat, h, w, tile_m, seed=22):
    """Multiplicative metre-scale field for a MACRO material, mean 1.0.

    Returns None for a material that is not in MACRO.  The field is applied in
    linear light before the tile is renormalised to its target mean, so it
    changes the structure and not the colour the material is meant to sit at.
    """
    cfg = MACRO.get(mat)
    if cfg is None:
        return None
    # joints: whole panels per tile in each axis, so the grid wraps
    nu = max(1, int(round(tile_m / cfg['panel_m'])))
    nv = max(1, int(round(tile_m / cfg['panel_m'])))
    half_u = max(1.0, 0.5 * cfg['joint_mm'] * 1e-3 / tile_m * w)
    half_v = max(1.0, 0.5 * cfg['joint_mm'] * 1e-3 / tile_m * h)
    x = (np.arange(w) + 0.5) * nu / w
    y = (np.arange(h) + 0.5) * nv / h
    du = np.abs(x - np.round(x)) * w / nu          # px to the nearest joint
    dv = np.abs(y - np.round(y)) * h / nv
    gu = np.clip(1.0 - du / half_u, 0.0, 1.0)
    gv = np.clip(1.0 - dv / half_v, 0.0, 1.0)
    gu = gu * gu * (3 - 2 * gu); gv = gv * gv * (3 - 2 * gv)   # smoothstep
    groove = np.maximum(gu[None, :], gv[:, None])
    d = 1.0 - cfg['joint_drop'] * groove
    # blotch: the pour / weathering patch, in metres
    if cfg['blotch'] > 0:
        sig = cfg['blotch_m'] / tile_m * w
        d = d * (1.0 + cfg['blotch'] * _periodic_noise(h, w, sig, seed))
    # streak: rain runs down a wall, so the field is long in v and narrow in u
    if cfg['streak'] > 0:
        st = _periodic_noise(h, w, max(1.0, 0.06 / tile_m * w), seed + 1)
        from scipy import ndimage as _nd
        st = _nd.uniform_filter1d(st, max(3, int(0.35 * h)), axis=0, mode='wrap')
        s = float(st.std())
        if s > 1e-9:
            d = d * (1.0 + cfg['streak'] * st / s)
    d = np.clip(d, 0.05, None)
    return (d / d.mean()).astype('f4')


_GDET = {}
# short config stamp for the cached detail pages, so retuning GROUND_DETAIL
# invalidates them instead of silently reusing the old ones
# GDET_ALGO (defined above GPM_TAG) is part of the stamp too.
GDET_TAG = '_' + hashlib.md5(
    (repr(sorted(GROUND_DETAIL.items())) + f'|algo{GDET_ALGO}').encode()
).hexdigest()[:6]
# the same stamp for the class tiles.  prep_textures() rewrites its tiles every
# run, so this is belt and braces, but it means a tile baked with one CLASS_DETAIL
# can never be picked up by a run using another one -- the name carries the config.
# one stamp per usage, so the two variants of a class tile can never collide in
# the cache and retuning either one invalidates only its own tiles
CDET_TAG = {u: '_cd' + hashlib.md5((repr(sorted(c.items()))
                                   + '|hp' + repr(sorted(CLASS_DETAIL_HIPASS_M.items()))
                                   + ('|ga' + repr(sorted(GROUND_DETAIL_AMP.items())) if u == 'ground' else '')
                                   ).encode()).hexdigest()[:6]
            for u, c in CLASS_DETAIL_USE.items()}


def class_detail(h, w, tile_m, cfg=None, hipass_m=None):
    """Achromatic multiplicative detail for one tiled CLASS material, mean 1.0.

    Same trick as ground_detail(): one achromatic, mean-1.0 multiplicative field
    in linear light, so the tile's hue cannot move.  Two differences, both
    forced by what a class tile is:

    * it is built at the tile's own (h, w) instead of box-averaged down from a
      square master, because the CC0 tiles are not square (concrete is 2048x1024)
      and a class tile does not have to line up with its neighbours in world
      space the way two photo pages do -- it repeats against itself.
    * the octave sigmas come from the tile's own metres-per-tile, so the field
      wraps at exactly one tile and adds no seam the tiling did not already have.

    sigma is taken from the width, which is the convention macro_detail() already
    uses on these same tiles, so on a 2:1 tile the field is stretched in v by the
    same factor the material itself is.
    """
    cfg = dict(CLASS_DETAIL, **(cfg or {}))
    h, w = int(h), int(w)
    ck = tuple(sorted((k, float(v)) for k, v in cfg.items()))
    key = (h, w, float(tile_m), ck, hipass_m)
    if key in _GDET:
        return _GDET[key]
    ppm = w / float(tile_m)
    f = np.zeros((h, w), 'f8')
    for k in range(int(cfg['octaves'])):
        sig = 0.5 * cfg['detail_m'] * ppm / (2.0 ** k)
        if sig < cfg['min_sigma_px']:
            break
        f += (2.0 ** k) ** (-cfg['slope']) * _periodic_noise(
            h, w, sig, int(cfg['seed']) + k)
    sd = float(f.std())
    if sd > 1e-9:
        f /= sd
    d = np.clip(1.0 + float(cfg['amp']) * f, 0.05, None)
    if hipass_m:
        # cycles per metre: the tile is tile_m wide and tile_m*h/w tall (the
        # sigma is taken from the width, see above)
        F = np.fft.fft2(d - d.mean())
        fu = np.fft.fftfreq(w)[None, :] * w / float(tile_m)
        fv = np.fft.fftfreq(h)[:, None] * w / float(tile_m)
        F[np.hypot(fu, fv) <= 1.0 / float(hipass_m)] = 0.0
        d = np.clip(1.0 + np.real(np.fft.ifft2(F)), 0.05, None)
    d = (d / d.mean()).astype('f4')
    _GDET[key] = d
    return d


def ground_detail(px, page_m=32.0, cfg=None):
    """Achromatic multiplicative detail for one photographed ground page.

    Mean exactly 1.0 and neutral: the same scalar multiplies all three
    channels, so the page's hue cannot move.  One master field is built at
    master_px over the 32 m page and every page is box-averaged down from it,
    which (a) makes the field identical in world space for every page, so two
    neighbouring pages meet without a seam whatever resolution each was baked
    at, and (b) low-passes the fine octaves away on the coarse pages instead of
    aliasing them.  The octaves are _periodic_noise, so the field wraps in both
    axes at exactly one page.
    """
    cfg = dict(GROUND_DETAIL, **(cfg or {}))
    px, mp = int(px), int(cfg['master_px'])
    ck = tuple(sorted((k, float(v)) for k, v in cfg.items()))
    key = (px, float(page_m), ck)
    if key in _GDET:
        return _GDET[key]
    mkey = (0, float(page_m), ck)
    f = _GDET.get(mkey)
    if f is None:
        ppm = mp / float(page_m)
        f = np.zeros((mp, mp), 'f8')
        for k in range(int(cfg['octaves'])):
            sig = 0.5 * cfg['detail_m'] * ppm / (2.0 ** k)
            if sig < cfg['min_sigma_px']:
                break
            f += (2.0 ** k) ** (-cfg['slope']) * _periodic_noise(
                mp, mp, sig, int(cfg['seed']) + k)
        sd = float(f.std())
        if sd > 1e-9:
            f /= sd
        _GDET[mkey] = f
    if px != mp:
        from PIL import Image
        g = np.asarray(Image.fromarray(f.astype('f4'), 'F').resize(
            (px, px), Image.BOX if px < mp else Image.BILINEAR), 'f8')
    else:
        g = f
    d = np.clip(1.0 + float(cfg['amp']) * g, 0.05, None)
    d = (d / d.mean()).astype('f4')
    _GDET[key] = d
    return d


def apply_ground_detail(src, dst, page_m=32.0, cfg=None, quality=96):
    """Write src multiplied by ground_detail into dst.  Returns the before and
    after mean linear RGB of the page, which is what the <1 % colour promise is
    measured on."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    lin = np.power(np.asarray(Image.open(src).convert('RGB'), 'f4') / 255.0, 2.2)
    m0 = lin.reshape(-1, 3).mean(0)
    d = ground_detail(lin.shape[1], page_m, cfg)
    if d.shape[0] != lin.shape[0]:
        d = ground_detail(lin.shape[0], page_m, cfg)
    lin = np.clip(lin * d[:, :, None], 0.0, 1.0)
    mraw = lin.reshape(-1, 3).mean(0)
    # The field's own mean is exactly 1.0, but the page's mean after the multiply
    # is not exactly the page's mean before it: the field and the photograph are
    # independent, not orthogonal, so over one finite page they correlate a
    # little, and the clip at 1.0 eats a little more.  Over a set of photo pages
    # at amp 0.11 that can leave a worst page at 0.0154 -- ABOVE the 0.01 the
    # extras promise.
    # Dividing the measured move back out makes the promise exact; the second
    # clip can only push a channel back down, never up.
    lin = np.clip(lin * (m0 / np.maximum(mraw, 1e-6))[None, None, :], 0.0, 1.0)
    m1 = lin.reshape(-1, 3).mean(0)
    apply_ground_detail.last_uncorrected = float(
        np.abs(mraw / np.maximum(m0, 1e-6) - 1.0).max())
    out = np.clip(np.power(lin, 1 / 2.2) * 255.0, 0, 255).astype('u1')
    Image.fromarray(out).save(dst, 'JPEG', quality=quality, optimize=True)
    return m0, m1


TINT_DECIDE = {}   # pre-relief class tints, for class decisions only
CANOPY_RATIO = {}  # canopy_fill's per-class colour factor, applied to the final vertex tint in emit()
SHADE_W = {}       # shade_fill's per-class weight raster (0..1 on the tint grid), for the stage's rule check



def hard_reclass_mask(cfg, report=None):
    """common.GROUND_HARD_RECLASS: -> (dict class name -> bool raster on the tint grid, grid) of the terrain lawn
    cells the ortho and the panorama world tint both measure as neutral hard ground, open sky, in regions >= min_area_m2,
    each region given the hard class stage 21's photo_scores picks from its median ortho colour.  Reads files only (ortho
    rasters, floor/tint_map.npz, lidar canopy, the terrain layer and its stage 21 classes), so any caller
    computes the same mask."""
    from scipy import ndimage
    import polish_lib as PL
    from s21_material import photo_scores
    report = {} if report is None else report
    r = np.load(ORTHO + '/rasters.npz'); Mo = r['Mo'].astype('f4'); ov = r['ovalid'] > 0
    x0, y0, c, nx, ny = [float(v) for v in r['grid']]; nx, ny = int(nx), int(ny)
    Mp = np.load(FLOOR + '/tint_map.npz')['M'].astype('f4')
    if Mp.shape[:2] != Mo.shape[:2]:
        report['skipped'] = 'tint_map grid != ortho grid'; return {}, None
    mm = np.load(T.SCR + '/material_ortho/ground_terrain_mat.npz'); z = np.load(C.W + '/layers/ground_terrain.npz')
    sel = (~mm['kill']) & (mm['up'] > 0.70)
    out = {}
    for src_cls in cfg['classes']:
        k = MI[src_cls]; ix = np.nonzero(sel & (mm['mat'] == k))[0]
        if not len(ix):
            continue
        t = z['pos'].astype('f8')[z['idx'][ix]]
        acc, den = PL.rasterise_tris(t, np.ones((len(ix), 1), 'f8'), nx, ny, x0, y0, c)
        own = den > 1e-9
        s_ = cfg['sigma_m'] / c; w = np.maximum(ndimage.gaussian_filter(own.astype('f4'), s_), 1e-4)
        wo = np.maximum(ndimage.gaussian_filter((own & ov).astype('f4'), s_), 1e-4)
        mo = np.stack([ndimage.gaussian_filter(Mo[..., q] * (own & ov), s_) for q in range(3)], -1) / wo[..., None]
        mp = np.stack([ndimage.gaussian_filter(Mp[..., q] * own, s_) for q in range(3)], -1) / w[..., None]
        bg = lambda a: np.log((a[..., 2] + 1e-4) / (a[..., 1] + 1e-4))
        ch = np.load(C.W + '/geom/chm.npz'); hx0, hy0, hc, hnx, hny = [float(v) for v in ch['grid']]
        jx = np.clip(((x0 + (np.arange(nx) + 0.5) * c - hx0) / hc).astype(int), 0, int(hnx) - 1)
        iy = np.clip(((y0 + (np.arange(ny) + 0.5) * c - hy0) / hc).astype(int), 0, int(hny) - 1)
        can = ch['canopy'].astype(bool)[np.ix_(iy, jx)]; dcan = ndimage.distance_transform_edt(~can) * c
        cand = (own & ov & (bg(mo) > cfg['ortho_bg_min']) & (bg(mo) <= cfg['ortho_bg_max'])
                & ((mo @ T.LUMW) >= cfg['ortho_luma_min']) & (bg(mp) > cfg['pano_bg_min']) & (dcan > cfg['ring_m']))
        if cfg.get('water_m') and 'wmask' in r.files:   # a shore bank is riprap or mud under the waterline haze, not a plaza
            cand &= ndimage.distance_transform_edt(~(r['wmask'] > 0)) * c > cfg['water_m']
        if cfg.get('open_m'):   # strips thinner than 2 x open_m are the neighbour's colour bled over the class edge
            k_ = int(round(cfg['open_m'] / c)); yy, xx = np.mgrid[-k_:k_ + 1, -k_:k_ + 1]
            cand = ndimage.binary_opening(cand, structure=(xx * xx + yy * yy) <= k_ * k_)
        lab, n = ndimage.label(cand)
        rs = dict(regions=0, area_m2=0.0, by_class={})
        for q in range(1, n + 1):
            m = lab == q; a = float(m.sum()) * c * c
            if a < cfg['min_area_m2']:
                continue
            o8 = np.clip(np.power(np.clip(Mo[m & ov], 0, 1), 1 / 2.2) * 255, 0, 255)   # real grass, even dry, leaves green cells
            gshare = float(np.mean((2 * o8[:, 1] - o8[:, 0] - o8[:, 2]) / (o8.sum(1) + 1e-3) > 0.03))
            if gshare > cfg['max_green_share']:
                rs['rejected_green'] = rs.get('rejected_green', 0) + 1; continue
            med = np.median(Mo[m & ov], 0).astype('f4')
            sc, _ = photo_scores(med[None], np.array([3]))
            best = max(cfg['to'], key=lambda h: sc[0, MI[h]])
            if sc[0, MI[best]] < cfg['min_score']:
                continue
            out.setdefault(best, np.zeros((ny, nx), bool))
            out[best] |= m
            rs['regions'] += 1; rs['area_m2'] += a; rs['by_class'][best] = round(rs['by_class'].get(best, 0) + a, 1)
        rs['area_m2'] = round(rs['area_m2'], 1); report[src_cls] = rs
    return out, (x0, y0, c, nx, ny)


def tint_relief(TINTC, TINT, cfg, report):
    """common.GROUND_TINT_RELIEF: scale each class's band_m log-luma relief
    down to the ortho's on that class's cells.  Edits TintMap.M in place."""
    from scipy import ndimage
    rp = T.SCR + '/ortho/rasters.npz'; cp = T.SCR + '/polish/class_map.npz'
    if not (os.path.exists(rp) and os.path.exists(cp)):
        report['skipped'] = 'no ortho or class map'; return
    o = np.load(rp); Mo = o['Mo'].astype('f4'); ov = o['ovalid']
    cls = np.load(cp)['cls']
    LW = T.LUMW.astype('f4')
    for ci, mname in enumerate(MATERIALS):
        if mname not in cfg['classes'] or mname not in TINTC:
            continue
        tm = TINTC[mname]
        if tm.M.shape[:2] != cls.shape or Mo.shape[:2] != cls.shape or np.asarray(TINT.M).shape[:2] != cls.shape:
            report[mname] = 'grid mismatch, skipped'; continue
        cell = float(tm.c)
        lo, hi = cfg['band_m'][0] / cell, cfg['band_m'][1] / cell
        cm = cls == ci
        w = cm.astype('f4')
        def band(L):
            def g(s_):
                return ndimage.gaussian_filter(L * w, s_) / np.maximum(ndimage.gaussian_filter(w, s_), 1e-4)
            return g(lo) - g(hi)
        Mt = np.asarray(tm.M, 'f4')
        dt = band(np.log(np.maximum(Mt @ LW, 1e-5)))
        do = band(np.log(np.maximum(Mo @ LW, 1e-5)))
        m = ndimage.binary_erosion(cm & ov, iterations=4)
        if m.sum() < 2000:
            report[mname] = f'{int(m.sum())} ortho cells, skipped'; continue
        st, so = float(dt[m].std()), float(do[m].std())
        k = float(np.clip(so / max(st, 1e-6), cfg['k_min'], 1.0))
        # k is fitted on luma; the same scale goes on each channel's own log band, so the
        # hue blotches (brown/green) shrink with the brightness ones
        fac = np.stack([np.exp(-(1.0 - k) * band(np.log(np.maximum(Mt[..., c], 1e-5))))
                        for c in range(3)], -1).astype('f4')
        fac[~cm] = 1.0
        # the class DECISIONS (the coarse-page drop) keep reading the measured tint: the
        # relief is a drawing amplitude, it must not move which faces take a photo
        TINT_DECIDE[mname] = TintMap(M=np.asarray(tm.M), grid=[tm.x0, tm.y0, tm.c, tm.nx, tm.ny])
        tm.M = Mt * fac
        R = np.asarray(TINT.M, 'f4')
        R[cm] = R[cm] * fac[cm]
        TINT.M = R
        report[mname] = dict(tint_band_sd=round(st, 4), ortho_band_sd=round(so, 4), k=round(k, 3),
                             cells=int(cm.sum()))


def canopy_fill(TINTC, TINT, cfg, report):
    """common.GROUND_CANOPY_FILL: under a lidar tree crown the ortho has no pixel and the
    panoramas saw the class in the crown's shade (blue sky-lit, dark) or not at all (fill), so the
    class tint, without this fill, draws purple-brown patches under every tree.  Level each listed class's low_m
    low-pass (log, per channel) onto the ref_m Gaussian mean (log) of the same class's OPEN
    cells (farther than open_m from any crown), with weight 1 under the crown fading to 0 at ring_m
    outside it; detail below low_m is kept.  Stores the factor in CANOPY_RATIO[m]; emit() multiplies every vertex tint
    of the class by it.  No raster is edited, so geometry and class decisions do not move."""
    from scipy import ndimage
    cp = T.SCR + '/polish/class_map.npz'; hp = C.W + '/geom/chm.npz'
    if not (os.path.exists(cp) and os.path.exists(hp)):
        report['skipped'] = 'no class map or lidar canopy'; return
    cls = np.load(cp)['cls']; ch = np.load(hp)
    for ci, mname in enumerate(MATERIALS):
        if mname not in cfg['classes'] or mname not in TINTC:
            continue
        tm = TINTC[mname]
        if tm.M.shape[:2] != cls.shape or np.asarray(TINT.M).shape[:2] != cls.shape:
            report[mname] = 'grid mismatch, skipped'; continue
        c = float(tm.c); ny, nx = cls.shape
        hx0, hy0, hc, hnx, hny = [float(v) for v in ch['grid']]
        jx = np.clip(((tm.x0 + (np.arange(nx) + 0.5) * c - hx0) / hc).astype(int), 0, int(hnx) - 1)
        iy = np.clip(((tm.y0 + (np.arange(ny) + 0.5) * c - hy0) / hc).astype(int), 0, int(hny) - 1)
        can = ch['canopy'].astype(bool)[np.ix_(iy, jx)]
        d = ndimage.distance_transform_edt(~can) * c
        w = np.where(can, 1.0, np.clip(1.0 - d / max(cfg['ring_m'], 1e-3), 0.0, 1.0)).astype('f4')
        cm = cls == ci
        op = (cm & (d > cfg['open_m'])).astype('f4')
        if op.sum() < 2000 or not (w[cm] > 0).any():
            report[mname] = f'{int(op.sum())} open cells, skipped'; continue
        # an earlier factor on this class (edge_fill runs first) is part of the colour being levelled, and the
        # open-lawn reference is the lawn as edge_fill left it
        prev = CANOPY_RATIO.get(mname)
        pf = np.asarray(prev.M, 'f4') if prev is not None else np.ones(tm.M.shape, 'f4')
        Mt = np.asarray(tm.M, 'f4') * pf; lg = np.log(np.maximum(Mt, 1e-5))
        cmf = cm.astype('f4'); sl = cfg['low_m'] / c
        dl = np.maximum(ndimage.gaussian_filter(cmf, sl), 1e-4)
        # reference: open cells, widening until the local weight is enough (a copse in a big lawn)
        ref = np.zeros_like(lg); got = np.zeros(cls.shape, bool)
        for k in (1, 2, 4, 8):
            s_ = cfg['ref_m'] * k / c; dr = ndimage.gaussian_filter(op, s_)
            ok = (dr > 0.02) & ~got
            for q in range(3):
                ref[..., q][ok] = (ndimage.gaussian_filter(lg[..., q] * op, s_)[ok] / dr[ok])
            got |= ok
        med = np.array([np.median(lg[..., q][op > 0]) for q in range(3)], 'f4')
        ref[~got] = med
        delta = np.zeros_like(lg)
        for q in range(3):
            delta[..., q] = ref[..., q] - ndimage.gaussian_filter(lg[..., q] * cmf, sl) / dl
        LW = T.LUMW.astype('f4')
        if cfg.get('luma', 'level') == 'lift':   # never darken: drop a negative luma part, keep the chroma move
            dlu = delta @ LW; delta = delta - np.minimum(dlu, 0.0)[..., None]
        delta = np.clip(delta, -cfg['max_log'], cfg['max_log']) * (cfg['gain'] * w)[..., None]
        # the layer is sampled a little past its own class edge: carry the edit out ring_m by nearest
        near = ndimage.distance_transform_edt(~cm, return_distances=True, return_indices=True)
        dd, (ii, jj) = near
        fac = np.exp(delta[ii, jj]).astype('f4'); fac[dd * c > cfg['ring_m']] = 1.0
        # the factor goes on the FINAL vertex tint in emit(), not on this layer.  Editing the layer instead
        # would (a) miss the photo mesh's structure sublayer, which draws the multi-view face colour, so
        # canopy/ring lawn vertices would keep the shade, and (b) lift some ring vertices just over
        # DARK_TINT_FLOOR, which switches off their stronger ortho lift (drawn darker than without the fill).  No raster moves, so TINT / TINT_DECIDE / refinement,
        # class decisions and photo pages are untouched.
        CANOPY_RATIO[mname] = TintMap(M=pf * fac, grid=[tm.x0, tm.y0, tm.c, tm.nx, tm.ny])
        sel = cm & can
        report[mname] = dict(canopy_cells=int(sel.sum()), weighted_cells=int((cm & (w > 0)).sum()),
                             open_cells=int(op.sum()),
                             canopy_dlogL_median=round(float(np.median((np.log(np.maximum(fac, 1e-6)) @ LW)[sel])), 4) if sel.any() else None,
                             ref_fallback_cells=int((cm & ~got).sum()))
    return


def edge_fill(TINTC, TINT, cfg, report):
    """common.GROUND_EDGE_FILL: the measured class tint greys out toward the class edge (the neighbouring
    path's colour bleeding in through class-map misregistration and pano blur), and the class tile's own hue draws
    that grey as mauve.  Level each listed class's low_m low-pass (log, per channel; stage 22 runs it before
    canopy_fill, so it reads the class layer after tint_relief and any factor already in CANOPY_RATIO) onto the ref_m Gaussian mean (log) of the class's interior cells (> edge_m from any other class), weight 1
    up to full_m inside the edge fading to 0 at edge_m, and weight 1 off the class (off_class: class-tile faces the
    class map calls another class).  The factor is multiplied into CANOPY_RATIO[m], which emit()
    applies to the FINAL vertex tint of every colour source of the class.  No raster is edited."""
    from scipy import ndimage
    cp = T.SCR + '/polish/class_map.npz'
    if not os.path.exists(cp):
        report['skipped'] = 'no class map'; return
    cls = np.load(cp)['cls']
    LW = T.LUMW.astype('f4')
    for ci, mname in enumerate(MATERIALS):
        if mname not in cfg['classes'] or mname not in TINTC:
            continue
        tm = TINTC[mname]
        if tm.M.shape[:2] != cls.shape:
            report[mname] = 'grid mismatch, skipped'; continue
        c = float(tm.c); cm = cls == ci
        prev = CANOPY_RATIO.get(mname)
        pf = np.asarray(prev.M, 'f4') if prev is not None else np.ones(tm.M.shape, 'f4')
        lg = np.log(np.maximum(np.asarray(tm.M, 'f4') * pf, 1e-5))
        d = ndimage.distance_transform_edt(cm) * c          # distance inside the class to the nearest other cell
        w = np.clip((cfg['edge_m'] - d) / max(cfg['edge_m'] - cfg['full_m'], 1e-3), 0.0, 1.0).astype('f4')
        w[~cm] = 0.0
        op = (cm & (d > cfg['edge_m'])).astype('f4')
        if op.sum() < 2000:
            report[mname] = f'{int(op.sum())} interior cells, skipped'; continue
        cmf = cm.astype('f4'); sl = cfg['low_m'] / c
        dl = np.maximum(ndimage.gaussian_filter(cmf, sl), 1e-4)
        ref = np.zeros_like(lg); got = np.zeros(cls.shape, bool)
        for k in (1, 2, 4, 8):   # widen until the interior weight is enough (a narrow strip of lawn)
            s_ = cfg['ref_m'] * k / c; dr = ndimage.gaussian_filter(op, s_)
            ok = (dr > 0.02) & ~got
            for q in range(3):
                ref[..., q][ok] = ndimage.gaussian_filter(lg[..., q] * op, s_)[ok] / dr[ok]
            got |= ok
        ref[~got] = np.array([np.median(lg[..., q][op > 0]) for q in range(3)], 'f4')
        # off the class (a class-tile face the class map does not call this class: 2.5 % of eye-level screen for the
        # lawn, 73 % of it grey) the layer holds the fill it was given; the face is still drawn as this class, so it
        # takes the interior colour at full weight (plain low-pass there, the class-masked one inside)
        w[~cm] = 1.0 if cfg.get('off_class', True) else 0.0
        delta = np.zeros_like(lg)
        for q in range(3):
            delta[..., q] = ref[..., q] - np.where(cm, ndimage.gaussian_filter(lg[..., q] * cmf, sl) / dl,
                                                   ndimage.gaussian_filter(lg[..., q], sl))
        if cfg.get('luma', 'level') == 'chroma':   # keep the measured luma, move only the hue
            delta = delta - (delta @ LW)[..., None]
        delta = np.clip(delta, -cfg['max_log'], cfg['max_log']) * (cfg['gain'] * w)[..., None]
        fac = np.exp(delta).astype('f4')
        CANOPY_RATIO[mname] = TintMap(M=pf * fac, grid=[tm.x0, tm.y0, tm.c, tm.nx, tm.ny])
        sel = cm & (w > 0)
        lf = np.log(np.maximum(fac, 1e-6))
        report[mname] = dict(weighted_cells=int(sel.sum()), interior_cells=int(op.sum()), off_class=bool(cfg.get('off_class', True)),
                             dlogL_median=round(float(np.median((lf @ LW)[sel])), 4),
                             dlogBG_median=round(float(np.median((lf[..., 2] - lf[..., 1])[sel])), 4),
                             ref_fallback_cells=int((cm & ~got).sum()))
    return


def shade_fill(TINTC, means, orthotint, cfg, report):
    """common.GROUND_SHADE_FILL: lawn the panoramas and the ortho saw in tree shade draws grey-blue
    even after the canopy and edge fills, because those factors are fitted to the class LAYER while emit() draws the layer
    through the dark-tint floors (ortho luminance, tile hue when crushed), and the canopy fill fades out 3 m from the
    crown while the cast shadow lies beside it.  Rebuild the class's drawn colour on the tint grid (layer / tile mean
    -> ground_tint_floor -> x the earlier factor), low-pass it (low_m, log), and where its log B/G sits bg_min..bg_full
    above the ref_m Gaussian mean of the class's sunlit open cells (> open_m from a lidar crown, ortho-lit with
    lit_ref) level the low-pass onto that reference (weight 0 -> 1 over the ramp, max_log cap, gain).  Multiplies
    into CANOPY_RATIO[m] (the final vertex tint of every colour source); carried carry_m off the class by nearest.
    Runs after prep_textures (needs the tile means).  No raster or class decision moves."""
    from scipy import ndimage
    cp = T.SCR + '/polish/class_map.npz'; hp = C.W + '/geom/chm.npz'; rp = ORTHO + '/rasters.npz'
    if not (os.path.exists(cp) and os.path.exists(hp)):
        report['skipped'] = 'no class map or lidar canopy'; return
    cls = np.load(cp)['cls']; ch = np.load(hp)
    lit = None
    if cfg.get('lit_ref', True) and os.path.exists(rp):
        lit = np.load(rp)['okU'].astype(bool)
    LW = T.LUMW.astype('f4')
    for ci, mname in enumerate(MATERIALS):
        if mname not in cfg['classes'] or mname not in TINTC or mname not in means:
            continue
        tm = TINTC[mname]
        if tm.M.shape[:2] != cls.shape or (lit is not None and lit.shape != cls.shape):
            report[mname] = 'grid mismatch, skipped'; continue
        c = float(tm.c); ny, nx = cls.shape; cm = cls == ci
        hx0, hy0, hc, hnx, hny = [float(v) for v in ch['grid']]
        jx = np.clip(((tm.x0 + (np.arange(nx) + 0.5) * c - hx0) / hc).astype(int), 0, int(hnx) - 1)
        iy = np.clip(((tm.y0 + (np.arange(ny) + 0.5) * c - hy0) / hc).astype(int), 0, int(hny) - 1)
        can = ch['canopy'].astype(bool)[np.ix_(iy, jx)]
        d = ndimage.distance_transform_edt(~can) * c
        op = cm & (d > cfg['open_m'])
        if lit is not None:
            op &= lit
        op = op.astype('f4')
        if op.sum() < 2000:
            report[mname] = f'{int(op.sum())} sunlit open cells, skipped'; continue
        # the colour emit() draws: layer / tile mean through the vertex-tint floors, times the fills so far
        M = np.asarray(means[mname], 'f4')
        prev = CANOPY_RATIO.get(mname)
        pf = np.asarray(prev.M, 'f4') if prev is not None else np.ones(tm.M.shape, 'f4')
        raw = np.clip(np.asarray(tm.M, 'f4') / np.maximum(M, 1e-4), 0.03, 1.0).reshape(-1, 3)
        Xc = (tm.x0 + (np.arange(nx) + 0.5) * c)[None, :].repeat(ny, 0).ravel()
        Yc = (tm.y0 + (np.arange(ny) + 0.5) * c)[:, None].repeat(nx, 1).ravel()
        drawn = ground_tint_floor(raw, Xc, Yc, np.broadcast_to(M, raw.shape), orthotint).reshape(ny, nx, 3)
        drawn = np.clip(drawn * pf, 0.03, 1.0)
        lg = np.log(np.maximum(drawn, 1e-5))
        cmf = cm.astype('f4'); sl = cfg['low_m'] / c
        dl = np.maximum(ndimage.gaussian_filter(cmf, sl), 1e-4)
        low = np.stack([ndimage.gaussian_filter(lg[..., q] * cmf, sl) / dl for q in range(3)], -1)
        ref = np.zeros_like(lg); got = np.zeros(cls.shape, bool)
        for k in (1, 2, 4, 8):   # widen until the sunlit weight is enough (a shaded copse in a big lawn)
            s_ = cfg['ref_m'] * k / c; dr = ndimage.gaussian_filter(op, s_)
            ok = (dr > 0.02) & ~got
            for q in range(3):
                ref[..., q][ok] = ndimage.gaussian_filter(lg[..., q] * op, s_)[ok] / dr[ok]
            got |= ok
        ref[~got] = np.array([np.median(lg[..., q][op > 0]) for q in range(3)], 'f4')
        delta = ref - low
        bshift = (low[..., 2] - low[..., 1]) - (ref[..., 2] - ref[..., 1])   # drawn log B/G above the sunlit lawn's
        w = np.clip((bshift - cfg['bg_min']) / max(cfg['bg_full'] - cfg['bg_min'], 1e-3), 0.0, 1.0).astype('f4')
        w[~cm] = 0.0
        if cfg.get('luma', 'level') == 'chroma':   # keep the drawn luma, move only the hue
            delta = delta - (delta @ LW)[..., None]
        delta = np.clip(delta, -cfg['max_log'], cfg['max_log']) * (cfg['gain'] * w)[..., None]
        # the layer is sampled a little past its own class edge: carry the edit out carry_m by nearest
        dd, (ii, jj) = ndimage.distance_transform_edt(~cm, return_distances=True, return_indices=True)
        fac = np.exp(delta[ii, jj]).astype('f4'); fac[dd * c > cfg['carry_m']] = 1.0
        CANOPY_RATIO[mname] = TintMap(M=pf * fac, grid=[tm.x0, tm.y0, tm.c, tm.nx, tm.ny])
        SHADE_W[mname] = TintMap(M=np.repeat(w[..., None], 3, 2), grid=[tm.x0, tm.y0, tm.c, tm.nx, tm.ny])
        sel = cm & (w > 0.5); lf = np.log(np.maximum(fac, 1e-6))
        report[mname] = dict(weighted_cells=int((cm & (w > 0)).sum()), shade_cells=int(sel.sum()), sunlit_cells=int(op.sum()),
                             dlogL_median=round(float(np.median((lf @ LW)[sel])), 4) if sel.any() else None,
                             dlogBG_median=round(float(np.median((lf[..., 2] - lf[..., 1])[sel])), 4) if sel.any() else None,
                             bshift_median=round(float(np.median(bshift[sel])), 4) if sel.any() else None,
                             ref_fallback_cells=int((cm & ~got).sum()))
    return


# (common.GROUND_DARK_LIFT): hue='tint' = the ortho lift sets luminance only
DARK_LIFT = getattr(C, 'GROUND_DARK_LIFT', None) or {}
DARK_LIFT_HUE = DARK_LIFT.get('hue', 'ortho')


def _lift_keep_hue(raw, lum, ortho_lift):
    """The ortho-lifted colours ortho_lift (n,3) with the tint raw's (n,3) chroma: raw
    scaled to the ortho's luminance, hue kept, clipped 0.03-1.  A tint darker than
    GROUND_DARK_LIFT hue_min_lum has no trustworthy chroma: dark_hue 'tile' draws it in
    the class tile's own hue at the ortho's luminance, 'ortho' keeps the ortho colour."""
    raw = np.asarray(raw, 'f4'); lum = np.asarray(lum, 'f4'); ortho_lift = np.asarray(ortho_lift, 'f4')
    ol = ortho_lift @ T.LUMW.astype('f4')
    g = (ol / np.maximum(lum, 1e-3))[:, None]
    out = np.clip(raw * g, 0.03, 1.0).astype('f4')
    dark = lum < float(DARK_LIFT.get('hue_min_lum', 0.0))
    if DARK_LIFT.get('dark_hue', 'ortho') == 'tile':
        # grey in raw (= tint / tile mean) space: the class tile's own hue at the ortho's luminance
        out[dark] = np.clip(ol[dark], 0.03, 1.0)[:, None]
    else:
        out[dark] = ortho_lift[dark]
    return out


def ground_tint_floor(raw, X, Y, M, orthotint):
    """The vertex-tint floors stage 22 puts on an up-facing ground tile vertex
    (dark_floor + hard_floor in main(), same constants) as a pure function, so the
    photo pages can be levelled onto the tile as it is drawn.  raw = tint / M (the
    COLOR_0 value, clipped to 0.03-1), M = the class tile's mean (n,3), at world X, Y.
    Under DARK_TINT_FLOOR luminance, where the ortho covers the point and is
    brighter, raw takes the ortho colour / M; anything still under HARD_TINT_FLOOR is
    scaled up to it, hue kept.  orthotint = (colour TintMap, valid TintMap) or None."""
    LW = T.LUMW.astype('f4')
    raw = np.array(raw, 'f4', copy=True)
    lum = raw @ LW
    if orthotint is not None:
        om, ov = orthotint
        ii = np.nonzero((lum < DARK_TINT_FLOOR) & (ov.at(X, Y)[:, 0] > 0.5))[0]
        if len(ii):
            ot = np.clip(om.at(X[ii], Y[ii]).astype('f4') / np.maximum(M[ii], 1e-4), 0.03, 1.0)
            keep = (ot @ LW) > lum[ii]            # never darken, as dark_floor
            if DARK_LIFT_HUE == 'tint':
                ot = _lift_keep_hue(raw[ii], lum[ii], ot)
            raw[ii[keep]] = ot[keep]
            lum = raw @ LW
    h = lum < HARD_TINT_FLOOR
    if h.any():
        raw[h] = np.clip(raw[h] * (HARD_TINT_FLOOR / np.maximum(lum[h], 1e-3))[:, None], 0.03, 1.0)
    return raw


def _feather_tile_look(X, Y, clsgrid, cls, tiles, lift=None):
    """Linear RGB of the class tile x tint at world X, Y (page texel grid): the tile
    image pre-resampled to the page texel, wrapped in world space.  lift: optional
    f(raw, X, Y, M) -> raw after stage 22's vertex-tint floors (ground_tint_floor)."""
    gx0, gy0, c = clsgrid
    ny, nx = cls.shape
    ci = np.clip(((Y - gy0) / c).astype(np.int32), 0, ny - 1)
    cj = np.clip(((X - gx0) / c).astype(np.int32), 0, nx - 1)
    k = cls[ci, cj]
    out = np.full(X.shape + (3,), np.nan, 'f4')
    for cid, (tl, tile_m, M, tint) in tiles.items():
        sel = k == cid
        if not sel.any():
            continue
        n = tl.shape[0]
        u = (np.floor(X[sel] / tile_m * n).astype(np.int64)) % n
        v = (np.floor(-Y[sel] / tile_m * n).astype(np.int64)) % n
        tw = np.asarray(tint.at(X[sel], Y[sel]), 'f4')
        raw = tw / np.maximum(M, 1e-4)[None, :]
        if lift is not None:
            raw = lift(np.clip(raw, 0.03, 1.0), X[sel], Y[sel], np.broadcast_to(M, raw.shape))
        out[sel] = tl[v, u] * raw
    return out


def photo_match(src, dst, box, page_m, tint, valid=None, cfg=None, quality=95, feather=None,
                drawn=None):
    """ Re-level one photographed ground page onto `tint` (TintMap, linear
    RGB, the colour a class-tile face at the same XY is drawn with).

    1. low frequency: gain = lowpass(tint) / lowpass(page), a normalised Gaussian of
       sigma_m over the page's valid texels on a cell_m grid, log-clamped (luma
       +-lum_max, chroma +-chroma_max), so the photo meets the tile it hands over to
       with no step and keeps its own detail below sigma_m;
    2. per texel: its chroma (log RGB minus log luma) may sit at most
       texel_chroma_max off the tint's chroma; the luma is kept.
    drawn: optional f(X, Y, tint_rgb) -> the colour a class tile is drawn with there
    (GROUND_PHOTO_MATCH level_to='drawn'); the page is levelled onto that instead.
    Returns a small stats dict."""
    from PIL import Image
    from scipy import ndimage
    Image.MAX_IMAGE_PIXELS = None
    cfg = cfg or GROUND_PHOTO_MATCH
    u8 = np.asarray(Image.open(src).convert('RGB'))
    px = u8.shape[0]
    lin = T.srgb_to_linear(u8).astype('f4')
    if valid is None or valid.shape != u8.shape[:2]:
        valid = np.ones(u8.shape[:2], bool)
    k = max(1, int(round(cfg['cell_m'] / (page_m / px))))
    nb = px // k
    wv = valid[:nb * k, :nb * k].reshape(nb, k, nb, k).astype('f4')
    den = wv.sum((1, 3))
    tex = page_m / px
    cc = box[0] + (np.arange(nb) + 0.5) * k * tex
    rr = box[1] + (np.arange(nb) + 0.5) * k * tex
    X, Y = np.meshgrid(cc, rr)
    Tc = np.asarray(tint.at(X.ravel(), Y.ravel()), 'f4')
    if drawn is not None:
        # level onto the tile as drawn (after the vertex-tint floors), not the raster
        Tc = np.asarray(drawn(X.ravel(), Y.ravel(), Tc), 'f4')
    Tc = Tc.reshape(nb, nb, 3)
    sig = cfg['sigma_m'] / (k * tex)
    g0 = ndimage.gaussian_filter(den, sig, mode='nearest')
    ok = g0 > 1e-3 * k * k
    gain = np.zeros((nb, nb, 3), 'f4')
    for c in range(3):
        sp = (lin[:nb * k, :nb * k, c].reshape(nb, k, nb, k) * wv).sum((1, 3))
        pl = ndimage.gaussian_filter(sp, sig, mode='nearest') / np.maximum(g0, 1e-6)
        tl = ndimage.gaussian_filter(Tc[..., c] * den, sig, mode='nearest') / np.maximum(g0, 1e-6)
        gain[..., c] = np.log(np.maximum(tl, 1e-4)) - np.log(np.maximum(pl, 1e-4))
    gain[~ok] = 0.0
    LW = T.LUMW.astype('f4')
    gl = np.clip(gain @ LW, -cfg['lum_max'], cfg['lum_max'])
    gc = np.clip(gain - (gain @ LW)[..., None], -cfg['chroma_max'], cfg['chroma_max'])
    gain = gl[..., None] + gc
    z = px / nb
    G = np.stack([ndimage.zoom(gain[..., c], z, order=1, mode='nearest', grid_mode=True)
                  for c in range(3)], -1)[:px, :px]
    if G.shape[:2] != (px, px):
        G = np.pad(G, ((0, px - G.shape[0]), (0, px - G.shape[1]), (0, 0)), mode='edge')
    out = lin * np.exp(G)
    # per-texel chroma sanity against the tint
    Tt = np.stack([ndimage.zoom(Tc[..., c], z, order=1, mode='nearest', grid_mode=True)
                   for c in range(3)], -1)[:px, :px]
    if Tt.shape[:2] != (px, px):
        Tt = np.pad(Tt, ((0, px - Tt.shape[0]), (0, px - Tt.shape[1]), (0, 0)), mode='edge')
    lo = np.log(np.maximum(out, 1e-4)); lt = np.log(np.maximum(Tt, 1e-4))
    co = lo - (lo @ LW)[..., None]; ct = lt - (lt @ LW)[..., None]
    dc = co - ct; cm = cfg['texel_chroma_max']
    clamped = (np.abs(dc) > cm).any(-1)
    Y0 = out @ LW
    o2 = np.exp(ct + np.clip(dc, -cm, cm))
    o2 = o2 * (Y0 / np.maximum(o2 @ LW, 1e-6))[..., None]
    out = np.clip(o2, 0.0, 1.0)
    fstat = {}
    if feather is not None and cfg.get('feather_m', 0) > 0 and valid.any() and not valid.all():
        # fade into the class tile's own look near the photographed area's edge
        d = ndimage.distance_transform_edt(valid) * tex
        wf = np.clip(d / cfg['feather_m'], 0.0, 1.0)
        wf = (wf * wf * (3.0 - 2.0 * wf)).astype('f4')
        band_ = valid & (wf < 1.0)
        if band_.any():
            ii, jj = np.nonzero(band_)
            Xb = (box[0] + (jj + 0.5) * tex).astype('f8'); Yb = (box[1] + (ii + 0.5) * tex).astype('f8')
            tlk = _feather_tile_look(Xb, Yb, feather['grid'], feather['cls'], feather['tiles'](tex),
                                     lift=feather.get('lift'))
            ok_ = np.isfinite(tlk).all(1)
            wb = wf[ii, jj][:, None]
            cur = out[ii, jj]
            out[ii[ok_], jj[ok_]] = np.clip(wb[ok_] * cur[ok_] + (1 - wb[ok_]) * tlk[ok_], 0, 1)
            fstat = dict(feather_texels=int(ok_.sum()), feather_frac_valid=float(ok_.sum() / valid.sum()))
    res = T.linear_to_srgb(out)
    Image.fromarray(res).save(dst, 'JPEG', quality=quality, optimize=True)
    def _mag(a):
        a = a.astype('i4'); return int((((a[..., 0] - a[..., 1]) > 12) & ((a[..., 2] - a[..., 1]) > 12) & valid).sum())
    return dict(lum_gain_p50=float(np.median(np.exp(gl[ok]))) if ok.any() else 1.0,
                lum_gain_p90dev=float(np.percentile(np.abs(gl[ok]), 90)) if ok.any() else 0.0,
                texel_chroma_clamped=float(clamped[valid].mean()) if valid.any() else 0.0,
                magenta_before=_mag(u8), magenta_after=_mag(res), **fstat)


def src_tag(path):
    """Short fingerprint of a source file (size + mtime_ns) for cache names.

    The --texscale caches below are keyed on the source, not only on page and
    size (facade_atlas_00_2048.jpg) with a hit taken on os.path.exists: s62
    swaps a new wall set into facades/ and s35 re-packs it, so the pages change
    content and LAYOUT under the same name, and a name-only cache would keep
    embedding the stale resized pages while the UVs come from the new
    facadetex.json.  model.glb (texscale 1.0) embeds the source directly.
    Keying the name on the source makes a re-pack a cache miss."""
    st = os.stat(path)
    return hashlib.sha1(f"{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:8]


GROUND_SUPERTILE = getattr(C, 'GROUND_SUPERTILE', None) or {}


def ground_period(mat):
    """Metres per repeat of a class's GROUND-use tile (the super tile when configured)."""
    st = GROUND_SUPERTILE.get(mat)
    return float(st['period_m']) if st else TILE_M[mat]


def _hex_lattice(n, period_m, hex_m, seed):
    """Per output texel of an n x n tile over period_m: the 3 triangle-lattice vertex
    ids and barycentric weights of a lattice that wraps exactly at period_m."""
    nx = max(3, int(round(period_m / hex_m)))
    ny = max(4, 2 * int(round(period_m / (hex_m * 0.8660254) / 2)))
    hx = period_m / nx; hy = period_m / ny
    return nx, ny, hx, hy


def hex_bake(src, src_m, period_m, n, hex_m, seed, normal=False, rows=512):
    """Bake a period_m x period_m tile (n x n texels) from a tileable source image
    (float array, src_m metres across) by hex-tile blending: every vertex of a
    periodic triangle lattice (spacing ~hex_m) takes a random offset and rotation
    of the source; each texel blends the 3 samples of its triangle.  Colour blends
    in a PCA-decorrelated, Gaussianised space with the variance-preserving operator
    of Heitz and Neyret (histogram preserved); normals blend with sharpened
    weights, their XY rotated with the sample, and renormalise."""
    from scipy import ndimage as _nd
    from scipy.special import ndtri
    H0 = src.shape[0]
    nx, ny, hx, hy = _hex_lattice(n, period_m, hex_m, seed)
    rng = np.random.default_rng(seed)
    key_n = 2 * nx * ny
    offs = rng.uniform(0, src_m, (key_n, 2)); ang = rng.uniform(0, 2 * np.pi, key_n)
    if not normal:
        flat = src.reshape(-1, 3).astype('f8'); mu = flat.mean(0)
        ev, V = np.linalg.eigh(np.cov((flat - mu).T)); V = V[:, ::-1]
        P = ((flat - mu) @ V).reshape(src.shape).astype('f4')
        qs = np.linspace(0, 1, 4097)[1:-1]
        tabs = []; G = np.empty_like(P)
        for c in range(3):
            srt = np.sort(P[..., c].ravel()); r = np.searchsorted(srt, P[..., c].ravel())
            G[..., c] = ndtri(np.clip((r + 0.5) / srt.size, 1e-6, 1 - 1e-6)).reshape(P.shape[:2])
            tabs.append(np.quantile(srt, qs))
        chans = [G[..., c] for c in range(3)]
    else:
        chans = [src[..., c] for c in range(3)]
    out = np.empty((n, n, 3), 'f4'); tex = period_m / n; spx = H0 / src_m
    for r0 in range(0, n, rows):
        yy, xx = np.mgrid[r0:min(n, r0 + rows), 0:n].astype('f8')
        X = (xx + 0.5) * tex; Y = (yy + 0.5) * tex
        t = Y / hy; s = X / hx - t / 2.0
        i0 = np.floor(s); j0 = np.floor(t); fs = s - i0; ft = t - j0
        up = (fs + ft) >= 1.0
        # triangle vertices in lattice coords and barycentrics
        vi = [np.where(up, i0 + 1, i0), i0 + 1, i0]; vj = [np.where(up, j0 + 1, j0), j0, j0 + 1]
        w = [np.where(up, fs + ft - 1.0, 1.0 - fs - ft), np.where(up, 1.0 - ft, fs), np.where(up, 1.0 - fs, ft)]
        acc = np.zeros(X.shape + (3,)); wsq = np.zeros(X.shape)
        for k in range(3):
            I = vi[k]; J = vj[k]
            key = (np.mod(2 * I + J, 2 * nx) * ny + np.mod(J, ny)).astype(np.int64)
            xv = (I + J / 2.0) * hx; yv = J * hy
            dx = X - xv; dy = Y - yv; a = ang[key]; ca = np.cos(a); sa = np.sin(a)
            u = (ca * dx - sa * dy + offs[key, 0]) * spx; v = (sa * dx + ca * dy + offs[key, 1]) * spx
            wk = w[k] if not normal else w[k] ** 3
            smp = np.stack([_nd.map_coordinates(ch, [v.ravel(), u.ravel()], order=1, mode='grid-wrap').reshape(X.shape)
                            for ch in chans], -1)
            if normal:
                nxv = smp[..., 0] * 2 - 1; nyv = smp[..., 1] * 2 - 1
                smp[..., 0] = ca * nxv - sa * nyv; smp[..., 1] = sa * nxv + ca * nyv; smp[..., 2] = smp[..., 2] * 2 - 1
            acc += wk[..., None] * smp; wsq += wk ** 2 if not normal else wk
        if normal:
            nv = acc / np.maximum(wsq, 1e-9)[..., None]
            nv /= np.maximum(np.linalg.norm(nv, axis=-1, keepdims=True), 1e-6)
            out[r0:r0 + len(yy)] = (nv * 0.5 + 0.5).astype('f4')
        else:
            g = acc / np.sqrt(np.maximum(wsq, 1e-9))[..., None]
            from scipy.special import ndtr
            pv = np.clip(ndtr(g), qs[0], qs[-1])
            Pc = np.stack([np.interp(pv[..., c], qs, tabs[c]) for c in range(3)], -1)
            out[r0:r0 + len(yy)] = (Pc @ V.T + mu).astype('f4')
    return out


def supertile_bake(mat, st, lin_src_path, M, px_scale=1.0):
    """The ground-use super tile of class `mat` (GROUND_SUPERTILE entry st): colour jpg
    path and normal jpg path (or None), both cached on the config."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    n = max(512, int(st['px'] * px_scale)); P = float(st['period_m']); sm = float(st['src_m'])
    src = st.get('src') or {}
    tag = hashlib.md5((repr(sorted((k, v) for k, v in st.items() if k != 'src')) + repr(sorted(src.items()))
                       + lin_src_path + repr(np.round(np.asarray(M, 'f8'), 5).tolist()) + '|hx3').encode()).hexdigest()[:8]
    fc = f"{TEXOUT}/{mat}_super_{n}_{tag}.jpg"; fn = f"{TEXOUT}/{mat}_super_nrm_{tag}.jpg"
    nsrc = src.get('normal')
    if os.path.exists(fc) and (not nsrc or os.path.exists(fn)):
        return fc, (fn if nsrc else None)
    spx = max(64, int(round(n * sm / P)))
    im = Image.open(src.get('color') or lin_src_path).convert('RGB').resize((spx, spx), Image.LANCZOS)
    lin = np.power(np.asarray(im, 'f4') / 255.0, 2.2)
    kc = st.get('keep_chroma')
    if st.get('chroma_cov') is not None:
        # colour the (log luma, log R/G, log B/G) statistics onto a measured reference
        # (lower Cholesky with luma first, reference rescaled to the source's luma
        # variance: luma is untouched, chroma takes the reference covariance and its
        # correlation with luma); kc is ignored
        l = np.maximum(lin.reshape(-1, 3).astype('f8'), 1e-4)
        X = np.stack([np.log(l @ T.LUMW), np.log(l[:, 0] / l[:, 1]), np.log(l[:, 2] / l[:, 1])], 1)
        mu = X.mean(0); Cs = np.cov((X - mu).T); Cr = np.asarray(st['chroma_cov'], 'f8')
        Cr = Cr * np.sqrt(Cs[0, 0] / Cr[0, 0]) ** np.array([[2, 1, 1], [1, 0, 0], [1, 0, 0]])
        Xn = (X - mu) @ (np.linalg.cholesky(Cr) @ np.linalg.inv(np.linalg.cholesky(Cs))).T + mu
        Lg, rg, bg = np.exp(Xn[:, 0]), np.exp(Xn[:, 1]), np.exp(Xn[:, 2])
        W = np.asarray(T.LUMW, 'f8'); Gc = Lg / (W[0] * rg + W[1] + W[2] * bg)
        lin = np.stack([rg * Gc, Gc, bg * Gc], 1).reshape(lin.shape).astype('f4')
        kc = 1.0
    elif kc is not None:
        g = lin @ T.LUMW; lin = kc * lin + (1 - kc) * g[:, :, None]
    b = np.clip(hex_bake(lin, sm, P, n, float(st['hex_m']), int(st['seed'])), 0, 1)
    if kc is None:
        y = b @ T.LUMW; b = (y / max(float(y.mean()), 1e-4))[:, :, None] * np.asarray(M, 'f4')[None, None, :]
    else:
        b = b * (np.asarray(M, 'f4') / np.maximum(b.reshape(-1, 3).mean(0), 1e-4))[None, None, :]
    Image.fromarray(np.clip(np.power(np.clip(b, 0, 1), 1 / 2.2) * 255, 0, 255).astype('u1')).save(fc, 'JPEG', quality=92, optimize=True)
    if nsrc:
        nn = max(256, int(st.get('normal_px', n // 2) * px_scale)); spn = max(64, int(round(nn * sm / P)))
        ns = np.asarray(Image.open(nsrc).convert('RGB').resize((spn, spn), Image.LANCZOS), 'f4') / 255.0
        nb = hex_bake(ns, sm, P, nn, float(st['hex_m']), int(st['seed']), normal=True)
        Image.fromarray(np.clip(nb * 255, 0, 255).astype('u1')).save(fn, 'JPEG', quality=90, optimize=True)
    return fc, (fn if nsrc else None)


def prep_textures(targets, texscale=1.0, normalmaps=True, classdetail=True):
    """Normalise each CC0 colour map so its mean is the 97th percentile of the
    colours that material has to reach, which keeps every tint at or below 1.0
    (glTF COLOR_0 can only darken).  Returns the per-material mean.

    When `classdetail` is on, every CC0 tile also takes the CLASS_DETAIL
    multiply (see class_detail()).  The per-tile mean linear shift the multiply
    introduces is left on prep_textures.detail_shift, keyed by material, for the
    stats block; the tile is then renormalised to M, as it is when CLASS_DETAIL
    is off, so the shift is what the multiply DID, not what survives into the
    model."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    idx = json.load(open(CC0 + '/index.json'))
    means = {}
    files = {}
    prep_textures.detail_shift = {}
    for mat in MATERIALS:
        px = max(256, int(TEX_PX[mat] * texscale))
        tgt = targets.get(mat)
        if tgt is None:
            tgt = np.array([0.35, 0.35, 0.35], 'f4')
        M = np.clip(tgt, 0.04, 0.88)
        means[mat] = M
        ent = idx['materials'].get(mat)
        if ent is None and mat == 'water':
            wc, wn, wm = water_textures(px, M)
            files[mat] = dict(color=wc, normal=(wn if normalmaps else None),
                              mr=wm, asset=None)
            continue
        if ent is None:
            files[mat] = None
            continue
        im = Image.open(ent['files']['color']).convert('RGB')
        if im.size[0] != px:
            im = im.resize((px, int(round(im.size[1] * px / im.size[0]))), Image.LANCZOS)
        a = np.asarray(im, 'f4') / 255.0
        lin = np.power(a, 2.2)
        g = lin @ T.LUMW
        k = KEEP_CHROMA[mat]
        lin = k * lin + (1 - k) * g[:, :, None]          # partly desaturate
        # How much of the tile's own high-frequency contrast to keep (HF_GAIN;
        # a class not listed there, lawn included, keeps all of it).
        hf = HF_GAIN.get(mat, 1.0)
        if hf != 1.0:
            from scipy import ndimage as _nd
            lo = np.stack([_nd.uniform_filter(lin[:, :, c], 9) for c in range(3)], -1)
            lin = np.clip(lo + (lin - lo) * hf, 0.0, 1.0)
        md = macro_detail(mat, lin.shape[0], lin.shape[1], TILE_M[mat])
        if md is not None:
            lin = np.clip(lin * md[:, :, None], 0.0, 1.0)
        # CLASS_DETAIL: the aggregate-scale relief the CC0 tile does not carry.
        # It goes in here, before the renormalisation to M, so the tile that
        # reaches the model still has exactly the mean the photographs measured
        # -- the shift recorded below is the move the multiply introduced, which
        # is the number the <0.01 promise is about, and the renormalise then
        # takes even that back out.
        # ... and it is baked once per USE (see CLASS_DETAIL_USE): the same
        # tile carries a wall and the ground beside it, and the amplitude that
        # is relief on the ground is mottling on the wall.
        cols = {}
        for use in (CLASS_DETAIL_USE if classdetail else {'ground': None}):
            # see CLASS_DETAIL_WALL_CLASSES: only the classes with a measured
            # improvement carry a second bake; the rest reuse the ground tile.
            if use != 'ground' and mat not in CLASS_DETAIL_WALL_CLASSES:
                continue
            lu = lin
            if classdetail:
                _p0 = lin.reshape(-1, 3).mean(0)
                _cfg = CLASS_DETAIL_USE[use]
                if use == 'ground' and mat in GROUND_DETAIL_AMP:
                    _cfg = dict(_cfg, amp=float(GROUND_DETAIL_AMP[mat]))
                cd = class_detail(lin.shape[0], lin.shape[1], TILE_M[mat],
                                  cfg=_cfg,
                                  hipass_m=CLASS_DETAIL_HIPASS_M.get(mat))
                lu = np.clip(lin * cd[:, :, None], 0.0, 1.0)
                _p1 = lu.reshape(-1, 3).mean(0)
                prep_textures.detail_shift.setdefault(use, {})[mat] = float(
                    np.abs(_p1 / np.maximum(_p0, 1e-6) - 1.0).max())
            m0 = lu.reshape(-1, 3).mean(0)
            if mat in LUMA_ONLY:
                _y = lu @ T.LUMW
                lu = (_y / max(float(_y.mean()), 1e-4))[:, :, None] * M[None, None, :]
            else:
                lu = lu * (M / np.maximum(m0, 1e-4))[None, None, :]
            out = np.clip(np.power(np.clip(lu, 0, 1), 1 / 2.2) * 255, 0, 255).astype('u1')
            fp = (f"{TEXOUT}/{mat}_color_{px}"
                  f"{CDET_TAG[use] if classdetail else ''}.jpg")
            Image.fromarray(out).save(fp, 'JPEG', quality=90, optimize=True)
            cols[use] = fp
        fp = cols['ground']
        nfp = None
        if normalmaps and ent['files'].get('normal'):
            nim = Image.open(ent['files']['normal']).convert('RGB')
            if nim.size[0] != NORMAL_PX:
                nim = nim.resize((NORMAL_PX, int(round(nim.size[1] * NORMAL_PX / nim.size[0]))),
                                 Image.LANCZOS)
            nfp = f"{TEXOUT}/{mat}_normal_{NORMAL_PX}.jpg"
            nim.save(nfp, 'JPEG', quality=88, optimize=True)
        nby = {}
        st = GROUND_SUPERTILE.get(mat)
        if st and st.get('mode') == 'hex':
            # (common.py GROUND_SUPERTILE): the ground use draws a baked
            # hex-blended super tile; the walls keep the plain tile at TILE_M
            cols.setdefault('wall', cols['ground'])
            nby['wall'] = nfp
            s_src = dict(st.get('src') or {})
            if not s_src.get('normal') and ent['files'].get('normal') and normalmaps:
                s_src['normal'] = ent['files']['normal']
            if not normalmaps:
                s_src.pop('normal', None)
            fcs, fns = supertile_bake(mat, dict(st, src=s_src), ent['files']['color'], M, texscale)
            cols['ground'] = fp = fcs
            nby['ground'] = fns
            if st.get('src'):
                ent = dict(ent, **{k: v for k, v in st['src'].items() if k not in ('color', 'normal')})
        files[mat] = dict(color=fp, color_by_use=cols, normal=nfp, normal_by_use=nby, asset=ent)
    for mat, ref in TILE_CONTRAST.items():        # per-area knob only (empty unless the area sets it)
        f = files.get(mat)
        if f and f.get('color_by_use', {}).get('ground'):
            nfp = tile_contrast(f['color_by_use']['ground'], mat, means[mat], ref, ground_period(mat), _GROUND_TC)
            if f['color'] == f['color_by_use']['ground']:
                f['color'] = nfp
            f['color_by_use']['ground'] = nfp
    return means, files


def water_knee_lum(lum):
    """Compress lifted water luminance above the crest.

    Below lum(WATER_CREST) * WATER_LIFT * WATER_KNEE_X the value is returned
    unchanged; above it only WATER_KNEE_SLOPE of the excess is kept.  The comment
    above WATER_KNEE_X gives the evidence that the bright water texels are baked
    reflections."""
    lum = np.asarray(lum, 'f8')
    k = float(WATER_CREST.astype('f8') @ T.LUMW.astype('f8')) * WATER_LIFT * WATER_KNEE_X
    return np.where(lum > k, k + (lum - k) * WATER_KNEE_SLOPE, lum)


# The knee is global, so reflection patches (20-70 m across) that sit under the
# crest keep their contrast against the dark water round them (about 1.6x the
# water median after the knee).  They are a LOW-FREQUENCY excess: the 12 m local
# median rises inside them.  So the
# local median (12 m window, 1 m cells, over the water tint raster) is knee'd
# against the water-wide median, and every vertex keeps its own ratio to its
# local median: the patch comes down as a whole, the texture inside and
# outside it survives.  Rendered, this cuts the patch contrast to about 1.2-1.3x
# with the water median and the texture away from the patches unchanged.
WATER_LOCAL_WIN = 12.0        # m, window of the local median
# The knee sits at 0.80 x the water-wide median: the raster-wide median sits
# above most of the water (open water is at the ~0.025 floor, the median near
# 0.0345), so a knee at the median would miss the 1.3-1.5x low-frequency patches
# that hold most of the brightest vertices.  Rendered, this cuts bright-patch
# pixels (over 1.25x the view's water median, in patches of 30 px or more) by
# roughly half with the water median unchanged.
WATER_LOCAL_T = 0.80          # knee at this multiple of the water-wide median
WATER_LOCAL_SLOPE = 0.1       # gain kept on the local median above the knee
_WLOC = {}
# A narrow bright streak on open water (about 8 m wide, lum 0.09-0.13 against a
# 0.025 floor) sits above the water-wide median knee and is too narrow for the
# 12 m local median, so neither takes it down.  So each vertex
# is also held to a ratio of its local BACKGROUND -- a low percentile over a
# window wider than the streak -- with the excess ratio knee'd.  Texture
# at or under WATER_BG_R times the background is untouched; gradients wider
# than the window move the background with them.
WATER_BG_WIN = float(os.environ.get('WATER_BG_WIN', '32'))    # m
WATER_BG_PCT = float(os.environ.get('WATER_BG_PCT', '25'))    # percentile
WATER_BG_R = float(os.environ.get('WATER_BG_R', '1.15'))      # knee ratio
WATER_BG_SLOPE = float(os.environ.get('WATER_BG_SLOPE', '0.10'))  # excess ratio kept
WATER_BG_ON = os.environ.get('WATER_BG', '1') == '1'
_WBG = {}


def _water_bg(tsrc):
    key = id(tsrc)
    if key not in _WBG:
        from scipy.ndimage import percentile_filter
        med, x0, y0, c, g = _water_local_med(tsrc)
        L = np.asarray(tsrc.M, 'f4') @ T.LUMW.astype('f4')
        f = 4
        ny, nx = (L.shape[0] // f) * f, (L.shape[1] // f) * f
        Lc = L[:ny, :nx].reshape(ny // f, f, nx // f, f).mean((1, 3))
        ok = Lc > 1e-4
        w = max(3, int(round(WATER_BG_WIN / c)))
        bg = percentile_filter(np.where(ok, Lc, g), WATER_BG_PCT, size=w,
                               mode='nearest')
        _WBG[key] = np.maximum(bg, 1e-4)
    return _WBG[key]


def _water_local_med(tsrc):
    key = id(tsrc)
    if key not in _WLOC:
        from scipy.ndimage import median_filter
        L = np.asarray(tsrc.M, 'f4') @ T.LUMW.astype('f4')
        f = 4                                    # 0.25 m -> 1 m cells
        ny, nx = (L.shape[0] // f) * f, (L.shape[1] // f) * f
        Lc = L[:ny, :nx].reshape(ny // f, f, nx // f, f).mean((1, 3))
        ok = Lc > 1e-4
        g = float(np.median(Lc[ok])) if ok.any() else 0.1
        w = max(3, int(round(WATER_LOCAL_WIN / (tsrc.c * f))))
        med = median_filter(np.where(ok, Lc, g), size=w, mode='nearest')
        _WLOC[key] = (med, tsrc.x0, tsrc.y0, tsrc.c * f, g)
    return _WLOC[key]


def water_local(cw, pos, tsrc):
    """Knee the water's local median.  cw: lifted (n,3) water
    colour at ENU pos; returns it rescaled in luminance only."""
    med, x0, y0, c, g = _water_local_med(tsrc)
    i = np.clip(((pos[:, 1] - y0) / c).astype(int), 0, med.shape[0] - 1)
    j = np.clip(((pos[:, 0] - x0) / c).astype(int), 0, med.shape[1] - 1)
    m = med[i, j].astype('f8')
    k = g * WATER_LOCAL_T
    m2 = np.where(m > k, k + (m - k) * WATER_LOCAL_SLOPE, m)
    s = m2 / np.maximum(m, 1e-9)
    if WATER_BG_ON:
        # the vertex's raster lum after the median knee, as a ratio
        # of its local background; the lift cancels out of the ratio
        lv = (tsrc.at(pos[:, 0], pos[:, 1]).astype('f8') @ T.LUMW.astype('f8')) * s
        q = lv / _water_bg(tsrc)[i, j].astype('f8')
        R = WATER_BG_R
        q2 = np.where(q > R, R + (q - R) * WATER_BG_SLOPE, q)
        s = s * (q2 / np.maximum(q, 1e-9))
    return (cw * s[:, None]).astype('f4')


def water_chroma(cw, mix=WATER_CHROMA_MIX, knee=False):
    """Rebase water colours onto WATER_CHROMA, keeping each one's luminance.

    `cw` is (n,3) linear RGB.  The returned colour has exactly the same
    luminance (T.LUMW dot product) as the input, so the water lift
    measured is carried through unchanged; only the hue moves.

    knee=True (the per-vertex water tint only, never the material mean) first
    runs the luminance through water_knee_lum(), which removes the panorama
    reflections baked into the water tint."""
    cw = np.asarray(cw, 'f4')
    flat = cw.ndim == 1
    c = np.atleast_2d(cw)
    lum = (c.astype('f8') @ T.LUMW.astype('f8'))
    if knee:
        lum = water_knee_lum(lum)
    k = float(WATER_CHROMA.astype('f8') @ T.LUMW.astype('f8'))
    tgt = lum[:, None] * (WATER_CHROMA.astype('f8') / k)[None, :]
    out = ((1.0 - mix) * c.astype('f8') + mix * tgt).astype('f4')
    return out[0] if flat else out


def _poskey(p):
    """The welding key emit() uses: a position rounded to 0.1 mm."""
    q = np.round(np.asarray(p, 'f8') / 1e-4).astype(np.int64)
    return q[:, 0] * 1000003 + q[:, 1] * 1009 + q[:, 2]


def skirt_wall_weight(z, ztop, zbot, zground, hold=None):
    """Weight of the WALL colour at height z in a skirt band.

    1.0 over the band's top `hold` metres (so the wall base is not a step),
    0.0 at and below the floor line `zground` (so the band meets the ground on
    the ground's own colour), smoothstep between.  A band
    with less than `hold` of itself above the floor line ramps across whatever
    it has."""
    if hold is None:
        hold = SKIRT_BLEND_M
    z = np.asarray(z, 'f8')
    ztop = np.asarray(ztop, 'f8'); zbot = np.asarray(zbot, 'f8')
    zg = np.clip(np.asarray(zground, 'f8'), zbot, ztop)
    hi = ztop - hold                      # at and above this: pure wall
    span = hi - zg
    t = np.where(span > 1e-6,
                 np.clip((z - zg) / np.maximum(span, 1e-6), 0.0, 1.0),
                 np.where(z >= ztop - 1e-6, 1.0,
                          np.clip((z - zg) / np.maximum(ztop - zg, 1e-6), 0.0, 1.0)))
    w = t * t * (3.0 - 2.0 * t)
    return np.where(z >= hi, 1.0, w).astype('f4')


def make_skirt_blend(spos, sidx, srows, db, floor_raster=None):
    """Colour the building skirt band so neither of its edges is a line.

    The band is a vertical strip hung off a wall's own bottom edge and carried
    down to 1 m under the floor beside it.  Painting the band with the ground
    raster over its whole height makes the wall base a ruler-straight step.

    Here the band is given a vertical gradient: its top SKIRT_BLEND_M metres
    carry the measured colour of the wall face the band hangs from, everything
    at or below the floor line carries the ground raster unchanged, and
    the two are blended across what is between.  As delivered the band is one
    quad per metre of wall with vertices only at its top and bottom edge, and
    COLOR_0 is interpolated linearly between them, so the band is SUBDIVIDED
    here -- rows of vertices at the two ends of the gradient -- otherwise the
    wall colour would smear over the whole band (often metres tall) and
    the ground line would land mid-ramp.  The split is in the band's own
    parameter, so two quads that share a vertical edge cut it at the same
    heights and no crack opens.

    Returns None, or dict(pos, idx, src, blend, report): `src` is the stitch
    row each new triangle came from, `blend` is f(pos, cw) -> cw for emit()."""
    make_skirt_blend.report = None
    if len(srows) < 1:
        return None
    if db is None:
        make_skirt_blend.report = 'skipped: no buildings layer'
        return None
    bz = np.load(C.W + '/layers/buildings.npz')
    bpos = bz['pos'].astype('f4'); bidx = bz['idx'].astype('i8')
    # every live wall face, not only the ones a panorama saw: fill_and_smooth
    # has already given the rest a neighbour or class-mean colour, and it is
    # that same colour the wall face itself is tinted with, so matching it is
    # what makes the band continue the wall
    wall = (~db['kill']) & (db['up'] < 0.5)
    if wall.sum() < 8 or len(bidx) != len(db['col']):
        make_skirt_blend.report = 'skipped: no measured wall faces'
        return None
    wi = np.nonzero(wall)[0]
    # Every wall face's edges, sampled every SKIRT_WALL_STEP metres and
    # carrying that face's colour.  Not the face vertices: an OSM mass is a
    # handful of large quads, so a wall's own corners are a measured median
    # 11.5 m from a band vertex, while the band is cut into 1 m quads
    # along the wall's bottom edge.  Sampling the edge puts a colour source
    # within half a step of every one of them.
    wt = bidx[wi]
    E = np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]], wt[:, [2, 0]]], 0)
    ef = np.concatenate([wi, wi, wi], 0)
    p0 = bpos[E[:, 0]].astype('f8'); p1 = bpos[E[:, 1]].astype('f8')
    ns = np.clip(np.ceil(np.linalg.norm(p1 - p0, axis=1) / SKIRT_WALL_STEP),
                 1, 600).astype(np.int64)
    cnt = ns + 1
    er = np.repeat(np.arange(len(E)), cnt)
    off = np.cumsum(cnt) - cnt
    t = (np.arange(int(cnt.sum())) - off[er]) / ns[er]
    wpts = p0[er] + (p1[er] - p0[er]) * t[:, None]
    wcol = db['col'][ef[er]].astype('f4')
    from scipy.spatial import cKDTree
    kd = cKDTree(wpts)

    tri = sidx[srows]
    P = spos[tri].astype('f8')
    z = P[:, :, 2]
    zt = z.max(1); zb = z.min(1)
    H = zt - zb
    istop = z >= (zt[:, None] - 1e-4)
    ntop = istop.sum(1)
    ok = (H > 0.05) & (ntop >= 1) & (ntop <= 2)
    # the floor line: by construction the band ends SKIRT_BURIED_M under the
    # lowest floor along its wall, and where the 0.5 m floor raster has
    # a height at the band's own XY that is used instead, since it is per
    # column where the construction rule is per wall edge.
    zg = zb + SKIRT_BURIED_M
    n_raster = 0
    if floor_raster is not None:
        Z, g = floor_raster
        x0, y0, cell, nx, ny = g
        cx = P[:, :, 0].mean(1); cy = P[:, :, 1].mean(1)
        ix = np.clip(np.round((cx - x0) / cell).astype(int), 0, int(nx) - 1)
        iy = np.clip(np.round((cy - y0) / cell).astype(int), 0, int(ny) - 1)
        zr = Z[iy, ix].astype('f8')
        use = np.isfinite(zr) & (zr > zb + 0.05) & (zr < zt - 0.05)
        zg = np.where(use, zr, zg)
        n_raster = int(use.sum())
    zg = np.clip(zg, zb, zt)

    # ---- subdivide: rows of vertices at the two ends of the gradient ----- #
    order = np.argsort(~istop, axis=1, kind='stable')
    V = np.take_along_axis(P, order[:, :, None], axis=1)
    two = (ntop == 2)
    A_top = V[:, 0]; B_bot = V[:, 2]
    A_bot = np.where(two[:, None], V[:, 2], V[:, 1])
    B_top = np.where(two[:, None], V[:, 1], V[:, 0])
    # band parameter (0 at the band bottom, 1 at the wall base) of each row
    sh = np.clip(1.0 - SKIRT_BLEND_M / np.maximum(H, 1e-6), 0.0, 1.0)
    sg = np.clip((zg - zb) / np.maximum(H, 1e-6), 0.0, 1.0)
    sh = np.maximum(sh, sg)
    lev = np.stack([np.ones_like(sh), sh, sg, np.zeros_like(sh)], 1)
    nlev = lev.shape[1]
    A = A_bot[:, None, :] + (A_top - A_bot)[:, None, :] * lev[:, :, None]
    B = B_bot[:, None, :] + (B_top - B_bot)[:, None, :] * lev[:, :, None]
    m = int(ok.sum())
    sel = np.nonzero(ok)[0]
    A = A[sel]; B = B[sel]; lev = lev[sel]
    # vertices: A rows then B rows, per source triangle
    pos2 = np.concatenate([A, B], 1).reshape(-1, 3)            # (m*2*nlev, 3)
    base = np.arange(m) * (2 * nlev)
    ia = base[:, None] + np.arange(nlev)[None, :]
    ib = base[:, None] + nlev + np.arange(nlev)[None, :]
    faces = []
    for k in range(nlev - 1):
        faces.append(np.stack([ia[:, k], ib[:, k], ib[:, k + 1]], 1))
        faces.append(np.stack([ia[:, k], ib[:, k + 1], ia[:, k + 1]], 1))
    idx2 = np.concatenate(faces, 0)
    loc2 = np.tile(np.arange(m), 2 * (nlev - 1))       # index into the split set
    src2 = np.tile(srows[sel], 2 * (nlev - 1))          # index into the stitch
    # the triangles that carry no band (a collapsed strip, or a row pair that
    # landed on the same height) are dropped
    T2 = pos2[idx2]
    ar = 0.5 * np.linalg.norm(np.cross(T2[:, 1] - T2[:, 0], T2[:, 2] - T2[:, 0]), axis=1)
    live = ar > 1e-7
    idx2 = idx2[live]; src2 = src2[live]; loc2 = loc2[live]
    # keep the original winding, so the shading normals do not flip
    n0 = np.cross(P[sel][:, 1] - P[sel][:, 0], P[sel][:, 2] - P[sel][:, 0])
    T2 = pos2[idx2]
    n2 = np.cross(T2[:, 1] - T2[:, 0], T2[:, 2] - T2[:, 0])
    flip = (n2 * n0[loc2]).sum(1) < 0
    idx2[flip] = idx2[flip][:, ::-1]

    # ---- the colour each new vertex takes ------------------------------- #
    zt2 = np.repeat(zt[sel], 2 * nlev)
    zb2 = np.repeat(zb[sel], 2 * nlev)
    zg2 = np.repeat(zg[sel], 2 * nlev)
    wv = skirt_wall_weight(pos2[:, 2], zt2, zb2, zg2)
    dist, ii = kd.query(pos2, k=1, distance_upper_bound=SKIRT_WALL_MAXD)
    found = np.isfinite(dist) & (ii < len(wpts))
    wv = np.where(found, wv, 0.0).astype('f4')
    wc = wcol[np.where(found, ii, 0)]
    key = _poskey(pos2)
    uk, first = np.unique(key, return_index=True)
    uw = wv[first]; uc = wc[first]
    rep = dict(source_triangles=int(len(tri)), triangles=int(len(idx2)),
               band_height_m=dict(p10=round(float(np.percentile(H[sel], 10)), 2),
                                  median=round(float(np.median(H[sel])), 2),
                                  p90=round(float(np.percentile(H[sel], 90)), 2)),
               wall_hold_m=SKIRT_BLEND_M, buried_m=SKIRT_BURIED_M,
               floor_line_from_raster=n_raster,
               wall_search_radius_m=SKIRT_WALL_MAXD,
               wall_samples=int(len(wpts)),
               vertices_with_a_wall_pct=round(100 * float(found.mean()), 1),
               vertices_taking_wall_colour_pct=round(100 * float((wv > 0).mean()), 1),
               rule="COLOR_0 = ground raster under the floor line, the wall "
                    "face's own measured colour over the top %.2f m, linear "
                    "between; band subdivided here to carry it" % SKIRT_BLEND_M)
    make_skirt_blend.report = rep

    def blend(pos, cw):
        k = _poskey(pos)
        j = np.clip(np.searchsorted(uk, k), 0, len(uk) - 1)
        hit = uk[j] == k
        w = np.where(hit, uw[j], 0.0).astype('f4')[:, None]
        return ((1.0 - w) * cw + w * uc[j]).astype('f4')

    return dict(pos=pos2.astype('f4'), idx=idx2.astype('i8'), src=src2,
                blend=blend, report=rep)


def water_textures(px, M):
    """The water surface, generated here: there is no CC0 water tile in the
    library, and a plain material (one flat colour, no texture, no normal)
    renders as a dark plane at eye level.

    A tileable wave height field (three octaves of band-limited periodic noise,
    stretched 1.5:1 across the prevailing wind fetch) drives all three maps:
    colour (the troughs keep the measured water colour, the crests lift toward
    the sky it reflects), a normal map (slopes under 6 degrees at 9 m per tile,
    which is what a light chop on open water is) and a metallic-roughness map (the
    crests are smoother, so they catch the specular and the troughs do not).
    The colour map is renormalised to the material's target mean M, exactly
    like every CC0 tile, so COLOR_0 still only darkens.

    The dominant octave is 12 cycles across the 9 m tile (0.75 m) at 1.5:1,
    i.e. 0.75 x 1.125 m; at 6 cycles (1.5 m) stretched 3:1 the ripple is a 1.5 x
    4.5 m corduroy that reads as a rolled metal sheet rather than chop.  The
    height amplitude is halved to hold the slope: measured over the tile, mean
    slope 4.14 deg / p95 8.18 deg, against 8.20 / 16.04 at full amplitude.

    The B channel of the MR map is flat 1.0 and metallic comes from
    metallicFactor; only roughness varies here.

    Returns (colour_path, normal_path, mr_path)."""
    from PIL import Image
    key = '%03d%03d%03d' % tuple(np.clip(np.asarray(M) * 999, 0, 999).astype(int))
    # the cached maps are named by a hash of this function's source and the colour
    # octaves, so an edit to the wave field, amplitude or MR channels never reuses a
    # stale map
    import inspect
    src = hashlib.sha1((inspect.getsource(water_textures) + repr(WATER_COLOUR_OCT)).encode()).hexdigest()[:8]
    cp = f"{TEXOUT}/water_color_{px}_{key}_{src}.jpg"
    np_ = f"{TEXOUT}/water_normal_{px}_{src}.jpg"
    mp = f"{TEXOUT}/water_mr_{px}_{src}.jpg"
    if os.path.exists(cp) and os.path.exists(np_) and os.path.exists(mp):
        return cp, np_, mp
    rng = np.random.default_rng(11)
    fy = np.fft.fftfreq(px)[:, None]
    fx = np.fft.fftfreq(px)[None, :]
    h = np.zeros((px, px), 'f8')
    hc = np.zeros((px, px), 'f8')          # the colour field: same octaves,
    for (f0, amp, stretch), cw in zip(     # fine-weighted
            ((12.0, 1.0, 1.5), (34.0, 0.45, 1.35), (88.0, 0.20, 1.2)),
            WATER_COLOUR_OCT):
        r = np.sqrt((fx * px * stretch) ** 2 + (fy * px) ** 2)
        band = np.exp(-((r - f0) / (f0 * 0.55)) ** 2)
        ph = np.exp(2j * np.pi * rng.random((px, px)))
        oct_ = np.real(np.fft.ifft2(band * ph)) / max(band.std(), 1e-9)
        h += amp * oct_
        hc += cw * oct_
    h = (h - h.mean()) / (h.std() + 1e-9)
    hc = (hc - hc.mean()) / (hc.std() + 1e-9)
    # 0 trough, 1 crest.  Two amplitudes: the colour map uses WATER_NOISE_AMP
    # (lower, so the water does not match the lawn's high-frequency noise) and
    # the roughness map WATER_PBR_AMP.
    hn = np.clip(hc * WATER_NOISE_AMP + 0.5, 0.0, 1.0)
    hn_pbr = np.clip(h * WATER_PBR_AMP + 0.5, 0.0, 1.0)
    # colour: the measured water, lifted toward its own sky reflection on crests
    trough, crest = WATER_TROUGH, WATER_CREST
    lin = trough[None, None, :] + (crest - trough)[None, None, :] * hn[:, :, None]
    lin = lin * (np.asarray(M, 'f4') / np.maximum(lin.reshape(-1, 3).mean(0), 1e-5))[None, None, :]
    Image.fromarray(np.clip(np.power(lin, 1 / 2.2) * 255, 0, 255).astype('u1')
                    ).save(cp, 'JPEG', quality=92, optimize=True)
    # normal from the same field: 9 m per tile, amplitude set for <6 deg slopes.
    # The amplitude scales with the wavelength: slope goes as amplitude over
    # wavelength, so 0.020 at this wavelength would double every slope.
    amp_m = 0.010
    dz = (2.0 * TILE_M['water'] / px)
    gy, gx = np.gradient(h * amp_m, dz)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    Image.fromarray(np.clip((n * 0.5 + 0.5) * 255, 0, 255).astype('u1')
                    ).save(np_, 'JPEG', quality=92, optimize=True)
    # glTF metallic-roughness: G = roughness, B = metallic.  Roughness keeps its
    # crest-to-trough variation; metallic is flat 1.0 so that the scalar
    # metallicFactor (0.04, PBR above) IS the metallic -- glTF multiplies the
    # two, and water is not a conductor.
    rough = (0.34 - 0.20 * hn_pbr)
    metal = np.ones_like(hn_pbr)
    mr = np.stack([np.zeros_like(hn_pbr), rough, metal], -1)
    Image.fromarray(np.clip(mr * 255, 0, 255).astype('u1')
                    ).save(mp, 'JPEG', quality=92, optimize=True)
    return cp, np_, mp


def foliage_texture(px=512):
    """A tileable leaf-cluster texture, generated here so the tree crowns are
    not flat green cylinders.  No external asset, so no licence question."""
    from PIL import Image
    fp = f"{TEXOUT}/foliage_color.jpg"
    if os.path.exists(fp):
        return fp
    rng = np.random.default_rng(4)
    a = np.zeros((px, px), 'f4')
    yy, xx = np.mgrid[0:px, 0:px]
    for r, n, amp in ((px * 0.055, 130, 1.0), (px * 0.028, 420, 0.7), (px * 0.014, 900, 0.5)):
        cx = rng.random(n) * px; cy = rng.random(n) * px
        for k in range(n):
            dx = np.minimum(np.abs(xx - cx[k]), px - np.abs(xx - cx[k]))
            dy = np.minimum(np.abs(yy - cy[k]), px - np.abs(yy - cy[k]))
            d2 = (dx / r) ** 2 + (dy / (r * 0.62)) ** 2
            a += amp * np.exp(-d2 * 2.2) * (0.5 + rng.random())
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    base = np.stack([0.058 + 0.055 * a, 0.105 + 0.085 * a,
                     0.042 + 0.038 * a], -1)
    out = np.clip(np.power(base, 1 / 2.2) * 255, 0, 255).astype('u1')
    Image.fromarray(out).save(fp, 'JPEG', quality=88, optimize=True)
    return fp


# --------------------------------------------------- colour fill/smooth --- #
def fill_and_smooth(cen, col, nviews, mat, mad=None, cell=0.75, fill_cell=3.0):
    """Low-pass the measured colour to the scale the poses support, and fill the
    faces no panorama reached from their own material's local mean.

    How hard the low-pass bites depends on how well the views agreed.  Where
    several panoramas saw a face and returned the same colour, the face keeps
    its own measured colour at mesh resolution -- that is signage,
    lettering and paving pattern surviving as detail.  Where they disagreed,
    the disagreement is pose error and gets smoothed away."""
    valid = nviews >= 1
    out = col.copy()
    def grid_mean(c, sel, cellsize, key_mat):
        k = (np.floor(cen[:, 0] / cellsize).astype(np.int64) * 1000003 +
             np.floor(cen[:, 1] / cellsize).astype(np.int64) * 1009 +
             np.floor(cen[:, 2] / max(cellsize, 1.5)).astype(np.int64))
        if key_mat:
            k = k * 11 + mat.astype(np.int64)
        uk, inv = np.unique(k, return_inverse=True)
        n = len(uk)
        num = np.zeros((n, 3), 'f8'); den = np.zeros(n, 'f8')
        for j in range(3):
            num[:, j] = np.bincount(inv[sel], weights=c[sel, j], minlength=n)
        den = np.bincount(inv[sel], minlength=n).astype('f8')
        m = np.where(den[:, None] > 0, num / np.maximum(den[:, None], 1), np.nan)
        return m[inv], den[inv]
    gm, gd = grid_mean(col, valid, cell, False)
    ok = np.isfinite(gm[:, 0]) & valid
    if mad is None:
        keep = np.full(len(col), 0.45, 'f4')
    else:
        # agreement -> how much of the face's own colour survives.  mad is the
        # median absolute deviation of luminance across the kept views, in
        # linear units; 0.02 is about 5 sRGB levels on a mid-grey.
        agree = np.clip((0.055 - mad) / 0.035, 0.0, 1.0)
        seen = np.clip((nviews.astype('f4') - 2.0) / 3.0, 0.0, 1.0)
        keep = (0.30 + 0.55 * agree * seen).astype('f4')
    kk = keep[:, None]
    out[ok] = (kk[ok] * col[ok] + (1 - kk[ok]) * gm[ok]).astype('f4')
    fill_and_smooth.detail_frac = float((keep[valid] > 0.6).mean()) if valid.any() else 0.0
    fill_and_smooth.keep_mean = float(keep[valid].mean()) if valid.any() else 0.0
    # fill: the same material's mean in a 3 m cell, then its global mean
    fm, fd = grid_mean(col, valid, fill_cell, True)
    need = ~valid
    got = need & np.isfinite(fm[:, 0])
    out[got] = fm[got].astype('f4')
    still = need & ~got
    if still.any():
        for mi in np.unique(mat[still]):
            m = still & (mat == mi)
            src = valid & (mat == mi)
            out[m] = col[src].mean(0) if src.any() else np.array([0.18, 0.18, 0.18], 'f4')
    return out, valid


def smooth_normals(pos, idx, blend=0.35):
    """Area-weighted vertex normals with one Laplacian smoothing round.

    Without a NORMAL attribute every viewer flat-shades a photogrammetric mesh
    whose facets are 13 cm across, which shows as faceted shadows.  Smooth
    normals cost 12 bytes a vertex and remove it."""
    n = len(pos)
    t = pos[idx]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])      # area weighted
    flat = idx.ravel()
    vn = np.empty((n, 3), 'f8')
    rep = np.repeat(fn, 3, axis=0)
    for k in range(3):
        vn[:, k] = np.bincount(flat, weights=rep[:, k], minlength=n)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(ln > 1e-12, vn / np.maximum(ln, 1e-12), np.array([0., 0., 1.]))
    fvn = vn[idx].mean(1)
    vs = np.empty((n, 3), 'f8')
    rep = np.repeat(fvn, 3, axis=0)
    for k in range(3):
        vs[:, k] = np.bincount(flat, weights=rep[:, k], minlength=n)
    ls = np.linalg.norm(vs, axis=1, keepdims=True)
    vs = np.where(ls > 1e-12, vs / np.maximum(ls, 1e-12), vn)
    out = blend * vn + (1 - blend) * vs
    lo = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(lo > 1e-12, out / np.maximum(lo, 1e-12), vn).astype('f4')



GT = OCP.GLTF_TRANSFORM


def simplify_group(pos, nrm, uv, col, idx, tag, ratio, error=5e-3,
                   max_area_delta=0.06):
    """Quadric-decimate one primitive, keeping every attribute.

    Weld first and lock the border: without
    a weld meshoptimizer gets stuck on attribute seams, without --lock-border it
    walks the outline of the part inwards.  Falls back to the input if the
    simplifier fails or the surface area moves by more than max_area_delta."""
    import subprocess
    from glblib import GLB as _GLB
    d = C.W + '/tmp'
    a = f"{d}/{tag}_in.glb"; b = f"{d}/{tag}_w.glb"; c = f"{d}/{tag}_s.glb"

    def clean():
        for f in (a, b, c):
            if os.path.exists(f):
                os.unlink(f)

    def area(p, i):
        t = np.asarray(p)[i]
        return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                                    axis=1).sum() * 0.5)
    ww = StreamGLBWriter(f"{d}/{tag}.bin")
    ww.add_node("m", pos, idx, uv=uv, color=col, normal=nrm,
                mat=ww.add_material_vcol("m"))
    ww.write(a)
    env = dict(os.environ, NODE_OPTIONS="--max-old-space-size=6144")
    r = subprocess.run([GT, "weld", a, b, "--vertex-layout", "separate"],
                       capture_output=True, text=True, cwd=OCP.GLTF_DIR, env=env)
    src = b if r.returncode == 0 and os.path.exists(b) else a
    r = subprocess.run([GT, "simplify", src, c, "--ratio", str(ratio),
                        "--error", str(error), "--lock-border", "true",
                        "--vertex-layout", "separate"],
                       capture_output=True, text=True, cwd=OCP.GLTF_DIR, env=env)
    if r.returncode != 0 or not os.path.exists(c):
        clean(); return pos, nrm, uv, col, idx, "simplify-failed"
    g = _GLB(c); pr = g.js['meshes'][0]['primitives'][0]
    at = pr['attributes']
    P2 = np.array(g.acc(at['POSITION']), 'f4')
    N2 = np.array(g.acc(at['NORMAL']), 'f4') if 'NORMAL' in at else None
    U2 = np.array(g.acc(at['TEXCOORD_0']), 'f4') if 'TEXCOORD_0' in at else None
    C2 = np.array(g.acc(at['COLOR_0']), 'u1') if 'COLOR_0' in at else None
    I2 = np.array(g.acc(pr['indices']), 'u4').reshape(-1, 3)
    g.close(); clean()
    if len(I2) < 3 or N2 is None or U2 is None or C2 is None:
        return pos, nrm, uv, col, idx, "left-alone: attributes lost"
    a0 = area(pos, idx); a1 = area(P2, I2)
    if a0 > 0 and abs((a1 - a0) / a0) > max_area_delta:
        return pos, nrm, uv, col, idx, f"left-alone: area moved {100*(a1-a0)/a0:+.2f}%"
    return P2, N2, U2, C2, I2, f"ok {len(idx)}->{len(I2)}"


# A tint may not turn concrete magenta.  These are caps on the linear-RGB
# saturation of the colour a material is allowed to be tinted to, set from the
# measured distribution per class: grass and brick really are saturated, poured
# concrete and asphalt are not, and where the projection says otherwise it is a
# person's coat or a reflection that a handful of views agreed on.
SAT_CAP = dict(lawn=0.62, paving=0.30, concrete=0.26, asphalt=0.24, gravel=0.50,
               sand=0.40, wood_deck=0.55, metal=0.30, brick=0.85, water=0.45)


# three per-area overrides, read from the area's area.yaml
# stage_knobs only.  An area that sets none of them gets the constant above for
# every value, and every branch below is skipped.
#   GROUND_SAT_CAP {from_ortho: true}: SAT_CAP per class raised to the class's own
#     ortho (nomesh_ground classes writes polish/nomesh_sat_cap.json, the p95 linear
#     saturation of its lit cells); fixed caps clip a bright sand class 0.288 -> 0.225.
#   GROUND_RAW_CLIP_MIN: floor of tint / M (default 0.03); a dark class (Tucson lawn,
#     30 % of lit-lawn vertices) loses its hue in the clip before the hard floor lifts it.
#   GROUND_TILE_CONTRAST {ratio, k_min}: the ground tile's detail finer than the ortho's
#     pixel (lo_m) scaled to the ortho's own finest-band log-luma sd for that class
#     (polish/nomesh_tile_contrast.json from nomesh_ground classes), where the tile's fine
#     band is over ratio x the ortho's: PavingStones070 0.05-0.3 m sd 0.375 vs Tucson's
#     paving ortho 0.10 at 0.3-1.2 m; k never below k_min.
def _area_stage_knob(name):
    return ((((getattr(C, '_AREA_CFG', None) or {}).get('stage_knobs') or {}).get(name)) or {}).get('value')


_GROUND_SAT = _area_stage_knob('GROUND_SAT_CAP')
if _GROUND_SAT and os.path.exists(T.SCR + '/polish/nomesh_sat_cap.json'):
    _d = json.load(open(T.SCR + '/polish/nomesh_sat_cap.json'))
    SAT_CAP = {k: max(v, float(_d.get(k, 0.0))) for k, v in SAT_CAP.items()}
RAW_CLIP_MIN = float(_area_stage_knob('GROUND_RAW_CLIP_MIN') or 0.03)
_GROUND_TC = _area_stage_knob('GROUND_TILE_CONTRAST')
TILE_CONTRAST = {}
if _GROUND_TC and os.path.exists(T.SCR + '/polish/nomesh_tile_contrast.json'):
    TILE_CONTRAST = json.load(open(T.SCR + '/polish/nomesh_tile_contrast.json'))
tile_contrast_report = {}


def tile_contrast(fp, mat, M, ref, period_m, cfg):
    """the baked ground tile's log-luma detail finer than ref['lo_m'] scaled by
    k = ortho band sd / tile fine-band sd (only where the tile's is over cfg['ratio'] x the
    ortho's; k >= cfg['k_min']); chromaticity kept, renormalised per channel to M as
    prep_textures does.  Returns the new file, or fp when k is 1."""
    from PIL import Image
    from scipy import ndimage as _nd
    a = np.asarray(Image.open(fp).convert('RGB'), 'f4') / 255.0
    lin = np.power(a, 2.2); y = np.maximum(lin @ T.LUMW, 1e-4); ly = np.log(y)
    ppm = a.shape[1] / period_m
    lp = _nd.gaussian_filter(ly, ref['lo_m'] * ppm / 2, mode='wrap')
    fine = float((_nd.gaussian_filter(ly, 0.05 * ppm / 2, mode='wrap') - lp).std())
    osd = float(ref['ortho_band_sd'])
    k = 1.0 if fine <= cfg.get('ratio', 2.5) * osd else max(float(cfg.get('k_min', 0.4)), osd / max(fine, 1e-6))
    tile_contrast_report[mat] = dict(tile_fine_sd=round(fine, 4), ortho_band_sd=round(osd, 4), lo_m=ref['lo_m'], k=round(k, 3))
    if k >= 0.999:
        return fp
    y2 = np.exp(lp + k * (ly - lp))
    lu = lin * (y2 / y)[:, :, None]
    lu = lu * (np.asarray(M, 'f4') / np.maximum(lu.reshape(-1, 3).mean(0), 1e-4))[None, None, :]
    out = np.clip(np.power(np.clip(lu, 0, 1), 1 / 2.2) * 255, 0, 255).astype('u1')
    nfp = fp[:-4] + f'_tc{int(round(k * 100)):03d}.jpg'
    Image.fromarray(out).save(nfp, 'JPEG', quality=90, optimize=True)
    return nfp


def cap_chroma(col, mat):
    """Pull any face whose colour is more saturated than its class allows back
    toward its own luminance.  Returns (colour, fraction capped)."""
    out = col.copy()
    mx = col.max(1); mn = col.min(1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    ncap = 0
    for mi, mname in enumerate(MATERIALS):
        cap = SAT_CAP[mname]
        m = (mat == mi) & (sat > cap)
        if not m.any():
            continue
        g = (col[m] @ T.LUMW)[:, None]
        k = (cap / np.maximum(sat[m], 1e-6))[:, None]
        out[m] = g + (col[m] - g) * k
        ncap += int(m.sum())
    return out, ncap / max(len(col), 1)


def axis_of(nrm, cen=None, force_top=None, cell=1.2):
    """0 = project from above, 1 = from +x, 2 = from +y.

    The axis is taken from a voxel-smoothed normal, not the raw face normal.
    On a 13 cm photogrammetric mesh the raw normal flips between neighbouring
    faces, and since each axis needs its own UV, a per-face choice shatters the
    surface into thousands of primitives and breaks smooth shading at every
    flip.  Smoothing first keeps a wall a wall and a deck a deck."""
    n = nrm
    if cen is not None:
        k = (np.floor(cen[:, 0] / cell).astype(np.int64) * 1000003 +
             np.floor(cen[:, 1] / cell).astype(np.int64) * 1009 +
             np.floor(cen[:, 2] / cell).astype(np.int64))
        uk, inv = np.unique(k, return_inverse=True)
        sm = np.empty((len(uk), 3), 'f8')
        sgn = np.sign(nrm[:, 2] + 1e-9)[:, None]      # fold, the mesh is two sided
        nn = nrm * sgn
        for j in range(3):
            sm[:, j] = np.bincount(inv, weights=nn[:, j], minlength=len(uk))
        ln = np.linalg.norm(sm, axis=1, keepdims=True)
        sm = np.where(ln > 1e-9, sm / np.maximum(ln, 1e-9), 0.0)
        n = 0.3 * nn + 0.7 * sm[inv]
    a = np.abs(n)
    ax = np.zeros(len(n), 'i1')
    vert = a[:, 2] < 0.45
    ax[vert & (a[:, 0] >= a[:, 1])] = 1
    ax[vert & (a[:, 1] > a[:, 0])] = 2
    if force_top is not None:
        ax[force_top] = 0
    return ax


def uv_for(pos, ax, scale):
    uv = np.empty((len(pos), 2), 'f4')
    if ax == 0:
        uv[:, 0] = pos[:, 0] / scale; uv[:, 1] = pos[:, 1] / scale
    elif ax == 1:
        uv[:, 0] = pos[:, 1] / scale; uv[:, 1] = pos[:, 2] / scale
    else:
        uv[:, 0] = pos[:, 0] / scale; uv[:, 1] = pos[:, 2] / scale
    return uv


class TintMap:
    """The world-XY floor colour raster (floor/tint_map.npz), sampled bilinearly at a vertex.

    Carrying the tint here instead of on the photo mesh is what removes the
    starburst gradients: the raster is a function of world position, so two
    vertices 4 m apart on the same terrain triangle do not bracket 4 m of
    unfiltered photo colour that the rasteriser then interpolates across."""

    def __init__(self, path=None, M=None, grid=None):
        if path is not None:
            d = np.load(path)
            M = d['M']; grid = d['grid']
        self.M = np.asarray(M).astype('f4')
        x0, y0, c, nx, ny = [float(v) for v in grid]
        self.x0, self.y0, self.c = x0, y0, c
        self.nx, self.ny = int(nx), int(ny)

    def at(self, X, Y):
        u = np.clip((X - self.x0) / self.c - 0.5, 0, self.nx - 1.0001)
        v = np.clip((Y - self.y0) / self.c - 0.5, 0, self.ny - 1.0001)
        j0 = np.floor(u).astype(np.int32); i0 = np.floor(v).astype(np.int32)
        fu = (u - j0)[:, None].astype('f4'); fv = (v - i0)[:, None].astype('f4')
        j1 = np.minimum(j0 + 1, self.nx - 1); i1 = np.minimum(i0 + 1, self.ny - 1)
        return (self.M[i0, j0] * (1 - fu) * (1 - fv) + self.M[i0, j1] * fu * (1 - fv) +
                self.M[i1, j0] * (1 - fu) * fv + self.M[i1, j1] * fu * fv)


def _retriangulate(pos, idx, parent, M):
    """Re-cut every triangle whose edges are in the midpoint table M.

    M[f, e] is the new vertex on edge e of face f, or -1.  Handling all eight
    cases is what keeps the surface closed: a face whose neighbour split their
    shared edge is re-cut too, so no T-junction is ever created."""
    v0, v1, v2 = idx[:, 0], idx[:, 1], idx[:, 2]
    m01, m12, m20 = M[:, 0], M[:, 1], M[:, 2]
    case = ((m01 >= 0).astype('i1') + 2 * (m12 >= 0).astype('i1') +
            4 * (m20 >= 0).astype('i1'))
    fs = []; ps = []

    def put(sel, *cols):
        if not sel.any():
            return
        fs.append(np.stack(cols, 1))
        ps.append(parent[sel])
    s0 = case == 0
    put(s0, v0[s0], v1[s0], v2[s0])
    s0 = case == 1                                        # v0-v1 split
    put(s0, v0[s0], m01[s0], v2[s0]); put(s0, m01[s0], v1[s0], v2[s0])
    s0 = case == 2                                        # v1-v2 split
    put(s0, v0[s0], v1[s0], m12[s0]); put(s0, v0[s0], m12[s0], v2[s0])
    s0 = case == 4                                        # v2-v0 split
    put(s0, v0[s0], v1[s0], m20[s0]); put(s0, m20[s0], v1[s0], v2[s0])
    s0 = case == 3
    put(s0, m01[s0], v1[s0], m12[s0]); put(s0, v0[s0], m01[s0], m12[s0])
    put(s0, v0[s0], m12[s0], v2[s0])
    s0 = case == 6
    put(s0, m12[s0], v2[s0], m20[s0]); put(s0, v1[s0], m12[s0], m20[s0])
    put(s0, v0[s0], v1[s0], m20[s0])
    s0 = case == 5
    put(s0, v0[s0], m01[s0], m20[s0]); put(s0, m01[s0], v1[s0], v2[s0])
    put(s0, m01[s0], v2[s0], m20[s0])
    s0 = case == 7
    put(s0, v0[s0], m01[s0], m20[s0]); put(s0, m01[s0], v1[s0], m12[s0])
    put(s0, m20[s0], m12[s0], v2[s0]); put(s0, m01[s0], m12[s0], m20[s0])
    return np.concatenate(fs), np.concatenate(ps)


def refine_for_tint(pos, idx, tint, tol_rel=0.12, a_min=0.5, max_rounds=32,
                    max_tris=2500000, a_max=None):
    """Longest-edge bisection until no triangle is both larger than a_min and
    carries more than tol_rel of relative luminance change between its corners
    in the world tint raster.

    Only the longest edge is cut.  Cutting all three at once preserves a
    sliver's aspect ratio, and the 3dfier TIN is mostly slivers (median edge
    2.2 m on a median area of 0.32 m2), so a uniform 1 m edge target turns
    8,837 path triangles into 5.1 million (measured on a paths layer).  Bisection instead makes
    slivers rounder and the count follows the area that actually needs it.

    Returns (pos, idx, parent) with parent indexing the input triangles, so the
    material class and the projected colour of the original face carry over."""
    pos = np.asarray(pos, 'f8'); idx = np.asarray(idx, 'i8')
    parent = np.arange(len(idx), dtype='i8')
    rounds = 0
    for _ in range(max_rounds):
        t = pos[idx]
        ar = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                                  axis=1)
        Lv = (tint.at(pos[:, 0], pos[:, 1]).astype('f8') @ T.LUMW.astype('f8'))
        tl = Lv[idx]
        need = ((ar > a_min) &
                ((tl.max(1) - tl.min(1)) > tol_rel * np.maximum(tl.mean(1), 1e-4)))
        if a_max is not None:
            # A hard cap on floor-triangle area.  The relative-change test
            # cannot protect a dark surface: the water tint sits at luminance
            # 0.05, so the gate's 12 % spread (TINT_TOL) is 0.006 in absolute terms and a
            # perfectly smooth gradient across one 24 m2 triangle trips it.
            # Without the cap, almost all fan area sits in a few hundred
            # triangles over 6 m2; capping the area at a_max makes a fan
            # impossible by the gate's own definition.
            need |= ar > a_max
        if not need.any() or len(idx) > max_tris:
            break
        rounds += 1
        n = len(pos)
        e = np.stack([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]], 1)
        lo = e.min(2); hi = e.max(2)
        key = lo * n + hi
        el = np.linalg.norm(pos[hi] - pos[lo], axis=2)
        which = np.argmax(el, axis=1)
        nf = np.nonzero(need)[0]
        marked = np.unique(key[nf, which[nf]])
        mid = 0.5 * (pos[marked // n] + pos[marked % n])
        newid = np.arange(n, n + len(marked), dtype='i8')
        pos = np.concatenate([pos, mid])
        j = np.searchsorted(marked, key)
        jc = np.minimum(j, len(marked) - 1)
        M = np.where(marked[jc] == key, newid[jc], -1)
        idx, parent = _retriangulate(pos, idx, parent, M)
    refine_for_tint.rounds = rounds
    return pos.astype('f4'), idx.astype('u4'), parent


# conforming longest-edge bisection ---------------------- #
def _lepp_edges(idx, n):
    e = np.stack([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]], 1)
    return e.min(2) * n + e.max(2)


# The stitch quads (the optional input polish/stitch.npz, skipped when absent
# or with --no-stitch)
# are a skirt hung off every boundary edge of a floor layer, with boundary edges
# longer than the producer's maximum segment length cut into pieces, so a skirt
# top vertex can sit in the MIDDLE of a terrain boundary edge where the terrain
# has no vertex (a T-junction).  Exact in f32, but the web copy's scene-wide 16-bit
# draco lattice moves that vertex ~1 mm off the edge and a sliver opens.  Weld: every
# stitch vertex within TJ_TOL_M (XY, then Z) of the interior of a boundary edge
# of the refined floor mesh is inserted into that edge, and the one triangle on
# the edge is split there, so both meshes carry the same vertex.  Deterministic:
# np.unique / lexsort only, output order fixed by (face, slot, t).
TJ_TOL_M = 1e-3


def weld_tjunctions(pos, idx, par, spts, tol=TJ_TOL_M, ztol=TJ_TOL_M):
    from scipy.spatial import cKDTree
    pd, idd = pos.dtype, idx.dtype
    P = np.asarray(pos, 'f8'); I = np.asarray(idx, 'i8'); par = np.asarray(par)
    st = dict(boundary_edges=0, inserted=0, faces_split=0)
    if len(I) == 0 or spts is None or len(spts) == 0:
        return pos, idx, par, st
    q = np.round(P * 1e5).astype(np.int64)
    _, g = np.unique(q, axis=0, return_inverse=True); g = g.ravel()
    G = g[I]; M = len(I)
    e = np.concatenate([G[:, [0, 1]], G[:, [1, 2]], G[:, [2, 0]]], 0)
    face = np.tile(np.arange(M), 3); slot = np.repeat(np.arange(3), M)
    es = np.sort(e, 1); key = es[:, 0] * (int(g.max()) + 1) + es[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bnd = np.nonzero(cnt[inv.ravel()] == 1)[0]
    st['boundary_edges'] = int(len(bnd))
    if not len(bnd):
        return pos, idx, par, st
    bf = face[bnd]; bs = slot[bnd]
    A = P[I[bf, bs]]; B = P[I[bf, (bs + 1) % 3]]
    S = np.asarray(spts, 'f8')[np.unique(np.round(np.asarray(spts, 'f8') * 1e5).astype(np.int64),
                                         axis=0, return_index=True)[1]]
    lo = np.minimum(A, B).min(0) - 1.0; hi = np.maximum(A, B).max(0) + 1.0
    S = S[(S[:, :2] >= lo[:2]).all(1) & (S[:, :2] <= hi[:2]).all(1)]
    if not len(S):
        return pos, idx, par, st
    d = B - A; L = np.linalg.norm(d[:, :2], axis=1)
    tree = cKDTree(S[:, :2])
    lists = tree.query_ball_point(0.5 * (A[:, :2] + B[:, :2]), 0.5 * L + tol)
    n = np.array([len(x) for x in lists]); ei = np.repeat(np.arange(len(bnd)), n)
    if not len(ei):
        return pos, idx, par, st
    pi = np.concatenate([np.asarray(x, 'i8') for x in lists if len(x)])
    p = S[pi]; a = A[ei]; dd = d[ei]; Le = np.maximum(L[ei], 1e-12)
    t = ((p[:, :2] - a[:, :2]) * dd[:, :2]).sum(1) / (Le * Le)
    dxy = np.linalg.norm(a[:, :2] + t[:, None] * dd[:, :2] - p[:, :2], axis=1)
    dz = np.abs(a[:, 2] + t * dd[:, 2] - p[:, 2])
    ok = (dxy < tol) & (dz < ztol) & (t * Le > tol) & ((1 - t) * Le > tol)
    ei = ei[ok]; pi = pi[ok]; t = t[ok]
    if not len(ei):
        return pos, idx, par, st
    o = np.lexsort((t, ei)); ei = ei[o]; pi = pi[o]; t = t[o]
    newv = np.arange(len(P), len(P) + len(ei))
    P2 = np.concatenate([P, S[pi]], 0)
    ins = {}
    for k in range(len(ei)):
        ins.setdefault((int(bf[ei[k]]), int(bs[ei[k]])), []).append(int(newv[k]))
    byface = {}
    for (f, s_), vs in sorted(ins.items()):
        byface.setdefault(f, {})[s_] = vs
    I2 = I.copy(); extra = []; epar = []
    for f in sorted(byface):
        sl = byface[f]; c0, c1, c2 = (int(x) for x in I[f])
        if len(sl) == 1:
            s_ = next(iter(sl)); ring = [int(I[f, s_])] + sl[s_] + [int(I[f, (s_ + 1) % 3])]
            apex = int(I[f, (s_ + 2) % 3])
            tris = [[ring[i], ring[i + 1], apex] for i in range(len(ring) - 1)]
        else:
            ring = []
            for s_ in range(3):
                ring.append(int(I[f, s_])); ring += sl.get(s_, [])
            cen = len(P2); P2 = np.concatenate([P2, P[I[f]].mean(0)[None]], 0)
            tris = [[ring[i], ring[(i + 1) % len(ring)], cen] for i in range(len(ring))]
        I2[f] = tris[0]; extra += tris[1:]; epar += [par[f]] * (len(tris) - 1)
    I2 = np.concatenate([I2, np.asarray(extra, 'i8').reshape(-1, 3)], 0)
    par2 = np.concatenate([par, np.asarray(epar, par.dtype)])
    st.update(inserted=int(len(ei)), faces_split=int(len(byface)),
              triangles_added=int(len(extra)))
    return P2.astype(pd), I2.astype(idd), par2, st


def refine_lepp(pos, idx, need_fn, max_rounds=48, max_tris=6000000, flip=True):
    pos = np.asarray(pos, 'f8'); idx = np.asarray(idx, 'i8')
    parent = np.arange(len(idx), dtype='i8'); rounds = 0
    for _ in range(max_rounds):
        n = len(pos)
        t = pos[idx]
        el = np.stack([np.linalg.norm(t[:, 1] - t[:, 0], axis=1),
                       np.linalg.norm(t[:, 2] - t[:, 1], axis=1),
                       np.linalg.norm(t[:, 0] - t[:, 2], axis=1)], 1)
        need = need_fn(pos, idx, t, el)
        if not need.any() or len(idx) > max_tris:
            break
        rounds += 1
        key = _lepp_edges(idx, n)
        lw = np.argmax(el, 1)
        ue, einv = np.unique(key, return_inverse=True)
        einv = einv.reshape(-1, 3)
        leid = einv[np.arange(len(idx)), lw]
        # never split a boundary edge (used by one face): a boundary midpoint
        # is a T-junction against the neighbouring mesh (stitch skirts, other
        # parts), which draco's 16-bit lattice opens into a visible sliver.  A face whose longest
        # edge is blocked cannot be bisected, so all its edges are blocked too;
        # grow that set to a fixed point, then run the closure on the rest.
        blk = np.bincount(einv.ravel(), minlength=len(ue)) == 1
        while True:
            frz = blk[leid]
            nb = blk.copy(); nb[einv[frz].ravel()] = True
            if nb.sum() == blk.sum():
                break
            blk = nb
        mk = np.zeros(len(ue), bool); mk[leid[need & ~blk[leid]]] = True
        while True:                       # closure: a marked edge drags the longest in
            add = mk[einv].any(1) & ~mk[leid]
            if not add.any():
                break
            mk[leid[add]] = True
        refine_lepp.blocked_faces = int(blk[leid].sum())
        marked = ue[mk]
        mid = 0.5 * (pos[marked // n] + pos[marked % n])
        pos = np.concatenate([pos, mid])
        newid = np.full(len(ue), -1, 'i8'); newid[mk] = n + np.arange(len(marked))
        M = newid[einv]                                     # midpoint id per face edge
        has = (M >= 0).any(1)
        out = [idx[~has]]; par = [parent[~has]]
        f = np.nonzero(has)[0]
        r = lw[f]                                           # rotate: longest edge first
        o = np.stack([r, (r + 1) % 3, (r + 2) % 3], 1)
        a, b, c = [idx[f, o[:, k]] for k in range(3)]
        mab = M[f, o[:, 0]]; mbc = M[f, o[:, 1]]; mca = M[f, o[:, 2]]
        pf = parent[f]
        s = mca < 0                                          # child (a, m, c)
        out += [np.stack([a[s], mab[s], c[s]], 1)]; par += [pf[s]]
        s = ~s
        out += [np.stack([a[s], mab[s], mca[s]], 1), np.stack([mab[s], c[s], mca[s]], 1)]
        par += [pf[s], pf[s]]
        s = mbc < 0                                          # child (m, b, c)
        out += [np.stack([mab[s], b[s], c[s]], 1)]; par += [pf[s]]
        s = ~s
        out += [np.stack([mab[s], b[s], mbc[s]], 1), np.stack([mab[s], mbc[s], c[s]], 1)]
        par += [pf[s], pf[s]]
        idx = np.concatenate(out); parent = np.concatenate(par)
        if flip:
            idx = flip_same_parent(pos, idx, parent)
    refine_lepp.rounds = rounds
    return pos, idx, parent


def _lepp_ang(a, b, c):
    """Angle at a in triangle (a, b, c)."""
    u = b - a; v = c - a
    return np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), (u * v).sum(1))


def flip_same_parent(pos, idx, parent, passes=8):
    """Delaunay edge flips between two faces cut from the SAME input triangle.

    Both lie in that triangle's plane, so a flip moves no surface at all; it
    only turns the needles longest-edge bisection leaves inside a 3dfier
    sliver into rounder triangles, which is what stops the cut count from
    growing much faster than the sliver's length."""
    idx = idx.copy()
    for _ in range(passes):
        n = len(pos)
        e = np.stack([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]], 1)
        key = (e.min(2) * n + e.max(2)).ravel()
        o = np.argsort(key, kind='stable'); ks = key[o]
        _, inv, cnt = np.unique(ks, return_inverse=True, return_counts=True)
        same = (ks[1:] == ks[:-1]) & (cnt[inv[:-1]] == 2)      # manifold pairs
        i0 = o[:-1][same]; i1 = o[1:][same]
        f0 = i0 // 3; e0 = i0 % 3; f1 = i1 // 3; e1 = i1 % 3
        ok = (parent[f0] == parent[f1]) & (f0 != f1)
        f0, e0, f1, e1 = f0[ok], e0[ok], f1[ok], e1[ok]
        if len(f0) == 0:
            break
        u = idx[f0, e0]; v = idx[f0, (e0 + 1) % 3]; p = idx[f0, (e0 + 2) % 3]
        q = idx[f1, (e1 + 2) % 3]
        good = (idx[f1, e1] == v) & (idx[f1, (e1 + 1) % 3] == u)   # consistent winding
        P = pos
        exc = _lepp_ang(P[p], P[u], P[v]) + _lepp_ang(P[q], P[v], P[u]) - np.pi
        nrm = np.cross(P[v] - P[u], P[p] - P[u])
        n1 = np.cross(P[u] - P[p], P[q] - P[p]); n2 = np.cross(P[v] - P[q], P[p] - P[q])
        valid = ((n1 * nrm).sum(1) > 0) & ((n2 * nrm).sum(1) > 0)
        cand = np.nonzero(good & valid & (exc > 1e-6))[0]
        if len(cand) == 0:
            break
        # independent set: an edge wins if it is the best candidate of both faces
        best = np.full(len(idx), -np.inf)
        np.maximum.at(best, f0[cand], exc[cand]); np.maximum.at(best, f1[cand], exc[cand])
        win = cand[(exc[cand] >= best[f0[cand]]) & (exc[cand] >= best[f1[cand]])]
        fa = np.bincount(np.concatenate([f0[win], f1[win]]), minlength=len(idx))
        win = win[(fa[f0[win]] == 1) & (fa[f1[win]] == 1)]      # drop ties
        if len(win) == 0:
            break
        idx[f0[win]] = np.stack([p[win], u[win], q[win]], 1)
        idx[f1[win]] = np.stack([q[win], v[win], p[win]], 1)
    return idx


def unit_cylinder(n=12, cap_bottom=True, uvs=False):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([np.cos(a), np.zeros(n), np.sin(a)], 1)
    pos = np.concatenate([ring, ring + [0, 1, 0]]).astype('f4')
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [[i, j, n + j], [i, n + j, n + i]]
    top = len(pos); pos = np.concatenate([pos, [[0, 1, 0]]]).astype('f4')
    for i in range(n):
        tris.append([n + i, n + ((i + 1) % n), top])
    if cap_bottom:
        bot = len(pos); pos = np.concatenate([pos, [[0, 0, 0]]]).astype('f4')
        for i in range(n):
            tris.append([(i + 1) % n, i, bot])
    uv = None
    if uvs:
        th = np.arctan2(pos[:, 2], pos[:, 0]) / (2 * math.pi)
        uv = np.stack([th * 3.0, pos[:, 1] * 2.0], 1).astype('f4')
    return pos, np.array(tris, 'u4'), uv


# ------------------------------------------------------------------ main -- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--min-faces', type=int, default=200)
    ap.add_argument('--web', action='store_true',
                    help='decimate the photo mesh and shrink the textures')
    ap.add_argument('--ratio', type=float, default=0.12)
    ap.add_argument('--core', action='store_true',
                    help='crop to the CORE box around the area origin')
    ap.add_argument('--texscale', type=float, default=1.0)
    ap.add_argument('--no-normalmap', action='store_true')
    ap.add_argument('--draco', action='store_true')
    ap.add_argument('--no-floor-tint', action='store_true',
                    help='ignore the floor tint raster: keep the photo floor and the per-face tint')
    ap.add_argument('--no-facades', action='store_true',
                    help='ignore the facade stages (s33-s35): flat material walls, photo remnants kept')
    ap.add_argument('--no-ortho', action='store_true',
                    help='ignore the ortho rasters: panorama colour only, no ortho backup')
    ap.add_argument('--no-classtint', action='store_true',
                    help='ignore stage 50: one tint raster for every class')
    ap.add_argument('--no-stitch', action='store_true',
                    help='ignore the stitch quads: leave the floor layers open at their edges')
    ap.add_argument('--no-scraps', action='store_true',
                    help='ignore the scrap masks: keep the photo-mesh scraps on the floor')
    ap.add_argument('--no-ground-detail', action='store_true',
                    help='skip the GROUND_DETAIL multiply on the photographed '
                         'ground pages (aggregate-scale relief so they do not '
                         'read flat at grazing angles)')
    ap.add_argument('--no-class-detail', action='store_true',
                    help='skip the CLASS_DETAIL multiply on the tiled class '
                         'material colour tiles (the same relief, on the '
                         'ground_terrain_3dfier_/stitch_/path_ materials rather '
                         'than the photographed pages)')
    ap.add_argument('--no-photo-ground', action='store_true',
                    help='ignore the photographic ground atlas (an optional '
                         'input): tiled material everywhere')
    a = ap.parse_args()
    out_path = a.out or (C.OUT + '/model.glb')
    t0 = time.time()
    memguard.require("composite", 10.0, need_disk_gb=6.0)

    # ---- gather every layer's faces -------------------------------------- #
    ke_path = FLOOR + '/kill_extra.npy'
    tint_path = FLOOR + '/tint_map.npz'
    ortho = {}
    use_ortho = (not a.no_ortho) and os.path.exists(FLOOR + '/tint_map_blend.npz')
    matdir = MT
    if use_ortho:
        tint_path = FLOOR + '/tint_map_blend.npz'
        # optional ortho-voted material classes (material_ortho/photo_mat.npz)
        # and a tint ownership record (ortho/orthotint.json, optional); each is
        # used when present
        if os.path.exists(MT_ORTHO + '/photo_mat.npz'):
            matdir = MT_ORTHO
        ojp = ORTHO + '/orthotint.json'
        oj = json.load(open(ojp)) if os.path.exists(ojp) else {}
        ortho = dict(tint_map=os.path.basename(tint_path),
                     material_source=os.path.basename(matdir),
                     ownership=oj.get('tint_ownership_by_layer'),
                     grid_cells=oj.get('grid_cells'),
                     face_colour_overrides={})
    facades = {}
    use_fac = (not a.no_facades) and os.path.exists(FACMAP)
    FACE_ATLAS = WALL_UV = WALL_NRM = None
    if use_fac:
        fm = np.load(FACMAP, allow_pickle=True)
        FACE_ATLAS = fm['face_atlas']; WALL_UV = fm['uv']; WALL_NRM = fm['fnrm']
        fdoc = json.load(open(FAC + '/facadetex.json'))
        facades = {k: v for k, v in fdoc.items() if k != 'per_wall'}
    floor_tint = {}
    TINT = None
    TINTC = None
    if not a.no_floor_tint and os.path.exists(tint_path):
        TINT = TintMap(tint_path)
        floor_tint['tint_map'] = dict(px=[TINT.nx, TINT.ny], cell_m=TINT.c,
                                    origin_enu_m=[TINT.x0, TINT.y0])
        # stage 50: one tint layer per material class, each filtered inside its
        # own class mask, so the colour hand-over stops at the curb instead of
        # fading 1.3 m across it
        cp = FLOOR + '/tint_class.npz'
        # tint_class_photo.npz (optional input, used with the photographed
        # ground atlas): the same raster with the material beside a photographed
        # region pulled onto that photograph's own low-frequency colour
        if (not a.no_photo_ground) and os.path.exists(FLOOR + '/tint_class_photo.npz'):
            cp = FLOOR + '/tint_class_photo.npz'
        if (not a.no_classtint) and os.path.exists(cp):
            dcl = np.load(cp)
            Mst = dcl['M']; gcl = dcl['grid']
            mats_cl = [str(x) for x in dcl['materials']]
            TINTC = {m: TintMap(M=Mst[i], grid=gcl) for i, m in enumerate(mats_cl)}
            # refine against the raster stage 22 actually samples, not the blend
            rp = FLOOR + '/tint_class_ref.npz'
            if (not a.no_photo_ground) and os.path.exists(FLOOR + '/tint_class_ref_photo.npz'):
                rp = FLOOR + '/tint_class_ref_photo.npz'
            if os.path.exists(rp):
                TINT = TintMap(rp)
                floor_tint['refine_raster'] = 'tint_class_ref.npz (per-class, as sampled)'
            floor_tint['tint_class'] = json.load(open(FLOOR + '/tint_class.json')) \
                if os.path.exists(FLOOR + '/tint_class.json') else dict(used=True)
            print(f"  class-confined tint: {len(TINTC)} layers", flush=True)
            if GROUND_TINT_RELIEF:
                tint_relief(TINTC, TINT, GROUND_TINT_RELIEF, floor_tint.setdefault('tint_relief', {}))
                print(f"  tint relief: {floor_tint['tint_relief']}", flush=True)
            if GROUND_EDGE_FILL:   # before the canopy fill, so the canopy is levelled onto the edge-filled lawn
                edge_fill(TINTC, TINT, GROUND_EDGE_FILL, floor_tint.setdefault('edge_fill', {}))
                print(f"  edge fill: {floor_tint['edge_fill']}", flush=True)
            if GROUND_CANOPY_FILL:
                canopy_fill(TINTC, TINT, GROUND_CANOPY_FILL, floor_tint.setdefault('canopy_fill', {}))
                print(f"  canopy fill: {floor_tint['canopy_fill']}", flush=True)
    # common.GROUND_HARD_RECLASS: lawn faces on measured hard ground draw the hard class (emit_floor)
    HARD_RC, HARD_G = {}, None
    if GROUND_HARD_RECLASS and TINTC is not None and not a.no_ortho:
        HARD_RC, HARD_G = hard_reclass_mask(GROUND_HARD_RECLASS, floor_tint.setdefault('hard_reclass', {}))
        print(f"  hard reclass: {floor_tint['hard_reclass']}", flush=True)
    ORTHOTINT = None
    if not a.no_ortho and os.path.exists(ORTHO + '/rasters.npz'):
        _or = np.load(ORTHO + '/rasters.npz')
        _g = _or['grid']
        ORTHOTINT = (TintMap(M=_or['Mo'].astype('f4'), grid=_g),
                     TintMap(M=np.repeat(_or['ovalid'].astype('f4')[:, :, None], 3, 2),
                             grid=_g))
        print("  ortho tint loaded for the dark-tint floor", flush=True)
    L = {}
    detail = {}
    targets_acc = {m: [] for m in MATERIALS}
    for name in ['photo', 'ground_terrain', 'paths', 'structures', 'buildings', 'water']:
        pf = PJ + f'/{name}_face.npz'
        mf = matdir + f'/{name}_mat.npz'
        if not (os.path.exists(pf) and os.path.exists(mf)):
            print("missing data for", name); continue
        d = np.load(pf); m = np.load(mf)
        col = d['col']; nviews = d['nviews']; area = d['area']
        mat = m['mat'].astype('i1'); kill = m['kill']; up = m['up']
        if name == 'photo' and not a.no_floor_tint and os.path.exists(ke_path):
            ke = np.load(ke_path)
            kill = kill | ke
            floor_tint['photo_faces_removed_by_floor_kill'] = int(ke.sum())
        if name == 'water' and (not a.no_scraps) and os.path.exists(KILLW):
            kw = np.load(KILLW)
            if len(kw) == len(kill):
                kill = kill | kw
                floor_tint['water_faces_removed_by_scrap_mask'] = int(kw.sum())
        if name == 'photo' and (not a.no_scraps) and os.path.exists(SCRAPS):
            ks = np.load(SCRAPS)
            kill = kill | ks
            floor_tint['photo_faces_removed_by_scrap_mask'] = int(ks.sum())
        if name == 'photo' and use_fac and os.path.exists(FACKILL):
            kf = np.load(FACKILL)
            kill = kill | kf
            facades['photo_faces_removed_by_facade_kill'] = int(kf.sum())
        if name == 'photo':
            CEN = np.load(C.G + '/centroid.npy', mmap_mode='r')
            cen = np.asarray(CEN[:len(col)]).astype('f4')
            P = np.load(C.G + '/pos.npy', mmap_mode='r')
            F = np.load(C.G + '/faces.npy', mmap_mode='r')
            nrm = np.zeros((len(col), 3), 'f4')
            step = 1 << 21
            for s0 in range(0, len(col), step):
                e0 = min(s0 + step, len(col))
                f = np.asarray(F[s0:e0])
                p0 = np.asarray(P[f[:, 0]]); p1 = np.asarray(P[f[:, 1]]); p2 = np.asarray(P[f[:, 2]])
                nn = np.cross(p1 - p0, p2 - p0)
                nrm[s0:e0] = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        else:
            z = np.load(C.W + f'/layers/{name}.npz')
            t = z['pos'].astype('f4')[z['idx']]
            cen = t.mean(1)
            nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            nrm = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        colf, valid = fill_and_smooth(cen, col, nviews, mat, mad=d['mad'])
        ovr = None
        # optional per-layer face-colour override (ortho/colour_override_<layer>.npz:
        # idx, col), used only with the ortho and skipped when absent
        opath = ORTHO + f'/colour_override_{name}.npz'
        # the ortho does not own the water.  The ortho face override gives
        # every water face the mean of the ortho cells that fall inside it, so a
        # moored hull, a wake or sun glint is painted straight onto the water
        # plane as white smears.  Skip the override on the
        # water layer and let the measured water tint ramp own
        # those faces; stage 50 already keeps the ortho off water cells at the
        # shore boundary (BND_SKIP), so this is the same rule one layer up.
        if name in ORTHO_SKIP_LAYERS:
            if os.path.exists(opath):
                ortho['face_colour_overrides'][name] = dict(
                    faces=0, area_m2=0.0,
                    skipped='ortho masked off this class: it paints hulls, '
                            'wake and glint onto the plane')
        elif use_ortho and os.path.exists(opath):
            od = np.load(opath)
            oi = od['idx']
            colf[oi] = od['col']
            valid[oi] = True
            ovr = np.zeros(len(colf), bool); ovr[oi] = True
            ortho['face_colour_overrides'][name] = dict(
                faces=int(len(oi)), area_m2=round(float(area[oi].sum()), 1))
        colf, capped = cap_chroma(colf, mat)
        detail[name] = dict(frac_faces_chroma_capped=round(capped, 4),
                            frac_area_keeping_mesh_resolution_detail=
                            round(fill_and_smooth.detail_frac, 4),
                            mean_own_colour_weight=round(fill_and_smooth.keep_mean, 3))
        L[name] = dict(col=colf, nviews=nviews, area=area, mat=mat, kill=kill,
                       up=up, nrm=nrm, cen=cen, valid=valid,
                       sublayer=m['sublayer'], ovr=ovr)
        live = ~kill
        for mi, mname in enumerate(MATERIALS):
            s = live & (mat == mi) & valid
            if s.sum() > 20:
                targets_acc[mname].append(colf[s])
        print(f"  {name}: {len(col):,} faces, {int(kill.sum()):,} killed, "
              f"{100*valid[live].mean():.1f}% with photo colour", flush=True)

    targets = {}
    for mname in MATERIALS:
        if targets_acc[mname]:
            v = np.concatenate(targets_acc[mname])
            targets[mname] = np.percentile(v, 97, axis=0).astype('f4')
    if 'water' in targets:
        # the generated water tile is renormalised to this mean, so the mean is
        # where the tile's hue comes from; rebase it exactly as the tint is,
        # which also keeps the tint (cw / M) near-neutral and clear of the 1.0
        # clip, so the water's hue survives the COLOR_0 ceiling
        _w0 = np.asarray(targets['water'], 'f4').copy()
        targets['water'] = water_chroma(_w0)
        floor_tint['water_tint'] = dict(
            material_mean_linear_rgb_before=[round(float(x), 4) for x in _w0],
            material_mean_linear_rgb_after=[round(float(x), 4) for x in targets['water']],
            chromaticity=[round(float(x), 4) for x in WATER_CHROMA],
            lift=WATER_LIFT, colour_noise_amp=WATER_NOISE_AMP,
            roughness_noise_amp=WATER_PBR_AMP,
            rule="the water tint and the water tile mean keep their measured "
                 "luminance and take the water ramp's chromaticity")
    means, files = prep_textures(targets, a.texscale, not a.no_normalmap,
                                 classdetail=not a.no_class_detail)
    if tile_contrast_report:                       # per-area knob only
        floor_tint['tile_contrast'] = tile_contrast_report
        print(f"  tile contrast: {tile_contrast_report}", flush=True)
    fol = foliage_texture()
    print("textures prepared", {k: [round(float(x), 3) for x in v] for k, v in means.items()},
          flush=True)
    if GROUND_SHADE_FILL and TINTC:   # after the tile means exist; the factor is only read in emit()
        shade_fill(TINTC, means, ORTHOTINT, GROUND_SHADE_FILL, floor_tint.setdefault('shade_fill', {}))
        print(f"  shade fill: {floor_tint['shade_fill']}", flush=True)

    # ---- assemble --------------------------------------------------------- #
    os.makedirs(C.W + '/tmp', exist_ok=True)
    w = StreamGLBWriter(C.W + '/tmp/comp.bin')
    kids = {k: [] for k in ('LAYER_ground', 'LAYER_paths', 'LAYER_structures',
                            'LAYER_buildings', 'LAYER_trees', 'LAYER_water')}
    matcache = {}
    cdet_used = {u: set() for u in CLASS_DETAIL_USE}

    def material(mname, use='ground'):
        """The glTF material for one class, for one USE of it.

        A class tile is baked twice -- once for the up-facing parts and once
        for the vertical ones -- because CLASS_DETAIL's amplitude has to differ
        between them (see CLASS_DETAIL_WALL).  A class with no second variant
        (water, or a run with --no-class-detail) falls back to the one tile and
        to ONE material, so nothing is duplicated for nothing."""
        f = files.get(mname)
        if use != 'ground' and not (f or {}).get('color_by_use', {}).get(use):
            use = 'ground'
        if use in cdet_used and mname in (files or {}) and files.get(mname):
            cdet_used[use].add(mname)
        if (mname, use) in matcache:
            return matcache[(mname, use)]
        met, rough = PBR[mname]
        mname_use = mname if use == 'ground' else f"{mname}_{use}"
        if f is None:
            mi = w.add_material_plain(
                list(np.clip(np.power(means[mname], 1 / 2.2), 0, 1)) + [1.0],
                f"mat_{mname_use}", rough=rough, metal=met)
        else:
            ti = w.add_image_file(f.get('color_by_use', {}).get(use, f['color']),
                                  wrap=1)
            mi = w.add_material_tex(ti, f"mat_{mname_use}")
            m = w.js['materials'][mi]
            m['pbrMetallicRoughness']['metallicFactor'] = met
            m['pbrMetallicRoughness']['roughnessFactor'] = rough
            _nf = (f.get('normal_by_use') or {}).get(use, f.get('normal'))
            if _nf:
                ni = w.add_image_file(_nf, wrap=1)
                m['normalTexture'] = {"index": ni, "scale": 0.8}
            if f.get('mr'):
                mri = w.add_image_file(f['mr'], wrap=1)
                m['pbrMetallicRoughness']['metallicRoughnessTexture'] = {"index": mri}
                # roughness is the texture's G channel, so the factor is unity;
                # metallic is NOT -- the B channel is flat 1.0 and the class's
                # own metallic stays a scalar; forcing it to 1.0 here would make
                # water a conductor.
                m['pbrMetallicRoughness']['roughnessFactor'] = 1.0
                m['pbrMetallicRoughness']['metallicFactor'] = met
            ent = f['asset']
            if ent is None:
                m['extras'] = dict(
                    material_class=mname, metres_per_tile=TILE_M[mname],
                    texture_source='generated in s22_composite.water_textures()',
                    licence='no third-party asset; generated by stages/s22_composite.py',
                    colour=('trough and crest linear RGB set from the measured '
                            'panorama tint of the water surface; the crest lift '
                            'and the %.2fx tint lift are inferred, not measured '
                            '-- the top-down tint carries no sun glitter'
                            % WATER_LIFT))
                # water optics (common.WATER_IOR / WATER_SPECULAR): without them
                # the studio environment's panels mirror as white glare
                _ext = {}
                if getattr(C, 'WATER_IOR', None):
                    _ext['KHR_materials_ior'] = {"ior": float(C.WATER_IOR)}
                if getattr(C, 'WATER_SPECULAR', None) is not None:
                    _ext['KHR_materials_specular'] = {"specularFactor": float(C.WATER_SPECULAR)}
                if _ext:
                    m.setdefault('extensions', {}).update(_ext)
                    _used = w.js.setdefault('extensionsUsed', [])
                    for _e in _ext:
                        if _e not in _used:
                            _used.append(_e)
                _sky_reflect(m, mname)
                matcache[(mname, use)] = mi
                return mi
            m['extras'] = dict(material_class=mname,
                               texture_source=ent['source'], asset_id=ent['asset_id'],
                               licence=ent['licence'], licence_url=ent['licence_url'],
                               page_url=ent['page_url'], download_url=ent['download_url'],
                               metres_per_tile=(ground_period(mname) if use == 'ground' else TILE_M[mname]))
            _sky_reflect(m, mname)
            if use == 'ground' and (GROUND_SUPERTILE.get(mname) or {}).get('mode') == 'hex':
                _st = GROUND_SUPERTILE[mname]
                m['extras']['super_tile'] = (
                    f"baked here (s22_composite.hex_bake): {_st['period_m']} m x {_st['period_m']} m "
                    f"hex-tile histogram-preserving blend (Heitz and Neyret) of the {_st['src_m']} m "
                    f"CC0 source, lattice ~{_st['hex_m']} m, random offset + rotation per vertex; "
                    "colour at 1 m and up comes from COLOR_0 (tint raster)")
            if not a.no_class_detail:
                cd = CLASS_DETAIL_USE[use]
                m['extras']['detail_map'] = (
                    "generated here (s22_composite.class_detail): a tileable "
                    "achromatic multiplicative field, mean 1.0 in linear light, "
                    f"{cd['octaves']}-octave periodic Gaussian noise from "
                    f"{cd['detail_m']} m down to the tile's own texel grid, "
                    f"relative amplitude {cd['amp']}; no asset was downloaded "
                    "for it and it carries no licence of its own")
                m['extras']['detail_map_use'] = (
                    f"{use}: this material is the '{use}' bake of the class "
                    f"tile.  The parts whose UVs are projected from above take "
                    f"amplitude {CLASS_DETAIL['amp']}, the vertical ones "
                    f"{CLASS_DETAIL_WALL['amp']} -- the same field at half the "
                    "contrast, because at eye level on a wall the ground "
                    "amplitude reads as blotches")
                m['extras']['detail_map_note'] = (
                    "applied to the CC0 tile in linear light before the tile is "
                    "renormalised to its measured mean, so the material's colour "
                    "is unchanged: it adds the aggregate-scale relief the tile "
                    "does not carry, not colour.  Disable with --no-class-detail.")
        _floor_lit(w.js['materials'][mi], mname)
        matcache[(mname, use)] = mi
        return mi

    faccache = {}
    fac_tex_bytes = 0

    def facade_material(pg):
        """One material per atlas page: the baked wall textures."""
        nonlocal fac_tex_bytes
        if pg in faccache:
            return faccache[pg]
        src = f"{FAC}/atlas_{pg:02d}.jpg"
        fp = src
        if a.texscale != 1.0:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(src)
            k = max(256, int(im.size[0] * a.texscale))
            fp = f"{TEXOUT}/facade_atlas_{pg:02d}_{k}_{src_tag(src)}.jpg"
            if not os.path.exists(fp):
                im.resize((k, k), Image.LANCZOS).save(fp, 'JPEG', quality=90,
                                                      optimize=True)
        ti = w.add_image_file(fp, wrap=1)
        mi = w.add_material_tex(ti, f"facade_atlas_{pg:02d}")
        m = w.js['materials'][mi]
        m['pbrMetallicRoughness']['metallicFactor'] = 0.0
        m['pbrMetallicRoughness']['roughnessFactor'] = 0.85
        # stage 35/60 normal + roughness pages of the same layout
        # (frames, recessed panes, sills, bands; glass smooth).  Metallic stays 0:
        # the MR page's B channel is 1.0 and metallicFactor 0.
        for _k in ('n', 'r'):
            _src = f"{FAC}/atlas_{pg:02d}_{_k}.jpg"
            if not getattr(C, 'FACADE_PBR', False) or not os.path.exists(_src):
                continue
            if _k == 'n' and a.no_normalmap:
                continue
            _fp = _src
            if a.texscale != 1.0:
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                _im = Image.open(_src)
                _kk = max(256, int(_im.size[0] * a.texscale))
                _fp = f"{TEXOUT}/facade_atlas_{pg:02d}_{_k}_{_kk}_{src_tag(_src)}.jpg"
                if not os.path.exists(_fp):
                    _im.resize((_kk, _kk), Image.LANCZOS).save(_fp, 'JPEG', quality=92,
                                                               optimize=True)
            _ti = w.add_image_file(_fp, wrap=1)
            if _k == 'n':
                m['normalTexture'] = {"index": _ti,
                                      "scale": float(getattr(C, 'FACADE_NORMAL_SCALE', 1.0))}
            else:
                m['pbrMetallicRoughness']['metallicRoughnessTexture'] = {"index": _ti}
                m['pbrMetallicRoughness']['roughnessFactor'] = 1.0
        m['extras'] = dict(
            material_class='building_facade',
            texture_source="stage 34 multi-view panorama projection onto the wall "
                           "plane, over a CC0 material with a generated window grid "
                           "(stage 35)",
            licence="Mapillary imagery CC BY-SA 4.0 (photographed facades); "
                    "ambientCG CC0 1.0 (material tiles); window grid generated here",
            licence_url="https://creativecommons.org/licenses/by-sa/4.0/",
            atlas_px=facades.get('atlas_px'),
            mad_cut_sRGB_levels=facades.get('mad_cut_sRGB_levels'),
            note="one tile per building wall; UVs are in [0,1] inside that tile")
        _sky_glass(m, 'building_facade')
        faccache[pg] = mi
        fac_tex_bytes += os.path.getsize(fp)
        return mi

    # ---- optional photographed ground atlas: an atlas object with pages, page_m,
    # pband, jpeg(), uv(), face_page(), page_at() and extras().  This stage does not
    # load one, so PHOTO is None and the photo-ground branches below are skipped.
    PHOTO = None
    PC_FEAT = None
    photo_stat = {}
    phcache = {}
    photo_tex_bytes = 0
    photo_det_shift = []
    photo_det_raw = []

    _fctx = {}

    def feather_ctx():
        """class map + the class tiles resampled to a page texel, for photo_match's feather."""
        if 'ctx' in _fctx:
            return _fctx['ctx']
        cp = T.SCR + '/polish/class_map.npz'
        if not os.path.exists(cp) or not GROUND_PHOTO_MATCH.get('feather_m'):
            _fctx['ctx'] = None; return None
        cls = np.load(cp)['cls']
        gx0, gy0, c = float(TINT.x0), float(TINT.y0), float(TINT.c)
        if cls.shape != np.asarray(TINT.M).shape[:2]:
            _fctx['ctx'] = None; return None
        from PIL import Image as _I
        cache = {}
        def tiles(tex):
            key = round(tex, 5)
            if key in cache:
                return cache[key]
            out = {}
            for cid, mn in enumerate(MATERIALS):
                f = files.get(mn) if files else None
                if not f or mn == 'water' or mn not in means:
                    continue
                n = max(8, int(round(ground_period(mn) / tex)))
                im = _I.open(f['color']).convert('RGB').resize((n, n), _I.LANCZOS)
                out[cid] = (T.srgb_to_linear(np.asarray(im)).astype('f4'), ground_period(mn),
                            np.asarray(means[mn], 'f4'), (TINTC.get(mn) if TINTC else None) or TINT)
            cache[key] = out
            return out
        lift = None
        if GROUND_PHOTO_MATCH.get('level_to') == 'drawn':
            lift = lambda raw, X, Y, M: ground_tint_floor(raw, X, Y, M, ORTHOTINT)
        _fctx['ctx'] = dict(cls=cls, grid=(gx0, gy0, c), tiles=tiles, lift=lift)
        return _fctx['ctx']

    def drawn_target():
        """f(X, Y, tint) -> the colour the class tile at X, Y is drawn with (tint/M
        through ground_tint_floor, times M), for photo_match; None unless level_to='drawn'."""
        if GROUND_PHOTO_MATCH.get('level_to') != 'drawn':
            return None
        cp = T.SCR + '/polish/class_map.npz'
        if not os.path.exists(cp):
            raise SystemExit("GROUND_PHOTO_MATCH level_to='drawn' needs " + cp)
        if 'drawn' not in _fctx:
            cls = np.load(cp)['cls']
            if cls.shape != np.asarray(TINT.M).shape[:2]:
                raise SystemExit('class_map grid != tint raster grid; cannot level to the drawn tile')
            Mc = np.full((len(MATERIALS), 3), np.nan, 'f4')
            for cid, mn in enumerate(MATERIALS):
                if mn != 'water' and mn in means:
                    Mc[cid] = np.asarray(means[mn], 'f4')
            _fctx['drawn'] = (cls, Mc)
        cls, Mc = _fctx['drawn']
        def f(X, Y, Tc):
            ci = np.clip(((Y - TINT.y0) / TINT.c).astype(np.int32), 0, cls.shape[0] - 1)
            cj = np.clip(((X - TINT.x0) / TINT.c).astype(np.int32), 0, cls.shape[1] - 1)
            M = Mc[np.clip(cls[ci, cj], 0, len(MATERIALS) - 1)]
            ok = np.isfinite(M).all(1)
            M = np.where(ok[:, None], M, 1.0).astype('f4')
            raw = np.clip(Tc / np.maximum(M, 1e-4), 0.03, 1.0)
            out = ground_tint_floor(raw, X, Y, M, ORTHOTINT) * M
            return np.where(ok[:, None], out, Tc)
        return f

    def photo_material(pg):
        """One material per atlas page: the best-view-per-patch photograph."""
        nonlocal photo_tex_bytes
        if pg in phcache:
            return phcache[pg]
        src = PHOTO.jpeg(pg)
        fp = src
        if a.texscale != 1.0:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(src)
            k = max(512, int(im.size[0] * a.texscale))
            fp = f"{TEXOUT}/photo_atlas_{pg:03d}_{k}_{src_tag(src)}.jpg"
            if not os.path.exists(fp):
                im.resize((k, k), Image.LANCZOS).save(fp, 'JPEG', quality=90,
                                                      optimize=True)
        # re-level the page onto the class-as-sampled tint raster (common.py
        # GROUND_PHOTO_MATCH) so it meets the class tile with no step and no colour cast
        if GROUND_PHOTO_MATCH and TINT is not None:
            from PIL import Image as _Im
            _Im.MAX_IMAGE_PIXELS = None
            if not hasattr(photo_material, 'ttag'):
                photo_material.ttag = hashlib.md5(np.ascontiguousarray(
                    np.asarray(TINT.M)[::7, ::7]).tobytes()).hexdigest()[:6]
            kpx0 = _Im.open(fp).size[0]
            mp = (f"{TEXOUT}/photo_atlas_{pg:03d}_{kpx0}_{src_tag(fp)}{GPM_TAG}"
                  f"_t{photo_material.ttag}.jpg")
            if not os.path.exists(mp):
                vp = f"{os.path.dirname(src)}/page_{int(pg):03d}.npz"
                vm = None
                if os.path.exists(vp):
                    vm = np.load(vp)['valid'].astype(bool)
                    if vm.shape[0] != kpx0:
                        vm = np.asarray(_Im.fromarray(vm.astype('u1') * 255).resize(
                            (kpx0, kpx0), _Im.NEAREST)) > 127
                pgd = PHOTO.pages[int(pg)]
                st_ = photo_match(fp, mp, pgd['box'], PHOTO.page_m, TINT, vm,
                                  feather=feather_ctx(), drawn=drawn_target())
                photo_stat.setdefault('match', []).append(dict(page=int(pg), **st_))
            fp = mp
        # The photograph is flat below its own texel, which is what a grazing
        # camera spends most of its pixels on; give it aggregate-scale relief
        # without moving the colour it measured.
        det = None
        if not a.no_ground_detail:
            from PIL import Image as _Im
            _Im.MAX_IMAGE_PIXELS = None
            kpx = _Im.open(fp).size[0]
            # the resolution and the config go in the name, so a --texscale run
            # and a retune never reuse each other's pages
            # and the source fingerprint, or a re-baked photo page
            # reuses the detail page made from the previous bake
            dp = (f"{TEXOUT}/photo_atlas_{pg:03d}_{kpx}_{src_tag(fp)}_det"
                  f"{GDET_TAG}.jpg")
            if not os.path.exists(dp):
                m0, m1 = apply_ground_detail(fp, dp, PHOTO.page_m)
                photo_det_shift.append(float(np.abs(m1 / np.maximum(m0, 1e-6)
                                                    - 1.0).max()))
                photo_det_raw.append(apply_ground_detail.last_uncorrected)
            fp = dp
            det = True
        ti = w.add_image_file(fp, wrap=1)
        mi = w.add_material_tex(ti, f"photo_ground_{pg:03d}")
        m = w.js['materials'][mi]
        m['pbrMetallicRoughness']['metallicFactor'] = 0.0
        m['pbrMetallicRoughness']['roughnessFactor'] = 0.90
        m['extras'] = PHOTO.extras(pg)
        if det:
            gd = GROUND_DETAIL
            m['extras']['detail_map'] = (
                "generated here (s22_composite.ground_detail): a tileable "
                "achromatic multiplicative field, mean 1.0 in linear light, "
                f"{gd['octaves']}-octave periodic Gaussian noise from "
                f"{gd['detail_m']} m down to the page's own texel grid, "
                f"relative amplitude {gd['amp']}; no asset was downloaded for "
                "it and it carries no licence of its own")
            m['extras']['detail_map_note'] = (
                "applied to the photograph in linear light and renormalised, "
                "so the page's mean linear RGB moves by under 1 %: it adds the "
                "aggregate-scale structure the panorama could not resolve, "
                "not colour.  Disable with --no-ground-detail.")
        phcache[pg] = mi
        photo_tex_bytes += os.path.getsize(fp)
        return mi

    def emit_floor(nodename, layer, faces_idx, pos_lookup, faces_arr, mname, ax,
                   facecol, facearea, extras, world_tint=False, simplify=None):
        """Split a floor group into the faces the atlas covers and the rest."""
        if PHOTO is None or ax != 0 or len(faces_arr) < 1:
            return emit(nodename, layer, faces_idx, pos_lookup, faces_arr, mname,
                        ax, facecol, facearea, extras, world_tint=world_tint,
                        simplify=simplify)
        used = np.unique(faces_arr.ravel())
        pu = pos_lookup(used)
        ri = np.searchsorted(used, faces_arr.ravel()).reshape(-1, 3)
        tri = pu[ri][:, :, :2]
        pgf = PHOTO.face_page(tri)
        if mname in PHOTO_DROP_CLASSES and TINTC is not None and (pgf >= 0).any():
            _ts = TINT_DECIDE.get(mname) or TINTC.get(mname) or TINT
            for _pg in np.unique(pgf[pgf >= 0]):
                if int(PHOTO.pband[_pg]) not in PHOTO_DROP_BANDS:
                    continue
                _sel = np.nonzero(pgf == _pg)[0]; _c = tri[_sel].mean(1)
                _w = np.asarray(_ts.at(_c[:, 0], _c[:, 1]), 'f8').mean(0)
                _gs = _w[1] / max(_w.sum(), 1e-6)
                if (_gs >= LAWN_GREEN_MIN) if mname == 'lawn' else (_gs < HARD_GREEN_MAX):
                    pgf[_sel] = -1
                    photo_stat['dropped_faces'] = photo_stat.get('dropped_faces', 0) + len(_sel)
                else:
                    photo_stat['kept_coarse_faces'] = photo_stat.get('kept_coarse_faces', 0) + len(_sel)
        _gpc = getattr(C, 'GROUND_PHOTO_COLOUR', None)
        if PC_FEAT is not None and _gpc and mname in _gpc['classes'] and (pgf >= 0).any():
            _fm, _fx0, _fy0, _fc = PC_FEAT; _c = tri.mean(1)
            _j = np.clip(((_c[:, 0] - _fx0) / _fc).astype(int), 0, _fm.shape[1] - 1)
            _i = np.clip(((_c[:, 1] - _fy0) / _fc).astype(int), 0, _fm.shape[0] - 1)
            _drop = (pgf >= 0) & ~_fm[_i, _j]
            photo_stat['photo_colour_dropped_faces'] = photo_stat.get('photo_colour_dropped_faces', 0) + int(_drop.sum())
            photo_stat['photo_colour_feature_faces'] = photo_stat.get('photo_colour_feature_faces', 0) + int(((pgf >= 0) & ~_drop).sum())
            pgf[_drop] = -1
        ex2 = dict(extras)
        ex2['tint'] = ("none; the baked best-view photograph carries the colour, "
                       "COLOR_0 is white")
        ex2['colour_source'] = PHOTO_CREDIT
        for pg in np.unique(pgf[pgf >= 0]):
            sel = np.nonzero(pgf == pg)[0]
            if len(sel) < 1:
                continue
            emit(f"{nodename}_ph{int(pg):03d}", layer, None, pos_lookup,
                 faces_arr[sel], mname, ax, None, None,
                 dict(ex2, material_class_under=mname), atlas=int(pg),
                 simplify=simplify)
            photo_stat['faces'] = photo_stat.get('faces', 0) + len(sel)
        rest = np.nonzero(pgf < 0)[0]
        if HARD_RC and mname in GROUND_HARD_RECLASS['classes'] and world_tint and len(rest):
            for _hc in sorted(HARD_RC):
                if _hc not in means or not len(rest):
                    continue
                _c = tri[rest].mean(1)
                _t3 = pu[ri[rest]]; _n = np.cross(_t3[:, 1] - _t3[:, 0], _t3[:, 2] - _t3[:, 0])
                _flat = np.abs(_n[:, 2]) >= GROUND_HARD_RECLASS['min_up'] * np.maximum(np.linalg.norm(_n, axis=1), 1e-12)
                _i = np.clip(((_c[:, 1] - HARD_G[1]) / HARD_G[2]).astype(np.int64), 0, int(HARD_G[4]) - 1)
                _j = np.clip(((_c[:, 0] - HARD_G[0]) / HARD_G[2]).astype(np.int64), 0, int(HARD_G[3]) - 1)
                _g = HARD_RC[_hc][_i, _j] & _flat
                if _g.sum() >= 3:
                    emit(nodename + '_rc' + _hc, layer, None, pos_lookup, faces_arr[rest[_g]], _hc, ax, None, None,
                         dict(extras, reclass=f'{mname}->{_hc} (ortho and panoramas measure hard ground)'),
                         world_tint=True, simplify=simplify)
                    photo_stat['hard_reclass_faces'] = photo_stat.get('hard_reclass_faces', 0) + int(_g.sum())
                    rest = rest[~_g]
        if GREY_LAWN_RECLASS and mname == 'lawn' and world_tint and TINTC is not None and len(rest) and 'paving' in means:
            _c = tri[rest].mean(1); _w = np.asarray((TINTC.get('lawn') or TINT).at(_c[:, 0], _c[:, 1]), 'f8')
            _gs = _w[:, 1] / np.maximum(_w.sum(1), 1e-6)
            _sat = (_w.max(1) - _w.min(1)) / np.maximum(_w.max(1), 1e-6)
            _g = (_gs < GREY_LAWN_GS) & (_sat < GREY_LAWN_SAT)
            if _g.sum() >= 3:
                _lt = TINTC.get('lawn') or TINT
                emit(nodename + '_rcpav', layer, None, pos_lookup, faces_arr[rest[_g]], 'paving', ax, None, None,
                     dict(extras, reclass='lawn->paving (world colour grey)'), world_tint=True,
                     tint_fn=lambda p, cw: _lt.at(p[:, 0], p[:, 1]).astype('f4'), simplify=simplify)
                photo_stat['reclass_faces'] = photo_stat.get('reclass_faces', 0) + int(_g.sum())
                rest = rest[~_g]
        photo_stat['faces_material'] = photo_stat.get('faces_material', 0) + len(rest)
        if len(rest) < 1:
            return None
        return emit(nodename, layer, faces_idx, pos_lookup, faces_arr[rest],
                    mname, ax,
                    None if facecol is None else facecol[rest],
                    None if facearea is None else facearea[rest],
                    extras, world_tint=world_tint, simplify=simplify)

    stats = dict(parts={}, layers={}, materials={})
    simp_notes = []
    ntri = 0
    dens_a = []; dens_d = []
    tint_stat = []
    clip_stat = []
    water_stat = []                # the water tint p99 cap
    dark_stat = []
    hard_stat = []

    def hard_floor(tint, mname, branch, ax):
        """Scale any still-near-black vertex up to HARD_TINT_FLOOR, hue kept.

        Runs after the ortho pass and also when there is no ortho to consult,
        so no vertex on a walked layer can leave this stage crushed to black.
        """
        lum = (tint @ T.LUMW).astype('f4')
        sel = lum < HARD_TINT_FLOOR
        if not sel.any():
            return tint
        g = (HARD_TINT_FLOOR / np.maximum(lum[sel], 1e-3))[:, None]
        tint[sel] = np.clip(tint[sel] * g, 0.03, 1.0)
        hard_stat.append((mname, branch, int(ax), int(sel.sum())))
        return tint

    def dark_floor(tint, pos, mname, layer, ax, M, branch):
        """The dark-tint floor.

        A vertex whose tint luminance is under DARK_TINT_FLOOR, on a layer the
        ortho covers, takes the ortho's own colour on that
        square metre instead.  The ortho may only lighten it.

        It runs in both tint branches (the pier decks and the 3dfier concrete
        read the world tint raster, not their own face colour), and on
        LAYER_structures it applies at any facing, so tilted dark flakes on
        structures are lifted too; on the other layers it applies to up-facing
        faces (projection axis 0) only.

        COLOR_0 only.  No position, normal, UV or index is touched.
        """
        if mname == 'water' or layer not in DARK_LIFT_LAYERS:
            return tint
        if ORTHOTINT is None or (ax != 0 and layer != 'LAYER_structures'):
            return hard_floor(tint, mname, branch, ax)
        om, ov = ORTHOTINT
        lum = (tint @ T.LUMW).astype('f4')
        okv = ov.at(pos[:, 0], pos[:, 1])[:, 0] > 0.5
        sel = (lum < DARK_TINT_FLOOR) & okv
        if not sel.any():
            return tint
        ot = np.clip(om.at(pos[sel, 0], pos[sel, 1]).astype('f4') /
                     np.maximum(M, 1e-4)[None, :], 0.03, 1.0)
        ol = (ot @ T.LUMW).astype('f4')
        keep = ol > lum[sel]            # never darken a surface
        ii = np.nonzero(sel)[0][keep]
        if len(ii):
            dark_stat.append((mname, branch, int(ax), len(ii),
                              float(lum[ii].sum()), float(ol[keep].sum())))
            if DARK_LIFT_HUE == 'tint':
                ot = _lift_keep_hue(tint[sel], lum[sel], ot)
            tint[ii] = ot[keep]
        return hard_floor(tint, mname, branch, ax)
    refstats = {}
    tjstats = {}

    def emit(nodename, layer, faces_idx, pos_lookup, faces_arr, mname, ax,
             facecol, facearea, extras, world_tint=False, simplify=None,
             atlas=None, tint_fn=None):
        """One primitive: the given faces, planar-projected UVs, per-vertex tint.

        With `atlas` set the primitive instead carries a photographic ground
        page (when an atlas is supplied): UV is the vertex's own world XY inside that 32 m
        page, COLOR_0 is white, and the baked photograph is the base colour."""
        nonlocal ntri
        f = faces_arr
        used, inv = np.unique(f.ravel(), return_inverse=True)
        pos = pos_lookup(used)
        # Weld by position first.  The photogrammetric mesh carries one vertex
        # per UV corner, so 39 % of its vertices are positional duplicates
        # (measured).  Left unwelded, every one of those source texture seams gets
        # its own normal and shades as a hard crease (faceted shading), and
        # the file carries duplicates it does not need, because the UVs here
        # are a function of position.
        wkey = _poskey(pos)
        uw, wfirst, winv = np.unique(wkey, return_index=True, return_inverse=True)
        pos = pos[wfirst]
        used = used[wfirst]
        idx = winv[inv].reshape(-1, 3).astype('u4')
        scale = ground_period(mname) if ax == 0 else TILE_M[mname]
        uv = uv_for(pos, ax, scale)
        nrm = smooth_normals(pos, idx)
        # (common.py GROUND_STEP_SHADE): a low floor step is shaded as the
        # ground it belongs to; a vertex any taller step touches keeps its normal
        if (GROUND_STEP_SHADE and ax != 0 and atlas is None and mname != 'water'
                and nodename.startswith(('ground_terrain', 'path', 'stitch_'))):
            _zt = pos[idx][:, :, 2]
            _tall = np.zeros(len(pos), bool)
            _tall[idx[(_zt.max(1) - _zt.min(1)) > GROUND_STEP_SHADE['max_step_m']].ravel()] = True
            nrm = np.array(nrm, copy=True)
            nrm[~_tall] = (0.0, 0.0, 1.0)
            photo_stat['step_shaded_vertices'] = photo_stat.get('step_shaded_vertices', 0) + int((~_tall).sum())
            photo_stat['step_kept_vertices'] = photo_stat.get('step_kept_vertices', 0) + int(_tall.sum())
        M = means[mname]
        if atlas is not None:
            uv = PHOTO.uv(pos, atlas)
            col8 = np.full((len(pos), 4), 255, 'u1')
            tint_stat.append(np.ones(len(pos), 'f4'))
            pos = C.yup(pos); nrm = C.yup(nrm)
            do_simp = bool(simplify)
            if a.web and len(idx) >= 2000 and do_simp:
                pos, nrm, uv, col8, idx, msg = simplify_group(
                    pos, nrm, uv, col8, idx, "wb" + str(len(stats['parts'])), a.ratio)
                if not msg.startswith('ok'):
                    simp_notes.append(f"{nodename}: {msg}")
            ni = w.child_node(nodename, pos, idx, uv=uv, color=col8, normal=nrm,
                              mat=photo_material(int(atlas)), extras=extras)
            kids[layer].append(ni)
            stats['parts'][nodename] = int(len(idx))
            stats['materials']['photographed_ground'] = \
                stats['materials'].get('photographed_ground', 0) + int(len(idx))
            ntri += len(idx)
            photo_stat['tris'] = photo_stat.get('tris', 0) + int(len(idx))
            return ni
        if world_tint:
            # the tint is a function of world position, read off the floor tint
            # raster at each vertex, so it cannot fan across a big triangle.
            # With stage 50 present the raster is this material class's own
            # layer, whose filtering never crossed the class boundary.
            tsrc = (TINTC.get(mname) or TINT) if TINTC else TINT
            cw = tsrc.at(pos[:, 0], pos[:, 1]).astype('f4')
            if mname == 'water':
                # the measured tint is a top-down projection of the water, which
                # carries no sun glitter and no sky at grazing incidence: at eye
                # level it reads as wet asphalt.  Lifted deliberately; the
                # measured mean is in the material extras.
                cw = cw * WATER_LIFT
                cw = water_local(cw, pos, tsrc)   # local-median knee
                # ...and it is near-neutral, which renders as wet ground.  Rebase
                # the hue onto the water ramp, keeping the luminance the lift
                # just set.
                cw = water_chroma(cw, knee=True)   # knee, see WATER_KNEE_X
            if tint_fn is not None:
                cw = tint_fn(pos, cw)
            cw, _ = cap_chroma(cw, np.full(len(cw), MI[mname], 'i1'))
            raw = cw / np.maximum(M, 1e-4)[None, :]
            clip_stat.append(np.array([float((raw > 1.0).any(1).mean()), len(raw)]))
            tint = np.clip(raw, RAW_CLIP_MIN, 1.0)
            if mname == 'water':
                # Water cap.  COLOR_0 on water is a neutral shade factor, not a
                # colour, and a small contiguous patch of water vertices can sit
                # at L 200-255, which renders as white smears on the water
                # surface itself (not a boat, a skirt or a tinted ground sliver).
                #   The cause is the 1.0 clip on `raw` above: WATER_LIFT multiplies the
                # measured top-down water colour by 1.55 before the divide, so
                # every vertex whose raster colour already sat at or above
                # M / 1.55 divides to raw >= 1.0 and the clip saturates it to
                # pure white.  The lift is wanted -- without it the water reads
                # as wet asphalt at eye level -- but it must not be allowed to
                # push a handful of vertices onto the ceiling, because 1.0 is a
                # hard stop and the rest of the water is nowhere near it.
                #   So cap water at its own batch's p99 instead of at 1.0.  The
                # water keeps its lift and its structure; only the few vertices
                # the lift drove into the clip come down, to the level the
                # water's own 99th percentile already sits at.
                wcap = float(np.percentile(tint, WATER_TINT_CAP_PCT))
                water_stat.append(np.array(
                    [float((tint > wcap).any(1).mean()), len(tint), wcap]))
                tint = np.minimum(tint, wcap)
            # the world raster is a measurement too, and where it measured a
            # pier deck or a slab of 3dfier concrete as near-black the ortho
            # overrides it exactly as it does in the photo-colour branch
            tint = dark_floor(tint, pos, mname, layer, ax, M, 'world_tint')
        else:
            # per-vertex tint = area-weighted mean of the faces that touch it
            tf = np.clip(facecol / np.maximum(M, 1e-4)[None, :], 0.03, 1.0)
            acc = np.zeros((len(used), 3), 'f8'); den = np.zeros(len(used), 'f8')
            wgt = np.repeat(facearea.astype('f8') + 1e-6, 3)
            flat = idx.ravel()
            for j in range(3):
                acc[:, j] = np.bincount(flat,
                                        weights=np.repeat(tf[:, j].astype('f8'), 3) * wgt,
                                        minlength=len(used))
            den = np.bincount(flat, weights=wgt, minlength=len(used))
            tint = np.where(den[:, None] > 0, acc / np.maximum(den[:, None], 1e-9), 0.5)
            tint = dark_floor(tint, pos, mname, layer, ax, M, 'photo_colour')
        if mname in CANOPY_RATIO:   # canopy fill on every colour source of the class, after the dark floors
            tint = np.clip(tint * CANOPY_RATIO[mname].at(pos[:, 0], pos[:, 1]), 0.03, 1.0)
        col8 = np.concatenate([np.clip(tint * 255, 0, 255).astype('u1'),
                               np.full((len(tint), 1), 255, 'u1')], 1)
        tint_stat.append(tint.mean(1))
        pos = C.yup(pos); nrm = C.yup(nrm)
        # A refined OSM floor must not be decimated again: the simplifier would
        # put the long edges straight back and the tint would fan across them.
        # The photo mesh is different -- its triangles are a measured 0.02 m2,
        # so even at the web copy's 12 % they stay far under the 2 m2 a fan
        # needs -- and a photo mesh of a million or more triangles left
        # undecimated would push the --web copy past its size target, so it is
        # simplified even where it reads the world raster.
        do_simp = (simplify if simplify is not None else not world_tint)
        if a.web and len(idx) >= 2000 and do_simp:
            pos, nrm, uv, col8, idx, msg = simplify_group(
                pos, nrm, uv, col8, idx, "wb" + str(len(stats['parts'])), a.ratio)
            if not msg.startswith('ok'):
                simp_notes.append(f"{nodename}: {msg}")
        # the projection axis is already the wall/ground decision: axis 0 is
        # projected from above, 1 and 2 are the vertical unwraps
        _mi = material(mname, 'ground' if ax == 0 else 'wall')
        ni = w.child_node(nodename, pos, idx, uv=uv, color=col8, normal=nrm,
                          mat=_mi, extras=extras)
        kids[layer].append(ni)
        stats['parts'][nodename] = int(len(idx))
        stats['materials'][mname] = stats['materials'].get(mname, 0) + int(len(idx))
        ntri += len(idx)
        return ni

    # ---------- the photogrammetric mesh ----------------------------------- #
    d = L.get('photo')
    if d is not None:
        P = np.load(C.G + '/pos.npy', mmap_mode='r')
        F = np.load(C.G + '/faces.npy', mmap_mode='r')
        TIL = np.load(C.G + '/facetile.npy')[:len(d['col'])]
        ax_all = axis_of(d['nrm'], d['cen'], force_top=(d['sublayer'] == 1))
        live = ~d['kill'] & (d['sublayer'] >= 1)
        if a.core:
            cc = d['cen']
            live &= ((cc[:, 0] >= CORE[0]) & (cc[:, 0] <= CORE[1]) &
                     (cc[:, 1] >= CORE[2]) & (cc[:, 1] <= CORE[3]))
        # The full model keeps the photogrammetry's own tiling, which is a
        # useful spatial split for swapping pieces.  The web copy does not: each
        # primitive there costs a weld+simplify round trip through
        # gltf-transform, so the cost grows with the primitive count and
        # hundreds of primitives cost far more than tens.
        tiles = [None] if a.web else list(range(int(TIL.max()) + 1))
        for tile in tiles:
            tm = live if tile is None else ((TIL == tile) & live)
            if tm.sum() < 3:
                continue
            for sub, (pre, layer) in PHOTO_SUB.items():
                sm = tm & (d['sublayer'] == sub)
                if sm.sum() < 3:
                    continue
                idxs = np.nonzero(sm)[0]
                key = d['mat'][idxs].astype(np.int64) * 4 + ax_all[idxs]
                uk, cnt = np.unique(key, return_counts=True)
                small = set(int(k) for k, c in zip(uk, cnt) if c < a.min_faces)
                if small and len(uk) > len(small):
                    big = int(uk[np.argmax(cnt)])
                    mv = np.isin(key, list(small))
                    d['mat'][idxs[mv]] = big // 4
                    ax_all[idxs[mv]] = big % 4
                for mi, mname in enumerate(MATERIALS):
                    mm = sm & (d['mat'] == mi)
                    if mm.sum() < 3:
                        continue
                    for ax in (0, 1, 2):
                        am = np.nonzero(mm & (ax_all == ax))[0]
                        if len(am) < 3:
                            continue
                        farr = np.asarray(F[am])
                        nm = (f"{pre}_{mname}_a{ax}" if tile is None
                              else f"{pre}_t{tile:02d}_{mname}_a{ax}")
                        # The photo GROUND that is kept is real, lidar-backed
                        # ground (for example low planting at lawn edges).  On
                        # its own projected colour it would not match the world
                        # raster the OSM lawn beside it reads, and the ground
                        # would read as a patchwork, so the floor sublayer
                        # reads the same class-confined raster as the terrain.  Structures
                        # (sublayer 2) keep their own colour: a boat or a shed
                        # is not the ground it stands on.
                        pw = (TINTC is not None and sub == 1 and ax == 0)
                        (emit_floor if (pw and sub == 1) else emit)(
                             nm, layer, am, lambda u: np.asarray(P[u]), farr,
                             mname, ax,
                             None if pw else d['col'][am],
                             None if pw else d['area'][am],
                             dict(layer=layer, material_class=mname,
                                  **({} if d.get('ovr') is None or not d['ovr'][am].any()
                                     else dict(colour_source=ORTHO_CREDIT)),
                                  source="the optional photo mesh layer; colour from "
                                         "Mapillary 360 panoramas",
                                  licence="Mapillary imagery CC BY-SA 4.0 (colour only); "
                                          "material texture CC0",
                                  tint=("per-vertex COLOR_0 read off the class tint "
                                        "raster at the vertex's own XY, the same "
                                        "raster the OSM floor uses" if pw else
                                        "per-vertex COLOR_0 = delighted multi-view "
                                        "colour / material mean")),
                             world_tint=pw, simplify=True)
                        if sub == 1 and ax == 0:
                            dens_a.append(d['area'][am])
                            dens_d.append(np.full(len(am), (TEX_PX[mname] / TILE_M[mname]) ** 2 if mname not in GROUND_SUPERTILE or GROUND_SUPERTILE[mname].get('mode') != 'hex' else (GROUND_SUPERTILE[mname]['px'] / GROUND_SUPERTILE[mname]['period_m']) ** 2))
        print(f"  photo mesh emitted, {ntri:,} tris, {time.time()-t0:.0f}s", flush=True)

    # ---------- (common.py GROUND_PHOTO_COLOUR): photo as colour ---- #
    # The listed classes draw their material everywhere; the photographed pages
    # give them colour only, as the band_m log band of the page mosaic added onto
    # the tint rasters (TINT and the class layer), faded to 0 over fade_m inside
    # the coverage edge, so stains and wear show but a page border or a mismatched
    # stone pattern cannot.  A page keeps drawing as full photo detail only on
    # "feature" cells: texels whose log-luma residual against a 0.5 m low-pass
    # exceeds knob x the drawn tile's own p99.5 of the same residual (markings,
    # drains, covers, text are far out of any tile's range), cleaned and grown.
    PC_FEAT = None
    GPC = getattr(C, 'GROUND_PHOTO_COLOUR', None)
    if GPC and PHOTO is not None and TINT is not None and TINTC:
        _t0 = time.time()
        from PIL import Image as _I
        from scipy import ndimage as _nd
        _cp = T.SCR + '/polish/class_map.npz'
        _cls = np.load(_cp)['cls'] if os.path.exists(_cp) else None
        if _cls is not None and np.asarray(TINT.M).shape[:2] == _cls.shape:
            c0 = float(TINT.c); ny_, nx_ = _cls.shape
            Xc = TINT.x0 + (np.arange(nx_) + 0.5) * c0; Yc = TINT.y0 + (np.arange(ny_) + 0.5) * c0
            XX, YY = np.meshgrid(Xc, Yc)
            own = PHOTO.page_at(XX.ravel(), YY.ravel()).reshape(ny_, nx_)
            cid = [MATERIALS.index(m) for m in GPC['classes'] if m in MATERIALS]
            incl = np.isin(_cls, cid)
            logP = np.zeros((ny_, nx_, 3), 'f4'); val = np.zeros((ny_, nx_), bool); fcnt = np.zeros((ny_, nx_), 'f4')
            thr = {}
            # the tile is resampled to the page texel (1.56 cm) whatever its own
            # size, so the --web --texscale builds get the same thresholds as model.glb;
            # the feature mask itself is cached (see _pcf below) and reused by them
            for m in GPC['classes']:
                if files.get(m):
                    _n = max(64, int(round(ground_period(m) / 0.015625)))
                    im = _I.open(files[m]['color']).convert('RGB').resize((_n, _n), _I.LANCZOS); tx = ground_period(m) / _n
                    Lt = np.log(np.maximum(T.srgb_to_linear(np.asarray(im)).astype('f4') @ T.LUMW, 0.005))
                    rt = np.abs(Lt - _nd.gaussian_filter(Lt, 0.5 / tx, mode='wrap'))
                    thr[MATERIALS.index(m)] = float(GPC['feature_knob'] * np.percentile(rt, 99.5))
            _pk = hashlib.md5((repr(sorted(GPC.items())) + repr([(p_['page'], os.path.getsize(PHOTO.jpeg(p_['page'])), int(os.path.getmtime(PHOTO.jpeg(p_['page']))))
                                                                for p_ in PHOTO.pages if os.path.exists(PHOTO.jpeg(p_['page']))])
                               + repr(np.asarray(_cls).shape) + str(int(os.path.getmtime(_cp)))).encode()).hexdigest()[:10]
            _pcf = f"{TEXOUT}/photo_colour_feat_{_pk}.npz"
            for pgi in np.unique(own[incl & (own >= 0)]):
                fpj = PHOTO.jpeg(int(pgi))
                if not os.path.exists(fpj):
                    continue
                pgd = PHOTO.pages[int(pgi)]; bx = pgd['box']; im = _I.open(fpj).convert('RGB')
                npx = im.size[0]; tx = PHOTO.page_m / npx; f = max(1, int(round(c0 / tx)))
                arr = T.srgb_to_linear(np.asarray(im)).astype('f4')
                lin_c = arr.reshape(npx // f, f, npx // f, f, 3).mean((1, 3))
                Lp = np.log(np.maximum(arr @ T.LUMW, 0.005))
                rp = np.abs(Lp - _nd.gaussian_filter(Lp, 0.5 / tx))
                sel = np.nonzero((own == pgi) & incl)
                jj = ((Xc[sel[1]] - bx[0]) / c0).astype(int); ii = ((Yc[sel[0]] - bx[1]) / c0).astype(int)
                okc = (jj >= 0) & (jj < npx // f) & (ii >= 0) & (ii < npx // f)
                sy, sx = sel[0][okc], sel[1][okc]; ii, jj = ii[okc], jj[okc]
                logP[sy, sx] = np.log(np.maximum(lin_c[ii, jj], 1e-4)); val[sy, sx] = True
                th = np.array([thr.get(int(c), 9.0) for c in _cls[sy, sx]], 'f4')
                rmax = rp.reshape(npx // f, f, npx // f, f)
                frac = np.zeros(len(sy), 'f4')
                for q in np.unique(th):
                    w_ = th == q
                    frac[w_] = (rmax[ii[w_], :, jj[w_], :] > q).mean((1, 2))
                fcnt[sy, sx] = frac
            wv = val.astype('f4')
            sa = GPC['band_m'][0] / c0; sb = GPC['band_m'][1] / c0
            da = np.maximum(_nd.gaussian_filter(wv, sa), 1e-4); db = np.maximum(_nd.gaussian_filter(wv, sb), 1e-4)
            fade = np.clip(_nd.distance_transform_edt(val) * c0 / GPC['fade_m'], 0, 1)
            B = np.zeros_like(logP)
            for ch in range(3):
                B[..., ch] = (_nd.gaussian_filter(logP[..., ch] * wv, sa) / da - _nd.gaussian_filter(logP[..., ch] * wv, sb) / db)
            B = np.clip(B, -GPC['max_log'], GPC['max_log']) * (fade * val)[..., None]
            G = np.exp(B).astype('f4')
            TINT.M = (np.asarray(TINT.M, 'f4') * G).astype('f4')
            for m in GPC['classes']:
                if TINTC.get(m) is not None and np.asarray(TINTC[m].M).shape[:2] == _cls.shape:
                    TINTC[m].M = (np.asarray(TINTC[m].M, 'f4') * G).astype('f4')
            if os.path.exists(_pcf):
                feat = np.load(_pcf)['feat'].astype(bool)   # same pages + config: the model.glb run's mask
            else:
                feat = _nd.binary_opening(val & (fcnt > GPC['feature_frac']), iterations=1)
                # a feature is a compact, isolated object (cover, drain, symbol), not a
                # surface pattern.  Without this filter, paver and joint grids (components over 2 m2,
                # up to hundreds of m2) pass as features and draw as hard-edged photo patches with black
                # joints.  Keep a core component only when its area is at most
                # feature_max_m2 and the core share of the coloured cells in a feature_iso_m box around
                # it stays at most feature_iso_max.
                _fmax = GPC.get('feature_max_m2'); _fiso = GPC.get('feature_iso_max')
                n_core = n_kept = 0
                if _fmax is not None or _fiso is not None:
                    _lab, n_core = _nd.label(feat)
                    _keep = np.ones(n_core + 1, bool); _keep[0] = False
                    if _fmax is not None:
                        _keep &= np.bincount(_lab.ravel(), minlength=n_core + 1) * c0 * c0 <= _fmax
                    if _fiso is not None and n_core:
                        _w = int(round(GPC.get('feature_iso_m', 3.0) / c0)) | 1
                        _d = _nd.uniform_filter(feat.astype('f4'), _w) / np.maximum(_nd.uniform_filter(val.astype('f4'), _w), 1e-3)
                        _keep &= np.concatenate([[0.0], np.asarray(_nd.maximum(_d, _lab, np.arange(1, n_core + 1)))]) <= _fiso
                    _keep[0] = False; n_kept = int(_keep.sum()); feat = _keep[_lab]
                feat = _nd.binary_dilation(feat, iterations=max(1, int(round(GPC['feature_grow_m'] / c0)))) & val
                photo_stat['photo_colour_feature_components'] = dict(core=int(n_core), kept=n_kept)
                np.savez_compressed(_pcf, feat=feat)
            PC_FEAT = (feat, TINT.x0, TINT.y0, c0)
            photo_stat['photo_colour'] = dict(
                cells_coloured=int(val.sum()), feature_cells=int(feat.sum()),
                feature_share=round(float(feat.sum() / max(val.sum(), 1)), 4),
                band_log_sd=round(float(B[val].std()), 4), tile_residual_thr={MATERIALS[k]: round(v, 3) for k, v in thr.items()},
                seconds=round(time.time() - _t0, 1))
            print(f"  photo as colour: {photo_stat['photo_colour']}", flush=True)

    # ---------- the OSM layers --------------------------------------------- #
    for name, layer in GROUP_TO_LAYER.items():
        d = L.get(name)
        if d is None:
            continue
        z = np.load(C.W + f'/layers/{name}.npz')
        meta = json.load(open(C.W + f'/layers/{name}.json'))
        pos_all = z['pos'].astype('f4'); idx_all = z['idx']
        # Floor faces steeper than axis_of()'s wall test get a wall unwrap.
        # Forcing EVERY floor face to axis 0 would also force the 3dfier stitch
        # walls: where the lawn polygon meets the flat water
        # polygon (or a path ribbon meets the terrain) 3dfier closes the height
        # step with a vertical wall, and a top-down projection on a vertical
        # face has one UV per column, so the lawn tile is smeared down the bank
        # as vertical streaks.  Those
        # faces are the ones whose OWN normal is steeper than axis_of()'s wall
        # test; they get a wall unwrap on the side axis their own normal picks.
        # Their COLOR_0 still reads the world raster at the vertex XY, which on
        # a vertical face is the colour of its top edge.
        if name in ('buildings', 'structures'):
            ft = None
        else:
            ft = np.ones(len(d['nrm']), bool)
            if name in FLOOR_WALL_LAYERS:
                ft = np.abs(d['nrm'][:, 2]) >= FLOOR_WALL_NZ
        ax_all = axis_of(d['nrm'], d['cen'], force_top=ft)
        if ft is not None and not ft.all():
            an = np.abs(d['nrm'][~ft])
            ax_all[~ft] = np.where(an[:, 0] >= an[:, 1], 1, 2).astype(ax_all.dtype)
        wt_layer = TINT is not None and name in FLOOR_LAYERS
        if wt_layer:
            tr0 = time.time()
            if name in TINT_EDGE_LAYERS and TINT_EMAX > 0:
                def _need(p, i, t, el):
                    ar = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0],
                                                       t[:, 2] - t[:, 0]), axis=1)
                    tl = (TINT.at(p[:, 0], p[:, 1]).astype('f8') @
                          T.LUMW.astype('f8'))[i]
                    dl = tl.max(1) - tl.min(1); mu = np.maximum(tl.mean(1), 1e-4)
                    return (((ar > TINT_AMIN) & (dl > TINT_TOL * mu)) |
                            (ar > TINT_AMAX) |
                            ((el.max(1) > TINT_EMAX) & (dl > TINT_ETOL * mu) &
                             (ar > TINT_EGATE * TINT_EMAX * TINT_EMAX)))
                rpos, ridx, rpar = refine_lepp(pos_all, idx_all, _need)
                rpos = rpos.astype('f4'); ridx = ridx.astype('u4')
                refine_for_tint.rounds = refine_lepp.rounds
            else:
                rpos, ridx, rpar = refine_for_tint(pos_all, idx_all, TINT,
                                                   tol_rel=TINT_TOL, a_min=TINT_AMIN,
                                                   a_max=TINT_AMAX)
            # weld the stitch skirt tops into this layer's boundary
            _sp = T.SCR + '/polish/stitch.npz'
            if (not a.no_stitch) and os.path.exists(_sp):
                _stp = np.load(_sp)['pos']; tjstats[name] = []
                # a second pass catches points that sit on a sub-edge made by the
                # pass before it (each pass inserts far fewer than the last); stop when
                # a pass inserts nothing
                for _pass in range(8):
                    rpos, ridx, rpar, _tj = weld_tjunctions(rpos, ridx, rpar, _stp)
                    tjstats[name].append(_tj)
                    print(f"  {name}: T-junction weld pass {_pass}: {_tj}", flush=True)
                    if not _tj.get('inserted'):
                        break
            refstats[name] = dict(triangles_before=int(len(idx_all)),
                                  triangles_after=int(len(ridx)),
                                  bisection_rounds=refine_for_tint.rounds,
                                  seconds=round(time.time() - tr0, 1))
            print(f"  {name}: refined {len(idx_all):,} -> {len(ridx):,} tris for the "
                  f"world tint, {time.time()-tr0:.0f}s", flush=True)
        for prt in meta['parts']:
            a0, b0 = prt['face_start'], prt['face_end']
            if b0 - a0 < 1:
                continue
            base = np.arange(a0, b0)
            live = base[~d['kill'][base]]
            if a.core and len(live):
                cc = d['cen'][live]
                live = live[(cc[:, 0] >= CORE[0]) & (cc[:, 0] <= CORE[1]) &
                            (cc[:, 1] >= CORE[2]) & (cc[:, 1] <= CORE[3])]
            if name == 'buildings' and use_fac and len(live):
                wf = live[FACE_ATLAS[live] >= 0]
                live = live[FACE_ATLAS[live] < 0]
                for pg in np.unique(FACE_ATLAS[wf]):
                    fs = wf[FACE_ATLAS[wf] == pg]
                    f3 = idx_all[fs]
                    p3 = pos_all[f3].reshape(-1, 3)
                    uv3 = WALL_UV[fs].reshape(-1, 2)
                    n3 = np.repeat(WALL_NRM[fs], 3, axis=0)
                    i3 = np.arange(len(p3), dtype='u4').reshape(-1, 3)
                    c3 = np.full((len(p3), 4), 255, 'u1')
                    nm = f"{prt['name']}_facade_a{int(pg)}"
                    ni = w.child_node(nm, C.yup(p3), i3, uv=uv3, color=c3,
                                      normal=C.yup(n3), mat=facade_material(int(pg)),
                                      extras=dict(layer=layer,
                                                  material_class='building_facade',
                                                  source=meta['source'],
                                                  licence=meta['licence'],
                                                  tint="none; the baked wall texture "
                                                       "carries the colour, COLOR_0 is white",
                                                  **prt.get('extras', {})))
                    kids[layer].append(ni)
                    stats['parts'][nm] = int(len(i3))
                    stats['materials']['building_facade'] = \
                        stats['materials'].get('building_facade', 0) + int(len(i3))
                    ntri += len(i3)
            if len(live) < 1:
                continue
            for mi, mname in enumerate(MATERIALS):
                mm = live[d['mat'][live] == mi]
                if len(mm) < 1:
                    continue
                for ax in (0, 1, 2):
                    am = mm[ax_all[mm] == ax]
                    if len(am) < 1:
                        continue
                    nm = f"{prt['name']}_{mname}" + (f"_a{ax}" if ax else "")
                    # The curb face of a path ribbon and the fascia of a deck
                    # are the near-vertical faces of these same layers.  Left on
                    # their own projected colour they come out a measured mean
                    # luminance of 0.066 against 0.091 for the tops, with a 10th
                    # percentile of 0.0 -- black -- because a 0.08 m sliver seen
                    # edge-on is what the panoramas measured worst.  Reading the
                    # world raster at their own XY paints them the colour of the
                    # ground they belong to, which is what closes the black line
                    # along every path and deck edge.
                    wt = wt_layer
                    ex = dict(layer=layer, material_class=mname,
                              source=meta['source'], licence=meta['licence'],
                              method=meta.get('method', '')[:400],
                              **prt.get('extras', {}))
                    if d.get('ovr') is not None and d['ovr'][am].any():
                        ex['colour_source'] = ORTHO_CREDIT
                    if wt:
                        ex['tint'] = ("per-vertex COLOR_0 read off the world-XY colour "
                                      "raster (%.2f m per texel, floor/%s) divided by "
                                      "the material mean; triangles bisected until a "
                                      "linear span across one holds under %.0f%% "
                                      "luminance change"
                                      % (TINT.c, os.path.basename(tint_path), 100 * TINT_TOL) +
                                      ("; that raster is the panorama tint where three or "
                                       "more views agreed and the " + ORTHO_CREDIT +
                                       " everywhere else" if use_ortho else ""))
                        sm = np.zeros(len(idx_all), bool); sm[am] = True
                        rs = np.nonzero(sm[rpar])[0]
                        if len(rs) < 1:
                            continue
                        (emit_floor if name != 'water' else emit)(
                             nm, layer, rs, lambda u: rpos[u], ridx[rs],
                             mname, ax, None, None, ex, world_tint=True)
                    else:
                        emit(nm, layer, am, lambda u: pos_all[u], idx_all[am],
                             mname, ax, d['col'][am], d['area'][am], ex)
                    if name == 'ground_terrain' and ax == 0:
                        dens_a.append(d['area'][am])
                        dens_d.append(np.full(len(am), (TEX_PX[mname] / TILE_M[mname]) ** 2 if mname not in GROUND_SUPERTILE or GROUND_SUPERTILE[mname].get('mode') != 'hex' else (GROUND_SUPERTILE[mname]['px'] / GROUND_SUPERTILE[mname]['period_m']) ** 2))
        stats['layers'][name] = dict(parts=len(meta['parts']))
        print(f"  {name} emitted, {ntri:,} tris cumulative", flush=True)

    # ---------- the stitch quads (optional input) --------------------------- #
    # The optional input polish/stitch.npz (pos, idx, mat, layer); skipped when
    # absent or with --no-stitch.
    # Vertical quads hung off every boundary edge of every floor layer, down to
    # the surface underneath: the terrain under a path ribbon or a deck, the
    # water plane at the shoreline, a fixed skirt at the model's outer rim.
    # Without them a draped ribbon that floats a measured 0.20 m over the
    # terrain shows the background through the gap under its lip from eye
    # height, which is the black sliver along every path.
    stitch_path = T.SCR + '/polish/stitch.npz'
    if TINT is not None and (not a.no_stitch) and os.path.exists(stitch_path):
        sz = np.load(stitch_path)
        spos = sz['pos'].astype('f4'); sidx = sz['idx'].astype('i8')
        smat = sz['mat']; slay = sz['layer']
        snames = [str(x) for x in sz['layers']]
        tri = spos[sidx]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(nrm, axis=1) + 1e-12
        # The stitch carries two kinds of quad: the vertical curb and fascia
        # faces, and the horizontal sheets (hole patches, the water backing and
        # the water fill).  A horizontal quad projected on a vertical plane
        # gets a constant V, so its texture smears into stripes, which show as
        # streaked bands along a path edge.
        # Pick the projection from the face normal, as every other layer does.
        up_s = np.abs(nrm[:, 2]) / ln
        sax = np.where(up_s > 0.70, 0,
                       np.where(np.abs(nrm[:, 0]) >= np.abs(nrm[:, 1]), 1, 2)).astype('i1')
        cen = tri.mean(1)
        # Which stitch rows are the building skirt: when polish/stitch.npz carries a
        # `tag` field, rows tagged 'skirt_*'; without one, the buildings-layer
        # stitch rows are the wall-base band.
        if 'tag' in sz.files:
            stag = np.asarray([str(x) for x in sz['tag']])
            is_skirt = np.char.startswith(stag, 'skirt_')
        else:
            is_skirt = ((slay == snames.index('buildings'))
                        if 'buildings' in snames else np.zeros(len(sidx), bool))
        fr = None
        # optional floor height raster (floor/floor_raster.npz: Z, grid), used
        # when present
        frp = FLOOR + '/floor_raster.npz'
        if os.path.exists(frp):
            _fz = np.load(frp)
            fr = (_fz['Z'], _fz['grid'])
        SK = make_skirt_blend(spos, sidx, np.nonzero(is_skirt)[0],
                              L.get('buildings'), floor_raster=fr)
        keep = np.ones(len(sidx), bool)
        if a.core:
            keep &= ((cen[:, 0] >= CORE[0]) & (cen[:, 0] <= CORE[1]) &
                     (cen[:, 1] >= CORE[2]) & (cen[:, 1] <= CORE[3]))
        nstitch = 0
        for li, lname in enumerate(snames):
            layer = GROUP_TO_LAYER.get(lname)
            if layer is None:
                continue
            for mi, mname in enumerate(MATERIALS):
                for ax in (0, 1, 2):
                    base = keep & (slay == li) & (smat == mi) & (sax == ax)
                    for sk in (False, True):
                        sel = np.nonzero(base & (is_skirt == sk))[0]
                        if len(sel) < 1:
                            continue
                        ex = dict(layer=layer, material_class=mname,
                                  source="stitch quads generated from the boundary "
                                         "edges of the OSM/lidar floor layers",
                                  licence="no third-party asset; generated by "
                                          "stages/s22_composite.py",
                                  method="vertical quad from the boundary edge down to "
                                         "the surface under it (terrain, water plane, or "
                                         "a fixed rim skirt), so no background is visible "
                                         "under a draped ribbon or at the shoreline",
                                  tint="per-vertex COLOR_0 read off the class tint raster "
                                       "at the quad's own XY")
                        fn = None
                        if sk and SK is not None:
                            fn = SK['blend']
                            ex['source'] = ("a skirt band generated under the wall's "
                                            "own bottom edge, merged into the stitch quads")
                            ex['method'] = ("vertical band from the wall bottom edge down "
                                            "to %.1f m under the floor beside it"
                                            % 1.0)
                            ex['tint'] = (
                                "per-vertex COLOR_0 blended vertically, so neither end "
                                "of the band is a straight step: the top %.2f m carries "
                                "the measured colour of the wall face the band hangs "
                                "from, everything at or below the floor line beside the "
                                "wall carries the class tint raster at the band's own XY "
                                "exactly as the rest of the stitch does, and the two are "
                                "smoothstepped across what is between.  The band is "
                                "subdivided here to carry the gradient: as built it has "
                                "vertices only at its top and bottom edge"
                                % SKIRT_BLEND_M)
                        if fn is not None:
                            ns = np.nonzero(np.isin(SK['src'], sel))[0]
                            if len(ns) < 1:
                                continue
                            pl, fa, ntr = (lambda u: SK['pos'][u]), SK['idx'][ns], len(ns)
                        else:
                            pl, fa, ntr = (lambda u: spos[u]), sidx[sel], len(sel)
                        emit(f"stitch_{lname}_{mname}_a{ax}" + ("_skirt" if sk else ""),
                             layer, sel, pl, fa, mname, ax,
                             None, None, ex, world_tint=True, tint_fn=fn)
                        nstitch += ntr
        floor_tint['stitch'] = dict(
            triangles=int(nstitch),
            skirt_triangles=int(is_skirt.sum()),
            skirt_blend=(SK['report'] if SK is not None
                         else make_skirt_blend.report or 'not applied'),
            report=json.load(open(T.SCR + '/polish/stitch.json'))
            if os.path.exists(T.SCR + '/polish/stitch.json') else None)
        print(f"  stitch quads emitted, {nstitch:,} tris", flush=True)

    # ---------- trees ------------------------------------------------------- #
    inv_doc = json.load(open(C.W + '/tree_inventory.json'))
    tp, ti_, tuv = unit_cylinder(10, uvs=True)
    cp_, ci_, cuv = unit_cylinder(14, uvs=True)
    tnrm = smooth_normals(tp, ti_)
    cnrm = smooth_normals(cp_, ci_)
    # the trunk borrows the plank material: run vertically on a cylinder it
    # reads as bark grain, and it means every primitive in the file carries the
    # same four attributes instead of the placeholders being a special case
    m_trunk = material('wood_deck', 'wall')   # a trunk is a vertical surface
    bark = np.clip(np.array([0.085, 0.058, 0.038], 'f4') /
                   np.maximum(means['wood_deck'], 1e-4), 0.03, 1.0)
    tcol = np.tile(np.concatenate([np.clip(bark * 255, 0, 255).astype('u1'),
                                   np.array([255], 'u1')])[None, :], (len(tp), 1))
    fti = w.add_image_file(fol, wrap=1)
    m_crown = w.add_material_tex(fti, "tree_crown_placeholder")
    w.js['materials'][m_crown]['pbrMetallicRoughness']['roughnessFactor'] = 0.92
    w.js['materials'][m_crown]['extras'] = dict(
        texture_source="generated in s22_composite.foliage_texture()",
        licence="no third-party asset; generated by stages/s22_composite.py",
        material_class="foliage_placeholder")
    mesh_trunk = w.add_mesh("unit_trunk", tp, ti_, uv=tuv * [1.5, 1.2],
                            color=tcol, normal=tnrm, mat=m_trunk)
    ntree_tri = 0
    crown_meshes = {}
    for k in range(6):
        v = 0.72 + 0.10 * k                       # per-tree foliage brightness
        col = np.tile(np.clip(np.array([[255 * v * 0.92, 255 * v,
                                         255 * v * 0.80, 255]], 'f4'),
                              0, 255).astype('u1'), (len(cp_), 1))
        crown_meshes[k] = w.add_mesh(f"unit_crown_{k}", cp_, ci_, uv=cuv,
                                     color=col, normal=cnrm, mat=m_crown)
    for tr in inv_doc['trees']:
        if a.core and not (CORE[0] <= tr['x_enu_m'] <= CORE[1] and
                           CORE[2] <= tr['y_enu_m'] <= CORE[3]):
            continue
        cb = max(0.6, tr['crown_base_m']); ch = max(0.8, tr['height_m'] - cb)
        rt = tr['trunk_radius_m']; rc = tr['crown_radius_m']
        # sha1, not hash(): Python salts str hashes per process
        # (PYTHONHASHSEED), so hash() would change a crown's brightness bin
        # (COLOR_0) between runs and two builds of the same inputs would
        # differ in their tree crowns.
        k = int(hashlib.sha1(tr['id'].encode()).hexdigest()[:8], 16) % 6
        trunk = w.node(tr['id'] + "_trunk", mesh=mesh_trunk, scale=(rt, cb, rt))
        crown = w.node(tr['id'] + "_crown", mesh=crown_meshes[k],
                       translation=(0.0, cb, 0.0), scale=(rc, ch, rc))
        n = w.node(tr['id'], children=[trunk, crown],
                   translation=(tr['x_enu_m'], tr['ground_z_enu_m'], -tr['y_enu_m']),
                   extras={kk: vv for kk, vv in tr.items() if kk not in ('evidence',)} |
                          dict(layer='LAYER_trees', placeholder=True,
                               swap_note="replace this node's children with a tree asset; "
                                         "translation is the trunk base, +Y is up, metres"))
        kids['LAYER_trees'].append(n)
        ntree_tri += len(ti_) + len(ci_)
    ntri += ntree_tri
    stats['parts']['trees_placeholders'] = ntree_tri
    stats['trees'] = dict(nodes=len(inv_doc['trees']), triangles=ntree_tri)

    for k, v in kids.items():
        if not v:
            continue
        w.node(k, children=v, root=True,
               extras=dict(layer_role=LAYER_DESC[k], children=len(v)))
        stats['layers'].setdefault(k, {})['nodes'] = len(v)
    raw = out_path if not a.draco else (C.W + '/tmp/raw_' + os.path.basename(out_path))
    nb = w.write(raw)
    if a.draco:
        import subprocess
        env = dict(os.environ, NODE_OPTIONS="--max-old-space-size=8192")
        r = subprocess.run([GT, "draco", raw, out_path], capture_output=True,
                           text=True, cwd=OCP.GLTF_DIR, env=env)
        if os.path.exists(out_path):
            os.unlink(raw)
        else:
            print("draco failed:", r.stderr[-400:]); os.rename(raw, out_path)
        nb = os.path.getsize(out_path)

    # ---- measured density -------------------------------------------------- #
    dd = np.concatenate(dens_d); aa = np.concatenate(dens_a)
    o = np.argsort(dd); cw = np.cumsum(aa[o])
    q = lambda p: float(dd[o][np.searchsorted(cw, cw[-1] * p)])
    cm = lambda x: round(100.0 / math.sqrt(max(x, 1e-9)), 3)
    stats['ground_texel_density'] = dict(
        ground_area_m2=round(float(aa.sum()), 1),
        texels_per_m2_median=round(q(0.5), 1),
        cm_per_texel_median=cm(q(0.5)),
        cm_per_texel_sharpest_decile=cm(q(0.9)),
        cm_per_texel_coarsest_decile=cm(q(0.1)),
        note="tileable material detail; the unique photographic information is "
             "the per-vertex tint, whose resolution is the mesh face size")
    stats['model_glb_bytes'] = nb
    stats['model_triangles'] = int(ntri)
    # every colour tile the model actually references, which
    # means the wall bake of a class too, wherever a vertical part asked for one
    used_col = set()
    for mn, v in files.items():
        if not v:
            continue
        for u in ('ground', 'wall'):
            if u == 'ground' or mn in cdet_used.get(u, ()):
                fpc = (v.get('color_by_use') or {}).get(u)
                if fpc:
                    used_col.add(fpc)
        used_col.add(v['color'])
    tex_bytes = sum(os.path.getsize(f) for f in
                    sorted(used_col) +
                    [v['normal'] for v in files.values() if v and v.get('normal')] + [fol])
    stats['texture_bytes'] = int(tex_bytes + fac_tex_bytes)
    stats['tint_mean'] = round(float(np.concatenate(tint_stat).mean()), 3)
    if clip_stat:
        cs = np.array(clip_stat)
        floor_tint['world_tint'] = dict(
            primitives=int(len(cs)),
            vertices=int(cs[:, 1].sum()),
            frac_vertices_clipped_at_one=round(
                float((cs[:, 0] * cs[:, 1]).sum() / max(cs[:, 1].sum(), 1)), 4),
            tolerance_rel_luminance=TINT_TOL, min_triangle_area_m2=TINT_AMIN,
            max_edge_m=dict(layers=list(TINT_EDGE_LAYERS), metres=TINT_EMAX,
                            tolerance_rel_luminance=TINT_ETOL, area_gate=TINT_EGATE,
                            method='conforming longest-edge bisection + same-parent flips'),
            refinement=refstats, tjunction_weld=tjstats)
    if water_stat:
        ws = np.array(water_stat)
        floor_tint['water_tint_cap'] = dict(
            primitives=int(len(ws)), vertices=int(ws[:, 1].sum()),
            percentile=WATER_TINT_CAP_PCT,
            frac_vertices_capped=round(
                float((ws[:, 0] * ws[:, 1]).sum() / max(ws[:, 1].sum(), 1)), 4),
            cap_min=round(float(ws[:, 2].min()), 4),
            cap_max=round(float(ws[:, 2].max()), 4),
            why="WATER_LIFT can push water vertices onto the hard 1.0 tint "
                "clip, which renders as white smears; capping at the batch's "
                "own p99 keeps the lift and drops the ceiling")
    if hard_stat:
        hv = sum(n for _, _, _, n in hard_stat)
        print('  hard tint floor %.2f lifted %d vertices on %d primitives'
              % (HARD_TINT_FLOOR, hv, len(hard_stat)), flush=True)
        floor_tint['hard_tint_floor'] = dict(
            luminance_floor=HARD_TINT_FLOOR, vertices=int(hv),
            primitives=len(hard_stat),
            what=('the floor of last resort: a vertex on '
                  + '/'.join(DARK_LIFT_LAYERS) + ' still under this luminance '
                  'after the ortho pass is scaled up to it, hue kept'))
    if dark_stat:
        bym = {}
        byb = {}
        for mn, branch, axv, n, lb, la in dark_stat:
            e = bym.setdefault(mn, [0, 0.0, 0.0])
            e[0] += n; e[1] += lb; e[2] += la
            k = f"{branch}_ax{axv}"
            byb[k] = byb.get(k, 0) + n
        # the vertices outside the up-facing photo-colour branch: the whole
        # world-tint branch, plus the non-up-facing structures faces
        newly = sum(v for k, v in byb.items()
                    if k.startswith('world_tint') or not k.endswith('ax0'))
        floor_tint['dark_tint_floor'] = dict(
            rule=("a vertex under %s whose measured tint luminance is below "
                  "%.2f, and which the ortho covers, takes "
                  "the ortho's own colour there; the ortho may only lighten it. "
                  "Applies in both tint branches, at any facing on "
                  "LAYER_structures and up-facing only on the other layers"
                  % ('/'.join(DARK_LIFT_LAYERS), DARK_TINT_FLOOR)),
            credit=ORTHO_CREDIT,
            vertices_lifted=int(sum(v[0] for v in bym.values())),
            vertices_lifted_by_branch_and_facing=byb,
            vertices_lifted_outside_up_facing_photo_branch=int(newly),
            by_material={k: dict(vertices=v[0],
                                 mean_tint_before=round(v[1] / max(v[0], 1), 4),
                                 mean_tint_after=round(v[2] / max(v[0], 1), 4))
                         for k, v in sorted(bym.items())})
    stats['floor_tint'] = floor_tint
    if use_fac:
        facades['atlas_pages_in_model'] = len(faccache)
        facades['atlas_bytes_in_model'] = int(fac_tex_bytes)
        facades['wall_triangles'] = int(stats['materials'].get('building_facade', 0))
    stats['facades'] = facades
    stats['ortho'] = ortho
    stats['photo_detail_retention'] = detail
    # what the photo hand-over and the step shading did
    stats['photo_ground_handover'] = dict(photo_stat, config=dict(
        photo_match=GROUND_PHOTO_MATCH, step_shade=GROUND_STEP_SHADE))
    stats['photo_ground_detail'] = (
        None if a.no_ground_detail else
        dict(GROUND_DETAIL, pages_written=len(photo_det_shift),
             worst_page_mean_linear_shift=round(max(photo_det_shift), 5)
             if photo_det_shift else None,
             median_page_mean_linear_shift=round(
                 float(np.median(photo_det_shift)), 5) if photo_det_shift else None,
             min_page_mean_linear_shift=round(min(photo_det_shift), 5)
             if photo_det_shift else None,
             worst_page_shift_before_mean_restore=round(max(photo_det_raw), 5)
             if photo_det_raw else None,
             note="mean-1.0 achromatic multiplicative field, applied in linear "
                  "light; the shift is the worst per-channel relative move of a "
                  "page's mean linear RGB and must stay under 0.01.  Each "
                  "page's mean is restored after the multiply, because the raw "
                  "multiply can leave a page over that ceiling.  "
                  "worst_page_shift_before_mean_restore is what the raw "
                  "multiply would have left."))
    def _cdet_use(use):
        sh = prep_textures.detail_shift.get(use, {})
        return dict(CLASS_DETAIL_USE[use],
                    tile_tag=CDET_TAG[use],
                    pages_written=len(sh),
                    materials=sorted(sh),
                    worst_page_mean_linear_shift=(
                        round(max(sh.values()), 5) if sh else None),
                    median_page_mean_linear_shift=(
                        round(float(np.median(list(sh.values()))), 5) if sh else None),
                    min_page_mean_linear_shift=(
                        round(min(sh.values()), 5) if sh else None),
                    worst_material=(max(sh, key=sh.get) if sh else None))

    _cd_all = {} if a.no_class_detail else {
        m: v for u in prep_textures.detail_shift
        for m, v in prep_textures.detail_shift[u].items()}
    stats['class_material_detail'] = (
        None if a.no_class_detail else
        dict(by_use={u: _cdet_use(u) for u in CLASS_DETAIL_USE},
             amp_by_use={u: c['amp'] for u, c in CLASS_DETAIL_USE.items()},
             pages_written=sum(len(v) for v in
                               prep_textures.detail_shift.values()),
             materials=sorted(set(_cd_all)),
             worst_page_mean_linear_shift=(
                 round(max(_cd_all.values()), 5) if _cd_all else None),
             median_page_mean_linear_shift=(
                 round(float(np.median(list(_cd_all.values()))), 5)
                 if _cd_all else None),
             min_page_mean_linear_shift=(
                 round(min(_cd_all.values()), 5) if _cd_all else None),
             worst_material=(max(_cd_all, key=_cd_all.get) if _cd_all else None),
             materials_by_use={u: sorted(v & set(_cd_all))
                               for u, v in cdet_used.items()},
             note="the same mean-1.0 achromatic multiply on the tiled class "
                  "colour tiles -- the materials the ground_terrain_3dfier_*, "
                  "stitch_* and path_* parts are drawn with.  One 'page' is one "
                  "class tile.  The amplitude is a property of the use, not "
                  "of the class, because eight of the nine classes are drawn "
                  "on both horizontal and vertical parts (concrete, for "
                  "example, on building_*, ground_*, path_* and stitch_* "
                  "parts).  A part whose UVs are projected from above (axis "
                  "0) takes the 'ground' bake, a vertically unwrapped one "
                  "(axes 1 and 2) the 'wall' bake at half the amplitude, which "
                  "is what stops a tall concrete wall reading as blotches.  Baking "
                  "that second tile for every class costs size, so it is baked only "
                  "for CLASS_DETAIL_WALL_CLASSES -- concrete and metal, the "
                  "only two with a measured improvement -- and every other "
                  "class falls back to its single ground tile and single "
                  "material, which keeps the file size down (and the --core crop's).  materials_by_use lists the classes each bake "
                  "actually reached in this model.  The shift is the worst "
                  "per-channel relative move the multiply introduces in a "
                  "tile's mean linear RGB and must stay under 0.01; the tile is "
                  "then renormalised to its measured mean, exactly as without the multiply, so "
                  "none of it reaches the model.  The generated water tile is "
                  "out of scope, see CLASS_DETAIL."))
    stats['simplify_ratio'] = a.ratio if a.web else None
    stats['simplify_notes'] = simp_notes[:40]
    stats['simplify_notes_total'] = len(simp_notes)
    print(f"\n{out_path} {nb/1e6:.1f} MB, {ntri:,} tris, {len(stats['parts'])} parts, "
          f"textures {tex_bytes/1e6:.1f} MB, {time.time()-t0:.0f}s")
    print("ground texel density:", json.dumps(stats['ground_texel_density']))
    print("triangles by material:", json.dumps(stats['materials']))
    tag = 'composite' + ('_web' if a.web else '') + ('_core' if a.core else '')
    json.dump(stats, open(C.W + f'/{tag}_stats.json', 'w'), indent=1)
    C.note(tag, {k: v for k, v in stats.items() if k != 'parts'})
    C.mark(tag)


if __name__ == '__main__':
    main()
