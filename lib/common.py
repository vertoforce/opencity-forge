"""Shared frame, OSM access and stage bookkeeping for an area build."""
import os, json, math, time, sys
try:
    import opencity_paths as OCP               # host roots, env-overridable
except ImportError:                            # imported with lib/ off sys.path
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import opencity_paths as OCP
import numpy as np

# Frame values.  A build always runs with OPENCITY_AREA set, and the area's
# area.yaml replaces every value below (paths, ORIGIN, REG_*, BBOX_WSEN).  With no
# area set (tests, tooling that only imports this module) they are neutral
# placeholders: a zero origin, an empty box and paths under the OPENCITY_* roots.
W   = OCP.CODE
CMB = OCP.COMBINED
G   = CMB + '/geom'
FW  = CMB + '/fusework'
CTX = OCP.SOURCE + '/context'
SRCM= OCP.DATA + '/photo_build'
PANO= OCP.SOURCE + '/panos'
OUT = OCP.DATA + '/model'
KEEP= OCP.WORK + '/keep'

ORIGIN = (0.0, 0.0, 0.0)                     # ENU origin, lat/lon/alt
# lidar -> photo frame registration (yaw + translation)
REG_YAW_DEG = 0.0
REG_T = np.zeros(3)
BBOX_WSEN = (0.0, 0.0, 0.0, 0.0)             # model extent, W, S, E, N

# ---- per-area config hook ---------------------------------------- #
# OPENCITY_AREA=<name> reads areas/<name>/area.yaml through pipeline/areacfg.py and
# overrides the frame values above (paths, ORIGIN, REG_*, BBOX_WSEN) here, before
# lla_to_enu captures ORIGIN; the area's knobs are applied at the end of this file.
# Unset, nothing is imported and every value stays as written.
_AREA_CFG = None
if os.environ.get('OPENCITY_AREA', '').strip():
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location('opencity_areacfg', os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pipeline', 'areacfg.py'))
    _areacfg = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_areacfg); del _spec, _ilu
    _AREA_CFG = _areacfg.load(_areacfg.active())
    _areacfg.apply_frame(globals(), _AREA_CFG)

def site(section, key, default=None):
    """Site data from the active area's area.yaml (render views, place lists):
    `section.key`, else `default`; KeyError when neither exists."""
    v = ((_AREA_CFG or {}).get(section) or {}).get(key)
    if v is None:
        v = default
    if v is None:
        raise KeyError(f'no {section}.{key} in the area config')
    return v

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3

def lla_to_ecef(lat, lon, alt):
    lat = np.radians(lat); lon = np.radians(lon)
    s = np.sin(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * s * s)
    return ((n + alt) * np.cos(lat) * np.cos(lon),
            (n + alt) * np.cos(lat) * np.sin(lon),
            (n * (1 - WGS84_E2) + alt) * s)

def lla_to_enu(lat, lon, alt, o=ORIGIN):
    lat0, lon0, alt0 = o
    x, y, z = lla_to_ecef(np.asarray(lat,'f8'), np.asarray(lon,'f8'), np.asarray(alt,'f8'))
    x0, y0, z0 = lla_to_ecef(lat0, lon0, alt0)
    dx, dy, dz = x-x0, y-y0, z-z0
    la, lo = math.radians(lat0), math.radians(lon0)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    return np.stack([-so*dx + co*dy,
                     -sl*co*dx - sl*so*dy + cl*dz,
                      cl*co*dx + cl*so*dy + sl*dz], axis=-1)

def enu_to_lla(e, n, u=0.0, o=ORIGIN):
    """Inverse of lla_to_enu, good to millimetres over a box a few hundred metres across."""
    lat0, lon0, alt0 = o
    la, lo = math.radians(lat0), math.radians(lon0)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    e = np.asarray(e,'f8'); n = np.asarray(n,'f8'); u = np.asarray(u,'f8')*np.ones_like(e)
    dx = -so*e - sl*co*n + cl*co*u
    dy =  co*e - sl*so*n + cl*so*u
    dz =            cl*n +    sl*u
    x0, y0, z0 = lla_to_ecef(lat0, lon0, alt0)
    x, y, z = x0+dx, y0+dy, z0+dz
    lon = np.degrees(np.arctan2(y, x))
    p = np.hypot(x, y); lat = np.arctan2(z, p*(1-WGS84_E2)); h = np.zeros_like(p)
    for _ in range(6):
        s = np.sin(lat); nn = WGS84_A/np.sqrt(1-WGS84_E2*s*s)
        h = p/np.cos(lat) - nn
        lat = np.arctan2(z, p*(1 - WGS84_E2*nn/(nn+h)))
    return np.degrees(lat), lon, h

def yup(P):
    """ENU (x=E, y=N, z=U) -> glTF Y-up (x, z, -y).  1 unit = 1 metre."""
    P = np.asarray(P); Y = np.empty_like(P)
    Y[:,0] = P[:,0]; Y[:,1] = P[:,2]; Y[:,2] = -P[:,1]
    return Y

# ---- stage bookkeeping --------------------------------------------------- #
def marker(name):  return f"{W}/markers/{name}.done"
def done(name):    return os.path.exists(marker(name))

def mark(name, info=None):
    with open(marker(name), 'w') as f:
        json.dump(info or {"at": time.strftime('%Y-%m-%dT%H:%M:%S')}, f)
    st = status_read()
    if name not in st['stages_done']: st['stages_done'].append(name)
    st['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    json.dump(st, open(W+'/status.json','w'), indent=1)

def status_read():
    try: return json.load(open(W+'/status.json'))
    except Exception:
        return {"area":((_AREA_CFG or {}).get("area") or {}).get("name"),"work_dir":W,"stages_done":[],"notes":{}}

def note(key, val):
    st = status_read(); st.setdefault('notes',{})[key] = val
    st['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    json.dump(st, open(W+'/status.json','w'), indent=1)

# ---- OSM ------------------------------------------------------------------ #
def osm_elements():
    """Raw Overpass elements for the model bbox (fetched by s02_osm)."""
    return json.load(open(W+'/osm/osm_raw.json'))['elements']

def osm_geoms():
    """Returns (nodes_by_id, ways) with way geometry in photo-ENU metres."""
    els = osm_elements()
    nb = {}
    for e in els:
        if e['type'] == 'node':
            nb[e['id']] = e
    ways = [e for e in els if e['type'] == 'way']
    return nb, ways

def way_xy(w):
    g = w.get('geometry')
    if not g: return None
    lat = np.array([p['lat'] for p in g]); lon = np.array([p['lon'] for p in g])
    return lla_to_enu(lat, lon, ORIGIN[2])[:, :2]


# ---- area: facades (per-area values read by s35_facadetex and stage 60) ---- #
# Wood-clad building walls are laid as vertical board siding by default: the CC0 wood
# tile is a horizontal deck plank, so stage 35 lays it turned 90 degrees on a building
# wall.  'horizontal' keeps lap/deck boards.  Ship hulls are not building walls and
# keep their planks.
FACADE_WOOD_BOARDS = 'vertical'
# Partial stage 60 ownership: a near wall whose owned photograph is under
# wpbake.PHOTO_MIN_AREA of the wall still ships it (photo where owned, the
# material fallback elsewhere) when the photograph covers at least this many
# square metres, so a full-height photographed strip on a long wall is not
# turned down by the fraction floor alone.
FACADE_PARTIAL_PHOTO_MIN_M2 = 12.0
# Photo-guided synthetic facade (the far grid).  Stage 34 also projects every
# panorama within FACADE_FAR_R_M of a wall onto a coarse FACADE_FAR_TEXEL_M grid;
# a wall whose near projection gives it no colour of its own takes the median of
# the far texels (unoccluded, not sky, not foliage) when they cover at least
# FACADE_FAR_MIN_M2, and its storey rows and bay pitch are read off the same grid.
# A far view is too soft to paste, but its colour and rhythm are still the building's.
FACADE_FAR_R_M = 60.0
FACADE_FAR_TEXEL_M = 0.25
FACADE_FAR_MIN_M2 = 8.0
# One building, one colour.  The far median is one colour per building part, so
# every drawn wall that takes it agrees with its siblings.  Without this rule a drawn
# wall whose near projection merely grazed it (trust on >= 0.03 of its texels, under
# the photo tint cut) skips the far median and keeps its OSM or class colour on the
# stage 35 neighbour tone match to a NEIGHBOUR's photographed wall, so two walls of one building
# draw at visibly different luminance.
# FACADE_FAR_GRAZE_TRUST: None = a drawn wall (no photo tint of its own) of a
# building the far grid measured always takes the far median, and the neighbour tone match is only
# the fallback for a building with no far measurement (the tone match is computed
# in either case and kept as level_reference for stage 60's exposure gate).
# A number keeps the neighbour tone match for walls grazed at or above that trust share.
FACADE_FAR_GRAZE_TRUST = None
# Facade completion.  On a partly photographed wall stage 60
# repeats the photographed module across the unseen part: horizontal shifts by the
# measured bay period, then vertical shifts by the measured storey period (never
# copying the ground storey upward).  Foliage and small saturated blobs (a sign's
# letters) are never a copy source.  No measured period, no completion.
# FACADE_FILL_MAX_RATIO caps the completed area at that multiple of the owned area.
FACADE_FILL = True
FACADE_FILL_BAY_M = (2.0, 12.0)
FACADE_FILL_STOREY_M = (2.6, 5.5)
FACADE_FILL_AC_MIN = 0.25
FACADE_FILL_MAX_RATIO = 6.0
# Block completion.  None = the texel mode: each unseen texel takes the nearest shift
# whose source texel is valid, so one hole is stitched from several shifts (staircase
# cuts, arrow-shaped pieces, pilaster bases repeated into the plinth, ghosted smears).
# With a dict, each bay-wide column block (edges on the flattest column-profile
# columns) and storey-high row block takes ONE shift, the nearest whose source covers
# min_src of its unseen texels; the rest stays material.
# Median mode.  min_src alone counts the foot band no photograph reaches anywhere, so
# a block can skip its neighbours and copy from a different wing several bays away
# (blank panels, a plinth sign repeated, a seam each bay).  So: holes are only unseen
# texels in rows the owned columns see at > row_min; a shift is kept only if source
# and target agree on the texels both see (sRGB-luma NCC >= ncc_min over >= ov_min of
# the seen rows); a block with no co-seen texel takes unverified shifts only at bay
# autocorrelation >= blind_ac; each hole texel is the median of the n_med best kept
# sources where >= n_min cover it (with a vote the accent veto is off: it would also
# drop every window pane); what is left, when at most row_hmax_m tall in its column
# and bounded by wall on both sides within row_gap_m, blends the row's median over
# row_win_m either side (plinth, sill and cornice bands continue); the rest stays
# material.
FACADE_FILL_BLOCK = dict(min_src=0.5, mode='median', ncc_min=0.5, ov_min=0.15, n_med=5, n_min=3,
                         blind_ac=0.6, row_fill=True, row_win_m=1.0, row_gap_m=12.0, row_min=0.15,
                         row_hmax_m=2.0)
# Unseen-texel blend.  An unseen photo texel (got False) and a texel the completion
# did not fill are undefined: the bake leaves them BLACK.  None = a blend whose 0.45 m
# feathers run outward over them, so every hole edge gets a dark fringe and every
# hole narrower than the feather a dark wedge.  'material': an undefined texel
# carries the material (or completion) colour before any feather (Porter-Duff: a
# transparent texel's colour must never reach the composite), so the feathers blend
# photo -> material only.
FACADE_BLEND_UNSEEN = 'material'
# Glass walls are drawn, never photographed.  A panorama's view of a glass curtain
# wall is a reflection: its tiles carry the street behind the camera, buildings
# across the road and the trees in front, as a dome-shaped island on the class grid.
# False: stage 60 draws the class grid over the whole wall, levelled per channel onto
# the building's own material glass walls (else the photo median).
FACADE_GLASS_PHOTO = False
# FACADE_GLASS_L_MAX: s35 classify()'s photo glass test is "dark, blue, strongly
# banded" with dark = mean linear L under this cut.  An area build gets 0.11
# (pipeline/areacfg.AREA_KNOB_DEFAULTS): a curtain wall whose photographs carry sky and
# cloud reflections measures L 0.09-0.10, above 0.085.
FACADE_GLASS_L_MAX = 0.085
# FACADE_GLASS_SIBLINGS: a wall the photograph calls glass hands that class to the
# unphotographed walls of its building (s35 sibling rule).  An area build gets False:
# one photographed glass bay would otherwise turn the whole building, its
# photographed front included, into synthetic curtain wall.
FACADE_GLASS_SIBLINGS = True
# Facade modules.  A synthetic wall is built from modules instead of
# painted rectangles: each opening gets a frame, a recessed pane, mullions and a
# sill; storey lines get a string course; the top a parapet cap; the foot a plinth
# and a dirt gradient; the ground storey a shopfront (fascia, bulkhead, glazed
# door) where OSM tags it retail/commercial or an amenity, or a lobby entrance
# where OSM tags an office/apartment/civic/institute.  Stage 35 bakes a height
# field (metres, into a tangent-space normal page) and a roughness page next to
# every colour page; stage 60 packs them, stage 22 attaches them to the facade
# material.  The measured colour, storey height and bay pitch stay the inputs;
# everything here is geometry and material, per building (seeded by the OSM id),
# never per wall id.  Lengths in metres, colours linear RGB.
FACADE_PBR = True
# texel size / tile cap of a synthetic wall beyond stage 35's NEAR_R: at 8 cm and
# 512 px a 70 m wall draws at 13.7 cm and a 7 cm window frame does not exist
FACADE_TEXEL_FAR = 0.05
FACADE_MAX_FAR = 1024
FACADE_NORMAL_SCALE = 1.0          # glTF normalTexture.scale on the facade pages
FACADE_MOD = dict(
    win_w_frac=(0.36, 0.62), win_h_frac=(0.46, 0.64),   # of bay / storey, per building
    mull=(('none', 0.15), ('v', 0.35), ('cross', 0.25), ('t', 0.25)),
    frame_m=0.07, mull_m=0.05, recess_m=0.14, frame_depth_m=0.05,
    sill_h_m=0.07, sill_over_m=0.07, sill_proj_m=0.05,
    band_p=0.6, band_h_m=0.22, band_proj_m=0.025, band_tone=1.07,
    parapet_m=0.55, parapet_proj_m=0.06, plinth_m=0.55, plinth_proj_m=0.02,
    frame_rgb=((0.62, 0.62, 0.60), (0.040, 0.035, 0.030), (0.20, 0.21, 0.22), (0.022, 0.022, 0.024)),
    glass_rgb=((0.045, 0.060, 0.080), (0.050, 0.068, 0.062), (0.050, 0.052, 0.056), (0.060, 0.052, 0.045)),
    sky_gain=(0.65, 1.9),              # pane bottom / top reflection multiplier on the glass colour
    blind_p=(0.10, 0.35), blind_rgb=(0.20, 0.19, 0.17), dark_p=0.12,
    curtain_band_p=0.55,               # curtain wall with opaque spandrel bands vs full glazing
    curtain_mull_m=1.5,
    shop_building=('retail', 'commercial'), shop_amenity_any=True,
    entrance_building=('office', 'apartments', 'civic', 'retail', 'commercial'),
    entrance_amenity_any=True,
    shop_h_frac=0.72, fascia_m=0.55, bulkhead_m=0.45, door_w_m=1.8, door_h_m=2.4,
    dirt_m=1.4, dirt_amt=(0.10, 0.22), dirt_rgb=(0.86, 0.80, 0.70),
    streak_amt=(0.04, 0.10), streak_m=1.2,
    tile_relief_m=0.012, macro_relief_m=0.30,
    rough=dict(brick=0.90, concrete=0.88, metal_panel=0.45, glass=0.30, wood=0.80),
    # glass roughness: at 0.05 the curtain walls mirror a viewer's studio
    # environment as white blocks, so it is blurred
    rough_glass=0.12, rough_curtain=0.24, curtain_blind=1.12, rough_frame=0.40, rough_sill=0.70, rough_dirt=0.05,
)


# ---- area config: ground class colour rules ------------------------- #
# World colour (the class tint, linear RGB) share of green G/(R+G+B) that makes a
# ground class "confident" for this area's lawns and hard ground.  Stage 22 drops a
# coarse photo page (16-32 texels/m) where the class is confident and draws the
# class tile + tint instead; grey "lawn" (green share and saturation both low) is
# drawn with the paving tile while keeping the lawn tint (grey_lawn_reclass, off by
# default).
GROUND_CLASS_COLOUR = dict(lawn_green_min=0.36, hard_green_max=0.37,
                           grey_lawn_green_max=0.345, grey_lawn_sat_max=0.12,
                           grey_lawn_reclass=False)


# ---- area config: ground class-tile detail (blotch) ----------------- #
# Stage 22's synthetic class-detail amplitude on a class's ground bake (default
# CLASS_DETAIL amp 0.20, tuned on the flat Concrete034).  Tiles that carry their own
# grain (Concrete047A aggregate, Asphalt012, the paving tile) get 0: on them the
# field's 0.25-1 m octaves triple the tile's own 0.1-1 m log-luma sd and draw dark
# ~1 m blotches with white speckle.  Gravel keeps the field: its tile has no
# structure above 0.1 m, so at amp 0 the gravel beds draw flat and featureless.
# Walls keep their own bake.
GROUND_DETAIL_AMP = dict(concrete=0.0, asphalt=0.0, paving=0.0)


# ---- area config: photographed-ground hand-over (ground) ----------- #
# Stage 22 re-levels every photographed ground page onto the class-as-sampled tint
# raster (floor/tint_class_ref*.npz, the colour a class-tile face at the same XY is
# drawn with) before it ships, so the photo | class-tile border carries no step and
# no page drifts to a colour cast.  sigma_m: the low-pass the page takes from the
# raster (its detail below that stays the photograph's own); lum_max / chroma_max:
# log clamps on that low-frequency gain; texel_chroma_max: the most a single texel's
# chroma (log RGB minus log luma) may sit off the raster's chroma, which removes
# magenta specks.  None disables it.
GROUND_PHOTO_MATCH = dict(sigma_m=1.0, cell_m=0.125, lum_max=1.2, chroma_max=0.6,
                          texel_chroma_max=0.15, feather_m=0.5, level_to='drawn')
# level_to: what the page is levelled onto.  'raster' is the tint raster as measured;
# 'drawn' is that raster after the same vertex-tint floors a class tile gets in stage
# 22 (DARK_TINT_FLOOR ortho lift, HARD_TINT_FLOOR), i.e. the colour the tile beside
# the page is actually drawn with.  Levelled onto 'raster', a page keeps the shade the
# floors lifted out of the tile beside it and draws a visible step; material factors,
# normal maps, KTX and mip filtering each change that step by under 0.5 L.
# feather_m: inside this distance of the edge of a page's photographed area the photo
# fades into the class tile's own look (tile x tint, sampled at the page texel), so
# the face where photo hands over to tile meets it without a step.
# Floor-layer faces that close a height step (3dfier polygon walls, stitch skirts)
# are shaded as the ground they belong to when the step is at most max_step_m tall:
# their horizontal normals would draw them as dark shards on flat lawn.  Taller banks
# keep their own shading.  None disables it.
GROUND_STEP_SHADE = dict(max_step_m=0.6)
# Dark-tint lift hue.  Stage 22 lifts a ground vertex whose tint is under
# DARK_TINT_FLOOR to the ortho where the ortho is brighter.  'ortho' takes the
# ortho's whole colour; 'tint' takes only its luminance and keeps the tint's own
# chroma (hue kept, as HARD_TINT_FLOOR does).  The ortho's tree and roof shadows are
# sky-blue, so 'ortho' renders blue-grey blotches between green ones.
# hue_min_lum: under this tint luminance the tint's chroma is noise (crushed-black
# photo_structure flakes), so those vertices take dark_hue: 'tile' = the class tile's
# own hue at the ortho's luminance, 'ortho' = the ortho's colour.
# None = 'ortho' colour for every lifted vertex.
GROUND_DARK_LIFT = dict(hue='tint', hue_min_lum=0.18, dark_hue='tile')


# ---- area config: ground tint relief ----------------------------- #
# The class tint rasters (s50) carry more metre-scale relief than the ground has:
# over lawn cells their log-luma sd in the 1-16 m band runs about 1.3x the ortho's on
# the same cells, which renders as brown/green blotches.  Stage 22 scales each listed
# class's band_m deviation down to the ortho's sd (never up, never below k_min; k
# fitted on luma, applied to each channel so hue blotches shrink too), inside the
# class mask, on the class layer and on the class-as-sampled raster alike so photo
# and tile stay matched.
GROUND_TINT_RELIEF = dict(band_m=(1.0, 16.0), k_min=0.5,
                          classes=('lawn', 'concrete', 'asphalt', 'paving', 'gravel'))

# Tint under tree crowns.  The ortho has no ground pixel under a lidar crown (canopy
# class) and the panoramas see the lawn there in the crown's shade (sky-lit: blue,
# dark) or not at all (views 0, filled), so the lawn tint draws purple-brown patches
# under every tree.  Stage 22 levels each listed class's low_m low-pass (log, per
# channel) onto the ref_m Gaussian mean of its open cells (> open_m from a crown),
# weight 1 under the crown fading to 0 at ring_m outside it; detail below low_m is
# kept, max_log caps the move; luma 'level' moves both ways, 'lift' never darkens.
# None = off.
GROUND_CANOPY_FILL = dict(classes=('lawn',), ring_m=3.0, open_m=6.0, ref_m=6.0, low_m=1.5, max_log=0.7, gain=1.0,
                          luma='level')

# Tint along a class edge.  A measured lawn tint greys out toward every lawn edge:
# the neighbouring path or kerb bleeds in through class-map misregistration and the
# panoramas' blur, while the ortho stays flat there.  The sparse_grass tile's own hue
# (log R/G +0.12, B/G -0.14) then draws that grey tint mauve-brown.  Stage 22 levels
# each listed class's low_m low-pass (log, per channel, BEFORE the canopy fill, which
# then levels crowns onto the edge-filled lawn; the other order leaves the crowns
# bluer than open lawn) onto the ref_m Gaussian mean of its own interior cells
# (farther than edge_m from any other class), weight 1 up to full_m from the edge
# fading to 0 at edge_m; detail below low_m is kept, max_log caps the move.
# off_class: a lawn-tile face on a cell the class map calls something else (OSM
# grass vs class-map mismatch) takes the interior colour at full weight.
# The factor is multiplied into the same final-vertex-tint ratio as the canopy fill.
# None = off.
GROUND_EDGE_FILL = dict(classes=('lawn',), full_m=1.0, edge_m=4.0, ref_m=6.0, low_m=0.75, max_log=0.7, gain=1.0,
                        luma='level', off_class=True)

# ---- area config: lawn drawn on measured hard ground ---- #
# Stage 21 can class OSM landuse=grass as lawn where the ortho and the panoramas both
# measure neutral hard ground (OSM lawn weight 2.5 beats a photo concrete score of at
# most 1.5).  Stage 22 would then draw the grass tile with that grey tint: grey-mauve
# dead grass.  Instead it draws such a region with the hard class stage 21's own
# photo_scores pick from the region's median ortho colour, tinted by that class's
# layer (colour and tile follow the measurement).
# A cell qualifies when its sigma_m Gaussian means (inside the terrain's lawn faces)
# of the ortho and of the panorama world tint are both neutral (log B/G > *_bg_min),
# the ortho is not shade or water (log B/G <= ortho_bg_max, luma >= ortho_luma_min),
# it is ring_m clear of any lidar crown (under a crown the ortho sees shade, not
# ground) and water_m from ortho water (a shore bank is riprap or mud, drawn as a
# jagged bright shard); then a disc opening of radius open_m (without it 1-2 m
# lawn-edge strips beside paths turn grey and ragged), regions >= min_area_m2 whose
# best hard score >= min_score and whose share of ortho cells with ExG > 0.03 is
# <= max_green_share, and only faces with |n_z| >= min_up.  None = off.
GROUND_HARD_RECLASS = dict(classes=('lawn',), to=('concrete', 'gravel', 'asphalt', 'paving'), sigma_m=0.5,
                           ortho_bg_min=-0.25, pano_bg_min=-0.25, ortho_bg_max=0.10, ortho_luma_min=0.07, ring_m=3.0,
                           water_m=3.0, open_m=1.0, min_up=0.95, min_area_m2=20.0, min_score=0.8, max_green_share=0.05)

# ---- area config: lawn drawn in tree shade ------------ #
# Lawn under and beside trees can still draw grey-blue after the canopy and edge
# fills: the ground is lawn (lit ortho pixels there are green), but the panoramas saw
# it in deep shade and the ortho cells are cast shadow.  The dark-tint lift and the
# canopy fill do not reach it: the canopy factor is fitted to the LAYER (not to the
# lifted colour it multiplies) and fades to 0 at 3 m from the lidar crown, while the
# cast shadow lies beside the crown, so the drawn low-pass colour ends bluer and
# brighter than the sunlit lawn around it.  Genuinely olive-brown open lawn has no
# blue shift and is left alone.
# Stage 22 shade_fill(): after the edge and canopy fills, rebuild the class's DRAWN
# colour on the tint grid (layer / tile mean -> ground_tint_floor, the same dark and
# hard floors emit() applies -> x the fills' factor), take its low_m low-pass (log),
# and measure its log B/G against the ref_m Gaussian mean of the class's sunlit open
# cells (> open_m from any lidar crown and, with lit_ref, an ortho-lit cell; widening
# x2 up to x8 where the local weight is under 0.02, class median beyond).  Weight 0
# at a blue shift of bg_min, 1 at bg_full; the low-pass is levelled onto the
# reference (per channel, 'level' moves luma too, 'chroma' keeps the drawn luma),
# max_log caps the move, gain scales it.  The factor multiplies the fills'
# CANOPY_RATIO, so it lands on the final vertex tint of every colour source of the
# class; carried carry_m past the class edge by nearest cell.  No raster or decision
# moves.  None = off.
GROUND_SHADE_FILL = dict(classes=('lawn',), open_m=6.0, ref_m=6.0, low_m=1.0, bg_min=0.10, bg_full=0.25,
                         max_log=0.7, gain=1.0, luma='level', lit_ref=True, carry_m=3.0)

# ---- area config: ground super tiles ------------------------------- #
# How games draw a lawn or a slab seen from 1 to 10 m: a detailed CC0 material at its
# real size, with the repetition hidden by hex-tile histogram-preserving blending
# (Heitz and Neyret, High-Performance By-Example Noise using a
# Histogram-Preserving Blending Operator; Mikkelsen, Practical Real-Time
# Hex-Tiling), and the colour at 1 m and up from the vertex tint (ortho/tint
# raster).  A glTF viewer has no custom shaders, so stage 22 BAKES the hex blend
# into one period_m x period_m tile (px texels) that the up-facing (ground-use) class
# parts repeat at period_m; walls keep TILE_M and the plain tile.  src: None keeps
# the class's index.json asset.  keep_chroma: the share of the lawn's green, straw
# and soil variation that is kept, with the tile renormalised per channel to the
# class mean; much above 0.55 the soil draws as a rusty red speckle.
# chroma_cov: the lawn tile's (log luma, log R/G, log B/G) covariance is coloured
# onto that of a rectified panorama lawn patch: the CC0 soil is R/G-rich and
# anti-correlated with luma, which draws a rust cast; real soil is grey-brown.
# sparse_grass is the CC0 candidate closest to a rectified panorama patch of real
# lawn in log-luma (16-50 cm) and green/red (4-50 cm) variation.  mode 'scale' only
# moves the ground repeat (paving: the tile's stones are scaled to real 0.2-0.25 m
# plaza pavers).
_PH = OCP.SHARED + '/texture/cc0/polyhaven_sparse_grass/'
GROUND_SUPERTILE = dict(
    lawn=dict(mode='hex', period_m=12.0, px=4096, normal_px=2048, src_m=2.0, hex_m=0.8,
              keep_chroma=0.55, seed=2601,
              chroma_cov=[[0.1159, 0.0088, 0.0474], [0.0088, 0.0284, 0.0343], [0.0474, 0.0343, 0.0658]],
              src=dict(color=_PH + 'sparse_grass_diff_4k.jpg', normal=_PH + 'sparse_grass_nor_gl_4k.jpg',
                       source='Poly Haven', asset_id='sparse_grass', title='Sparse Grass', licence='CC0 1.0',
                       licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
                       page_url='https://polyhaven.com/a/sparse_grass',
                       download_url='https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/sparse_grass/')),
    concrete=dict(mode='hex', period_m=12.0, px=4096, normal_px=2048, src_m=3.0, hex_m=1.1,
                  keep_chroma=None, seed=2602, src=None),
    paving=dict(mode='scale', period_m=2.7))
# ---- area config: photo as colour, material as detail ------------ #
# The listed classes draw their material everywhere.  The photo pages add only their
# band_m (Gaussian sigma, m) log band onto the tint rasters, clamped to +-max_log and
# faded in over fade_m from the coverage edge, so real stains and wear show at
# 0.5-1 m and coarser while page borders and mismatched stone patterns cannot.  A
# page keeps drawing as full photo only on feature cells: a 0.25 m cell where more
# than feature_frac of the page texels have a log-luma residual (vs a 0.5 m
# low-pass) over feature_knob x the drawn tile's own p99.5 of that residual, opened
# once and grown by feature_grow_m.  None disables it.
# feature_max_m2 / feature_iso_m / feature_iso_max: a feature is compact and
# isolated.  A core component (before the grow) keeps full photo only when its area
# is at most feature_max_m2 and the core share of the coloured cells in a
# feature_iso_m box around it is at most feature_iso_max.  Without these, the dark
# joint grids of paver photos on concrete pass the residual test across large
# connected areas.  feature_max_m2=0 keeps no photo detail anywhere.  None disables
# a test.
GROUND_PHOTO_COLOUR = dict(classes=('paving', 'concrete', 'lawn'), band_m=(0.25, 1.0), max_log=0.4,
                           fade_m=1.0, feature_knob=2.0, feature_frac=0.10, feature_grow_m=0.5,
                           feature_max_m2=0.0, feature_iso_m=3.0, feature_iso_max=0.25)

# ---- area config: sidewalk panel joints ------------------- #
# Stage 94 (s94_ground_joints.py, run by pipeline/nomesh_swaps.py) scores the listed
# ground materials into poured panels: a grid of dark contraction joints and a
# per-panel tone baked into the ground super tile, with each face's UVs rotated so
# the grid runs along the nearest OSM highway (roads and footways, orientation mod 90
# deg, Gaussian-weighted over orient_sigma_m, quantised to orient_q_deg; faces with
# no street within reach keep the world axes).  Real sidewalk joints measure 1-1.5
# cm wide, log-luma depth 0.33-0.48, panel sides 1.4-2.7 m; the Concrete047A super
# tile has no structure above 0.3 m.  panel_m is rounded so a whole number of panels
# fills the tile period.  Joints are skipped on a copy whose tile texel is coarser
# than joint_w_m / min_texel_frac.  None disables the stage.
GROUND_JOINTS = dict(materials=('mat_concrete',), panel_m=2.0, joint_w_m=0.015, joint_log=0.50,
                     panel_sd=0.04, orient_sigma_m=6.0, orient_q_deg=10.0, orient_min=0.15,
                     min_texel_frac=0.5, seed=2694, jpeg_q=92, chroma_keep=0.35, luma_keep=0.5, chroma_sigma_m=4.0,
                     edge_sigma_m=5.0, edge_min=0.05)
# edge_sigma_m / edge_min: where the material's own outline in the stage-21 class map
# (kerbs, lawn and building edges) is coherent within edge_sigma_m (strength >=
# edge_min of its p99.5), the grid follows that outline instead of the street, which
# can run tens of degrees off a slab's own kerb.  edge_sigma_m None = streets only.
# chroma_keep / chroma_sigma_m: the same stage pulls these materials' vertex-tint
# chroma (log RGB minus log luma) toward its own chroma_sigma_m Gaussian low-pass,
# keeping chroma_keep of the local deviation (luma untouched), which removes purple
# and green smears at 0.3-3 m.  chroma_keep None disables it.
# luma_keep: the same for log luma (1.0 = untouched), against soft light/dark smears
# between the panel joints.

# ---- area config: stage 60 per-view wall outline alignment ---- #
# Texture2LoD3 (CVPRW): place a view by fitting the wall's outline to the facade
# segmentation of the panorama, not by refining its 6-DoF pose.  Here the
# segmentation is Mapillary's own (the detections cache stage 40 already holds),
# rasterised for FACADE_QUAD_CLASS, resampled on the wall plane padded by
# FACADE_QUAD_PAD_M at FACADE_QUAD_CELL_M, and the wall box is slid over it
# (+-FACADE_QUAD_SEARCH_M, summed-area tables).  A fit counts when its IoU with
# the mask is >= FACADE_QUAD_IOU_MIN and at least one vertical and one
# horizontal edge have >= FACADE_QUAD_EDGE_MIN non-building just outside them.
# Only views the correlation could not place use it; a fitted view is tied to
# the correlated set's frame through the correlated views' own fits
# (via 'quad', ownership weight FACADE_QUAD_W_FRAME) or, when none of them fits,
# placed absolutely (via 'quad_abs', weight FACADE_QUAD_W_ABS; 0 = never owns).
FACADE_QUAD_ALIGN = True
FACADE_QUAD_CLASS = 'construction--structure--building'
FACADE_QUAD_PAD_M = 3.0
FACADE_QUAD_CELL_M = 0.10
FACADE_QUAD_SEARCH_M = 2.5
FACADE_QUAD_IOU_MIN = 0.60
FACADE_QUAD_EDGE_MIN = 0.60
FACADE_QUAD_W_FRAME = 1.0
FACADE_QUAD_W_ABS = 0.0

# ---- area config: water surface optics -------------------- #
# A glTF viewer's neutral studio environment has bright overhead panels that a water
# surface mirrors as white glare (water's colour map is <= 0.08 linear, so the glare
# is all specular).  Stage 22 gives mat_water KHR_materials_ior (water is 1.33, glTF
# defaults to 1.5, so F0 drops 0.040 -> 0.020) and KHR_materials_specular.  None
# drops either extension.
WATER_IOR = 1.33
WATER_SPECULAR = 0.45

# ---- area config: glossy surfaces mirror the area's own sky ------ #
# A studio environment still lights every material, so water keeps white glint
# columns and metal roofs and railings blow out white; raising metal roughness or
# capping metallic does not help, the environment is the lever.  Stage 22 writes
# extras.sky_reflect = the value below on every mat_<class> material, for a viewer
# that gives those materials the area's sky image as their envMap at that
# envMapIntensity.  Water is higher because WATER_SPECULAR 0.45 is tuned to hide the
# studio panels.  {} turns it off (a viewer then does nothing).
SKY_REFLECT = dict(water=1.8, metal=1.0)
# SKY_GLASS: glass mirrors the sky, opaque walls keep the studio.  A facade atlas
# packs glass texels (roughness page 0.12 / 0.24 / 0.30: panes, curtain wall,
# photographed glass) and opaque ones (frames 0.40, walls >= 0.85) in ONE material, so
# SKY_REFLECT's per-material envMap would relight every wall.  The rule is per texel
# instead: stage 22 writes extras.sky_glass = [gain, r0, r1] on the classes below,
# and a viewer swaps only the SPECULAR radiance to the sky image x gain where the
# roughness page is <= r0, fading back to the studio by r1; diffuse lighting stays on
# the studio everywhere.  Gain 1.3 puts glass at the render/albedo ratio of opaque
# walls.  {} turns it off.
SKY_GLASS = dict(building_facade=(1.3, 0.31, 0.36))
# FLOOR_LIT: a ground class is never lit from below.  GROUND_STEP_SHADE gives the
# vertical faces of low floor steps (stitch quads, 3dfier step walls, path kerbs) an
# UP vertex normal so they shade as the ground they belong to, but every class
# material is doubleSided, and a viewer flips the normal of a back-facing fragment, so
# an up normal seen from the face's back points DOWN and draws the step dark.
# Stitch quads often face their back to the camera and 3dfier terrain folds are seen
# from both sides, so no single winding rule fixes it.  Stage 22 writes
# extras.floor_lit = up_min on every class material except `skip`; a viewer keeps the
# stored normal on a back-facing fragment of such a material when that normal points
# up (world y > up_min), so it is lit the same from either side.  Side normals (tall
# banks, building walls sharing mat_concrete_wall) have world y ~ 0 and keep the flip.
# None / {} turns it off (a viewer then does nothing).
FLOOR_LIT = dict(up_min=0.5, skip=('water',))

# ---- per-area config: knob overrides (see the hook above) ------------------- #
if _AREA_CFG is not None:
    _areacfg.apply_knobs(globals(), _AREA_CFG)
