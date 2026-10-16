"""The OSM tag -> layer decision table.

Ported BY HAND (rules, not code) from louis-e/arnis, a Rust OSM-to-Minecraft
generator whose value here is a decision table refined over a lot of real map
data.  Source files: arnis src/element_processing/ plus src/data_processing.rs
for the dispatch order.  Every entry cites the arnis file and line range it came
from; line numbers refer to arnis commit 80553b383f855b5d63fdafff8e26e5b9af2c1e7c.  Nothing is copied: arnis emits integer block
layers from a palette chosen for visual read; this module emits metres of polygon.

Unit note.  arnis works in Minecraft blocks and at its default world scale one
block is one metre, so a block constant is a metre constant.  Two conversions
are deliberate and flagged at each use:

  * arnis stamps a road as a SQUARE Chebyshev neighbourhood of radius
    `block_range` around every Bresenham point, so its rendered width is
    2*block_range + 1 blocks (highways.rs:1852-1854, 2874-2887).  The +1 is the
    centre voxel.  This pipeline draws a continuous buffered polygon, which has
    no centre voxel, so its width is 2*block_range metres.
  * arnis lifts a bridge deck by `layer * LAYER_HEIGHT_STEP` where the step is
    6 blocks (bridges.rs:10-16).  Six metres of clearance per layer is a
    Minecraft-scale exaggeration for a footbridge, so DECK_LAYER_STEP_M is
    kept at arnis's 6.0 but the gate conditions (free endpoint, terrain dip,
    span) are what decides; see bridge_clearance().

Consumers: s02_osm.py (classifies the whole fetch and records the counts),
s10a_3dfier.py (builds the planar partition), lib/s2x_lib.py (ribbon widths).
"""

# --------------------------------------------------------------------------- #
# Layer priority.  arnis has no single ordering: element dispatch is a
# first-match if/else chain (data_processing.rs:224-380) that puts
# building > highway > landuse > natural > amenity > leisure > barrier >
# waterway > man_made, and ground-filling areas are then re-sorted by
# DESCENDING ring area so the smallest polygon is the last writer
# (data_processing.rs:150-178).  On top of that a sealed-surface mask stops
# landuse ever repainting a road, a paved area or water
# (data_processing.rs:773-780, landuse.rs:139-166).
#
# Flattened for a planar partition, that is the order below: the class that
# wins a square metre is the first one in this list that claims it.
# --------------------------------------------------------------------------- #
PRIORITY = ['building', 'bridge', 'water', 'road', 'terrain']

MIN_BUILDING_AREA_M2 = 8.0     # this pipeline's own; arnis's analogue is MIN_FACADE_FOOTPRINT
                               # = 12 blocks (building_facade.rs:23)

# --------------------------------------------------------------------------- #
# 1. Paths and roads
# --------------------------------------------------------------------------- #
# highways.rs:2725-2745 -- base half-width in BLOCKS per highway=*.  Types
# marked lane_scaled take max(base, lane formula); the rest ignore `lanes`.
HIGHWAY_HALF_WIDTH_BLOCKS = {
    # not lane-scaled (highways.rs:2725-2731)
    'footway': 1, 'pedestrian': 1, 'path': 1, 'track': 1, 'steps': 1,
    'escape': 1,
    # lane-scaled (highways.rs:2732-2744)
    'motorway': 5, 'primary': 5, 'trunk': 5,
    'secondary': 4,
    'tertiary': 2, 'service': 2,
    'secondary_link': 1, 'tertiary_link': 1,
}
HIGHWAY_DEFAULT_HALF_WIDTH_BLOCKS = 2      # highways.rs:2745, covers residential,
                                           # cycleway, unclassified, living_street,
                                           # motorway_link, ...
HIGHWAY_NOT_LANE_SCALED = {'footway', 'pedestrian', 'path', 'track', 'steps', 'escape'}

# highways.rs:2716-2721 -- default lane count when lanes=* is absent.
DEFAULT_LANES = {'motorway': 2, 'primary': 2, 'trunk': 2, 'secondary': 2,
                 'tertiary': 2}
DEFAULT_LANES_OTHER = 1
LANE_WIDTH_M = 3.5                 # highways.rs:2746-2758, the 3.5 in the formula
LANES_CLAMP = (1, 16)              # highways.rs:2746-2758
HALF_WIDTH_CLAMP_BLOCKS = (1, 8)   # highways.rs:2759, MAX_BLOCK_RANGE = 8

# highways.rs:2843-2846, 2776-2780 -- never part of the road surface at all.
HIGHWAY_NOT_A_SURFACE = {'street_lamp', 'crossing', 'bus_stop'}


def path_width(tags):
    """(width_m, source) for a highway centreline, arnis's rule in metres.

    Precedence, highways.rs:2746-2764:
      width=*  ->  block_range = round(width/2)              (HWY-4, wins outright)
      else lane-scaled types: max(base, round((lanes*3.5 - 1)/2))   (HWY-3)
      else the per-type base                                        (HWY-1)
    then clamped to 1..8 blocks (HWY-5).  Returns 2*block_range metres, not
    2*block_range+1, because a buffered polygon replaces the stamped voxels.
    """
    hw = tags.get('highway')
    base = HIGHWAY_HALF_WIDTH_BLOCKS.get(hw, HIGHWAY_DEFAULT_HALF_WIDTH_BLOCKS)
    src = 'arnis_highway_default'
    w = _num(tags.get('width'))
    if w and w > 0:
        half = round(w / 2.0)
        src = 'osm_width'
    else:
        half = base
        if hw not in HIGHWAY_NOT_LANE_SCALED:
            lanes = _num(tags.get('lanes'))
            if lanes is None:
                lanes = DEFAULT_LANES.get(hw, DEFAULT_LANES_OTHER)
            lanes = max(LANES_CLAMP[0], min(LANES_CLAMP[1], int(round(lanes))))
            lane_half = round((lanes * LANE_WIDTH_M - 1.0) / 2.0)
            if lane_half > half:
                half = lane_half
                src = 'arnis_lane_formula'
    half = max(HALF_WIDTH_CLAMP_BLOCKS[0], min(HALF_WIDTH_CLAMP_BLOCKS[1], half))
    return float(2 * half), src


def is_path_area(tags, closed):
    """highways.rs:1415-1432 -- highway + area=yes is flood-filled as a plaza,
    the centreline is skipped entirely."""
    return bool(closed and tags.get('area') == 'yes' and tags.get('highway'))


# --------------------------------------------------------------------------- #
# 2. surface=* -> ground surface class
# --------------------------------------------------------------------------- #
# surfaces.rs:9-43.  arnis returns a Minecraft block; the class on the right is
# the equivalent ground surface type here.  Unknown values return None and the caller
# keeps its own default (surfaces.rs:42).
SURFACE_CLASS = {
    'sand': 'sand',                                            # SAND
    'grass': 'lawn', 'grass_paver': 'lawn',                    # GRASS_BLOCK
    'artificial_turf': 'lawn',                                 # GREEN_WOOL
    'dirt': 'bare', 'ground': 'bare', 'earth': 'bare',
    'soil': 'bare', 'unpaved': 'bare',                         # DIRT
    'mud': 'bare',                                             # MUD
    'mulch': 'bare', 'woodchips': 'bare',                      # PODZOL
    'clay': 'bare',                                            # TERRACOTTA
    'tartan': 'paving',                                        # RED_TERRACOTTA
    'pebblestone': 'paving', 'cobblestone': 'paving',
    'unhewn_cobblestone': 'paving', 'stepping_stones': 'paving',  # COBBLESTONE
    'stone': 'paving', 'rock': 'paving',                       # STONE
    'bricks': 'paving', 'brick': 'paving',                     # BRICK
    'metal': 'paving',                                         # IRON_BLOCK
    'ice': 'paving',                                           # PACKED_ICE
    'wood': 'wood',                                            # OAK_PLANKS
    'gravel': 'gravel', 'fine_gravel': 'gravel',
    'compacted': 'gravel',                                     # GRAVEL
    # surfaces.rs:34-41 -- the shared hard-road "asphalt mix"
    'asphalt': 'paving', 'concrete': 'paving', 'paving_stones': 'paving',
    'sett': 'paving', 'paved': 'paving', 'cement': 'paving',
    'chipseal': 'paving', 'bitmac': 'paving',
    'concrete:plates': 'paving', 'concrete:lanes': 'paving',
}

# highways.rs:1453-1460 -- per-type default when surface=* is absent or unknown.
HIGHWAY_DEFAULT_SURFACE = {
    'footway': 'paving', 'pedestrian': 'paving', 'service': 'paving',
    'steps': 'paving',            # GRAY_CONCRETE
    'path': 'bare',               # DIRT_PATH
    'escape': 'sand',             # SAND
}
HIGHWAY_DEFAULT_SURFACE_OTHER = 'road'   # DEFAULT_ROAD_MIX


# --------------------------------------------------------------------------- #
# 3. landuse / leisure / natural -> ground surface class
# --------------------------------------------------------------------------- #
# landuse.rs:29-58
LANDUSE_SURFACE = {
    'greenfield': 'lawn', 'meadow': 'lawn', 'grass': 'lawn',
    'orchard': 'lawn', 'forest': 'lawn',              # GRASS_BLOCK
    'farmland': 'bare',                               # FARMLAND
    'cemetery': 'bare',                               # PODZOL
    'construction': 'bare',                           # COARSE_DIRT
    'vineyard': 'bare', 'brownfield': 'bare', 'farmyard': 'bare',
    'landfill': 'bare',                               # landuse.rs:46-54
    'traffic_island': 'paving',                       # STONE_BLOCK_SLAB
    'education': 'paving', 'religious': 'paving',     # POLISHED_ANDESITE
    'industrial': 'paving', 'quarry': 'paving',       # STONE mix
    'military': 'paving',                             # GRAY_CONCRETE mix
    'railway': 'gravel',                              # GRAVEL
    'garages': 'paving',
}
LANDUSE_DEFAULT_SURFACE = 'lawn'                      # landuse.rs:57, unknown -> GRASS_BLOCK
# landuse.rs:36-39 -- skipped entirely, nothing is placed.  arnis's reason: too
# broad to be a real ground class.  Mirrored at data_processing.rs:63-69 so they
# never even enter the area sort.
LANDUSE_SKIP = {'residential', 'commercial'}

# leisure.rs:27-41
LEISURE_SURFACE = {
    'park': 'lawn', 'nature_reserve': 'lawn', 'garden': 'lawn',
    'disc_golf_course': 'lawn', 'golf_course': 'lawn',     # GRASS_BLOCK
    'playground': 'lawn', 'recreation_ground': 'lawn', 'pitch': 'lawn',
    'beach_resort': 'lawn', 'dog_park': 'lawn',            # GREEN_STAINED_HARDENED_CLAY
    'schoolyard': 'paving',                                # BLACK_CONCRETE
    'bathing_place': 'sand',                               # SMOOTH_SANDSTONE
    'outdoor_seating': 'paving',                           # SMOOTH_STONE
    'water_park': 'paving', 'slipway': 'paving',           # LIGHT_GRAY_CONCRETE
    'ice_rink': 'paving',                                  # PACKED_ICE
}
LEISURE_DEFAULT_SURFACE = 'lawn'                           # leisure.rs:40
# leisure.rs:27-41 -- these map to the WATER block, i.e. they are water areas.
LEISURE_WATER = {'swimming_pool', 'swimming_area', 'marina'}

# natural.rs:109-128
NATURAL_SURFACE = {
    'scrub': 'scrub', 'grassland': 'lawn', 'wood': 'lawn',
    'heath': 'lawn', 'tree_row': 'lawn',                   # GRASS_BLOCK
    'shrubbery': 'lawn', 'tundra': 'lawn', 'hill': 'lawn',
    'sand': 'sand', 'dune': 'sand',
    'beach': 'sand', 'shoal': 'sand',                      # SAND
    'bare_rock': 'paving', 'saddle': 'paving', 'ridge': 'paving',
    'cliff': 'paving',                                     # STONE
    'blockfield': 'paving', 'mountain_range': 'paving',    # COBBLESTONE
    'glacier': 'paving',                                   # PACKED_ICE
    'mud': 'bare', 'wetland': 'bare',                      # MUD
}
NATURAL_DEFAULT_SURFACE = 'lawn'                           # natural.rs:127
NATURAL_WATER = {'water', 'reef', 'bay'}                   # natural.rs:118 -> WATER


# Ground surface class -> vertex colour for the floor where no photo reaches.
SURFACE_RGB = dict(lawn=(104, 132, 72), paving=(176, 172, 164),
                   sand=(198, 186, 156), gravel=(168, 160, 146),
                   wood=(150, 140, 125), bare=(126, 122, 112),
                   road=(96, 96, 98), scrub=(96, 116, 74),
                   water=(44, 74, 96))


# --------------------------------------------------------------------------- #
# 4. Water: areas vs waterways
# --------------------------------------------------------------------------- #
# data_processing.rs:333-348 / water_areas.rs:597-603 -- only waterway=dock and
# waterway=riverbank become water AREAS; every other waterway value is a linear
# channel.  natural=water and landuse=reservoir ways are taken earlier by the
# natural/landuse arms, which paint the WATER block, so they are areas too.
WATERWAY_AREA = {'dock', 'riverbank'}
# waterways.rs:58-79 -- structures and points, not channels: render nothing.
WATERWAY_NOT_A_CHANNEL = {
    'dam', 'weir', 'lock_gate', 'waterfall', 'rapids', 'boatyard', 'fuel',
    'dock', 'riverbank', 'water_point', 'turning_point', 'sluice_gate',
    'fish_pass', 'security_lock', 'milestone', 'check_dam', 'floating_barrier'}
# waterways.rs:95-107 -- channel width in BLOCKS (= metres in this pipeline).
WATERWAY_WIDTH_M = {'river': 8.0, 'canal': 6.0, 'stream': 3.0, 'fairway': 12.0,
                    'flowline': 2.0, 'brook': 2.0, 'ditch': 2.0, 'drain': 1.0}
WATERWAY_DEFAULT_WIDTH_M = 4.0        # waterways.rs:106
WATERWAY_MAX_WIDTH_M = 128.0          # waterways.rs:111-125, MAX_WATERWAY_WIDTH
LANDUSE_WATER = {'reservoir', 'basin'}

# waterway=dock is a water basin fill in arnis, not a pier (data_processing.rs:333-348,
# water_areas.rs:597-603), so this table sends it to `water`, NOT to `bridge`.
# man_made=pier stays a deck (man_made.rs:63-104).


def waterway_width(tags):
    """waterways.rs:95-125.  width=* wins if >= 1 m, clamped to 128 m."""
    w = _num(tags.get('width'))
    if w is not None and round(w) >= 1:
        return float(min(round(w), WATERWAY_MAX_WIDTH_M)), 'osm_width'
    return WATERWAY_WIDTH_M.get(tags.get('waterway'),
                                WATERWAY_DEFAULT_WIDTH_M), 'arnis_waterway_default'


# --------------------------------------------------------------------------- #
# 5. Bridges, piers, docks, boardwalks
# --------------------------------------------------------------------------- #
DECK_LAYER_STEP_M = 6.0            # bridges.rs:10-16, LAYER_HEIGHT_STEP
FLAT_TERRAIN_DIP_M = 4.0           # bridges.rs:10-16, FLAT_TERRAIN_DIP_THRESHOLD
SHORT_BRIDGE_SPAN_M = 30.0         # bridges.rs:10-16, SHORT_BRIDGE_LENGTH_BLOCKS
ARCH_MIN_CLEARANCE_M = 8.0         # bridges.rs:327-333
PIER_DECK_ABOVE_GROUND_M = 1.0     # man_made.rs:63-104, deck at ground + 1 block
PIER_DEFAULT_WIDTH_M = 3.0         # man_made.rs:63-104, DEFAULT 3 blocks
PIER_PILLAR_SPACING_M = 4.0        # man_made.rs:63-104, support_spacing = 4
BOARDWALK_POST_SPACING_M = 4.0     # bridge_styles.rs:22-62, BOARDWALK_POST_INTERVAL
BEAM_PILLAR_SPACING_M = 8.0        # bridge_styles.rs:244-245, BEAM_PILLAR_INTERVAL

# bridge_styles.rs:212-242, bridge:structure first, then bridge=*.
BRIDGE_STYLE = {
    'arch': 'arch', 'truss': 'truss', 'suspension': 'suspension',
    'simple-suspension': 'suspension', 'suspension_bridge': 'suspension',
    'cable-stayed': 'cable_stayed', 'cable_stayed': 'cable_stayed',
    'beam': 'beam', 'covered': 'covered', 'boardwalk': 'boardwalk',
    'viaduct': 'beam',
}
BRIDGE_DEFAULT_STYLE = 'beam'      # bridge_styles.rs:241


def bridge_style(tags):
    """bridge_styles.rs:212-242.  bridge:structure beats bridge=*."""
    s = tags.get('bridge:structure')
    if s and s in BRIDGE_STYLE:
        return BRIDGE_STYLE[s], 'bridge:structure'
    b = tags.get('bridge')
    if b and b in BRIDGE_STYLE:
        return BRIDGE_STYLE[b], 'bridge'
    return BRIDGE_DEFAULT_STYLE, 'default'


def effective_layer(tags):
    """bridges.rs:767-773.  max(layer, 0); a bridge with no layer=* counts as 1."""
    l = _num(tags.get('layer'))
    if l is None:
        return 1 if (tags.get('bridge') and tags.get('bridge') != 'no') else 0
    return max(int(l), 0)


def bridge_clearance(layer, has_free_endpoint, terrain_dip_m, span_m,
                     style='beam'):
    """arnis's deck-lift heuristic, bridges.rs:299-334, in metres.

    The deck is lifted to terrain_max + clearance only when ALL of:
      * the group has a boundary endpoint (an endpoint used by one member only),
      * the terrain dip under it is under FLAT_TERRAIN_DIP_M, and
      * the total span is at least SHORT_BRIDGE_SPAN_M.
    Otherwise the deck follows the contour (clearance 0).  An arch with the same
    dip/span gate gets at least ARCH_MIN_CLEARANCE_M (bridges.rs:327-333).
    """
    flat_and_long = (terrain_dip_m < FLAT_TERRAIN_DIP_M and span_m >= SHORT_BRIDGE_SPAN_M)
    c = (max(layer, 0) * DECK_LAYER_STEP_M
         if (has_free_endpoint and flat_and_long) else 0.0)
    if style == 'arch' and flat_and_long:
        c = max(c, ARCH_MIN_CLEARANCE_M)
    return float(c)


# --------------------------------------------------------------------------- #
# 6. Building heights
# --------------------------------------------------------------------------- #
# buildings.rs:2507-2532 -- blocks per upper floor.  A building:part uses one
# cycle for every part so stacked volumes stay flush.
FLOOR_CYCLE_PART_M = 4.0
FLOOR_CYCLE_RESIDENTIAL_M = 3.0
FLOOR_CYCLE_OTHER_M = 4.0
GROUND_FLOOR_BONUS_M = 2.0          # buildings.rs:2507-2532, added once at ground
RESIDENTIAL_BUILDINGS = {
    'house', 'detached', 'semidetached_house', 'terrace', 'bungalow', 'villa',
    'cabin', 'residential', 'apartments', 'dormitory', 'farm', 'hut', 'shed',
    'static_caravan', 'yes'}
MIN_HEIGHT_PART_M = 1.0             # buildings.rs:2738-2742
MIN_HEIGHT_BUILDING_M = 3.0
# buildings.rs:2556-2645 -- storeys inferred from building=* and footprint when
# nothing is tagged.  Only the most common values here; arnis's full table is a
# weighted random pick; this table takes its modal storey count so the build is stable.
INFER_LEVELS = {
    'bungalow': 1, 'static_caravan': 1, 'house': 2, 'detached': 2, 'villa': 2,
    'farm': 2, 'terrace': 2, 'apartments': 3, 'residential': 3, 'dormitory': 3,
    'retail': 1, 'shop': 1, 'supermarket': 1, 'parking': 3, 'hospital': 5,
    'garage': 1, 'garages': 1, 'carport': 1, 'shed': 1, 'hut': 1,
}
INFER_HALL_M = {'industrial': 7.0, 'warehouse': 7.0, 'hangar': 7.0,
                'barn': 7.0, 'stable': 7.0,
                'church': 12.0, 'chapel': 12.0, 'mosque': 12.0,
                'synagogue': 12.0, 'temple': 12.0, 'religious': 12.0,
                'cathedral': 21.0, 'roof': 4.0, 'ship': 6.0}
# buildings.rs:2556-2645, building=yes / unknown banded by footprint area.
INFER_BY_AREA = ((40.0, 1), (150.0, 2), (600.0, 3), (float('inf'), 3))


def floor_cycle(tags):
    """buildings.rs:2507-2532."""
    if tags.get('building:part') and tags.get('building:part') != 'no':
        return FLOOR_CYCLE_PART_M
    b = tags.get('building') or 'yes'
    return (FLOOR_CYCLE_RESIDENTIAL_M if b in RESIDENTIAL_BUILDINGS
            else FLOOR_CYCLE_OTHER_M)


def building_height(tags, area_m2=None):
    """(height_m, min_height_m, source) following arnis's precedence.

    buildings.rs:2693-2742: height=* OVERRIDES building:levels.
    buildings.rs:2942-2971: the base is min_height, else building:min_level,
      else layer, else 0.
    buildings.rs:2721-2736: roof:height is subtracted from the walls only when
      roof:shape is present and is not "flat".
    buildings.rs:2669-2689 / 2556-2645: levels, then inference from building=*
      and footprint area.
    """
    fc = floor_cycle(tags)
    is_part = bool(tags.get('building:part') and tags.get('building:part') != 'no')
    floor_min = MIN_HEIGHT_PART_M if is_part else MIN_HEIGHT_BUILDING_M

    # base of the walls
    min_h = _num(tags.get('min_height'))
    min_level = _num(tags.get('building:min_level'))
    if min_h is None and min_level and min_level > 0:
        min_h = min_level * fc + GROUND_FLOOR_BONUS_M
    if min_h is None:
        # buildings.rs:2942-2971 uses layer=* as the last base fallback, but only
        # as a stacking hint for multi-shell roof PARTS.  Restricted to parts
        # here: over-water buildings can carry layer=1 purely as an OSM crossing
        # layer, which would otherwise lift them 3-4 m into the air.
        lay = _num(tags.get('layer'))
        min_h = lay * fc if (lay and lay > 0 and is_part) else 0.0

    h = _num(tags.get('height'))
    if h is not None and h > 0:
        eff = max(1.0, h - min_h)
        src = 'osm_height'
    else:
        lv = _num(tags.get('building:levels'))
        ml = _num(tags.get('building:min_level')) or 0.0
        if lv is not None and (lv - ml) >= 1.0:
            bonus = GROUND_FLOOR_BONUS_M if ml == 0 else 0.0
            eff = max(floor_min, (lv - ml) * fc + bonus)
            src = 'osm_levels'
        else:
            b = tags.get('building') or tags.get('building:part') or 'yes'
            if b in INFER_HALL_M:
                eff = INFER_HALL_M[b]; src = 'arnis_infer_hall'
            elif b in INFER_LEVELS:
                eff = INFER_LEVELS[b] * fc + GROUND_FLOOR_BONUS_M
                src = 'arnis_infer_levels'
            else:
                lv2 = INFER_BY_AREA[-1][1]
                if area_m2 is not None:
                    for cut, n in INFER_BY_AREA:
                        if area_m2 < cut:
                            lv2 = n; break
                eff = lv2 * fc + GROUND_FLOOR_BONUS_M
                src = 'arnis_infer_area'
    # roof:height only comes off a pitched roof
    rh = _num(tags.get('roof:height'))
    shape = tags.get('roof:shape')
    if rh and shape and shape != 'flat':
        eff = max(min(floor_min, eff), eff - max(0.0, rh))
        src += '+roof_height'
    return float(max(eff, floor_min)), float(min_h), src


# --------------------------------------------------------------------------- #
# 7. The dispatch
# --------------------------------------------------------------------------- #
def classify(tags, closed=False, area_m2=None):
    """Tags -> one partition layer, or None if the element carries no geometry.

    Follows arnis's first-match dispatch (data_processing.rs:224-380), with the
    ground-filling classes collapsed onto `terrain` and everything sealed put
    ahead of them, which is what the sealed-surface mask does there.
    """
    t = tags or {}

    # building / building:part -- data_processing.rs:224-380, first arm
    b = t.get('building'); bp = t.get('building:part')
    if (b and b != 'no') or (bp and bp != 'no'):
        if not closed:
            return None
        h, min_h, src = building_height(t, area_m2)
        return dict(layer='building', kind='building', geom='area',
                    width_m=None, surface='paving',
                    height_m=h, min_height_m=min_h, height_source=src,
                    rule='building/building:part (data_processing.rs:224-380); '
                         'height ' + src + ' (buildings.rs:2669-2742)')

    # bridge decks -- bridges.rs:738-744
    if t.get('bridge') and t.get('bridge') != 'no':
        style, ssrc = bridge_style(t)
        w, wsrc = path_width(t) if t.get('highway') else (PIER_DEFAULT_WIDTH_M, 'default')
        return dict(layer='bridge', kind='bridge_' + style,
                    geom='area' if (closed and t.get('area') == 'yes') else 'line',
                    width_m=w, surface=('wood' if style == 'boardwalk' else 'paving'),
                    cap='square', bridge_style=style, style_source=ssrc,
                    layer_tag=effective_layer(t), width_source=wsrc,
                    pillar_spacing_m=(BOARDWALK_POST_SPACING_M if style == 'boardwalk'
                                      else BEAM_PILLAR_SPACING_M),
                    rule='bridge=* deck (bridges.rs:738-744); style '
                         '(bridge_styles.rs:212-242); lift gate (bridges.rs:299-334)')

    # man_made=pier -- man_made.rs:63-104
    if t.get('man_made') == 'pier':
        w = _num(t.get('width')) or PIER_DEFAULT_WIDTH_M
        return dict(layer='bridge', kind='pier',
                    geom='area' if closed else 'line', width_m=float(w),
                    surface='wood', cap='square',
                    deck_above_ground_m=PIER_DECK_ABOVE_GROUND_M,
                    pillar_spacing_m=PIER_PILLAR_SPACING_M,
                    width_source=('osm_width' if t.get('width') else 'arnis_default_3'),
                    rule='man_made=pier: fixed deck 1 m above ground, pillars every '
                         '4 m, never samples the water level (man_made.rs:63-104)')

    # water areas
    ww = t.get('waterway')
    if ww in WATERWAY_AREA:
        return dict(layer='water', kind='waterway_' + ww, geom='area',
                    width_m=None, surface='water',
                    rule='waterway=dock|riverbank is a water AREA, not a pier '
                         '(data_processing.rs:333-348, water_areas.rs:597-603)')
    if t.get('natural') in NATURAL_WATER or t.get('water'):
        return dict(layer='water', kind='natural_water', geom='area',
                    width_m=None, surface='water',
                    rule='natural=water|bay|reef or water=* (natural.rs:109-128, '
                         'water_areas.rs:77-86)')
    if t.get('landuse') in LANDUSE_WATER:
        return dict(layer='water', kind='landuse_water', geom='area',
                    width_m=None, surface='water',
                    rule='landuse=reservoir|basin paints WATER (landuse.rs:29-58)')
    if t.get('leisure') in LEISURE_WATER:
        return dict(layer='water', kind='leisure_' + t['leisure'], geom='area',
                    width_m=None, surface='water',
                    rule='leisure=marina|swimming_pool paint WATER (leisure.rs:27-41)')
    # linear waterways
    if ww and ww not in WATERWAY_NOT_A_CHANNEL:
        w, wsrc = waterway_width(t)
        return dict(layer='water', kind='waterway_' + ww, geom='line', width_m=w,
                    surface='water', cap='round', width_source=wsrc,
                    rule='linear waterway channel, width (waterways.rs:95-125)')

    # highways
    hw = t.get('highway')
    if hw and hw not in HIGHWAY_NOT_A_SURFACE:
        surf = (SURFACE_CLASS.get(t.get('surface'))
                or HIGHWAY_DEFAULT_SURFACE.get(hw, HIGHWAY_DEFAULT_SURFACE_OTHER))
        if is_path_area(t, closed):
            return dict(layer='road', kind='highway_' + hw + '_area', geom='area',
                        width_m=None, surface=surf,
                        rule='highway + area=yes: flood-filled plaza, centreline '
                             'skipped (highways.rs:1415-1432)')
        w, wsrc = path_width(t)
        return dict(layer='road', kind='highway_' + hw, geom='line', width_m=w,
                    surface=surf, cap='round', width_source=wsrc,
                    rule='highway ribbon, half-width (highways.rs:2725-2764), '
                         'surface (surfaces.rs:9-43, highways.rs:1453-1460)')

    # ground-filling areas.  arnis order: landuse, then natural, then amenity,
    # then leisure (data_processing.rs:224-380).
    lu = t.get('landuse')
    if lu:
        if lu in LANDUSE_SKIP:
            return None            # landuse.rs:36-39
        return dict(layer='terrain', kind='landuse_' + lu, geom='area',
                    width_m=None,
                    surface=LANDUSE_SURFACE.get(lu, LANDUSE_DEFAULT_SURFACE),
                    rule='landuse ground class (landuse.rs:29-58); '
                         'residential/commercial skipped (landuse.rs:36-39)')
    nat = t.get('natural')
    if nat and t.get('amenity') != 'fountain':
        return dict(layer='terrain', kind='natural_' + nat, geom='area',
                    width_m=None,
                    surface=NATURAL_SURFACE.get(nat, NATURAL_DEFAULT_SURFACE),
                    rule='natural ground class (natural.rs:109-128)')
    le = t.get('leisure')
    if le:
        surf = (SURFACE_CLASS.get(t.get('surface'))
                or LEISURE_SURFACE.get(le, LEISURE_DEFAULT_SURFACE))
        return dict(layer='terrain', kind='leisure_' + le, geom='area',
                    width_m=None, surface=surf,
                    rule='leisure ground class (leisure.rs:27-41); surface=* '
                         'overrides the category default (leisure.rs:44-50)')
    if t.get('amenity') == 'parking':
        return dict(layer='road', kind='amenity_parking', geom='area',
                    width_m=None,
                    surface=SURFACE_CLASS.get(t.get('surface'), 'paving'),
                    rule='amenity=parking is a sealed surface '
                         '(floodfill_cache.rs:325-367)')
    return None


def _num(v):
    if v is None:
        return None
    try:
        return float(str(v).strip().split()[0].rstrip('m'))
    except Exception:
        return None
