"""Stage 84 -- the OBJECT inventory: one tagged row per recognisable thing in
the area that the s80 furniture inventory does not already own.

s80 answers "where is the street furniture"; this
stage answers "what else is here": vehicles, boats, utility poles, traffic
lights and signs by type, banners, cabinets, cameras, ground fixtures, and the
things no sensor vocabulary names (planters, kiosks, shelters, artwork,
playground items) which are carried as sized placeholders so a downstream tool
can substitute a model later.

Three sensors, fused with the s80 rules (multi-view agreement, model-floor
height, per-source evidence gates):

  photo   Mapillary detections triangulated across panorama pairs
          (lib/objects/objphoto.py; the ray geometry is s80's, imported).
  aerial  the area's ENU ortho, colour/shape only, inside OSM
          parking, road and water priors (lib/objects/objaerial.py).
  lidar   3DEP returns 0.5-4 m above the model floor, clustered at 0.5 m and
          sized by PCA (lib/objects/objlidar.py).
  osm     the tags s80 does not claim: playground, shelter, artwork, memorial,
          surveillance, kiosk.

Every row carries kind, subkind, position, footprint, height, yaw, colour,
sources, confidence, and static vs transient.  Transient kinds (cars, boats,
cones) are NOT one object per sighting: every sighting inside one vehicle
footprint is merged into one POSITION MARKER with the number of panoramas that saw
something there, because a parked car is a property of the capture day, not of the site.

  build   s84_object_inventory.py            (runs the three passes if needed)
  fuse    s84_object_inventory.py --fuse-only
Outputs: object_inventory.json / .csv in the work dir, report in scratch.
"""
import os, sys, json, math, time, argparse, collections, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
import common as C
import s80_furniture_inventory as S
from objects.objcommon import SCR, hash_unit
from objects import objclasses as K
from objects import objaerial as A

TRANSIENT_MERGE_M = {'car': 2.5, 'truck': 3.5, 'bus': 4.0, 'boat': 4.0,
                     'motorcycle': 1.5, 'bicycle': 1.5, 'traffic_cone': 1.0,
                     'rail_vehicle': 6.0}
# Wheeled vehicles are merged with an ELLIPSE, not a disc, because that is how
# they pack: nose to tail they are >4 m apart along the road, side by side they
# are ~2.6 m apart across it.  A disc big enough to swallow the same car seen on
# six different capture runs also swallows the car in the next bay; an ellipse
# aligned with the road does not.  Radii are (along the road, across it).
VEHICLE_ELLIPSE = {'car': (4.6, 2.2), 'truck': (6.5, 2.4), 'bus': (9.0, 2.4),
                   'motorcycle': (2.5, 1.0), 'bicycle': (2.0, 1.0)}
LIDAR_MIN_PTS = 20          # a lidar-only object needs this many returns
LIDAR_MIN_PTS_UNKNOWN = 30  # an unnamed lidar blob needs more
AERIAL_CAR_FILL = 0.6
AERIAL_CAR_CONTRAST = 18.0
BLOB_CONFIRM_M = 2.5        # an aerial blob needs another sensor this close
PROP_MATCH_M = 3.0          # distance at which a new object is an old prop
POLE_PROP_MATCH_M = 5.0     # a pole-like detection this close to a placed pole
                            # IS that pole: the Mapillary pole classes fire on
                            # lamp columns and sign posts as well
# Kinds that belong to a street.  object--support--pole fires on every vertical
# the camera passes, so these kinds must be within STREET_NEAR_M of a road,
# parking or service corridor to be kept (otherwise poles and traffic lights
# land across lawns).
STREET_KINDS = ('utility_pole', 'traffic_light', 'sign_board', 'sign_frame',
                'traffic_sign_face', 'jersey_barrier', 'parking_meter',
                'utility_cabinet', 'mailbox', 'traffic_cone')
STREET_NEAR_M = 12.0
# Wheeled vehicles belong on a DRIVEABLE surface.  A "near a road or a path" test
# keeps most mis-triangulated cars, on lawn and on footpaths, because a path is
# where the panorama rig walked.  The test is the
# OSM surface raster texlib already builds for the material pass
# (texlib.build_surface_raster: arnis surface class + OSM feature kind per
# 0.25 m cell, cut from the same geojson as the terrain TIN, so raster and
# geometry cannot disagree).  'road' as a SURFACE class covers only the cells
# arnis calls DEFAULT_ROAD_MIX -- a small share of a typical area -- because footway,
# pedestrian, service and steps all default to 'paving', so the driveable set
# has to be named by KIND as well.
ROADWAY_KINDS = ('highway_secondary', 'highway_residential',
                 'highway_unclassified', 'highway_service', 'amenity_parking',
                 'landuse_garages')
# A car whose triangulation landed just off the kerb is a real car with a metre
# or two of error, so it is SNAPPED onto the nearest driveable cell before the
# drop is considered.  Beyond this the row is not a car on the street at all.
VEHICLE_SNAP_M = 3.0
VEHICLE_KINDS = ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'rail_vehicle')
# An unknown_cluster row buys a placeholder box with no class name behind it, so
# it has to earn the space.  unknown_cluster confidences split in two with a gap
# between them: a group at the 0.80 ceiling (either aerial+lidar agreement, or
# >= 45 lidar returns on their own) and lidar-only blobs at 0.65-0.77 carrying
# 30-42 returns, barely over LIDAR_MIN_PTS_UNKNOWN.  So the cut goes at 0.80.
UNKNOWN_MIN_CONF = 0.80


def roadway_masks():
    """The driveable surface, off texlib's 0.25 m OSM surface raster.

    -> (distance to the nearest driveable cell in metres, indices of that cell,
    the OSM feature name per driveable cell, grid).  Distance and indices come
    from one EDT over the whole area, so a per-object query is a few array
    lookups."""
    import texlib as T
    from scipy import ndimage
    surf, kind, kinds, grid = T.build_surface_raster()
    ki = {k: i for i, k in enumerate(kinds)}
    sel = np.zeros(max(len(kinds), int(kind.max()) + 1), bool)
    for k in ROADWAY_KINDS:
        if k in ki:
            sel[ki[k]] = True
    drive = (surf == T.SURF_CLASSES.index('road')) | sel[kind]
    dist, idx = ndimage.distance_transform_edt(~drive, sampling=grid[2],
                                               return_indices=True)
    # the name of the driveable feature per cell, so a kept vehicle can say
    # which roadway it is standing on rather than which mesh layer carries it
    names = np.where(drive, np.array(kinds + ['road'], object)[
        np.where(sel[kind], kind, len(kinds))], None)
    return dist.astype('f4'), idx.astype('i4'), names, grid


def snap_to_roadway(dist, idx, names, grid, x, y, r=VEHICLE_SNAP_M):
    """-> (x, y, moved_m, roadway) on the nearest driveable cell, or None when
    the nearest one is further than r metres away."""
    x0, y0, c, nx, ny = grid
    j = int((x - x0) / c); i = int((y - y0) / c)
    if not (0 <= i < int(ny) and 0 <= j < int(nx)):
        return None
    d = float(dist[i, j])
    if d > r:
        return None
    if d <= 0.0:
        return (x, y, 0.0, names[i, j])
    si = int(idx[0][i, j]); sj = int(idx[1][i, j])
    xs = x0 + (sj + 0.5) * c; ys = y0 + (si + 0.5) * c
    # the EDT measures cell centre to cell centre; the object is somewhere
    # inside its cell, so the move it actually makes is measured again here and
    # the limit applies to THAT, not to the raster's approximation of it
    moved = math.hypot(xs - x, ys - y)
    if moved > r:
        return None
    return (xs, ys, round(moved, 2), names[si, sj])


def ship_polys(buf=3.0):
    """Footprints of the OSM-mapped ships and big vessels: a 'boat' detected inside
    one is the same hull, not a second object."""
    from shapely.geometry import Polygon
    els = json.load(open(C.W + '/osm/osm_raw.json'))['elements']
    out = []
    for el in els:
        t = el.get('tags') or {}
        if el['type'] != 'way' or not el.get('geometry'):
            continue
        if not (t.get('historic') in ('ship', 'naval') or t.get('building') == 'ship'
                or t.get('seamark:type') == 'mooring'):
            continue
        P = [S.enu_of(g['lat'], g['lon']) for g in el['geometry']]
        if len(P) < 3:
            continue
        try:
            out.append((Polygon(P).buffer(buf), t.get('name')))
        except Exception:
            pass
    return out


def parking_mask(cell=0.5, pad=5.0):
    """OSM amenity=parking polygons and parking aisles only, so a car slot can
    be tagged parked (a map feature) or traffic (a moment on the road)."""
    import cv2
    m = json.load(open(C.W + '/ortho/ortho_enu.json'))
    ex = m['enu_extent']
    x0 = ex['xmin'] - pad; y1 = ex['ymax'] + pad
    nx = int(math.ceil((ex['xmax'] + pad - x0) / cell))
    ny = int(math.ceil((y1 - (ex['ymin'] - pad)) / cell))
    img = np.zeros((ny, nx), 'u1')
    els = json.load(open(C.W + '/osm/osm_raw.json'))['elements']
    for el in els:
        t = el.get('tags') or {}
        g = el.get('geometry') or []
        if el['type'] != 'way' or len(g) < 2:
            continue
        pts = np.array([[(S.enu_of(q['lat'], q['lon'])[0] - x0) / cell,
                         (y1 - S.enu_of(q['lat'], q['lon'])[1]) / cell] for q in g],
                       'f4').round().astype('i4')
        if t.get('amenity') == 'parking':
            cv2.fillPoly(img, [pts], 1)
        elif t.get('highway') in ('service',) and t.get('service') in (
                'parking_aisle', 'driveway'):
            cv2.polylines(img, [pts], False, 1, int(round(5.0 / cell)))
    return img, (x0, y1, cell)


def near_mask(mask, grid, x, y, r):
    """True when any set cell of the mask is within r metres of (x, y)."""
    x0, y1, cell = grid
    j = int((x - x0) / cell); i = int((y1 - y) / cell)
    k = max(1, int(round(r / cell)))
    i0, i1 = max(0, i - k), min(mask.shape[0], i + k + 1)
    j0, j1 = max(0, j - k), min(mask.shape[1], j + k + 1)
    if i0 >= i1 or j0 >= j1:
        return False
    return bool(mask[i0:i1, j0:j1].any())


# -- AERIAL-ONLY BOATS NEED A BOAT'S SHAPE, WATER AND EYES.
# objaerial.detect_boats() also fires on pier fingers, on the shoreline and on
# dark water by the bank, and colour alone does not separate those from hulls.
# The lidar flight and the ortho are usually flown on different days, so "no
# lidar hull" is never a verdict on its own; single-panorama boat rays can land
# on a neighbouring berth, so a hit count is not one either.  The rules:
#   * FINGER: pier fingers are narrow and long, so width < 2.2 m and
#     length/width >= 4.0 is not a hull (a dinghy is about 1.5 m wide at a
#     length/width near 3);
#   * SHORE: a shoreline strip sits only partly over water, so a footprint
#     under 75 % over open model water is not a hull;
#   * BLIND: dark water seen by >= 20 clear high-confidence panoramas within
#     40 m with no boat ray on it, and hull lidar cover < 5 %, is not a hull.
# Only rows whose sole source is the ortho are judged; s85 retires them the
# way it thins sign props (node kept, draws nothing).
AB_FINGER_W = 2.2
AB_FINGER_ASPECT = 4.0
AB_MIN_WATER = 0.75
AB_PANO_R = 40.0
AB_PANO_MIN_VIS = 20
AB_LIDAR_COV = 0.05
AB_HULL_H = (1.3, 8.0)     # m above the local water returns: over dock freeboard


def aerial_boat_evidence(rows, surf=None, verbose=False):
    """-> {id: dict(evidence..., retire=rule or None)} for every boat row.
    Deterministic; reads the ortho-frame lidar, the pano poses and the
    Mapillary detections.  Rows are not modified."""
    from scipy.spatial import cKDTree
    from shapely.geometry import Polygon, Point
    from objects import objphoto as OP
    surf = surf or S.Surface()
    boats = [o for o in rows if o.get('kind') == 'boat']
    if not boats:
        return {}

    def rect(o, pad=0.0):
        L, W = o['footprint_m']; a = math.radians(o.get('yaw_deg') or 0.0)
        ca, sa = math.cos(a), math.sin(a)
        return Polygon([(o['x'] + u * ca - v * sa, o['y'] + u * sa + v * ca)
                        for u, v in ((-L / 2 - pad, -W / 2 - pad), (L / 2 + pad, -W / 2 - pad),
                                     (L / 2 + pad, W / 2 + pad), (-L / 2 - pad, W / 2 + pad))])
    polys = [rect(o, 0.75) for o in boats]
    cxy = np.array([[o['x'], o['y']] for o in boats], 'f8'); btree = cKDTree(cxy)
    zw = float(np.median([o['z'] for o in boats]))
    P = json.load(open(C.W + '/pano_poses.json'))['poses']
    pid = sorted(k for k, v in P.items() if v.get('confidence') == 'high')
    if not pid:   # poses mode has no skyline refinement, so every pose is 'usable', none 'high'
        pid = sorted(k for k, v in P.items() if v.get('confidence') == 'usable')
    XY = np.array([[P[k]['x'], P[k]['y']] for k in pid], 'f8').reshape(-1, 2); ptree = cKDTree(XY)
    reg = np.load(C.FW + '/lidar_reg.npy')
    lcls = np.load(C.FW + '/loaded.npz')['lidar_cls']
    if len(lcls) != len(reg):
        lcls = np.zeros(len(reg), 'i2')
    ltree = cKDTree(reg[:, :2])

    def los_clear(x0, y0, x1, y1):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 0.5))
        return not any(surf.in_building(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
                       for t in np.linspace(0.05, 0.95, n))
    vis = collections.defaultdict(set); hits = collections.defaultdict(set)
    near = sorted(set(i for o in boats for i in ptree.query_ball_point([o['x'], o['y']], AB_PANO_R)))
    for i in near:
        k = pid[i]; O = np.array([P[k]['x'], P[k]['y'], P[k]['z']], 'f8')
        for bi in btree.query_ball_point(O[:2], AB_PANO_R):
            o = boats[bi]
            if math.hypot(O[0] - o['x'], O[1] - o['y']) > 3 and los_clear(O[0], O[1], o['x'], o['y']):
                vis[bi].add(k)
        kinds, D, _O, _s = OP.rays_for_pano(k, P[k])
        for kk, dv in zip(kinds, D):
            if kk != 'boat' or dv[2] > -0.005:
                continue
            t = (zw + 0.3 - O[2]) / dv[2]
            rng = math.hypot(t * dv[0], t * dv[1])
            if rng > AB_PANO_R + 10:
                continue
            az = np.array([dv[0], dv[1]]) / math.hypot(dv[0], dv[1])
            best = None     # the row this ray's waterline range fits best
            for s in np.arange(max(2.0, 0.6 * rng), 1.5 * rng + 1, 0.5):
                px, py = O[0] + az[0] * s, O[1] + az[1] * s
                for bi in btree.query_ball_point([px, py], 25):
                    if polys[bi].contains(Point(px, py)):
                        e = abs(s - rng)
                        if best is None or e < best[0] or (e == best[0] and bi < best[1]):
                            best = (e, bi)
            if best is not None:
                hits[best[1]].add(k)
    out = {}
    for bi, o in enumerate(boats):
        L, W = (float(v) for v in o['footprint_m'])
        a = math.radians(o.get('yaw_deg') or 0.0); ca, sa = math.cos(a), math.sin(a)
        uu, vv = np.meshgrid(np.linspace(-L / 2, L / 2, 11), np.linspace(-W / 2, W / 2, 5))
        gi, gj = surf._ij((o['x'] + uu * ca - vv * sa).ravel(), (o['y'] + uu * sa + vv * ca).ravel())
        solid = (np.isfinite(surf.z_ground_terrain[gi, gj]) | np.isfinite(surf.z_paths[gi, gj])
                 | np.isfinite(surf.z_structures[gi, gj]) | np.isfinite(surf.z_buildings[gi, gj]))
        water = float((np.isfinite(surf.z_water[gi, gj]) & ~solid).mean())
        idx = np.array(ltree.query_ball_point([o['x'], o['y']], 30.0), int)
        w = idx[np.isin(lcls[idx], (9, 18))]
        wl = float(np.median(reg[w, 2])) if len(w) > 5 else float(o['z'])
        pr = rect(o, 0.3)
        idx = np.array(ltree.query_ball_point([o['x'], o['y']], max(L, W) / 2 + 1.0), int)
        if len(idx):
            idx = idx[np.array([pr.contains(Point(p)) for p in reg[idx, :2]], bool)]
        li, lj = surf._ij(reg[idx, 0], reg[idx, 1])
        wat = (~np.isfinite(surf.z_ground_terrain[li, lj]) & ~np.isfinite(surf.z_paths[li, lj])
               & ~np.isfinite(surf.z_structures[li, lj]))
        h = reg[idx, 2] - wl
        hull = idx[wat & (h > AB_HULL_H[0]) & (h < AB_HULL_H[1]) & (lcls[idx] != 2)]
        cells = len(set(map(tuple, np.floor(reg[hull, :2] / 0.5).astype(int)))) if len(hull) else 0
        cov = cells / max(pr.area / 0.25, 1.0)
        ev = dict(width_m=round(W, 2), aspect=round(L / max(W, 0.01), 2), water_frac=round(water, 2),
                  pano_views=len(vis[bi]), pano_rays=len(hits[bi]), lidar_hull_cover=round(cov, 3))
        rule = None
        if set(o.get('sources') or ()) == {'aerial'}:
            if W < AB_FINGER_W and L / max(W, 0.01) >= AB_FINGER_ASPECT:
                rule = ('pier finger: %.2f m wide (< %.1f) at length/width %.1f (>= %.1f)'
                        % (W, AB_FINGER_W, L / W, AB_FINGER_ASPECT))
            elif water < AB_MIN_WATER:
                rule = ('shoreline: %.0f %% of the footprint over open water (< %.0f %%)'
                        % (water * 100, AB_MIN_WATER * 100))
            elif len(vis[bi]) >= AB_PANO_MIN_VIS and not hits[bi] and cov < AB_LIDAR_COV:
                rule = ('unseen: %d clear panoramas within %.0f m, no boat ray, lidar hull '
                        'cover %.1f %%' % (len(vis[bi]), AB_PANO_R, cov * 100))
        ev['retire'] = rule
        out[o['id']] = ev
        if verbose:
            print(o['id'], '+'.join(o['sources']), ev, flush=True)
    return out


def load_props():
    """The s80 inventory: rows already placed, used as the 'already owned' set
    and as the OSM ground truth for the re-find rate."""
    try:
        d = json.load(open(C.W + '/prop_inventory.json'))
    except Exception:
        return []
    return d['props'] if isinstance(d, dict) else d


# --------------------------------------------------------------------------- #
# sensor -> kind
# --------------------------------------------------------------------------- #
def name_lidar(cl, on_water):
    """Size names a lidar cluster; that is all lidar alone can do."""
    lo, sh, h = cl['long_m'], cl['short_m'], cl['height_m']
    if on_water and lo >= 2.5 and h <= 4.0:
        return 'boat', 'moored'
    if 3.2 <= lo <= 6.8 and 1.3 <= sh <= 2.8 and 1.1 <= h <= 2.3:
        return 'car', 'car'
    if 4.0 <= lo <= 8.0 and 1.8 <= sh <= 3.2 and 2.3 < h <= 4.0:
        return 'truck', 'truck'
    if 1.5 <= lo <= 5.0 and sh >= 1.2 and 2.0 <= h <= 4.0:
        return 'kiosk', 'lidar_box'
    if 0.6 <= lo <= 2.2 and sh <= 2.0 and 0.5 <= h <= 1.5:
        return 'unknown_cluster', 'low_box'
    return 'unknown_cluster', 'other'


def osm_pass():
    els = json.load(open(C.W + '/osm/osm_raw.json'))['elements']
    out = []
    for el in els:
        t = el.get('tags') or {}
        if not t:
            continue
        kind = sub = None
        for (k, v), kd, sb in K.OSM_RULES:
            if kd is None:
                continue
            if k in t and (v is None or t[k] == v):
                kind, sub = kd, (sb or t[k]); break
        if kind is None:
            continue
        if el['type'] == 'node':
            x, y = S.enu_of(el['lat'], el['lon'])
            foot = None
        elif el.get('geometry'):
            g = el['geometry']
            P = np.array([S.enu_of(p['lat'], p['lon']) for p in g], 'f8')
            x, y = float(P[:, 0].mean()), float(P[:, 1].mean())
            d = P - [x, y]
            cov = d.T @ d / max(len(d), 1)
            ev, evec = np.linalg.eigh(cov)
            foot = (float(np.ptp(d @ evec[:, 1])), float(np.ptp(d @ evec[:, 0])))
        else:
            continue
        out.append(dict(kind=kind, subkind=sub, x=x, y=y, source='osm',
                        osm_type=el['type'], osm_id=el['id'],
                        osm_tag=';'.join('%s=%s' % kv for kv in sorted(t.items())
                                         if kv[0] in ('leisure', 'amenity', 'tourism',
                                                      'historic', 'man_made', 'playground',
                                                      'attraction', 'shop')),
                        name=t.get('name'), footprint_m=foot))
    return out


# --------------------------------------------------------------------------- #
# fuse
# --------------------------------------------------------------------------- #
def fuse(verbose=True):
    t0 = time.time()
    surf = S.Surface()
    photo = json.load(open(SCR + '/photo_objects.json'))
    lidar = json.load(open(SCR + '/lidar_clusters.json'))['clusters']
    aer = json.load(open(SCR + '/aerial_objects.json'))
    props = load_props()
    road, plaza, grid = A.enu_mask()
    rdist, ridx, rnames, rgrid = roadway_masks()
    park_img, park_grid = parking_mask()
    ships = ship_polys()
    rgb, valid, gsd, ox, oy = A.load_ortho()
    from scipy.spatial import cKDTree

    prop_xy = np.array([[p['x'], p['y']] for p in props], 'f8') if props else np.zeros((0, 2))
    prop_kd = cKDTree(prop_xy) if len(prop_xy) else None

    def near_prop(x, y, classes, r=PROP_MATCH_M):
        if prop_kd is None:
            return None, 9e9
        best, bd = None, r
        for i in prop_kd.query_ball_point([x, y], r):
            if classes and props[i]['cls'] not in classes:
                continue
            d = math.hypot(props[i]['x'] - x, props[i]['y'] - y)
            if d < bd:
                best, bd = props[i], d
        return best, bd

    # ---- raw detections, one list per source ------------------------------ #
    det = []
    for o in photo['objects']:
        det.append(dict(kind=o['kind'], subkind=o['subkind'], x=o['x'], y=o['y'],
                        src='photo', views=o['n_views'], pairs=o['n_pairs'],
                        spread_m=o['spread_m'], yaw=o.get('yaw_deg'),
                        z_photo=o['z_est'], subkinds=o.get('subkinds')))
    for cl in lidar:
        _z, sname = surf.at(cl['x'], cl['y'], r=0.5)
        kind, sub = name_lidar(cl, sname == 'water')
        det.append(dict(kind=kind, subkind=sub, x=cl['x'], y=cl['y'], src='lidar',
                        points=cl['points'], height_m=cl['height_m'],
                        footprint_m=(cl['long_m'], cl['short_m']), yaw=cl['yaw_deg']))
    for o in aer['cars']:
        det.append(dict(kind='car', subkind='car', x=o['x'], y=o['y'], src='aerial',
                        footprint_m=(o['length_m'], o['width_m']), yaw=o['yaw_deg'],
                        rgb=o['rgb'], fill=o['fill'], contrast=o['contrast']))
    for o in aer['boats']:
        det.append(dict(kind='boat', subkind='moored', x=o['x'], y=o['y'], src='aerial',
                        footprint_m=(o['length_m'], o['width_m']), yaw=o['yaw_deg'],
                        rgb=o['rgb'], brightness=o['brightness']))
    for o in aer['blobs']:
        det.append(dict(kind='unknown_cluster', subkind='ortho_blob', x=o['x'],
                        y=o['y'], src='aerial', footprint_m=(o['length_m'], o['width_m']),
                        yaw=o['yaw_deg'], rgb=o['rgb'], contrast=o['contrast']))
    for o in osm_pass():
        det.append(dict(kind=o['kind'], subkind=o['subkind'], x=o['x'], y=o['y'],
                        src='osm', osm_id=o['osm_id'], osm_type=o['osm_type'],
                        osm_tag=o['osm_tag'], name=o['name'],
                        footprint_m=o['footprint_m']))
    raw_counts = collections.Counter((d['src'], d['kind']) for d in det)

    # ---- merge: transient kinds become position markers, static kinds fuse ------ #
    Ap0, Bp0 = S.path_bearings()
    objs = []
    for kind in sorted(set(d['kind'] for d in det)):
        spec = K.SPEC[kind]
        rows = [d for d in det if d['kind'] == kind]
        r = TRANSIENT_MERGE_M.get(kind, spec['match_m'] * 0.6)
        ell = VEHICLE_ELLIPSE.get(kind)
        P = np.array([[d['x'], d['y']] for d in rows], 'f8')
        kd = cKDTree(P)
        if ell:
            bear = np.array([S.nearest_path_yaw(Ap0, Bp0, x, y)[0] or 0.0
                             for x, y in P], 'f8')
        order = sorted(range(len(rows)),
                       key=lambda i: -(rows[i].get('views', 0) * 10
                                       + rows[i].get('points', 0)
                                       + (100 if rows[i]['src'] == 'osm' else 0)
                                       + (50 if rows[i]['src'] == 'aerial' else 0)))
        taken = np.zeros(len(rows), bool)
        for i in order:
            if taken[i]:
                continue
            if ell:
                a = math.radians(bear[i])
                ca, sa = math.cos(a), math.sin(a)
                cand = [j for j in kd.query_ball_point(P[i], max(ell))
                        if not taken[j]]
                idx = []
                for j in cand:
                    dx, dy = P[j] - P[i]
                    u = dx * ca + dy * sa; v = -dx * sa + dy * ca
                    if (u / ell[0]) ** 2 + (v / ell[1]) ** 2 <= 1.0:
                        idx.append(j)
            else:
                idx = [j for j in kd.query_ball_point(P[i], r) if not taken[j]]
            if not idx:
                continue
            taken[idx] = True
            grp = [rows[j] for j in idx]
            objs.append(merge_group(kind, grp))
    if verbose:
        print('merged %d detections into %d candidate objects (%.0fs)'
              % (len(det), len(objs), time.time() - t0), flush=True)

    # ---- gates, floor, tags ---------------------------------------------- #
    kept, dropped = [], []
    Ap, Bp = Ap0, Bp0
    for o in objs:
        spec = K.SPEC[o['kind']]
        srcs = set(o['sources'])
        # already placed by s80?
        fam = K.CONFIRMS.get(o['kind'])
        if fam:
            m, d = near_prop(o['x'], o['y'], fam,
                             POLE_PROP_MATCH_M if o['kind'] in
                             ('utility_pole', 'sign_frame', 'traffic_sign_face',
                              'sign_board') else PROP_MATCH_M)
            if m is not None:
                o['drop'] = 'already placed as %s %s (%.1f m)' % (m['cls'], m['id'], d)
                o['confirms_prop'] = m['id']
                dropped.append(o); continue
        # evidence gates
        if srcs == {'photo'} and o['views'] < spec['min_views']:
            o['drop'] = 'photo only, %d panoramas (needs %d)' % (o['views'], spec['min_views'])
            dropped.append(o); continue
        if srcs == {'lidar'}:
            need = LIDAR_MIN_PTS_UNKNOWN if o['kind'] == 'unknown_cluster' else LIDAR_MIN_PTS
            if o['points'] < need:
                o['drop'] = 'lidar only, %d returns (needs %d)' % (o['points'], need)
                dropped.append(o); continue
        if srcs == {'aerial'}:
            if o['kind'] == 'unknown_cluster':
                o['drop'] = 'ortho blob with no second sensor within %.1f m' % BLOB_CONFIRM_M
                dropped.append(o); continue
            if o['kind'] == 'car' and (o.get('fill', 0) < AERIAL_CAR_FILL
                                       or o.get('contrast', 0) < AERIAL_CAR_CONTRAST):
                o['drop'] = 'ortho car blob below the fill/contrast gate'
                dropped.append(o); continue
        if o['kind'] in STREET_KINDS and 'osm' not in srcs:
            if not near_mask(road, grid, o['x'], o['y'], STREET_NEAR_M):
                o['drop'] = 'street object more than %.0f m from any road or ' \
                            'parking surface' % STREET_NEAR_M
                dropped.append(o); continue
        # roadway prior for wheeled vehicles: a car standing on the lawn, on a
        # footpath or on a deck is a triangulation that missed the street, not
        # a car that was there.  Snap first, drop second.
        if o['kind'] in VEHICLE_KINDS:
            snap = snap_to_roadway(rdist, ridx, rnames, rgrid, o['x'], o['y'])
            if snap is None:
                o['drop'] = ('vehicle off every road and parking surface, and '
                             'more than %.1f m from one' % VEHICLE_SNAP_M)
                dropped.append(o); continue
            o['roadway'] = snap[3]
            if snap[2] > 0.0:
                o['x'] = round(float(snap[0]), 3)
                o['y'] = round(float(snap[1]), 3)
                o['snapped_to_roadway_m'] = snap[2]
        # floor
        z, sname = surf.at(o['x'], o['y'], r=0.5)
        if not np.isfinite(z):
            o['drop'] = 'off the model floor'
            dropped.append(o); continue
        if sname == 'water' and not spec['water_ok']:
            o['drop'] = 'stands in water'
            dropped.append(o); continue
        if sname != 'water' and o['kind'] == 'boat':
            o['drop'] = 'boat on dry land'
            dropped.append(o); continue
        if o['kind'] == 'boat':
            from shapely.geometry import Point
            pt = Point(o['x'], o['y'])
            hit = next((nm for poly, nm in ships if poly.contains(pt)), None)
            if hit is not None:
                o['drop'] = 'inside %s, already modelled as a structure' % (hit or 'a mapped vessel')
                dropped.append(o); continue
        if surf.in_building(o['x'], o['y']):
            o['drop'] = 'inside a building mass'
            dropped.append(o); continue
        o['z'] = round(float(z), 3); o['surface'] = sname
        # footprint / height: measured beats the placeholder
        box = spec['box_m']
        fp = o.get('footprint_m')
        o['footprint_m'] = [round(float(fp[0]), 2), round(float(fp[1]), 2)] if fp \
            else [box[0], box[1]]
        o['footprint_from'] = 'measured' if fp else 'class placeholder'
        o['height_m'] = round(float(o['height_m']), 2) if o.get('height_m') else box[2]
        o['height_from'] = 'lidar' if o.get('height_m') and 'lidar' in srcs else 'class placeholder'
        o['box_m'] = box
        # yaw
        if o.get('yaw') is not None:
            o['yaw_deg'] = round(float(o['yaw']) % 360.0, 1)
        elif o['kind'] in ('car', 'truck', 'bus', 'motorcycle'):
            yaw, d = S.nearest_path_yaw(Ap, Bp, o['x'], o['y'])
            o['yaw_deg'] = round(yaw, 1) if (yaw is not None and d < 20) else \
                round(360.0 * hash_unit((o['x'], o['y'])), 1)
            o['yaw_from'] = 'nearest road/path bearing' if d < 20 else 'hash'
        else:
            o['yaw_deg'] = round(360.0 * hash_unit((o['x'], o['y'])), 1)
            o['yaw_from'] = 'deterministic hash'
        o.setdefault('yaw_from', o.get('yaw_src', 'measured long axis'))
        # colour
        if o.get('rgb') is None:
            if max(o['footprint_m']) >= 1.2 and o['kind'] not in (
                    'utility_pole', 'traffic_light', 'banner', 'cctv_camera',
                    'sign_board', 'sign_frame', 'traffic_sign_face'):
                c = A.ortho_colour(rgb, gsd, ox, oy, o['x'], o['y'],
                                   r=max(0.4, min(o['footprint_m']) / 2.0))
                if c:
                    o['rgb'] = c
                    o['colour_from'] = 'ortho median over the footprint'
            else:
                o['colour_from'] = None
        else:
            o['colour_from'] = 'ortho blob pixels'
        if o.get('rgb'):
            o['colour_hex'] = '#%02x%02x%02x' % tuple(int(v) for v in o['rgb'])
        # tags
        if o['kind'] in ('car', 'truck', 'bus'):
            o['subkind'] = 'parked' if A.mask_at(park_img, park_grid, o['x'], o['y']) \
                else 'traffic'
        o['static'] = bool(spec['static'])
        o['role'] = 'scenery' if spec['static'] else 'transient_candidate'
        o['asset'] = spec['asset']
        o['confidence'] = round(confidence(o), 2)
        if o['kind'] == 'unknown_cluster' and o['confidence'] < UNKNOWN_MIN_CONF:
            o['drop'] = ('unnamed cluster at confidence %.2f, under the %.2f '
                         'placeholder floor' % (o['confidence'], UNKNOWN_MIN_CONF))
            dropped.append(o); continue
        kept.append(o)
    kept.sort(key=lambda o: (o['kind'], round(o['x'], 2), round(o['y'], 2)))
    for i, o in enumerate(kept):
        o['id'] = 'obj_%04d' % (i + 1)
        for k in ('yaw', 'src', 'drop'):
            o.pop(k, None)
    # evidence for the ortho-only boats, recorded on the row (ids are
    # final by now, so nothing else renumbers); s85 retires row['evidence']
    # ['aerial_gate']['retire'] (and recomputes it when a row lacks the tag)
    for oid, ev in aerial_boat_evidence(kept, surf).items():
        next(o for o in kept if o['id'] == oid).setdefault('evidence', {})['aerial_gate'] = ev
    rep = report(kept, dropped, det, raw_counts, props, photo, lidar, aer, surf)
    out = dict(what='tagged inventory of the objects in the area that the '
                    's80 furniture inventory does not already place',
               frame='photo-build ENU, x=East y=North z=Up, metres',
               generated=time.strftime('%Y-%m-%dT%H:%M:%S'),
               sources=dict(
                   photo='Mapillary detections (CC BY-SA 4.0 metadata), '
                         '%d panoramas, triangulated pairs' % photo['panoramas'],
                   aerial=aer['method'] + '; ' + aer['ortho']['source'],
                   lidar='USGS 3DEP returns, registered to the photo frame by the '
                         'combined build',
                   osm='OpenStreetMap tags (ODbL)'),
               objects=kept, report=rep)
    json.dump(out, open(C.W + '/object_inventory.json', 'w'), indent=1)
    json.dump(dict(dropped=[{k: v for k, v in d.items() if k != 'members'}
                            for d in dropped]),
              open(SCR + '/object_dropped.json', 'w'), indent=1)
    write_csv(kept, C.W + '/object_inventory.csv')
    json.dump(rep, open(SCR + '/reports/object_report.json', 'w'), indent=1)
    print(json.dumps({k: rep[k] for k in ('counts_by_kind', 'counts_by_source',
                                          'agreement', 'osm_refind')}, indent=1)[:4000])
    return out


def merge_group(kind, grp):
    """One object from every detection that landed inside the merge radius."""
    w = []
    for g in grp:
        w.append({'osm': 6.0, 'aerial': 3.0, 'lidar': 2.0,
                  'photo': 1.0 + 0.1 * g.get('views', 0)}[g['src']])
    w = np.array(w)
    X = np.array([[g['x'], g['y']] for g in grp])
    p = (X * w[:, None]).sum(0) / w.sum()
    o = dict(kind=kind, x=float(p[0]), y=float(p[1]),
             sources=sorted(set(g['src'] for g in grp)),
             n_detections=len(grp),
             views=sum(g.get('views', 0) for g in grp),
             points=max([g.get('points', 0) for g in grp] or [0]),
             sightings=sum(1 for g in grp if g['src'] == 'photo'))
    subs = collections.Counter()
    for g in grp:
        subs[g.get('subkind') or kind] += g.get('views', 1)
    o['subkind'] = subs.most_common(1)[0][0]
    o['subkind_votes'] = {k: int(v) for k, v in subs.most_common(5)}
    for key in ('footprint_m', 'height_m', 'rgb', 'fill', 'contrast',
                'osm_id', 'osm_type', 'osm_tag', 'name', 'brightness'):
        for pref in ('osm', 'aerial', 'lidar', 'photo'):
            v = next((g.get(key) for g in grp if g['src'] == pref and g.get(key)), None)
            if v is not None:
                o[key] = v; break
    for pref in ('aerial', 'lidar', 'photo'):
        v = next((g.get('yaw') for g in grp if g['src'] == pref
                  and g.get('yaw') is not None), None)
        if v is not None:
            o['yaw'] = float(v)
            o['yaw_src'] = {'aerial': 'ortho minimum-area rectangle',
                            'lidar': 'lidar cluster long axis',
                            'photo': 'triangulated point cloud long axis'}[pref]
            break
    o['evidence'] = {s: sum(1 for g in grp if g['src'] == s)
                     for s in o['sources']}
    if o['views']:
        o['evidence']['photo_panoramas'] = o['views']
    if o['points']:
        o['evidence']['lidar_returns'] = o['points']
    return o


def confidence(o):
    s = set(o['sources'])
    c = 0.0
    if 'osm' in s:
        c = max(c, 0.9)
    if 'photo' in s:
        c = max(c, min(0.85, 0.3 + 0.04 * o['views']))
    if 'lidar' in s:
        c = max(c, min(0.8, 0.35 + 0.01 * o['points']))
    if 'aerial' in s:
        c = max(c, 0.6)
    c += 0.12 * (len(s) - 1)
    return float(min(0.99, c))


CSV_COLS = ['id', 'kind', 'subkind', 'x', 'y', 'z', 'surface', 'yaw_deg',
            'footprint_m', 'height_m', 'colour_hex', 'static', 'role', 'asset',
            'sources', 'confidence', 'views', 'points', 'n_detections', 'name']


def write_csv(rows, path):
    import csv
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(CSV_COLS)
        for o in rows:
            wr.writerow([json.dumps(o.get(c)) if isinstance(o.get(c), (list, dict))
                         else o.get(c, '') for c in CSV_COLS])


# --------------------------------------------------------------------------- #
# report: counts, agreement, and how much of OSM the sensors re-found
# --------------------------------------------------------------------------- #
def report(kept, dropped, det, raw_counts, props, photo, lidar, aer, surf):
    from scipy.spatial import cKDTree
    by_kind = collections.Counter(o['kind'] for o in kept)
    by_src = collections.Counter()
    for o in kept:
        for s in o['sources']:
            by_src[s] += 1
    multi = collections.Counter(len(o['sources']) for o in kept)
    # agreement: of the objects one sensor proposed, how many did another confirm
    agree = {}
    for src in ('photo', 'lidar', 'aerial'):
        prop = [o for o in kept if src in o['sources']]
        both = [o for o in prop if len(o['sources']) > 1]
        agree[src] = dict(kept_with_this_source=len(prop),
                          also_seen_by_another=len(both),
                          rate=round(len(both) / max(len(prop), 1), 3))
    # per-kind agreement for the transient pair that matters
    for kind in ('car', 'boat'):
        rows = [o for o in kept if o['kind'] == kind]
        agree['%s_sources' % kind] = dict(collections.Counter(
            '+'.join(o['sources']) for o in rows))
    # cross-sensor: of the RAW detections one sensor made of a kind, how many
    # have a detection of the same kind from another sensor within its fusion
    # radius.  This is measured before the gates, so it is a property of the
    # sensors, not of the inventory.
    cross = {}
    bykind = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in det:
        bykind[d['kind']][d['src']].append((d['x'], d['y']))
    for kind, per in bykind.items():
        if len(per) < 2:
            continue
        r = K.SPEC[kind]['match_m']
        for a in per:
            for b in per:
                if a == b:
                    continue
                kdb = cKDTree(np.array(per[b]))
                hit = sum(1 for p in per[a] if kdb.query_ball_point(p, r))
                cross['%s: %s in %s within %.1f m' % (kind, a, b, r)] = dict(
                    n=len(per[a]), matched=hit,
                    rate=round(hit / max(len(per[a]), 1), 3))
    # OSM re-find: for every OSM-mapped object in the area (the s80 inventory's
    # own OSM rows plus this stage's OSM rows), did photo or lidar see it?
    dets_by_src = collections.defaultdict(list)
    for d in det:
        dets_by_src[d['src']].append((d['x'], d['y']))
    kd = {s: cKDTree(np.array(v)) for s, v in dets_by_src.items() if v}
    osm_rows = [p for p in props if p.get('source') == 'osm']
    refind = {}
    for p in osm_rows:
        cls = p['cls']
        r = S.MATCH_M.get(cls, 3.0)
        hit_p = bool(kd['photo'].query_ball_point([p['x'], p['y']], r)) if 'photo' in kd else False
        hit_l = bool(kd['lidar'].query_ball_point([p['x'], p['y']], r)) if 'lidar' in kd else False
        e = refind.setdefault(cls, dict(osm_objects=0, photo=0, lidar=0, either=0,
                                        already_fused_in_s80=0))
        e['osm_objects'] += 1
        e['photo'] += int(hit_p); e['lidar'] += int(hit_l)
        e['either'] += int(hit_p or hit_l)
        e['already_fused_in_s80'] += int(len(set(p.get('sources', []))) > 1)
    for cls, e in refind.items():
        e['recall_either'] = round(e['either'] / max(e['osm_objects'], 1), 3)
    tot = dict(osm_objects=sum(e['osm_objects'] for e in refind.values()),
               either=sum(e['either'] for e in refind.values()))
    tot['recall_either'] = round(tot['either'] / max(tot['osm_objects'], 1), 3)
    drop_reasons = collections.Counter(
        (d.get('drop') or '').split('(')[0].strip() for d in dropped)
    return dict(
        objects=len(kept),
        counts_by_kind=dict(by_kind.most_common()),
        counts_by_source={k: int(v) for k, v in by_src.most_common()},
        objects_by_source_count={str(k): int(v) for k, v in sorted(multi.items())},
        static=sum(1 for o in kept if o['static']),
        transient_candidates=sum(1 for o in kept if not o['static']),
        with_asset=sum(1 for o in kept if o['asset']),
        placeholder_only=sum(1 for o in kept if not o['asset']),
        raw_detections={'%s/%s' % k: int(v) for k, v in raw_counts.most_common()},
        raw_detections_total=len(det),
        dropped=len(dropped),
        drop_reasons={k: int(v) for k, v in drop_reasons.most_common()},
        vehicles=dict(
            kept=sum(1 for o in kept if o['kind'] in VEHICLE_KINDS),
            snapped_to_roadway=sum(1 for o in kept if o.get('snapped_to_roadway_m')),
            median_snap_m=round(float(np.median(
                [o['snapped_to_roadway_m'] for o in kept
                 if o.get('snapped_to_roadway_m')] or [0.0])), 2),
            by_surface=dict(collections.Counter(
                o.get('surface') for o in kept
                if o['kind'] in VEHICLE_KINDS).most_common()),
            by_roadway=dict(collections.Counter(
                o.get('roadway') for o in kept
                if o['kind'] in VEHICLE_KINDS).most_common()),
            rule='kept only on an OSM road or parking surface (%s), after a '
                 'snap of up to %.1f m onto the nearest one'
                 % ('/'.join(ROADWAY_KINDS), VEHICLE_SNAP_M)),
        agreement=agree,
        cross_sensor=cross,
        osm_refind=dict(per_class=refind, total=tot),
        subkinds={k: dict(collections.Counter(
            o['subkind'] for o in kept if o['kind'] == k).most_common(8))
            for k in by_kind},
        vocabulary=vocabulary(),
        panoramas=photo['panoramas'], lidar_clusters=len(lidar),
        aerial=dict(cars=len(aer['cars']), boats=len(aer['boats']),
                    blobs=len(aer['blobs'])))


def vocabulary():
    """What each sensor can even name, measured: the Mapillary census counts
    per kind inside the bbox, and the classes no sensor vocabulary has."""
    try:
        cen = json.load(open(SCR + '/census_photo.json'))
    except Exception:
        return {}
    per = collections.Counter(); skipped = collections.Counter()
    for r in cen['classes']:
        if not r['n_in_bbox']:
            continue
        kk = K.photo_kind(r['value'])
        if kk:
            per[kk[0]] += r['n_in_bbox']
        else:
            skipped[r['value'].split('--')[0]] += r['n_in_bbox']
    asked = {
        'boats': 'photo class object--vehicle--boat, plus the ortho over water',
        'cars': 'photo class object--vehicle--car, plus the ortho inside parking/road',
        'trucks': 'photo class object--vehicle--truck',
        'bicycles': 'photo class object--vehicle--bicycle',
        'planters': 'NO photo class; only the ortho blob pass and lidar clusters',
        'sculptures/statues': 'NO photo class; OSM tourism=artwork / historic=memorial only',
        'kiosks': 'NO photo class; lidar box clusters 2-4 m tall only',
        'tables': 'NO photo class; s80 already places OSM leisure=picnic_table',
        'umbrellas': 'NO photo class, no OSM tag, no lidar signature at 0.5 m cells',
        'playground equipment': 'NO photo class; OSM leisure=playground area only',
        'flags': 'NO photo class; s80 places OSM man_made=flagpole',
        'fountains': 'NO photo class; OSM amenity=fountain only if mapped',
        'signs by type': 'photo classes regulatory--*/warning--*/information--*, '
                         'kept as traffic_sign_face subkinds',
        'banners': 'photo class object--banner',
        'bike-share stations': 'NO photo class; OSM amenity=bicycle_parking only',
        'food carts': 'NO photo class and no OSM tag is read',
    }
    return dict(photo_detections_in_bbox_by_kind={k: int(v) for k, v in per.most_common()},
                photo_classes_present=len([r for r in cen['classes'] if r['n_in_bbox']]),
                photo_detections_not_an_object_class={k: int(v) for k, v in skipped.most_common()},
                requested_classes=asked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fuse-only', action='store_true')
    ap.add_argument('--boat-evidence', action='store_true',
                    help='print the aerial-boat gate evidence of object_inventory.json')
    ap.add_argument('--rerun', action='store_true', help='re-run the three passes')
    a = ap.parse_args()
    if a.boat_evidence:
        rows = json.load(open(C.W + '/object_inventory.json'))['objects']
        aerial_boat_evidence(rows, verbose=True)
        return
    py = sys.executable
    if a.rerun:
        for m in ('objects.objcensus', 'objects.objphoto', 'objects.objaerial',
                  'objects.objlidar'):
            subprocess.check_call([py, '-m', m], cwd=C.W)
    fuse()


if __name__ == '__main__':
    main()
