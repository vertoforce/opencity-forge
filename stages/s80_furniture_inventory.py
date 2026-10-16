"""Stage 80 -- the street-furniture inventory: one row per small object that
belongs on the map as a real 3D prop.

Same shape as the tree work (s03/s04/s05 build
tree_inventory.json, s38 swaps models in): this stage only MEASURES and writes
an inventory; s81 bakes the asset library and s82 rewrites the published GLBs.

Three independent sources, fused at the end:

  osm     tagged nodes and ways inside the model bbox (osm/osm_raw.json, the
          fetch s02_osm.py already made).  Authoritative for class and id.
  lidar   pole-like vertical returns in the registered 3DEP point cloud that no
          OSM node explains: lamp posts, sign posts, masts.
  photo   Mapillary per-image detections (the cache s40_masks.py filled),
          triangulated across panoramas with the poses in pano_poses.json.

Frame: photo-build ENU, x=East y=North z=Up, metres (common.py).  The GLB
convention the model uses is (x, z, -y), applied in s82, not here.

Outputs (scratch, then copied to the work dir by `fuse`):
  prop_inventory.json   {"props": [...], "counts": {...}, "sources": {...}}
  prop_inventory.csv    the same rows, flat
"""
import os, sys, json, math, time, gzip, base64, argparse, collections, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import opencity_paths as OCP                    # noqa: E402  host roots, env-overridable
import numpy as np
import common as C

SCR = OCP.SHARED + '/props'
DET = OCP.SOURCE + '/detections'
# per-area paths when an area is set (pipeline/areacfg.module_paths);
# otherwise the defaults above.
_AP = C._areacfg.module_paths(C) if hasattr(C, '_areacfg') else None
if _AP:
    SCR = _AP['PROPS']; DET = _AP['DET']
CELL = 0.5
os.makedirs(SCR, exist_ok=True)

# --------------------------------------------------------------------------- #
# classes.  One entry per placeable prop class; `asset` is the key
# s81/s82 look up in the baked asset library.
# --------------------------------------------------------------------------- #
CLASSES = {
    'bench':          dict(asset='bench',          radius=0.9,  align='path'),
    'lamp_post':      dict(asset='lamp_post',      radius=0.3,  align='hash'),
    'bollard':        dict(asset='bollard',        radius=0.2,  align='hash'),
    'trash_can':      dict(asset='trash_can',      radius=0.35, align='hash'),
    'bike_rack':      dict(asset='bike_rack',      radius=1.0,  align='path'),
    'picnic_table':   dict(asset='picnic_table',   radius=1.1,  align='path'),
    'sign_post':      dict(asset='sign_post',      radius=0.3,  align='path'),
    'flagpole':       dict(asset='flagpole',       radius=0.3,  align='hash'),
    'railing_segment':dict(asset='railing_segment',radius=1.1,  align='way'),
    'mooring_post':   dict(asset='mooring_post',   radius=0.25, align='hash'),
    'small_boat':     dict(asset='small_boat',     radius=2.5,  align='way'),
    'drinking_water': dict(asset='bollard',        radius=0.25, align='hash'),
    'mast':           dict(asset='flagpole',       radius=0.3,  align='hash'),
    'fire_hydrant':   dict(asset='bollard',        radius=0.25, align='hash'),
}

# OSM tag -> class.  Only tags that name a physical small object.
OSM_NODE_RULES = [
    (('amenity', 'bench'), 'bench'),
    (('amenity', 'waste_basket'), 'trash_can'),
    (('amenity', 'bicycle_parking'), 'bike_rack'),
    (('amenity', 'drinking_water'), 'drinking_water'),
    (('highway', 'street_lamp'), 'lamp_post'),
    (('barrier', 'bollard'), 'bollard'),
    (('man_made', 'flagpole'), 'flagpole'),
    (('man_made', 'mast'), 'mast'),
    (('man_made', 'lighthouse'), 'mast'),
    (('leisure', 'picnic_table'), 'picnic_table'),
    (('tourism', 'information'), 'sign_post'),
    (('tourism', 'artwork'), 'sign_post'),
    (('emergency', 'fire_hydrant'), 'fire_hydrant'),
    (('mooring', None), 'mooring_post'),
    (('seamark:type', 'mooring'), 'mooring_post'),
]
# linear runs: instanced segments along the way
OSM_WAY_LINEAR = [
    (('barrier', 'fence'), 'railing_segment'),
    (('barrier', 'railing'), 'railing_segment'),
    (('barrier', 'handrail'), 'railing_segment'),
    (('barrier', 'guard_rail'), 'railing_segment'),
    (('barrier', 'wall'), 'railing_segment'),
]
# closed ways that are a single object: place one prop at the centroid
OSM_WAY_AREA = [
    (('historic', 'ship'), 'small_boat'),
    (('historic', 'naval'), 'small_boat'),
    (('amenity', 'bench'), 'bench'),
    (('barrier', 'bollard'), 'bollard'),
]
SEG_LEN = 2.0          # m, one railing instance per this much of the way

# Mapillary detection classes trusted to be a placeable object
PHOTO_CLASSES = {
    'object--street-light': 'lamp_post',
    'object--bench': 'bench',
    'object--trash-can': 'trash_can',
    'object--traffic-sign--front': 'sign_post',
    'object--support--pole': 'sign_post',
    'object--fire-hydrant': 'fire_hydrant',
    'object--bike-rack': 'bike_rack',
}
PHOTO_MAX_PER_CLASS_PER_PANO = 12
PHOTO_GAP_M = 1.0         # two rays "agree" when their closest approach is this
PHOTO_MAX_RANGE_M = 35.0  # ignore a ray longer than this; detections get mushy
PHOTO_MIN_BASE_M = 1.5    # two cameras closer than this do not triangulate
PHOTO_PAIR_K = 12         # nearest panos considered as the partner of each pano
PHOTO_MIN_ANGLE_DEG = 12.0   # two near-parallel rays cross anywhere; ignore them
# How far above the model floor the triangulated point may sit, per class.
# A bench or a bin is detected down to its feet; a street light is often
# detected as the luminaire alone, which is 4-6 m up, so it gets a tall window.
PHOTO_DZ_WINDOW = {'lamp_post': (-1.0, 9.0), 'sign_post': (-1.0, 3.0),
                   'bench': (-1.0, 1.5), 'trash_can': (-1.0, 1.5),
                   'fire_hydrant': (-1.0, 1.5), 'bike_rack': (-1.0, 1.5)}
MATCH_M = {'lamp_post': 4.0, 'bench': 3.0, 'trash_can': 3.0, 'sign_post': 3.0,
           'bike_rack': 4.0, 'fire_hydrant': 3.0, 'bollard': 2.0,
           'mast': 5.0, 'picnic_table': 3.0}


def seeded(key, salt=''):
    h = hashlib.sha256((str(key) + '|' + salt).encode()).digest()
    return int.from_bytes(h[:8], 'little')


def hash_yaw(key):
    return 360.0 * (((seeded(key) >> 8) & 0xFFFF) / 65536.0)


# --------------------------------------------------------------------------- #
# surfaces: one 0.5 m raster per floor layer, so a prop can be put ON the model
# --------------------------------------------------------------------------- #
class Surface:
    """Height and class of the model's floor at any XY, measured off the same
    layer meshes the GLB is built from (layers/*.npz)."""
    F = SCR + '/surface.npz'

    def __init__(self, rebuild=False):
        if os.path.exists(self.F) and not rebuild:
            d = np.load(self.F)
            self.__dict__.update({k: d[k] for k in d.files})
            self.grid = [float(v) for v in d['grid']]
            return
        from s10a_3dfier import raster_tin
        pos = np.load(C.W + '/layers/ground_terrain.npz')['pos']
        gx0 = math.floor(pos[:, 0].min() - 5); gy0 = math.floor(pos[:, 1].min() - 5)
        nx = int(math.ceil((pos[:, 0].max() + 5 - gx0) / CELL))
        ny = int(math.ceil((pos[:, 1].max() + 5 - gy0) / CELL))
        out = {}
        for name in ('ground_terrain', 'paths', 'structures', 'water', 'buildings'):
            if not os.path.exists(C.W + '/layers/%s.npz' % name):
                # 3dfier writes no file for an empty layer (an area with no paths
                # has no paths layer); an absent layer covers no cell.
                out['z_' + name] = np.full((ny, nx), np.nan, 'f4'); continue
            z = np.load(C.W + '/layers/%s.npz' % name)
            V = z['pos'].astype('f8'); F = z['idx'].astype('i4')
            t0 = time.time()
            Z, _TI = raster_tin(V, F, gx0, gy0, CELL, nx, ny)
            out['z_' + name] = Z.astype('f4')
            print('%-16s %dx%d %5.1f%% covered %5.0fs'
                  % (name, nx, ny, np.isfinite(Z).mean() * 100, time.time() - t0),
                  flush=True)
        out['grid'] = np.array([gx0, gy0, CELL, nx, ny], 'f8')
        np.savez_compressed(self.F, **out)
        self.__dict__.update(out)
        self.grid = [gx0, gy0, CELL, nx, ny]

    def _ij(self, x, y):
        gx0, gy0, c, nx, ny = self.grid
        j = np.clip(((np.asarray(x) - gx0) / c).astype(np.int32), 0, int(nx) - 1)
        i = np.clip(((np.asarray(y) - gy0) / c).astype(np.int32), 0, int(ny) - 1)
        return i, j

    def _disc(self, Z, x, y, r, fn):
        """fn over the cells inside radius r of (x,y); NaN when none are set."""
        n = max(1, int(round(r / CELL)))
        vals = []
        for dy in range(-n, n + 1):
            for dx in range(-n, n + 1):
                if dx * dx + dy * dy > n * n:
                    continue
                i, j = self._ij(x + dx * CELL, y + dy * CELL)
                vals.append(Z[i, j])
        v = np.array(vals)
        v = v[np.isfinite(v)]
        return fn(v) if len(v) else float('nan')

    def at(self, x, y, r=0.0):
        """-> (z, surface_name).  Deck wins over path wins over terrain; water
        is only reported when nothing solid covers the cell."""
        for nm, key in (('deck', 'z_structures'), ('path', 'z_paths'),
                        ('terrain', 'z_ground_terrain')):
            z = self._disc(self.__dict__[key], x, y, r, np.nanmedian) if r > 0 else \
                float(self.__dict__[key][self._ij(x, y)])
            if np.isfinite(z):
                return float(z), nm
        z = float(self.z_water[self._ij(x, y)])
        if np.isfinite(z):
            return z, 'water'
        return float('nan'), 'none'

    def floor_z(self, x, y):
        """Height of the model floor at XY: deck, else path, else terrain."""
        i, j = self._ij(x, y)
        for k in ('z_structures', 'z_paths', 'z_ground_terrain'):
            z = self.__dict__[k][i, j]
            if np.isfinite(z):
                return float(z)
        return float('nan')

    def in_building(self, x, y, r=0.0):
        z = self._disc(self.z_buildings, x, y, r, np.nanmax) if r > 0 else \
            float(self.z_buildings[self._ij(x, y)])
        return bool(np.isfinite(z))

    def terrain(self, x, y):
        i, j = self._ij(x, y)
        z = self.z_ground_terrain[i, j]
        if not np.isfinite(z):
            z = self.z_paths[i, j]
        return float(z)


def enu_of(lat, lon):
    p = C.lla_to_enu(np.asarray([lat], 'f8'), np.asarray([lon], 'f8'),
                     np.asarray([0.0], 'f8'))[0]
    return float(p[0]), float(p[1])


# --------------------------------------------------------------------------- #
# 1. OSM
# --------------------------------------------------------------------------- #
def tag_class(t, rules):
    for (k, v), cls in rules:
        if k in t and (v is None or t[k] == v):
            return cls, '%s=%s' % (k, t[k])
    return None, None


def osm_pass():
    els = json.load(open(C.W + '/osm/osm_raw.json'))['elements']
    w, s, e, n = C.BBOX_WSEN
    out = []
    seen = set()
    ways_pts = []
    for el in els:
        t = el.get('tags') or {}
        if not t:
            continue
        if el['type'] == 'node':
            cls, why = tag_class(t, OSM_NODE_RULES)
            if cls is None:
                continue
            if not (w <= el['lon'] <= e and s <= el['lat'] <= n):
                continue
            key = ('node', el['id'])
            if key in seen:
                continue
            seen.add(key)
            x, y = enu_of(el['lat'], el['lon'])
            h = None
            for hk in ('height', 'lamp_height'):
                try:
                    h = float(str(t[hk]).split()[0])
                except Exception:
                    pass
            out.append(dict(cls=cls, source='osm', osm_type='node', osm_id=el['id'],
                            osm_tag=why, x=x, y=y, confidence=1.0,
                            height_m=h, name=t.get('name')))
        elif el['type'] == 'way':
            g = el.get('geometry') or []
            if len(g) < 2:
                continue
            pts = np.array([[*enu_of(p['lat'], p['lon'])] for p in g], 'f8')
            closed = (el['nodes'][0] == el['nodes'][-1]) if el.get('nodes') else False
            cls, why = tag_class(t, OSM_WAY_AREA) if closed else (None, None)
            if cls is not None:
                cen = pts[:-1].mean(0) if closed else pts.mean(0)
                d = pts - cen
                cov = d.T @ d
                ev, evec = np.linalg.eigh(cov)
                ax = evec[:, int(np.argmax(ev))]
                length = float(np.ptp(d @ ax))
                out.append(dict(cls=cls, source='osm', osm_type='way', osm_id=el['id'],
                                osm_tag=why, x=float(cen[0]), y=float(cen[1]),
                                confidence=1.0, yaw_deg=float(math.degrees(
                                    math.atan2(ax[1], ax[0]))) % 360.0,
                                length_m=round(length, 2), name=t.get('name')))
                continue
            cls, why = tag_class(t, OSM_WAY_LINEAR)
            if cls is None:
                continue
            ways_pts.append((el['id'], why, t.get('name'), pts))
    # linear runs -> instanced segments
    nseg = 0
    for wid, why, nm, pts in ways_pts:
        seg = np.diff(pts, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        total = float(L.sum())
        if total < 1.0:
            continue
        cum = np.concatenate([[0.0], np.cumsum(L)])
        k = max(1, int(round(total / SEG_LEN)))
        step = total / k
        for m in range(k):
            sm = (m + 0.5) * step
            i = int(np.searchsorted(cum, sm) - 1)
            i = max(0, min(i, len(L) - 1))
            f = (sm - cum[i]) / max(L[i], 1e-9)
            p = pts[i] + f * seg[i]
            yaw = math.degrees(math.atan2(seg[i][1], seg[i][0])) % 360.0
            out.append(dict(cls='railing_segment', source='osm', osm_type='way',
                            osm_id=wid, osm_tag=why, x=float(p[0]), y=float(p[1]),
                            confidence=1.0, yaw_deg=yaw, length_m=round(step, 3),
                            run_index=m, run_segments=k, name=nm))
            nseg += 1
    json.dump(out, open(SCR + '/osm_props.json', 'w'))
    c = collections.Counter(o['cls'] for o in out)
    print('osm props', len(out), dict(c), '(%d railing segments)' % nseg, flush=True)
    return out


# --------------------------------------------------------------------------- #
# 2. lidar poles
# --------------------------------------------------------------------------- #
POLE_MIN_H = 2.2      # m above terrain: below this it is furniture, not a pole
POLE_MAX_H = 18.0
POLE_MIN_PTS = 2      # returns in a 0.5 m cell; 3DEP is thin on poles
POLE_ISO = 0.35       # ring points / core points must be under this


def lidar_pass(surf):
    reg = np.load(C.FW + '/lidar_reg.npy')
    cls = np.load(C.FW + '/loaded.npz')['lidar_cls']
    if len(cls) != len(reg):
        cls = np.zeros(len(reg), 'i2')
    gx0, gy0, c, nx, ny = surf.grid
    x, y, z = reg[:, 0], reg[:, 1], reg[:, 2]
    inb = ((x > gx0) & (x < gx0 + nx * c) & (y > gy0) & (y < gy0 + ny * c)
           & (cls != 2) & (cls != 9))
    x, y, z = x[inb], y[inb], z[inb]
    i, j = surf._ij(x, y)
    zt = surf.z_ground_terrain[i, j]
    zp = surf.z_paths[i, j]
    zd = surf.z_structures[i, j]
    base = np.where(np.isfinite(zd), zd, np.where(np.isfinite(zp), zp, zt))
    hag = z - base
    keep = np.isfinite(hag) & (hag > POLE_MIN_H) & (hag < POLE_MAX_H)
    x, y, hag = x[keep], y[keep], hag[keep]
    i, j = surf._ij(x, y)
    cnt = np.zeros((int(ny), int(nx)), 'i4')
    np.add.at(cnt, (i, j), 1)
    hmax = np.zeros((int(ny), int(nx)), 'f4')
    np.maximum.at(hmax, (i, j), hag.astype('f4'))
    # core = 3x3 (1.5 m), ring = 7x7 minus core (up to 3.5 m).  A pole is a
    # column of returns with almost nothing around it; a tree or a wall is not.
    from scipy.ndimage import uniform_filter, label
    core = uniform_filter(cnt.astype('f4'), 3) * 9
    wide = uniform_filter(cnt.astype('f4'), 7) * 49
    ring = wide - core
    cand = (cnt >= POLE_MIN_PTS) & (core >= POLE_MIN_PTS) & (ring <= POLE_ISO * core)
    # never inside a building footprint
    cand &= ~np.isfinite(surf.z_buildings)
    lab, n = label(cand)
    out = []
    if n:
        idx = np.arange(1, n + 1)
        for k in idx:
            m = lab == k
            if m.sum() > 12:            # 3 m^2 of "pole" is a wall, not a pole
                continue
            jj, ii = np.meshgrid(np.arange(int(nx)), np.arange(int(ny)))
            wj = jj[m]; wi = ii[m]; wc = cnt[m].astype('f8')
            px = gx0 + (wj + 0.5) * c
            py = gy0 + (wi + 0.5) * c
            X = float((px * wc).sum() / wc.sum()); Y = float((py * wc).sum() / wc.sum())
            H = float(hmax[m].max())
            out.append(dict(x=X, y=Y, height_m=round(H, 2),
                            points=int(wc.sum()), cells=int(m.sum())))
    # drop poles standing in trees (the tree inventory is the better source)
    tr = json.load(open(C.W + '/tree_inventory.json'))['trees']
    T = np.array([[t['x_enu_m'], t['y_enu_m']] for t in tr], 'f8')
    kept = []
    for o in out:
        d = np.hypot(T[:, 0] - o['x'], T[:, 1] - o['y']).min() if len(T) else 9e9
        if d < 2.5:
            o['reject'] = 'inside a tree crown (%.1f m from tree)' % d
            continue
        kept.append(o)
    json.dump(kept, open(SCR + '/lidar_poles.json', 'w'))
    print('lidar pole candidates', len(out), 'kept after tree veto', len(kept), flush=True)
    return kept


# --------------------------------------------------------------------------- #
# 3. Mapillary detections, triangulated
#
# Rule, stated once here because the report quotes it:
#   * only panoramas whose pose tier is high or usable are used (pano_poses.json);
#   * a detection contributes one ray, through the BASE of its polygon (the
#     bottom 10% of the outline, where the object meets the ground), in the
#     equirect convention recorded in pano_poses.json['meta'];
#   * two rays from two different panoramas at least 1.5 m apart are a match
#     when their closest approach is <= 1.0 m and the midpoint is within 35 m
#     of both cameras and inside the model bbox;
#   * matched midpoints are single-link clustered at 1.0 m and a cluster is
#     kept when at least 2 distinct panoramas voted for it.
# --------------------------------------------------------------------------- #
def det_path(pid):
    return '%s/%s/%s.json.gz' % (DET, str(pid)[-2:], pid)


def rays_for_pano(pid, pose, want):
    """-> list of (cls, origin(3), dir(3), area) for one panorama."""
    import mapbox_vector_tile
    p = det_path(pid)
    if not os.path.exists(p):
        return []
    try:
        data = json.loads(gzip.open(p, 'rt').read()).get('data', [])
    except Exception:
        return []
    R = np.array(pose['R'], 'f8')
    Cc = np.array([pose['x'], pose['y'], pose['z']], 'f8')
    per = collections.defaultdict(list)
    for d in data:
        cls = want.get(d.get('value') or '')
        if cls is None or not d.get('geometry'):
            continue
        try:
            dec = mapbox_vector_tile.decode(base64.decodebytes(d['geometry'].encode()))
        except Exception:
            continue
        for lay in dec.values():
            for ft in lay.get('features', []):
                rings = []

                def walk(cc):
                    if not len(cc):
                        return
                    if isinstance(cc[0], (int, float)):
                        return
                    if isinstance(cc[0][0], (int, float)):
                        rings.append(np.asarray(cc, 'f8')); return
                    for q in cc:
                        walk(q)
                walk((ft.get('geometry') or {}).get('coordinates', []))
                for r in rings:
                    if len(r) < 3:
                        continue
                    uf = r[:, 0] / 4096.0
                    vf = 1.0 - r[:, 1] / 4096.0        # tile y is up (s40_masks)
                    area = float((uf.max() - uf.min()) * (vf.max() - vf.min()))
                    base = vf >= np.percentile(vf, 90)
                    u = float(uf[base].mean()); v = float(vf[base].max())
                    lon = 2 * math.pi * (u - 0.5)
                    lat = math.pi * (0.5 - v)
                    X = np.array([math.cos(lat) * math.sin(lon), -math.sin(lat),
                                  math.cos(lat) * math.cos(lon)], 'f8')
                    dw = R.T @ X
                    dw /= np.linalg.norm(dw)
                    per[cls].append((area, Cc, dw))
    out = []
    for cls, lst in per.items():
        lst.sort(key=lambda t: -t[0])
        for area, o, dw in lst[:PHOTO_MAX_PER_CLASS_PER_PANO]:
            out.append((cls, o, dw, area))
    return out


def closest_point(o1, d1, o2, d2):
    """-> (midpoint, gap, t1, t2) of the closest approach of two rays."""
    w = o1 - o2
    a = float(d1 @ d1); b = float(d1 @ d2); c = float(d2 @ d2)
    d = float(d1 @ w); e = float(d2 @ w)
    den = a * c - b * b
    if abs(den) < 1e-9:
        return None, 9e9, 0.0, 0.0
    t1 = (b * e - c * d) / den
    t2 = (a * e - b * d) / den
    p1 = o1 + t1 * d1; p2 = o2 + t2 * d2
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2)), t1, t2


def photo_pass(surf, limit=None):
    t0 = time.time()
    P = json.load(open(C.W + '/pano_poses.json'))['poses']
    ids = [k for k, v in P.items() if v.get('confidence') in ('high', 'usable')
           and os.path.exists(det_path(k))]
    gx0, gy0, c, nx, ny = surf.grid
    ids = [k for k in ids if gx0 - 20 < P[k]['x'] < gx0 + nx * c + 20
           and gy0 - 20 < P[k]['y'] < gy0 + ny * c + 20]
    if limit:
        ids = ids[:limit]
    XY = np.array([[P[k]['x'], P[k]['y']] for k in ids], 'f8')
    print('panoramas used', len(ids), flush=True)
    from scipy.spatial import cKDTree
    tree = cKDTree(XY)
    cache = {}

    def rays(k):
        if k not in cache:
            if len(cache) > 400:
                cache.clear()
            cache[k] = rays_for_pano(k, P[k], PHOTO_CLASSES)
        return cache[k]

    nray = 0
    votes = collections.defaultdict(list)     # cls -> list of (point, a, b)
    pairs = 0
    for ai, k in enumerate(ids):
        ra = rays(k)
        nray += len(ra)
        if not ra:
            continue
        nb = tree.query(XY[ai], k=min(PHOTO_PAIR_K + 1, len(ids)))[1]
        for bi in np.atleast_1d(nb):
            bi = int(bi)
            if bi <= ai:
                continue
            if np.linalg.norm(XY[ai] - XY[bi]) < PHOTO_MIN_BASE_M:
                continue
            rb = rays(ids[bi])
            if not rb:
                continue
            pairs += 1
            for cls1, o1, d1, _a1 in ra:
                for cls2, o2, d2, _a2 in rb:
                    if cls1 != cls2:
                        continue
                    pt, gap, t1, t2 = closest_point(o1, d1, o2, d2)
                    if pt is None or gap > PHOTO_GAP_M:
                        continue
                    if not (0.5 < t1 < PHOTO_MAX_RANGE_M and 0.5 < t2 < PHOTO_MAX_RANGE_M):
                        continue
                    if not (gx0 < pt[0] < gx0 + nx * c and gy0 < pt[1] < gy0 + ny * c):
                        continue
                    cosang = abs(float(d1 @ d2))
                    if cosang > math.cos(math.radians(PHOTO_MIN_ANGLE_DEG)):
                        continue
                    zs = surf.floor_z(pt[0], pt[1])
                    lo, hi = PHOTO_DZ_WINDOW.get(cls1, (-1.0, 2.0))
                    if not np.isfinite(zs) or not (lo <= pt[2] - zs <= hi):
                        continue
                    votes[cls1].append((pt, ai, bi))
        if ai % 250 == 0:
            print('  pano %d/%d  pairs %d  votes %d  %.0fs'
                  % (ai, len(ids), pairs, sum(len(v) for v in votes.values()),
                     time.time() - t0), flush=True)
    out = []
    for cls, vs in votes.items():
        if not vs:
            continue
        pts = np.array([v[0] for v in vs], 'f8')
        pa = np.array([[v[1], v[2]] for v in vs], 'i4')
        # grid-peak clustering, not single-link: single-link chains a whole
        # street of poles into one blob (thousands of views in one cluster).
        gi = np.floor((pts[:, 0] - gx0) / c).astype('i4')
        gj = np.floor((pts[:, 1] - gy0) / c).astype('i4')
        acc = collections.Counter(zip(gi.tolist(), gj.tolist()))
        order = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
        from scipy.spatial import cKDTree as KD
        kd = KD(pts[:, :2])
        taken = np.zeros(len(pts), bool)
        for (ci, cj), _n in order:
            cx = gx0 + (ci + 0.5) * c; cy = gy0 + (cj + 0.5) * c
            idx = np.array(kd.query_ball_point([cx, cy], PHOTO_GAP_M), 'i4')
            if not len(idx):
                continue
            idx = idx[~taken[idx]]
            if len(idx) < 1:
                continue
            panos = set(pa[idx].ravel().tolist())
            if len(panos) < 2:
                continue
            taken[idx] = True
            p = pts[idx].mean(0)
            sp = float(np.hypot(*(pts[idx][:, :2] - p[:2]).T).max())
            out.append(dict(cls=cls, x=float(p[0]), y=float(p[1]),
                            z_est=float(p[2]), n_views=len(panos),
                            n_pairs=int(len(idx)), spread_m=round(sp, 2)))
    json.dump(dict(objects=out, panoramas=len(ids), rays=nray, pairs=pairs,
                   seconds=round(time.time() - t0, 1)),
              open(SCR + '/photo_objects.json', 'w'))
    print('photo objects', len(out), collections.Counter(o['cls'] for o in out),
          'from %d rays in %d panoramas, %.0fs' % (nray, len(ids), time.time() - t0),
          flush=True)
    return out


# --------------------------------------------------------------------------- #
# 4. fuse
# --------------------------------------------------------------------------- #
POLE_CLASSES = ('lamp_post', 'sign_post', 'mast', 'flagpole')
# Evidence gates, set from the measurements in the report: a source on its own
# has to be strong, a source corroborated by another source does not.
LIDAR_MIN_POINTS_ALONE = 6     # a 2-return "pole" alone is noise
# Set from the measured view-count distribution: the Mapillary class
# object--support--pole fires on every wire pole and fence post, so a pole that
# nothing else confirms has to be seen from many panoramas before it is placed.
PHOTO_MIN_VIEWS = {'lamp_post': 8, 'sign_post': 16, 'bench': 6, 'trash_can': 6,
                   'fire_hydrant': 6, 'bike_rack': 4}
DEFAULT_H = {'lamp_post': 5.0, 'sign_post': 2.5, 'mast': 10.0, 'flagpole': 8.0}
WATER_OK = ('small_boat', 'mooring_post')


def path_bearings():
    """Every footway/cycleway/pedestrian segment in ENU, for bench alignment."""
    els = json.load(open(C.W + '/osm/osm_raw.json'))['elements']
    A, B = [], []
    for el in els:
        if el['type'] != 'way':
            continue
        t = el.get('tags') or {}
        if t.get('highway') not in ('footway', 'path', 'pedestrian', 'steps',
                                    'cycleway', 'living_street', 'service'):
            continue
        g = el.get('geometry') or []
        pts = [enu_of(p['lat'], p['lon']) for p in g]
        for i in range(len(pts) - 1):
            A.append(pts[i]); B.append(pts[i + 1])
    return np.array(A, 'f8'), np.array(B, 'f8')


def nearest_path_yaw(A, B, x, y):
    """-> (yaw of the nearest path segment in degrees, distance m)."""
    if not len(A):
        return None, 9e9
    ab = B - A
    L2 = (ab ** 2).sum(1)
    L2[L2 < 1e-9] = 1e-9
    t = np.clip(((np.array([x, y]) - A) * ab).sum(1) / L2, 0, 1)
    p = A + t[:, None] * ab
    d = np.hypot(p[:, 0] - x, p[:, 1] - y)
    k = int(np.argmin(d))
    return float(math.degrees(math.atan2(ab[k, 1], ab[k, 0])) % 360.0), float(d[k])


# -- A TALL POLE AT A STREET IS A STREET POLE, NOT A FLAGPOLE.
# fuse() calls every lidar pole over 8 m with no OSM match a 'mast', and
# CLASSES draws 'mast' as the pirate-kit flagpole (brown pole, grey flag).
# Distance from a lidar mast to the nearest OSM street carriageway EDGE
# (centreline minus lanes x LANE_M / 2; primary..residential, not service/parking
# aisles) and to the nearest tram track (railway=tram, electrified=contact_line)
# separates street lights from other masts: the street-edge and tram distances
# of street lights fall under ~5-6 m, with a gap of several metres before other
# masts, so STREET_EDGE_M = 5 and TRAM_M = 6 sit in that gap.  Masts at a street
# kerb or a tram line flip; masts away from the street (on footways or parking
# aisles) stay.  A flipped pole takes the kit street light (lamp_post), the
# model every lamp_post row uses, at its own lidar height.  Applied AFTER
# the ids are assigned, so no prop id changes (the id is the sort position).
STREET_EDGE_M = 5.0
TRAM_M = 6.0
LANE_M = 3.2
STREET_HW = ('primary', 'secondary', 'tertiary', 'unclassified', 'residential')
STREET_LANES_DEFAULT = 2


def street_lines():
    """-> (street segs A, B, half-width per seg), (tram segs A, B)."""
    els = json.load(open(C.W + '/osm/osm_raw.json'))['elements']
    SA, SB, SH, TA, TB = [], [], [], [], []
    for el in els:
        if el['type'] != 'way':
            continue
        t = el.get('tags') or {}
        pts = [enu_of(p['lat'], p['lon']) for p in (el.get('geometry') or [])]
        if t.get('railway') == 'tram':
            TA += pts[:-1]; TB += pts[1:]
        elif t.get('highway') in STREET_HW:
            try:
                ln = int(str(t.get('lanes', STREET_LANES_DEFAULT)).split(';')[0])
            except ValueError:
                ln = STREET_LANES_DEFAULT
            SA += pts[:-1]; SB += pts[1:]; SH += [ln * LANE_M / 2.0] * (len(pts) - 1)
    f = lambda L: np.array(L, 'f8').reshape(-1, 2)
    return (f(SA), f(SB), np.array(SH, 'f8')), (f(TA), f(TB))


def _seg_dist(A, B, x, y):
    if not len(A):
        return np.array([9e9])
    ab = B - A
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    t = np.clip(((np.array([x, y]) - A) * ab).sum(1) / L2, 0, 1)
    p = A + t[:, None] * ab
    return np.hypot(p[:, 0] - x, p[:, 1] - y)


def street_masts(props):
    """Recast masts at a street edge or a tram track as lamp_post.
    -> list of (id, flipped, street_edge_m, tram_m) for every mast."""
    (SA, SB, SH), (TA, TB) = street_lines()
    out = []
    for p in props:
        if p['cls'] != 'mast':
            continue
        de = float((_seg_dist(SA, SB, p['x'], p['y']) - SH).min()) if len(SA) else 9e9
        dt = float(_seg_dist(TA, TB, p['x'], p['y']).min())
        flip = de <= STREET_EDGE_M or dt <= TRAM_M
        out.append((p['id'], flip, round(de, 1), round(dt, 1)))
        if flip:
            p['recast_from'] = dict(
                cls='mast', asset=p['asset'],
                rule='lidar pole %.1f m from a street edge, %.1f m from the '
                     'nearest tram track: street pole (<= %.0f / %.0f m)'
                     % (de, dt, STREET_EDGE_M, TRAM_M))
            p['cls'] = 'lamp_post'
            p['asset'] = CLASSES['lamp_post']['asset']
    return out


# -- OFF-STREET MASTS LEFT BY street_masts().  An off-street mast is one of
# three things:
#   a tall street light in open ground (curved or straight arm): the lidar holds
#     ground returns and a small head at 8-9 m with nothing in between (a thin
#     pole, a lamp head), and a photo street-light cluster of many views lies
#     within OFFSTREET_LAMP_M;
#   no pole in any panorama: the point is a portico, wall or eave right at a
#     building footprint (the lidar column is wall or eave returns);
#   anything else (e.g. a pole only a few views call a street light) stays a
#     mast.
# The building distance and the view count both separate the groups with a
# clear gap: OFFSTREET_BUILDING_M = 1.0, and OFFSTREET_LAMP_VIEWS = 8 (the
# lamp_post photo gate, PHOTO_MIN_VIEWS).  Applied after street_masts(), after
# the ids are assigned, so no other prop id changes; a retired mast goes to
# 'dropped'.
OFFSTREET_BUILDING_M = 1.0
OFFSTREET_LAMP_M = 1.5
OFFSTREET_LAMP_VIEWS = 8


def offstreet_masts(props, photo, surf):
    """Recast masts that photos show as street lights; retire masts that are
    building-wall returns.  -> list of (id, action, lamp_views, at_building)."""
    lamps = [o for o in photo if o['cls'] == 'lamp_post']
    out = []
    for p in props:
        if p['cls'] != 'mast':
            continue
        at_bld = surf.in_building(p['x'], p['y'], OFFSTREET_BUILDING_M)
        lv = max([o['n_views'] for o in lamps
                  if math.hypot(o['x'] - p['x'], o['y'] - p['y']) <= OFFSTREET_LAMP_M],
                 default=0)
        if at_bld:
            act = 'retire'
            p['drop'] = ('lidar mast within %.1f m of a building footprint: wall '
                         'or eave returns, no pole in the panoramas' % OFFSTREET_BUILDING_M)
        elif lv >= OFFSTREET_LAMP_VIEWS:
            act = 'lamp_post'
            p['recast_from'] = dict(
                cls='mast', asset=p['asset'],
                rule='photo street-light cluster of %d views within %.1f m '
                     '(>= %d): off-street street light' % (lv, OFFSTREET_LAMP_M, OFFSTREET_LAMP_VIEWS))
            p['cls'] = 'lamp_post'
            p['asset'] = CLASSES['lamp_post']['asset']
        else:
            act = 'stays mast'
        out.append((p['id'], act, lv, at_bld))
    return out


def fuse(surf):
    osm = json.load(open(SCR + '/osm_props.json'))
    poles = json.load(open(SCR + '/lidar_poles.json'))
    photo = json.load(open(SCR + '/photo_objects.json'))['objects']
    props = []
    for o in osm:
        p = dict(o)
        p.setdefault('sources', ['osm'])
        p['evidence'] = {'osm': 1}
        props.append(p)

    def match(x, y, classes, maxd):
        best, bd = None, maxd
        for p in props:
            if p['cls'] not in classes:
                continue
            d = math.hypot(p['x'] - x, p['y'] - y)
            if d < bd:
                best, bd = p, d
        return best, bd

    # city asset inventories (pipeline/city_props.py), only with an area set: a
    # city row within MATCH_M of an OSM row of its class is evidence on it,
    # otherwise it is a prop of its own.
    n_c_match = n_c_new = 0
    cp = SCR + '/city_props.json'
    if _AP and os.path.exists(cp):
        for o in json.load(open(cp))['rows']:
            m, d = match(o['x'], o['y'], (o['cls'],), MATCH_M.get(o['cls'], 3.0))
            if m is not None:
                m['sources'].append('city'); m['evidence']['city'] = 1
                for k in ('height_m', 'length_m'):
                    if m.get(k) is None and o.get(k) is not None:
                        m[k] = o[k]
                n_c_match += 1
            else:
                p = dict(o); p['sources'] = ['city']; p['evidence'] = {'city': 1}
                props.append(p); n_c_new += 1
        print(f'city rows: {n_c_match} matched an OSM prop, {n_c_new} new', flush=True)

    # lidar poles
    n_l_match = 0
    for o in poles:
        h = o['height_m']
        cls = 'mast' if h > 8.0 else ('lamp_post' if h >= 3.5 else 'sign_post')
        m, d = match(o['x'], o['y'], POLE_CLASSES, MATCH_M.get(cls, 4.0))
        if m is not None:
            m['sources'].append('lidar')
            m['evidence']['lidar'] = 1
            m['lidar_height_m'] = h
            m['lidar_offset_m'] = round(d, 2)
            n_l_match += 1
            continue
        props.append(dict(cls=cls, source='lidar', sources=['lidar'],
                          evidence={'lidar': 1}, x=o['x'], y=o['y'],
                          confidence=round(min(0.9, 0.4 + 0.05 * o['points']), 2),
                          lidar_height_m=h, height_m=h,
                          lidar_points=o['points'], osm_id=None, osm_type=None,
                          osm_tag=None))
    # photo objects
    n_p_match = 0
    for o in photo:
        cls = o['cls']
        fam = POLE_CLASSES if cls in POLE_CLASSES else (cls,)
        m, d = match(o['x'], o['y'], fam, MATCH_M.get(cls, 3.0))
        if m is not None:
            m['sources'].append('photo')
            m['evidence']['photo'] = o['n_views']
            m['photo_offset_m'] = round(d, 2)
            n_p_match += 1
            continue
        props.append(dict(cls=cls, source='photo', sources=['photo'],
                          evidence={'photo': o['n_views']}, x=o['x'], y=o['y'],
                          confidence=round(min(0.9, 0.3 + 0.1 * o['n_views']), 2),
                          photo_views=o['n_views'], photo_spread_m=o['spread_m'],
                          height_m=DEFAULT_H.get(cls), osm_id=None, osm_type=None,
                          osm_tag=None))
    # evidence gates: drop single-source objects that are not strong enough
    gated = []
    for p in props:
        srcs = set(p['sources'])
        if srcs == {'lidar'} and p.get('lidar_points', 0) < LIDAR_MIN_POINTS_ALONE:
            p['drop'] = ('lidar only, %d returns (needs %d, or a photo match)'
                         % (p.get('lidar_points', 0), LIDAR_MIN_POINTS_ALONE))
            gated.append(p); continue
        if srcs == {'photo'}:
            need = PHOTO_MIN_VIEWS.get(p['cls'], 3)
            if p.get('photo_views', 0) < need:
                p['drop'] = ('photo only, %d views (needs %d, or an OSM/lidar match)'
                             % (p.get('photo_views', 0), need))
                gated.append(p); continue
        gated.append(None)
    props = [p for p, g in zip(props, gated) if 'drop' not in p]
    gated = [p for p in gated if p is not None and 'drop' in p]
    # place, orient, check
    A, B = path_bearings()
    kept, dropped = [], list(gated)
    for p in props:
        cfg = CLASSES[p['cls']]
        r = cfg['radius']
        inb = surf.in_building(p['x'], p['y'], r=0.0)
        if inb:
            # a ship mapped as an OSM building is already massed by 3dfier, so a
            # boat prop on top of one would be a second hull in the same water
            p['drop'] = ('ship already modelled as a building mass'
                         if p['cls'] == 'small_boat' else 'inside a building mass')
            dropped.append(p); continue
        z, surfname = surf.at(p['x'], p['y'], r=min(r, 1.0))
        if not np.isfinite(z):
            p['drop'] = 'off the model floor'
            dropped.append(p); continue
        if surfname == 'water' and p['cls'] not in WATER_OK:
            p['drop'] = 'stands in water'
            dropped.append(p); continue
        p['z'] = round(float(z), 3)
        p['surface'] = surfname
        if 'yaw_deg' not in p or p['yaw_deg'] is None:
            if cfg['align'] == 'path':
                yaw, d = nearest_path_yaw(A, B, p['x'], p['y'])
                if yaw is not None and d < 15.0:
                    # a bench sits along the path and faces it
                    p['yaw_deg'] = round((yaw + 90.0) % 360.0, 2)
                    p['yaw_from'] = 'nearest path segment %.1f m away' % d
                else:
                    p['yaw_deg'] = round(hash_yaw(p.get('osm_id') or
                                                  (p['x'], p['y'])), 2)
                    p['yaw_from'] = 'deterministic hash (no path within 15 m)'
            else:
                p['yaw_deg'] = round(hash_yaw(p.get('osm_id') or (p['x'], p['y'])), 2)
                p['yaw_from'] = 'deterministic hash'
        else:
            p['yaw_deg'] = round(float(p['yaw_deg']), 2)
            p.setdefault('yaw_from', 'osm way direction')
        if p.get('height_m') is None:
            p['height_m'] = DEFAULT_H.get(p['cls'])
        kept.append(p)
    kept.sort(key=lambda p: (p['cls'], round(p['x'], 3), round(p['y'], 3)))
    for i, p in enumerate(kept):
        p['id'] = 'prop_%04d' % (i + 1)
        p['asset'] = CLASSES[p['cls']]['asset']
        p['sources'] = sorted(set(p['sources']))
    sm = street_masts(kept)
    for i_, f_, de_, dt_ in sm:
        print('mast %s %s street_edge %.1f m tram %.1f m'
              % (i_, 'FLIP->lamp_post' if f_ else 'stays mast', de_, dt_), flush=True)
    pm = offstreet_masts(kept, photo, surf)
    for i_, a_, lv_, b_ in pm:
        print('off-street mast %s %s street_light_views %d at_building %s'
              % (i_, a_, lv_, b_), flush=True)
    dropped += [p for p in kept if 'drop' in p]
    kept = [p for p in kept if 'drop' not in p]
    counts = collections.Counter(p['cls'] for p in kept)
    per_source = collections.Counter(p['source'] for p in kept)
    multi = collections.Counter('+'.join(p['sources']) for p in kept)
    doc = dict(
        props=kept,
        counts=dict(counts), per_source=dict(per_source), by_source_set=dict(multi),
        dropped=[dict(cls=p['cls'], why=p['drop'], x=round(p['x'], 1),
                      y=round(p['y'], 1), source=p['source']) for p in dropped],
        dropped_counts=dict(collections.Counter('%s: %s' % (p['cls'], p['drop'])
                                                for p in dropped)),
        matched=dict(lidar_poles_matched_to_osm=n_l_match,
                     lidar_poles_total=len(poles),
                     photo_objects_matched_to_existing=n_p_match,
                     photo_objects_total=len(photo)),
        gates=dict(lidar_min_points_alone=LIDAR_MIN_POINTS_ALONE,
                   photo_min_views=PHOTO_MIN_VIEWS,
                   note='a single-source object must clear its gate; an object '
                        'confirmed by a second source is kept whatever its count'),
        rules=dict(
            photo=('rays through the base of each detection polygon from panoramas '
                   'with a high or usable pose; two rays from two panoramas >= %.1f m '
                   'apart agree when their closest approach is <= %.1f m and both ray '
                   'lengths are 0.5-%.0f m; agreeing midpoints are single-link '
                   'clustered at %.1f m and a cluster needs >= 2 distinct panoramas; '
                   'the two rays must also differ in direction by >= %.0f deg and the '
                   'point must sit inside a per-class height window above the model '
                   'floor (%s)'
                   % (PHOTO_MIN_BASE_M, PHOTO_GAP_M, PHOTO_MAX_RANGE_M, PHOTO_GAP_M,
                      PHOTO_MIN_ANGLE_DEG, json.dumps(PHOTO_DZ_WINDOW))),
            lidar=('non-ground 3DEP returns %.1f-%.1f m above the model floor, gridded '
                   'at %.2f m; a cell is pole-like when it holds >= %d returns, its '
                   '3x3 core holds >= %d and the 7x7 ring around that core holds '
                   '<= %.0f%% of the core; clusters over 3 m2 and clusters within '
                   '2.5 m of an inventoried tree are rejected'
                   % (POLE_MIN_H, POLE_MAX_H, CELL, POLE_MIN_PTS, POLE_MIN_PTS,
                      POLE_ISO * 100)),
            match=('a lidar pole or photo object is the same object as an existing '
                   'prop when it is within %s m of one of the same class family'
                   % json.dumps(MATCH_M))),
        frame='photo-build ENU, x=East y=North z=Up, metres; s82 writes (x, z, -y)',
        street_masts=dict(rule='mast -> lamp_post when <= %.0f m from a street '
                               'edge or <= %.0f m from a tram track'
                               % (STREET_EDGE_M, TRAM_M),
                          rows=[dict(id=i_, flipped=f_, street_edge_m=de_, tram_m=dt_)
                                for i_, f_, de_, dt_ in sm]),
        offstreet_masts=dict(rule='mast retired when <= %.1f m from a building '
                             'footprint; mast -> lamp_post when a photo '
                             'street-light cluster of >= %d views is <= %.1f m away'
                             % (OFFSTREET_BUILDING_M, OFFSTREET_LAMP_VIEWS, OFFSTREET_LAMP_M),
                             rows=[dict(id=i_, action=a_, street_light_views=lv_,
                                        at_building=b_) for i_, a_, lv_, b_ in pm]),
        generated=time.strftime('%Y-%m-%dT%H:%M:%S'))
    json.dump(doc, open(SCR + '/prop_inventory.json', 'w'), indent=1)
    json.dump(doc, open(C.W + '/prop_inventory.json', 'w'), indent=1)
    cols = ['id', 'cls', 'asset', 'source', 'sources', 'x', 'y', 'z', 'surface',
            'yaw_deg', 'height_m', 'confidence', 'osm_type', 'osm_id', 'osm_tag']
    with open(C.W + '/prop_inventory.csv', 'w') as f:
        f.write(','.join(cols) + '\n')
        for p in kept:
            f.write(','.join('"%s"' % ('+'.join(p[c]) if c == 'sources'
                                       else ('' if p.get(c) is None else p.get(c)))
                             for c in cols) + '\n')
    print('props', len(kept), dict(counts), flush=True)
    print('per source', dict(per_source), 'source sets', dict(multi), flush=True)
    print('dropped', len(dropped), doc['dropped_counts'], flush=True)
    print('matched', doc['matched'], flush=True)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['surface', 'osm', 'lidar', 'photo', 'fuse', 'all'])
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    if a.mode in ('surface', 'osm', 'lidar', 'photo', 'fuse', 'all'):
        surf = Surface(rebuild=a.rebuild) if a.mode != 'osm' else None
    if a.mode in ('osm', 'all'):
        osm_pass()
    if a.mode in ('lidar', 'all'):
        lidar_pass(surf)
    if a.mode in ('photo', 'all'):
        photo_pass(surf, limit=a.limit)
    if a.mode in ('fuse', 'all'):
        fuse(surf)


if __name__ == '__main__':
    main()
