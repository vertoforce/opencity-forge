"""Stage s10a: build a planar partition of the OSM layers and run 3dfier on it.

3dfier (tudelft3d) is run as an external CLI in Docker, pinned by tag+digest --
no host install.  It lifts each polygon class by its own percentile of the lidar
returns and then *stitches* the classes together, so a building wall, a pier
deck and the lawn meet exactly at their shared edge instead of interpenetrating
a raster grid.

Its watertightness is conditional on the input being a planar partition.  Fed
overlapping OSM polygons, it returns a mesh full of open edges, all of them
boundaries of isolated grass patches floating in a void.  This stage fixes that upstream: every OSM polygon is
clipped against the ones above it in a fixed priority order

    buildings > bridges/piers > water > paths > landuse > remainder terrain

and the leftover of the model extent becomes one remainder-terrain polygon, so
the union of the inputs is exactly the extent with no overlaps and no holes.

Both versions are actually run and the open edges of each output counted, so the
before/after is measured rather than asserted.

Outputs consumed downstream:
  layers/buildings.npz     LoD1 massing, walls noded down to the terrain
  layers/water.npz         forced-flat water surface
  layers/ground_terrain.npz the lidar floor under LAYER_ground
  layers/structures.npz    bridge / pier decks
  geom/backstop.npz        the floor as a 1 m raster, for path re-draping

Resumable: markers/3dfier_{partition,run,layers}.done plus markers/3dfier.done.
"""
import sys, os, json, math, time, subprocess, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
from scipy import ndimage
import shapely
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union
import common as C
import memguard as MG
import s2x_lib as L
import osm_rules as R

# 3dfier pinned by tag + digest (image label org.opencontainers.image.version=v1.5.0).
IMAGE = ("tudelft3d/3dfier:v1.5.0@sha256:"
         "95a8e1defe7d8f2e105758bc5ed96e7964f97bda6fb07e577c8ff67cf9dd8264")

WORK = C.W + '/tmp3d'
# Model extent (ENU metres): the area's own bbox (common.BBOX_WSEN), whole metres
# outward.  With no area set the bbox is empty, and main() refuses to run.
_HAS_AREA = bool(os.environ.get('OPENCITY_AREA', '').strip()
                 and ((C._AREA_CFG or {}).get('paths') or {}).get('scratch_dir'))
_w, _s, _e, _n = C.BBOX_WSEN
_P = C.lla_to_enu([_s, _s, _n, _n], [_w, _e, _e, _w], [C.ORIGIN[2]] * 4)
EXT = (float(math.floor(_P[:, 0].min())), float(math.floor(_P[:, 1].min())),
       float(math.ceil(_P[:, 0].max())), float(math.ceil(_P[:, 1].max())))
CELL = 1.0                      # raster used for the z-fight clamp and backstop
MIN_PART_AREA = 0.25            # slivers below this fall through to the remainder
SUBDIV_EDGE_M = 4.0             # terrain triangles are split below this edge length
SUBDIV_MAX_DEPTH = 3
PHOTO_CLEAR = 0.05              # the floor sits this far under the photo ground

COL = dict(lawn=(104, 132, 72), paving=(176, 172, 164), sand=(198, 186, 156),
           gravel=(168, 160, 146), wood=(150, 140, 125), bare=(126, 122, 112),
           road=(96, 96, 98), water=(44, 74, 96), scrub=(96, 116, 74))


# --------------------------------------------------------------------------- #
# stage bookkeeping
# --------------------------------------------------------------------------- #
def sub_done(n):
    return os.path.exists(C.W + f'/markers/3dfier_{n}.done')


def sub_mark(n, info):
    json.dump(info, open(C.W + f'/markers/3dfier_{n}.done', 'w'), indent=1)


# --------------------------------------------------------------------------- #
# 1. the planar partition
# --------------------------------------------------------------------------- #
def _explode(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g] if g.area > 0 else []
    return [p for p in getattr(g, 'geoms', []) if isinstance(p, Polygon) and p.area > 0]


def _water_mask_polygon():
    """The measured water mask (lidar class 9 + OSM) as a polygon.

    The published water layer is this mask, not the OSM polygons, and it is
    larger than them; feeding 3dfier only the OSM water would leave part of the
    water surface as terrain.
    """
    S = np.load(C.FW + '/surfaces.npz')
    x0, y0, c, nx, ny = [float(v) for v in S['grid']]
    m = ndimage.binary_closing(S['m_water'], np.ones((3, 3)))
    fy, fx = np.nonzero(m)
    boxes = [box(x * c + x0, y * c + y0, (x + 1) * c + x0, (y + 1) * c + y0)
             for y, x in zip(fy, fx)]
    g = unary_union(boxes)
    return shapely.simplify(g, 0.4), int(m.sum()), float(m.sum() * c * c)


def collect_layers():
    """OSM -> shapely polygons per 3dfier class, before any clipping."""
    els = C.osm_elements()
    ways = [e for e in els if e['type'] == 'way']
    rels = [e for e in els if e['type'] == 'relation']
    out = {k: [] for k in R.PRIORITY}
    tally = collections.Counter()

    for w in sorted(ways, key=lambda w: w['id']):
        t = w.get('tags', {}) or {}
        d = R.classify(t, closed=(w['nodes'][0] == w['nodes'][-1]
                                  and len(w.get('geometry', [])) >= 4))
        if d is None:
            continue
        lay = d['layer']
        if d['geom'] == 'area':
            poly = L.way_polygon(w, 0.0, do_clip=False)
        else:
            ln = L.way_line(w)
            poly = None if ln is None else L.keep_polygons(
                ln.buffer(d['width_m'] / 2.0, cap_style=d.get('cap', 'round'),
                          join_style='round', quad_segs=2))
        poly = L.clip(poly) if poly is not None else None
        if poly is None or poly.area < (R.MIN_BUILDING_AREA_M2
                                        if lay == 'building' else 0.5):
            continue
        out[lay].append(dict(poly=poly, osm_id=int(w['id']), osm_type='way',
                             name=t.get('name'), **d))
        tally[lay] += 1

    for r in sorted(rels, key=lambda r: r['id']):
        t = r.get('tags', {}) or {}
        if t.get('type') != 'multipolygon':
            continue
        d = R.classify(t, closed=True)
        if d is None or d['geom'] != 'area':
            continue
        poly = L.rel_polygon(r)
        if poly is None or poly.area < 1.0:
            continue
        out[d['layer']].append(dict(poly=poly, osm_id=int(r['id']),
                                    osm_type='relation', name=t.get('name'), **d))
        tally[d['layer']] += 1

    # the measured water surface, which the OSM polygons under-cover
    wm, wcells, warea = _water_mask_polygon()
    wclip = L.clip(wm) if wm is not None else None
    # a dry box (no class-9 returns) has no measured water; L.clip() then returns
    # None and the water layer is skipped.
    if wclip is None or wclip.is_empty:
        return out, dict(tally), dict(water_mask_cells=wcells, water_mask_area_m2=warea)
    out['water'].append(dict(poly=wclip, osm_id=None, osm_type='measured',
                             name='measured_water_mask', layer='water',
                             kind='water_measured', geom='area', surface=None,
                             width_m=None, rule='measured lidar class-9 mask'))
    tally['water'] += 1
    return out, dict(tally), dict(water_mask_cells=wcells, water_mask_area_m2=warea)


# The partition's smallest-claims-first rule (see partition()) ignores height.  Where an OSM building:part
# sits ON a lower part (its min_height is the lower part's height), the smaller
# of the two takes the shared area and the other loses it.  For a tower part
# whose min_height equals its podium's height, smallest-claims-first gives most
# of the tower's footprint to the podium and the upper storey over the podium
# disappears.  A planar partition holds one column per
# cell, and 3dfier extrudes every column from the ground, so the right owner of
# a stacked cell is the UPPER part: its column from the ground to its roof is
# the lower part's storey plus its own.  Rule: before the claim loop, each
# building:part is cut by the footprints of the parts that sit on it (upper
# base within STACK_GAP_M of its top, and above its own base); the parts are
# the pre-partition OSM volumes s10a_layers' support test uses.  A part that
# floats over a lower one with a real gap (a bridge over a shed) is not "on" it
# and does not cut it.  s10a_layers then sees the upper piece standing over its
# support and drops its base to the floor.
STACK_GAP_M = 1.0


def stack_clip(items):
    """Cut every building:part item by the parts stacked directly on it.
    Returns (items, stats); items that vanish are removed."""
    import s10a_layers as SL
    vol = {v['id']: v for v in SL._osm_volumes() if v['part']}

    def v_of(it):
        return vol.get(it.get('osm_id')) if it.get('osm_type') == 'way' else None
    out = []; cuts = []
    for lo in items:
        vl = v_of(lo)
        if vl is None or vl['top'] <= vl['base'] + STACK_GAP_M:
            out.append(lo); continue
        ups = []
        for u in items:
            vu = v_of(u)
            if (u is lo or vu is None or abs(vu['base'] - vl['top']) > STACK_GAP_M
                    or vu['base'] <= vl['base'] + STACK_GAP_M
                    or not u['poly'].intersects(lo['poly'])):
                continue
            ups.append((vu['id'], u['poly']))
        if not ups:
            out.append(lo); continue
        a0 = lo['poly'].area
        p = shapely.make_valid(lo['poly'].difference(unary_union([g for _, g in ups])))
        cuts.append(dict(lower=vl['id'], by=sorted(i for i, _ in ups),
                         area_m2=round(a0, 1), cut_m2=round(a0 - p.area, 1)))
        if p.is_empty or p.area < MIN_PART_AREA:
            continue
        out.append(dict(lo, poly=p))
    return out, dict(cuts=cuts, parts_cut=len(cuts),
                     cut_m2=round(sum(c['cut_m2'] for c in cuts), 1))


def partition(raw):
    """Clip every polygon against every polygon already accepted.

    Priority runs across layers in R.PRIORITY order and, inside a layer,
    smallest-area-first: arnis re-sorts its ground-filling areas by DESCENDING
    ring area so the smallest, most specific polygon is the LAST writer
    (data_processing.rs:150-178).  Last-writer-wins on a raster is
    smallest-claims-first on a partition, and it is what puts a building:part
    ahead of its parent building and a lawn patch ahead of the larger green polygon it sits in.
    """
    extent = box(*EXT)
    feats = []
    acc = []                       # accepted geometries, in claim order
    bnds = np.zeros((0, 4))
    stats = {}
    raw = dict(raw)
    raw['building'], stats['stack_clip'] = stack_clip(raw['building'])
    for lay in R.PRIORITY:
        kept = 0
        area_in = 0.0
        area_out = 0.0
        items = sorted(raw[lay], key=lambda it: (it['poly'].area, it.get('osm_id') or 0))
        for it in items:
            p = it['poly']
            area_in += p.area
            p = p.intersection(extent)
            if p.is_empty:
                continue
            if len(acc):
                x0, y0, x1, y1 = p.bounds
                hit = np.nonzero(~((bnds[:, 2] < x0) | (bnds[:, 0] > x1) |
                                   (bnds[:, 3] < y0) | (bnds[:, 1] > y1)))[0]
                if len(hit):
                    u = unary_union([acc[i] for i in hit])
                    try:
                        p = p.difference(u)
                    except Exception:
                        p = shapely.make_valid(p).difference(u)
            new = []
            for q in _explode(shapely.make_valid(p)):
                if q.area < MIN_PART_AREA:
                    continue
                q = q.buffer(0)
                for qq in _explode(q):
                    if qq.area < MIN_PART_AREA:
                        continue
                    feats.append(dict(it, poly=qq))
                    new.append(qq)
                    kept += 1
                    area_out += qq.area
            if new:
                acc += new
                bnds = np.vstack([bnds, np.array([g.bounds for g in new])])
        stats[lay] = dict(features_in=len(raw[lay]), parts_out=kept,
                          area_in_m2=round(area_in, 1), area_kept_m2=round(area_out, 1))
    used = unary_union(acc) if acc else None
    rem = extent if used is None else extent.difference(used)
    nrem = 0
    rem_area = 0.0
    for q in _explode(shapely.make_valid(rem)):
        if q.area < MIN_PART_AREA:
            continue
        feats.append(dict(poly=q, layer='terrain', kind='remainder', geom='area',
                          surface='bare', width_m=None, osm_id=None,
                          osm_type='remainder', name=None,
                          rule='model extent minus every other layer'))
        nrem += 1
        rem_area += q.area
    stats['remainder'] = dict(parts_out=nrem, area_m2=round(rem_area, 1))
    stats['extent_area_m2'] = round(extent.area, 1)
    stats['clipped_area_m2'] = round(sum(f['poly'].area for f in feats), 1)
    feats, ns = node_and_polygonize(feats, extent)
    stats['noding'] = ns
    stats['covered_area_m2'] = round(sum(f['poly'].area for f in feats), 1)
    return feats, stats


def node_and_polygonize(feats, extent, grid=0.001):
    """Turn the clipped set into a *noded* planar partition.

    Clipping alone still leaves T-junctions: where three polygons meet, one of
    them carries a vertex the other two do not, and 3dfier triangulates each
    polygon on its own, so the shared edge comes back split on one side and
    whole on the other -- an open edge on both.  Unioning every boundary first
    inserts a node at every crossing, and re-polygonising the arrangement gives
    faces that agree vertex-for-vertex along every shared edge.
    """
    from shapely.ops import polygonize
    src = [shapely.set_precision(f['poly'], grid) for f in feats]
    keep = [(f, p) for f, p in zip(feats, src)
            if p is not None and not p.is_empty and p.area >= MIN_PART_AREA]
    lines = unary_union([p.boundary for _, p in keep] + [extent.boundary])
    faces = [f for f in polygonize(lines) if f.area >= MIN_PART_AREA]
    bnds = np.array([p.bounds for _, p in keep]) if keep else np.zeros((0, 4))
    out = []
    orphan = 0
    orphan_area = 0.0
    for fc in faces:
        rp = fc.representative_point()
        x, y = rp.x, rp.y
        hit = np.nonzero((bnds[:, 0] <= x) & (bnds[:, 2] >= x) &
                         (bnds[:, 1] <= y) & (bnds[:, 3] >= y))[0]
        owner = None
        for i in hit:
            if keep[i][1].contains(rp):
                owner = keep[i][0]
                break
        if owner is None:
            orphan += 1
            orphan_area += fc.area
            owner = dict(layer='terrain', kind='remainder', geom='area',
                         surface='bare', width_m=None, osm_id=None,
                         osm_type='remainder', name=None,
                         rule='noding leftover, folded into the remainder terrain')
        out.append(dict(owner, poly=fc))
    return out, dict(faces_in=len(keep), faces_out=len(out),
                     unowned_faces=orphan, unowned_area_m2=round(orphan_area, 1),
                     precision_grid_m=grid)


# --------------------------------------------------------------------------- #
# 2. writing 3dfier's inputs
# --------------------------------------------------------------------------- #
LIFT = {'building': 'Building', 'bridge': 'Bridge/Overpass', 'water': 'Water',
        'road': 'Road', 'terrain': 'Terrain'}
FILE = {'building': 'building', 'bridge': 'bridge', 'water': 'water',
        'road': 'road', 'terrain': 'terrain'}


def write_geojson(feats, d):
    os.makedirs(d, exist_ok=True)
    per = collections.defaultdict(list)
    for k, f in enumerate(feats):
        fid = f"{f['layer']}_{k}"
        f['fid'] = fid
        per[f['layer']].append({
            "type": "Feature",
            "properties": {"fid": fid, "kind": f.get('kind'),
                           "osm_id": f.get('osm_id'),
                           "surface": f.get('surface')},
            "geometry": mapping(f['poly'])})
    json.dump({f['fid']: {k: v for k, v in f.items() if k != 'poly'}
               for f in feats}, open(f"{d}/fids.json", 'w'), indent=0)
    counts = {}
    for lay, fs in per.items():
        json.dump({"type": "FeatureCollection", "features": fs},
                  open(f"{d}/{FILE[lay]}.geojson", 'w'))
        counts[lay] = len(fs)
    for lay in LIFT:
        p = f"{d}/{FILE[lay]}.geojson"
        if not os.path.exists(p):
            json.dump({"type": "FeatureCollection", "features": []}, open(p, 'w'))
            counts[lay] = 0
    return counts


def write_las(path, pad=True):
    """Registered lidar as LAS in the model's own ENU metres.

    3dfier never reprojects, so handing it the model's ENU frame directly means its
    output needs no transform on the way back.
    """
    import laspy
    P = np.load(C.FW + '/lidar_reg.npy').astype('f8')
    cls = np.load(C.FW + '/loaded.npz')['lidar_cls'].astype('u1')
    npad = 0
    if pad:
        # Polygon *vertices* need a return within radius_vertex_elevation; the
        # model rectangle can extend past the lidar footprint, so the outer
        # ring of the remainder terrain may have no returns at all.  Fill only
        # where no class-2 return exists within 3 m, using the same
        # nearest-measured-cell value the raster floor already used.
        g = (cls == 2) | (cls == 8)
        Zm, meas, gx0, gy0, nx, ny = _ground_raster(P[g])
        far = ndimage.distance_transform_edt(~meas) * CELL > 3.0
        idx = ndimage.distance_transform_edt(~meas, return_distances=False,
                                             return_indices=True)
        Zf = Zm[tuple(idx)]
        sel = far & np.isfinite(Zf)
        sel[1::2, :] = False        # 2 m spacing
        sel[:, 1::2] = False
        py, px = np.nonzero(sel)
        pts = np.stack([px * CELL + gx0 + CELL / 2, py * CELL + gy0 + CELL / 2,
                        Zf[py, px]], 1)
        npad = len(pts)
        P = np.vstack([P, pts])
        cls = np.concatenate([cls, np.full(npad, 2, 'u1')])
    h = laspy.LasHeader(point_format=1, version="1.2")
    h.scales = [0.001, 0.001, 0.001]
    h.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(h)
    las.x = P[:, 0]; las.y = P[:, 1]; las.z = P[:, 2]
    las.classification = cls
    las.write(path)
    return len(P), npad


def _ground_raster(G):
    gx0, gy0, x1, y1 = EXT
    nx = int(round((x1 - gx0) / CELL)); ny = int(round((y1 - gy0) / CELL))
    ix = np.clip(((G[:, 0] - gx0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((G[:, 1] - gy0) / CELL).astype(int), 0, ny - 1)
    lin = iy * nx + ix
    o = np.argsort(lin); ls = lin[o]; zs = G[o, 2]
    u, s = np.unique(ls, return_index=True); e = np.append(s[1:], len(ls))
    z = np.full(nx * ny, np.nan, 'f8')
    for uu, a, b in zip(u, s, e):
        z[uu] = np.median(zs[a:b])
    Z = z.reshape(ny, nx)
    return Z, ~np.isnan(Z), gx0, gy0, nx, ny


CONFIG = """input_polygons:
  - datasets:
      - /data/building.geojson
    uniqueid: fid
    lifting: Building
  - datasets:
      - /data/bridge.geojson
    uniqueid: fid
    lifting: Bridge/Overpass
  - datasets:
      - /data/water.geojson
    uniqueid: fid
    lifting: Water
  - datasets:
      - /data/road.geojson
    uniqueid: fid
    lifting: Road
  - datasets:
      - /data/terrain.geojson
    uniqueid: fid
    lifting: Terrain

lifting_options:
  Building:
    roof:
      height: percentile-90
    ground:
      height: percentile-10
      use_LAS_classes:
        - 2
    lod: 1
    floor: true
    inner_walls: true
    triangulate: true
  Water:
    height: percentile-10
    use_LAS_classes:
      - 2
      - 9
  Road:
    height: percentile-50
    filter_outliers: true
    flatten: true
    use_LAS_classes:
      - 2
  Bridge/Overpass:
    height: percentile-50
    flatten: true
  Terrain:
    simplification_tinsimp: 0.1
    use_LAS_classes:
      - 2
      - 9

input_elevation:
  - datasets:
      - /data/lidar.las
    omit_LAS_classes:
      - 7
      - 18
    thinning: 1

options:
  building_radius_vertex_elevation: 3.0
  radius_vertex_elevation: 1.0
  threshold_jump_edges: 0.5
  extent: %.2f, %.2f, %.2f, %.2f
""" % EXT


def _config_for(d):
    """CONFIG minus the dataset blocks whose layer file has no features: 3dfier
    aborts on an empty layer ("field 'fid' not found").  When every layer has
    features this returns CONFIG unchanged, byte for byte."""
    head, rest = CONFIG.split('lifting_options:', 1)
    blocks = head.split('  - datasets:')
    keep = [blocks[0]]
    for b in blocks[1:]:
        f = b.split('/data/', 1)[1].split('\n', 1)[0].strip()
        try:
            n = len(json.load(open(f"{d}/{f}")).get('features', []))
        except (OSError, ValueError):
            n = 1
        if n:
            keep.append(b)
    return '  - datasets:'.join(keep) + 'lifting_options:' + rest


def run_3dfier(d, tag):
    open(f"{d}/config.yml", 'w').write(_config_for(d))
    os.chmod(d, 0o777)
    t0 = time.time()
    p = subprocess.run(
        ['docker', 'run', '--rm', '-v', f'{d}:/data', IMAGE,
         '/data/config.yml', '--OBJ', f'/data/out_{tag}.obj'],
        capture_output=True, text=True, timeout=1800)
    el = time.time() - t0
    log = (p.stdout or '') + (p.stderr or '')
    open(f"{d}/log_{tag}.txt", 'w').write(log)
    ok = os.path.exists(f"{d}/out_{tag}.obj")
    return ok, el, log


# --------------------------------------------------------------------------- #
# 3. reading 3dfier's output
# --------------------------------------------------------------------------- #
def load_obj(p):
    V = []; F = []; fobj = []; fmat = []
    cur = None; mat = None
    for line in open(p):
        if line[:2] == 'v ':
            V.append(line.split()[1:4])
        elif line[:2] == 'f ':
            t = line.split()[1:]
            idx = [int(x.split('/')[0]) for x in t]
            idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            for k in range(1, len(idx) - 1):
                F.append((idx[0], idx[k], idx[k + 1])); fobj.append(cur); fmat.append(mat)
        elif line[:2] == 'o ':
            cur = line[2:].strip()
        elif line.startswith('usemtl'):
            mat = line.split()[1].strip()
    return (np.array(V, 'f8'), np.array(F, 'i8'),
            np.array(fobj, object), np.array(fmat, object))


def open_edges(V, F, tol=1e-6):
    q = np.round(V / tol).astype('i8')
    _, inv = np.unique(q, axis=0, return_inverse=True)
    G = inv[F]
    e = np.sort(np.concatenate([G[:, [0, 1]], G[:, [1, 2]], G[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum()), int((c > 2).sum()), int(len(u))


# --------------------------------------------------------------------------- #
# raster / TIN helpers
# --------------------------------------------------------------------------- #
def raster_tin(V, F, gx0, gy0, cell, nx, ny):
    """Z and owning-triangle index per cell centre, by barycentric scan."""
    Z = np.full((ny, nx), np.nan, 'f8')
    TI = np.full((ny, nx), -1, 'i4')
    a = V[F[:, 0]]; b = V[F[:, 1]]; c = V[F[:, 2]]
    x0 = np.minimum(np.minimum(a[:, 0], b[:, 0]), c[:, 0])
    x1 = np.maximum(np.maximum(a[:, 0], b[:, 0]), c[:, 0])
    y0 = np.minimum(np.minimum(a[:, 1], b[:, 1]), c[:, 1])
    y1 = np.maximum(np.maximum(a[:, 1], b[:, 1]), c[:, 1])
    i0 = np.clip(np.floor((x0 - gx0) / cell).astype(int), 0, nx - 1)
    i1 = np.clip(np.ceil((x1 - gx0) / cell).astype(int), 0, nx - 1)
    j0 = np.clip(np.floor((y0 - gy0) / cell).astype(int), 0, ny - 1)
    j1 = np.clip(np.ceil((y1 - gy0) / cell).astype(int), 0, ny - 1)
    for k in range(len(F)):
        ii = np.arange(i0[k], i1[k] + 1); jj = np.arange(j0[k], j1[k] + 1)
        if not len(ii) or not len(jj):
            continue
        X, Y = np.meshgrid(ii * cell + gx0 + cell / 2, jj * cell + gy0 + cell / 2)
        ax, ay = a[k, 0], a[k, 1]
        v0x, v0y = b[k, 0] - ax, b[k, 1] - ay
        v1x, v1y = c[k, 0] - ax, c[k, 1] - ay
        den = v0x * v1y - v1x * v0y
        if abs(den) < 1e-12:
            continue
        px = X - ax; py = Y - ay
        u = (px * v1y - v1x * py) / den
        v = (v0x * py - px * v0y) / den
        m = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        if not m.any():
            continue
        z = a[k, 2] + u * (b[k, 2] - a[k, 2]) + v * (c[k, 2] - a[k, 2])
        J, I = np.meshgrid(jj, ii, indexing='ij')
        Z[J[m], I[m]] = z[m]
        TI[J[m], I[m]] = k
    return Z, TI


def subdivide(V, F, max_edge, max_depth):
    """Midpoint-split any triangle whose longest edge exceeds max_edge."""
    V = list(map(tuple, V))
    mid = {}

    def m_of(i, j):
        k = (i, j) if i < j else (j, i)
        r = mid.get(k)
        if r is None:
            a = V[i]; b = V[j]
            V.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2))
            r = len(V) - 1
            mid[k] = r
        return r

    cur = [tuple(f) for f in F]
    for _ in range(max_depth):
        nxt = []
        split = 0
        for (i, j, k) in cur:
            a = V[i]; b = V[j]; c = V[k]
            e = max(math.hypot(a[0] - b[0], a[1] - b[1]),
                    math.hypot(b[0] - c[0], b[1] - c[1]),
                    math.hypot(c[0] - a[0], c[1] - a[1]))
            if e <= max_edge:
                nxt.append((i, j, k)); continue
            ij = m_of(i, j); jk = m_of(j, k); ki = m_of(k, i)
            nxt += [(i, ij, ki), (ij, j, jk), (ki, jk, k), (ij, jk, ki)]
            split += 1
        cur = nxt
        if split == 0:
            break
    return np.array(V, 'f8'), np.array(cur, 'i8')


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def stage_partition():
    raw, tally, wm = collect_layers()
    feats, st = partition(raw)
    os.makedirs(WORK + '/after', exist_ok=True)
    os.makedirs(WORK + '/before', exist_ok=True)
    # "after" = the planar partition
    ca = write_geojson(feats, WORK + '/after')
    # "before" = the same OSM layers, unclipped and with no remainder terrain:
    # the unpartitioned baseline, for the open-edge comparison
    braw = []
    for lay in R.PRIORITY:
        braw += [dict(it) for it in raw[lay]]
    cb = write_geojson(braw, WORK + '/before')
    npts, npad = write_las(WORK + '/after/lidar.las')
    if not os.path.exists(WORK + '/before/lidar.las'):
        os.link(WORK + '/after/lidar.las', WORK + '/before/lidar.las')
    info = dict(counts_after=ca, counts_before=cb, tally=tally, partition=st,
                water_mask=wm, las_points=npts, las_pad_points=npad,
                extent=list(EXT))
    json.dump(info, open(WORK + '/partition.json', 'w'), indent=1)
    sub_mark('partition', info)
    print("partition:", json.dumps(st, indent=1))
    print("features after:", ca, " before:", cb,
          f" las {npts:,} pts ({npad:,} pad)")
    return info


def stage_run():
    res = {}
    for tag in ('before', 'after'):
        d = WORK + '/' + tag
        ok, el, log = run_3dfier(d, tag)
        if not ok:
            print(log[-3000:])
            raise SystemExit(f"3dfier {tag} produced no OBJ")
        V, F, fo, fm = load_obj(f"{d}/out_{tag}.obj")
        o, nm, tot = open_edges(V, F)
        res[tag] = dict(seconds=round(el, 2), vertices=int(len(V)),
                        triangles=int(len(F)),
                        objects=int(len(set(fo.tolist()))),
                        unique_edges=tot, open_edges=o, non_manifold_edges=nm,
                        by_class={k: int(v) for k, v in
                                  collections.Counter(fm.tolist()).items()},
                        z_min=round(float(V[:, 2].min()), 2),
                        z_max=round(float(V[:, 2].max()), 2))
        print(tag, json.dumps(res[tag]))
    res['open_edge_drop_pct'] = round(
        100.0 * (res['before']['open_edges'] - res['after']['open_edges'])
        / max(res['before']['open_edges'], 1), 1)
    json.dump(res, open(WORK + '/run.json', 'w'), indent=1)
    sub_mark('run', res)
    return res


def main():
    if not _HAS_AREA:
        sys.exit('s10a_3dfier: set OPENCITY_AREA to an area with paths.scratch_dir')
    os.makedirs(WORK, exist_ok=True)
    MG.require('s10a_3dfier', 6.0, need_disk_gb=3.0)
    t0 = time.time()
    if not sub_done('partition'):
        stage_partition()
    if not sub_done('run'):
        stage_run()
    import s10a_layers
    if not sub_done('layers'):
        s10a_layers.build()
    write_summary()
    C.mark('threedfier', json.load(open(WORK + '/summary.json')))
    print(f"s10a done in {time.time()-t0:.0f}s")


def write_summary():
    P = json.load(open(WORK + '/partition.json'))
    R_ = json.load(open(WORK + '/run.json'))
    Lz = json.load(open(WORK + '/layers.json'))
    st = Lz['layers']['structures']
    s = dict(
        image=IMAGE,
        extent_m=list(EXT),
        partition=P['partition'],
        input_features=dict(before=P['counts_before'], after=P['counts_after']),
        lidar=dict(points=P['las_points'], pad_points=P['las_pad_points'],
                   note='ENU metres, the model frame; 3dfier never reprojects, so its '
                        'output needs no transform back. Pad points are the '
                        'nearest-measured-cell ground fill for any strip between '
                        'the lidar footprint and the model rectangle, at 2 m spacing.'),
        run=R_,
        repair=Lz['repair'],
        layers=Lz['layers'],
        zfight=Lz['zfight'],
        defects=[
            dict(what='pier and bridge decks land on the water surface',
                 detail='Bridge/Overpass lifting is percentile-50 of the returns near '
                        'each vertex with no class filter; a pier over open water gets '
                        'the class-9 water surface from those returns',
                 measured=dict(deck_correction_median_m=st['deck_correction_median_m'],
                               deck_correction_max_m=st['deck_correction_max_m'],
                               decks=st['parts'],
                               fix='deck height re-measured from returns above the water '
                                   'plane; arnis man_made.rs:63-104 ground+1 m fallback')),
            dict(what='no-data vertices',
                 detail='vertices with no elevation point inside the search radius are '
                        'written as z = -99.99',
                 measured=dict(vertices=Lz['repair']['nodata_vertices'],
                               fix='replaced with the nearest valid vertex height')),
            dict(what='torn shared edges (T-junctions)',
                 detail='each input polygon is triangulated on its own, so a vertex on '
                        'one side of a shared edge leaves the other side whole; this is '
                        'the dominant source of open edges and does not depend on '
                        'triangulation density',
                 measured=dict(open_edges_raw=Lz['repair']['open_edges_raw'],
                               open_edges_after_weld=Lz['repair']['open_edges_after_repair'],
                               points_welded=Lz['repair']['tjunction_points'])),
        ])
    json.dump(s, open(WORK + '/summary.json', 'w'), indent=1)
    return s


if __name__ == '__main__':
    with MG.PeakRSS(label='s10a_3dfier') as pk:
        main()
    print(f"PEAK_RSS_GB {pk.peak:.2f}")
