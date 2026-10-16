"""s10a part 2: turn 3dfier's OBJ into the build's layer files.

3dfier hands back one 2.5D surface per class.  This module:
  * repairs its two known output defects -- no-data (-99.99) vertices and
    T-junctions where a shared edge is split on one side only,
  * uses Terrain + Road as the walkable floor under LAYER_ground, pushed under
    the photo ground so the photo mesh keeps every pixel it has,
  * uses Building as LAYER_buildings massing, with the wall bases dropped onto
    the lowered floor so nothing floats,
  * uses Water as LAYER_water,
  * uses Bridge as the LAYER_structures decks -- after correcting the deck
    height, which 3dfier gets wrong over water (see DECK note below),
  * rewrites geom/backstop.npz, the floor raster paths are draped onto.
"""
import sys, os, json, math, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import common as C
import osm_rules as R
import s2x_lib as L
import s10a_3dfier as S

CELL = S.CELL
CLEAR = S.PHOTO_CLEAR
FLOOR_CLASSES = ('Terrain', 'TerrainWall', 'Road', 'RoadWall')
BUILD_CLASSES = ('Building', 'BuildingWall', 'BuildingFloor')
WATER_CLASSES = ('Water', 'WaterWall')
BRIDGE_CLASSES = ('Bridge', 'BridgeWall')
SKIRT = 0.35
# OSM building=roof is an open shelter (often with amenity=shelter), not a closed
# mass.  3dfier extrudes it from the ground to its lidar roof like any building,
# which gives a dark windowless cube.  For these ways the footprint's BuildingFloor triangles go to
# the floor layer (ground visible under the roof), the mass keeps only a
# ROOF_BAND_M slab under its roof, and square POST_M posts at <= POST_SPACING_M
# along the footprint inset carry it.  Their coplanar outer faces can exceed
# s33's 0.4 m wall minimum, so s33 skips the last post_faces faces of an
# open_shelter part explicitly; such parts get no building skirt.
ROOF_BAND_M = 1.0
# 0.50 m is a plausible timber/steel column section; much thinner posts under a
# large shelter roof render as thin dark sticks.
POST_M = 0.50
POST_SPACING_M = 4.5
POST_INSET_M = 0.45
# 3dfier's WaterWall fills the height jump from a water
# polygon to every neighbour, so where a building stands on the water edge a
# water-textured wall climbs to its roof (tens of metres above the water).  A steep
# water face (|n_z| <= WATER_FLAT_NZ) whose top is more than WATER_MAX_RISE_M above its local
# water level (the flat-water vertex nearest the face's lowest corner, so a pond above
# another water body keeps its own rim) is dropped.  Real rims rise about 1.3 m at most and a bogus wall
# over 3 m, so 3.0 separates them.  The level (level_z_enu_m) stays the median of the
# undropped vertices.  The area knob stage_knobs.WATER_MAX_RISE_M
# {value: <m> | null (off)} overrides this default per area.
WATER_MAX_RISE_M = 3.0
WATER_FLAT_NZ = 0.95


def _roof_ways():
    return {e['id'] for e in C.osm_elements()
            if e['type'] == 'way' and e.get('tags', {}).get('building') == 'roof'}


def _open_objs(props):
    rw = _roof_ways()
    return {o for o, p in props.items()
            if p.get('layer') == 'building' and p.get('osm_id') in rw}


def _posts(fp, zfun, ztop):
    """Square posts along the footprint inset: every corner of the simplified
    inset ring, plus evenly spaced ones so no gap exceeds POST_SPACING_M."""
    ring = fp.buffer(-POST_INSET_M)
    if ring.is_empty:
        return np.zeros((0, 3)), np.zeros((0, 3), int), 0
    if ring.geom_type != 'Polygon':
        ring = max(ring.geoms, key=lambda g: g.area)
    xy = np.asarray(ring.simplify(0.3).exterior.coords)[:-1]
    pts = []
    for k in range(len(xy)):
        a, b = xy[k], xy[(k + 1) % len(xy)]
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / POST_SPACING_M)))
        for j in range(n):
            pts.append((a + (b - a) * j / n, (b - a) / (np.linalg.norm(b - a) + 1e-9)))
    POS, IDX = [], []
    h = POST_M / 2
    for c, u in pts:
        v = np.array([-u[1], u[0]])
        q = [c + h * (su * u + sv * v) for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        zb = zfun(c) - 0.10
        base = len(POS) * 0 + sum(len(x) for x in POS)
        ring4 = np.array([[x[0], x[1], zb] for x in q] + [[x[0], x[1], ztop] for x in q])
        f = []
        for i in range(4):
            i2 = (i + 1) % 4
            f += [[i, i2, 4 + i2], [i, 4 + i2, 4 + i]]
        # top and bottom caps make each post a closed box.  An open post's top and
        # bottom rims are boundary edges, and a stitch pass would hang a skin off
        # them down to the terrain, so each post would render as the stitch quad.
        f += [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
        f = np.array(f)
        # make every side face point away from the post centre
        cen = ring4.mean(0)
        for r in range(len(f)):
            t = ring4[f[r]]
            nrm = np.cross(t[1] - t[0], t[2] - t[0])
            if np.dot(nrm, t.mean(0) - cen) < 0:
                f[r] = f[r][::-1]
        POS.append(ring4); IDX.append(f + base)
    return np.vstack(POS), np.vstack(IDX), len(pts)


# what the open shelter's posts and slab underside are painted with.  The
# panoramas see a post as a thin sliver in the shadow of the roof, so stage 20
# projects a near-black mean linear colour (about 0.015-0.04) onto the posts,
# stage 21 splits them into several materials by that noise, and stage 33 groups
# the coplanar outer post faces into facade walls, so each post would be a stack
# of textures and a photo strip, near black.  The slab underside is the same
# (projected 0.00-0.03).  So s20 and s21 call shelter_faces() and give both one
# colour, SHELTER_RGB (the light concrete the mass already gets), and one flat
# material, SHELTER_MAT; s33 skips the posts through the part's post_faces count.
SHELTER_MAT = 'concrete'
SHELTER_RGB = (170, 168, 163)
SHELTER_UNDER_SHADE = 0.92


def shelter_lin(shade=1.0):
    """SHELTER_RGB as the linear colour stage 20 writes per face."""
    c = np.asarray(SHELTER_RGB, 'f8') / 255.0 * shade
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype('f4')


def shelter_faces(meta, pos, idx):
    """(post_faces, underside_faces) of every open_shelter part of the
    buildings layer: the posts are the last post_faces faces of the part (see
    _posts' call site), the underside is every other face of the part whose
    normal points down (z < -0.7)."""
    post, under = [], []
    for p in meta['parts']:
        o = (p.get('extras') or {}).get('open_shelter')
        if not o:
            continue
        a, b = p['face_start'], p['face_end']
        npf = int(o.get('post_faces') or 0)
        post.append(np.arange(b - npf, b))
        f = np.arange(a, b - npf)
        t = pos[idx[f]].astype('f8')
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
        under.append(f[nz < -0.7])
    cat = lambda x: np.concatenate(x).astype('i8') if x else np.zeros(0, 'i8')
    return cat(post), cat(under)


# --------------------------------------------------------------------------- #
# repairs
# --------------------------------------------------------------------------- #
def repair_nodata(V, F):
    """3dfier writes -99.99 where it found no elevation point for a vertex."""
    cut = -50.0
    if hasattr(C, '_areacfg') and C._areacfg.build_area_cfg(C) is not None:
        # with an area set the cut comes from the area's own ground: 3dfier can
        # write no-data near -50 as well as -99.99 (a vertex at -49.99 would leave
        # a 50 m pit under a fixed -50 cut), so anything 10 m below the lowest DTM
        # cell is no-data.
        dtm = np.load(C.FW + '/surfaces.npz')['dtm']
        cut = float(np.nanmin(dtm)) - 10.0
    bad = V[:, 2] < cut
    n = int(bad.sum())
    if n:
        good = ~bad
        kt = cKDTree(V[good][:, :2])
        _, j = kt.query(V[bad][:, :2], k=1)
        V[bad, 2] = V[good][j, 2]
    return V, n


def weld(V, F, tol=1e-6):
    q = np.round(V / tol).astype('i8')
    u, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return V[first], inv[F], inv


# --------------------------------------------------------------------------- #
# photo-ground reference rasters
# --------------------------------------------------------------------------- #
def photo_ground():
    """(PZmin, has, grid) -- the 3x3 minimum of the photo ground's per-cell p10.

    The 3x3 minimum of the per-cell p10 keeps the floor below the photo ground,
    which is the z-fight guarantee.
    """
    gx0, gy0, x1, y1 = S.EXT
    nx = int(round((x1 - gx0) / CELL)); ny = int(round((y1 - gy0) / CELL))
    lay = np.load(C.W + '/geom/face_layer.npy')
    Cn = np.load(C.G + '/centroid.npy', mmap_mode='r')
    g = np.asarray(Cn[lay == 1])
    cache = C.W + '/tmp3d/photo_ground_p10.npz'
    if os.path.exists(cache):
        z = np.load(cache)
        if z['PZmin'].shape == (ny, nx):     # a cache from another extent is stale
            return z['PZmin'], z['has'], (gx0, gy0, nx, ny), g
    jx = np.clip(((g[:, 0] - gx0) / CELL).astype(int), 0, nx - 1)
    jy = np.clip(((g[:, 1] - gy0) / CELL).astype(int), 0, ny - 1)
    lin = jy * nx + jx
    o = np.argsort(lin, kind='stable')
    ls = lin[o]; zs = g[o, 2]
    uu, ss = np.unique(ls, return_index=True); ee = np.append(ss[1:], len(ls))
    pz = np.full(nx * ny, np.nan, 'f8')
    for u, a, b in zip(uu, ss, ee):
        pz[u] = np.percentile(zs[a:b], 10)
    PZ = pz.reshape(ny, nx)
    PZmin = ndimage.generic_filter(np.where(np.isnan(PZ), 1e6, PZ), np.min, size=3,
                                   mode='constant', cval=1e6)
    has = PZmin < 1e5
    np.savez_compressed(cache, PZmin=PZmin.astype('f4'), has=has)
    return PZmin, has, (gx0, gy0, nx, ny), g


def clamp_under_photo(V, F, PZmin, has, grid, rounds=4):
    """Push the floor under the photo ground, vertices first then per triangle."""
    gx0, gy0, nx, ny = grid
    req = np.where(has, PZmin - CLEAR, np.inf)
    ix = np.clip(((V[:, 0] - gx0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((V[:, 1] - gy0) / CELL).astype(int), 0, ny - 1)
    V[:, 2] = np.minimum(V[:, 2], req[iy, ix])
    hist = []
    for it in range(rounds):
        Z, TI = S.raster_tin(V, F, gx0, gy0, CELL, nx, ny)
        viol = has & np.isfinite(Z) & (Z > req + 1e-6)
        n = int(viol.sum())
        hist.append(n)
        if n == 0:
            break
        vy, vx = np.nonzero(viol)
        drop = Z[vy, vx] - req[vy, vx]
        tri = TI[vy, vx]
        # each offending triangle's vertices come down by the worst violation
        per = collections.defaultdict(float)
        for t, dz in zip(tri, drop):
            if t >= 0 and dz > per[t]:
                per[t] = dz
        dv = np.zeros(len(V))
        for t, dz in per.items():
            for v in F[t]:
                dv[v] = max(dv[v], dz)
        V[:, 2] -= dv
    return V, hist


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build():
    t0 = time.time()
    d = S.WORK + '/after'
    props = json.load(open(d + '/fids.json'))
    V, F, fo, fm = S.load_obj(d + '/out_after.obj')
    before = dict(vertices=int(len(V)), triangles=int(len(F)))
    V, nnodata = repair_nodata(V, F)
    V, F, inv = weld(V, F)
    fo = np.asarray(fo); fm = np.asarray(fm)
    o0, nm0, _ = S.open_edges(V, F)
    F, nsplit, nfaces_split, fo, fm = _split_with_tags(V, F, fo, fm)
    o1, nm1, _ = S.open_edges(V, F)
    repair = dict(nodata_vertices=nnodata, tjunction_points=int(nsplit),
                  faces_resplit=int(nfaces_split),
                  open_edges_raw=o0, open_edges_after_repair=o1,
                  non_manifold_raw=nm0, non_manifold_after_repair=nm1,
                  triangles_after_repair=int(len(F)))
    print("repair:", json.dumps(repair))

    cls = np.asarray(fm)
    sel = {k: np.isin(cls, v) for k, v in
           dict(floor=FLOOR_CLASSES, building=BUILD_CLASSES,
                water=WATER_CLASSES, bridge=BRIDGE_CLASSES).items()}

    PZmin, has, grid, gpts = photo_ground()
    gx0, gy0, nx, ny = grid

    opened = _open_objs(props)
    shelter_floor = np.isin(fo, list(opened)) & (cls == 'BuildingFloor')
    if shelter_floor.any():
        Fo = F.copy()
        t = V[Fo[shelter_floor]]
        nz = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2]
        fl = np.nonzero(shelter_floor)[0][nz < 0]
        Fo[fl] = Fo[fl][:, ::-1]
        sel_floor = sel['floor'] | shelter_floor
    else:
        Fo, sel_floor = F, sel['floor']
    print(f"open shelters: {sorted(opened)}; {int(shelter_floor.sum())} floor tris to the floor layer")
    out = {}
    out['floor'] = _floor(V, Fo[sel_floor], fo[sel_floor], props,
                          PZmin, has, grid, gpts)
    floorZ = out['floor']['Zr']
    out['buildings'] = _buildings(V, F[sel['building']], fo[sel['building']],
                                  cls[sel['building']], props, floorZ, grid)
    out['water'] = _water(V, F[sel['water']], cls[sel['water']])
    out['structures'] = _structures(V, F[sel['bridge']], fo[sel['bridge']], props)
    zf = (_zfight(out['floor'], grid, gpts) if len(gpts) else
          dict(photo_ground_faces_tested=0, note='no photo mesh: nothing to z-fight'))
    stats = dict(image=S.IMAGE, obj_before_repair=before, repair=repair,
                 layers={k: {kk: vv for kk, vv in v.items()
                             if kk not in ('Zr',)} for k, v in out.items()},
                 zfight=zf, seconds=round(time.time() - t0, 1))
    json.dump(stats, open(C.W + '/tmp3d/layers.json', 'w'), indent=1)
    S.sub_mark('layers', stats)
    C.note('threedfier_layers', {k: v for k, v in stats.items() if k != 'layers'})
    C.note('threedfier_layer_counts', stats['layers'])
    print("zfight:", json.dumps(zf))
    return stats


def _split_with_tags(V, F, fo, fm):
    """repair_tjunctions, carrying the object/class tag onto the new faces."""
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    key = e[:, 0] * (len(V)) + e[:, 1]
    u, idx, cnt = np.unique(key, return_index=True, return_counts=True)
    op = idx[cnt == 1]
    pair = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])[op]
    face = np.tile(np.arange(len(F)), 3)[op]
    kt = cKDTree(V)
    ins = collections.defaultdict(dict)
    nsplit = 0
    for k in range(len(op)):
        a, b = pair[k]
        A = V[a]; B = V[b]; dd = B - A
        L2 = float(dd @ dd)
        if L2 < 1e-12:
            continue
        hits = []
        for i in kt.query_ball_point(0.5 * (A + B), r=math.sqrt(L2) / 2 + 0.02):
            if i == a or i == b:
                continue
            t = float((V[i] - A) @ dd) / L2
            if 1e-6 < t < 1 - 1e-6 and np.linalg.norm(V[i] - (A + t * dd)) < 0.02:
                hits.append((t, i))
        if not hits:
            continue
        f = face[k]
        c = [j for j in range(3) if F[f][j] == a and F[f][(j + 1) % 3] == b]
        if not c:
            c = [j for j in range(3) if F[f][j] == b and F[f][(j + 1) % 3] == a]
            if not c:
                continue
            hits = [(1.0 - t, i) for t, i in hits]
        ins[f][c[0]] = sorted(hits)
        nsplit += len(hits)
    out = []; oo = []; mm = []
    for f in range(len(F)):
        if f not in ins:
            out.append(F[f]); oo.append(fo[f]); mm.append(fm[f]); continue
        loop = []
        for c in range(3):
            loop.append(F[f][c])
            for t, i in ins[f].get(c, []):
                loop.append(i)
        for k in range(1, len(loop) - 1):
            out.append([loop[0], loop[k], loop[k + 1]])
            oo.append(fo[f]); mm.append(fm[f])
    return (np.array(out, 'i8'), nsplit, len(ins),
            np.array(oo, object), np.array(mm, object))


# --------------------------------------------------------------------------- #
def _compact(V, F):
    used, inv = np.unique(F.ravel(), return_inverse=True)
    return V[used], inv.reshape(-1, 3).astype('u4'), used


def _floor(V, F, fo, props, PZmin, has, grid, gpts):
    gx0, gy0, nx, ny = grid
    Vf, Ff, used = _compact(V, F)
    surf = np.array([str(props.get(str(o), {}).get('surface') or 'bare') for o in fo])
    tris_in = len(Ff)
    # carry the surface class to vertices before subdividing
    vsurf = np.array(['bare'] * len(Vf), object)
    for s in ('bare', 'gravel', 'sand', 'wood', 'scrub', 'lawn', 'road', 'paving', 'water'):
        m = surf == s
        if m.any():
            vsurf[np.unique(Ff[m])] = s
    Vs, Fs = S.subdivide(Vf, Ff, S.SUBDIV_EDGE_M, S.SUBDIV_MAX_DEPTH)
    # vertices added by subdivision inherit from the nearest original vertex
    if len(Vs) > len(Vf):
        kt = cKDTree(Vf[:, :2])
        _, j = kt.query(Vs[len(Vf):, :2], k=1)
        vsurf = np.concatenate([vsurf, vsurf[j]])
    Vs, hist = clamp_under_photo(Vs, Fs, PZmin, has, grid)
    Zr, TI = S.raster_tin(Vs, Fs, gx0, gy0, CELL, nx, ny)
    # colour: the reconstruction's own ground points where they reach, then the
    # OSM surface class from the decision table
    src_xy, gcol = _ground_point_colour()
    if len(gcol):
        kt = cKDTree(src_xy)
        dd, jj = kt.query(Vs[:, :2], k=8, distance_upper_bound=6.0)
    else:   # no photo points (no-mesh mode): the OSM class colour everywhere
        dd = np.full((len(Vs), 8), np.inf); jj = np.zeros((len(Vs), 8), 'i8')
        gcol = np.zeros((1, 3), 'f8')
    ok = np.isfinite(dd)
    col = np.zeros((len(Vs), 3), 'f8')
    hasc = ok.any(1)
    w = np.where(ok, 1.0 / np.maximum(dd, 0.25) ** 2, 0.0)
    jc = np.where(ok, jj, 0)
    num = (gcol[jc] * w[..., None]).sum(1); den = w.sum(1)
    col[hasc] = num[hasc] / den[hasc][:, None]
    cls_col = np.array([R.SURFACE_RGB.get(s, R.SURFACE_RGB['bare']) for s in vsurf], 'f8')
    col[~hasc] = cls_col[~hasc]
    rgba = np.concatenate([np.clip(col, 0, 255).astype('u1'),
                           np.full((len(col), 1), 255, 'u1')], 1)
    np.savez_compressed(C.W + '/layers/ground_terrain.npz',
                        pos=Vs.astype('f4'), idx=Fs.astype('u4'), color=rgba,
                        part=np.zeros(len(Fs), 'i4'))
    area = float(np.isfinite(Zr).sum() * CELL * CELL)
    meta = dict(group='ground_terrain',
        source='3dfier (tudelft3d) constrained-Delaunay terrain over the OSM '
               'planar partition, lifted from USGS 3DEP lidar '
               'class 2/9 returns registered to the photo frame',
        licence='3DEP public domain; polygon outlines OpenStreetMap ODbL 1.0; '
                'colours from the Mapillary photo reconstruction (CC BY-SA 4.0) '
                'where they reach, OSM surface classes (ODbL) elsewhere',
        method=f'3dfier Terrain + Road classes from the planar partition, run in '
               f'Docker ({S.IMAGE}); triangles longer than {S.SUBDIV_EDGE_M} m '
               f'subdivided; then pushed to at most the 3x3 minimum of the photo '
               f"ground's per-cell 10th percentile minus {CLEAR} m so it backs the "
               f'photo mesh holes without fighting it for depth; re-checked by '
               f'rasterising the TIN and lowering any triangle still above it',
        parts=[dict(index=0, name='ground_terrain_3dfier', face_start=0,
                    face_end=int(len(Fs)),
                    extras=dict(cell_m=CELL, vertices=int(len(Vs)),
                                faces=int(len(Fs)),
                                triangles_from_3dfier=int(tris_in),
                                area_m2=round(area, 1),
                                photo_coloured_pct=round(float(hasc.mean() * 100), 1),
                                clamp_violation_cells_by_round=hist,
                                role='continuous walkable floor under the whole land '
                                     'area; polygon-conforming TIN, not a raster grid'))])
    json.dump(meta, open(C.W + '/layers/ground_terrain.json', 'w'), indent=1)
    # the raster form paths are re-draped onto
    fill = np.isfinite(Zr)
    # cells the floor does not own (water, building footprints, bridge decks) take
    # the nearest floor height, so a path clipping the corner of one does not jump
    j = ndimage.distance_transform_edt(~fill, return_distances=False,
                                       return_indices=True)
    Zb = Zr[tuple(j)]
    np.savez_compressed(C.W + '/geom/backstop.npz', Z=Zb.astype('f4'), fill=fill,
                        grid=np.array([gx0, gy0, CELL, nx, ny], 'f8'))
    C.note('ground_terrain', meta['parts'][0]['extras'])
    return dict(triangles=int(len(Fs)), triangles_from_3dfier=int(tris_in),
                vertices=int(len(Vs)), area_m2=round(area, 1),
                photo_coloured_pct=round(float(hasc.mean() * 100), 1),
                clamp_rounds=hist, Zr=Zr)


_GC = {}


def _ground_point_colour():
    """The photo reconstruction's own near-ground points and their colours."""
    if 'c' not in _GC:
        d = np.load(C.FW + '/loaded.npz')
        P = d['photo']; RGB = d['photo_rgb']
        Su = np.load(C.FW + '/surfaces.npz')
        sx0, sy0, sc, snx, sny = [float(v) for v in Su['grid']]
        iy = np.clip(((P[:, 1] - sy0) / sc).astype(int), 0, int(sny) - 1)
        ix = np.clip(((P[:, 0] - sx0) / sc).astype(int), 0, int(snx) - 1)
        h = P[:, 2] - Su['dtm'][iy, ix]
        g = (h > -0.4) & (h < 0.45)
        _GC['xy'] = P[g][:, :2]
        _GC['c'] = RGB[g].astype('f8')
    return _GC['xy'], _GC['c']


def _part_span(g3, r3, eff, minh, hsrc):
    """(base, roof, fix) for one 3dfier building object; the rules are in
    _buildings' docstring."""
    h3 = r3 - g3
    tagged = hsrc in ('osm_height', 'osm_levels')
    if minh > 0.0 and tagged:
        # an elevated part: the lidar inside its footprint measures whatever
        # stands on top of it, so the tag wins outright (buildings.rs:2693-2742)
        return g3 + minh, g3 + minh + eff, 'osm_min_height_plus_' + hsrc
    if h3 < 2.0 and eff >= 2.0:
        return g3 + minh, g3 + minh + eff, 'osm_' + hsrc + '_no_lidar_roof'
    if hsrc == 'osm_height' and h3 > minh + eff + 2.0:
        return g3 + minh, g3 + minh + eff, 'lidar_overshoot_vs_osm_height'
    return g3, r3, None


def _part_geom(Vb, Fb, oarr, carr, props, o):
    """Roof/floor vertex split, 3dfier ground/roof z, the provisional span and
    the XY footprint (union of the roof triangles) of one building object."""
    import shapely
    m = oarr == o
    pr = props.get(o, {})
    sub = Fb[m]
    roof_v = np.unique(sub[carr[m] == 'Building'])
    floor_v = np.unique(sub[carr[m] == 'BuildingFloor'])
    allv = np.unique(sub)
    if not len(roof_v) or not len(floor_v):
        roof_v = allv[Vb[allv, 2] >= Vb[allv, 2].mean()]
        floor_v = allv[Vb[allv, 2] < Vb[allv, 2].mean()]
    g3 = float(np.median(Vb[floor_v, 2]))       # 3dfier ground, class-2 p10
    r3 = float(np.median(Vb[roof_v, 2]))
    eff = float(pr.get('height_m') or 0.0)
    minh = float(pr.get('min_height_m') or 0.0)
    span = _part_span(g3, r3, eff, minh, pr.get('height_source') or 'none')
    tris = sub[carr[m] == 'Building']
    fp = shapely.union_all([shapely.Polygon(Vb[t, :2]) for t in tris
                            if shapely.Polygon(Vb[t, :2]).area > 1e-6])
    return dict(verts=(roof_v, floor_v, allv, g3, r3), span=span, g3=g3,
                minh=minh, fp=fp)


# An OSM building:part with min_height is a volume that starts above the ground
# because another part of the same building stands under it.  3dfier's planar
# partition makes every footprint disjoint, so where an upper part overlaps a lower one the shared
# area goes to the upper part alone and the lower part's volume under it is
# gone.  Keeping min_height there floats the mass, and a building skirt then fills
# the gap with one tall grey band.
# The support test therefore runs on the PRE-partition OSM footprints
# (osm/osm_raw.json), not on the layer.  Raised masses are usually stacked, with
# nearly all of their footprint over OSM parts that reach their base; few truly
# overhang.
# Rule: OSM volumes are resolved in ascending min_height; one is standing if it
# starts on the ground, or if at least SUPPORT_FRAC of its footprint is over
# standing volumes whose top reaches within SUPPORT_GAP_M of its base.  A raised
# layer part whose OSM way is standing lost its support to the partition: its
# base drops to the floor and its tagged roof stays, so the column becomes a
# wall the facade stage can texture.  A raised part that is not standing in OSM
# is a true overhang or bridge and keeps its min_height.  Building outlines that
# contain parts are not volumes (Simple 3D Buildings) and never support.
SUPPORT_FRAC = 0.5
SUPPORT_GAP_M = 1.0


# the support test is per OSM way, but the partition hands every
# overlap to the upper part, so what is left of a stacked way on the layer can
# be exactly the part of it that stands over nothing (an undercroft strip, or a
# piece over no OSM part below it).  Dropping such a piece to the floor builds a
# solid wall where the photograph shows soffit, pilotis and open ground.  So
# each raised piece of a stacked way is split by the OSM parts
# under it: per face of the arrangement of those footprints, the storeys that
# exist there give the lowest z the column is continuous from (0 = ground).
# A piece that stands (continuous from the ground) drops to the floor; a piece
# that is over a gap keeps its base at the top of the gap (for these pieces
# its own min_height) and becomes a real overhang with a soffit, with no skirt
# hung under it.  The partition pieces are disjoint and 3dfier gives each its
# own walls (the grounded neighbour's wall on the shared edge runs from the
# floor to its roof), so no gap or overlap opens at the split.  A piece with
# more than MIXED_M2 of both kinds would need a mesh cut; it drops whole and is
# counted (overhang_mixed_pieces).
# s10a_3dfier's stack_clip gives stacked cells to the upper part, so a tower's
# piece covers its podium plus its outline offset; without it the leftovers
# where the outline runs 0.1-4 m past the podium's would be full-height slivers
# over empty ground.  A piece that is mostly
# grounded (grounded >= hang) is a standing column with an outline offset, not
# an overhang: it drops whole and is counted as overhang_minor_pieces, not as
# mixed.
MIXED_M2 = 0.5
_VOLS = {}
_MINOR = {}


def _osm_volumes():
    """Pre-partition OSM building volumes: [dict(id, part, fp, top, base)]."""
    if 'v' in _VOLS:
        return _VOLS['v']
    from shapely.geometry import Polygon
    els = C.osm_elements()
    nodes = {e['id']: e for e in els if e['type'] == 'node'}

    def num(t, k):
        try:
            return float(str(t.get(k)).split()[0])
        except (TypeError, ValueError):
            return None
    vols = []
    for w in els:
        t = w.get('tags', {})
        if w['type'] != 'way' or not ('building' in t or 'building:part' in t):
            continue
        nd = [nodes[n] for n in w.get('nodes', []) if n in nodes]
        if len(nd) < 4:
            continue
        xy = C.lla_to_enu(np.array([n['lat'] for n in nd]),
                          np.array([n['lon'] for n in nd]), C.ORIGIN[2])[:, :2]
        fp = Polygon(xy).buffer(0)
        if fp.is_empty or fp.area < 1e-3:
            continue
        top = num(t, 'height')
        if top is None and num(t, 'building:levels') is not None:
            top = 3.0 * num(t, 'building:levels')
        base = num(t, 'min_height')
        if base is None and num(t, 'building:min_level') is not None:
            base = 3.0 * num(t, 'building:min_level')
        vols.append(dict(id=w['id'], part='building:part' in t, fp=fp,
                         top=top or 0.0, base=base or 0.0))
    parts = [v['fp'] for v in vols if v['part']]
    vols = [v for v in vols if v['part']
            or not any(v['fp'].buffer(0.5).contains(q) for q in parts)]
    _VOLS['v'] = vols
    return vols


def _osm_stacked_ways():
    """{OSM way id: OSM base m} of raised building volumes that stand on lower
    volumes in the pre-partition OSM footprints (see the rule above)."""
    import shapely
    standing = []; stacked = {}
    for v in sorted(_osm_volumes(), key=lambda v: v['base']):
        if v['base'] <= SUPPORT_GAP_M:
            standing.append(v); continue
        under = [s['fp'] for s in standing if s['top'] >= v['base'] - SUPPORT_GAP_M
                 and s['base'] < v['base'] - SUPPORT_GAP_M]
        cov = (shapely.union_all(under).intersection(v['fp']).area / v['fp'].area
               if under else 0.0)
        if cov >= SUPPORT_FRAC:
            standing.append(v); stacked[v['id']] = v['base']
    return stacked


def _column_bottom(q, base, parts):
    """Lowest OSM height (0 = ground) from which the storeys at point q run
    continuously up to `base` (gaps <= SUPPORT_GAP_M are closed)."""
    z = 0.0; zb = 0.0
    for b, t in sorted((v['base'], v['top']) for v in parts if v['fp'].contains(q)):
        if b >= base - SUPPORT_GAP_M:
            break
        if b > z + SUPPORT_GAP_M:
            zb = b
        z = max(z, t)
    return base if z < base - SUPPORT_GAP_M else zb


def _piece_split(fp, base):
    """{column bottom (OSM m): area m2} over one layer footprint."""
    import shapely
    from shapely.ops import polygonize
    parts = [v for v in _osm_volumes() if v['part'] and v['fp'].intersects(fp)]
    lines = shapely.union_all([fp.boundary] + [v['fp'].boundary for v in parts])
    out = collections.Counter()
    for f in polygonize(lines):
        f = f.intersection(fp)
        if f.area < 1e-4:
            continue
        out[round(_column_bottom(f.representative_point(), base, parts), 2)] += f.area
    return out


def _unsupported_parts(geo, props):
    """(drop, overhang, mixed): layer objects raised on this layer whose OSM way
    is stacked, split by what stands under the piece itself.  drop = stands on
    the ground (base drops to the floor); overhang = {o: (column bottom m,
    area m2)} = over a gap (keeps a raised base); mixed = both, dropped."""
    stacked = _osm_stacked_ways()
    drop = set(); overhang = {}; mixed = set(); minor = set()
    for o, g in geo.items():
        oid = props.get(o, {}).get('osm_id')
        raised = g['minh'] > 0.0 and g['span'][0] - g['g3'] > SUPPORT_GAP_M
        if not raised or oid not in stacked:
            continue
        sp = _piece_split(g['fp'], stacked[oid])
        grounded = sum(a for z, a in sp.items() if z <= SUPPORT_GAP_M)
        hang = {z: a for z, a in sp.items() if z > SUPPORT_GAP_M}
        if sum(hang.values()) < MIXED_M2:
            drop.add(o)
        elif grounded < MIXED_M2 and len([a for a in hang.values() if a >= MIXED_M2]) <= 1:
            z = max(hang, key=hang.get)
            overhang[o] = (z, round(sum(hang.values()), 1))
        elif grounded >= sum(hang.values()):
            drop.add(o); minor.add(o)
        else:
            drop.add(o); mixed.add(o)
    _MINOR['n'] = len(minor)
    return drop, overhang, mixed


def _buildings(V, F, fo, cls, props, Zr, grid):
    """3dfier LoD1 massing, with its two z defects corrected.

    3dfier has no notion of OSM min_height / building:min_level: it extrudes
    every building:part from the ground to that part's own lidar roof.  That
    makes a raised volume (min_height > 0) into a solid standing on the
    pavement, and it makes a low part under a tower as tall as the tower,
    because the lidar over its footprint sees the tower above it.  It also
    returns a roof at ground level where the returns inside the footprint are
    water (for example a vessel mapped as building=ship) or missing.

    The corrections use osm_rules.building_height, which is arnis's precedence
    (buildings.rs:2693-2742, 2942-2971): height beats building:levels, the base
    is min_height, then building:min_level, then layer, then 0.  Every part
    records which rule moved it.
    """
    gx0, gy0, nx, ny = grid
    Vb, Fb, used = _compact(V, F)
    ix = np.clip(((Vb[:, 0] - gx0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((Vb[:, 1] - gy0) / CELL).astype(int), 0, ny - 1)
    fz = Zr[iy, ix]
    objs = sorted(set(str(o) for o in fo))
    oarr = np.array([str(x) for x in fo])
    carr = np.array([str(x) for x in cls])
    parts = []; POS = []; IDX = []; COL = []; nv = 0; nf = 0
    heights = []; moved = []; fixes = collections.Counter()
    used_names = set()
    geo = {o: _part_geom(Vb, Fb, oarr, carr, props, o) for o in objs}
    floating, overhang, mixed = _unsupported_parts(geo, props)
    if mixed:
        print(f"WARNING: {len(mixed)} raised pieces are part grounded, part overhang; dropped whole: {sorted(mixed)}")
    fixes['overhang_mixed_pieces'] = len(mixed)
    fixes['overhang_minor_pieces'] = _MINOR.get('n', 0)
    open_ways = _roof_ways()

    def zfun(c):
        i = int(np.clip((c[1] - gy0) / CELL, 0, ny - 1)); j = int(np.clip((c[0] - gx0) / CELL, 0, nx - 1))
        w = Zr[max(0, i - 2):i + 3, max(0, j - 2):j + 3]
        return float(np.nanmin(w)) if np.isfinite(w).any() else ground_base
    for o in objs:
        m = oarr == o
        pr = props.get(o, {})
        sub = Fb[m]
        # re-read here, not from geo: parts share welded wall vertices, so an
        # earlier part's z edit moves this one's floor/roof medians
        roof_v = np.unique(sub[carr[m] == 'Building'])
        floor_v = np.unique(sub[carr[m] == 'BuildingFloor'])
        allv = np.unique(sub)
        if not len(roof_v) or not len(floor_v):
            roof_v = allv[Vb[allv, 2] >= Vb[allv, 2].mean()]
            floor_v = allv[Vb[allv, 2] < Vb[allv, 2].mean()]
        g3 = float(np.median(Vb[floor_v, 2]))       # 3dfier ground, class-2 p10
        r3 = float(np.median(Vb[roof_v, 2]))
        h3 = r3 - g3
        eff = float(pr.get('height_m') or 0.0)      # osm_rules wall span
        minh = float(pr.get('min_height_m') or 0.0)
        hsrc = pr.get('height_source') or 'none'
        base, roof, fix = _part_span(g3, r3, eff, minh, hsrc)
        if o in floating:
            # stacked on a lower OSM part the partition removed: keep the tagged
            # roof, drop the base to the floor (see _unsupported_parts)
            base, fix, minh = g3, 'osm_min_height_dropped_partitioned_support', 0.0
        ovh = overhang.get(o)
        if ovh is not None:
            # over a gap in the OSM stack: a real overhang, soffit at the gap top
            base, fix, minh = g3 + ovh[0], 'osm_min_height_kept_true_overhang', ovh[0]
        if fix:
            fixes[fix] += 1
        # the base sits on the lowered floor unless the part is deliberately elevated
        if minh <= 0.0:
            local = fz[allv]
            tgt = float(np.nanmin(local)) - 0.10 if np.isfinite(local).any() else base
            base = min(base, tgt)
        moved.append(abs(base - g3))
        is_open = pr.get('osm_id') in open_ways
        ground_base = base
        if is_open:
            band = min(ROOF_BAND_M, 0.25 * (roof - base))
            base = roof - band
            fix = fix or None
            fixes['open_shelter_roof_slab'] += 1
        Vb[floor_v, 2] = base
        Vb[roof_v, 2] = roof
        wall = np.setdiff1d(allv, np.union1d(roof_v, floor_v))
        if len(wall):
            t = (Vb[wall, 2] - g3) / (h3 if abs(h3) > 1e-6 else 1.0)
            Vb[wall, 2] = base + np.clip(t, 0, 1) * (roof - base)
        vs, invx = np.unique(sub.ravel(), return_inverse=True)
        pnt = Vb[vs]
        wall_rgb = L.parse_colour(None, (170, 168, 163))
        rgba = np.tile(np.array(L.rgba(wall_rgb), 'u1'), (len(vs), 1))
        rv = np.intersect1d(vs, roof_v)
        if len(rv):
            rgba[np.searchsorted(vs, rv)] = L.rgba(L.shade(wall_rgb, 0.80))
        name = L.uniquify('building_' + L.slug(pr.get('name'), pr.get('osm_id') or o),
                          used_names)
        npost = 0; nfx = 0
        if is_open:
            pp, pf, npost = _posts(geo[o]['fp'], zfun, base + 0.05)
            if npost:
                prgb = SHELTER_RGB
                IDXx = pf + len(pnt)
                pnt = np.vstack([pnt, pp])
                rgba = np.vstack([rgba, np.tile(np.array(L.rgba(prgb), 'u1'), (len(pp), 1))])
                invx = np.concatenate([invx.reshape(-1, 3), IDXx]).ravel()
                nfx = len(pf)
            base = ground_base
        h = roof - base
        heights.append(h)
        POS.append(pnt); COL.append(rgba); IDX.append(invx.reshape(-1, 3) + nv)
        parts.append(dict(index=len(parts), name=name, face_start=nf,
                          face_end=nf + len(sub) + nfx,
                          extras=dict(osm_id=pr.get('osm_id'), osm_name=pr.get('name'),
                                      fid=o, faces=int(len(sub) + nfx),
                                      open_shelter=(dict(slab_bottom_z_m=round(roof - min(ROOF_BAND_M, 0.25 * (roof - ground_base)), 2),
                                                         posts=int(npost), post_m=POST_M,
                                                         post_faces=int(nfx))
                                                    if is_open else None),
                                      overhang=(dict(soffit_z_m=round(base, 2),
                                                     open_below_osm_m=ovh[0],
                                                     area_m2=ovh[1])
                                                if ovh is not None else None),
                                      height_m=round(h, 2),
                                      height_source=('3dfier_lidar_p90' if not fix
                                                     else fix),
                                      height_from_3dfier_m=round(h3, 2),
                                      osm_rule_height_m=round(eff, 2),
                                      osm_rule_source=hsrc,
                                      min_height_m=(round(minh, 2) if minh else None),
                                      base_z_m=round(base, 2),
                                      roof_z_m=round(roof, 2),
                                      ground_z_3dfier_m=round(g3, 2))))
        nv += len(pnt); nf += len(sub) + nfx
    _write('buildings', POS, IDX, COL, parts,
           source='OpenStreetMap building/building:part footprints (ODbL 1.0), '
                  'clipped to the planar partition, massed by 3dfier from the '
                  'registered 3DEP lidar',
           licence='OpenStreetMap contributors, ODbL 1.0; lidar 3DEP public domain',
           method=_buildings.__doc__.strip(),
           extra=dict(stats=dict(parts=len(parts),
                                 median_height_m=round(float(np.median(heights)), 2),
                                 height_corrections=dict(fixes),
                                 uncorrected=len(parts) - sum(v for k, v in fixes.items() if not k.startswith("overhang_")),
                                 base_drop_median_m=round(float(np.median(moved)), 3))))
    return dict(parts=len(parts), triangles=int(nf),
                median_height_m=round(float(np.median(heights)), 2),
                height_corrections=dict(fixes),
                uncorrected=len(parts) - sum(v for k, v in fixes.items() if not k.startswith("overhang_")),
                base_drop_median_m=round(float(np.median(moved)), 3))


def _water_max_rise():
    """stage_knobs.WATER_MAX_RISE_M.value of the active area (None/false = off), else
    WATER_MAX_RISE_M (with OPENCITY_AREA unset the code default applies)."""
    sk = ((getattr(C, '_AREA_CFG', None) or {}).get('stage_knobs') or {})
    if 'WATER_MAX_RISE_M' not in sk:
        return WATER_MAX_RISE_M
    v = (sk['WATER_MAX_RISE_M'] or {}).get('value')
    return None if v is None or v is False else float(v)


def _water_rise_drop(Vw, Fw, max_rise, lvl):
    """(drop mask, rise per face): steep faces whose top is over max_rise above the local
    water level (z of the flat-water vertex nearest each face's lowest vertex; lvl when
    the layer has no flat face).  Flat faces are never dropped."""
    n = len(Fw)
    if n == 0:
        return np.zeros(0, bool), np.zeros(0)
    T = Vw[Fw].astype('f8')
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(nrm, axis=1)
    flat = np.abs(nrm[:, 2]) > WATER_FLAT_NZ * np.maximum(ln, 1e-12)
    if flat.any():
        fv = np.unique(Fw[flat])
        lo = T[np.arange(n), T[:, :, 2].argmin(1)]
        loc = Vw[fv[cKDTree(Vw[fv, :2]).query(lo[:, :2])[1]], 2].astype('f8')
    else:
        loc = np.full(n, lvl)
    rise = np.where(flat, 0.0, T[:, :, 2].max(1) - loc)
    if max_rise is None:
        return np.zeros(n, bool), rise
    return rise > max_rise, rise


def _water(V, F, cls):
    Vw, Fw, used = _compact(V, F)
    lvl = float(np.median(Vw[:, 2]))
    max_rise = _water_max_rise()
    drop, rise = _water_rise_drop(Vw, Fw, max_rise, lvl)
    T = Vw[Fw].astype('f8')
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
    rise_doc = dict(water_max_rise_m=max_rise, faces_dropped=int(drop.sum()),
                    area_dropped_m2=round(float(area[drop].sum()), 1),
                    max_rise_dropped_m=round(float(rise[drop].max()), 2) if drop.any() else None,
                    max_rise_kept_m=round(float(rise[~drop].max()), 2) if (~drop).any() else None,
                    rule='steep water faces more than water_max_rise_m above their '
                         'local water level dropped (3dfier WaterWall up a building side)')
    if drop.any():
        Vw, Fw, _ = _compact(Vw, Fw[~drop])
    print("water rise rule:", json.dumps(rise_doc))
    col = np.tile(np.array(L.rgba(R.SURFACE_RGB['water']), 'u1'), (len(Vw), 1))
    wz = L.water_z()                       # the area's derived flat water plane
    _write('water', [Vw], [Fw], [col],
           [dict(index=0, name='water_3dfier', face_start=0, face_end=int(len(Fw)),
                 extras=dict(level_z_enu_m=round(lvl, 3),
                             faces=int(len(Fw)), vertices=int(len(Vw)),
                             derived_water_z_enu_m=wz,
                             delta_to_derived_m=None if wz is None else round(lvl - wz, 3),
                             water_rise=rise_doc))],
           source='3dfier Water class over the measured water mask (lidar class 9 '
                  'plus OSM natural=water / leisure=marina), lifted from 3DEP lidar',
           licence='3DEP public domain; outline partly OpenStreetMap ODbL 1.0',
           method='percentile-10 of the class 2 and 9 returns near each vertex, '
                  'then forced flat per polygon (3dfier Water lifting)',
           extra=dict(level_z_enu_m=round(lvl, 3), water_rise=rise_doc))
    C.note('water', dict(level_z_enu_m=round(lvl, 3), faces=int(len(Fw)),
                         source='3dfier', water_rise_faces_dropped=rise_doc['faces_dropped']))
    return dict(parts=1, triangles=int(len(Fw)), level_z_enu_m=round(lvl, 3),
                water_rise_faces_dropped=rise_doc['faces_dropped'])


def _structures(V, F, fo, props):
    """Bridge / pier decks.

    DECK NOTE: 3dfier's Bridge lifting is percentile-50 of the returns near each
    vertex, with no class filter.  A pier over open water gets those returns from
    the class-9 water surface, so 3dfier puts the deck ON the water (a median
    within about 0.1 m of the water plane).  arnis's rule for man_made=pier is the fix
    (man_made.rs:63-104): a fixed deck one metre above the ground, which never
    samples the water level.  This stage keeps 3dfier's polygon-conforming deck
    outline and lifts each deck to the height measured from the lidar returns that are
    actually above the water, falling back to arnis's ground + 1 m.
    """
    Vb, Fb, used = _compact(V, F)
    # the water plane the decks stand over: the area's derived WATER_Z, else (no water
    # body, so every deck is a bridge over ground) the lowest ground cell
    W = L.water_z()
    if W is None:
        W = float(np.nanmin(np.load(C.FW + '/surfaces.npz')['dtm']))
    lid = np.load(C.FW + '/lidar_reg.npy').astype('f8')
    cls = np.load(C.FW + '/loaded.npz')['lidar_cls']
    lid = lid[~np.isin(cls, [7, 18])]
    del cls
    band_lo, band_hi = W + 0.20, W + 6.0
    kt = cKDTree(lid[:, :2])
    objs = sorted(set(str(o) for o in fo))
    parts = []; POS = []; IDX = []; COL = []; nv = 0; nf = 0
    src_count = collections.Counter()
    corr = []
    used_names = set()
    for o in objs:
        m = np.array([str(x) == o for x in fo])
        sub = Fb[m]
        vs, invx = np.unique(sub.ravel(), return_inverse=True)
        p = Vb[vs].copy()
        pr = props.get(o, {})
        kind = pr.get('kind', 'pier')
        z3d = float(np.median(p[:, 2]))
        # lidar returns above the water inside this deck's footprint
        cen = p[:, :2].mean(0)
        rad = float(np.linalg.norm(p[:, :2] - cen, axis=1).max()) + 0.5
        idx = kt.query_ball_point(cen, r=rad)
        z = lid[idx, 2] if len(idx) else np.zeros(0)
        z = z[(z > band_lo) & (z < band_hi)]
        if len(z) >= 15:
            deck = _deck_z(z, kind, band_lo, band_hi)
            src = 'lidar_above_water'
        else:
            deck = z3d + R.PIER_DECK_ABOVE_GROUND_M
            src = 'arnis_ground_plus_1m'
        deck = float(min(max(deck, W + 0.25), W + 6.0))
        src_count[src] += 1
        corr.append(deck - z3d)
        p[:, 2] = deck
        rgb = R.SURFACE_RGB['wood'] if kind == 'pier' else (130, 130, 135)
        # deck slab with a skirt, capped both faces
        low = p.copy(); low[:, 2] = deck - SKIRT
        P2 = np.vstack([p, low])
        top = invx.reshape(-1, 3)
        bot = top[:, ::-1] + len(p)
        # side walls along the boundary of this object's triangle set
        e = np.sort(np.concatenate([top[:, [0, 1]], top[:, [1, 2]], top[:, [2, 0]]]), axis=1)
        key = e[:, 0] * len(p) + e[:, 1]
        u, ii, cc = np.unique(key, return_index=True, return_counts=True)
        bd = np.concatenate([top[:, [0, 1]], top[:, [1, 2]], top[:, [2, 0]]])[ii[cc == 1]]
        side = np.concatenate([
            np.stack([bd[:, 0], bd[:, 1], bd[:, 1] + len(p)], 1),
            np.stack([bd[:, 0], bd[:, 1] + len(p), bd[:, 0] + len(p)], 1)])
        Ftot = np.concatenate([top, bot, side])
        col = np.vstack([np.tile(np.array(L.rgba(rgb), 'u1'), (len(p), 1)),
                         np.tile(np.array(L.rgba(L.shade(rgb, 0.8)), 'u1'), (len(p), 1))])
        name = L.uniquify(f"{kind}_" + L.slug(pr.get('name'), pr.get('osm_id') or o),
                          used_names)
        POS.append(P2); COL.append(col); IDX.append(Ftot + nv)
        parts.append(dict(index=len(parts), name=name, face_start=nf,
                          face_end=nf + len(Ftot),
                          extras=dict(osm_id=pr.get('osm_id'), osm_name=pr.get('name'),
                                      fid=o, kind=kind,
                                      deck_z_m=round(deck, 2),
                                      deck_height_above_water_m=round(deck - W, 2),
                                      deck_z_from_3dfier_m=round(z3d, 2),
                                      deck_correction_m=round(deck - z3d, 2),
                                      deck_source=src, skirt_m=SKIRT,
                                      pillar_spacing_m=pr.get('pillar_spacing_m'),
                                      faces=int(len(Ftot)))))
        nv += len(P2); nf += len(Ftot)
    _write('structures', POS, IDX, COL, parts,
           source='OpenStreetMap man_made=pier and bridge=* ways (ODbL 1.0) clipped '
                  'to the planar partition; deck outline from 3dfier Bridge/Overpass, '
                  'deck height from the registered 3DEP lidar',
           licence='OpenStreetMap contributors, ODbL 1.0; lidar 3DEP public domain',
           method=_structures.__doc__.strip(),
           extra=dict(water_plane_z_m=W,
                      stats=dict(parts=len(parts), deck_source=dict(src_count),
                                 deck_correction_median_m=(round(float(np.median(corr)), 2) if len(corr) else None),
                                 deck_correction_max_m=(round(float(np.max(corr)), 2) if len(corr) else None))))
    return dict(parts=len(parts), triangles=int(nf),
                deck_source=dict(src_count),
                deck_correction_median_m=(round(float(np.median(corr)), 2) if len(corr) else None),
                deck_correction_max_m=(round(float(np.max(corr)), 2) if len(corr) else None))


def _deck_z(z, kind, lo, hi):
    if kind.startswith('bridge'):
        return float(np.median(z))
    bins = np.arange(lo, hi + 0.01, 0.25)
    h, _ = np.histogram(z, bins=bins)
    if h.max() == 0:
        return float(np.median(z))
    b = bins[int(np.flatnonzero(h >= 0.20 * h.max())[0])] + 0.125
    sel = z[np.abs(z - b) < 0.45]
    return float(np.median(sel)) if len(sel) else float(b)


def _write(group, POS, IDX, COL, parts, source, licence, method, extra=None):
    if not parts:   # an area without this layer (e.g. no piers): write it empty
        POS, IDX, COL = [np.zeros((0, 3), 'f4')], [np.zeros((0, 3), 'u4')], [np.zeros((0, 4), 'u1')]
    pos = np.vstack(POS).astype('f4')
    idx = np.vstack(IDX).astype('u4')
    col = np.vstack(COL).astype('u1')
    part = np.concatenate([np.full(p['face_end'] - p['face_start'], p['index'], 'i4')
                           for p in parts] + [np.zeros(0, 'i4')]).astype('i4')
    np.savez_compressed(C.W + f'/layers/{group}.npz', pos=pos, idx=idx,
                        color=col, part=part)
    doc = dict(group=group, source=source, licence=licence, method=method,
               parts=parts)
    if extra:
        doc.update(extra)
    json.dump(doc, open(C.W + f'/layers/{group}.json', 'w'), indent=1)
    print(f"{group}: {len(parts)} parts, {len(idx):,} tris, {len(pos):,} verts")


def _zfight(floor, grid, gpts):
    """The z-fight test, run on the new floor."""
    gx0, gy0, nx, ny = grid
    Zr = floor['Zr']
    ix = np.clip(((gpts[:, 0] - gx0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((gpts[:, 1] - gy0) / CELL).astype(int), 0, ny - 1)
    fz = Zr[iy, ix]
    ok = np.isfinite(fz)
    dz = gpts[ok, 2] - fz[ok]
    below = dz < 0
    return dict(photo_ground_faces_tested=int(ok.sum()),
                photo_ground_faces_without_floor=int((~ok).sum()),
                median_above_floor_m=round(float(np.median(dz)), 3),
                p5_above_floor_m=round(float(np.percentile(dz, 5)), 3),
                p95_above_floor_m=round(float(np.percentile(dz, 95)), 3),
                faces_below_floor=int(below.sum()),
                faces_below_floor_pct=round(float(below.mean() * 100), 4),
                worst_below_m=round(float(dz.min()), 3),
                passes_under_0p2pct=bool(below.mean() * 100 < 0.2))


if __name__ == '__main__':
    build()
