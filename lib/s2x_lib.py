"""Shared geometry helpers for the vector layers (buildings, structures, paths).

ENU metres throughout (x=E, y=N, z=U).  Every function here is pure; the calling
stages (s10a_3dfier, s10a_layers) own the IO.
"""
import os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import common as C

import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

LAYERS = C.W + '/layers'
# margin (m) of the clip box around the area bbox.  Vector features are clipped to the
# box so nothing reaches far past the terrain; the layer partition then cuts every
# feature to the bbox itself, so the margin only keeps a feature that straddles the
# bbox edge whole while it is classified and filtered by area.
CLIP_PAD_M = 100.0


def water_z():
    """The area's flat water plane (ENU m): stage_knobs.WATER_Z.value, which
    pipeline/derive_knobs.py sets to the median of the lidar class-9 returns (a hand
    entry in area.yaml wins).  None when no area is set or the area has no water body."""
    sk = ((getattr(C, '_AREA_CFG', None) or {}).get('stage_knobs') or {})
    v = (sk.get('WATER_Z') or {}).get('value')
    return None if v is None else float(v)


def clip_box():
    """The area bbox in ENU metres plus CLIP_PAD_M on every side; None when no area is
    set (the bbox is empty), and then clip() leaves geometry whole."""
    if not any(C.BBOX_WSEN) or getattr(C, '_areacfg', None) is None:
        return None
    return shapely.box(*C._areacfg.enu_ext(C, pad=CLIP_PAD_M))


def clip(g):
    """Intersect a (Multi)Polygon with the model extent; None if it falls outside."""
    if g is None:
        return None
    box = clip_box()
    if box is None:
        return keep_polygons(g)
    try:
        g = g.intersection(box)
    except Exception:
        g = shapely.make_valid(g).intersection(box)
    return keep_polygons(g)

# ---- raster sampling ------------------------------------------------------ #
class Grid:
    def __init__(self):
        S = np.load(C.FW + '/surfaces.npz')
        self.x0, self.y0, self.cell, nx, ny = [float(v) for v in S['grid']]
        self.nx, self.ny = int(nx), int(ny)
        self.dtm = S['dtm'].astype('f8')
        self.m_water = S['m_water']
        self.S = S

    def ij(self, x, y):
        ix = np.clip(((np.asarray(x) - self.x0) / self.cell).astype(int), 0, self.nx - 1)
        iy = np.clip(((np.asarray(y) - self.y0) / self.cell).astype(int), 0, self.ny - 1)
        return ix, iy

    def sample(self, arr, x, y):
        ix, iy = self.ij(x, y)
        return arr[iy, ix]

    def dtm_at(self, x, y):
        return self.sample(self.dtm, x, y)


# ---- OSM -> shapely ------------------------------------------------------- #
def slug(s, fallback):
    if not s:
        return str(fallback)
    s = s.lower().replace('&', ' and ')
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s or str(fallback)


def way_polygon(w, min_area=0.0, do_clip=True):
    """Closed way -> valid (Multi)Polygon in ENU metres, or None."""
    xy = C.way_xy(w)
    if xy is None or len(xy) < 4:
        return None
    if np.hypot(*(xy[0] - xy[-1])) > 1e-6:
        xy = np.vstack([xy, xy[:1]])
    p = Polygon(xy)
    if not p.is_valid:
        p = shapely.make_valid(p)
    p = keep_polygons(p)
    if do_clip:
        p = clip(p)
    if p is None or p.area < min_area:
        return None
    return p


def way_line(w):
    xy = C.way_xy(w)
    if xy is None or len(xy) < 2:
        return None
    # drop repeated points
    keep = np.r_[True, (np.abs(np.diff(xy, axis=0)).sum(1) > 1e-9)]
    xy = xy[keep]
    if len(xy) < 2:
        return None
    return LineString(xy)


def rel_polygon(rel, ways_by_id=None):
    """Multipolygon relation (outer/inner members with geometry) -> (Multi)Polygon."""
    outers, inners = [], []
    for m in rel.get('members', []):
        if m.get('type') != 'way':
            continue
        g = m.get('geometry')
        if not g:
            continue
        lat = np.array([p['lat'] for p in g]); lon = np.array([p['lon'] for p in g])
        xy = C.lla_to_enu(lat, lon, C.ORIGIN[2])[:, :2]
        (inners if m.get('role') == 'inner' else outers).append(xy)
    def rings(parts):
        out = []
        for xy in _stitch(parts):
            if len(xy) >= 4:
                out.append(xy)
        return out
    o = rings(outers); i = rings(inners)
    if not o:
        return None
    polys = []
    for ring in o:
        holes = [h for h in i if Polygon(ring).contains(Polygon(h).representative_point())]
        try:
            p = Polygon(ring, holes)
        except Exception:
            p = Polygon(ring)
        if not p.is_valid:
            p = shapely.make_valid(p)
        polys.append(p)
    return clip(keep_polygons(unary_union(polys)))


def _stitch(parts, tol=1e-6):
    """Join open way fragments into closed rings where possible."""
    parts = [np.asarray(p) for p in parts if len(p) >= 2]
    out = []
    while parts:
        cur = list(parts.pop(0))
        changed = True
        while changed:
            changed = False
            if np.hypot(*(np.asarray(cur[0]) - np.asarray(cur[-1]))) < tol:
                break
            for k, q in enumerate(parts):
                if np.hypot(*(q[0] - np.asarray(cur[-1]))) < tol:
                    cur += list(q[1:]); parts.pop(k); changed = True; break
                if np.hypot(*(q[-1] - np.asarray(cur[-1]))) < tol:
                    cur += list(q[::-1][1:]); parts.pop(k); changed = True; break
                if np.hypot(*(q[-1] - np.asarray(cur[0]))) < tol:
                    cur = list(q[:-1]) + cur; parts.pop(k); changed = True; break
                if np.hypot(*(q[0] - np.asarray(cur[0]))) < tol:
                    cur = list(q[::-1][:-1]) + cur; parts.pop(k); changed = True; break
        cur = np.asarray(cur)
        if np.hypot(*(cur[0] - cur[-1])) > tol:
            cur = np.vstack([cur, cur[:1]])
        out.append(cur)
    return out


def keep_polygons(g):
    if g is None or g.is_empty:
        return None
    if isinstance(g, Polygon):
        return g if g.area > 0 else None
    ps = [p for p in getattr(g, 'geoms', []) if isinstance(p, Polygon) and p.area > 0]
    if not ps:
        return None
    return ps[0] if len(ps) == 1 else MultiPolygon(ps)


# ---- mesh building -------------------------------------------------------- #
class Mesh:
    """Accumulates vertices / triangles / colours for one part."""
    def __init__(self):
        self.V = []; self.F = []; self.Cv = []

    def add(self, verts, faces, rgba):
        n = sum(len(v) for v in self.V)
        self.V.append(np.asarray(verts, 'f8'))
        self.F.append(np.asarray(faces, 'i8') + n)
        self.Cv.append(np.tile(np.asarray(rgba, 'u1'), (len(verts), 1)))


# ---- colours -------------------------------------------------------------- #
_NAMED = {'red': (178, 60, 52), 'white': (238, 238, 234), 'black': (60, 60, 62),
          'grey': (150, 150, 150), 'gray': (150, 150, 150), 'blue': (60, 88, 150),
          'green': (86, 120, 80), 'yellow': (210, 196, 120), 'brown': (130, 104, 82),
          'beige': (220, 212, 190), 'tan': (200, 184, 152), 'silver': (190, 190, 192)}


def parse_colour(s, default):
    if not s:
        return default
    s = s.strip().lower()
    if s.startswith('#'):
        h = s[1:]
        if len(h) == 3:
            h = ''.join(c * 2 for c in h)
        if len(h) == 6:
            try:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return default
    return _NAMED.get(s, default)


def shade(rgb, f):
    return tuple(int(max(0, min(255, round(c * f)))) for c in rgb)


def rgba(rgb, a=255):
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(a))


# ---- output --------------------------------------------------------------- #


def uniquify(name, used):
    if name not in used:
        used.add(name); return name
    k = 2
    while f'{name}_{k}' in used:
        k += 1
    used.add(f'{name}_{k}')
    return f'{name}_{k}'
