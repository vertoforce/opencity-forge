"""Shared machinery for the texture stages: poses, projection, occlusion,
delighting and the OSM surface raster.

Everything here is measured off files that already exist:
  * panorama poses           work/pano_poses.json (pipeline/nomesh.py poses)
  * per-panorama skyline     _shared/pano_pose/work/sky/<id>.npz, an optional external
                             input: lat (sky-boundary latitude in radians, one per
                             equirect column) and valid (bool per column).  No stage
                             here writes it; without it the wall stages skip the sky
                             check.
  * lidar DSM/DTM 0.5 m      _shared/pano_pose/dsm/dsm_fine.npz
  * OSM planar partition     tmp3d/after/*.geojson + fids.json
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import opencity_paths as OCP                    # noqa: E402  host roots, env-overridable
import numpy as np

W    = OCP.CODE
SCR  = OCP.SHARED + '/texture'
PSCR = OCP.SHARED + '/pano_pose'
IMD  = OCP.SOURCE + '/panos/images_pano'
META = OCP.SOURCE + '/panos/metadata/images.jsonl'
SKY  = PSCR + '/work/sky'
POSES_JSON = W + '/pano_poses.json'
# per-area paths: an area build (OPENCITY_AREA set and its
# area.yaml has paths.scratch_dir) moves these; otherwise the defaults
# above stay.
if os.environ.get('OPENCITY_AREA', '').strip():
    import common as _C
    _sd = ((_C._AREA_CFG or {}).get('paths') or {}).get('scratch_dir')
    if _sd:
        W = _C.W; SCR = _sd + '/texture'; PSCR = _sd + '/pano_pose'
        IMD = _C.PANO + '/images_pano'; META = _C.PANO + '/metadata/images.jsonl'
        SKY = PSCR + '/work/sky'
        POSES_JSON = W + '/pano_poses.json'

os.makedirs(SCR, exist_ok=True)

# ---------------------------------------------------------------- poses ---- #
def load_poses():
    """-> (dict id -> pose, provenance string).

    pose: x, y, z (photo-ENU metres), R (3x3 world->camera), confidence,
    residual_deg, captured_at (epoch ms) and w/h of the source jpg.
    """
    meta = {}
    for l in open(META):
        d = json.loads(l)
        meta[d['id']] = d
    out = {}
    if not os.path.exists(POSES_JSON):
        raise FileNotFoundError(f'{POSES_JSON} missing: run pipeline/nomesh.py poses first')
    doc = json.load(open(POSES_JSON))
    for pid, p in doc['poses'].items():
        if p.get('method') == 'none':
            continue
        m = meta.get(pid, {})
        out[pid] = dict(x=p['x'], y=p['y'], z=p['z'],
                        R=np.array(p['R'], 'f8'),
                        confidence=p['confidence'],
                        residual_deg=p.get('residual_deg', 9.9),
                        loss=p.get('loss', 1.0),
                        captured_at=m.get('captured_at'),
                        w=m.get('width'), h=m.get('height'))
    src = (f"pano_poses.json ({doc['meta'].get('method','')[:40]}...), "
           f"tiers {doc['meta'].get('counts')}")
    return out, src


def thin_poses(poses, spacing_m=2.0, keep=('high', 'usable')):
    """Spatial thinning: keep the best-residual pose in each spacing_m cell.

    The panoramas are sequences a metre apart; without this the projection pays
    for hundreds of near-identical views of the same square metre."""
    sel = {k: v for k, v in poses.items() if v['confidence'] in keep}
    best = {}
    for pid, p in sel.items():
        c = (int(math.floor(p['x'] / spacing_m)), int(math.floor(p['y'] / spacing_m)))
        b = best.get(c)
        if b is None or p['residual_deg'] < sel[b]['residual_deg']:
            best[c] = pid
    return {pid: sel[pid] for pid in best.values()}


# ------------------------------------------------------------- imagery ---- #
def pano_rgb(pid, width=2048):
    """Equirect RGB at `width` x `width/2`, loaded through the JPEG DCT scaler."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(f"{IMD}/{pid}.jpg")
    im.draft('RGB', (width, width // 2))
    im = im.convert('RGB')
    if im.size != (width, width // 2):
        im = im.resize((width, width // 2), Image.BILINEAR)
    return np.asarray(im, 'u1')


def skyline(pid):
    """(lat[ncol] radians, valid[ncol]) from the panorama's optional skyline file,
    or (None, None) when it has none."""
    f = f"{SKY}/{pid}.npz"
    if not os.path.exists(f):
        return None, None
    d = np.load(f)
    return d['lat'].astype('f4'), d['valid']


def project(P, C, R):
    """World points -> equirect direction.  Returns (lon, lat, rng) in radians/m.

    Convention is the one pano_poses.json uses: X = R @ (P - C),
    lon = atan2(X0, X2), lat = atan2(-X1, hypot(X0, X2))."""
    d = P - np.asarray(C, 'f4')
    X = d @ R.T.astype('f4')
    rng = np.linalg.norm(d, axis=1)
    lon = np.arctan2(X[:, 0], X[:, 2])
    lat = np.arctan2(-X[:, 1], np.hypot(X[:, 0], X[:, 2]))
    return lon, lat, rng


def sample_equirect(img, lon, lat):
    """Bilinear sample of an equirect image at (lon, lat) radians -> float RGB."""
    H, Wd = img.shape[:2]
    u = (lon / (2 * math.pi) + 0.5) * Wd - 0.5
    v = (0.25 - lat / (2 * math.pi)) * Wd - 0.5
    u = np.mod(u, Wd)
    v = np.clip(v, 0.0, H - 1.0001)
    u0 = np.mod(np.floor(u), Wd).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    fu = (u - u0)[:, None]; fv = (v - v0)[:, None]
    u1 = (u0 + 1) % Wd; v1 = np.minimum(v0 + 1, H - 1)
    a = img[v0, u0].astype('f4'); b = img[v0, u1].astype('f4')
    c = img[v1, u0].astype('f4'); d = img[v1, u1].astype('f4')
    return (a * (1 - fu) * (1 - fv) + b * fu * (1 - fv) +
            c * (1 - fu) * fv + d * fu * fv)


# ------------------------------------------------------------------ DSM ---- #
class DSM:
    """0.5 m max-Z lidar surface over the area, plus the bare-earth DTM."""
    def __init__(self):
        d = np.load(PSCR + '/dsm/dsm_fine.npz')
        self.Z = np.where(np.isfinite(d['dsm']), d['dsm'], -1e4).astype('f4')
        self.dtm = d['dtm'].astype('f4')
        self.water = d['water']
        x0, y0, c, nx, ny = [float(v) for v in d['grid']]
        self.x0, self.y0, self.c = x0, y0, c
        self.nx, self.ny = int(nx), int(ny)
        # fill holes in the DSM with the DTM where there is one, else ground level
        hole = self.Z < -1e3
        fill = np.where(np.isfinite(self.dtm), self.dtm, -4.4).astype('f4')
        self.Z[hole] = fill[hole]
        wd = np.load(PSCR + '/dsm/dsm_wide.npz')
        self.WZ = np.where(np.isfinite(wd['dsm']), wd['dsm'], -1e4).astype('f4')
        wx0, wy0, wc, wnx, wny = [float(v) for v in wd['grid']]
        self.wx0, self.wy0, self.wc = wx0, wy0, wc
        self.wnx, self.wny = int(wnx), int(wny)

    def sample(self, X, Y):
        """Nearest-cell DSM height, falling back to the wide grid outside."""
        j = np.rint((X - self.x0) / self.c - 0.5).astype(np.int32)
        i = np.rint((Y - self.y0) / self.c - 0.5).astype(np.int32)
        inside = (j >= 0) & (j < self.nx) & (i >= 0) & (i < self.ny)
        out = np.full(X.shape, -1e4, 'f4')
        out[inside] = self.Z[i[inside], j[inside]]
        if (~inside).any():
            wj = np.clip(np.rint((X[~inside] - self.wx0) / self.wc - 0.5), 0, self.wnx - 1).astype(np.int32)
            wi = np.clip(np.rint((Y[~inside] - self.wy0) / self.wc - 0.5), 0, self.wny - 1).astype(np.int32)
            out[~inside] = self.WZ[wi, wj]
        return out


# occlusion ray-march radii: fine near the camera where it matters
def march_radii(r_max=220.0):
    return np.concatenate([np.arange(0.75, 25.0, 0.25),
                           np.arange(25.0, 80.0, 0.75),
                           np.arange(80.0, r_max, 3.0)]).astype('f4')


NAZ = 1440           # 0.25 degree azimuth bins


class Occluder:
    """Per-camera line-of-sight test against the lidar DSM.

    The line of sight to a point at horizontal range r and height Pz is the
    straight line z(t) = Cz + t*(Pz-Cz)/r, so the DSM blocks it exactly when
    (Z(t) - Cz)/t exceeds (Pz - Cz)/r somewhere before the point.  Taking the
    running maximum of (Z-Cz)/t along each azimuth turns that into one lookup.
    """
    def __init__(self, dsm, C, r_max=220.0, back_off_m=1.75):
        self.C = np.asarray(C, 'f8')
        self.radii = march_radii(r_max)
        self.back_off = back_off_m
        az = (np.arange(NAZ) + 0.5) * (2 * math.pi / NAZ) - math.pi
        self.az0 = -math.pi
        self.daz = 2 * math.pi / NAZ
        X = self.C[0] + np.cos(az)[:, None] * self.radii[None, :]
        Y = self.C[1] + np.sin(az)[:, None] * self.radii[None, :]
        Z = dsm.sample(X, Y)
        slope = (Z - self.C[2]) / self.radii[None, :]
        slope[Z < -1e3] = -9.9
        self.cmax = np.maximum.accumulate(slope, axis=1).astype('f4')

    def visible(self, P, margin_tan=0.012):
        """True where the DSM does not block the ray.  margin_tan is a slope
        slack (0.012 ~ 0.7 deg) that absorbs pose error and DSM noise."""
        d = P[:, :2] - self.C[None, :2]
        r = np.hypot(d[:, 0], d[:, 1])
        az = np.arctan2(d[:, 1], d[:, 0])
        ai = np.clip(((az - self.az0) / self.daz).astype(np.int32), 0, NAZ - 1)
        rr = np.maximum(r - self.back_off, 0.5)
        ri = np.searchsorted(self.radii, rr) - 1
        ok_r = ri >= 0
        ri = np.clip(ri, 0, len(self.radii) - 1)
        slope_pt = (P[:, 2] - self.C[2]) / np.maximum(r, 0.3)
        vis = self.cmax[ai, ri] <= slope_pt + margin_tan
        vis &= ok_r
        return vis


def shadow_map(dsm, sun_az_deg, sun_el_deg, step=0.5, r_max=140.0):
    """Cast-shadow mask on the 0.5 m DSM for one sun direction.

    True = the DSM surface at that cell is in its own cast shadow."""
    if sun_el_deg <= 0.5:
        return np.ones(dsm.Z.shape, bool)
    ax = math.radians(sun_az_deg); el = math.radians(sun_el_deg)
    dx = math.cos(ax); dy = math.sin(ax)
    tan_el = math.tan(el)
    ny, nx = dsm.Z.shape
    Xc = dsm.x0 + (np.arange(nx) + 0.5) * dsm.c
    Yc = dsm.y0 + (np.arange(ny) + 0.5) * dsm.c
    GX, GY = np.meshgrid(Xc, Yc)
    Z0 = dsm.Z
    sh = np.zeros(Z0.shape, bool)
    t = step
    while t < r_max:
        Zs = dsm.sample(GX + dx * t, GY + dy * t)
        sh |= (Zs > Z0 + t * tan_el + 0.35)
        t += step if t < 25 else 1.5
    return sh


# --------------------------------------------------------------- the sun --- #
def sun_position(epoch_s, lat_deg, lon_deg):
    """NOAA solar position.  Returns (azimuth deg from East toward North to match
    the model's ENU yaw convention, elevation deg)."""
    jd = epoch_s / 86400.0 + 2440587.5
    n = jd - 2451545.0
    g = math.radians((357.52911 + 0.98560028 * n) % 360)
    q = (280.46646 + 0.9856474 * n) % 360
    lam = math.radians((q + 1.914602 * math.sin(g) + 0.019993 * math.sin(2 * g)) % 360)
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    lst = math.radians((gmst * 15 + lon_deg) % 360)
    ha = lst - ra
    la = math.radians(lat_deg)
    el = math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha))
    azn = math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(dec) * math.cos(la))
    az_from_north = (math.degrees(azn) + 180.0) % 360.0     # compass bearing
    az_enu = (90.0 - az_from_north) % 360.0                 # from +x toward +y
    return az_enu, math.degrees(el)


# ------------------------------------------------- OSM surface raster ------ #
SURF_CELL = 0.25
SURF_CLASSES = ['none', 'lawn', 'paving', 'road', 'wood', 'gravel', 'sand',
                'bare', 'scrub', 'water', 'building']


def surface_ext():
    """The surface raster's rectangle: a fixed +-260 x +-250 m with no area set, or an area's
    own bbox (pipeline/areacfg.enu_ext) plus 10 m, rounded out to whole metres."""
    if os.environ.get('OPENCITY_AREA', '').strip():
        import common as _C
        if hasattr(_C, '_areacfg') and _C._areacfg.build_area_cfg(_C) is not None:
            return _C._areacfg.enu_ext(_C, pad=10.0)
    return -260.0, -250.0, 260.0, 250.0


def build_surface_raster(cache=None, rebuild=False):
    """OSM planar-partition surface class and feature kind per 0.25 m cell.

    Built from the same geojson the terrain TIN was cut from, so the raster and
    the geometry cannot disagree.  Returns (surf, kind, kinds_list, grid)."""
    cache = cache or (SCR + '/surface_raster.npz')
    x0, y0, x1, y1 = surface_ext()
    nx = int((x1 - x0) / SURF_CELL); ny = int((y1 - y0) / SURF_CELL)
    if os.path.exists(cache) and not rebuild:
        d = np.load(cache, allow_pickle=True)
        g = [float(v) for v in d['grid']]
        # size check: a cache from another extent is rebuilt, not reused
        if g == [x0, y0, SURF_CELL, float(nx), float(ny)]:
            return d['surf'], d['kind'], list(d['kinds']), g
        print(f'surface raster cache grid {g} != {[x0, y0, SURF_CELL, nx, ny]}: rebuilding', flush=True)
    from shapely.geometry import shape
    import shapely
    d3 = W + '/tmp3d/after'
    # painted low priority first, so the arnis PRIORITY order wins
    order = ['terrain', 'water', 'road', 'bridge', 'building']
    surf = np.zeros((ny, nx), 'i1')
    kind = np.zeros((ny, nx), 'i2')
    kinds = ['none']
    ki = {'none': 0}
    Xc = x0 + (np.arange(nx) + 0.5) * SURF_CELL
    Yc = y0 + (np.arange(ny) + 0.5) * SURF_CELL
    for name in order:
        fp = f"{d3}/{name}.geojson"
        if not os.path.exists(fp):
            continue
        for ft in json.load(open(fp))['features']:
            pr = ft.get('properties', {})
            s = pr.get('surface') or ('building' if name == 'building' else 'bare')
            if name == 'building':
                s = 'building'
            if s not in SURF_CLASSES:
                s = 'bare'
            k = str(pr.get('kind') or name)
            if k not in ki:
                ki[k] = len(kinds); kinds.append(k)
            try:
                g = shape(ft['geometry'])
            except Exception:
                continue
            if g.is_empty:
                continue
            gx0, gy0, gx1, gy1 = g.bounds
            j0 = max(0, int((gx0 - x0) / SURF_CELL) - 1)
            j1 = min(nx, int((gx1 - x0) / SURF_CELL) + 2)
            i0 = max(0, int((gy0 - y0) / SURF_CELL) - 1)
            i1 = min(ny, int((gy1 - y0) / SURF_CELL) + 2)
            if j1 <= j0 or i1 <= i0:
                continue
            GX, GY = np.meshgrid(Xc[j0:j1], Yc[i0:i1])
            inside = shapely.contains_xy(g, GX, GY)
            if not inside.any():
                continue
            sb = surf[i0:i1, j0:j1]; kb = kind[i0:i1, j0:j1]
            sb[inside] = SURF_CLASSES.index(s); kb[inside] = ki[k]
            surf[i0:i1, j0:j1] = sb; kind[i0:i1, j0:j1] = kb
    np.savez_compressed(cache, surf=surf, kind=kind,
                        kinds=np.array(kinds, object),
                        grid=np.array([x0, y0, SURF_CELL, nx, ny], 'f8'))
    return surf, kind, kinds, [x0, y0, SURF_CELL, nx, ny]


def surface_at(ras, grid, X, Y):
    x0, y0, c, nx, ny = grid
    j = np.clip(((X - x0) / c).astype(np.int32), 0, int(nx) - 1)
    i = np.clip(((Y - y0) / c).astype(np.int32), 0, int(ny) - 1)
    return ras[i, j]


# --------------------------------------------------- one-panorama sampler -- #
SRGB_LUT = (np.linspace(0, 1, 256) ** 2.2).astype('f4')       # cheap sRGB->linear
LUMW = np.array([0.2126, 0.7152, 0.0722], 'f4')


def srgb_to_linear(u8):
    return SRGB_LUT[np.asarray(u8, 'u1')]


def linear_to_srgb(lin):
    return np.clip(np.power(np.clip(lin, 0, 1), 1 / 2.2) * 255.0, 0, 255).astype('u1')


class PanoSampler:
    """Everything needed to take colour from one panorama at a set of world
    points: range gate, DSM line-of-sight, the panorama's own measured skyline,
    and a bilinear equirect sample.

    Shared by the delight solve (s19) and the projection (s20) so the two cannot
    disagree about which samples are admissible."""

    def __init__(self, dsm, pid, pose, r_max=45.0, width=2048, r_min=2.0):
        self.pid = pid
        self.pose = pose
        self.C = np.array([pose['x'], pose['y'], pose['z']], 'f8')
        self.R = pose['R']
        self.r_max = r_max
        self.r_min = r_min
        self.width = width
        self.occ = Occluder(dsm, self.C, r_max=r_max + 20.0)
        self._img = None

    @property
    def img(self):
        if self._img is None:
            self._img = pano_rgb(self.pid, self.width)
        return self._img

    def admissible(self, P):
        """-> (index into P, lon, lat, range, incidence weight input).

        Applies the range gate, the DSM line-of-sight test and the sky mask."""
        d = P[:, :2] - self.C[None, :2].astype('f4')
        r = np.hypot(d[:, 0], d[:, 1])
        near = np.nonzero((r > self.r_min) & (r < self.r_max))[0]
        if len(near) == 0:
            return near, None, None, None
        vis = self.occ.visible(P[near])
        sel = near[vis]
        if len(sel) == 0:
            return sel, None, None, None
        lon, lat, rng = project(P[sel], self.C, self.R)
        sl, sv = skyline(self.pid)
        if sl is not None:
            col = np.clip(((lon / (2 * math.pi) + 0.5) * len(sl)).astype(np.int32),
                          0, len(sl) - 1)
            keep = ~(sv[col] & (lat > sl[col] - 0.02))
            sel = sel[keep]; lon = lon[keep]; lat = lat[keep]; rng = rng[keep]
        return sel, lon, lat, rng

    def sample(self, P):
        """-> (index, linear RGB float32, range).  Empty arrays if nothing hits."""
        sel, lon, lat, rng = self.admissible(P)
        if len(sel) == 0:
            return sel, np.zeros((0, 3), 'f4'), np.zeros(0, 'f4')
        rgb = sample_equirect(self.img, lon, lat)
        lin = np.power(np.clip(rgb / 255.0, 0, 1), 2.2).astype('f4')
        return sel, lin, rng.astype('f4')

    def free(self):
        self._img = None
