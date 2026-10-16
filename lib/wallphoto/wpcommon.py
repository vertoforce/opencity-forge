"""wallphoto/ -- one panorama per 1-2 m patch, on walls and structure faces.

Used by stages s60_wall_bestview and s62_wall_integrate; it writes only its own
set directory and imports texlib, s33's wall inventory, s35's material
synthesiser and align.alcommon's transient masks read-only.

Why this exists
---------------
A wall texture made as a top-2 weighted average of up to six panoramas per texel
at ~5 cm is soft: even after the image-space alignment below the views disagree by a
decimetre or two, so averaging them is a box blur of that size applied to a 1 cm photograph.  Here every patch of a
wall takes its colour from exactly ONE panorama, sampled at the panorama's
native equirect width (5760 or 11000 px, which is 0.5-1.6 cm on a wall at
5-15 m), and the differences between neighbouring patches are removed
afterwards by a per-view levelling solve plus a narrow feather.

The pose error is ~1 m, which slides a whole wall's photograph sideways.  Two
things fix that here, both measured and reported per wall:

  * every view of a wall is aligned to the wall's REFERENCE view by a masked
    normalised cross-correlation on the wall plane (Texture2LoD3's "fix it in
    image space, not in pose space"), so neighbouring patches from different
    panoramas line up with each other;
  * the whole aligned set is then pinned to the wall's own geometry by its
    silhouette -- the roofline against the sky for the vertical shift, the
    building corner against the sky for the horizontal one -- using the sky
    boundary in each panorama's skyline file, when it has one.

Grids
-----
rect            a planar quad: origin (bottom-left), u (along), v (up), n (out)
patch lattice   PATCH_M along each axis, snapped so the patches tile the rect
chip grid       CHIP_TX texels over the rect padded by PAD_M, for alignment
fine grid       per-rect texel size from the 25th percentile view range
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import opencity_paths as OCP                    # noqa: E402  host roots, env-overridable

W     = OCP.CODE
SCR   = OCP.SHARED + '/texture'
WP    = SCR + '/wallphoto'
LOGS  = W + '/logs'
FAC   = SCR + '/facades'
IMD   = OCP.SOURCE + '/panos/images_pano'

# per-area paths when an area is set (OPENCITY_AREA set and its area.yaml has
# paths.scratch_dir; pipeline/areacfg.module_paths); otherwise the defaults above.
_AP = None
if os.environ.get('OPENCITY_AREA', '').strip():
    import common as _C
    _AP = _C._areacfg.module_paths(_C) if hasattr(_C, '_areacfg') else None
if _AP:
    W = _AP['W']; SCR = _AP['SCR']; WP = SCR + '/wallphoto'; LOGS = W + '/logs'
    FAC = SCR + '/facades'; IMD = _AP['IMD']

for d in (WP, LOGS):
    os.makedirs(d, exist_ok=True)

# ---- the view gate -------------------------------------------------------- #
R_MIN     = 2.0        # m; nearer than this the rig itself is in the way
R_MAX     = 25.0       # m; outer range gate
# A wall no panorama passes inside 25 m is not better served by a material
# tile: at 35 m a 5760 px equirect is 3.8 cm on the wall and an 11000 px one is
# 2.0 cm, both finer than the 8 cm stage 35 fallback.  So walls with no closer
# view get their own, longer, gate; every wall that HAS a view inside 25 m
# keeps 25 m, because a far view of a wall that also has a near one only adds
# blur and parallax.  Carried per rect in rect.meta['r_max'].
R_MAX_FAR = 40.0
MIN_COS   = 0.30       # front-facing enough to carry texture (72 deg)
if _AP:   # stage_knobs.FACADE_PHOTO_QUALITY.min_cos, when the area sets it
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'pipeline'))
    import facade_quality as _FQ
    MIN_COS = _FQ.min_cos(MIN_COS)
PATCH_M   = 1.5        # m, one panorama per patch
PAD_M     = 1.0        # m of rect padding the chips carry, for the 2D search
CHIP_TX   = 0.04       # m per texel on the alignment chips
SEARCH_M  = 1.0        # m, the 2D alignment search radius
NCC_MIN   = 0.35       # a shift has to correlate this well to be believed
FEATHER_M = 0.12       # m, the blend band between two patches' owners
NATIVE_MAX = 11000     # decode width cap for the bake

# fine texel size: 1.8 mm per metre of range, i.e. 1.8 cm at 10 m, which is
# just over one pixel of a 5760 px panorama at that range (1.09 cm) and half a
# pixel of an 11000 px one.  Clamped, then coarsened for walls far from the
# area centre the way stage 34 does, then capped so no tile exceeds MAX_DIM.
TEX_PER_M = 0.0018
TEX_MIN, TEX_MAX = 0.012, 0.045
FAR_R = 120.0
MAX_DIM = 2048

LUMW = np.array([0.2126, 0.7152, 0.0722], 'f4')


def log(msg, name='wallphoto'):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(f'{LOGS}/{name}.log', 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass


def setdir(setname):
    d = f'{WP}/{setname}'
    for sub in ('', '/chips', '/tiles', '/work', '/markers'):
        os.makedirs(d + sub, exist_ok=True)
    return d


def mark(setname, name, info=None):
    with open(f'{setdir(setname)}/markers/{name}.done', 'w') as f:
        json.dump(info or {'at': time.strftime('%Y-%m-%dT%H:%M:%S')}, f)


def have(setname, name):
    return os.path.exists(f'{setdir(setname)}/markers/{name}.done')


# ---- rects ---------------------------------------------------------------- #
class Rect:
    """A planar quad with an in-plane basis.  u along the surface, v straight
    up, n outward; origin is the bottom-left corner in that basis."""

    __slots__ = ('id', 'origin', 'u', 'n', 'width', 'height', 'meta')

    def __init__(self, rid, origin, u, n, width, height, meta=None):
        self.id = int(rid)
        self.origin = np.asarray(origin, 'f8')
        self.u = np.asarray(u, 'f8')
        self.n = np.asarray(n, 'f8')
        self.width = float(width)
        self.height = float(height)
        self.meta = meta or {}

    @property
    def v(self):
        return np.array([0.0, 0.0, 1.0])

    @property
    def centre(self):
        return self.origin + self.u * (self.width / 2) + self.v * (self.height / 2)

    def points(self, uu, vv, lift=0.06):
        """(u, v) in rect metres -> world points, lifted off the plane."""
        uu = np.asarray(uu, 'f8'); vv = np.asarray(vv, 'f8')
        return (self.origin[None, :] + self.u[None, :] * uu[:, None] +
                np.array([0., 0., 1.])[None, :] * vv[:, None] +
                self.n[None, :] * lift)

    def to_dict(self):
        return dict(id=self.id, origin=[float(x) for x in self.origin],
                    u=[float(x) for x in self.u], n=[float(x) for x in self.n],
                    width_m=self.width, height_m=self.height, meta=self.meta)

    @staticmethod
    def from_dict(d):
        return Rect(d['id'], d['origin'], d['u'], d['n'], d['width_m'],
                    d['height_m'], d.get('meta'))


def walls_as_rects(path=None):
    """s33's building walls."""
    doc = json.load(open(path or (FAC + '/walls.json')))
    out = []
    for w in doc['walls']:
        out.append(Rect(w['id'], w['origin'], w['u'], w['normal'],
                        w['width_m'], w['height_m'],
                        dict(part=w['part'], part_name=w['part_name'],
                             osm_id=w['osm_id'], area_m2=w['area_m2'],
                             buried=w['buried'], osm=w['osm'],
                             faces=w['faces'])))
    return out


def load_rects(setname):
    d = json.load(open(f'{setdir(setname)}/rects.json'))
    return [Rect.from_dict(r) for r in d['rects']], d


def save_rects(setname, rects, extra=None):
    json.dump(dict(n=len(rects), rects=[r.to_dict() for r in rects],
                   **(extra or {})),
              open(f'{setdir(setname)}/rects.json', 'w'), indent=1)


# ---- patch lattice -------------------------------------------------------- #
def rect_rmax(rect):
    """The range gate this rect's views are held to (see R_MAX_FAR)."""
    try:
        return float(rect.meta.get('r_max', R_MAX))
    except Exception:
        return R_MAX


def patch_shape(rect, pm=PATCH_M):
    npu = max(1, int(round(rect.width / pm)))
    npv = max(1, int(round(rect.height / pm)))
    return npu, npv


def patch_of_uv(rect, uu, vv, npu=None, npv=None):
    if npu is None:
        npu, npv = patch_shape(rect)
    pu = np.clip((np.asarray(uu) / rect.width * npu).astype(np.int32), 0, npu - 1)
    pv = np.clip((np.asarray(vv) / rect.height * npv).astype(np.int32), 0, npv - 1)
    return pv * npu + pu


# ---- texel sizes ---------------------------------------------------------- #
def fine_texel(rect, dq, cap_m=None):
    """Texel size for this rect's photo tile, in metres.

    It follows the range: 1.8 mm per metre of view range, so a wall seen from
    35 m is baked at 4.5 cm (the clamp) instead of the 1.8 cm a wall seen from
    10 m gets.  `cap_m` is the texel size the material tile this photograph
    replaces already has: a photographed wall must never come out COARSER than
    the tile it displaces, which the distance-from-the-centre coarsening below
    would otherwise do on the far walls the 40 m gate brings in (up to 9.4 cm
    against an 8.0 cm material tile)."""
    t = float(np.clip(TEX_PER_M * dq, TEX_MIN, TEX_MAX))
    r = float(np.hypot(rect.centre[0], rect.centre[1]))
    t *= 1.0 + max(0.0, (r - FAR_R) / FAR_R)
    if cap_m:
        t = min(t, float(cap_m))
    return max(t, max(rect.width, rect.height) / MAX_DIM)


def chip_texel(rect):
    return max(CHIP_TX, max(rect.width + 2 * PAD_M,
                            rect.height + 2 * PAD_M) / 1400.0)


def grid_axes(rect, tx, pad=0.0):
    nu = max(2, int(round((rect.width + 2 * pad) / tx)))
    nv = max(2, int(round((rect.height + 2 * pad) / tx)))
    uu = -pad + (np.arange(nu) + 0.5) * ((rect.width + 2 * pad) / nu)
    vv = -pad + (np.arange(nv) + 0.5) * ((rect.height + 2 * pad) / nv)
    return uu, vv


# ---- colour --------------------------------------------------------------- #
_SRGB = (np.linspace(0, 1, 256) ** 2.2).astype('f4')


def to_linear(u8):
    return _SRGB[np.asarray(u8, 'u1')]


def to_srgb(lin):
    return np.clip(np.power(np.clip(lin, 0, 1), 1 / 2.2) * 255.0, 0, 255).astype('u1')


def luma(rgb_u8):
    return (np.asarray(rgb_u8, 'f4') @ LUMW)


def green_veto(rgb_u8):
    """Foliage in front of a wall that the lidar DSM did not carry.  Stage 35's
    test, on 8-bit sRGB: green clearly above both other channels."""
    f = np.asarray(rgb_u8, 'f4') / 255.0
    g = f[..., 1]
    return (g - np.maximum(f[..., 0], f[..., 2])) > 0.045


