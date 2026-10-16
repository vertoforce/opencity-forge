"""Shared helpers for the ground polish passes.

Everything here works on the 0.25 m world-XY tint grid that stage 22 samples, so a class map, a tint raster and a mask are all the same
array shape and can be indexed together.
"""
import os
try:
    import opencity_paths as OCP                # host roots, env-overridable
except ImportError:                            # imported with lib/ off sys.path
    import sys
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import opencity_paths as OCP
import numpy as np
import texlib as T
from s21_material import MATERIALS

SCR = T.SCR
FLOOR = SCR + '/floor'
ORTHO = SCR + '/ortho'
PJ = SCR + '/project'
MT = SCR + '/material_ortho'
POL = SCR + '/polish'
os.makedirs(POL, exist_ok=True)
W = OCP.CODE

# per-area paths when an area is set (OPENCITY_AREA set and its area.yaml has
# paths.scratch_dir; pipeline/areacfg.module_paths); otherwise the defaults above.
_AP = None
if os.environ.get('OPENCITY_AREA', '').strip():
    import common as _C
    _AP = _C._areacfg.module_paths(_C) if hasattr(_C, '_areacfg') else None
if _AP:
    W = _AP['W']


def grid():
    d = np.load(FLOOR + '/tint_map_blend.npz')
    x0, y0, c, nx, ny = [float(v) for v in d['grid']]
    return x0, y0, c, int(nx), int(ny)


def rasterise_tris(tri, vals, nx, ny, gx0, gy0, cell, acc=None, den=None):
    """Area-rasterise triangles into (ny,nx,K) accumulators. vals: (n,K)."""
    K = vals.shape[1]
    if acc is None:
        acc = np.zeros((ny, nx, K), 'f8')
    if den is None:
        den = np.zeros((ny, nx), 'f8')
    wcell = cell * cell
    j0 = np.clip(np.floor((tri[:, :, 0].min(1) - gx0) / cell).astype(int), 0, nx - 1)
    j1 = np.clip(np.ceil((tri[:, :, 0].max(1) - gx0) / cell).astype(int), 0, nx - 1)
    i0 = np.clip(np.floor((tri[:, :, 1].min(1) - gy0) / cell).astype(int), 0, ny - 1)
    i1 = np.clip(np.ceil((tri[:, :, 1].max(1) - gy0) / cell).astype(int), 0, ny - 1)
    for k in range(len(tri)):
        jj = np.arange(j0[k], j1[k] + 1); ii = np.arange(i0[k], i1[k] + 1)
        if not len(jj) or not len(ii):
            continue
        X, Y = np.meshgrid(jj * cell + gx0 + cell / 2, ii * cell + gy0 + cell / 2)
        ax_, ay_ = tri[k, 0, 0], tri[k, 0, 1]
        v0x = tri[k, 1, 0] - ax_; v0y = tri[k, 1, 1] - ay_
        v1x = tri[k, 2, 0] - ax_; v1y = tri[k, 2, 1] - ay_
        dn = v0x * v1y - v1x * v0y
        if abs(dn) < 1e-12:
            continue
        px = X - ax_; py = Y - ay_
        u = (px * v1y - v1x * py) / dn
        v = (v0x * py - px * v0y) / dn
        m = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        if not m.any():
            continue
        J, I = np.meshgrid(jj, ii)
        jm = J[m]; im = I[m]
        for q in range(K):
            acc[im, jm, q] += vals[k, q] * wcell
        den[im, jm] += wcell
    return acc, den


def class_map(force=False):
    """Per-cell material class of the TOP surface, 0.25 m, int8 (-1 = none).

    Priority: water plane and decks and path ribbons sit above the terrain, and
    the terrain sits above any photo ground, so a cell
    takes the class of the highest layer that covers it.  That is what a viewer
    looking straight down sees, and it is the boundary the tint must be crisp
    across."""
    p = POL + '/class_map.npz'
    if os.path.exists(p) and not force:
        d = np.load(p)
        return d['cls'], d['src']
    gx0, gy0, cell, nx, ny = grid()
    cls = np.full((ny, nx), -1, 'i1')
    src = np.full((ny, nx), -1, 'i1')          # 0 terrain 1 paths 2 structures 3 water 4 photo
    K = len(MATERIALS)
    order = [('photo', 4), ('ground_terrain', 0), ('water', 3),
             ('structures', 2), ('paths', 1)]
    for name, sid in order:
        mm = np.load(MT + f'/{name}_mat.npz')
        kill = mm['kill'].copy()
        # the stages that delete faces after s21 have to be visible here, or the
        # class map claims a floor that the model does not carry
        extra = {'photo': [FLOOR + '/kill_extra.npy', POL + '/kill_scraps.npy',
                           SCR + '/facades/kill_facade.npy'],
                 'water': [POL + '/kill_layer_water.npy']}.get(name, [])
        for kp in extra:
            if os.path.exists(kp):
                kx = np.load(kp)
                if len(kx) >= len(kill):
                    kill |= kx[:len(kill)]
        sel = (~kill) & (mm['up'] > 0.70)
        if name == 'photo':
            sel &= (mm['sublayer'] == 1)
            P = np.load(W + '/geom/pos.npy', mmap_mode='r') if os.path.exists(W + '/geom/pos.npy') else None
            import common as C
            P = np.load(C.G + '/pos.npy', mmap_mode='r')
            F = np.load(C.G + '/faces.npy', mmap_mode='r')
            ix = np.nonzero(sel)[0]
            # in blocks, so a large mesh stays within memory
            acc = np.zeros((ny, nx, K), 'f8'); den = np.zeros((ny, nx), 'f8')
            mat = mm['mat']
            for s in range(0, len(ix), 400000):
                b = ix[s:s + 400000]
                t = np.asarray(P[np.asarray(F[b])], 'f8')
                cen = t.mean(1)
                ar = 0.5 * np.abs(np.cross(t[:, 1, :2] - t[:, 0, :2],
                                           t[:, 2, :2] - t[:, 0, :2]))
                j = np.clip(((cen[:, 0] - gx0) / cell).astype(int), 0, nx - 1)
                i = np.clip(((cen[:, 1] - gy0) / cell).astype(int), 0, ny - 1)
                flat = i * nx + j
                for q in range(K):
                    w = np.where(mat[b] == q, ar, 0.0)
                    if w.any():
                        acc[:, :, q] += np.bincount(flat, weights=w,
                                                    minlength=nx * ny).reshape(ny, nx)
                den += np.bincount(flat, weights=ar, minlength=nx * ny).reshape(ny, nx)
        else:
            import common as C
            z = np.load(C.W + f'/layers/{name}.npz')
            t = z['pos'].astype('f8')[z['idx']]
            mat = mm['mat']
            ix = np.nonzero(sel)[0]
            vals = np.zeros((len(ix), K), 'f8')
            vals[np.arange(len(ix)), np.clip(mat[ix], 0, K - 1)] = 1.0
            acc, den = rasterise_tris(t[ix], vals, nx, ny, gx0, gy0, cell)
        hit = den > 1e-9
        best = np.argmax(acc, axis=2).astype('i1')
        cls = np.where(hit, best, cls)
        src = np.where(hit, np.int8(sid), src)
        print(f"  class map {name}: {int(hit.sum()):,} cells", flush=True)
    np.savez_compressed(p, cls=cls, src=src)
    return cls, src


