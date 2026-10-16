"""Stage 50 -- the class-confined world tint.

The ground tint raster (pipeline/nomesh_ground.py tint) is one colour raster for
the whole area, smoothed with a 0.5 m Gaussian, with the ortho blended in over a
0.5 m ownership transition.  Both filters run straight across the curb, so a path's colour
leaks into the lawn and the lawn's into the path: measured, the 10-90 %
luminance hand-over across a lawn/paving boundary spans about 1.3 m, which is
why paths fade into lawn instead of ending at a curb.

This stage cuts the raster into one layer per material class:

  * a cell belongs to the class of the TOP surface that covers it (the path
    ribbon above the terrain, the deck above the water), so the class map is
    what a viewer looking down sees;
  * every filter -- outlier clamp, smoothing, the ortho hand-over -- runs as a
    normalised convolution INSIDE one class mask, so no colour crosses a class
    boundary;
  * cells outside a class are filled with that class's nearest in-class value,
    so a vertex sampled a few centimetres on the wrong side of the boundary
    still reads its own class's colour;
  * near a boundary the ortho (crisp, finer than the panorama tint) takes the cell from the panorama
    tint (blurred by construction) wherever the ortho is usable.

Per class the smoothing scale is set by what the material is meant to carry:
the lawn keeps only low-frequency colour, because its high frequency is the
grass texture's job, while paving keeps detail down to the cell.

Writes <scr>/floor/tint_class.npz, which stage 22 samples per material class.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
from scipy import ndimage
import texlib as T
import common as C
import polish_lib as P
from s21_material import MATERIALS

FLOOR = T.SCR + '/floor'
ORTHO = T.SCR + '/ortho'
OUT = FLOOR + '/tint_class.npz'

# metres of Gaussian smoothing inside the class, and how many local sigmas of
# luminance a cell may sit away from its class's local median before it is
# pulled back.  A blob of tree crown or a person's shadow painted onto the lawn
# is exactly such an outlier.
SIGMA_M = dict(lawn=1.10, sand=0.70, gravel=0.50, asphalt=0.40, paving=0.30,
               concrete=0.30, wood_deck=0.30, metal=0.30, brick=0.30, water=2.00)
CLAMP_K = dict(lawn=1.10, sand=1.60, gravel=1.80, asphalt=2.00, paving=2.20,
               concrete=2.20, wood_deck=2.20, metal=2.50, brick=2.50, water=1.00)
# the same clamp again on colour rather than brightness: how many local sigmas
# of chroma a cell may sit from its class's local colour.  A salmon-brown smear
# on a lawn is a tint outlier in exactly this sense, with no object under it to
# justify it.
CHROMA_K = dict(lawn=1.10, sand=1.60, gravel=1.80, asphalt=1.80, paving=2.00,
                concrete=2.00, wood_deck=2.00, metal=2.50, brick=2.50, water=1.00)
BND_M = 1.00          # within this of another class the ortho takes over
BND_OWN = 0.85
# ... except on water: within a metre of the shore the ortho pixel is dock,
# rip-rap or moored hull, and handing those cells to the ortho paints a
# shattered bright band along the shoreline.
BND_SKIP = ('water',)
WATER_SHORE_ERODE_M = 1.00
# the shoreline feather: over this distance inward from the land the water's
# own tint hands over to the tint of the ground it meets, so the water ends in a
# wet margin instead of the hard colour step the eye-level views show.  The
# ortho still does not own these cells (BND_SKIP), so the shattered bright band
# does not come back: what is blended in is the smoothed land raster.
WATER_FEATHER_M = 0.50
WATER_FEATHER_MAX = 0.75     # blend weight in the outermost water cell
MIN_CELLS = 400


def masked_gauss(A, m, sigma_px):
    """Normalised convolution: the average of the in-mask cells only."""
    mf = ndimage.gaussian_filter(m.astype('f4'), sigma_px, mode='nearest')
    if A.ndim == 2:
        af = ndimage.gaussian_filter(np.where(m, A, 0).astype('f4'), sigma_px,
                                     mode='nearest')
        return np.where(mf > 1e-6, af / np.maximum(mf, 1e-6), 0.0)
    out = np.empty_like(A)
    for k in range(A.shape[2]):
        af = ndimage.gaussian_filter(np.where(m, A[:, :, k], 0).astype('f4'),
                                     sigma_px, mode='nearest')
        out[:, :, k] = np.where(mf > 1e-6, af / np.maximum(mf, 1e-6), 0.0)
    return out


def main():
    t0 = time.time()
    gx0, gy0, cell, nx, ny = P.grid()
    cls, src = P.class_map()
    Mp = np.load(FLOOR + '/tint_map.npz')['M'].astype('f4')
    Mb = np.load(FLOOR + '/tint_map_blend.npz')['M'].astype('f4')
    R = np.load(ORTHO + '/rasters.npz')
    Mo = R['Mo'].astype('f4'); ovalid = R['ovalid']; own = R['own'].astype('f4')
    Mo_use = np.where(ovalid[:, :, None], Mo, Mp)

    # boundary cells: a different class within BND_M
    lab = cls.astype('i2')
    dif = np.zeros(cls.shape, bool)
    r = int(round(BND_M / cell))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            s = np.roll(np.roll(lab, dy, 0), dx, 1)
            dif |= (s != lab) & (s >= 0) & (lab >= 0)
    own_b = np.where(dif & ovalid, np.maximum(own, BND_OWN), own).astype('f4')
    Mb2 = ((1 - own_b)[:, :, None] * Mp + own_b[:, :, None] * Mo_use).astype('f4')

    rep = dict(grid=dict(px=[nx, ny], cell_m=cell, origin_enu_m=[gx0, gy0]),
               boundary_cells=int(dif.sum()),
               boundary_ortho_ownership=dict(
                   rule=f"within {BND_M} m of another class the ortho owns at least "
                        f"{BND_OWN:.2f} of the cell wherever it is usable",
                   mean_own_before=round(float(own.mean()), 4),
                   mean_own_after=round(float(own_b.mean()), 4)),
               classes={})
    K = len(MATERIALS)
    stack = np.zeros((K, ny, nx, 3), 'f2')
    present = []
    LUM = np.array([0.2126, 0.7152, 0.0722], 'f4')
    for ci, mname in enumerate(MATERIALS):
        m = cls == ci
        if mname == 'water' and m.any():
            # keep the shore rim out of the water's own statistics and values
            r = int(round(WATER_SHORE_ERODE_M / cell))
            m = ndimage.binary_erosion(m, np.ones((2 * r + 1, 2 * r + 1), bool))
        n = int(m.sum())
        if n < MIN_CELLS:
            stack[ci] = Mb.astype('f2')
            rep['classes'][mname] = dict(cells=n, used_global=True)
            continue
        present.append(mname)
        sg = SIGMA_M[mname] / cell
        A = (Mb if mname in BND_SKIP else Mb2).copy()
        big = max(sg, 3.0 / cell)
        L = A @ LUM
        med = masked_gauss(L, m, big)                          # local class level
        dev = L - med
        sd = np.sqrt(np.maximum(masked_gauss(dev * dev, m, big), 1e-8))
        k = CLAMP_K[mname]
        fac = np.where(np.abs(dev) > k * sd,
                       (med + np.sign(dev) * k * sd) / np.maximum(L, 1e-5), 1.0)
        clamped = int((m & (np.abs(dev) > k * sd)).sum())
        A = A * np.clip(fac, 0.25, 4.0)[:, :, None]
        # the same clamp on colour: hold the ratio RGB/luminance near the class's
        # own local colour, so a cell keeps its brightness but cannot be a
        # different material's hue
        L2 = np.maximum(A @ LUM, 1e-5)
        rat = A / L2[:, :, None]
        cmed = masked_gauss(rat, m, big)
        cdev = rat - cmed
        csd = np.sqrt(np.maximum(masked_gauss(cdev * cdev, m, big), 1e-10))
        kc = CHROMA_K[mname]
        over = np.abs(cdev) > kc * csd
        rat = np.where(over, cmed + np.sign(cdev) * kc * csd, rat)
        chroma_clamped = int((m & over.any(2)).sum())
        A = rat * L2[:, :, None]
        Ssm = masked_gauss(A, m, sg)
        # Fill outside the class by smoothing the class's own values outward,
        # not by nearest neighbour.  A nearest-neighbour fill steps at the
        # Voronoi boundaries between two distant parts of the class, and a big
        # triangle that crosses one of those steps -- the water plane under a
        # dock, a flat terrain triangle under a path -- becomes a vertex-colour
        # fan (a nearest fill turns about 4 % of the floor into fans, against
        # 0.05 % without it).  A cascade of wider
        # and wider normalised convolutions fills every cell with something
        # smooth, so nothing outside the class can step.
        full = Ssm.copy()
        filled = m.copy()
        for sig in (2.0, 6.0, 18.0, 54.0, 162.0):
            if filled.all():
                break
            g = masked_gauss(Ssm, m, sig)
            mf = ndimage.gaussian_filter(m.astype('f4'), sig, mode='nearest')
            ok = (~filled) & (mf > 1e-5)
            full[ok] = g[ok]
            filled |= ok
        if not filled.all():
            idxn = ndimage.distance_transform_edt(~filled, return_distances=False,
                                                  return_indices=True)
            full = full[tuple(idxn)]
        if mname == 'water':
            # hand the last half metre of water over to the land it meets
            wet = cls == ci
            d = ndimage.distance_transform_edt(wet, sampling=cell)
            wgt = (WATER_FEATHER_MAX *
                   np.clip(1.0 - d / WATER_FEATHER_M, 0.0, 1.0)) * wet
            land = ndimage.gaussian_filter(Mb.astype('f4'), (1.0, 1.0, 0),
                                           mode='nearest')
            full = (full * (1 - wgt)[:, :, None] +
                    land * wgt[:, :, None]).astype('f4')
            rep['water_shore_feather'] = dict(
                metres=WATER_FEATHER_M, max_weight=WATER_FEATHER_MAX,
                cells_feathered=int((wgt > 0.01).sum()),
                area_m2=round(float((wgt > 0.01).sum()) * cell * cell, 1),
                blended_with='the ground tint raster, smoothed 0.25 m')
        stack[ci] = full.astype('f2')
        Lb = (Mb[m] @ LUM); Ls = (Ssm[m] @ LUM)
        rep['classes'][mname] = dict(
            cells=n, area_m2=round(n * cell * cell, 1),
            smoothing_sigma_m=SIGMA_M[mname], clamp_k=CLAMP_K[mname],
            cells_clamped=clamped,
            cells_clamped_pct=round(100.0 * clamped / max(n, 1), 2),
            cells_chroma_clamped=chroma_clamped,
            cells_chroma_clamped_pct=round(100.0 * chroma_clamped / max(n, 1), 2),
            luminance_std_before=round(float(Lb.std()), 4),
            luminance_std_after=round(float(Ls.std()), 4),
            mean_abs_grad_before=round(float(grad_in_mask(Mb @ LUM, m)), 5),
            mean_abs_grad_after=round(float(grad_in_mask(Ssm @ LUM, m)), 5))
        print(f"  {mname:<10} {n:>8,} cells  sigma {SIGMA_M[mname]:.2f} m  "
              f"clamped {clamped:,}  {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(OUT, M=stack, grid=np.array([gx0, gy0, cell, nx, ny], 'f8'),
                        materials=np.array(MATERIALS))
    rep['present'] = present
    rep['bytes'] = os.path.getsize(OUT)
    rep['seconds'] = round(time.time() - t0)
    json.dump(rep, open(FLOOR + '/tint_class.json', 'w'), indent=1)
    C.note('class_tint', rep)
    print(json.dumps({k: v for k, v in rep.items() if k != 'classes'}, indent=1))
    print("wrote", OUT, rep['bytes'] // 1024 // 1024, "MB", rep['seconds'], "s")


def grad_in_mask(L, m):
    gy, gx = np.gradient(L.astype('f4'))
    g = np.sqrt(gy * gy + gx * gx)
    mm = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return float(g[mm].mean()) if mm.any() else float('nan')


if __name__ == '__main__':
    main()
