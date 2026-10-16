"""Aerial pass: objects read off the area's ENU ortho.

No neural network.  The venv has OpenCV but no torch, and a segmentation model
would have to be downloaded, weighted and licence-checked for three classes that
colour and shape priors separate; what is used instead is recorded in
the inventory as method='ortho colour/shape' so nobody mistakes it for a
learned detector.  The priors that make that work:

  cars    only inside an OSM parking polygon or a road corridor; a car is a
          4-14 m2 blob whose colour differs from the local asphalt median, with
          a long axis of 3.2-6.5 m and a short axis of 1.4-2.6 m.
  boats   only over open water (the model's water raster, minus any deck), and
          a boat is a bright blob (hulls and covers are white, water is dark),
          4-200 m2, aspect >= 1.8.
  blobs   on paths and plaza: a compact 0.7-4 m patch whose colour differs from
          the local paving, kept only as a CANDIDATE that another sensor has to
          confirm (planters, kiosks, umbrellas, play equipment).

Everything is measured in ortho pixels then converted to the model's ENU with
the affine in ortho/ortho_enu.json, which the ortho stages already registered
to the model (registration_shift_applied_m is baked into that affine).

  python -m objects.objaerial      -> SCR/aerial_objects.json
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import cv2
import common as C
from objects.objcommon import SCR, enu_of

ORTHO = C.W + '/ortho/ortho_enu.tif'
META = C.W + '/ortho/ortho_enu.json'

ROAD_W = {'secondary': 12.0, 'residential': 8.0, 'service': 5.0,
          'unclassified': 8.0, 'tertiary': 10.0, 'motorway_link': 8.0,
          'primary': 12.0, 'living_street': 6.0}

CAR_AREA = (4.0, 16.0)        # m2 of the filled blob
CAR_LONG = (3.2, 6.8)         # m
CAR_SHORT = (1.3, 2.8)        # m
BOAT_AREA = (4.0, 400.0)
BOAT_LONG = (2.5, 40.0)
BLOB_SIZE = (0.7, 4.0)


def _ortho_source():
    """The ortho's source line from ortho_enu.json, else 'area ortho'."""
    try:
        m = json.load(open(META))
    except (OSError, ValueError):
        return 'area ortho'
    return m.get('source') or m.get('credit') or 'area ortho'


def load_ortho():
    import rasterio
    r = rasterio.open(ORTHO)
    a = r.read()
    rgb = np.transpose(a[:3], (1, 2, 0)).copy()
    valid = a[3] > 0
    m = json.load(open(META))
    gsd = float(m['gsd_m']); x0 = m['enu_affine'][2]; y0 = m['enu_affine'][5]
    return rgb, valid, gsd, x0, y0


def to_px(x, y, gsd, x0, y0):
    return (x - x0) / gsd, (y0 - y) / gsd


def to_enu(col, row, gsd, x0, y0):
    return x0 + (col + 0.5) * gsd, y0 - (row + 0.5) * gsd


def osm_masks(shape, gsd, x0, y0):
    """-> (road_or_parking mask, path/plaza mask) as uint8 rasters."""
    els = json.load(open(C.W + '/osm/osm_raw.json'))['elements']
    road = np.zeros(shape, 'u1'); plaza = np.zeros(shape, 'u1')
    for el in els:
        t = el.get('tags') or {}
        g = el.get('geometry') or []
        if el['type'] != 'way' or len(g) < 2:
            continue
        pts = np.array([to_px(*enu_of(p['lat'], p['lon']), gsd, x0, y0)
                        for p in g], 'f4')
        closed = el.get('nodes') and el['nodes'][0] == el['nodes'][-1]
        hw = t.get('highway')
        if t.get('amenity') == 'parking' and closed:
            cv2.fillPoly(road, [pts.round().astype('i4')], 1)
        elif hw in ROAD_W:
            w = int(round(ROAD_W[hw] / gsd))
            cv2.polylines(road, [pts.round().astype('i4')], False, 1, w)
        elif hw in ('footway', 'pedestrian', 'path', 'cycleway', 'steps'):
            if closed and t.get('area') == 'yes':
                cv2.fillPoly(plaza, [pts.round().astype('i4')], 1)
            else:
                cv2.polylines(plaza, [pts.round().astype('i4')], False, 1,
                              int(round(4.0 / gsd)))
    return road, plaza


def water_mask(shape, gsd, x0, y0):
    """Open water from the model's own floor rasters (0.5 m), upsampled."""
    import s80_furniture_inventory as S
    surf = S.Surface()
    gx0, gy0, c, nx, ny = surf.grid
    wet = np.isfinite(surf.z_water) & ~np.isfinite(surf.z_structures) \
        & ~np.isfinite(surf.z_paths) & ~np.isfinite(surf.z_ground_terrain)
    rows = np.arange(shape[0]); cols = np.arange(shape[1])
    X = x0 + (cols + 0.5) * gsd
    Y = y0 - (rows + 0.5) * gsd
    j = np.clip(((X - gx0) / c).astype('i4'), 0, int(nx) - 1)
    i = np.clip(((Y - gy0) / c).astype('i4'), 0, int(ny) - 1)
    return wet[np.ix_(i, j)].astype('u1'), surf


def blobs(mask, min_px, max_px):
    n, lab, stats, cen = cv2.connectedComponentsWithStats(mask.astype('u1'), 8)
    out = []
    for k in range(1, n):
        a = stats[k, cv2.CC_STAT_AREA]
        if a < min_px or a > max_px:
            continue
        out.append((k, lab, stats[k], cen[k], a))
    return out, lab


def pixels_of(lab, k, st):
    """Component pixels, found inside the component's own bounding box; a
    full-image scan per blob costs 33 Mpx each and there are thousands."""
    x0, y0 = int(st[cv2.CC_STAT_LEFT]), int(st[cv2.CC_STAT_TOP])
    w, h = int(st[cv2.CC_STAT_WIDTH]), int(st[cv2.CC_STAT_HEIGHT])
    sub = lab[y0:y0 + h, x0:x0 + w] == k
    ys, xs = np.nonzero(sub)
    return ys + y0, xs + x0


def rect_of(lab, k, st):
    ys, xs = pixels_of(lab, k, st)
    pts = np.stack([xs, ys], 1).astype('f4')
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    return cx, cy, w, h, ang, pts


def detect_cars(rgb, road, gsd, x0, y0):
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype('f4')
    sat = (rgb.max(2).astype('f4') - rgb.min(2).astype('f4'))
    # local asphalt: a wide median, computed on a 4x downsample for speed
    sml = cv2.resize(L, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(sml.astype('u1'), 31).astype('f4')
    bg = cv2.resize(bg, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
    d = np.abs(L - bg)
    m = ((d > 18) | (sat > 45)) & (road > 0)
    m = cv2.morphologyEx(m.astype('u1'), cv2.MORPH_CLOSE,
                         np.ones((9, 9), 'u1'))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), 'u1'))
    px2 = gsd * gsd
    cand, lab = blobs(m, CAR_AREA[0] / px2, CAR_AREA[1] / px2 * 3)
    out = []
    for k, lab_, st, cen, a in cand:
        cx, cy, w, h, ang, pts = rect_of(lab, k, st)
        lo, sh = max(w, h) * gsd, min(w, h) * gsd
        area = a * px2
        if not (CAR_LONG[0] <= lo <= CAR_LONG[1] and CAR_SHORT[0] <= sh <= CAR_SHORT[1]):
            continue
        if not (CAR_AREA[0] <= area <= CAR_AREA[1]):
            continue
        fill = area / max(lo * sh, 1e-6)
        if fill < 0.55:
            continue
        ys, xs = pixels_of(lab, k, st)
        col = rgb[ys, xs].mean(0)
        # ENU yaw of the long axis: ortho rows run -y, so negate the angle
        yaw = (-(ang if w >= h else ang + 90.0)) % 180.0
        ex, ey = to_enu(cx, cy, gsd, x0, y0)
        out.append(dict(cls='car', x=ex, y=ey, length_m=round(lo, 2),
                        width_m=round(sh, 2), area_m2=round(area, 2),
                        yaw_deg=round(float(yaw), 1), fill=round(float(fill), 2),
                        rgb=[int(v) for v in col],
                        contrast=round(float(d[ys, xs].mean()), 1)))
    return out


def detect_boats(rgb, wet, gsd, x0, y0):
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype('f4')
    w = wet > 0
    if w.sum() < 100:
        return []
    med = float(np.median(L[w])); mad = float(np.median(np.abs(L[w] - med))) + 1e-6
    m = (L > med + max(12.0, 4.0 * mad)) & w
    m = cv2.morphologyEx(m.astype('u1'), cv2.MORPH_CLOSE, np.ones((7, 7), 'u1'))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), 'u1'))
    px2 = gsd * gsd
    cand, lab = blobs(m, BOAT_AREA[0] / px2, BOAT_AREA[1] / px2)
    out = []
    for k, lab_, st, cen, a in cand:
        cx, cy, ww, hh, ang, pts = rect_of(lab, k, st)
        lo, sh = max(ww, hh) * gsd, min(ww, hh) * gsd
        area = a * px2
        if lo < BOAT_LONG[0] or lo > BOAT_LONG[1] or sh < 0.8:
            continue
        if lo / max(sh, 1e-6) < 1.6:
            continue
        ys, xs = pixels_of(lab, k, st)
        col = rgb[ys, xs].mean(0)
        yaw = (-(ang if ww >= hh else ang + 90.0)) % 180.0
        ex, ey = to_enu(cx, cy, gsd, x0, y0)
        out.append(dict(cls='boat', x=ex, y=ey, length_m=round(lo, 2),
                        width_m=round(sh, 2), area_m2=round(area, 2),
                        yaw_deg=round(float(yaw), 1),
                        rgb=[int(v) for v in col],
                        brightness=round(float(L[ys, xs].mean() - med), 1)))
    return out


def detect_blobs(rgb, plaza, gsd, x0, y0):
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype('f4')
    sat = (rgb.max(2).astype('f4') - rgb.min(2).astype('f4'))
    sml = cv2.resize(L, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(sml.astype('u1'), 21).astype('f4'),
                    (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
    d = np.abs(L - bg)
    m = ((d > 22) | (sat > 55)) & (plaza > 0)
    m = cv2.morphologyEx(m.astype('u1'), cv2.MORPH_OPEN, np.ones((5, 5), 'u1'))
    px2 = gsd * gsd
    cand, lab = blobs(m, (BLOB_SIZE[0] ** 2) / px2, (BLOB_SIZE[1] ** 2) / px2)
    out = []
    for k, lab_, st, cen, a in cand:
        cx, cy, ww, hh, ang, pts = rect_of(lab, k, st)
        lo, sh = max(ww, hh) * gsd, min(ww, hh) * gsd
        if lo > BLOB_SIZE[1] or sh < BLOB_SIZE[0] or lo / max(sh, 1e-6) > 2.5:
            continue
        ys, xs = pixels_of(lab, k, st)
        col = rgb[ys, xs].mean(0)
        ex, ey = to_enu(cx, cy, gsd, x0, y0)
        out.append(dict(cls='blob', x=ex, y=ey, length_m=round(lo, 2),
                        width_m=round(sh, 2), area_m2=round(a * px2, 2),
                        yaw_deg=round(float((-(ang if ww >= hh else ang + 90.0)) % 180.0), 1),
                        rgb=[int(v) for v in col],
                        contrast=round(float(d[ys, xs].mean()), 1)))
    return out


def main():
    t0 = time.time()
    rgb, valid, gsd, x0, y0 = load_ortho()
    print('ortho %dx%d gsd %.4f m' % (rgb.shape[1], rgb.shape[0], gsd), flush=True)
    road, plaza = osm_masks(rgb.shape[:2], gsd, x0, y0)
    wet, surf = water_mask(rgb.shape[:2], gsd, x0, y0)
    road &= valid.astype('u1'); plaza &= valid.astype('u1'); wet &= valid.astype('u1')
    print('masks: road/parking %.0f m2, plaza/path %.0f m2, water %.0f m2'
          % (road.sum() * gsd * gsd, plaza.sum() * gsd * gsd, wet.sum() * gsd * gsd),
          flush=True)
    cars = detect_cars(rgb, road, gsd, x0, y0)
    boats = detect_boats(rgb, wet, gsd, x0, y0)
    bl = detect_blobs(rgb, plaza, gsd, x0, y0)
    out = dict(method='ortho colour/shape (OpenCV %s), no learned model' % cv2.__version__,
               ortho=dict(file=ORTHO, gsd_m=gsd,
                          source=_ortho_source()),
               mask_area_m2=dict(road=float(road.sum() * gsd * gsd),
                                 plaza=float(plaza.sum() * gsd * gsd),
                                 water=float(wet.sum() * gsd * gsd)),
               cars=cars, boats=boats, blobs=bl,
               seconds=round(time.time() - t0, 1))
    json.dump(out, open(SCR + '/aerial_objects.json', 'w'), indent=1)
    print('aerial: %d cars, %d boats, %d blobs, %.0fs'
          % (len(cars), len(boats), len(bl), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()


def enu_mask(cell=0.5, pad=5.0):
    """The same road/parking and path/plaza masks, rasterised on a coarse ENU
    grid so the fuse stage can ask 'is this point on a road?' cheaply.

    -> (road, plaza, (x0, y0, cell)) with row 0 at the TOP (y decreasing)."""
    m = json.load(open(META))
    ex = m['enu_extent']
    x0 = ex['xmin'] - pad; y1 = ex['ymax'] + pad
    nx = int(math.ceil((ex['xmax'] + pad - x0) / cell))
    ny = int(math.ceil((y1 - (ex['ymin'] - pad)) / cell))
    road, plaza = osm_masks((ny, nx), cell, x0, y1)
    return road, plaza, (x0, y1, cell)


def mask_at(mask, grid, x, y):
    x0, y1, cell = grid
    j = int((x - x0) / cell); i = int((y1 - y) / cell)
    if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]:
        return bool(mask[i, j])
    return False


def ortho_colour(rgb, gsd, x0, y0, x, y, r=0.6):
    """Median RGB of the ortho inside radius r of an ENU point."""
    c0 = int((x - x0) / gsd); r0 = int((y0 - y) / gsd)
    k = max(1, int(r / gsd))
    sub = rgb[max(0, r0 - k):r0 + k + 1, max(0, c0 - k):c0 + k + 1]
    if sub.size == 0:
        return None
    med = np.median(sub.reshape(-1, 3), axis=0)
    return [int(v) for v in med]
