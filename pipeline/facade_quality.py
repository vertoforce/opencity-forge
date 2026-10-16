"""Per-texel photo quality for the facade passes.

Configured by stage_knobs.FACADE_PHOTO_QUALITY in an area's area.yaml; when it is absent
(or OPENCITY_AREA is unset) every function here returns None / the caller's own default.

  value: {seg_veto: true, min_cos: 0.5, measure: true}

seg_veto  a panorama pixel Mapillary's own semantic segmentation (stage 40's
          detections cache, already on disk: no model is run) labels as sky,
          vegetation, terrain, water, road / sidewalk / any flat class, curb,
          fence, pole, sign, light, vehicle, person or any other object is not
          wall, so it carries no facade colour.  Kept: construction--structure--*,
          construction--barrier--wall, the void--unlabeled / void--static
          remainder and shop signs / adverts (those hang on the wall).
min_cos   the incidence gate for a texel to carry photo colour (the stock gate is
          0.20 in s34 and 0.30 in wallphoto); a texel seen more edge-on than this
          is smeared along the wall by the pose error.
measure   return the veto hits even when seg_veto is off, so builds with and
          without the veto report the same count.
fallback  {k_views: 4, min_cos: 0.45, unowned: true}: a texel
          its owner panorama cannot carry (vetoed, edge-on, occluded) takes the
          best OTHER correlation-aligned panorama of the wall that passes the same
          veto, ranked per texel by incidence and range, before it falls back to
          material.  k_views: how many non-owner aligned views are sampled per
          wall; unowned: also fill patches nobody owned, except the ones the
          agreement gate dropped for doubling; align_unvetoed: the alignment
          chips keep the not-wall pixels (registration, not colour), so the
          views a vetoed texel needs are aligned in the first place.

Mask cache: <area>/detections/facadeveto/<last 2>/<pid>.png, 2048 x 1024,
255 = not wall.
"""
import os, sys, gzip, json, base64, math
import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib')
if R not in sys.path:
    sys.path.insert(0, R)

KEEP_PREFIX = ('construction--structure--', 'construction--barrier--wall',
               'void--unlabeled', 'void--static', 'object--sign--store',
               'object--sign--advertisement')
DILATE_DEG = 0.25


def knob():
    """The FACADE_PHOTO_QUALITY dict for an area build, else None."""
    import common as C
    if not hasattr(C, '_areacfg') or C._areacfg.build_area_cfg(C) is None:
        return None
    v = ((((getattr(C, '_AREA_CFG', None) or {}).get('stage_knobs') or {})
          .get('FACADE_PHOTO_QUALITY')) or {}).get('value')
    return dict(v) if isinstance(v, dict) else None


def min_cos(default, key='min_cos'):
    """key min_cos: the wallphoto (photo texel) gate; s34_min_cos: stage 34's
    colour/far projection gate (kept at its own default unless set)."""
    k = knob()
    return float(k[key]) if k and k.get(key) is not None else default


def fallback():
    """the FACADE_PHOTO_QUALITY.fallback dict, else None (no area set, or the key
    absent / false: the bake is the plain photo bake byte for byte)."""
    k = knob()
    f = k.get('fallback') if k else None
    if not f:
        return None
    out = dict(k_views=4, min_cos=0.45, unowned=True, align_unvetoed=True)
    if isinstance(f, dict):
        out.update(f)
    return out


def wants_mask():
    k = knob()
    return bool(k and (k.get('seg_veto') or k.get('measure')))


def vetoes():
    k = knob()
    return bool(k and k.get('seg_veto'))


def is_wall(v):
    return any(v.startswith(p) for p in KEEP_PREFIX)


def _paths(pid):
    from align import alcommon as A
    pid = str(pid)
    return (f'{A.DET}/{pid[-2:]}/{pid}.json.gz',
            f'{A.DET}/facadeveto/{pid[-2:]}/{pid}.png')


def rasterise(dets, mw, mh):
    import cv2
    import mapbox_vector_tile
    import s40_masks as S40
    m = np.zeros((mh, mw), 'u1')
    for d in dets:
        v = d.get('value') or ''
        g = d.get('geometry')
        if is_wall(v) or not g:
            continue
        try:
            dec = mapbox_vector_tile.decode(base64.decodebytes(g.encode('utf-8')))
        except Exception:
            continue
        for lay in dec.values():
            for ft in lay.get('features', []):
                # every ring of a feature in ONE fillPoly call, so an interior
                # ring stays a hole (a sky polygon has the skyline's buildings cut
                # out of it; filling ring by ring would fill those holes and veto the
        # buildings as sky, up to ~95 % of a panorama)
                polys = []
                for r in S40.rings(ft.get('geometry', {})):
                    if len(r) < 3:
                        continue
                    px = np.empty((len(r), 2), 'i4')
                    px[:, 0] = np.clip(r[:, 0] / 4096.0 * mw, 0, mw - 1)
                    px[:, 1] = np.clip((1.0 - r[:, 1] / 4096.0) * mh, 0, mh - 1)
                    polys.append(px)
                if polys:
                    cv2.fillPoly(m, polys, 255)
    k = max(1, int(round(DILATE_DEG / 360.0 * mw)) * 2 + 1)
    return cv2.dilate(m, np.ones((k, k), 'u1'))


def load(pid):
    """-> bool (MASK_H, MASK_W) True where the pixel is not wall, or None (no
    knob, or no detections cached for this panorama)."""
    if not wants_mask():
        return None
    from align import alcommon as A
    from PIL import Image
    src, dst = _paths(pid)
    if os.path.exists(dst):
        return np.asarray(Image.open(dst), 'u1') > 0
    if not os.path.exists(src):
        return None
    d = json.load(gzip.open(src))
    dets = d.get('data', []) if isinstance(d, dict) else d
    m = rasterise(dets, A.MASK_W, A.MASK_H)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.%d.tmp.png' % os.getpid()
    Image.fromarray(m).save(tmp); os.replace(tmp, dst)
    return m > 0


def lookup(mask, lon, lat):
    """Mask value at panorama angles (texlib.project's lon/lat)."""
    from align import alcommon as A
    if mask is None:
        return np.zeros(np.shape(lon), bool)
    u = (np.asarray(lon) / (2 * math.pi) + 0.5) * A.MASK_W - 0.5
    v = (0.25 - np.asarray(lat) / (2 * math.pi)) * A.MASK_W - 0.5
    return A.mask_lookup(mask, u, v, A.MASK_W)
