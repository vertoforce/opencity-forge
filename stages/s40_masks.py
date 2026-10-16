"""Stage 40 -- Mapillary per-image detections, cached, and a transient mask
per panorama.

Veto the pixels of people, riders, vehicles and animals before any multi-view
statistic is taken, using the segmentation Mapillary has already computed.

  GET graph.mapillary.com/<image_id>/detections?fields=id,value,geometry
  geometry is a base64 Mapbox vector tile, layer "mpy-or", extent 4096.

The y axis of that tile is up, so an image row is (1 - y/4096) * height: road
polygons land below the equirect horizon and sky polygons above it, and they
swap if the flip is dropped.

Cache      $OPENCITY_DATA/_source/detections/<2>/<id>.json.gz
Masks      .../detections/masks/<2>/<id>.png   2048 x 1024, 255 = transient
Resumable  per image; an image with both files is skipped.
"""
import os, sys, json, gzip, time, base64, random, argparse, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import opencity_paths as OCP                    # noqa: E402  host roots, env-overridable
import numpy as np
from align import alcommon as A

GRAPH = 'https://graph.mapillary.com'
FIELDS = 'id,value,geometry'
DILATE_DEG = 0.35            # mask grows by this much, for segmentation slop


def token():
    return open(OCP.TOKEN_FILE).read().strip()


class Client:
    def __init__(self, tok):
        self.tok = tok
        self.local = threading.local()

    @property
    def s(self):
        import requests
        s = getattr(self.local, 's', None)
        if s is None:
            s = requests.Session()
            s.headers['User-Agent'] = 's40_masks.py/1.0'
            a = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
            s.mount('https://', a)
            self.local.s = s
        return s

    def get(self, url, params, tries=5, timeout=90):
        last = None
        for i in range(tries):
            try:
                r = self.s.get(url, params=params, timeout=timeout)
                if r.status_code == 200:
                    return r
                if r.status_code < 500 and r.status_code != 429:
                    return None if r.status_code == 404 else r.raise_for_status()
                last = f'{r.status_code}'
            except Exception as e:
                last = repr(e)[:120]
            time.sleep(min(30, 2 ** i + random.random()))
        raise RuntimeError(f'GET failed after {tries}: {last}')


def rings(geom):
    """-> list of (n,2) float arrays in tile units, one per polygon ring."""
    out = []

    def walk(c, depth):
        if not len(c):
            return
        if isinstance(c[0], (int, float)):
            return
        if isinstance(c[0][0], (int, float)):
            out.append(np.asarray(c, 'f4'))
            return
        for x in c:
            walk(x, depth + 1)
    walk(geom.get('coordinates', []), 0)
    return out


def rasterise(dets, mw=A.MASK_W, mh=A.MASK_H):
    """-> (mask uint8, n_transient_detections)."""
    import cv2
    import mapbox_vector_tile
    m = np.zeros((mh, mw), 'u1')
    n = 0
    for d in dets:
        v = d.get('value') or ''
        if not A.is_transient(v):
            continue
        g = d.get('geometry')
        if not g:
            continue
        try:
            dec = mapbox_vector_tile.decode(base64.decodebytes(g.encode('utf-8')))
        except Exception:
            continue
        for lay in dec.values():
            for ft in lay.get('features', []):
                for r in rings(ft.get('geometry', {})):
                    if len(r) < 3:
                        continue
                    # tile y is up: image row = (1 - y/4096) * height
                    px = np.empty((len(r), 2), 'i4')
                    px[:, 0] = np.clip(r[:, 0] / 4096.0 * mw, 0, mw - 1)
                    px[:, 1] = np.clip((1.0 - r[:, 1] / 4096.0) * mh, 0, mh - 1)
                    cv2.fillPoly(m, [px], 255)
                    n += 1
    if n:
        k = max(1, int(round(DILATE_DEG / 360.0 * mw)) * 2 + 1)
        m = cv2.dilate(m, np.ones((k, k), 'u1'))
    return m, n


def shard_dirs():
    """Pre-create every shard.  On NFS a negative dentry cached by one thread
    survives another thread's mkdir, so creating them lazily loses requests."""
    for a in range(10):
        for b in range(10):
            os.makedirs(f'{A.DET}/{a}{b}', exist_ok=True)
            os.makedirs(f'{A.DETM}/{a}{b}', exist_ok=True)


def one(cl, pid, force=False):
    dp = A.det_path(pid); mp = A.mask_path(pid)
    if not force and os.path.exists(dp) and os.path.exists(mp):
        return None
    if os.path.exists(dp) and not force:
        dets = json.loads(gzip.open(dp, 'rt').read()).get('data', [])
    else:
        r = cl.get(f'{GRAPH}/{pid}/detections',
                   params={'access_token': cl.tok, 'fields': FIELDS, 'limit': '2000'})
        doc = r.json() if r is not None else {'data': []}
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        tmp = dp + '.tmp'
        with gzip.open(tmp, 'wt') as f:
            json.dump(doc, f)
        os.replace(tmp, dp)
        dets = doc.get('data', [])
    m, nt = rasterise(dets)
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    from PIL import Image
    tmp = mp + '.tmp.png'
    Image.fromarray(m).save(tmp, optimize=True)
    os.replace(tmp, mp)
    return dict(pid=pid, n_det=len(dets), n_transient=nt,
                masked_frac=float((m > 0).mean()),
                masked_frac_lower=float((m[A.MASK_H // 2:] > 0).mean()),
                classes=[d.get('value') for d in dets])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    ids = A.image_ids()
    if a.limit:
        ids = ids[:a.limit]
    A.log(f's40: {len(ids)} images', 'align_masks')
    shard_dirs()
    cl = Client(token())
    from collections import Counter
    cls = Counter(); tcls = Counter()
    fr = []; frl = []; ndet = []; ntr = []
    done = 0; fetched = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, cl, p, a.force): p for p in ids}
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
            except Exception as e:
                A.log(f'  {futs[f]} failed: {repr(e)[:160]}', 'align_masks')
                continue
            if r is None:
                continue
            fetched += 1
            for v in r['classes']:
                cls[v] += 1
                if A.is_transient(v):
                    tcls[v] += 1
            fr.append(r['masked_frac']); frl.append(r['masked_frac_lower'])
            ndet.append(r['n_det']); ntr.append(r['n_transient'])
            if fetched % 250 == 0:
                A.log(f'  {done}/{len(ids)} images, {fetched} new, '
                      f'{(time.time()-t0)/60:.1f} min', 'align_masks')
    # a pass over every cached mask, so the report covers the whole set even on
    # a resumed run
    tot = 0
    for p in ids:
        mp = A.mask_path(p)
        if os.path.exists(mp):
            tot += 1
    rep = dict(
        images=len(ids), masks_on_disk=tot, newly_processed=fetched,
        detections_per_image_mean=round(float(np.mean(ndet)), 1) if ndet else None,
        transient_polygons_per_image_mean=round(float(np.mean(ntr)), 2) if ntr else None,
        image_area_masked_frac_mean=round(float(np.mean(fr)), 5) if fr else None,
        lower_hemisphere_masked_frac_mean=round(float(np.mean(frl)), 5) if frl else None,
        transient_classes=dict(tcls.most_common(30)),
        top_classes_all=dict(cls.most_common(15)),
        mask_px=[A.MASK_W, A.MASK_H], dilate_deg=DILATE_DEG,
        y_convention='tile y is up; image row = (1 - y/4096) * height, verified '
                     'on road-below-horizon / sky-above-horizon polygons',
        cache=A.DET, masks=A.DETM, seconds=round(time.time() - t0))
    json.dump(rep, open(A.AL + '/masks.json', 'w'), indent=1)
    A.log(f's40 done: {json.dumps(rep)[:400]}', 'align_masks')
    if tot >= len(ids) * 0.98:
        A.mark('masks', rep)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
