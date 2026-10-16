"""Census of the Mapillary detection cache: which object classes exist in this
bbox at all, and how often, before any of them is triangulated.

Reads only the `value` field of every cached detection, so it never decodes a
polygon; that makes the pass cheap enough to run over the whole cache.

  python -m objects.objcensus            -> SCR/census_photo.json
"""
import os, sys, json, gzip, time, collections
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objects.objcommon import SCR, det_path, load_poses
import numpy as np
import common as C


def one(path):
    try:
        d = json.loads(gzip.open(path, 'rt').read()).get('data', [])
    except Exception:
        return collections.Counter(), 0
    c = collections.Counter(x.get('value') or '?' for x in d)
    return c, 1


def scan(paths, workers=8):
    tot = collections.Counter()
    npan = 0
    with ProcessPoolExecutor(workers) as ex:
        for c, n in ex.map(one, paths, chunksize=32):
            tot.update(c); npan += n
    return tot, npan


def main():
    t0 = time.time()
    P = load_poses()
    w, s, e, n = C.BBOX_WSEN
    inb = {k: v for k, v in P.items()
           if -400 < v['x'] < 400 and -400 < v['y'] < 400}
    # bbox in ENU: derive from the model extent used by the rest of the build
    pos = np.load(C.W + '/layers/ground_terrain.npz')['pos']
    x0, x1 = float(pos[:, 0].min()) - 20, float(pos[:, 0].max()) + 20
    y0, y1 = float(pos[:, 1].min()) - 20, float(pos[:, 1].max()) + 20
    inb = {k: v for k, v in P.items()
           if x0 < v['x'] < x1 and y0 < v['y'] < y1}
    print('panos with pose+detections: %d, inside the model bbox: %d'
          % (len(P), len(inb)), flush=True)
    c_in, n_in = scan([det_path(k) for k in inb])
    c_all, n_all = scan([det_path(k) for k in P])
    rows = []
    for v, n in c_all.most_common():
        rows.append(dict(value=v, n_in_bbox=int(c_in.get(v, 0)), n_all=int(n)))
    out = dict(bbox_enu=[x0, y0, x1, y1], panos_all=n_all, panos_in_bbox=n_in,
               detections_in_bbox=int(sum(c_in.values())),
               detections_all=int(sum(c_all.values())),
               classes=rows, seconds=round(time.time() - t0, 1))
    json.dump(out, open(SCR + '/census_photo.json', 'w'), indent=1)
    print('classes: %d, detections in bbox: %d, %.0fs'
          % (len(rows), sum(c_in.values()), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
