#!/usr/bin/env python3
"""Stage 94: sidewalk panel joints on the ground concrete.

Runs from pipeline/nomesh_swaps.py on the full model, after s22.
For each material in common.GROUND_JOINTS['materials']:
  * the base-colour super tile gets a grid of dark contraction joints and a
    per-panel tone (a whole number of panels per tile period, tileable, mean
    colour per channel kept, so the tint/photo levelling of stage 22 still holds);
  * every primitive's faces are grouped by the orientation of the nearest OSM
    highway (mod 90 deg, Gaussian-weighted, quantised), vertices shared by two
    groups are split, and TEXCOORD_0 = world XY rotated into that group's frame
    / tile period, so the joints run along the street.
Positions, normals and colours are copied unchanged (same points, no crack can
open).  GROUND_JOINTS = None copies the input byte for byte.

    s94_ground_joints.py --in IN.glb --out OUT.glb [--report R.json]
"""
import os, sys, io, json, argparse, shutil
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import common as C
from glblib import GLB
from s38_tree_models import Appender, compact_write

CFG = getattr(C, 'GROUND_JOINTS', None)
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
DT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


def read_acc(g, js, i):
    a = js['accessors'][i]; bv = js['bufferViews'][a['bufferView']]
    if a.get('sparse'):
        raise SystemExit('s94: sparse accessor not supported')
    n = NC[a['type']]; dt = np.dtype(DT[a['componentType']]); es = n * dt.itemsize
    b = bytes(g.bv_bytes(a['bufferView'])); o = a.get('byteOffset', 0); st = bv.get('byteStride') or es
    if st == es:
        return np.frombuffer(b[o:o + a['count'] * es], dt).reshape(a['count'], n).copy(), a
    raw = np.frombuffer(b[o:o + (a['count'] - 1) * st + es].ljust(a['count'] * st, b'\0'), np.uint8).reshape(a['count'], st)[:, :es]
    return np.ascontiguousarray(raw).view(dt).reshape(a['count'], n).copy(), a


def highway_segments():
    d = json.load(open(C.W + '/osm/osm_raw.json'))['elements']
    nodes = {e['id']: (e['lat'], e['lon']) for e in d if e['type'] == 'node'}
    segs = []
    for e in d:
        t = e.get('tags', {})
        if e['type'] != 'way' or 'highway' not in t or t['highway'] == 'steps':
            continue
        ll = np.array([nodes[n] for n in e['nodes'] if n in nodes])
        if len(ll) < 2:
            continue
        xy = C.lla_to_enu(ll[:, 0], ll[:, 1], np.zeros(len(ll)))[:, :2]
        segs.append(np.concatenate([xy[:-1], xy[1:]], 1))
    return np.concatenate(segs, 0)


def orient_field(segs, sig, cell=1.0):
    """(angle raster rad mod pi/2, magnitude raster, grid) of the local street direction:
    each segment splats length x (cos 4a, sin 4a) along itself, Gaussian blur sigma sig."""
    from scipy import ndimage as nd
    lo = segs.reshape(-1, 2).min(0) - 4 * sig; hi = segs.reshape(-1, 2).max(0) + 4 * sig
    nx, ny = (np.ceil((hi - lo) / cell)).astype(int) + 1
    A = np.zeros((2, ny, nx)); Wt = np.zeros((ny, nx))
    for x0, y0, x1, y1 in segs:
        L = np.hypot(x1 - x0, y1 - y0)
        if L < 1e-3:
            continue
        a = np.arctan2(y1 - y0, x1 - x0); k = max(2, int(L / (cell * 0.5)) + 1); t = (np.arange(k) + 0.5) / k
        j = ((x0 + t * (x1 - x0) - lo[0]) / cell).astype(int); i = ((y0 + t * (y1 - y0) - lo[1]) / cell).astype(int)
        np.add.at(A[0], (i, j), np.cos(4 * a) * L / k); np.add.at(A[1], (i, j), np.sin(4 * a) * L / k); np.add.at(Wt, (i, j), L / k)
    A = np.stack([nd.gaussian_filter(A[c], sig / cell) for c in range(2)]); Wt = nd.gaussian_filter(Wt, sig / cell)
    ang = (np.arctan2(A[1], A[0]) / 4) % (np.pi / 2)
    mag = np.hypot(A[0], A[1]) / max(Wt.max(), 1e-9)   # coherent street mass, 1 = densest street cell
    return ang, mag, (lo[0], lo[1], cell, nx, ny)


def edge_field(cls_id, sig, pre=0.5):
    """(angle mod pi/2, strength 0-1, grid) of the material's own outline: gradient of the
    ground class map (<scratch>/polish/class_map.npz from pipeline/nomesh_ground.py, 0.25 m) for class cls_id, as (cos 4a, sin 4a)
    weighted by |g|^2 and Gaussian-blurred over sig; strength = |mean vector| / its p99.5."""
    from scipy import ndimage as nd
    import texlib as T
    import polish_lib as PL
    cm = np.load(T.SCR + '/polish/class_map.npz')['cls']; g0 = PL.grid()
    cell = float(g0[2]); m = nd.gaussian_filter((cm == cls_id).astype('f4'), pre / cell)
    gy = nd.sobel(m, 0); gx = nd.sobel(m, 1); w = gx * gx + gy * gy; a = np.arctan2(gy, gx)
    c = nd.gaussian_filter(w * np.cos(4 * a), sig / cell); s_ = nd.gaussian_filter(w * np.sin(4 * a), sig / cell)
    mag = np.hypot(c, s_)
    return (np.arctan2(s_, c) / 4) % (np.pi / 2), mag / max(float(np.percentile(mag, 99.5)), 1e-12), \
        (float(g0[0]), float(g0[1]), cell, cm.shape[1], cm.shape[0])


def sample(r, grid, x, y):
    x0, y0, cell, nx, ny = grid
    j = np.clip(((x - x0) / cell).astype(int), 0, nx - 1); i = np.clip(((y - y0) / cell).astype(int), 0, ny - 1)
    return r[i, j]


def colour_float(arr, acc):
    if acc['componentType'] == 5126: return arr.astype('f8')
    return arr.astype('f8') / float(np.iinfo(arr.dtype).max)


def colour_back(x, arr, acc):
    if acc['componentType'] == 5126: return x.astype('f4')
    mx = np.iinfo(arr.dtype).max; return np.clip(np.round(x * mx), 0, mx).astype(arr.dtype)


def chroma_calm(prims_xy_rgb, keep, sig, luma_keep=1.0, cell=0.5):
    """Vertex tint chroma (log RGB - log luma) pulled toward its own sig-metre Gaussian
    low-pass over this material's vertices (normalised convolution on a cell raster):
    c' = lp + keep * (c - lp); log luma likewise with luma_keep (1 = unchanged).
    Returns the new linear RGB per prim."""
    from scipy import ndimage as nd
    W = np.array([.2126, .7152, .0722])
    xy = np.concatenate([p[0] for p in prims_xy_rgb]); rgb = np.concatenate([p[1] for p in prims_xy_rgb])
    l = np.maximum(rgb, 1e-4); Y = l @ W; c = np.log(l) - np.log(Y)[:, None]
    lo = xy.min(0) - 4 * sig; j = ((xy[:, 0] - lo[0]) / cell).astype(int); i = ((xy[:, 1] - lo[1]) / cell).astype(int)
    sh = (i.max() + int(8 * sig / cell) + 2, j.max() + int(8 * sig / cell) + 2)
    den = np.zeros(sh); np.add.at(den, (i, j), 1.0); den = nd.gaussian_filter(den, sig / cell)
    lY = np.log(Y); X = np.concatenate([c, lY[:, None]], 1); lp = np.empty_like(X)
    for k in range(4):
        acc = np.zeros(sh); np.add.at(acc, (i, j), X[:, k]); lp[:, k] = nd.gaussian_filter(acc, sig / cell)[i, j] / np.maximum(den[i, j], 1e-9)
    c2 = lp[:, :3] + keep * (c - lp[:, :3]); Y2 = np.exp(lp[:, 3] + luma_keep * (lY - lp[:, 3]))
    e = np.exp(c2); out = e * (Y2 / (e @ W))[:, None]
    stat = dict(chroma_sd_before=round(float(c.std(0).mean()), 4), chroma_sd_after=round(float(c2.std(0).mean()), 4),
                luma_sd_before=round(float(lY.std()), 4), luma_sd_after=round(float(np.log(Y2).std()), 4))
    res, o = [], 0
    for p in prims_xy_rgb:
        res.append(np.clip(out[o:o + len(p[0])], 0, 1)); o += len(p[0])
    return res, stat


def joint_tile(img_bytes, period, cfg):
    """Jointed, panel-toned copy of a tile image (jpeg bytes) -> (jpeg bytes, info)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(io.BytesIO(img_bytes)).convert('RGB'); n = im.size[0]
    tex = period / n
    if tex > cfg['joint_w_m'] / cfg['min_texel_frac']:
        return None, dict(skipped=f'texel {tex * 100:.2f} cm too coarse for {cfg["joint_w_m"] * 100:.1f} cm joints')
    a = np.asarray(im, 'f4') / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    mu0 = lin.reshape(-1, 3).mean(0)
    k = max(1, int(round(period / cfg['panel_m']))); pm = period / k
    ss = 8; u = (np.arange(n * ss) + 0.5) / (n * ss) * period
    dj = np.abs(((u + pm / 2) % pm) - pm / 2)
    prof = np.exp(-cfg['joint_log'] * np.exp(-4 * np.log(2) * (dj / cfg['joint_w_m']) ** 2)).reshape(n, ss).mean(1)
    rng = np.random.default_rng(int(cfg['seed']))
    tone = np.exp(rng.normal(0, cfg['panel_sd'], (k, k)))
    pi = np.minimum((np.arange(n) * tex / pm).astype(int), k - 1)
    f = prof[None, :] * prof[:, None] * tone[pi[:, None], pi[None, :]]
    out = lin * f[:, :, None]
    out *= (mu0 / np.maximum(out.reshape(-1, 3).mean(0), 1e-6))[None, None, :]
    out = np.clip(out, 0, 1); s = np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
    b = io.BytesIO(); Image.fromarray(np.clip(s * 255 + 0.5, 0, 255).astype('u1')).save(b, 'JPEG', quality=int(cfg['jpeg_q']), optimize=True)
    return b.getvalue(), dict(px=n, texel_cm=round(tex * 100, 3), panels_per_tile=k, panel_m=round(pm, 4))


def main():
    ap_ = argparse.ArgumentParser(); ap_.add_argument('--in', dest='inp', required=True); ap_.add_argument('--out', required=True)
    ap_.add_argument('--report', default=None); a = ap_.parse_args()
    if not CFG:
        shutil.copyfile(a.inp, a.out); print('s94: GROUND_JOINTS off, copied'); return
    g = GLB(a.inp); js = json.loads(json.dumps(g.js)); ap = Appender(js)
    segs = highway_segments(); ang, mag, grid = orient_field(segs, float(CFG['orient_sigma_m']))
    import s21_material as _m21
    q = np.radians(float(CFG['orient_q_deg'])); nq = int(round((np.pi / 2) / q))
    rep = dict(config=CFG, segments=int(len(segs)), materials={})
    for mname in CFG['materials']:
        mis = [i for i, m in enumerate(js['materials']) if m.get('name') == mname]
        if not mis:
            rep['materials'][mname] = 'absent'; continue
        mi = mis[0]; m = js['materials'][mi]
        period = float(m.get('extras', {}).get('metres_per_tile') or 0)
        bt = m.get('pbrMetallicRoughness', {}).get('baseColorTexture')
        if not period or bt is None:
            rep['materials'][mname] = 'no period/base texture'; continue
        tx = js['textures'][bt['index']]
        if 'source' not in tx:
            raise SystemExit('s94: base texture is not a plain image (run before any KTX2 compression)')
        jb, info = joint_tile(bytes(g.bv_bytes(js['images'][tx['source']]['bufferView'])), period, CFG)
        mr = dict(tile=info, prims=0, faces=0, verts_in=0, verts_out=0, groups={})
        if jb is None:
            rep['materials'][mname] = mr; print(f's94: {mname}: {info}'); continue
        img = dict(js['images'][tx['source']]); img['bufferView'] = ap.bv(jb); img['name'] = (img.get('name') or mname) + '_joints'
        img.pop('uri', None); js['images'].append(img)
        nt = dict(tx); nt['source'] = len(js['images']) - 1; js['textures'].append(nt)
        m['pbrMetallicRoughness']['baseColorTexture'] = dict(bt, index=len(js['textures']) - 1)
        m.setdefault('extras', {})['panel_joints'] = (
            f"added here (s94_ground_joints): {info['panels_per_tile']} x {info['panels_per_tile']} panels of {info['panel_m']} m per "
            f"{period} m tile, joints {CFG['joint_w_m'] * 100:.1f} cm wide at log depth {CFG['joint_log']}, panel tone sd {CFG['panel_sd']}; "
            f"UVs rotated per face to the nearest OSM highway (mod 90 deg, {CFG['orient_q_deg']} deg steps)")
        cls_name = m.get('extras', {}).get('material_class')
        ef = edge_field(_m21.MI[cls_name], float(CFG['edge_sigma_m'])) if CFG.get('edge_sigma_m') and cls_name in _m21.MI else None
        gh = np.zeros(nq, 'i8'); todo = []
        for mesh in js['meshes']:
            for p in mesh['primitives']:
                if p.get('material') != mi or p.get('mode', 4) != 4 or 'indices' not in p:
                    continue
                if 'KHR_draco_mesh_compression' in p.get('extensions', {}):
                    raise SystemExit('s94: draco primitive (run before the draco pass)')
                todo.append((p, {k: read_acc(g, js, v) for k, v in p['attributes'].items()}))
        if CFG.get('chroma_keep') is not None and todo and all('COLOR_0' in t[1] for t in todo):
            xyc = []
            for p, attrs in todo:
                P = attrs['POSITION'][0].astype('f8'); cf = colour_float(*attrs['COLOR_0']); xyc.append((np.stack([P[:, 0], -P[:, 2]], 1), cf[:, :3]))
            new_rgb, cst = chroma_calm(xyc, float(CFG['chroma_keep']), float(CFG['chroma_sigma_m']), float(CFG.get('luma_keep', 1.0))); mr['tint_chroma'] = cst
            for (p, attrs), nr in zip(todo, new_rgb):
                arr, acc = attrs['COLOR_0']; cf = colour_float(arr, acc); cf[:, :3] = nr; attrs['COLOR_0'] = (colour_back(cf, arr, acc), acc)
        for p, attrs in todo:
            I, ia = read_acc(g, js, p['indices']); I = I.reshape(-1, 3).astype(np.int64)
            P = attrs['POSITION'][0].astype('f8'); X = P[:, 0]; Y = -P[:, 2]
            cx = X[I].mean(1); cy = Y[I].mean(1)
            th = sample(ang, grid, cx, cy); mg = sample(mag, grid, cx, cy)
            kf = np.where(mg >= float(CFG['orient_min']), np.round(th / q).astype(int) % nq, 0)
            if ef is not None:   # the material's own outline wins where it is coherent
                te = sample(ef[0], ef[2], cx, cy); me = sample(ef[1], ef[2], cx, cy); ok = me >= float(CFG['edge_min'])
                kf = np.where(ok, np.round(te / q).astype(int) % nq, kf); mr['faces_by_outline'] = mr.get('faces_by_outline', 0) + int(ok.sum())
            key = I * nq + kf[:, None]
            uk, inv = np.unique(key.ravel(), return_inverse=True)
            src_v = uk // nq; kv = uk % nq; t = kv * q
            ct, st = np.cos(t), np.sin(t); xv, yv = X[src_v], Y[src_v]
            uv = np.stack([(xv * ct + yv * st) / period, (-xv * st + yv * ct) / period], 1).astype('f4')
            newp = dict(p); newp['attributes'] = {}
            for k_, (arr, acc) in attrs.items():
                out = uv if k_ == 'TEXCOORD_0' else arr[src_v]
                newp['attributes'][k_] = ap.acc(out.astype(arr.dtype), acc['type'], acc['componentType'], 34962,
                                               minmax=(k_ == 'POSITION'), normalized=bool(acc.get('normalized')))
            ni = inv.reshape(-1).astype('u4' if len(uk) > 65535 or ia['componentType'] == 5125 else 'u2')
            newp['indices'] = ap.acc(ni, 'SCALAR', 5125 if ni.dtype == np.uint32 else 5123, 34963)
            p.clear(); p.update(newp)
            mr['prims'] += 1; mr['faces'] += len(I); mr['verts_in'] += len(P); mr['verts_out'] += len(uk)
            gh += np.bincount(kf, minlength=nq)
        mr['groups'] = {f'{i * float(CFG["orient_q_deg"]):.0f}': round(float(c) / max(gh.sum(), 1), 4) for i, c in enumerate(gh)}
        rep['materials'][mname] = mr; print(f"s94: {mname}: {json.dumps(mr)}", flush=True)
    compact_write(g, js, ap.blobs, a.out)
    if a.report:
        json.dump(rep, open(a.report, 'w'), indent=1)


if __name__ == '__main__':
    main()
