"""Stage 81 -- bake the CC0 prop library into one npz plus one shared texture.

The analogue of s38_tree_models.py's `assets` mode, for street furniture.  Input
is the CC0 kit meshes in assets/props/src described by assets/props/sources.json;
output, written next to them in the area's work copy (<work>/assets/props/), is

  <work>/assets/props/prop_assets.npz    triangle soup per class: pos, nrm, uv, col
  <work>/assets/props/prop_palette.jpg   ONE 512x512 texture, one swatch per class
  <work>/assets/props/manifest.json      per-class licence, url, triangles, measured size

Why a triangle soup and not the kit GLBs: s82 appends the geometry into the
output model's own buffer, and every primitive there must carry NORMAL,
TEXCOORD_0 and COLOR_0, with a REPEAT-sampled base
colour texture that has a licence in its extras.  Colour rides in COLOR_0, the
palette gives the material something to sample; the kit textures are not copied.

Offline: reads only assets/props/src; fetching the kits is a separate step (fetch/fetch_cc0.py).
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
import common as C

# A lamp with an arm, or a sign with a panel, has its bounding box centre off
# to one side of the mast; placing by the bbox centre then stands the pole an
# arm's length away from the inventory point.  These classes are centred on
# their BASE footprint instead.
POLE_BASE_CENTRED = ('lamp_post', 'sign_post', 'flagpole', 'mooring_post')
BASE_SLICE = 0.12      # fraction of the height that counts as "the base"

AD = C.W + '/assets/props'
TEX_PX = 512
CELLS = 4                      # 4x4 swatches = up to 16 classes
CELL_PX = TEX_PX // CELLS


def palette_texture(classes, px=TEX_PX):
    """One neutral, lightly textured swatch per class; COLOR_0 carries colour.

    Generated here, so there is no third-party texture licence to track."""
    from PIL import Image
    rng = np.random.default_rng(81)   # any fixed seed; it only changes the palette grain
    img = np.zeros((px, px, 3), 'f4')
    for k, cls in enumerate(classes):
        i, j = divmod(k, CELLS)
        n = rng.normal(0.0, 0.035, (CELL_PX, CELL_PX, 1))
        # a faint vertical grain: reads as wood/metal at prop size
        g = (np.sin(np.arange(CELL_PX) * 0.7) * 0.02)[None, :, None]
        cell = np.clip(1.0 + n + g, 0.80, 1.0)
        img[i * CELL_PX:(i + 1) * CELL_PX, j * CELL_PX:(j + 1) * CELL_PX] = cell
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype('u1'))
    p = AD + '/prop_palette.jpg'
    out.save(p, quality=92)
    return p


def cell_uv(k, n_vert, pos, rng):
    """UV inside swatch k, jittered with position so the grain is not flat."""
    i, j = divmod(k, CELLS)
    u0 = (j + 0.12) / CELLS; v0 = (i + 0.12) / CELLS
    span = 0.76 / CELLS
    p = pos - pos.min(0)
    ext = np.maximum(p.max(0), 1e-6)
    u = u0 + span * np.clip((p[:, 0] / ext[0] * 0.6 + p[:, 2] / ext[2] * 0.4), 0, 1)
    v = v0 + span * np.clip(p[:, 1] / ext[1], 0, 1)
    return np.stack([u, v], 1).astype('f4')


def _geom_colour(g):
    """-> (n,4) u1 per vertex for one kit mesh.

    Kit GLBs carry their colour three different ways: real vertex colours, a
    baseColorFactor on the material, or a small palette texture.  trimesh's
    to_color() only picks up the first reliably, so the other two are read off
    the material here -- without this every prop comes out white."""
    n = len(g.vertices)
    vis = getattr(g, 'visual', None)
    kind = 'default'
    col = np.full((n, 4), 200, 'u1'); col[:, 3] = 255
    try:
        vc = np.asarray(vis.vertex_colors, 'u1')
        if vc.shape == (n, 4) and vc[:, :3].std() > 2.0:
            return vc, 'vertex colours'
    except Exception:
        pass
    mat = getattr(vis, 'material', None)
    if mat is not None:
        img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
        uv = getattr(vis, 'uv', None)
        if img is not None and uv is not None and len(uv) == n:
            im = np.asarray(img.convert('RGB'))
            h, w = im.shape[:2]
            u = np.clip((np.asarray(uv)[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
            v = np.clip((1.0 - np.asarray(uv)[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(int)
            col[:, :3] = im[v, u]
            return col, 'sampled from the kit texture'
        f = getattr(mat, 'baseColorFactor', None)
        if f is None:
            f = getattr(mat, 'diffuse', None)
        if f is not None:
            f = np.asarray(f, 'f8').ravel()
            if f.max() <= 1.001:
                f = f * 255.0
            col[:, :3] = np.clip(f[:3], 0, 255).astype('u1')
            return col, 'material base colour'
    return col, kind


def load_one(path):
    """-> (pos (n,3) f4, nrm, col (n,4) u1, how) triangle soup, Y up."""
    import trimesh
    sc = trimesh.load(path, process=False)
    geoms = []
    if isinstance(sc, trimesh.Scene):
        for name, g in sc.geometry.items():
            T = np.eye(4)
            try:
                T = sc.graph.get(sc.graph.geometry_nodes[name][0])[0]
            except Exception:
                pass
            g = g.copy(); g.apply_transform(T)
            geoms.append(g)
    else:
        geoms = [sc]
    P, Cc, how = [], [], set()
    for g in geoms:
        v = np.asarray(g.vertices, 'f8'); f = np.asarray(g.faces, 'i4')
        if not len(f):
            continue
        vc, h = _geom_colour(g)
        how.add(h)
        P.append(v[f].reshape(-1, 3))
        Cc.append(vc[f].reshape(-1, 4))
    pos = np.concatenate(P).astype('f4')
    col = np.concatenate(Cc)
    tri = pos.reshape(-1, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), np.array([0.0, 1.0, 0.0]))
    nrm = np.repeat(n, 3, axis=0).astype('f4')
    return pos, nrm, col, sorted(how)


def bake(verbose=True):
    src = json.load(open(AD + '/sources.json'))
    assets = src['assets']
    classes = sorted(assets)
    if len(classes) > CELLS * CELLS:
        raise SystemExit('palette has %d cells, %d classes' % (CELLS ** 2, len(classes)))
    tex = palette_texture(classes)
    rng = np.random.default_rng(7)
    store, man = {}, {}
    for k, cls in enumerate(classes):
        a = assets[cls]
        p = AD + '/src/' + a['file']
        pos, nrm, col, how = load_one(p)
        # normalise: base at y = 0, and centred on origin in X/Z -- on the base
        # footprint for a pole, on the whole bounding box for everything else
        lo = pos.min(0); hi = pos.max(0)
        if cls in POLE_BASE_CENTRED:
            m = pos[:, 1] <= lo[1] + BASE_SLICE * max(hi[1] - lo[1], 1e-6)
            b = pos[m] if m.any() else pos
            cx = (float(b[:, 0].min()) + float(b[:, 0].max())) / 2
            cz = (float(b[:, 2].min()) + float(b[:, 2].max())) / 2
            centred_on = 'base footprint (lowest %d%% of the height)' % (BASE_SLICE * 100)
        else:
            cx = (lo[0] + hi[0]) / 2; cz = (lo[2] + hi[2]) / 2
            centred_on = 'bounding box'
        pos[:, 0] -= cx
        pos[:, 2] -= cz
        pos[:, 1] -= lo[1]
        uv = cell_uv(k, len(pos), pos, rng)
        store['pos_' + cls] = pos
        store['nrm_' + cls] = nrm
        store['uv_' + cls] = uv
        store['col_' + cls] = col
        ext = (pos.max(0) - pos.min(0))
        man[cls] = dict(file=a['file'], triangles=len(pos) // 3,
                        size_m=[round(float(x), 3) for x in ext],
                        swatch=k, source=a.get('source'), page_url=a.get('page_url'),
                        download_url=a.get('download_url'),
                        licence=a.get('licence'),
                        archive_sha256=a.get('archive_sha256'),
                        original_mesh=a.get('original_mesh'),
                        edits=a.get('edits'), facing=a.get('facing', '+Z'),
                        centred_on=centred_on,
                        offset_applied_m=[round(float(cx), 4), round(float(cz), 4)],
                        colour_from=how,
                        mean_rgb=[int(v) for v in col[:, :3].mean(0).round()])
        if verbose:
            print('%-16s %5d tris  %5.2f x %5.2f x %5.2f m  %s'
                  % (cls, len(pos) // 3, ext[0], ext[1], ext[2],
                     '%s rgb%s' % (','.join(how), col[:, :3].mean(0).round().astype(int).tolist())),
                  flush=True)
    np.savez_compressed(AD + '/prop_assets.npz', **store)
    doc = dict(assets=man, texture=dict(
        file='prop_palette.jpg', px=TEX_PX,
        origin='generated in s81_prop_assets.palette_texture()',
        licence='no third-party asset; generated by stages/s81_prop_assets.py',
        bytes=os.path.getsize(tex),
        note='one swatch per class, near-white with a faint grain; the colour of '
             'a prop is in COLOR_0, not in this image'),
        classes=classes, generated=time.strftime('%Y-%m-%dT%H:%M:%S'),
        note='triangle soup per class, Y up, footprint centred on X/Z, base at y=0, '
             'metres at the real-world size')
    json.dump(doc, open(AD + '/manifest.json', 'w'), indent=1)
    print('baked %d classes -> %s (%.0f kB) + %s'
          % (len(classes), AD + '/prop_assets.npz',
             os.path.getsize(AD + '/prop_assets.npz') / 1e3, tex), flush=True)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', nargs='?', default='bake', choices=['bake'])
    ap.parse_args()
    bake()


if __name__ == '__main__':
    main()
