"""Stage 38 -- swap the placeholder tree cylinders for real low-poly tree models.

The assembled model carries one node per inventory tree (``tree_0001`` ...) whose
children are two placeholder cylinders, ``*_trunk`` and ``*_crown``.  This stage
replaces those two children with one instanced low-poly tree model, chosen by the
species group and scaled so the instance's total height and crown radius are the
inventory's numbers.  The ``tree_NNNN`` node itself keeps its name, its
translation and every inventory field in its extras, so the placement the earlier
stages verified is exactly the placement this stage ships; the only extras it
rewrites are the two that described the placeholder (``placeholder``,
``swap_note``) plus two that name the model now standing there.

    s38_tree_models.py assets                      # bake the asset library (once)
    s38_tree_models.py swap IN.glb OUT.glb [opts]  # the swap
    s38_tree_models.py swap IN.glb OUT.glb --identity   # control: compact, no swap
    s38_tree_models.py check MODEL.glb             # the tree placement checks

Options: --draco (decompress the input and re-encode the output with
gltf-transform, for the web copies), --inventory PATH, --report PATH.

The models are the Quaternius Ultimate Nature Pack, CC0.  Nothing in the pack
ships a texture --
every part is a flat material colour -- so the one texture this stage adds is
generated here, foliage in the top half and bark in the bottom half, and every
asset shares it.  One material, one texture, one primitive per variant.
"""
import os, sys, re, json, math, time, struct, hashlib, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import opencity_paths as OCP                    # noqa: E402  host roots, env-overridable
import numpy as np
import common as C
import memguard
from glblib import GLB

AD = C.W + '/assets/trees'
SCRATCH = OCP.SHARED + '/trees'
GT = OCP.GLTF_TRANSFORM
KIT_NAME = 'Quaternius Ultimate Nature Pack'
KIT_PAGE = 'https://quaternius.com/packs/ultimatenature.html'
KIT_URL = 'https://drive.google.com/drive/folders/1-Kl0L_Jg8awbh0S5T-z3zxh4mVlnxTpa'
KIT_LICENCE = ('CC0 1.0 Universal (public domain dedication), '
               'Quaternius Ultimate Nature Pack')

# ---- the six assets, one per species group ------------------------------- #
# Every one is a separate model in the kit, all six well under the 5,000
# triangle budget; the triangle counts are measured at bake time and land in
# the manifest.
ASSETS = {
    'deciduous_broadleaf': dict(file='CommonTree_5.glb', src='CommonTree_5.obj', src_sha256='92999b816fc0e8941f3bf45c480debd06af226a76b9848081b4396ca1fdcc0d6',
        role="default: a broad round-crowned broadleaf (Platanus, Quercus, Ulmus, Acer, Zelkova...)"),
    'conifer':             dict(file='PineTree_5.glb', src='PineTree_5.obj', src_sha256='8dad1b3c5435d9696fd709c1bed77e289e7ce2d8d427b99495dd40b900e3773e',
        role="needleleaved: a layered conic evergreen (Pinus, Pseudotsuga, Thuja, Taxodium...)"),
    'columnar':            dict(file='BirchTree_5.glb', src='BirchTree_5.obj', src_sha256='cf98f18c29d92c29a9a0f372c8083f1d1dc7bf1773dc97539994f29d0fd2fe63',
        role="narrow upright: poplars and fastigiate/columnar cultivars"),
    'small_ornamental':    dict(file='CommonTree_4.glb', src='CommonTree_4.obj', src_sha256='64882415a3170fb4cfd8b07e3b72ecbcae3420deee7a2b0c3fdc8cded12260f2',
        role="small flowering/ornamental with a visible branch structure (Cornus, Prunus, Pyrus...)"),
    'young_street':        dict(file='BirchTree_2.glb', src='BirchTree_2.obj', src_sha256='490f569998d2398200de204090313f927da4a45bb32fadc2b5048f5e09eec4fa',
        role="a recently planted street tree: short, thin, small crown"),
    'shrub':               dict(file='Bush_1.glb', src='Bush_1.obj', src_sha256='0cac6fca9e9a5d899fa1859867bf9cc2dbcc1817b31a0668d8b6faf2b033788b',
        role="a shrub or a tree under 2.5 m: crown to the ground, no clear trunk"),
}
FOLIAGE_MATS = ('leaf', 'grass', 'plant')      # src material names, lowercased

# ---- species -> group ---------------------------------------------------- #
CONIFER_GENERA = {'Pinus', 'Pseudotsuga', 'Abies', 'Picea', 'Thuja', 'Tsuga',
                  'Sequoia', 'Sequoiadendron', 'Chamaecyparis', 'Cedrus',
                  'Cupressus', 'Cryptomeria', 'Calocedrus', 'Juniperus',
                  'Taxus', 'Araucaria', 'Taxodium', 'Metasequoia', 'Larix'}
ORNAMENTAL_GENERA = {'Cornus', 'Prunus', 'Malus', 'Amelanchier', 'Cercis',
                     'Crataegus', 'Syringa', 'Styrax', 'Chionanthus',
                     'Lagerstroemia', 'Stewartia', 'Parrotia', 'Hamamelis',
                     'Koelreuteria', 'Magnolia', 'Pyrus', 'Sorbus', 'Laburnum'}
COLUMNAR_GENERA = {'Populus', 'Liriodendron'}
# small maples are ornamentals; every other Acer is a full-size broadleaf
ORNAMENTAL_SPECIES = re.compile(r'^Acer\s+(palmatum|circinatum|griseum|japonicum)', re.I)
COLUMNAR_CULTIVAR = re.compile(r"fastigiat|columnar|musashino|pyramidal|"
                               r"sentry|obelisk|silhouette|skyrocket|"
                               r"streetwise|emerald\s+sentinel", re.I)
SHRUB_MAX_H = 2.5        # m: anything shorter is a shrub, whatever it is named
YOUNG_MAX_H = 4.0        # m: a short broadleaf with a small crown is a sapling
YOUNG_MAX_R = 1.6        # m


def group_of(t):
    """Species string (plus the leaf_type the join stage derived, plus size)
    -> one of the six asset groups.  Default is deciduous_broadleaf."""
    sp = (t.get('species') or '').strip()
    genus = sp.split()[0] if sp else ''
    why = 'default'
    if sp and COLUMNAR_CULTIVAR.search(sp):
        g, why = 'columnar', 'cultivar name is columnar/fastigiate'
    elif genus in CONIFER_GENERA:
        g, why = 'conifer', 'conifer genus'
    elif genus in ORNAMENTAL_GENERA or (sp and ORNAMENTAL_SPECIES.match(sp)):
        g, why = 'small_ornamental', 'ornamental genus'
    elif genus in COLUMNAR_GENERA:
        g, why = 'columnar', 'columnar genus'
    elif sp:
        g, why = 'deciduous_broadleaf', 'broadleaf genus'
    elif t.get('leaf_type') == 'needleleaved':
        g, why = 'conifer', 'no species, leaf_type needleleaved'
    else:
        g, why = 'deciduous_broadleaf', 'no species, default'
    h = float(t['height_m']); r = float(t['crown_radius_m'])
    if h < SHRUB_MAX_H:
        return 'shrub', f'{why}; under {SHRUB_MAX_H} m so a shrub'
    if g != 'conifer' and h < YOUNG_MAX_H and r < YOUNG_MAX_R:
        return 'young_street', f'{why}; under {YOUNG_MAX_H} m with a small crown'
    return g, why


# ---- a very small glTF reader for the kit files -------------------------- #
CT = {5120: 'i1', 5121: 'u1', 5122: 'i2', 5123: 'u2', 5125: 'u4', 5126: 'f4'}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def read_glb_bytes(b):
    clen, _ = struct.unpack('<II', b[12:20])
    js = json.loads(b[20:20 + clen])
    off = 20 + clen
    struct.unpack('<II', b[off:off + 8])
    binoff = off + 8

    def acc(i):
        a = js['accessors'][i]; bv = js['bufferViews'][a['bufferView']]
        base = binoff + bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        dt = np.dtype(CT[a['componentType']]); nc = NC[a['type']]
        arr = np.frombuffer(b, dtype=dt, count=a['count'] * nc, offset=base)
        return arr.reshape(a['count'], nc) if nc > 1 else arr
    return js, acc


def node_world(js, ni, M=None):
    n = js['nodes'][ni]
    T = np.eye(4)
    if 'matrix' in n:
        T = np.array(n['matrix'], 'f8').reshape(4, 4).T
    else:
        if 'scale' in n: T[:3, :3] = np.diag(n['scale'])
        if 'rotation' in n:
            x, y, z, w = n['rotation']
            R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], 'f8')
            T[:3, :3] = R @ T[:3, :3]
        if 'translation' in n: T[:3, 3] = n['translation']
    return T if M is None else M @ T


# ---- the one shared texture ---------------------------------------------- #
def palette_texture(px=512):
    """Foliage in the top half, bark in the bottom half, generated here so the
    swap adds no third-party image and no licence question.  The foliage half is
    a field of overlapping leaf clusters in two tones with shaded gaps between
    them, so a crown triangle that samples it reads as clustered leaves rather
    than flat paint; the bark half is ridged brown."""
    from PIL import Image
    rng = np.random.default_rng(3802)
    h = px // 2
    yy, xx = np.mgrid[0:h, 0:px].astype('f4')

    # ---- foliage: overlapping leaf clusters, each with its own tone -------- #
    height = np.zeros((h, px), 'f4')      # how much leaf is stacked here
    tone = np.zeros((h, px), 'f4')        # light/dark mix carried by the cluster
    wsum = np.zeros((h, px), 'f4')
    # a slow vertical band so a small triangle landing high in the region gets a
    # lighter canopy tone and one landing low gets a darker one
    band = 0.5 + 0.5 * np.sin(yy / h * np.pi * 2.0 - 0.6)
    for r, n, amp in ((px * 0.075, 26, 1.00), (px * 0.038, 90, 0.85),
                      (px * 0.019, 260, 0.65), (px * 0.010, 620, 0.45)):
        cx = rng.random(n) * px
        cy = rng.random(n) * h
        ang = rng.random(n) * np.pi
        lt = rng.random(n)                      # this cluster's own tone
        for k in range(n):
            dx = np.minimum(np.abs(xx - cx[k]), px - np.abs(xx - cx[k]))
            dy = np.minimum(np.abs(yy - cy[k]), h - np.abs(yy - cy[k]))
            ca, sa = np.cos(ang[k]), np.sin(ang[k])
            u = (dx * ca + dy * sa) / r
            v = (-dx * sa + dy * ca) / (r * 0.55)
            d = u * u + v * v
            w = np.exp(-d * 2.6) * amp
            height += w
            tone += w * lt[k]
            wsum += w
    tone = tone / np.maximum(wsum, 1e-6)
    hn = height / max(float(height.max()), 1e-6)
    # gaps: where little leaf is stacked the canopy shows its own shadow
    gap = np.clip(hn * 3.2, 0, 1)
    # leaf veins/edges: the derivative of the stack picks out cluster rims
    gx = np.gradient(hn, axis=1); gy = np.gradient(hn, axis=0)
    rim = np.clip(1.0 - np.hypot(gx, gy) * 26.0, 0.55, 1.0)
    mix = np.clip(0.45 * tone + 0.35 * band + 0.35 * hn, 0, 1)
    speck = (rng.random((h, px)).astype('f4') - 0.5) * 0.05
    # two tones of leaf, plus a deep shadow colour in the gaps
    dark = np.array([0.055, 0.105, 0.045], 'f4')     # shaded blue-green
    mid = np.array([0.105, 0.190, 0.070], 'f4')
    light = np.array([0.215, 0.315, 0.110], 'f4')    # sunlit yellow-green
    shadow = np.array([0.028, 0.050, 0.026], 'f4')
    m = mix[..., None]
    leaf = np.where(m < 0.5, dark + (mid - dark) * (m / 0.5),
                    mid + (light - mid) * ((m - 0.5) / 0.5))
    fol = shadow + (leaf - shadow) * gap[..., None]
    fol = fol * rim[..., None] + speck[..., None]

    # ---- bark: vertical ridges ------------------------------------------- #
    ph = rng.random(px) * 6.283
    ridge = 0.5 + 0.5 * np.sin(xx * 0.85 + ph[None, :] * 0.0 + np.sin(xx * 0.21) * 1.7)
    s = rng.random((h, px)).astype('f4')
    for _ in range(3):
        s = (s + np.roll(s, 1, 0) + np.roll(s, 2, 0)) / 3.0
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    b = np.clip(0.55 * ridge + 0.45 * s, 0, 1)
    crack = np.clip(1.0 - np.exp(-((np.sin(xx * 0.27 + s * 2.0)) ** 2) * 6.0), 0.45, 1.0)
    bark = np.stack([0.070 + 0.085 * b, 0.052 + 0.062 * b, 0.040 + 0.045 * b], -1)
    bark *= crack[..., None]

    img = np.concatenate([np.clip(fol, 0, 1), np.clip(bark, 0, 1)], 0)
    out = np.clip(np.power(img, 1 / 2.2) * 255, 0, 255).astype('u1')
    return Image.fromarray(out)


FOL_V = (0.010, 0.490)       # the two texture regions, in glTF UV (v=0 is the
BARK_V = (0.510, 0.990)      # top row of the image)
TILE = 0.30                  # asset units per texture tile; a tree is ~1.5 units


def uv_for(pos, fol, rng):
    """One UV per triangle corner, placed so each triangle samples its own patch
    of its region.  Triangles are de-indexed, so no triangle can straddle the
    foliage/bark boundary and nothing bleeds."""
    n = len(pos)
    uv = np.zeros((n, 2), 'f4')
    tris = np.arange(n).reshape(-1, 3)
    for t in tris:
        p = pos[t]
        e1 = p[1] - p[0]; e2 = p[2] - p[0]
        nn = np.cross(e1, e2)
        ax = int(np.argmax(np.abs(nn)))
        keep = [i for i in range(3) if i != ax]
        q = p[:, keep] / TILE
        q = q - q.min(0)
        lo, hi = (FOL_V if fol[t[0]] else BARK_V)
        span = np.array([0.98, hi - lo])
        ext = np.maximum(q.max(0), 1e-6)
        sc = float(min(1.0, min(span[0] / ext[0], span[1] / ext[1])))
        q = q * sc
        off = np.array([0.01, lo]) + rng.random(2) * np.maximum(span - q.max(0), 0)
        uv[t] = q + off
    return uv


def bake_assets(kit_zip=None, verbose=True):
    """Read the six models of the CC0 kit (two-material GLBs in <work>/assets/trees/src;
    this repository ships no converter, so each chosen model has to be exported from
    the pack's OBJ to that layout first), de-index them, re-origin them on
    the trunk base, give them UVs into the generated texture and measure the
    height and crown radius the swap scales against."""
    os.makedirs(AD + '/src', exist_ok=True)
    man = dict(what="low-poly tree models that replace the placeholder cylinders",
               source=dict(name=KIT_NAME, page=KIT_PAGE, download=KIT_URL,
                           licence=KIT_LICENCE,
                           format="the pack ships OBJ/FBX/blend; the bake reads each "
                                  "chosen model as a two-material GLB (leaf_foliage, "
                                  "bark) from <work>/assets/trees/src; this repository "
                                  "ships no converter, so each chosen model has to be "
                                  "exported from the pack's OBJ to that layout first, "
                                  "and the bake itself is offline",
                           credit="Quaternius (quaternius.com); CC0 needs no attribution, "
                                  "this records it anyway"),
               texture=dict(file="tree_palette.jpg",
                            source="generated in s38_tree_models.palette_texture()",
                            licence="no third-party asset; generated by "
                                    "stages/s38_tree_models.py",
                            layout="foliage v 0.010-0.490, bark v 0.510-0.990"),
               assets={}, groups={}, rules={})
    store = {}
    for grp, spec in ASSETS.items():
        name = spec['file']
        dst = AD + '/src/' + name
        data = open(dst, 'rb').read()          # offline, from the converted copies
        sha = hashlib.sha256(data).hexdigest()
        js, acc = read_glb_bytes(data)
        P, N, F = [], [], []
        for ni, nd in enumerate(js['nodes']):
            if 'mesh' not in nd: continue
            M = node_world(js, ni)
            for pr in js['meshes'][nd['mesh']]['primitives']:
                mat = js['materials'][pr['material']]['name'].lower()
                pos = np.asarray(acc(pr['attributes']['POSITION']), 'f8')
                nrm = np.asarray(acc(pr['attributes']['NORMAL']), 'f8')
                idx = np.asarray(acc(pr['indices'])).reshape(-1, 3).ravel()
                pos = (M[:3, :3] @ pos.T).T + M[:3, 3]
                nrm = (M[:3, :3] @ nrm.T).T
                P.append(pos[idx]); N.append(nrm[idx])
                isfol = any(k in mat for k in FOLIAGE_MATS)
                F.append(np.full(len(idx), isfol, bool))
        pos = np.concatenate(P); nrm = np.concatenate(N); fol = np.concatenate(F)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        # re-origin: trunk base at y=0, trunk axis at x=z=0
        bark = ~fol
        axis = pos[bark][:, [0, 2]].mean(0) if bark.sum() else pos[:, [0, 2]].mean(0)
        pos[:, 0] -= axis[0]; pos[:, 2] -= axis[1]
        pos[:, 1] -= pos[:, 1].min()
        H = float(pos[:, 1].max())
        R = float(np.hypot(pos[fol][:, 0], pos[fol][:, 2]).max()) if fol.sum() else \
            float(np.hypot(pos[:, 0], pos[:, 2]).max())
        cb = float(pos[fol][:, 1].min()) if fol.sum() else 0.0
        # the trunk is the bark below the crown, not the branches inside it
        low = bark & (pos[:, 1] <= max(cb * 0.8, 0.05 * H))
        tr = float(np.hypot(pos[low][:, 0], pos[low][:, 2]).max()) if low.sum() else 0.0
        rng = np.random.default_rng(abs(int(hashlib.sha1(grp.encode()).hexdigest()[:8], 16)))
        uv = uv_for(pos, fol, rng)
        store[grp + '_pos'] = pos.astype('f4')
        store[grp + '_nrm'] = nrm.astype('f4')
        store[grp + '_uv'] = uv.astype('f4')
        store[grp + '_fol'] = fol
        man['assets'][grp] = dict(
            model=name, url=KIT_PAGE, download=KIT_URL, licence=KIT_LICENCE,
            role=spec['role'], source_obj=spec.get('src'),
            source_obj_sha256=spec.get('src_sha256'), sha256_glb=sha,
            bytes_glb=len(data),
            triangles=int(len(pos) // 3), vertices=int(len(pos)),
            textures=1, materials_after_bake=1,
            asset_height_units=round(H, 4), asset_crown_radius_units=round(R, 4),
            asset_crown_base_units=round(cb, 4), asset_trunk_radius_units=round(tr, 4),
            crown_base_fraction=round(cb / H, 4), crown_radius_over_height=round(R / H, 4))
        if verbose:
            print(f"  {grp:20s} {name:26s} {len(pos)//3:5d} tri  H {H:.3f}  R {R:.3f}  "
                  f"crown base {cb/H:.2f}H  trunk r {tr:.3f}")
    man['groups'] = {g: ASSETS[g]['role'] for g in ASSETS}
    man['rules'] = dict(
        order=["cultivar name matching /fastigiat|columnar|musashino|pyramidal|sentry|"
               "obelisk|silhouette|skyrocket|streetwise|emerald sentinel/ -> columnar",
               "genus in the conifer list -> conifer",
               "genus in the ornamental list, or Acer palmatum/circinatum/griseum/japonicum"
               " -> small_ornamental",
               "genus Populus or Liriodendron -> columnar",
               "any other species string -> deciduous_broadleaf",
               "no species but leaf_type=needleleaved -> conifer",
               "nothing at all -> deciduous_broadleaf (the default)",
               f"then, whatever the species said: height < {SHRUB_MAX_H} m -> shrub",
               f"then, non-conifers with height < {YOUNG_MAX_H} m and crown radius"
               f" < {YOUNG_MAX_R} m -> young_street"],
        conifer_genera=sorted(CONIFER_GENERA),
        ornamental_genera=sorted(ORNAMENTAL_GENERA),
        columnar_genera=sorted(COLUMNAR_GENERA))
    np.savez_compressed(AD + '/tree_assets.npz', **store)
    palette_texture().save(AD + '/tree_palette.jpg', 'JPEG', quality=92, optimize=True)
    man['texture']['bytes'] = os.path.getsize(AD + '/tree_palette.jpg')
    man['built'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    json.dump(man, open(AD + '/manifest.json', 'w'), indent=1)
    print(f"assets -> {AD}  ({sum(a['triangles'] for a in man['assets'].values())} "
          f"triangles across {len(man['assets'])} models, "
          f"{man['texture']['bytes']/1024:.0f} kB texture)")
    return man


# ---- the swap ------------------------------------------------------------ #
TINTS = (0.78, 0.89, 1.00)          # per-tree foliage brightness, 3 variants
CBASE_BINS = (0.25, 0.40, 0.55, 0.70)   # crown base as a fraction of height
TRUNK_FLOOR = 0.45                  # do not thin an asset trunk past this
TREE_RE = re.compile(r'^tree_\d+$')


def seeded(tree_id, salt=''):
    """A hash that is stable across runs (Python's hash() is salted per process)."""
    h = hashlib.sha1((salt + tree_id).encode()).hexdigest()
    return int(h[:8], 16)


def load_assets():
    z = np.load(AD + '/tree_assets.npz')
    man = json.load(open(AD + '/manifest.json'))
    A = {}
    for g in ASSETS:
        A[g] = dict(pos=z[g + '_pos'].astype('f8'), nrm=z[g + '_nrm'].astype('f8'),
                    uv=z[g + '_uv'], fol=z[g + '_fol'], m=man['assets'][g])
    return A, man


def variant_geometry(a, tint, rb, trunk_f):
    """One mesh variant: the asset warped so its crown starts at `rb` of its
    height, its trunk thinned by `trunk_f`, its foliage tinted by `tint`.
    Height and crown radius are untouched, so the node scale still lands them
    exactly on the inventory."""
    pos = a['pos'].copy(); nrm = a['nrm'].copy(); fol = a['fol']
    H = a['m']['asset_height_units']; cb = a['m']['asset_crown_base_units']
    if rb is not None and cb > 0.05 * H:
        nb = rb * H
        lo = pos[:, 1] <= cb
        s_lo = nb / cb; s_hi = (H - nb) / (H - cb)
        pos[:, 1] = np.where(lo, pos[:, 1] * s_lo, nb + (pos[:, 1] - cb) * s_hi)
        nrm[:, 1] /= np.where(lo, s_lo, s_hi)
    if trunk_f < 0.999 and cb > 0.05 * H:
        # thin the clear stem only, ramping back to 1.0 where the branches start
        y = pos[:, 1]; top = rb * H if (rb is not None and cb > 0.05 * H) else cb
        w = np.clip((y - 0.6 * top) / max(0.4 * top, 1e-6), 0, 1)
        f = trunk_f + (1.0 - trunk_f) * w
        f = np.where(fol, 1.0, f)
        pos[:, 0] *= f; pos[:, 2] *= f
        nrm[:, 0] /= np.maximum(f, 1e-6); nrm[:, 2] /= np.maximum(f, 1e-6)
    if a['m'].get('model') == ASSETS['shrub']['file']:
        # Bush_1 is three stacked lobes; each lobe's underside faces down
        # (asset normal y -0.5 at 0.41 and -0.6 at 0.81 of its height), so every
        # dome renders with dark horizontal seams between the lobes.  Shade it
        # as ONE dome instead: normals radial from a point 0.15 H up the axis.
        # Silhouette untouched.
        c = np.array([0.0, 0.15 * float(pos[:, 1].max()), 0.0])
        nrm = pos - c
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    col = np.full((len(pos), 4), 255, 'u1')
    col[fol, :3] = np.clip(np.array([tint * 0.96, tint, tint * 0.88]) * 255, 0, 255).astype('u1')
    return pos.astype('f4'), nrm.astype('f4'), a['uv'], col


def trunk_factors(inv, A):
    """One thinning factor per group: how much thinner than the asset the
    median tree in that group actually is, floored so nothing becomes a wire."""
    out = {}
    per = {g: [] for g in ASSETS}
    for t in inv:
        g, _ = group_of(t)
        m = A[g]['m']
        if m['asset_trunk_radius_units'] <= 0: continue
        per[g].append(t['trunk_radius_m'] * m['asset_crown_radius_units'] /
                      max(t['crown_radius_m'] * m['asset_trunk_radius_units'], 1e-9))
    for g in ASSETS:
        out[g] = round(float(max(TRUNK_FLOOR, np.median(per[g]))), 3) if per[g] else 1.0
    return out


# ---- GLB rewrite --------------------------------------------------------- #
def _tex_indices(obj, found):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith('Texture') and isinstance(v, dict) and 'index' in v:
                found.add(v['index'])
            else:
                _tex_indices(v, found)
    elif isinstance(obj, list):
        for v in obj: _tex_indices(v, found)


def compact_write(src, js, blobs, outp):
    """Write `js` to `outp`, dropping everything no scene reaches and copying
    every surviving bufferView byte-for-byte out of `src`.  No image is ever
    decoded, so this runs on the full model at a few hundred MB of RSS."""
    keep_n = set()
    stack = [i for sc in js['scenes'] for i in sc['nodes']]
    while stack:
        i = stack.pop()
        if i in keep_n: continue
        keep_n.add(i); stack += js['nodes'][i].get('children', [])
    nmap = {o: k for k, o in enumerate(sorted(keep_n))}
    nodes = []
    for o in sorted(keep_n):
        n = dict(js['nodes'][o])
        if 'children' in n:
            n['children'] = [nmap[c] for c in n['children'] if c in nmap]
            if not n['children']: del n['children']
        nodes.append(n)
    used_m = sorted({n['mesh'] for n in nodes if 'mesh' in n})
    mmap_ = {o: k for k, o in enumerate(used_m)}
    for n in nodes:
        if 'mesh' in n: n['mesh'] = mmap_[n['mesh']]
    meshes = [dict(js['meshes'][o]) for o in used_m]
    acc_used, bv_used, mat_used = set(), set(), set()
    for m in meshes:
        prims = []
        for pr in m['primitives']:
            pr = json.loads(json.dumps(pr))
            for ai in list(pr['attributes'].values()): acc_used.add(ai)
            if 'indices' in pr: acc_used.add(pr['indices'])
            if 'material' in pr: mat_used.add(pr['material'])
            dr = pr.get('extensions', {}).get('KHR_draco_mesh_compression')
            if dr: bv_used.add(dr['bufferView'])
            prims.append(pr)
        m['primitives'] = prims
    mats_o = sorted(mat_used)
    matmap = {o: k for k, o in enumerate(mats_o)}
    materials = [js['materials'][o] for o in mats_o]
    tex_used = set(); _tex_indices(materials, tex_used)
    tex_o = sorted(tex_used); texmap = {o: k for k, o in enumerate(tex_o)}
    textures = [dict(js['textures'][o]) for o in tex_o]
    img_o = sorted({t['source'] for t in textures if 'source' in t})
    imgmap = {o: k for k, o in enumerate(img_o)}
    smp_o = sorted({t['sampler'] for t in textures if 'sampler' in t})
    smpmap = {o: k for k, o in enumerate(smp_o)}
    for t in textures:
        if 'source' in t: t['source'] = imgmap[t['source']]
        if 'sampler' in t: t['sampler'] = smpmap[t['sampler']]
    images = [dict(js['images'][o]) for o in img_o]
    samplers = [js['samplers'][o] for o in smp_o]
    materials = json.loads(json.dumps(materials))

    def remap_tex(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.endswith('Texture') and isinstance(v, dict) and 'index' in v:
                    v['index'] = texmap[v['index']]
                else: remap_tex(v)
        elif isinstance(o, list):
            for v in o: remap_tex(v)
    remap_tex(materials)
    for m in meshes:
        for pr in m['primitives']:
            if 'material' in pr: pr['material'] = matmap[pr['material']]
    accs_o = sorted(acc_used)
    accmap = {o: k for k, o in enumerate(accs_o)}
    accessors = [dict(js['accessors'][o]) for o in accs_o]
    for m in meshes:
        for pr in m['primitives']:
            pr['attributes'] = {k: accmap[v] for k, v in pr['attributes'].items()}
            if 'indices' in pr: pr['indices'] = accmap[pr['indices']]
    for a in accessors:
        if 'bufferView' in a: bv_used.add(a['bufferView'])
    for im in images:
        if 'bufferView' in im: bv_used.add(im['bufferView'])
    bvs_o = sorted(bv_used)
    bvmap = {o: k for k, o in enumerate(bvs_o)}
    for a in accessors:
        if 'bufferView' in a: a['bufferView'] = bvmap[a['bufferView']]
    for im in images:
        if 'bufferView' in im: im['bufferView'] = bvmap[im['bufferView']]
    for m in meshes:
        for pr in m['primitives']:
            dr = pr.get('extensions', {}).get('KHR_draco_mesh_compression')
            if dr: dr['bufferView'] = bvmap[dr['bufferView']]
    # emit
    out = dict(js)
    out['nodes'] = nodes; out['meshes'] = meshes; out['accessors'] = accessors
    out['materials'] = materials; out['textures'] = textures
    out['images'] = images; out['samplers'] = samplers
    out['scenes'] = [dict(sc, nodes=[nmap[i] for i in sc['nodes'] if i in nmap])
                     for sc in js['scenes']]
    for k in ('textures', 'images', 'samplers', 'materials'):
        if not out[k]: del out[k]
    bvout = []; off = 0
    plan = []
    for o in bvs_o:
        bv = dict(js['bufferViews'][o])
        blob = blobs[bv.pop('_blob')] if '_blob' in bv else None
        n = bv['byteLength']
        plan.append((o, blob, n))
        bv2 = {k: v for k, v in bv.items() if k != 'byteOffset'}
        bv2['buffer'] = 0; bv2['byteOffset'] = off; bv2['byteLength'] = n
        bvout.append(bv2)
        off += n + ((4 - n % 4) % 4)
    out['bufferViews'] = bvout
    out['buffers'] = [{"byteLength": off}]
    jb = json.dumps(out, separators=(',', ':')).encode('utf8')
    jb += b' ' * ((4 - len(jb) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + off
    with open(outp, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))
        f.write(struct.pack('<II', len(jb), 0x4E4F534A)); f.write(jb)
        f.write(struct.pack('<II', off, 0x004E4942))
        for o, blob, n in plan:
            if blob is not None:
                f.write(blob)
            else:
                sbv = js['bufferViews'][o]
                so = src.bin_off + sbv.get('byteOffset', 0)
                rem = n; pos = so
                while rem:
                    c = min(rem, 1 << 22)
                    f.write(src.mm[pos:pos + c]); pos += c; rem -= c
            pad = (4 - n % 4) % 4
            if pad: f.write(b'\x00' * pad)
    return total


class Appender:
    """Adds accessors/bufferViews to an existing glTF JSON without touching its
    binary chunk; the bytes ride along in `blobs` and compact_write emits them."""
    def __init__(self, js):
        self.js = js; self.blobs = []

    def bv(self, data, target=None):
        self.blobs.append(bytes(data))
        bv = {"buffer": 0, "byteLength": len(data), "_blob": len(self.blobs) - 1}
        if target: bv['target'] = target
        self.js['bufferViews'].append(bv)
        return len(self.js['bufferViews']) - 1

    def acc(self, arr, typ, ctype, target, minmax=False, normalized=False):
        arr = np.ascontiguousarray(arr)
        a = {"bufferView": self.bv(arr.tobytes(), target), "componentType": ctype,
             "count": int(arr.shape[0]), "type": typ}
        if minmax:
            a['min'] = [float(x) for x in arr.min(0)]
            a['max'] = [float(x) for x in arr.max(0)]
        if normalized: a['normalized'] = True
        self.js['accessors'].append(a)
        return len(self.js['accessors']) - 1

    def mesh(self, name, pos, nrm, uv, col, mat):
        idx = np.arange(len(pos), dtype='u4')
        prim = {"attributes": {
                    "POSITION": self.acc(pos, "VEC3", 5126, 34962, minmax=True),
                    "NORMAL": self.acc(nrm, "VEC3", 5126, 34962),
                    "TEXCOORD_0": self.acc(uv, "VEC2", 5126, 34962),
                    "COLOR_0": self.acc(col, "VEC4", 5121, 34962, normalized=True)},
                "indices": self.acc(idx, "SCALAR", 5125, 34963), "mode": 4,
                "material": mat}
        self.js['meshes'].append({"primitives": [prim], "name": name})
        return len(self.js['meshes']) - 1

    def node(self, n):
        self.js['nodes'].append(n)
        return len(self.js['nodes']) - 1


def tri_of_mesh(js, mi):
    return sum(js['accessors'][pr['indices']]['count'] // 3
               for pr in js['meshes'][mi]['primitives'] if 'indices' in pr)


def instanced_triangles(js):
    """Triangles a renderer draws, counting one node as one instance."""
    seen = set(); stack = [i for sc in js['scenes'] for i in sc['nodes']]; n = 0
    while stack:
        i = stack.pop()
        if i in seen: continue
        seen.add(i)
        nd = js['nodes'][i]
        if 'mesh' in nd: n += tri_of_mesh(js, nd['mesh'])
        stack += nd.get('children', [])
    return n


def do_swap(inp, outp, inv_path, identity=False):
    t0 = time.time()
    inv = json.load(open(inv_path))['trees']
    byid = {t['id']: t for t in inv}
    A, man = load_assets()
    tf = trunk_factors(inv, A)
    src = GLB(inp)
    js = json.loads(json.dumps(src.js))
    rep = dict(input=inp, output=outp, identity=bool(identity),
               input_bytes=os.path.getsize(inp),
               triangles_before=instanced_triangles(js),
               assets={g: dict(model=A[g]['m']['model'],
                               triangles=A[g]['m']['triangles'],
                               trunk_thinning=tf[g]) for g in ASSETS})
    ap = Appender(js)
    names_before, tri_before, tree_nodes = [], 0, []
    for i, nd in enumerate(js['nodes']):
        if TREE_RE.match(nd.get('name', '')):
            tree_nodes.append(i); names_before.append(nd['name'])
            for c in nd.get('children', []):
                cn = js['nodes'][c]
                if 'mesh' in cn: tri_before += tri_of_mesh(js, cn['mesh'])
    rep['tree_nodes'] = len(tree_nodes)
    rep['tree_triangles_before'] = tri_before
    if identity:
        rep['note'] = "control run: nothing swapped, only the unreachable data dropped"
        rep['bytes'] = compact_write(src, js, ap.blobs, outp)
        rep['triangles_after'] = rep['triangles_before']
        rep['seconds'] = round(time.time() - t0, 1)
        src.close()
        return rep
    # one material, one texture, shared by every variant
    for k in ('images', 'textures', 'samplers', 'materials'):
        js.setdefault(k, [])
    bvi = ap.bv(open(AD + '/tree_palette.jpg', 'rb').read())
    js['images'].append({"bufferView": bvi, "mimeType": "image/jpeg",
                         "name": "tree_palette"})
    # REPEAT, not CLAMP: every UV this stage writes sits inside its region so
    # the wrap mode never bites, and the published-model checks require REPEAT on every
    # material that carries a base colour texture.
    js['samplers'].append({"magFilter": 9729, "minFilter": 9987,
                           "wrapS": 10497, "wrapT": 10497})
    js['textures'].append({"sampler": len(js['samplers']) - 1,
                           "source": len(js['images']) - 1})
    js['materials'].append({
        "name": "tree_low_poly",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": len(js['textures']) - 1},
                                 "metallicFactor": 0.0, "roughnessFactor": 0.93},
        "doubleSided": True,
        "extras": dict(licence=man['texture']['licence'],
                       texture_source=man['texture']['source'],
                       models=man['source']['name'], model_url=man['source']['page'],
                       model_licence=man['source']['licence'],
                       material_class="tree_model")})
    mati = len(js['materials']) - 1
    cache, meas, used = {}, {}, {}
    hist = {g: 0 for g in ASSETS}; why_hist = {}
    err_h, err_r, err_cb, err_t, err_xyz, dist = [], [], [], [], [], []
    skipped = []
    for i in tree_nodes:
        nd = js['nodes'][i]
        t = byid.get(nd['name'])
        if t is None:
            skipped.append(nd['name']); continue
        g, why = group_of(t)
        hist[g] += 1; why_hist[why] = why_hist.get(why, 0) + 1
        s = seeded(t['id'])
        k_tint = s % len(TINTS)
        rb_t = float(np.clip(t['crown_base_m'] / max(t['height_m'], 1e-6), 0.15, 0.75))
        k_cb = int(np.argmin([abs(rb_t - b) for b in CBASE_BINS]))
        key = (g, k_tint, k_cb)
        if key not in cache:
            pos, nrm, uv, col = variant_geometry(A[g], TINTS[k_tint],
                                                 CBASE_BINS[k_cb], tf[g])
            cache[key] = ap.mesh(f"tree_{g}_t{k_tint}_c{k_cb}", pos, nrm, uv, col, mati)
            fol = A[g]['fol']; bark = ~fol
            hh = float(pos[:, 1].max())
            rr = float(np.hypot(pos[fol][:, 0], pos[fol][:, 2]).max()) if fol.any() else \
                 float(np.hypot(pos[:, 0], pos[:, 2]).max())
            cbv = float(pos[fol][:, 1].min()) if fol.any() else 0.0
            low = bark & (pos[:, 1] <= max(cbv * 0.8, 0.05 * hh))
            trv = float(np.hypot(pos[low][:, 0], pos[low][:, 2]).max()) if low.any() else 0.0
            meas[key] = (hh, rr, cbv, trv)
            used[f"{g}|tint{k_tint}|cbase{CBASE_BINS[k_cb]}"] = len(pos) // 3
        hh, rr, cbv, trv = meas[key]
        sy = t['height_m'] / hh
        sxz = t['crown_radius_m'] / rr
        yaw = 2 * math.pi * (((s >> 8) & 0xFFFF) / 65536.0)
        ni = ap.node({"name": "model_" + t['id'], "mesh": cache[key],
                      "rotation": [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)],
                      "scale": [sxz, sy, sxz],
                      "extras": dict(tree_model=A[g]['m']['model'], species_group=g,
                                     group_reason=why, yaw_deg=round(math.degrees(yaw), 2),
                                     licence=man['source']['licence'],
                                     source=man['source']['page'])})
        nd['children'] = [ni]
        ex = nd.setdefault('extras', {})
        ex['placeholder'] = False
        ex['swap_note'] = (f"low-poly model {A[g]['m']['model']} ({g}), scaled to the "
                           f"inventory height and crown radius; translation is still the "
                           f"trunk base, +Y up, metres")
        ex['tree_model'] = A[g]['m']['model']; ex['species_group'] = g
        # what was actually written, measured on the variant geometry
        err_h.append(hh * sy - t['height_m'])
        err_r.append(rr * sxz - t['crown_radius_m'])
        err_cb.append(cbv * sy - t['crown_base_m'])
        if trv > 0: err_t.append(trv * sxz - t['trunk_radius_m'])
        want = (t['x_enu_m'], t['ground_z_enu_m'], -t['y_enu_m'])
        got = nd.get('translation', [0, 0, 0])
        err_xyz.append(max(abs(a_ - b_) for a_, b_ in zip(want, got)))
        dist.append(max(sy / sxz, sxz / sy) if min(sy, sxz) > 0 else 1.0)

    for i, nd in enumerate(js['nodes']):
        if nd.get('name') == 'LAYER_trees':
            nd.setdefault('extras', {})['layer_role'] = (
                "tree placements; each node holds one instanced low-poly model "
                f"({man['source']['name']}, {man['source']['licence']}) scaled to the "
                "inventory height and crown radius")
    names_after = [js['nodes'][i]['name'] for i in tree_nodes]
    q = lambda v, p: round(float(np.percentile(np.abs(v), p)), 4) if len(v) else None
    rep.update(
        swapped=len(err_h), skipped_not_in_inventory=skipped,
        groups=hist, group_reasons=why_hist,
        variants=len(cache), variant_triangles=used,
        mesh_data_triangles_added=int(sum(used.values())),
        tree_triangles_after=int(sum(len(A[g]['pos']) // 3 * n for g, n in hist.items())),
        names_preserved=bool(names_after == names_before),
        max_translation_delta_m=round(float(max(err_xyz)), 6) if err_xyz else None,
        height_error_m=dict(max=q(err_h, 100), p95=q(err_h, 95)),
        crown_radius_error_m=dict(max=q(err_r, 100), p95=q(err_r, 95)),
        height_error_pct=round(float(np.abs(np.array(err_h) /
            np.array([byid[n]['height_m'] for n in names_before if n in byid])).max() * 100), 4),
        crown_radius_error_pct=round(float(np.abs(np.array(err_r) /
            np.array([byid[n]['crown_radius_m'] for n in names_before if n in byid])).max() * 100), 4),
        crown_base_error_m=dict(median=q(err_cb, 50), p90=q(err_cb, 90), max=q(err_cb, 100)),
        trunk_radius_error_m=dict(median=q(err_t, 50), p90=q(err_t, 90), max=q(err_t, 100)),
        vertical_stretch_vs_asset=dict(median=round(float(np.median(dist)), 2),
                                       p90=round(float(np.percentile(dist, 90)), 2),
                                       max=round(float(np.max(dist)), 2)))
    rep['bytes'] = compact_write(src, js, ap.blobs, outp)
    rep['triangles_after'] = instanced_triangles(js)
    rep['seconds'] = round(time.time() - t0, 1)
    src.close()
    return rep


def do_check(path, inv_path):
    """The tree placement checks, on any file, so the swap can be
    proved on a private copy without writing to the published directory."""
    inv = json.load(open(inv_path))['trees']
    g = GLB(path); js = g.js
    nodes = js['nodes']
    names = [n.get('name') for n in nodes]
    byname = {n.get('name'): n for n in nodes}
    ntree = sum(1 for n in nodes if n.get('name', '').startswith('tree_')
                and not n['name'].endswith(('_trunk', '_crown')))
    d = []
    for t in inv:
        n = byname.get(t['id'])
        if n is None: continue
        tr = n['translation']
        d.append(max(abs(tr[0] - t['x_enu_m']), abs(tr[1] - t['ground_z_enu_m']),
                     abs(tr[2] + t['y_enu_m'])))
    miss = {'NORMAL': 0, 'TEXCOORD_0': 0, 'COLOR_0': 0}
    for me in js['meshes']:
        for pr in me['primitives']:
            for k in miss:
                if k not in pr['attributes']: miss[k] += 1
    nolic = [m.get('name') for m in js.get('materials', [])
             if 'baseColorTexture' in m.get('pbrMetallicRoughness', {})
             and not (m.get('extras') or {}).get('licence')]
    samp = js.get('samplers', []); tiling = set()
    for m in js.get('materials', []):
        t = m.get('pbrMetallicRoughness', {}).get('baseColorTexture')
        if t is not None: tiling.add(js['textures'][t['index']].get('sampler', 0))
    roots = [nodes[i].get('name') for i in js['scenes'][0]['nodes']]
    out = dict(
        file=path, bytes=os.path.getsize(path),
        node_names_unique=(len(names) == len(set(names))),
        scene_roots=roots,
        tree_nodes_counted=ntree, inventory_trees=len(inv),
        one_node_per_inventoried_tree=(ntree == len(inv)),
        max_placement_delta_m=round(float(max(d)), 6) if d else None,
        placements_checked=len(d),
        primitives_missing_attributes=miss,
        textured_materials_without_licence=nolic,
        texture_sampler_wrapS=[samp[i].get('wrapS') for i in sorted(tiling)
                               if i < len(samp)])
    out['pass'] = bool(out['node_names_unique'] and out['one_node_per_inventoried_tree']
                       and (out['max_placement_delta_m'] or 0) < 1e-3
                       and all(v == 0 for v in miss.values()) and not nolic
                       and all(w == 10497 for w in out['texture_sampler_wrapS']))
    g.close()
    return out


def gt(args, log=None):
    r = subprocess.run([GT] + args, cwd=OCP.GLTF_DIR, capture_output=True,
                       text=True, env=dict(os.environ, NODE_OPTIONS='--max-old-space-size=6144'))
    if r.returncode != 0:
        raise RuntimeError("gltf-transform " + " ".join(args[:2]) + ": " + r.stderr[-500:])
    return r.stdout


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('mode', choices=['assets', 'swap', 'check'])
    ap_.add_argument('src', nargs='?'); ap_.add_argument('dst', nargs='?')
    ap_.add_argument('--inventory', default=C.W + '/tree_inventory.json')
    ap_.add_argument('--identity', action='store_true')
    ap_.add_argument('--draco', action='store_true',
                     help='input is draco-compressed: decompress, swap, re-encode')
    ap_.add_argument('--report', default=None)
    ap_.add_argument('--scratch', default=SCRATCH + '/work')
    a = ap_.parse_args()
    if a.mode == 'assets':
        bake_assets(); return
    if a.mode == 'check':
        r = do_check(a.src, a.inventory)
        print(json.dumps(r, indent=1))
        if a.report: json.dump(r, open(a.report, 'w'), indent=1)
        sys.exit(0 if r['pass'] else 1)
    memguard.require('tree_swap', 3.0, need_disk_gb=1.0)
    os.makedirs(a.scratch, exist_ok=True)
    with memguard.PeakRSS(label='tree_swap') as pk:
        if a.draco:
            base = os.path.basename(a.dst).replace('.glb', '')
            plain_in = f"{a.scratch}/{base}_in_plain.glb"
            plain_out = f"{a.scratch}/{base}_out_plain.glb"
            gt(['copy', a.src, plain_in])
            rep = do_swap(plain_in, plain_out, a.inventory, a.identity)
            gt(['draco', plain_out, a.dst])
            rep['draco'] = dict(input_bytes=os.path.getsize(a.src),
                                plain_input_bytes=os.path.getsize(plain_in),
                                plain_output_bytes=rep['bytes'],
                                output_bytes=os.path.getsize(a.dst))
            rep['bytes'] = os.path.getsize(a.dst)
            rep['input_bytes'] = os.path.getsize(a.src)
            for f in (plain_in, plain_out):
                if os.path.exists(f): os.unlink(f)
        else:
            rep = do_swap(a.src, a.dst, a.inventory, a.identity)
    rep['peak_rss_gb'] = round(pk.peak, 2)
    print(json.dumps(rep, indent=1))
    if a.report:
        json.dump(rep, open(a.report, 'w'), indent=1)


if __name__ == '__main__':
    main()
