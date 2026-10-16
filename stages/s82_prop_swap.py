"""Stage 82 -- put the street furniture into the published GLBs.

Same shape and the same GLB machinery as
s38_tree_models.py (its reader, Appender and compact_write are imported, not
copied): this stage edits published GLBs in place and nothing else in the build
reads its output.

It adds ONE new scene root, LAYER_props, holding one node per inventory row:

  LAYER_props
    prop_0001   translation (x, z, -y) metres, rotation about +Y, scale,
                extras {source, sources, osm_type, osm_id, class, confidence,
                        surface, yaw_from, asset, licence}
    ...

Geometry is instanced: one mesh per prop class, reused by every node of that
class, so 1,000 props cost one copy of eleven small meshes.

  swap   $OPENCITY_PY s82_prop_swap.py swap IN OUT [--draco]
  check  $OPENCITY_PY s82_prop_swap.py check FILE
"""
import os, sys, json, math, time, argparse, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import opencity_paths as OCP  # noqa: E402
import numpy as np
import common as C
from s38_tree_models import (GLB, Appender, compact_write, instanced_triangles,
                             tri_of_mesh, gt)
import s80_furniture_inventory as S

AD = C.W + '/assets/props'
SCRATCH = (OCP.SHARED + '/props')
# per-area paths when an area is set (pipeline/areacfg.module_paths);
# otherwise the defaults above.
_AP = C._areacfg.module_paths(C) if hasattr(C, '_areacfg') else None
if _AP:
    SCRATCH = _AP['PROPS']
INV = C.W + '/prop_inventory.json'
LAYER = 'LAYER_props'
LINEAR = ('railing_segment',)      # run along their long horizontal axis
JITTER = 0.04                      # +-4% deterministic size jitter per instance


def seeded(key, salt=''):
    return S.seeded(key, salt)


# The Kenney City Kit sign_post carries its street-name blade
# as 36 MINT vertices, measured mean rgb (88.7, 194.7, 134.7).  s85's BOX_COLOUR
# note forbids mint for placeholders, and the same rule applies to the kit
# assets themselves: sign_post is usually the most numerous prop class, and its
# mint blades render as light mint (mean rgb (127.6, 192.8, 171.0)); that many
# pastel blades read as placeholder geometry, not street furniture.
# A real US street-name blade is MUTCD green, near (0, 99, 65).  Rather than
# flatten the 36 vertices to one colour and lose the kit's shading, scale them
# per channel so their MEAN lands on BLADE_GREEN and the relative variation
# survives.  Only vertices the mint test flags are touched, so the post, the
# pole and every other class are untouched.
BLADE_GREEN = (30.0, 95.0, 63.0)
# BLADE_GREEN is then taken to the base colour that renders as MUTCD street-name
# green (25, 100, 70) through the prop render curve: the renderer lifts dark base
# colours, so (30, 95, 63) as a base renders at mean rgb (79.1, 138.7, 121.7),
# and per channel the base is scaled by (target / rendered) ** (1 / RENDER_E),
# with RENDER_E the prop render exponent below.
BLADE_RENDER_TARGET = (25.0, 100.0, 70.0)
BLADE_RENDERED = (79.1, 138.7, 121.7)
RENDER_E = 0.44
BLADE_GREEN = tuple(round(b * (t / r) ** (1.0 / RENDER_E), 1) for b, t, r in
                    zip(BLADE_GREEN, BLADE_RENDER_TARGET, BLADE_RENDERED))
# Besides sign_post, the only kit assets carrying vertices the mint test flags
# are traffic_light (10 such vertices, mean rgb (80, 186, 130)) and small_boat
# (87 vertices, mean (91, 197, 136)).
# traffic_light is LEFT ALONE ON PURPOSE: its 10 flagged vertices sit at y
# 3.07-3.19 m on a 4.0 m post and the asset also carries 120 red-ish and 102
# amber-ish vertices, so that green is the signal's go lens, which is the colour
# a traffic light is supposed to be, not a placeholder.
# small_boat IS a placeholder pastel: the flagged vertices are the topsides band
# at y 0.70-1.17 m of a 1.46 m hull, on a boat whose dominant colours are white
# ((221,221,234) and (243,243,247)).  The mint band is scaled to a light marine
# grey of the SAME luminance (0.2126/0.7152/0.0722 weights: 170.0 before,
# 169.6 after), so the hull keeps its shading and only loses the green cast.
BOAT_HULL = (165.0, 170.0, 178.0)
BLADE_TARGET = {'sign_post': BLADE_GREEN, 'small_boat': BOAT_HULL}


def _is_mint(c8):
    """The mint test, per vertex: green is the max channel, chroma over 0.10,
    green over 150 (the 150 floor separates a pastel blade from dark tree
    foliage)."""
    a = c8[:, :3].astype(np.float32)
    mx, mn, g = a.max(1), a.min(1), a[:, 1]
    return (g >= mx - 0.01) & ((mx - mn) / 255.0 > 0.10) & (g > 150)


def deblade(cls, col):
    """Pull a kit class's mint vertices onto that class's real-world colour."""
    tgt = BLADE_TARGET.get(cls)
    if tgt is None:
        return col, 0, np.zeros(len(col), bool)
    c8 = col if col.dtype == np.uint8 else np.clip(col * 255.0, 0, 255).astype(np.uint8)
    m = _is_mint(c8)
    if not m.any():
        return col, 0, m
    out = c8.copy()
    mean = c8[m][:, :3].astype(np.float32).mean(0)
    k = np.array(tgt, dtype=np.float32) / np.maximum(mean, 1e-6)
    out[m, :3] = np.clip(c8[m][:, :3].astype(np.float32) * k, 0, 255).astype(np.uint8)
    return out, int(m.sum()), m


# -- STREET FURNITURE IS TONED TO THE GROUND IT STANDS ON.
# Untoned, prop_* pixels render far brighter than the ground they stand on
# (about 1.7x the terrain, brighter than the sky), and the bright classes (trash
# can, flagpole, lamp post) most of all.
# Cause: the Kenney kits' COLOR_0 is light pastel paint -- mesh means
# lamp_post (180,188,224), flagpole (210,206,218), trash_can (174,189,189),
# bench (180,143,143), sign_post (119,138,156) -- and nothing downstream
# darkens it: prop_low_poly has no baseColorFactor and prop_palette.jpg,
# sampled at the assets' own UVs, is 249-251/255 (its 172 GLOBAL mean is the
# grain between swatches, not what a vertex reads).  COLOR_0 x texel then goes
# through the same base->pixel curve as the ground (s85 GROUND_RENDER_K/E,
# 5.0356*b**0.7163): predicted 169-230 for those classes, measured 161-220.
#
# Rule: each class's mean COLOR_0 x texel luminance (Rec.709) is capped at the
# median luminance of the ground under that class's own instances, read by
# s85's GroundColour off THE INPUT GLB (COLOR_0 x atlas texel of the ground_/
# path_/stitch_/photo_ground prims, annulus GROUND_SAMPLE_GAP_M..+RING_M) --
# the same base space the kit colour is in, so no render calibration is
# involved and the target moves with the ground if a texture stage changes.
# One factor per class scales RGB uniformly, so hue, chroma ratio and the kit's
# own shading survive (the bench stays red-brown wood, the lamp blue-grey) and
# only brightness moves.  Classes already at or under their ground (railing,
# bollard, picnic_table) come out untouched.  The deblade vertices (the MUTCD
# blade green, the boat hull band) already carry a chosen real-world colour and
# are left as they are; small_boat is exempt (a hull on water, not furniture
# on ground).  If the input has no textured ground the stage keeps kit paint.
TONE_EXEMPT = ('small_boat',)
# -- UPRIGHT PROPS RENDER BRIGHTER THAN FLAT GROUND.  Toning a class's BASE
# luminance to its ground's base luminance does not make it RENDER like its
# ground: an upright post is lit side-on by the sun and ambient, the flat ground
# is not.  PROP_RENDER_RATIO is each class's rendered luminance over its
# ground's (bollard and the small classes stay at 1.0).  A base factor k moves
# rendered L by about k ** RENDER_E, with RENDER_E = 0.44 the prop render
# exponent, so each class's factor is divided by its ratio ** (1 / RENDER_E).
# Hue is untouched (uniform RGB scale) and the target is the ground's rendered
# L, not darker.
PROP_RENDER_RATIO = {'flagpole': 1.136, 'lamp_post': 1.124,
                         'sign_post': 1.272, 'bench': 1.175,
                         'trash_can': 1.276, 'bike_rack': 1.090}
W709 = np.array([0.2126, 0.7152, 0.0722])


def tone_to_ground(A, props, src, js):
    """Cap each used class's kit luminance at its ground's; edits A in place.
    -> {class: report}."""
    import cv2
    import s85_object_place as O
    t0 = time.time()
    grd = O.GroundColour(src, js)
    rin = O.GROUND_SAMPLE_GAP_M
    by = collections.defaultdict(list)
    for p in props:
        g = grd.at(float(p['x']), float(p['y']), rin, rin + O.GROUND_SAMPLE_RING_M)
        if g is not None:
            by[p['asset']].append(float(np.asarray(g, 'f8') @ W709))
    allg = [v for vs in by.values() for v in vs]
    rep = dict(ground_prims=grd.prims, ground_cells=int(len(grd.key)),
               seconds=0.0, classes={})
    if not allg:
        rep['skipped'] = 'no textured ground in the input; kit paint kept'
        return rep
    tex = cv2.imread(AD + '/prop_palette.jpg', cv2.IMREAD_COLOR)[:, :, ::-1]
    h, w = tex.shape[:2]
    for cls in sorted({p['asset'] for p in props}):
        a = A[cls]
        if cls in TONE_EXEMPT:
            rep['classes'][cls] = dict(exempt=True)
            continue
        gs = by.get(cls) or allg
        tgt = float(np.median(gs))
        col = a['col']
        c = col[:, :3].astype('f8')
        if col.dtype != np.uint8:
            c = c * 255.0
        uv = np.asarray(a['uv'], 'f8')
        tx = tex[(np.mod(uv[:, 1], 1.0) * (h - 1)).astype('i4'),
                 (np.mod(uv[:, 0], 1.0) * (w - 1)).astype('i4')].astype('f8') / 255.0
        free = ~a['blade']
        kit = float(((c * tx)[free] @ W709).mean())
        k = min(1.0, tgt / max(kit, 1e-6))
        up = PROP_RENDER_RATIO.get(cls, 1.0) ** (-1.0 / RENDER_E)
        k *= up
        if k < 1.0:
            c2 = c.copy()
            c2[free] *= k
            out = col.copy()
            if col.dtype == np.uint8:
                out[:, :3] = np.clip(np.round(c2), 0, 255).astype(np.uint8)
            else:
                out[:, :3] = (c2 / 255.0).astype(col.dtype)
            a['col'] = out
        rep['classes'][cls] = dict(
            ground_L=round(tgt, 1), kit_L=round(kit, 1), factor=round(k, 3),
            render_correction=round(up, 3),
            ground_samples=len(by.get(cls) or []),
            ground_from='own instances' if by.get(cls) else 'all props')
    rep['seconds'] = round(time.time() - t0, 1)
    return rep


# -- ONE SIGN POST PER SIGN, NOT ONE PER RAY PAIR.
# Most sign_post rows are photo-sourced, many on the minimum evidence (photo:
# 2), and their nearest-neighbour distances cluster just above s80's 3.0 m
# same-object match radius.  That is the signature of one real sign triangulated again and again from different
# panorama pairs, each ghost landing just outside the radius that would have
# merged it, and the ghosts render as a dense cluster of posts.
# Rule: greedy non-maximum suppression at SIGN_POST_SPACING_M, strongest
# evidence first (an OSM row, then lidar support, then photo detections,
# panoramas, confidence; id breaks ties), so the kept row of each cluster is
# its best-evidenced one.  An OSM row is never thinned (it is a mapped object
# with an id).  NOTHING IS DELETED: a thinned row keeps its node, its
# translation and its extras (so check's one-node-per-row holds) and draws no
# mesh, the same typical-position treatment s85's thin_sign_props() gives.
SIGN_POST_SPACING_M = 10.0
THIN_CLASSES = ('sign_post',)


def _evidence_key(p):
    src = set(p.get('sources') or [p.get('source')])
    ev = p.get('evidence') or {}
    return (-int('osm' in src), -int('lidar' in src), -int(ev.get('photo') or 0),
            -int(p.get('photo_views') or 0), -float(p.get('confidence') or 0.0),
            p['id'])


def thin_sign_posts(props):
    """-> {id: {kept, metres, rule}} for the rows that draw nothing."""
    out = {}
    for cls in THIN_CLASSES:
        rows = sorted((p for p in props if p['cls'] == cls), key=_evidence_key)
        kept = []
        for p in rows:
            x, y = float(p['x']), float(p['y'])
            src = set(p.get('sources') or [p.get('source')])
            near = None
            if 'osm' not in src:
                for kx, ky, kid in kept:
                    d = math.hypot(kx - x, ky - y)
                    if d < SIGN_POST_SPACING_M and (near is None or d < near[0]):
                        near = (d, kid)
            if near is None:
                kept.append((x, y, p['id']))
            else:
                out[p['id']] = dict(rule='one %s per %.0f m, strongest evidence '
                                    'kept' % (cls, SIGN_POST_SPACING_M),
                                    kept=near[1], metres=round(near[0], 2))
    return out


def load_assets():
    man = json.load(open(AD + '/manifest.json'))
    z = np.load(AD + '/prop_assets.npz')
    A = {}
    for cls in man['classes']:
        pos = z['pos_' + cls]
        ext = pos.max(0) - pos.min(0)
        run = 'x' if ext[0] >= ext[2] else 'z'
        col, nblade, blade = deblade(cls, z['col_' + cls])
        if nblade:
            print('  deblade %s: %d mint vertices -> %s'
                  % (cls, nblade, BLADE_TARGET[cls]))
        A[cls] = dict(pos=pos, nrm=z['nrm_' + cls], uv=z['uv_' + cls],
                      col=col, blade=blade, ext=ext, run=run,
                      m=man['assets'][cls])
    return A, man


def y_quat(deg):
    t = math.radians(deg) / 2.0
    return [0.0, math.sin(t), 0.0, math.cos(t)]


def node_rotation(cls, A, yaw_enu_deg):
    """Y rotation that points the asset the way the inventory says.

    ENU yaw is measured from +x (East) toward +y (North); the GLB frame is
    (x, z, -y), so an ENU bearing a is the GLB direction (cos a, -sin a) in
    (X, Z).  A rotation t about +Y sends +Z to (sin t, cos t) and +X to
    (cos t, -sin t); so an asset that faces +Z needs t = 90 + a and an asset
    that runs along +X needs t = a."""
    if cls in LINEAR and A[cls]['run'] == 'x':
        return yaw_enu_deg
    if cls in LINEAR:
        return 90.0 + yaw_enu_deg
    return 90.0 + yaw_enu_deg


def instance_scale(p, a):
    """-> (sx, sy, sz) for one prop, from the inventory measurement."""
    ext = a['ext']
    s = 1.0
    h = p.get('height_m')
    if h and ext[1] > 1e-6 and p['cls'] in ('lamp_post', 'sign_post', 'mast',
                                            'flagpole'):
        s = float(np.clip(h / ext[1], 0.5, 2.5))
    j = 1.0 + JITTER * ((((seeded(p['id'], 'size') >> 8) & 0xFFFF) / 65535.0) * 2 - 1)
    sx = sy = sz = s * j
    if p['cls'] in LINEAR and p.get('length_m'):
        run = float(p['length_m'])
        if a['run'] == 'x':
            sx = run / max(float(ext[0]), 1e-6)
        else:
            sz = run / max(float(ext[2]), 1e-6)
        sy = 1.0
    if p['cls'] == 'small_boat' and p.get('length_m'):
        want = float(p['length_m'])
        long_ext = float(max(ext[0], ext[2]))
        k = float(np.clip(want / max(long_ext, 1e-6), 0.4, 6.0))
        sx = sy = sz = k
    return float(sx), float(sy), float(sz)


def do_swap(inp, outp, inv_path=INV):
    t0 = time.time()
    doc = json.load(open(inv_path))
    props = doc['props']
    A, man = load_assets()
    src = GLB(inp)
    js = json.loads(json.dumps(src.js))
    rep = dict(input=inp, output=outp, input_bytes=os.path.getsize(inp),
               triangles_before=instanced_triangles(js), props=len(props))
    # a re-run replaces the layer: drop the old root, compact_write forgets it
    old = [i for i in js['scenes'][0]['nodes']
           if js['nodes'][i].get('name') == LAYER]
    if old:
        js['scenes'][0]['nodes'] = [i for i in js['scenes'][0]['nodes'] if i not in old]
        rep['replaced_existing_layer'] = True
    ap = Appender(js)
    for k in ('images', 'textures', 'samplers', 'materials'):
        js.setdefault(k, [])
    bvi = ap.bv(open(AD + '/prop_palette.jpg', 'rb').read())
    js['images'].append({"bufferView": bvi, "mimeType": "image/jpeg",
                         "name": "prop_palette"})
    # REPEAT, like the tree material: the published-model checks require it on any material
    # that carries a base colour texture.
    js['samplers'].append({"magFilter": 9729, "minFilter": 9987,
                           "wrapS": 10497, "wrapT": 10497})
    js['textures'].append({"sampler": len(js['samplers']) - 1,
                           "source": len(js['images']) - 1})
    lic = {c: A[c]['m'].get('licence') for c in A}
    js['materials'].append({
        "name": "prop_low_poly",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": len(js['textures']) - 1},
                                 "metallicFactor": 0.0, "roughnessFactor": 0.9},
        "doubleSided": True,
        "extras": dict(licence=man['texture']['licence'],
                       texture_source=man['texture']['origin'],
                       asset_licences=lic,
                       asset_urls={c: A[c]['m'].get('page_url') for c in A},
                       material_class="street_furniture")})
    mati = len(js['materials']) - 1
    thin = thin_sign_posts(props)
    rep['sign_posts_thinned'] = len(thin)
    rep['sign_post_rule'] = ('greedy NMS at %.0f m, strongest evidence kept; a '
                             'thinned row keeps its node and extras and draws '
                             'no mesh' % SIGN_POST_SPACING_M)
    drawn = [p for p in props if p['id'] not in thin]
    rep['tone_to_ground'] = tone_to_ground(A, drawn, src, js)
    meshes, tri_per_class = {}, {}
    used = collections.Counter()
    for cls in sorted({p['asset'] for p in props}):
        a = A[cls]
        meshes[cls] = ap.mesh('prop_' + cls, a['pos'], a['nrm'], a['uv'],
                              a['col'], mati)
        tri_per_class[cls] = len(a['pos']) // 3
    kids = []
    for p in props:
        cls = p['asset']
        a = A[cls]
        sx, sy, sz = instance_scale(p, a)
        rot = node_rotation(p['cls'], A, float(p['yaw_deg']))
        nd = {"name": p['id'], "mesh": meshes[cls],
              "translation": [round(float(p['x']), 4), round(float(p['z']), 4),
                              round(-float(p['y']), 4)],
              "rotation": y_quat(rot),
              "scale": [round(sx, 5), round(sy, 5), round(sz, 5)],
              "extras": dict(prop_class=p['cls'], asset=cls,
                             source=p['source'], sources=p['sources'],
                             osm_type=p.get('osm_type'), osm_id=p.get('osm_id'),
                             osm_tag=p.get('osm_tag'),
                             confidence=p.get('confidence'),
                             surface=p.get('surface'), yaw_deg=p['yaw_deg'],
                             yaw_from=p.get('yaw_from'),
                             height_m=p.get('height_m'),
                             evidence=p.get('evidence'),
                             licence=a['m'].get('licence'),
                             asset_url=a['m'].get('page_url'))}
        if p.get('recast_from'):
            # s80 street_masts() recast a lidar mast as a street pole
            nd['extras']['recast_from'] = p['recast_from']
        if p['id'] in thin:
            del nd['mesh']
            nd['extras']['thinned'] = thin[p['id']]
            nd['extras']['static'] = False
            nd['extras']['role'] = 'typical_position'
            kids.append(ap.node(nd))
            used[cls + ':thinned'] += 1
            continue
        kids.append(ap.node(nd))
        used[cls] += 1
    root = ap.node({"name": LAYER, "children": kids, "extras": dict(
        layer_role=('street furniture and small objects; one node per row of '
                    'prop_inventory.json, each an instance of a CC0 low-poly '
                    'model scaled to the measured size'),
        sources=doc.get('per_source'), counts=doc.get('counts'),
        inventory='prop_inventory.json',
        frame='translation is (x_enu, z_up, -y_enu) metres, the model convention')})
    js['scenes'][0]['nodes'].append(root)
    rep.update(nodes_added=len(kids) + 1,
               instances_per_class=dict(used),
               asset_triangles=tri_per_class,
               mesh_data_triangles_added=int(sum(tri_per_class.values())),
               instanced_triangles_added=int(sum(tri_per_class[c] * n
                                                 for c, n in used.items()
                                                 if c in tri_per_class)))
    rep['bytes'] = compact_write(src, js, ap.blobs, outp)
    rep['triangles_after'] = instanced_triangles(js)
    rep['seconds'] = round(time.time() - t0, 1)
    src.close()
    return rep


def do_check(path, inv_path=INV, surface=True):
    """Every claim this stage makes about the file, measured on the file."""
    doc = json.load(open(inv_path))
    props = {p['id']: p for p in doc['props']}
    A, man = load_assets()
    g = GLB(path); js = g.js
    nodes = js['nodes']
    names = [n.get('name') for n in nodes]
    byname = {n.get('name'): n for n in nodes}
    roots = [nodes[i].get('name') for i in js['scenes'][0]['nodes']]
    layer = byname.get(LAYER)
    kids = [nodes[i] for i in (layer or {}).get('children', [])]
    dmax, missing_extras, badrot = 0.0, [], 0
    for nd in kids:
        p = props.get(nd.get('name'))
        if p is None:
            missing_extras.append(nd.get('name')); continue
        tr = nd['translation']
        dmax = max(dmax, abs(tr[0] - p['x']), abs(tr[1] - p['z']),
                   abs(tr[2] + p['y']))
        ex = nd.get('extras') or {}
        for k in ('source', 'prop_class', 'confidence', 'licence'):
            if k not in ex:
                missing_extras.append('%s:%s' % (nd.get('name'), k)); break
        if 'osm' in (ex.get('sources') or []) and ex.get('osm_id') is None:
            missing_extras.append('%s:osm_id' % nd.get('name'))
        q = nd.get('rotation') or [0, 0, 0, 1]
        if abs(q[0]) > 1e-9 or abs(q[2]) > 1e-9:
            badrot += 1
    miss = {'NORMAL': 0, 'TEXCOORD_0': 0, 'COLOR_0': 0}
    for me in js['meshes']:
        for pr in me['primitives']:
            if 'KHR_draco_mesh_compression' in (pr.get('extensions') or {}):
                att = (pr['extensions']['KHR_draco_mesh_compression']['attributes'])
            else:
                att = pr['attributes']
            for k in miss:
                if k not in att and k not in pr['attributes']:
                    miss[k] += 1
    nolic = [m.get('name') for m in js.get('materials', [])
             if 'baseColorTexture' in m.get('pbrMetallicRoughness', {})
             and not (m.get('extras') or {}).get('licence')]
    samp = js.get('samplers', []); tiling = set()
    for m in js.get('materials', []):
        t = m.get('pbrMetallicRoughness', {}).get('baseColorTexture')
        if t is not None:
            tiling.add(js['textures'][t['index']].get('sampler', 0))
    prop_tris = 0
    for nd in kids:
        if 'mesh' in nd:
            prop_tris += tri_of_mesh(js, nd['mesh'])
    out = dict(file=path, bytes=os.path.getsize(path),
               scene_roots=roots, layer_present=bool(layer),
               prop_nodes=len(kids), inventory_rows=len(props),
               one_node_per_inventory_row=(len(kids) == len(props)),
               node_names_unique=(len(names) == len(set(names))),
               max_placement_delta_m=round(float(dmax), 6),
               nodes_missing_extras=missing_extras[:20],
               nodes_with_non_yaw_rotation=badrot,
               primitives_missing_attributes=miss,
               textured_materials_without_licence=nolic,
               texture_sampler_wrapS=[samp[i].get('wrapS') for i in sorted(tiling)
                                      if i < len(samp)],
               prop_instanced_triangles=prop_tris,
               total_instanced_triangles=instanced_triangles(js))
    g.close()
    if surface:
        surf = S.Surface()
        inb, inw, offmodel = [], [], []
        for p in doc['props']:
            r = S.CLASSES[p['cls']]['radius']
            if p['cls'] != 'small_boat' and surf.in_building(p['x'], p['y'], r=0.0):
                inb.append(p['id'])
            z, sn = surf.at(p['x'], p['y'], r=min(r, 1.0))
            if sn == 'water' and p['cls'] not in S.WATER_OK:
                inw.append(p['id'])
            if not np.isfinite(z):
                offmodel.append(p['id'])
            elif abs(z - p['z']) > 0.05:
                offmodel.append(p['id'] + ':z%.2f' % (z - p['z']))
        out.update(props_inside_a_building=inb[:20], n_inside_building=len(inb),
                   props_standing_in_water=inw[:20], n_in_water=len(inw),
                   props_off_the_floor=offmodel[:20], n_off_floor=len(offmodel))
    out['pass'] = bool(out['layer_present'] and out['one_node_per_inventory_row']
                       and out['node_names_unique']
                       and out['max_placement_delta_m'] < 1e-3
                       and not out['nodes_missing_extras']
                       and out['nodes_with_non_yaw_rotation'] == 0
                       and all(v == 0 for v in miss.values()) and not nolic
                       and all(w == 10497 for w in out['texture_sampler_wrapS'])
                       and out.get('n_inside_building', 0) == 0
                       and out.get('n_in_water', 0) == 0
                       and out.get('n_off_floor', 0) == 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['swap', 'check'])
    ap.add_argument('src', nargs='?'); ap.add_argument('dst', nargs='?')
    ap.add_argument('--inventory', default=INV)
    ap.add_argument('--draco', action='store_true',
                    help='input is draco-compressed: decompress, swap, re-encode')
    ap.add_argument('--report', default=None)
    ap.add_argument('--scratch', default=SCRATCH + '/work')
    a = ap.parse_args()
    if a.mode == 'check':
        r = do_check(a.src, a.inventory)
        print(json.dumps(r, indent=1))
        if a.report:
            json.dump(r, open(a.report, 'w'), indent=1)
        sys.exit(0 if r['pass'] else 2)
    os.makedirs(a.scratch, exist_ok=True)
    if a.draco:
        # same route as s38: `copy` decodes the draco stream, `draco` re-encodes
        base = os.path.basename(a.dst).replace('.glb', '')
        plain_in = '%s/%s_in_plain.glb' % (a.scratch, base)
        plain_out = '%s/%s_out_plain.glb' % (a.scratch, base)
        gt(['copy', a.src, plain_in])
        r = do_swap(plain_in, plain_out, a.inventory)
        gt(['draco', plain_out, a.dst])
        r['draco'] = dict(input_bytes=os.path.getsize(a.src),
                          plain_input_bytes=os.path.getsize(plain_in),
                          plain_output_bytes=r['bytes'],
                          output_bytes=os.path.getsize(a.dst))
        r['bytes'] = os.path.getsize(a.dst)
        r['input_bytes'] = os.path.getsize(a.src)
        for f in (plain_in, plain_out):
            if os.path.exists(f):
                os.unlink(f)
    else:
        r = do_swap(a.src, a.dst, a.inventory)
    print(json.dumps(r, indent=1))
    if a.report:
        json.dump(r, open(a.report, 'w'), indent=1)


if __name__ == '__main__':
    main()
