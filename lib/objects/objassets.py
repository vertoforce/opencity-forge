"""Stage 84's asset half: the CC0 model library s85 places.

Every mesh is either a Kenney CC0 kit model (downloaded from kenney.nl, archive
sha256 recorded, License.txt quoted) or generated here with trimesh primitives,
so the library carries no licence that restricts redistribution.  The kits are unpacked
from the scratch copy of the zip; nothing here goes to the network except
`fetch`, and `fetch` skips a kit that is already on disk.

Baking is s81_prop_assets.bake() with its asset directory pointed at
assets/objects, so the two libraries are identical in shape: one triangle soup
per class in an npz, one generated palette texture, one manifest.

  fetch   python -m objects.objassets fetch    (kenney.nl, official download URL)
  build   python -m objects.objassets build    (unpack, rescale, bake)
"""
import os, sys, json, time, hashlib, zipfile, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import common as C
import s81_prop_assets as S81
from objects.objcommon import SCR

KITS = {
    'car-kit': dict(
        url='https://kenney.nl/media/pages/assets/car-kit/1a312ec241-1775131960/kenney_car-kit.zip',
        page='https://kenney.nl/assets/car-kit', zip='kenney_car-kit.zip'),
    'watercraft-kit': dict(
        url='https://kenney.nl/media/pages/assets/watercraft-kit/a335cfed49-1713519620/kenney_watercraft-pack.zip',
        page='https://kenney.nl/assets/watercraft-kit', zip='kenney_watercraft-pack.zip'),
    'city-kit-roads': dict(
        url='https://kenney.nl/media/pages/assets/city-kit-roads/74288c9459-1787042796/kenney_city-kit-roads.zip',
        page='https://kenney.nl/assets/city-kit-roads', zip='kenney_city-kit-roads.zip'),
}
KD = SCR + '/kits'
AD = C.W + '/assets/objects'

# class -> (kit, member, axis to scale on, target size in metres)
# axis 'x'/'y'/'z' is the axis of the ORIGINAL model that the target applies to.
PICK = {
    'car_sedan':    ('car-kit', 'sedan.glb', 'long', 4.5),
    'car_suv':      ('car-kit', 'suv.glb', 'long', 4.8),
    'car_van':      ('car-kit', 'van.glb', 'long', 5.0),
    'car_taxi':     ('car-kit', 'taxi.glb', 'long', 4.6),
    'truck':        ('car-kit', 'truck.glb', 'long', 7.0),
    'delivery':     ('car-kit', 'delivery.glb', 'long', 6.5),
    'boat_speed':   ('watercraft-kit', 'boat-speed-a.glb', 'long', 6.0),
    'boat_sail':    ('watercraft-kit', 'boat-sail-a.glb', 'long', 9.0),
    'boat_row':     ('watercraft-kit', 'boat-row-large.glb', 'long', 4.5),
    'boat_tow':     ('watercraft-kit', 'boat-tow-a.glb', 'long', 12.0),
    'traffic_light': ('city-kit-roads', 'traffic-light.glb', 'y', 4.0),
    'utility_pole': ('city-kit-roads', 'electricity-pole-single.glb', 'y', 9.0),
    'sign_board':   ('city-kit-roads', 'road-sign-empty.glb', 'y', 2.2),
    'traffic_cone': ('city-kit-roads', 'construction-cone.glb', 'y', 0.7),
    'jersey_barrier': ('city-kit-roads', 'construction-barrier.glb', 'long', 2.0),
}
GENERATED = ('planter', 'utility_cabinet')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fetch():
    os.makedirs(KD, exist_ok=True)
    for name, k in KITS.items():
        p = KD + '/' + k['zip']
        if os.path.exists(p):
            print('have %s' % k['zip'], flush=True); continue
        # -L: kenney.nl redirects a kit's download URL to its newest version.
        subprocess.check_call(['curl', '-fsSL', '-m', '300', '-o', p, k['url']])
        if not zipfile.is_zipfile(p):
            os.remove(p)
            sys.exit('%s: download is not a zip archive; see %s' % (k['zip'], k['page']))
        print('fetched %s %d bytes' % (k['zip'], os.path.getsize(p)), flush=True)


def licence_of(zp):
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.lower().endswith('license.txt'):
                t = z.read(n).decode('utf8', 'replace')
                for line in t.splitlines():
                    if 'creative commons' in line.lower() or 'cc0' in line.lower():
                        return line.strip()
                return t.strip().splitlines()[0]
    return None


_TEXCACHE = {}


def kit_texture(zp):
    """Models/GLB format/Textures/colormap.png out of a kit zip."""
    from PIL import Image
    import io
    if zp in _TEXCACHE:
        return _TEXCACHE[zp]
    with zipfile.ZipFile(zp) as z:
        n = next(x for x in z.namelist()
                 if x.endswith('GLB format/Textures/colormap.png'))
        img = Image.open(io.BytesIO(z.read(n))).convert('RGB')
    _TEXCACHE[zp] = img
    return img


def gen_planter():
    """A square planter box: four walls and a soil top.  Generated here."""
    import trimesh
    outer = trimesh.creation.box(extents=[1.2, 0.85, 1.2])
    outer.apply_translation([0, 0.425, 0])
    inner = trimesh.creation.box(extents=[0.98, 0.8, 0.98])
    inner.apply_translation([0, 0.52, 0])
    try:
        m = outer.difference(inner)
        if m is None or not len(m.faces):
            raise ValueError
    except Exception:
        m = outer
    soil = trimesh.creation.box(extents=[0.98, 0.06, 0.98])
    soil.apply_translation([0, 0.80, 0])
    m.visual.vertex_colors = np.tile([150, 140, 125, 255], (len(m.vertices), 1))
    soil.visual.vertex_colors = np.tile([70, 55, 40, 255], (len(soil.vertices), 1))
    return trimesh.util.concatenate([m, soil])


def gen_cabinet():
    """A street utility cabinet: a rounded box on a plinth.  Generated here."""
    import trimesh
    body = trimesh.creation.box(extents=[0.9, 1.1, 0.5])
    body.apply_translation([0, 0.62, 0])
    plinth = trimesh.creation.box(extents=[1.0, 0.12, 0.6])
    plinth.apply_translation([0, 0.06, 0])
    body.visual.vertex_colors = np.tile([120, 128, 120, 255], (len(body.vertices), 1))
    plinth.visual.vertex_colors = np.tile([90, 90, 92, 255], (len(plinth.vertices), 1))
    return trimesh.util.concatenate([body, plinth])


def build():
    import trimesh
    os.makedirs(AD + '/src', exist_ok=True)
    assets = {}
    for cls, (kit, member, axis, target) in sorted(PICK.items()):
        zp = KD + '/' + KITS[kit]['zip']
        inner = 'Models/GLB format/' + member
        with zipfile.ZipFile(zp) as z:
            data = z.read(inner)
        tmp = SCR + '/work/_kit_%s.glb' % cls
        open(tmp, 'wb').write(data)
        m = trimesh.load(tmp, process=False)
        # The kit GLBs reference Textures/colormap.png, which is a file NEXT to
        # them in the zip, not an embedded image: every material comes out with
        # no baseColorTexture and every model bakes white.  Attach the kit's own
        # colormap here so s81's load_one can sample it per vertex.
        img = kit_texture(zp)
        for g in (m.geometry.values() if hasattr(m, 'geometry') else [m]):
            try:
                g.visual.material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=img, metallicFactor=0.0)
            except Exception:
                pass
        ext = (m.bounds[1] - m.bounds[0]) if hasattr(m, 'bounds') else None
        if ext is None:
            raise SystemExit('no bounds for ' + cls)
        cur = {'x': ext[0], 'y': ext[1], 'z': ext[2],
               'long': max(ext[0], ext[2])}[axis]
        s = float(target) / float(cur)
        m.apply_scale(s)
        m.export(AD + '/src/%s.glb' % cls)
        ext2 = m.bounds[1] - m.bounds[0]
        assets[cls] = dict(
            file='%s.glb' % cls, nominal_size_m=[round(float(v), 3) for v in ext2],
            facing='+Z' if ext2[2] >= ext2[0] else '+X',
            source='Kenney %s' % kit, page_url=KITS[kit]['page'],
            download_url=KITS[kit]['url'], archive_sha256=sha256(zp),
            licence=licence_of(zp), original_mesh=inner,
            edits=['scaled uniformly x%.4f so the %s axis is %.2f m' % (s, axis, target),
                   'kit vertex colours / palette texture kept in COLOR_0 by s81 load_one',
                   'base dropped to y=0 and XZ centred by s81_prop_assets.bake'])
        os.unlink(tmp)
        print('%-16s %-28s %5.2f x %5.2f x %5.2f m' % (cls, member, *ext2), flush=True)
    for cls, fn in (('planter', gen_planter), ('utility_cabinet', gen_cabinet)):
        m = fn()
        m.export(AD + '/src/%s.glb' % cls)
        ext = m.bounds[1] - m.bounds[0]
        assets[cls] = dict(file='%s.glb' % cls,
                           nominal_size_m=[round(float(v), 3) for v in ext],
                           facing='+Z',
                           source='original model built from trimesh primitives '
                                  'by lib/objects/objassets.py',
                           page_url=None, download_url=None, archive_sha256=None,
                           licence='CC0 1.0, original model, no third-party asset',
                           original_mesh=None, edits=['built from boxes'])
        print('%-16s %-28s %5.2f x %5.2f x %5.2f m' % (cls, 'generated', *ext), flush=True)
    json.dump(dict(assets=assets, generated=time.strftime('%Y-%m-%dT%H:%M:%S'),
                   kits={k: dict(v, sha256=sha256(KD + '/' + v['zip']))
                         for k, v in KITS.items()}),
              open(AD + '/sources.json', 'w'), indent=1)
    # bake with s81's machinery, pointed at this library
    S81.AD = AD
    S81.CELLS = 5
    S81.CELL_PX = S81.TEX_PX // 5
    S81.POLE_BASE_CENTRED = ('utility_pole', 'traffic_light', 'sign_board',
                             'traffic_cone')
    doc = S81.bake()
    os.rename(AD + '/prop_assets.npz', AD + '/object_assets.npz')
    os.rename(AD + '/prop_palette.jpg', AD + '/object_palette.jpg')
    doc['texture']['file'] = 'object_palette.jpg'
    doc['npz'] = 'object_assets.npz'
    json.dump(doc, open(AD + '/manifest.json', 'w'), indent=1)
    print('library: %d classes' % len(doc['assets']), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['fetch', 'build'])
    a = ap.parse_args()
    if a.mode == 'fetch':
        fetch()
    else:
        fetch(); build()


if __name__ == '__main__':
    main()
