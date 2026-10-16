#!/usr/bin/env python3
"""No-COLMAP build mode helpers (`opencity build AREA`, build_mode poses).

The photo mesh is optional: the texture comes from panoramas projected onto the
lidar + OSM geometry, and that needs panorama poses, not a COLMAP mesh.

    nomesh.py manifest AREA    pose-only "photo dir" for fuse_context --no-photo:
                               manifest.json with the ENU origin and attribution
    nomesh.py poses AREA       W/pano_poses.json, W/panos.json, W/pano_convention.json
                               from Mapillary's own computed poses, registered to the
                               lidar ground (needs the fused surfaces.npz)
    nomesh.py empty-geom AREA  G/ as a zero-face photo mesh, so stages that read the
                               photo mesh arrays see "no photo surface" instead of
                               crashing on missing files

Pose recipe (pano_poses.json against a lidar DTM):
  * x, y, yaw/pitch/roll: Mapillary computed_geometry + computed_rotation.
    Mapillary computed poses are typically ~1-2 m and ~0.3 deg off a
    skyline-registered pose (median 1.7 m / 0.3 deg); this mode does no skyline
    refinement, so every pose is written with confidence 'usable' and method
    'mapillary_computed+lidar_z'.
  * z: lidar DTM under the camera + CAM_H_M, a typical street-level panorama
    camera height above the ground (median 2.26 m, p10 1.83 m, p90 2.77 m).
  * equirect convention: wc_xyz, the convention whose projection best matches the
    photo colours (corr 0.221 vs <= 0.119 for the others).  It is a property of the
    Mapillary API, not of the area, so it is fixed here, not re-measured.
Runs on the CPU (file IO and small numpy).
"""
import json, math, os, sys
try:
    import opencity_paths as OCP                # host roots, env-overridable
except ImportError:                            # imported with lib/ off sys.path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
    import opencity_paths as OCP
if __name__ == '__main__':                       # run as ./nomesh.py under /usr/bin/env python3
    OCP.reexec_in_venv()
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import areacfg as A                                   # noqa: E402

CAM_H_M = 2.262
CONVENTION = dict(best='wc_xyz',
                  source='fixed: a property of the Mapillary API, not of the area')


def rodrigues(v):
    v = np.asarray(v, 'f8'); th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def lla_to_enu(lat, lon, alt, o):
    a, e2 = 6378137.0, 6.69437999014e-3

    def ecef(la, lo, h):
        la, lo = np.radians(la), np.radians(lo); s = np.sin(la)
        n = a / np.sqrt(1 - e2 * s * s)
        return (n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo), (n * (1 - e2) + h) * s
    x, y, z = ecef(np.asarray(lat, 'f8'), np.asarray(lon, 'f8'), np.asarray(alt, 'f8'))
    x0, y0, z0 = ecef(o[0], o[1], o[2])
    dx, dy, dz = x - x0, y - y0, z - z0
    la, lo = math.radians(o[0]), math.radians(o[1])
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    return np.stack([-so * dx + co * dy, -sl * co * dx - sl * so * dy + cl * dz,
                     cl * co * dx + cl * so * dy + sl * dz], -1)


def cfgp(name):
    c = A.load(name)
    return c, c['paths'], c['paths'].get('scratch_dir')


def cmd_manifest(name):
    c, p, sd = cfgp(name)
    ctx = p['CTX']
    st = json.load(open(f'{ctx}/lidar_stats.json'))
    o = c['area']['origin']
    alt = o['alt_m']
    if c['area'].get('origin_provisional', True) or not alt:
        # lidar frame: put the origin at the mean lidar ground height so
        # ground sits near z = 0
        alt = round(float(st['ground']['z_mean']), 3)
    origin = [float(o['lat']), float(o['lon']), float(alt)]
    creators = set()
    meta = f"{p['PANO']}/metadata/images.jsonl"
    if os.path.exists(meta):
        for l in open(meta):
            cr = (json.loads(l).get('creator') or {})
            if cr.get('username'):
                creators.add(cr['username'])
    out = p['SRCM']
    os.makedirs(out, exist_ok=True)
    json.dump({'area': name, 'tool': 'pipeline/nomesh.py (no COLMAP)',
               'crs': {'type': 'ENU metres', 'origin_lat_lon_alt': origin,
                       'note': 'lidar-defined frame: NAVD88 heights about this origin; no photo reconstruction'},
               'georeference_error_m': None, 'bbox_wsen': c['area']['bbox_wsen'],
               'attribution': {'licence': 'CC BY-SA 4.0 (Mapillary)', 'creators': sorted(creators)}},
              open(f'{out}/manifest.json', 'w'), indent=1)
    # record the frame in the area config so common.ORIGIN matches fuse's frame
    y = A._yaml(); txt = open(c['_file']).read()
    d = y.safe_load(txt)
    want = dict(origin={'lat': origin[0], 'lon': origin[1], 'alt_m': origin[2]},
                origin_provisional=False,
                frame='lidar (no-mesh mode): NAVD88 heights about the origin')
    # rewrite only when the frame changed, so a rebuild leaves the file (and its comments) untouched
    if any(d['area'].get(k) != v for k, v in want.items()):
        d['area'].update(want)
        head = ''.join(l for l in txt.splitlines(True) if l.startswith('#'))
        with open(c['_file'], 'w') as f:
            f.write(head); y.safe_dump(d, f, sort_keys=False, default_flow_style=None, width=100)
    print(f'wrote {out}/manifest.json, origin {origin}, {len(creators)} creators')


def cmd_poses(name):
    c, p, sd = cfgp(name)
    o = c['area']['origin']; org = (o['lat'], o['lon'], o['alt_m'])
    S = np.load(f"{p['FW']}/surfaces.npz")
    x0, y0, gc, nx, ny = [float(v) for v in S['grid']]; nx, ny = int(nx), int(ny)
    dtm = S['dtm']
    rows, poses = [], {}
    for l in open(f"{p['PANO']}/metadata/images.jsonl"):
        d = json.loads(l)
        if d.get('camera_type') != 'spherical':
            continue
        g = d.get('computed_geometry') or d.get('geometry'); rot = d.get('computed_rotation')
        if not g or not rot:
            continue
        lon, lat = g['coordinates']
        e = lla_to_enu(lat, lon, org[2], org)
        x, y = float(e[0]), float(e[1])
        i, j = int((y - y0) / gc), int((x - x0) / gc)
        if not (0 <= i < ny and 0 <= j < nx) or not np.isfinite(dtm[i, j]):
            continue
        z = float(dtm[i, j]) + CAM_H_M
        R = rodrigues(rot)                    # wc_xyz: X = R @ (P - C)
        rows.append(dict(id=str(d['id']), lat=lat, lon=lon, alt=z + org[2], rot=rot,
                         w=d.get('width'), h=d.get('height'), x=x, y=y, z=z))
        poses[str(d['id'])] = dict(x=x, y=y, z=z, R=R.tolist(), method='mapillary_computed+lidar_z',
                                   confidence='usable', residual_deg=9.9, loss=1.0)
    W = p['W']
    os.makedirs(W, exist_ok=True)
    json.dump(rows, open(f'{W}/panos.json', 'w'))
    json.dump(dict(CONVENTION, panos_total=len(rows), panos_in_box=len(rows)),
              open(f'{W}/pano_convention.json', 'w'), indent=1)
    meta = dict(frame=f'lidar ENU about {list(org)}', method=(
        f'Mapillary computed pose, z = lidar DTM + {CAM_H_M} m (typical street-level panorama camera height); '
        'no skyline refinement (Mapillary computed poses are typically ~1.7 m / 0.3 deg off a skyline-registered pose)'),
        counts={'usable': len(poses)}, n_panoramas=len(poses),
        rotation='R is world->camera, X = R @ (P - C), convention wc_xyz')
    json.dump(dict(meta=meta, poses=poses), open(f'{W}/pano_poses.json', 'w'))
    print(f'wrote {W}/pano_poses.json: {len(poses)} panoramas (of the fetched set) inside the fused grid')


def cmd_empty_geom(name):
    c, p, sd = cfgp(name)
    G = p['G']; os.makedirs(G, exist_ok=True)
    f3 = np.zeros((0, 3), 'f4')
    arrays = dict(pos=f3, faces=np.zeros((0, 3), 'i4'), centroid=f3, uv=np.zeros((0, 2), 'f4'),
                  facetile=np.zeros(0, 'i4'), face_hag=np.zeros(0, 'f4'), face_dlidar=np.zeros(0, 'f4'))
    for k, v in arrays.items():
        np.save(f'{G}/{k}.npy', v)
    json.dump({'note': 'no photo mesh (no-mesh mode): zero-face arrays'}, open(f'{G}/NO_PHOTO_MESH.json', 'w'))
    # with no photo mesh the face-class outputs (geom/face_layer.npy, faceclass.json) are empty
    W = p['W']; os.makedirs(f'{W}/geom', exist_ok=True)
    np.save(f'{W}/geom/face_layer.npy', np.zeros(0, 'i1'))
    json.dump({'faces': 0, 'note': 'no photo mesh: zero photo faces (pipeline/nomesh.py)'},
              open(f'{W}/faceclass.json', 'w'), indent=1)
    print(f'wrote zero-face photo mesh arrays to {G}')


def cmd_dsm(name):
    """<scratch>/pano_pose/dsm/dsm_fine.npz (0.5 m) and dsm_wide.npz (2 m) from the
    area's registered lidar, with pipeline/dsm_build.build() (same max-Z DSM,
    class-2 min DTM, class-9 water).  Here the wide grid is the area's own lidar, so
    occlusion beyond the fetched box is not modelled; cmd_dsm_wide replaces it with a
    3 km coarse EPT fetch for far skyline occluders."""
    from importlib.machinery import SourceFileLoader
    import importlib.util as U
    c, p, sd = cfgp(name)
    spec = U.spec_from_loader('pp_dsm_build', SourceFileLoader('pp_dsm_build', f'{HERE}/dsm_build.py'))
    DB = U.module_from_spec(spec); spec.loader.exec_module(DB)
    xyz = np.load(f"{p['FW']}/lidar_reg.npy").astype('f8')
    cls = np.load(f"{p['FW']}/loaded.npz")['lidar_cls']
    out = f'{sd}/pano_pose/dsm'; os.makedirs(out, exist_ok=True)
    lo, hi = xyz[:, :2].min(0), xyz[:, :2].max(0)
    for tag, cell, pad in (('fine', 0.5, 2.0), ('wide', 2.0, 40.0)):
        x0, y0 = np.floor((lo - pad) / cell) * cell
        nx, ny = int(np.ceil((hi[0] + pad - x0) / cell)), int(np.ceil((hi[1] + pad - y0) / cell))
        r = DB.build(xyz, cls, float(x0), float(y0), cell, nx, ny, want_water=(tag == 'fine'))
        np.savez_compressed(f'{out}/dsm_{tag}.npz', **r)
        print(f'dsm_{tag}: {nx} x {ny} cells of {cell} m, {int(np.isfinite(r["dsm"]).sum()):,} with returns')


def cmd_dsm_wide(name):
    """<scratch>/pano_pose/dsm/dsm_wide.npz: a 3 x 3 km, 2 m max-Z DSM about the area
    origin from a coarse-LOD EPT fetch of the area's own lidar dataset (lidar_catalog.json),
    so far skyline occluders (250-1450 m out) are in the DSM.  pipeline/dsm_build
    stage_fetch + build, with OUT, EPT and the registration set to the area's own
    (identity in poses mode).  The local 2 m grid is kept as dsm_wide_local.npz; the
    fetched PLY is deleted after use."""
    import importlib.util as U, math
    from importlib.machinery import SourceFileLoader
    c, p, sd = cfgp(name)
    spec = U.spec_from_loader('pp_dsm_build', SourceFileLoader('pp_dsm_build', f'{HERE}/dsm_build.py'))
    DB = U.module_from_spec(spec); spec.loader.exec_module(DB)
    out = f'{sd}/pano_pose/dsm'; os.makedirs(out, exist_ok=True)
    cat = json.load(open(f"{p['CTX']}/lidar_catalog.json"))
    url = (cat.get('probed') or cat['candidates'])[0].get('ept_url') or cat['candidates'][0]['url']
    DB.OUT, DB.EPT = out, url
    import common as CC                     # the area's frame (OPENCITY_AREA is set by the CLI)
    th = math.radians(float(CC.REG_YAW_DEG)); t = np.asarray(CC.REG_T, 'f8')
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
    DB.register = lambda enu: enu @ Rz.T + t
    if os.path.exists(f'{out}/dsm_wide.npz') and not os.path.exists(f'{out}/dsm_wide_local.npz'):
        os.rename(f'{out}/dsm_wide.npz', f'{out}/dsm_wide_local.npz')
    ply = DB.stage_fetch()
    xyz, cls = DB.ply_to_registered(ply)
    n = int(2 * DB.WIDE_HALF / DB.WIDE_CELL)
    r = DB.build(xyz, cls, -DB.WIDE_HALF, -DB.WIDE_HALF, DB.WIDE_CELL, n, n)
    # acceptance: the fetched surface against the area's own fine DSM
    f = np.load(f'{out}/dsm_fine.npz'); fx0, fy0, fc, fnx, fny = [float(v) for v in f['grid']]
    jj, ii = np.meshgrid(np.arange(int(fnx)), np.arange(int(fny)))
    X = fx0 + (jj + 0.5) * fc; Y = fy0 + (ii + 0.5) * fc
    wi = ((Y + DB.WIDE_HALF) / DB.WIDE_CELL).astype(int); wj = ((X + DB.WIDE_HALF) / DB.WIDE_CELL).astype(int)
    dz = r['dtm'][wi, wj] - f['dtm']
    ok = np.isfinite(dz)
    acc = dict(cells=int(ok.sum()), dtm_median_abs_dz=round(float(np.median(np.abs(dz[ok]))), 3),
               dtm_median_signed=round(float(np.median(dz[ok])), 3), ept=url,
               filled=round(float(np.isfinite(r['dsm']).mean()), 3), points=int(len(xyz)))
    json.dump(acc, open(f'{out}/accept_wide.json', 'w'), indent=1)
    print('dsm_wide accept', json.dumps(acc))
    if not (acc['cells'] > 1000 and acc['dtm_median_abs_dz'] < 0.5):
        sys.exit('dsm-wide: the fetched DTM does not agree with the area lidar; kept the local grid')
    np.savez_compressed(f'{out}/dsm_wide.npz', **r)
    os.remove(ply)
    print(f'dsm_wide: {n} x {n} cells of {DB.WIDE_CELL} m')


if __name__ == '__main__':
    {'manifest': cmd_manifest, 'poses': cmd_poses, 'empty-geom': cmd_empty_geom,
     'dsm': cmd_dsm, 'dsm-wide': cmd_dsm_wide}[sys.argv[1]](sys.argv[2])
