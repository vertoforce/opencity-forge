"""opencity area commands: new-area, fetch, build, derive, check, publish, areas, stage.

Each area has areas/<name>/area.yaml (pipeline/areacfg.py) and its data under
$OPENCITY_WORK/areas/<name>/.  These commands chain the tools below:

  fetch   fetch/fetch_cc0.py                 shared CC0 material tiles (once)
          pipeline/fetch_panos.py            Mapillary panoramas (CC BY-SA)
          fetch/fetch_context.py             USGS 3DEP lidar + OSM
          pipeline/fetch_ortho.py            NAIP ortho (public domain)
  build   fetch/fetch_area.py, fetch/fuse_context.py, pipeline/nomesh.py
                                             manifest, one ENU frame, panorama poses, lidar DSM
          pipeline/derive_knobs.py           data-derived knobs -> derived.yaml
          POSES_STAGES                       the stage chain (OPENCITY_AREA set, the repo read-only)
  check   gates on the built model
  publish copy the model into OPENCITY_PUBLISH

Long commands take --detach: setsid nohup, pid file and a completion marker in
<scratch>/run/ (<cmd>.pid, <cmd>.done or <cmd>.failed, log in <scratch>/logs/).
"""
import argparse, json, math, os, re, shutil, subprocess, sys, time
try:
    import opencity_paths as OCP                # host roots, env-overridable
except ImportError:                            # imported with lib/ off sys.path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
    import opencity_paths as OCP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import areacfg as A                                   # noqa: E402

PY = OCP.PY
FETCH_DIR = OCP.FETCH
SCRATCH_ROOT = OCP.WORK + '/areas'
PUBLISH_ROOT = OCP.PUBLISH   # publish root (OPENCITY_PUBLISH)
GLTF = OCP.GLTF_TRANSFORM
# paths an area stage must never write (bound read-only in its namespace)
PROTECT = [OCP.CODE, OCP.COMBINED,
           OCP.DATA]
# s02 queries public Overpass mirrors, which can answer 504 under load; retry the
# stage whole, with a long pause, rather than fail the build.
STAGE_RETRIES = {'s02_osm.py': (3, 300)}


def restricted(src):
    """A source block from area.yaml sources: its terms forbid redistribution or sale
    when it says `restricted: true`.  license public excludes such a source."""
    return isinstance(src, dict) and bool(src.get('restricted'))


def stage_path(script):
    """A chain entry's file: a bare sNN_*.py lives in stages/, anything else is repo-relative."""
    return os.path.join(ROOT, script) if '/' in script else os.path.join(OCP.STAGES, script)


def stage_skip(cfg, stage):
    """Reason to skip a stage or output check on this area, from its derived knobs, or None."""
    sk = cfg.get('stage_knobs') or {}
    if stage == 'water' and 'WATER_Z' in sk and sk['WATER_Z'].get('value') is None:
        return 'no water body (derived WATER_Z is null: ' + str(sk['WATER_Z'].get('reason', ''))[:80] + ')'
    return None


def die(msg):
    sys.exit(f'opencity: {msg}')


def parse_bbox(s):
    try:
        w, s_, e, n = (float(v) for v in s.split(','))
    except ValueError:
        die(f'bad --bbox {s!r}; write --bbox=W,S,E,N')
    if not (w < e and s_ < n):
        die('bbox must be W,S,E,N with W<E and S<N')
    return w, s_, e, n


def bbox_size_m(b):
    w, s, e, n = b
    lat = math.radians((s + n) / 2)
    return (e - w) * 111320 * math.cos(lat), (n - s) * 110574


def cfg_of(name):
    try:
        return A.load(name)
    except A.AreaConfigError as e:
        die(str(e))


def scratch(cfg):
    return cfg['paths'].get('scratch_dir') or os.path.join(SCRATCH_ROOT, cfg['area']['name'])


def bbox_arg(cfg):
    b = cfg['area'].get('fetch_bbox_wsen') or cfg['area']['bbox_wsen']
    return ','.join(f'{v:.6f}' for v in b)


# ------------------------------------------------------------ detach helper
def detach(cfg, job, argv):
    sd = scratch(cfg)
    run, logs = os.path.join(sd, 'run'), os.path.join(sd, 'logs')
    os.makedirs(run, exist_ok=True); os.makedirs(logs, exist_ok=True)
    pidf = os.path.join(run, f'{job}.pid')
    if os.path.exists(pidf):
        try:
            pid = int(open(pidf).read().split()[0]); os.kill(pid, 0)
            die(f'{job} already running, pid {pid} ({pidf})')
        except (ProcessLookupError, ValueError):
            pass
    for ext in ('.done', '.failed'):
        p = os.path.join(run, job + ext)
        if os.path.exists(p):
            os.remove(p)
    log = os.path.join(logs, f'{job}.log')
    me = [PY, os.path.join(HERE, 'cli.py')] + [a for a in argv if a != '--detach']
    sh = (f'echo $$ > {pidf}; {" ".join(map(_q, me))} >> {_q(log)} 2>&1; rc=$?; '
          f'if [ $rc = 0 ]; then date -Is > {run}/{job}.done; else echo "rc=$rc $(date -Is)" > {run}/{job}.failed; fi; '
          f'rm -f {pidf}')
    p = subprocess.Popen(['setsid', 'nohup', 'bash', '-c', sh], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        if os.path.exists(pidf):
            break
        time.sleep(0.1)
    pid = open(pidf).read().strip() if os.path.exists(pidf) else f'? (launcher {p.pid})'
    print(f'{job}: detached pid {pid}\n  log    {log}\n  marker {run}/{job}.done (or {job}.failed)')


def _q(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


def run(cmd, env=None, cwd=None):
    print('+', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, cwd=cwd)
    if r.returncode:
        die(f'step failed (exit {r.returncode}): {os.path.basename(cmd[1] if len(cmd) > 1 else cmd[0])}')


# ------------------------------------------------------------ new-area
def cmd_new_area(a):
    name = a.name
    if not name.replace('_', '').isalnum() or name.lower() != name:
        die('area name: lower-case letters, digits and _')
    b = parse_bbox(a.bbox)
    wm, hm = bbox_size_m(b)
    if wm * hm > 1.5e6 and not a.force:
        die(f'bbox is {wm:.0f} x {hm:.0f} m; the pipeline is sized for a box of about 1 km2 or less. --force to override')
    p = A.area_file(name)
    if os.path.exists(p) and not a.force:
        die(f'{p} exists (--force to overwrite)')
    sd = os.path.join(SCRATCH_ROOT, name)
    lat0, lon0 = (b[1] + b[3]) / 2, (b[0] + b[2]) / 2
    lic = a.license
    ortho = 'naip'
    tz = a.timezone or guess_us_timezone(lon0, lat0)
    sys.path.insert(0, HERE); import city_trees; _ct = city_trees.pick(b)   # a city tree layer covering the box
    zone = int((lon0 + 180) // 6) + 1
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    wk = os.path.join(sd, 'work')
    cfg = {
        'schema_version': 2,
        'derive': True,
        'license': lic,
        'area': {
            'name': name, 'title': a.title or name.replace('_', ' ').title(),
            'bbox_wsen': [round(v, 6) for v in b],
            'fetch_bbox_wsen': [round(v, 6) for v in b],
            # provisional: bbox centre at 0 m.  `opencity build` (nomesh.py manifest)
            # sets the height to the mean lidar ground height of the box.
            'origin': {'lat': round(lat0, 6), 'lon': round(lon0, 6), 'alt_m': 0.0},
            'origin_provisional': True,
            'crs_epsg': epsg,
            'timezone': tz,
        },
        'paths': {
            'scratch_dir': sd, 'work_dir': wk, 'publish_name': a.publish_name or f'opencity_{name}',
            'W': wk, 'CMB': os.path.join(sd, 'fusion', name), 'G': os.path.join(sd, 'fusion', name, 'geom'),
            'FW': os.path.join(sd, 'fusion', name), 'CTX': os.path.join(sd, 'src', name, 'context'),
            'SRCM': os.path.join(sd, 'out', name), 'PANO': os.path.join(sd, 'src', name),
            'OUT': os.path.join(sd, 'publish'), 'KEEP': os.path.join(sd, 'keep'),
        },
        'sources': {
            'mapillary': {'token_file': OCP.TOKEN_FILE, 'max_panos': a.max_panos,
                          'licence': 'CC BY-SA 4.0'},
            'lidar': {'dataset': 'auto (OpenTopography / USGS 3DEP catalog)', 'licence': 'USGS 3DEP, public domain',
                      'local': [], 'fallback': 'usgs_3dep'},
            'osm': {'licence': '(c) OpenStreetMap contributors, ODbL 1.0'},
            # NAIP (public domain) is the default everywhere.  A local ortho is declared
            # by hand (primary, licence, and restricted: true when its terms forbid sale).
            'ortho': {'primary': 'naip', 'licence': 'USDA NAIP, public domain',
                      'endpoint': 'USGS National Map USGSNAIPImagery ImageServer',
                      'local': [], 'fallback': 'naip'},
            # local first, national fallback: each input lists the local sources that
            # cover this box and the national/OSM fallback every stage can run on alone.
            'trees': {'local': ([f'{_ct}_tree_inventory'] if _ct else []),
                      'fallback': 'lidar_crowns (USGS 3DEP CHM) + OSM natural=tree'},
            'props': {'local': [],
                      'fallback': 'OSM + lidar poles + Mapillary detections'},
        },
        # poses: panoramas projected onto lidar + OSM geometry (no photo mesh)
        'build_mode': 'poses',
        'knobs': {},
        # default on for every area new-area writes
        'stage_knobs': {
            'FACADE_PHOTO_QUALITY': {
                'value': {'seg_veto': True, 'min_cos': 0.5, 'measure': True}, 'class': 'hand',
                'reason': 'pipeline/facade_quality.py: seg_veto: Mapillary '
                          'segmentation not-wall pixels carry no facade colour; min_cos 0.5: no photo '
                          'texel seen more edge-on than 60 deg'},
            'FACADE_DATA_COLOUR': {
                'value': {'enabled': True, 'k_parts': 5, 'r_m': 150.0}, 'class': 'hand',
                'reason': 'stages/s35_facadetex.py: an unphotographed wall '
                          'takes the median colour of the 5 nearest photographed buildings, not the '
                          'class-default grey'},
            'OBJECT_PLACEHOLDER_HIDE': {
                'value': True, 'class': 'hand',
                'reason': 'stages/s85_object_place.py: rows with no CC0 model '
                          'stay tagged nodes and draw no cage/box'},
            # default on (stages/s10a_layers.py)
            'WATER_MAX_RISE_M': {
                'value': 3.0, 'class': 'hand',
                'reason': 's10a_layers.WATER_MAX_RISE_M: a steep water face '
                          'more than 3 m above its local water level is a 3dfier WaterWall up a '
                          'building side and is dropped (kept rims measured <= 1.29 m)'},
        },
    }
    os.makedirs(os.path.dirname(p), exist_ok=True)
    yaml = A._yaml()
    with open(p, 'w') as f:
        f.write(f'# {name}: written by `opencity new-area`.\n'
                f'# derive: true -> `opencity build` writes derived.yaml in the scratch dir and those\n'
                f'# values become this area\'s knob defaults.  Put a hand override in knobs:\n'
                f'# with value:, class: hand and reason:.  Box {wm:.0f} x {hm:.0f} m.\n')
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None, width=100)
    for d in ('run', 'logs', 'work/markers', 'publish'):
        os.makedirs(os.path.join(sd, d), exist_ok=True)
    print(f'wrote {p}\n  box {wm:.0f} x {hm:.0f} m, EPSG:{epsg}, ortho: {ortho}, timezone: {tz}'
          f'{"" if a.timezone else " (looked up from the box centre; pass --timezone to set it)"}, '
          f'city trees: {_ct or "none"}\n  data under {sd}\n  next: opencity fetch {name} --detach')


def guess_us_timezone(lon, lat):
    """Default IANA zone for new-area: an offline polygon lookup (timezonefinder) at the
    box centre, else a coarse contiguous-US guess from longitude with Arizona on
    America/Phoenix.  new-area prints the result and --timezone overrides it."""
    # an offline polygon lookup first: longitude alone puts Arizona in
    # America/Denver, which keeps daylight time, so the sun would be 1 h off.
    try:
        from timezonefinder import TimezoneFinder
        z = TimezoneFinder().timezone_at(lng=lon, lat=lat)
        if z:
            return z
    except ImportError:
        print('timezone: timezonefinder not installed; guessing from longitude', flush=True)
    if not (24 <= lat <= 50 and -125 <= lon <= -66):
        die('outside the contiguous US: pass --timezone (IANA name)')
    if -114.82 <= lon <= -109.05 and 31.33 <= lat <= 37.0:
        return 'America/Phoenix'       # Arizona keeps standard time (the Navajo Nation does not)
    for edge, z in ((-114.5, 'America/Los_Angeles'), (-102.0, 'America/Denver'),
                    (-86.5, 'America/Chicago')):
        if lon < edge:
            return z
    return 'America/New_York'


# ------------------------------------------------------------ fetch
def cmd_fetch(a, argv):
    cfg = cfg_of(a.area)
    if a.detach:
        return detach(cfg, 'fetch', argv)
    sd, name = scratch(cfg), cfg['area']['name']
    if not os.path.exists(OCP.SHARED + '/texture/cc0/index.json'):   # shared CC0 material tiles
        run([PY, os.path.join(ROOT, 'fetch', 'fetch_cc0.py')])
    bb = bbox_arg(cfg)
    src = cfg['sources']
    if not a.skip_mapillary:
        # metadata first, thin, then download only the selection: one street can hold
        # thousands of panoramas, tens of GB at original quality.
        m = src['mapillary']
        run([PY, os.path.join(HERE, 'fetch_panos.py'), f'--bbox={bb}', '--out', f'{sd}/src/{name}',
             '--max-panos', str(m.get('max_panos', 300)), '--spacing', str(m.get('pano_spacing_m', 3.0))])
    if not a.skip_context:
        run([PY, f'{FETCH_DIR}/fetch_context.py', f'{name}={bb}', '--work-root', f'{sd}/context_work',
             '--src-root', f'{sd}/src'], cwd=FETCH_DIR)
    osrc = src.get('ortho', {})
    prim = osrc.get('primary')
    lic = a.license or cfg.get('license', 'private')
    if restricted(osrc) and lic == 'public':
        print(f'ortho: license public excludes the restricted source {prim}; falling back to naip', flush=True)
        prim = 'naip'
    od = f'{sd}/ortho'
    have = (json.load(open(f'{od}/source.json')).get('source') or 'local') if os.path.exists(f'{od}/source.json') else None
    if prim == 'naip' and have and have != 'naip' and not a.skip_ortho:     # never mix two sources' tiles in one dir
        aside = f'{sd}/ortho_{have}'; shutil.rmtree(aside, ignore_errors=True); os.rename(od, aside)
        print(f'ortho: {have} tiles moved to {aside}', flush=True)
    if prim == 'naip' and not a.skip_ortho:
        run([PY, os.path.join(HERE, 'fetch_ortho.py'), f'--bbox={bb}', '--out', f'{sd}/ortho', '--source', 'naip']
            + (['--epsg', str(cfg['area']['crs_epsg'])] if cfg['area'].get('crs_epsg') else []))
    elif prim and not a.skip_ortho:
        print(f'ortho: {prim} is a local source; its tiles and source.json must already be in {od}', flush=True)
    print(f'fetch {name}: done')


# ------------------------------------------------------------ build
def _sandbox_cmd(cmd, cfg):
    """Run cmd in a private mount namespace where the PROTECT paths (repo, DATA, default COMBINED) are read-only."""
    binds = ' '.join(f'mount --bind {_q(p)} {_q(p)} && mount -o remount,bind,ro {_q(p)} &&'
                     for p in PROTECT if os.path.exists(p))
    return ['unshare', '-m', '--propagation', 'private', 'bash', '-c',
            f'{binds} exec {" ".join(map(_q, cmd))}']


def cmd_build(a, argv):
    cfg = cfg_of(a.area)
    if a.detach:
        return detach(cfg, 'build', argv)
    return build_poses(cfg, a.steps.split(',') if a.steps else POSES_STEPS, getattr(a, 'from_stage', None))


# ------------------------------------------------------------ build
# No COLMAP: poses from Mapillary + lidar, geometry from lidar + OSM (3dfier), the
# texture projected from the panoramas.  Stage flags skip the passes that exist
# only to repair or blend a photo mesh.
POSES_STEPS = ['manifest', 'fuse', 'poses', 'geom', 'dsm', 'derive', 'stages', 'web']
POSES_STAGES = [
    ('s02_osm.py', []), ('s03b_city_trees.py', []), ('s03_trees.py', []),
    ('s04_tree_merge.py', []), ('s05_inventory.py', []), ('s05b_inventory_join.py', []),
    ('s10a_3dfier.py', []), ('s19_delight.py', []),
    # Mapillary per-image detections + transient masks (people, cars) for the facade veto
    # and the s80/s84 photo passes.  Graph API, token read from file, 4 workers.
    ('s40_masks.py', ['--workers', '4']),
    ('s20_project.py', []),
    # ground colour without a photo mesh: gated panorama + ortho tint rasters, then the
    # s50 class tint and the s22 world tint.  s22 must not run --no-floor-tint --no-classtint
    # here: that fans one colour per 3dfier sliver face across the TIN (streaks) and
    # leaves the class tiles at the 0.88 cap.
    ('pipeline/nomesh_ground.py', ['tint']), ('s21_material.py', []),
    ('pipeline/nomesh_ground.py', ['classes']), ('s50_classtint.py', []),
    # facades: s33..s60, then the facade swap, on the Mapillary poses; the wallfill,
    # glass grid, one-colour and tone rules come with s34/s35/s60.
    ('s33_walls.py', []), ('s34_wallproj.py', []), ('s35_facadetex.py', []),
    ('pipeline/nomesh_facades.py', ['prewp']),
] + [('s60_wall_bestview.py', [c, '--force', '--workers', '8'])
     for c in ('rects', 'cover', 'chips', 'align', 'refine', 'own', 'bake', 'relevel', 'atlas')] + [
    ('s35_facadetex.py', []),
] + [('s60_wall_bestview.py', [c, '--force', '--workers', '8'])
     for c in ('bake', 'relevel', 'atlas')] + [
    ('pipeline/nomesh_facades.py', ['swap']),
    # props and objects: OSM + optional city inventories (pipeline/city_props.py) + lidar
    # + photo -> s80; s84 adds the aerial pass on the ENU ortho; the CC0 kits are baked
    # per area; the swaps run after s22.
    ('pipeline/nomesh_ground.py', ['ortho-enu']), ('pipeline/city_props.py', []),
    ('s80_furniture_inventory.py', ['all', '--rebuild']), ('s81_prop_assets.py', ['bake']),
    ('lib/objects/objassets.py', ['build']), ('s84_object_inventory.py', ['--rerun']),
    ('s22_composite.py', ['--no-ortho', '--no-stitch', '--no-scraps', '--no-photo-ground']),
    ('pipeline/nomesh_swaps.py', []),
]
NOMESH = os.path.join(HERE, 'nomesh.py')


def restore_panos(cfg):
    """An area's images_pano may be deleted to save disk, and s19/s20/s34/s60 then die on
    missing files.  Before the stages, every panorama the area's poses use
    (work/pano_poses.json) must be on disk; missing ones are fetched by id (fetch_area
    --ids-file, the call fetch_panos.py makes), and the build stops if any is still missing."""
    p = cfg['paths']
    if not p.get('PANO') or not p.get('W') or not os.path.exists(os.path.join(p['W'], 'pano_poses.json')):
        return
    pp = os.path.join(p['W'], 'pano_poses.json'); pdir = os.path.join(p['PANO'], 'images_pano')
    want = sorted(json.load(open(pp)).get('poses', {}))
    have = {f[:-4] for f in os.listdir(pdir)} if os.path.isdir(pdir) else set()
    miss = [i for i in want if i not in have]
    if not miss:
        return
    print(f'restore_panos: {len(miss)} of {len(want)} posed panoramas missing from {pdir}; fetching them by id', flush=True)
    ids = os.path.join(p['PANO'], 'metadata', 'restore_ids.txt'); os.makedirs(os.path.dirname(ids), exist_ok=True)
    open(ids, 'w').write('\n'.join(miss) + '\n')
    meta = os.path.join(p['PANO'], 'metadata', 'images.jsonl'); keep = meta + '.pre_restore'
    if os.path.exists(meta):                     # --ids-file rewrites images.jsonl to the fetched ids only
        shutil.copy2(meta, keep)
    run([PY, f'{FETCH_DIR}/fetch_area.py', f'--bbox={bbox_arg(cfg)}', '--out', p['PANO'], '--max-images', '0',
         '--pano', 'only', '--no-clusters', '--workers', '8', '--ids-file', ids, '--quality', 'original'])
    if os.path.exists(keep):
        os.replace(keep, meta)
    have = {f[:-4] for f in os.listdir(pdir)} if os.path.isdir(pdir) else set()
    still = [i for i in want if i not in have]
    if still:
        die(f'restore_panos: {len(still)} of {len(want)} posed panoramas still missing after the fetch: {still[:5]}')
    print(f'restore_panos: {len(want)} posed panoramas on disk', flush=True)


def build_poses(cfg, steps, from_stage=None):
    name, sd, p = cfg['area']['name'], scratch(cfg), cfg['paths']
    env = dict(os.environ, OPENCITY_AREA=name)   # PYTHONPATH (lib/, stages/) comes from opencity_paths
    ensure_work(cfg)
    if 'manifest' in steps:
        run([PY, NOMESH, 'manifest', name])
        cfg = cfg_of(name)
    if 'fuse' in steps:
        run([PY, f'{FETCH_DIR}/fuse_context.py', name, '--no-photo', '--photo-dir', p['SRCM'],
             '--context-dir', p['CTX'], '--work-root', f'{sd}/fusion', '--out-root', f'{sd}/out_fused',
             '--src-root', f'{sd}/src'], cwd=FETCH_DIR)
    for st, sub in (('poses', 'poses'), ('geom', 'empty-geom'), ('dsm', 'dsm')):
        if st in steps:
            run([PY, NOMESH, sub, name], env=env)
    if 'derive' in steps:
        run([PY, os.path.join(HERE, 'derive_knobs.py'), name])
    if 'stages' in steps:
        cfg = cfg_of(name)
        restore_panos(cfg)
        todo = POSES_STAGES
        if from_stage:                            # resume a failed chain at its stage
            hit = [i for i, (sc, _) in enumerate(POSES_STAGES) if os.path.basename(sc) == os.path.basename(from_stage)]
            if not hit:
                die(f'--from {from_stage}: not a poses-mode stage')
            todo = POSES_STAGES[hit[0]:]
            print(f'stages from {from_stage}: {len(todo)} of {len(POSES_STAGES)}', flush=True)
            # a resumed stage must really rerun.  s10a keeps sub-step markers
            # (3dfier_layers.done) that would otherwise skip the layer build.  Markers written
            # at or after the from-stage's last start are cleared; each stage's start time is
            # recorded below.
            md = os.path.join(p['W'], 'markers'); st0 = os.path.join(md, '_start_' + os.path.basename(todo[0][0]))
            if os.path.exists(st0):
                t0 = os.path.getmtime(st0); gone = []
                for f in os.listdir(md):
                    q = os.path.join(md, f)
                    if not f.startswith('_start_') and os.path.getmtime(q) >= t0:
                        os.remove(q); gone.append(f)
                print(f'--from: cleared {len(gone)} markers written since {time.ctime(t0)}: {sorted(gone)[:12]}', flush=True)
            else:
                print(f'--from: no start record for {todo[0][0]}; stage markers kept (sub-steps may skip)', flush=True)
            # the s10a layer sub-step (s10a_layers.py, cheap, no Docker) always reruns on
            # --from s10a: its marker predates the start record, so the clear above keeps it
            # and the layer build would be skipped.
            if os.path.basename(todo[0][0]) == 's10a_3dfier.py':
                q = os.path.join(md, '3dfier_layers.done')
                if os.path.exists(q):
                    os.replace(q, q + '.pre_from'); print('--from s10a: 3dfier_layers.done moved aside (layer build reruns)', flush=True)
        started = set()
        for script, args in todo:
            why = stage_skip(cfg, script)
            if why:
                print(f'skip {script}: {why}'); continue
            tries, pause = STAGE_RETRIES.get(script, (1, 0))
            sm = os.path.join(p['W'], 'markers', '_start_' + os.path.basename(script))
            if script not in started:              # first run of this script in this chain
                started.add(script); os.makedirs(os.path.dirname(sm), exist_ok=True); open(sm, 'w').write(time.strftime('%Y-%m-%dT%H:%M:%S'))
            for k in range(tries):
                r = subprocess.run(_sandbox_cmd([PY, stage_path(script)] + args, cfg), env=env, cwd=p['W'])
                if r.returncode == 0:
                    break
                if k + 1 < tries:
                    print(f'{script} exit {r.returncode}; retry in {pause}s', flush=True); time.sleep(pause)
            else:
                die(f'stage {script} failed')
    if 'web' in steps:
        make_web(cfg)
    print(f'build {name} (poses mode): done ({",".join(steps)})')


def model_dir(cfg):
    """Where the area's finished model lives: s22's C.OUT."""
    return cfg['paths']['OUT']


def make_web(cfg):
    """model_web.glb: model.glb with textures resized to 2048 px (gltf-transform resize; no meshopt)."""
    d = model_dir(cfg)
    src, dst = f'{d}/model.glb', f'{d}/model_web.glb'
    if not os.path.exists(src):
        die(f'no {src}')
    if not os.path.exists(GLTF):
        die(f'gltf-transform not found at {GLTF}; install it with: '
            f'npm install --prefix {OCP.GLTF_DIR} @gltf-transform/cli@4.5.0')
    # resize only: no meshopt/EXT_meshopt_compression.  A viewer that never calls
    # setMeshoptDecoder renders a meshopt web copy black.
    run([GLTF, 'resize', src, dst, '--width', '2048', '--height', '2048'])
    print(f'web copy {os.path.getsize(dst)/1e6:.1f} MB (model.glb {os.path.getsize(src)/1e6:.1f} MB)')


# ------------------------------------------------------------ derive
def cmd_derive(a):
    subprocess.run([PY, os.path.join(HERE, 'derive_knobs.py'), a.area], check=False)


# ------------------------------------------------------------ check
def _validate_sections(text):
    """Issue codes per section (ERROR, WARNING, INFO, HINT) of `gltf-transform validate`.

    The report prints every section header, so the word ERROR appears even when the
    section says 'No errors found.'; count the table rows under each header instead."""
    out, cur = {}, None
    for line in text.splitlines():
        t = line.strip()
        if t in ('ERROR', 'WARNING', 'INFO', 'HINT'):
            cur = t; out.setdefault(cur, []); continue
        m = re.match(r'^\u2502\s*([A-Z][A-Z0-9_]{3,})\s*\u2502', t)
        if cur and m and m.group(1) != 'code':
            out[cur].append(m.group(1))
    return out


def _glb_stats(p):
    import trimesh
    sc = trimesh.load(p, force='scene', process=False)
    faces = sum(len(g.faces) for g in sc.geometry.values() if hasattr(g, 'faces'))
    return dict(nodes=len(sc.geometry), faces=int(faces), extent_m=[round(float(v), 1) for v in sc.extents])


def cmd_check(a):
    cfg = cfg_of(a.area)
    name = cfg['area']['name']
    fails, rows = 0, []

    def gate(label, ok, detail=''):
        nonlocal fails
        fails += (not ok)
        rows.append(f'{"PASS" if ok else "FAIL"}  {label}  {detail}')

    fused = model_dir(cfg)
    need = ('model.glb', 'model_web.glb')
    for f in need:
        gate(f'{f} exists', os.path.exists(f'{fused}/{f}'), fused)
    if os.path.exists(f'{fused}/model.glb'):
        st = _glb_stats(f'{fused}/model.glb')
        wm, hm = bbox_size_m(cfg['area']['bbox_wsen'])
        span = max(st['extent_m'][0], st['extent_m'][2]) if len(st['extent_m']) == 3 else 0
        gate('model covers the box', span >= 0.8 * min(wm, hm), f'model {st["extent_m"]} m vs box {wm:.0f}x{hm:.0f} m')
        gate('model has geometry', st['faces'] > 1000, f'{st["faces"]:,} faces in {st["nodes"]} nodes')
        if os.path.exists(GLTF):
            r = subprocess.run([GLTF, 'validate', f'{fused}/model.glb'], capture_output=True, text=True)
            sec = _validate_sections(r.stdout)
            n_err = len(sec.get('ERROR', []))
            warn = sec.get('WARNING', [])
            codes = ', '.join(f'{c} x{warn.count(c)}' for c in sorted(set(warn)))
            gate('gltf-transform validate', r.returncode == 0 and n_err == 0,
                 f'{n_err} errors' + (f'; {len(warn)} warnings: {codes}' if warn else '') +
                 (f'; first error {sec["ERROR"][0]}' if n_err else ''))
    m = f'{fused}/manifest.json'
    if os.path.exists(m):
        mj = json.load(open(m))
        sc = mj.get('selfcheck') or mj.get('verify') or {}
        if isinstance(sc, dict) and 'ok' in sc:
            gate('fuse_context selfcheck', bool(sc['ok']))
    dp = A.derived_path(cfg)
    gate('derived knobs written', os.path.exists(dp), dp)
    # intermediate outputs: WARN, not FAIL (the model gates above decide)
    wk = cfg['paths']['W']
    for s, label, out in (('osm', 's02_osm.py output', f'{wk}/osm/osm_layers.json'),
                          ('water', 's10a water layer', f'{wk}/layers/water.npz')):
        why = stage_skip(cfg, s)
        ok = os.path.exists(out)
        rows.append(f'{"SKIP" if why else ("PASS" if ok else "WARN")}  {label}  {why or out}')
    print('\n'.join(rows))
    print(f'check {name}: {"ALL PASS" if not fails else f"{fails} FAIL"}')
    sys.exit(1 if fails else 0)


# ------------------------------------------------------------ publish
def cmd_publish(a):
    cfg = cfg_of(a.area)
    name = cfg['area']['name']
    pub = cfg['paths'].get('publish_name') or f'opencity_{name}'
    if pub.startswith('_'):
        die(f'publish name {pub!r} is reserved')
    src = model_dir(cfg)
    if not os.path.exists(f'{src}/model.glb'):
        die(f'no {src}/model.glb; run opencity build {name} first')
    used = sources_used(cfg)
    lic = a.license or cfg.get('license', 'private')
    bad = [f"{k}: {v['source']}" for k, v in used.items() if restricted(v)]
    if lic == 'public' and bad:
        die('refusing to publish as public: restricted source(s) fed this build: ' + '; '.join(bad)
            + '. Rebuild with license public (falls back to NAIP) or publish as private.')
    used['license_profile'] = lic; used['restricted_sources'] = bad
    dst = os.path.join(PUBLISH_ROOT, pub)
    if a.dry_run:
        print(f'would copy {src} -> {dst}'); return
    OCP.refuse_if_frozen(dst, 'opencity publish')
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src)):
        if f.endswith(('.glb', '.json', '.png', '.jpg', '.txt')):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f)); n += 1
    with open(os.path.join(dst, 'ATTRIBUTION.txt'), 'w') as f:
        f.write('Imagery: Mapillary contributors, CC BY-SA 4.0 (creators in the source metadata).\n'
                'Map data: (c) OpenStreetMap contributors, ODbL 1.0.\n')
        for k in ('lidar', 'ortho', 'trees', 'props'):
            f.write(f"{k.capitalize()}: {used[k]['credit']} [{used[k]['tier']}]\n")
        f.write('Derived 3D model: CC BY-SA 4.0 (share-alike from the Mapillary imagery and ODbL map data).\n')
    json.dump(used, open(os.path.join(dst, 'SOURCES.json'), 'w'), indent=1)
    print(f'licence profile {lic}; restricted sources: {bad or "none"}')
    print(f'published {n} files to {dst}')


def sources_used(cfg):
    """Which source fed each input of this build, read from the stage outputs (not the
    config): local or fallback, with its credit.  Written to SOURCES.json and
    ATTRIBUTION.txt by publish ("local first, national fallback")."""
    sd, p = scratch(cfg), cfg['paths']
    def j(path):
        try:
            return json.load(open(path))
        except Exception:
            return None
    out = {'design': 'local first, national fallback'}
    ls = j(os.path.join(p['CTX'], 'lidar_stats.json')) or {}
    out['lidar'] = dict(tier='fallback', source='usgs_3dep', dataset=ls.get('dataset'),
                        credit=f"USGS 3D Elevation Program {ls.get('dataset') or ''}, public domain".replace('  ', ' '))
    o = j(f'{sd}/ortho/source.json') or {}
    if o.get('source') == 'naip':
        out['ortho'] = dict(tier='fallback', source='naip', year=o.get('year'), gsd_m=o.get('gsd_m'),
                            credit=f"USDA NAIP {o.get('year')} ({o.get('gsd_m')} m), public domain, via USGS The National Map")
    elif o:
        oc = (cfg.get('sources') or {}).get('ortho') or {}
        out['ortho'] = dict(tier='local', source=o.get('source') or oc.get('primary') or 'local',
                            restricted=bool(o.get('restricted') or restricted(oc)),
                            credit=o.get('credit') or o.get('licence') or oc.get('licence') or 'local ortho')
    else:
        out['ortho'] = dict(tier='none', source=None, credit='no ortho')
    t = j(os.path.join(p['W'], 'inventory', 'city_trees.json')) or {}
    nt = (t.get('counts') or {}).get('combined_in_extent', 0)
    out['trees'] = (dict(tier='local', source=f"{t.get('city')}_tree_inventory", rows=nt, credit=t.get('licence') or f"{t.get('city')} open data")
                    if nt else dict(tier='fallback', source='lidar_crowns + OSM', rows=0,
                                    credit='trees from USGS 3DEP lidar crowns and OpenStreetMap'))
    cp = j(f'{sd}/props/city_props.json')
    out['props'] = (dict(tier='local', source=f"{cp.get('city')}_city_props", rows=len(cp.get('rows') or []), credit=cp.get('licence'))
                    if cp else dict(tier='fallback', source='OSM + lidar + Mapillary detections', rows=0,
                                    credit='street furniture from OpenStreetMap, lidar and Mapillary detections'))
    return out


# subdirs of W that stages write into without creating (counted from C.W + '/<dir>/' literals)
WORK_SUBDIRS = ('layers', 'geom', 'osm', 'assets', 'tmp3d', 'tex', 'tmp', 'prev', 'inventory',
                'ortho', 'markers')


# area-independent assets every area reads (CC0 class tiles): linked, never copied
SHARED_ASSETS = {'texture/cc0': OCP.SHARED + '/texture/cc0',
                 # CC0 prop kit: sources linked, each area bakes its own prop_assets.npz
                 # next to them (s81); the object kit is copied (SHARED_COPIES)
                 'work/assets/props/src': ROOT + '/assets/props/src',
                 'work/assets/props/sources.json': ROOT + '/assets/props/sources.json',
                 }


# small kits a stage writes into (lib/objects/objassets.py build writes derived GLBs into src/): copied
SHARED_COPIES = {'work/assets/objects/src': ROOT + '/assets/objects/src',
                 'work/assets/objects/sources.json': ROOT + '/assets/objects/sources.json'}


def ensure_work(cfg):
    for d in WORK_SUBDIRS:
        os.makedirs(os.path.join(cfg['paths']['W'], d), exist_ok=True)
    os.makedirs(cfg['paths']['OUT'], exist_ok=True)     # s22 writes model.glb here; nothing else creates it
    sd = scratch(cfg)
    for rel, src in SHARED_COPIES.items():
        dst = os.path.join(sd, rel)
        if os.path.islink(dst):
            os.unlink(dst)
        if not os.path.exists(dst) and os.path.exists(src):
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
    for rel, src in SHARED_ASSETS.items():
        dst = os.path.join(sd, rel)
        if not os.path.lexists(dst) and os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)


def cmd_stage(a):
    """Run one stage script for an area: OPENCITY_AREA set, the repo read-only."""
    cfg = cfg_of(a.area)
    script = a.script if a.script.endswith('.py') else a.script + '.py'
    path = script if os.path.isabs(script) else stage_path(script)
    if not os.path.exists(path):
        die(f'no stage script {path}')
    ensure_work(cfg)
    env = dict(os.environ, OPENCITY_AREA=cfg['area']['name'])
    r = subprocess.run(_sandbox_cmd([PY, path] + a.args, cfg), env=env, cwd=cfg['paths']['W'])
    sys.exit(r.returncode)


def cmd_areas(a):
    if not os.path.isdir(A.AREAS):
        return
    for n in sorted(os.listdir(A.AREAS)):
        if os.path.exists(A.area_file(n)):
            c = A.load(n)
            wm, hm = bbox_size_m(c['area']['bbox_wsen'])
            print(f'{n:20s} {wm:5.0f} x {hm:4.0f} m  derive={bool(c.get("derive"))}  {c["area"].get("title", "")}')


COMMANDS = ('new-area', 'fetch', 'build', 'derive', 'check', 'publish', 'areas', 'stage')


def main(argv):
    ap = argparse.ArgumentParser(prog='opencity', description='open map data to a photo-textured 3D model')
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('new-area', help='write areas/<name>/area.yaml')
    p.add_argument('name'); p.add_argument('--bbox', required=True, help='--bbox=W,S,E,N (note the =)')
    p.add_argument('--title'); p.add_argument('--publish-name'); p.add_argument('--timezone', help='IANA zone (default: looked up from the box centre; a longitude guess, contiguous US, without timezonefinder)')
    p.add_argument('--license', default='private', choices=['private', 'public'],
                   help='private (default): local first, best source; public: no restricted source, NAIP fallback')
    p.add_argument('--max-panos', type=int, default=300); p.add_argument('--force', action='store_true')
    p = sp.add_parser('fetch', help='pull Mapillary, lidar, OSM, ortho')
    p.add_argument('area'); p.add_argument('--detach', action='store_true')
    for s in ('mapillary', 'context', 'ortho'):
        p.add_argument(f'--skip-{s}', action='store_true')
    p.add_argument('--license', choices=['private', 'public'], help='override area.yaml license for this fetch')
    p = sp.add_parser('build', help='frame, poses, derive knobs, the stage chain, the web copy')
    p.add_argument('area'); p.add_argument('--detach', action='store_true')
    p.add_argument('--steps', help='comma list of steps: ' + ','.join(POSES_STEPS))
    p.add_argument('--from', dest='from_stage', help='start the stages step at this stage (resume a failed chain)')
    p = sp.add_parser('derive', help='derive data-driven knobs (report only for derive: false areas)')
    p.add_argument('area')
    p = sp.add_parser('check', help='gates on the built model')
    p.add_argument('area')
    p = sp.add_parser('publish', help='copy the model into OPENCITY_PUBLISH')
    p.add_argument('area'); p.add_argument('--dry-run', action='store_true')
    p.add_argument('--license', choices=['private', 'public'], help='tag (and check) this publish; default area.yaml license')
    sp.add_parser('areas', help='list configured areas')
    p = sp.add_parser('stage', help='run one stage script for an area (sandboxed)')
    p.add_argument('area'); p.add_argument('script'); p.add_argument('args', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    full = argv
    {'new-area': lambda: cmd_new_area(a), 'fetch': lambda: cmd_fetch(a, full),
     'build': lambda: cmd_build(a, full), 'derive': lambda: cmd_derive(a),
     'check': lambda: cmd_check(a), 'publish': lambda: cmd_publish(a),
     'areas': lambda: cmd_areas(a), 'stage': lambda: cmd_stage(a)}[a.cmd]()
