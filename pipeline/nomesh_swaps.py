"""The swap chain: replace placeholder nodes in the GLB s22 writes with CC0 models,
on C.OUT/model.glb in place:

  s82 props     s80 inventory (OSM + city inventories + lidar + photo) -> CC0 props
  s85 objects   s84 inventory (OSM + lidar + photo + aerial) -> cars, kiosks, poles
  s94 joints    sidewalk panel joints on the concrete class (common.GROUND_JOINTS)

Trees keep their placeholder cylinders.  A step that fails keeps the previous file
and is reported.

  usage: OPENCITY_AREA=<name> python pipeline/nomesh_swaps.py [--steps s82,s85,s94]
"""
import os, sys, json, shutil, subprocess, argparse, time
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R + '/lib')
if not os.environ.get('OPENCITY_AREA', '').strip():
    sys.exit('nomesh_swaps: OPENCITY_AREA must name an area')
import common as C                                    # noqa: E402
AP = C._areacfg.module_paths(C)
if AP is None:
    sys.exit('nomesh_swaps: the area config has no paths.scratch_dir')
PY = sys.executable
REP = AP['SCR'] + '/swaps'


def steps_for(names):
    S = {
        's82': ['s82_prop_swap.py', 'swap', '@IN', '@OUT', '--report', REP + '/s82.json'],
        's85': ['s85_object_place.py', 'place', '@IN', '@OUT', '--report', REP + '/s85.json'],
        's94': ['s94_ground_joints.py', '--in', '@IN', '--out', '@OUT', '--report', REP + '/s94.json'],
    }
    return [(n, S[n]) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', default='s82,s85,s94')
    ap.add_argument('--model', default=C.OUT + '/model.glb')
    a = ap.parse_args()
    os.makedirs(REP, exist_ok=True)
    src = a.model
    base = AP['SCR'] + '/swaps/work'
    cur, done, failed = src, [], []
    env = dict(os.environ)                       # PYTHONPATH (lib/, stages/) comes from opencity_paths
    for name, cmd in steps_for(a.steps.split(',')):
        o = f'{base}_{name}.glb'
        c = [PY] + [os.path.join(R, 'stages', cmd[0])] + [o if x == '@OUT' else (cur if x == '@IN' else x) for x in cmd[1:]]
        t = time.time()
        r = subprocess.run(c, cwd=C.W, env=env, capture_output=True, text=True)
        open(f'{REP}/{name}.log', 'w').write(r.stdout + r.stderr)
        if r.returncode != 0 or not os.path.exists(o):
            print(f'  {name} failed (rc {r.returncode}), keeping {os.path.basename(cur)}: '
                  f'{(r.stderr or r.stdout).strip().splitlines()[-1:]}', flush=True)
            failed.append(name); continue
        if cur != src:
            os.unlink(cur)
        cur = o; done.append(name)
        print(f'  {name} ok {time.time() - t:.0f}s, {os.path.getsize(o) / 1e6:.1f} MB', flush=True)
    if cur != src:
        shutil.move(cur, src)
    json.dump(dict(model=src, done=done, failed=failed), open(REP + '/swaps.json', 'w'), indent=1)
    print(f'swaps: done {done}, failed {failed} -> {src}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
