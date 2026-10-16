"""Facade chain glue for an area build.

The facade chain runs s33 -> s34 -> s35 -> prewp -> s60 (rects..atlas) -> s35 -> s60
(bake, relevel, atlas), then swaps the walls set into facades/ before s22.  This
file provides the two glue steps (prewp, swap).  `swap` calls
s62_wall_integrate.swap(), which reads only texlib.SCR and wpcommon paths (both
per-area under OPENCITY_AREA).

  prewp   snapshot s35's atlases into facades/_pre_wallphoto/ (wpatlas's tone
          reference); an existing snapshot is replaced
  swap    copy the walls pack over facades/ (s62's stale-set check included)
"""
import os, sys, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))

if not os.environ.get('OPENCITY_AREA', '').strip():
    sys.exit('nomesh_facades: OPENCITY_AREA must name an area')
import common as C                                    # noqa: E402
if C._areacfg.build_area_cfg(C) is None:
    sys.exit('nomesh_facades: the area config has no paths.scratch_dir')
import texlib as T                                    # noqa: E402

FAC = T.SCR + '/facades'
BAK = FAC + '/_pre_wallphoto'


def prewp():
    if os.path.isdir(BAK):
        shutil.rmtree(BAK)
    os.makedirs(BAK)
    n = 0
    for f in sorted(os.listdir(FAC)):
        if f.startswith('atlas_') or f in ('facadetex.json', 'wallmap.npz'):
            shutil.copy2(FAC + '/' + f, BAK + '/' + f); n += 1
    print(f'prewp: {n} files -> {BAK}')


def swap(setname='walls'):
    import s62_wall_integrate as S62
    r = S62.swap(setname)
    print('swap:', json.dumps(r))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'prewp':
        prewp()
    elif cmd == 'swap':
        swap(*(sys.argv[2:3] or ['walls']))
    else:
        sys.exit(__doc__)
