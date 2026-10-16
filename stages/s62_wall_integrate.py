#!/usr/bin/env python3
"""Stage 62 -- put the stage 60 wall atlases in place of the stage 35 ones.

The atlases stage 60 writes carry the same three files stage 22 reads
(atlas_NN.jpg, wallmap.npz, facadetex.json), so integration is a swap, not a
change to the compositor.  The stage 35 files it replaces are kept in
facades/_pre_wallphoto/ and `--revert` puts them back.  `opencity build` runs
it through pipeline/nomesh_facades.py swap, before s22 composites the model.
"""
import os, sys, json, time, shutil, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import opencity_paths as OCP                    # noqa: E402  host roots, env-overridable
import common as C                             # noqa: E402,F401  applies the area frame
import texlib as T                             # noqa: E402
from wallphoto import wpcommon as H            # noqa: E402

PY = OCP.PY
FAC = T.SCR + '/facades'
BAK = FAC + '/_pre_wallphoto'
FILES = ('wallmap.npz', 'facadetex.json')


def swap(setname='walls', revert=False):
    OUT = H.setdir(setname) + '/out'
    os.makedirs(BAK, exist_ok=True)
    if revert:
        n = 0
        for f in os.listdir(BAK):
            shutil.copy2(BAK + '/' + f, FAC + '/' + f)
            n += 1
        for f in os.listdir(FAC):
            if f.startswith('atlas_') and not os.path.exists(BAK + '/' + f):
                os.unlink(FAC + '/' + f)
        return dict(reverted=n)
    # This swap deletes facades/atlas_* and copies the set's own pages
    # over them, so a stage 35 run that happened AFTER the set was packed would be
    # thrown away silently (the composite would never see a tone-matched texel).
    # Fail loudly instead: the set has to be re-packed
    # (s60_wall_bestview.py atlas --set <name>) after s35.
    fac_j, out_j = FAC + '/facadetex.json', OUT + '/facadetex.json'
    if (os.path.exists(fac_j) and os.path.exists(out_j)
            and os.path.getmtime(fac_j) > os.path.getmtime(out_j) + 1
            and not os.environ.get('ALLOW_STALE_SET')):
        raise SystemExit(
            "STALE SET: stage 35 wrote %s at %s but set '%s' was packed at %s.\n"
            "Swapping now would discard the newer stage 35 atlas.  Re-pack with\n"
            "  %s s60_wall_bestview.py atlas --set %s\n"
            "or set ALLOW_STALE_SET=1 if the old pages really are wanted."
            % (fac_j, time.ctime(os.path.getmtime(fac_j)), setname,
               time.ctime(os.path.getmtime(out_j)), PY, setname))
    if not os.listdir(BAK):
        for f in os.listdir(FAC):
            if f.startswith('atlas_') or f in FILES:
                shutil.copy2(FAC + '/' + f, BAK + '/' + f)
    for f in os.listdir(FAC):
        if f.startswith('atlas_'):
            os.unlink(FAC + '/' + f)
    n = 0
    for f in sorted(os.listdir(OUT)):
        if f.startswith('atlas_') or f in FILES:
            shutil.copy2(OUT + '/' + f, FAC + '/' + f)
            n += 1
    return dict(files=n, backup=BAK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--set', dest='setname', default='walls',
                    help="which stage 60 rect set's atlases to swap in")
    a = ap.parse_args()
    print(json.dumps(swap(setname=a.setname, revert=a.revert)))


if __name__ == '__main__':
    main()
