#!/usr/bin/env python3
"""Stage 60 -- photographic building walls: one panorama per 1-2 m patch.

A top-2 weighted average of up to six panoramas per texel at 5 cm is a blur:
the views of a wall typically disagree by 10 to 25 cm even after the
image-space alignment below, so that average is a 20 cm blur over a 1 cm photograph, and no
fusion statistic can undo it.

This stage takes each patch of each wall from ONE panorama -- the nearest
unoccluded front-facing view inside 25 m, sampled at the panorama's native
equirect width (5760 or 11000 px) -- and fixes the ~1 m pose error in image
space instead of in pose space:

  * every view of a wall is correlated against that wall's reference view on
    the wall plane (masked NCC), so patches from different panoramas line up;
  * the whole set is pinned to the wall's own outline by the silhouette: the
    roofline against the measured sky boundary, and the corner by a 1D match
    of the predicted silhouette profile against the measured one.  Where the
    gap is a wrong OSM storey count rather than a pose error the texture is
    stretched from the base, so the ground line and the roofline both land.

Subcommands: rects, cover, chips, align, refine, own, bake, relevel, atlas, all.
Everything is resumable: each step writes its own marker and the chips and
tiles are skipped when they already exist.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
from wallphoto import wpcommon as H

SET = 'walls'


def main():
    global SET
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('rects', 'cover', 'chips', 'align', 'refine',
                                    'own', 'bake', 'relevel', 'atlas',
                                    'all'))
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--only', default=None, help='comma-separated wall ids')
    ap.add_argument('--spacing', type=float, default=1.0)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--set', dest='setname', default=SET,
                    help="the rect set to work on: a directory name under the "
                         "wallphoto scratch dir (each rect carries its own range "
                         "gate, see wpcommon.R_MAX_FAR)")
    a = ap.parse_args()
    SET = a.setname
    only = set(int(x) for x in a.only.split(',')) if a.only else None
    steps = ([a.cmd] if a.cmd != 'all' else
             ['rects', 'cover', 'chips', 'align', 'refine', 'own', 'bake',
              'relevel', 'atlas'])
    for s in steps:
        if a.cmd == 'all' and H.have(SET, s) and not a.force:
            H.log(f"s60: {s} already done")
            continue
        t0 = time.time()
        if s == 'rects':
            rects = H.walls_as_rects()
            H.save_rects(SET, rects, extra=dict(
                source="s33_walls.py wall inventory (facades/walls.json)"))
            H.mark(SET, 'rects', dict(rects=len(rects)))
            H.log(f"s60 rects: {len(rects)} wall planes")
        elif s == 'cover':
            from wallphoto import wpcover
            wpcover.run(SET, workers=a.workers, spacing=a.spacing, only=only)
        elif s == 'chips':
            from wallphoto import wpalign
            wpalign.run_chips(SET, workers=min(a.workers, 16), only=only)
        elif s == 'align':
            from wallphoto import wpalign
            wpalign.run_align(SET, workers=a.workers, only=only)
        elif s == 'refine':
            from wallphoto import wpalign
            wpalign.run_refine(SET, workers=a.workers, only=only)
        elif s == 'own':
            from wallphoto import wpbake
            wpbake.run_own(SET)
        elif s == 'bake':
            from wallphoto import wpbake
            wpbake.run_bake(SET, workers=min(a.workers, 10), only=only)
        elif s == 'relevel':
            # the walls the level gate and the far gate turned down get their
            # next panoramas; anything that still fails is put back untouched
            from wallphoto import wpatlas
            wpatlas.run_relevel(SET)
        elif s == 'atlas':
            from wallphoto import wpatlas
            r = wpatlas.run_atlas(SET)
            print(json.dumps(r, indent=1)[:1500])
        H.log(f"s60 {s}: {round(time.time()-t0)}s")


if __name__ == '__main__':
    main()
