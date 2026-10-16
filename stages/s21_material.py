"""Stage 21 -- a material class for every face, from OSM plus photo evidence.

Two independent sources decide each face:

  * the OSM planar partition the geometry was cut from (osm_rules.classify ->
    surface class), rasterised at 0.25 m, which is authoritative where it says
    something specific (lawn, wood, water, road) and useless over the large share
    of an area it calls "bare" -- the catch-all remainder polygon; and
  * the colour stage 20 projected from the posed panoramas: greenness,
    saturation, warmth and luminance, which is what actually separates lawn
    from concrete from asphalt inside that remainder.

The stage reports the class area table and the share of area each source
decided, and flags the photogrammetric debris to delete: faces with neither
lidar support nor corroboration from a second panorama.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
import common as C
import texlib as T

PJ = T.SCR + '/project'
OUT = T.SCR + '/material'
os.makedirs(OUT, exist_ok=True)

MATERIALS = ['lawn', 'paving', 'concrete', 'asphalt', 'gravel', 'sand',
             'wood_deck', 'metal', 'brick', 'water']
MI = {m: i for i, m in enumerate(MATERIALS)}

# OSM surface class -> the material it implies, and how strongly.  'bare' and
# 'none' are the catch-all remainder: they carry no information at all.
OSM_TO_MAT = {
    'lawn': ('lawn', 2.5), 'scrub': ('lawn', 2.0),
    'wood': ('wood_deck', 3.0), 'water': ('water', 4.0),
    'road': ('asphalt', 2.5), 'gravel': ('gravel', 2.0),
    'sand': ('sand', 2.5), 'building': ('concrete', 0.5),
    'paving': ('paving', 1.2), 'bare': (None, 0.0), 'none': (None, 0.0),
}
# kinds that mean a pedestrian hard surface: poured concrete, not stone pavers
CONCRETE_KINDS = {'highway_footway', 'highway_steps', 'highway_footway_area',
                  'highway_service', 'amenity_parking'}
ASPHALT_KINDS = {'highway_secondary', 'highway_residential', 'highway_unclassified',
                 'highway_service', 'amenity_parking', 'highway_cycleway'}


def srgb(lin):
    return np.clip(np.power(np.clip(lin, 0, 1), 1 / 2.2) * 255.0, 0, 255)


def photo_scores(col_lin, nviews):
    """-> (scores[n, nmat], have_evidence[n]).

    Colour features on the delighted, multi-view colour.  Weak evidence scores
    below 1.0 so a specific OSM class still wins; strong evidence (a clearly
    green surface, a clearly black one) scores above it and overrides."""
    c = srgb(col_lin)
    s = c.sum(1) + 1e-3
    L = c @ np.array([0.2126, 0.7152, 0.0722], 'f4')
    exg = (2 * c[:, 1] - c[:, 0] - c[:, 2]) / s
    warm = (c[:, 0] - c[:, 2]) / (c[:, 0] + c[:, 2] + 1e-3)
    mx = c.max(1); mn = c.min(1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    n = len(c)
    sc = np.zeros((n, len(MATERIALS)), 'f4')
    have = nviews >= 2
    # lawn: green.  0.045 is weak, 0.09 overrides any OSM class but water.
    sc[:, MI['lawn']] = np.clip((exg - 0.035) / 0.055, 0, 1) * 3.2
    # asphalt: dark and neutral
    sc[:, MI['asphalt']] = (np.clip((62 - L) / 30.0, 0, 1) *
                            np.clip((0.22 - sat) / 0.14, 0, 1) * 2.2)
    # concrete: mid-to-light and neutral
    sc[:, MI['concrete']] = (np.clip((L - 55) / 45.0, 0, 1) *
                             np.clip((0.20 - sat) / 0.12, 0, 1) * 1.5)
    # gravel / bare dirt: mid luminance, slightly warm, moderate saturation
    sc[:, MI['gravel']] = (np.clip((warm - 0.045) / 0.06, 0, 1) *
                           np.clip((135 - L) / 55.0, 0, 1) *
                           np.clip((L - 45) / 25.0, 0, 1) * 1.4)
    # sand: bright and warm
    sc[:, MI['sand']] = (np.clip((L - 128) / 40.0, 0, 1) *
                         np.clip((warm - 0.07) / 0.06, 0, 1) * 1.6)
    # wood: warm, mid, more saturated than concrete
    sc[:, MI['wood_deck']] = (np.clip((warm - 0.10) / 0.08, 0, 1) *
                              np.clip((sat - 0.16) / 0.12, 0, 1) *
                              np.clip((150 - L) / 60.0, 0, 1) * 1.5)
    # brick: warm and saturated, darker than sand
    sc[:, MI['brick']] = (np.clip((warm - 0.16) / 0.09, 0, 1) *
                          np.clip((sat - 0.24) / 0.12, 0, 1) * 1.6)
    # metal / glass facade: neutral, and either bright or bluish
    sc[:, MI['metal']] = (np.clip((0.14 - sat) / 0.10, 0, 1) *
                          np.clip((-warm - 0.01) / 0.06, 0, 1) * 1.3)
    sc[~have] = 0.0
    return sc, have



# ------------------------------------------------------------ ortho vote --- #
class OrthoRaster:
    """The 0.25 m ortho rasters (texture/ortho/rasters.npz), sampled at a face centroid."""

    def __init__(self, path):
        d = np.load(path)
        self.M = d['Mo'].astype('f4')
        self.okU = d['okU']; self.covered = d['covered']
        self.conf = d['conf']; self.wmask = d['wmask']
        x0, y0, c, nx, ny = [float(v) for v in d['grid']]
        self.x0, self.y0, self.c = x0, y0, c
        self.nx, self.ny = int(nx), int(ny)

    def at(self, X, Y):
        j = np.clip(((X - self.x0) / self.c).astype(np.int32), 0, self.nx - 1)
        i = np.clip(((Y - self.y0) / self.c).astype(np.int32), 0, self.ny - 1)
        return (self.M[i, j], self.okU[i, j] & ~self.covered[i, j],
                self.conf[i, j], self.wmask[i, j])


def ortho_scores(ocol, water_cell, allow_hard):
    """Votes the ortho is entitled to cast.

    Looking straight down it can separate grass from a hard surface from bare
    gravel, it can see open water, and luminance separates asphalt from the
    pale hard surfaces.  What it cannot do is tell poured concrete from stone
    pavers -- both are flat, neutral and mid-bright from 700 m up.  So the
    hard-surface vote lands on concrete, which is what an unclassified flat
    face already defaults to: the ortho confirms "this is a hard surface, not
    grass", it does not invent a paver pattern.  Voting that same score onto
    'paving' instead would move unclassified ground from a concrete texture
    to a paver texture on no evidence at all."""
    c = srgb(ocol)
    s = c.sum(1) + 1e-3
    L = c @ np.array([0.2126, 0.7152, 0.0722], 'f4')
    exg = (2 * c[:, 1] - c[:, 0] - c[:, 2]) / s
    warm = (c[:, 0] - c[:, 2]) / (c[:, 0] + c[:, 2] + 1e-3)
    mx = c.max(1); mn = c.min(1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    sc = np.zeros((len(c), len(MATERIALS)), 'f4')
    sc[:, MI['lawn']] = np.clip((exg - 0.035) / 0.055, 0, 1) * 3.0
    sc[:, MI['gravel']] = (np.clip((warm - 0.045) / 0.06, 0, 1) *
                           np.clip((135 - L) / 55.0, 0, 1) *
                           np.clip((L - 45) / 25.0, 0, 1) * 1.3)
    hard = (np.clip((0.18 - sat) / 0.12, 0, 1) *
            np.clip((L - 60) / 45.0, 0, 1) * 1.1)
    sc[:, MI['concrete']] = np.where(allow_hard, hard, 0.0)
    # asphalt is measurably dark and neutral; it is the one class beyond the
    # four the ortho vote covers, and it is the same test photo_scores()
    # applies to panorama colour
    sc[:, MI['asphalt']] = (np.clip((62 - L) / 30.0, 0, 1) *
                            np.clip((0.22 - sat) / 0.14, 0, 1) * 1.8)
    sc[:, MI['water']] = np.where(water_cell, 3.0, 0.0)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ortho', action='store_true',
                    help="let the ortho rasters vote where no "
                         'panorama saw the face; writes to material_ortho/')
    a = ap.parse_args()
    out = OUT + ('_ortho' if a.ortho else '')
    os.makedirs(out, exist_ok=True)
    ORAS = OrthoRaster(T.SCR + '/ortho/rasters.npz') if a.ortho else None
    t0 = time.time()
    surf, kind, kinds, grid = T.build_surface_raster()
    ki = {k: i for i, k in enumerate(kinds)}
    lay_photo = np.load(C.W + '/geom/face_layer.npy')
    dsm = T.DSM()
    water_z = json.load(open(C.W + '/layers/water.json'))['level_z_enu_m']

    report = {}
    table = {m: 0.0 for m in MATERIALS}
    decided = dict(osm_only=0.0, photo_only=0.0, photo_override=0.0,
                   agreed=0.0, fallback=0.0, ortho_vote=0.0)
    ovote_rep = dict(area_eligible=0.0, area_voted=0.0, area_changed=0.0,
                     changed_from={}, changed_to={}, by_layer={})

    for name in ['photo', 'ground_terrain', 'paths', 'structures', 'buildings', 'water']:
        fp = PJ + f'/{name}_face.npz'
        if not os.path.exists(fp):
            print("missing", fp); continue
        d = np.load(fp)
        col = d['col']; nviews = d['nviews']; area = d['area']
        nf = len(col)
        # ---- geometry per face -------------------------------------------- #
        if name == 'photo':
            CEN = np.load(C.G + '/centroid.npy', mmap_mode='r')
            cen = np.asarray(CEN[:nf]).astype('f4')
            hag = np.load(C.G + '/face_hag.npy', mmap_mode='r')
            hag = np.asarray(hag[:nf]).astype('f4')
            dlid = np.asarray(np.load(C.G + '/face_dlidar.npy', mmap_mode='r')[:nf]).astype('f4')
            # face normal upness
            P = np.load(C.G + '/pos.npy', mmap_mode='r')
            F = np.load(C.G + '/faces.npy', mmap_mode='r')
            up = np.zeros(nf, 'f4')
            step = 1 << 21
            for s0 in range(0, nf, step):
                e0 = min(s0 + step, nf)
                f = np.asarray(F[s0:e0])
                p0 = np.asarray(P[f[:, 0]]); p1 = np.asarray(P[f[:, 1]]); p2 = np.asarray(P[f[:, 2]])
                nn = np.cross(p1 - p0, p2 - p0)
                up[s0:e0] = np.abs(nn[:, 2]) / np.maximum(np.linalg.norm(nn, axis=1), 1e-12)
            sublayer = lay_photo[:nf]
        else:
            z = np.load(C.W + f'/layers/{name}.npz')
            pos = z['pos'].astype('f4'); idx = z['idx']
            t = pos[idx]
            cen = t.mean(1)
            nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            up = np.abs(nn[:, 2]) / np.maximum(np.linalg.norm(nn, axis=1), 1e-12)
            gz = dsm.sample(cen[:, 0], cen[:, 1])
            hag = cen[:, 2] - np.where(gz > -1e3, gz, cen[:, 2])
            dlid = np.zeros(nf, 'f4')
            sublayer = np.zeros(nf, 'i1')

        osurf = T.surface_at(surf, grid, cen[:, 0], cen[:, 1])
        okind = T.surface_at(kind, grid, cen[:, 0], cen[:, 1])

        # ---- OSM prior ---------------------------------------------------- #
        sc = np.zeros((nf, len(MATERIALS)), 'f4')
        osm_informative = np.zeros(nf, bool)
        for si, sname in enumerate(T.SURF_CLASSES):
            m = osurf == si
            if not m.any():
                continue
            mat, wgt = OSM_TO_MAT.get(sname, (None, 0.0))
            if mat is None:
                continue
            if sname == 'paving':
                kk = okind[m]
                is_conc = np.isin(kk, [ki[k] for k in CONCRETE_KINDS if k in ki])
                is_asph = np.isin(kk, [ki[k] for k in ASPHALT_KINDS if k in ki])
                idxs = np.nonzero(m)[0]
                sc[idxs[is_conc], MI['concrete']] += 2.0
                sc[idxs[is_asph], MI['asphalt']] += 1.6
                sc[idxs[~(is_conc | is_asph)], MI['paving']] += 1.6
            else:
                sc[m, MI[mat]] += wgt
            osm_informative |= m
        # 'bare'/'none' carry nothing
        for sname in ('bare', 'none'):
            osm_informative &= ~(osurf == T.SURF_CLASSES.index(sname))

        # ---- photo evidence ------------------------------------------------ #
        psc, have = photo_scores(col, nviews)
        sc = sc + psc

        # ---- the ortho's vote, only where no panorama saw the face --------- #
        ovoted = np.zeros(nf, bool)
        if ORAS is not None:
            ocol, ook, oconf, owater = ORAS.at(cen[:, 0], cen[:, 1])
            gz2 = dsm.sample(cen[:, 0], cen[:, 1])
            topmost = np.abs(cen[:, 2] - gz2) <= 1.5
            # a wall is invisible from above; so is anything under a lid
            eligible = (~have) & ook & (oconf >= 0.5) & (up > 0.5) & topmost
            hard_ok = ~osm_informative
            water_cell = (owater & (up > 0.9) &
                          (np.abs(cen[:, 2] - water_z) < 0.5))
            osc = ortho_scores(ocol, water_cell, hard_ok)
            osc[~eligible] = 0.0
            sc = sc + osc
            ovoted = eligible & (osc.max(1) > 0)
            ovote_rep['area_eligible'] += float(area[eligible].sum())
            ovote_rep['area_voted'] += float(area[ovoted].sum())

        # ---- structural rules that beat both ------------------------------- #
        vertical = up < 0.35
        if name == 'water':
            sc[:] = 0; sc[:, MI['water']] = 10.0
        if name == 'buildings' or (name == 'photo' and (sublayer == 3).any()):
            isfac = (sublayer == 3) if name == 'photo' else np.ones(nf, bool)
            wall = isfac & vertical
            roof = isfac & ~vertical
            # a wall is brick or metal/glass or concrete, never lawn or paving
            for bad in (MI['lawn'], MI['paving'], MI['asphalt'], MI['sand'],
                        MI['gravel'], MI['water']):
                sc[wall, bad] = 0.0
            sc[wall, MI['concrete']] += 0.8
            sc[wall, MI['metal']] += 0.6
            for bad in (MI['lawn'], MI['water'], MI['sand']):
                sc[roof, bad] = 0.0
            sc[roof, MI['concrete']] += 0.7
            sc[roof & (srgb(col) @ np.array([0.2126, 0.7152, 0.0722], 'f4') < 70) & have,
               MI['asphalt']] += 1.2
        if name == 'structures' or (name == 'photo' and (sublayer == 2).any()):
            isst = (sublayer == 2) if name == 'photo' else np.ones(nf, bool)
            deck = isst & ~vertical
            rail = isst & vertical
            over_water = cen[:, 2] < water_z + 4.0
            sc[deck & over_water, MI['wood_deck']] += 1.6
            for bad in (MI['lawn'], MI['water']):
                sc[rail, bad] = 0.0
            sc[rail, MI['metal']] += 1.0
        # nothing that is not water is allowed to be water
        if name != 'water':
            nw = ~((osurf == T.SURF_CLASSES.index('water')) & (up > 0.8))
            sc[nw, MI['water']] = 0.0

        # ---- decide -------------------------------------------------------- #
        # a path ribbon is never lawn; a terrain wall is never metal
        if name == 'paths':
            sc[:, MI['lawn']] = 0.0
        best = sc.argmax(1)
        nothing = sc.max(1) <= 1e-6
        # fallback for a face with no OSM class and no photo evidence: concrete
        # on the flat; on the vertical, metal only where it is a building
        best[nothing] = MI['concrete']
        facade_like = ((name == 'buildings') |
                       ((name == 'photo') & (sublayer == 3)))
        best[nothing & vertical & facade_like] = MI['metal']

        osm_best = np.full(nf, -1, 'i4')
        # recompute the OSM-only argmax for the attribution
        for si, sname in enumerate(T.SURF_CLASSES):
            m = osurf == si
            mat, wgt = OSM_TO_MAT.get(sname, (None, 0.0))
            if mat is None or not m.any():
                continue
            if sname == 'paving':
                kk = okind[m]; idxs = np.nonzero(m)[0]
                is_conc = np.isin(kk, [ki[k] for k in CONCRETE_KINDS if k in ki])
                is_asph = np.isin(kk, [ki[k] for k in ASPHALT_KINDS if k in ki])
                osm_best[idxs[is_conc]] = MI['concrete']
                osm_best[idxs[is_asph]] = MI['asphalt']
                osm_best[idxs[~(is_conc | is_asph)]] = MI['paving']
            else:
                osm_best[m] = MI[mat]
        src = np.full(nf, 4, 'i1')          # 4 = fallback
        src[osm_informative & ~have] = 0    # osm only
        src[~osm_informative & have] = 1    # photo only
        src[osm_informative & have & (best == osm_best)] = 3   # agreed
        src[osm_informative & have & (best != osm_best)] = 2   # photo override
        if ORAS is not None:
            prev = np.load(OUT + f'/{name}_mat.npz')['mat'].astype('i1')
            if len(prev) != nf:     # size check: stale from another extent
                raise SystemExit(f'{OUT}/{name}_mat.npz has {len(prev)} faces, the layer has {nf}: '
                                 'rerun s21_material.py without --ortho first')
            chg = ovoted & (best.astype('i1') != prev)
            src[ovoted] = 5                                    # ortho vote
            ovote_rep['area_changed'] += float(area[chg].sum())
            for mi2, mname in enumerate(MATERIALS):
                f_ = float(area[chg & (prev == mi2)].sum())
                t_ = float(area[chg & (best == mi2)].sum())
                if f_ > 0:
                    ovote_rep['changed_from'][mname] = round(
                        ovote_rep['changed_from'].get(mname, 0.0) + f_, 1)
                if t_ > 0:
                    ovote_rep['changed_to'][mname] = round(
                        ovote_rep['changed_to'].get(mname, 0.0) + t_, 1)
            ovote_rep['by_layer'][name] = dict(
                area_m2=round(float(area.sum()), 1),
                area_eligible_m2=round(float(area[eligible].sum()), 1),
                area_voted_m2=round(float(area[ovoted].sum()), 1),
                area_changed_m2=round(float(area[chg].sum()), 1),
                faces_changed=int(chg.sum()))

        # ---- debris ---------------------------------------------------------#
        kill = np.zeros(nf, bool)
        if name == 'photo':
            in_model = (sublayer >= 1) & (sublayer <= 3)
            unseen = nviews < 1
            thin = nviews < 2
            kill = ((unseen & (dlid > 1.5)) | (thin & (dlid > 3.0)))
            # airborne slabs: photogrammetry leaves large planes hanging above
            # the scene where the depth fusion bridged two surfaces.  Anything
            # more than 5 m up that the lidar does not back and fewer than four
            # panoramas corroborate is one of those, not architecture.
            airborne = (hag > 5.0) & (dlid > 0.8) & (nviews < 4)
            kill |= airborne
            # water-edge shards: photo geometry standing over open water that is
            # neither a pier deck nor supported by lidar
            over_w = (T.surface_at(surf, grid, cen[:, 0], cen[:, 1]) ==
                      T.SURF_CLASSES.index('water'))
            shard = over_w & (dlid > 1.0) & (nviews < 3) & (cen[:, 2] > water_z - 1.0)
            kill |= shard
            kill &= in_model
            per_sub = {}
            for cl, cn in ((1, 'ground'), (2, 'structures'), (3, 'facades')):
                sm = sublayer == cl
                per_sub[cn] = dict(
                    faces=int(sm.sum()), area_m2=round(float(area[sm].sum()), 1),
                    killed_faces=int((sm & kill).sum()),
                    killed_area_pct=round(float(100 * area[sm & kill].sum() /
                                                max(area[sm].sum(), 1e-9)), 2))
            report['debris'] = dict(
                faces_in_model=int(in_model.sum()),
                area_in_model_m2=round(float(area[in_model].sum()), 1),
                killed=int(kill.sum()),
                killed_pct_of_model_faces=round(float(100 * kill.sum() /
                                                      max(in_model.sum(), 1)), 2),
                killed_area_m2=round(float(area[kill].sum()), 1),
                killed_area_pct=round(float(100 * area[kill].sum() /
                                            max(area[in_model].sum(), 1e-9)), 2),
                by_sublayer=per_sub,
                of_which_water_edge_shards=int((shard & in_model).sum()),
                of_which_airborne_slabs=int((airborne & in_model).sum()),
                rule='no panorama saw it and it is >1.5 m from any lidar point, '
                     'or fewer than 2 panoramas and >3.0 m from lidar, or it '
                     'stands over open water unsupported, or it hangs more '
                     'than 5 m up with no lidar backing and fewer than four '
                     'panoramas')

        if name == 'buildings':
            # an open shelter's posts and slab underside are one
            # material (s10a_layers.SHELTER_MAT), not the concrete / wood_deck
            # / brick patchwork that the shadowed panorama colour would vote
            import s10a_layers as SL
            bz = np.load(C.W + '/layers/buildings.npz')
            bm = json.load(open(C.W + '/layers/buildings.json'))
            sf = np.concatenate(SL.shelter_faces(bm, bz['pos'], bz['idx']))
            sf = sf[sf < nf]
            best[sf] = MI[SL.SHELTER_MAT]; kill[sf] = False
            print(f"  open shelter: {len(sf)} post + underside faces set to {SL.SHELTER_MAT}", flush=True)
        np.savez_compressed(out + f'/{name}_mat.npz', mat=best.astype('i1'),
                            ovoted=ovoted,
                            src=src, kill=kill, up=up.astype('f4'),
                            hag=hag.astype('f4'), osurf=osurf.astype('i1'),
                            okind=okind.astype('i2'), sublayer=sublayer)
        live = ~kill
        per = {}
        for mi2, mname in enumerate(MATERIALS):
            m = (best == mi2) & live
            if m.any():
                per[mname] = round(float(area[m].sum()), 1)
                table[mname] += float(area[m].sum())
        for si2, sname in enumerate(['osm_only', 'photo_only', 'photo_override',
                                     'agreed', 'fallback', 'ortho_vote']):
            m = (src == si2) & live
            decided[sname] += float(area[m].sum())
        report[name] = dict(faces=int(nf), area_m2=round(float(area[live].sum()), 1),
                            by_material=per,
                            photo_evidence_frac_area=round(
                                float(area[have & live].sum() / max(area[live].sum(), 1e-9)), 4))
        print(f"{name:<16} {json.dumps(report[name])}", flush=True)

    tot = sum(table.values())
    report['class_area_table_m2'] = {k: round(v, 1) for k, v in
                                     sorted(table.items(), key=lambda x: -x[1])}
    report['class_area_pct'] = {k: round(100 * v / max(tot, 1e-9), 2) for k, v in
                                sorted(table.items(), key=lambda x: -x[1])}
    dtot = sum(decided.values())
    report['decided_by_area_pct'] = {k: round(100 * v / max(dtot, 1e-9), 2)
                                     for k, v in decided.items()}
    report['decided_by_area_pct']['photo_any'] = round(
        100 * (decided['photo_only'] + decided['photo_override'] + decided['agreed']) /
        max(dtot, 1e-9), 2)
    report['decided_by_area_pct']['osm_alone'] = round(
        100 * decided['osm_only'] / max(dtot, 1e-9), 2)
    if ORAS is not None:
        ovote_rep['area_eligible'] = round(ovote_rep['area_eligible'], 1)
        ovote_rep['area_voted'] = round(ovote_rep['area_voted'], 1)
        ovote_rep['area_changed'] = round(ovote_rep['area_changed'], 1)
        prev_rep = json.load(open(OUT + '/material.json'))
        ovote_rep['class_area_pct_before'] = prev_rep['class_area_pct']
        ovote_rep['class_area_m2_before'] = prev_rep['class_area_table_m2']
        ovote_rep['rule'] = (
            'the ortho votes only on a face no panorama saw (fewer than two '
            'views), that faces up (|n.z| > 0.5), that is the topmost surface '
            'at its x,y (within 1.5 m of the lidar DSM) and whose 0.25 m ortho '
            'cell is at least half usable pixels.  It can vote lawn, gravel or '
            'water anywhere it is eligible; the paving vote is offered only '
            'where the face had no class at all, and it lands on concrete, the '
            'class such a face already defaults to, because a nadir photo cannot '
            'separate poured concrete from stone pavers; scoring it onto paving '
            'would move unclassified ground onto a paver texture with nothing to '
            'back it')
        report['ortho_vote'] = ovote_rep
    report['seconds'] = round(time.time() - t0)
    json.dump(report, open(out + '/material.json', 'w'), indent=1)
    print("\nclass area table:", json.dumps(report['class_area_pct'], indent=1))
    print("decided by:", json.dumps(report['decided_by_area_pct'], indent=1))
    print("debris:", json.dumps(report.get('debris', {}), indent=1))
    C.note('material' + ('_ortho' if a.ortho else ''),
           {k: v for k, v in report.items()
            if k in ('class_area_table_m2', 'class_area_pct',
                     'decided_by_area_pct', 'debris', 'ortho_vote')})
    C.mark('material' + ('_ortho' if a.ortho else ''))


if __name__ == '__main__':
    main()
