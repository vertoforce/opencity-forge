"""Rebuild the tree layer by joining a city's point tree inventory to the lidar.

The tree layer is built inventory-first rather than by segmenting the canopy
into blobs: a published point inventory is the anchor for where a tree is, what species it is and how thick its trunk is; the lidar is asked only
for height and crown spread, and only where it actually has a crown there.

Evidence order for one placement:

  position    inventory point, else OSM node, else lidar crown apex
  species     inventory scientific name, else OSM species/genus tag
  trunk       inventory DBH (a measured diameter at breast height), else a
              DBH-from-height allometry fitted on the area's own matched pairs
  height      1. a lidar crown matched to the point            (best)
              2. the max of the canopy height model in an 11 x 11 cell window
              3. the field-recorded height in the inventory, feet to metres
              4. an allometry fitted on the area's own matched pairs
  crown       the matched lidar crown, else the same fitted allometry

Every record says which tier it used, so a bad-looking tree can be traced to the
evidence that placed it.
"""
import sys, os, json, math, csv, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
from scipy.spatial import cKDTree
import common as C
from s05_inventory import classify          # the s05 crown classifier, reused as is

INV = C.W + '/inventory/city_trees.json'
MIN_FIT_PAIRS = 8         # below this the allometry is a prior, not a fit
CROWN_MATCH_MIN, CROWN_MATCH_MAX = 3.0, 6.0   # metres, grows with crown radius
OSM_MATCH = 4.0                               # metres
CHM_WIN = 2.5                                 # metres, ~11 x 11 cells at 0.5 m
MIN_CANOPY = 2.5                              # metres, what s03 calls canopy

# species -> genus -> leaf type, the arnis fallback chain.  The common street and
# park conifer genera (and Ginkgo) are listed; an unlisted genus falls back to
# broadleaved deciduous.
NEEDLE = {'Pinus': ('needleleaved', 'evergreen'),
          'Pseudotsuga': ('needleleaved', 'evergreen'),
          'Abies': ('needleleaved', 'evergreen'),
          'Picea': ('needleleaved', 'evergreen'),
          'Thuja': ('needleleaved', 'evergreen'),
          'Tsuga': ('needleleaved', 'evergreen'),
          'Sequoia': ('needleleaved', 'evergreen'),
          'Chamaecyparis': ('needleleaved', 'evergreen'),
          'Cedrus': ('needleleaved', 'evergreen'),
          'Taxodium': ('needleleaved', 'deciduous'),
          'Metasequoia': ('needleleaved', 'deciduous'),
          'Larix': ('needleleaved', 'deciduous'),
          'Ginkgo': ('broadleaved', 'deciduous')}


def leaf_of(species, leaf_type):
    g = (species or '').split()[0] if species else None
    if g in NEEDLE: return NEEDLE[g]
    if leaf_type:
        t = leaf_type.lower()
        if t.startswith('deciduous conifer'): return ('needleleaved', 'deciduous')
        if t.startswith('conifer'): return ('needleleaved', 'evergreen')
        if t.startswith('deciduous'): return ('broadleaved', 'deciduous')
        if t.startswith('evergreen'): return ('broadleaved', 'evergreen')
    if species: return ('broadleaved', 'deciduous')
    return (None, None)


def greedy_pairs(A, B, radius_of):
    """One-to-one nearest assignment, shortest distance first."""
    if not len(A) or not len(B): return {}, {}
    kb = cKDTree(B)
    pairs = []
    for i, a in enumerate(A):
        for j in kb.query_ball_point(a, radius_of(i)):
            pairs.append((float(np.linalg.norm(a - B[j])), i, j))
    pairs.sort()
    ab, ba = {}, {}
    for d, i, j in pairs:
        if i in ab or j in ba: continue
        ab[i] = (j, d); ba[j] = (i, d)
    return ab, ba


def main():
    t0 = time.time()
    src = json.load(open(INV))
    inv = src['trees']
    crowns = json.load(open(C.W + '/crowns_scored.json'))
    O = json.load(open(C.W + '/trees_raw.json'))['osm_trees']
    prev = json.load(open(C.W + '/prev/tree_inventory_before_join.json'))['trees'] \
        if os.path.exists(C.W + '/prev/tree_inventory_before_join.json') else []

    Z = np.load(C.W + '/geom/chm.npz')
    chm = Z['chm']; gx0, gy0, cell, nx, ny = Z['grid']; nx = int(nx); ny = int(ny)
    # The raw CHM is just "height above ground": inside a building footprint it is
    # the roof, so a window maximum next to a building reads the roof height.  Blank the same cells s03 blanks before it
    # looks for crowns, so the window can only ever see canopy.
    bmask = Z['bmask']; wmask = Z['wmask']; pmask = Z['pmask']
    chm_open = chm.copy()
    chm_open[bmask] = 0.0
    chm_open[wmask & ~pmask] = 0.0
    # and the undilated footprints, only to flag a placement that lands in one
    braw_p = C.W + '/geom/bmask_raw.npz'
    if os.path.exists(braw_p):
        braw = np.load(braw_p)['bmask_raw']
    else:
        from s03_trees import rasterise_polys, osm_layers
        _, _builds, _, _, _ = osm_layers()
        braw = rasterise_polys([b[0] for b in _builds], gx0, gy0, cell, nx, ny)
        np.savez_compressed(braw_p, bmask_raw=braw,
                            grid=np.array([gx0, gy0, cell, nx, ny], 'f8'))
    S = np.load(C.FW + '/surfaces.npz')
    dx0, dy0, dc, dnx, dny = [float(v) for v in S['grid']]; dnx = int(dnx); dny = int(dny)
    dtm = S['dtm']

    def dtm_at(x, y):
        return float(dtm[int(np.clip((y - dy0) / dc, 0, dny - 1)),
                         int(np.clip((x - dx0) / dc, 0, dnx - 1))])

    def chm_max(x, y, rad=CHM_WIN):
        i0 = int(np.clip((x - gx0 - rad) / cell, 0, nx - 1))
        i1 = int(np.clip((x - gx0 + rad) / cell, 0, nx - 1))
        j0 = int(np.clip((y - gy0 - rad) / cell, 0, ny - 1))
        j1 = int(np.clip((y - gy0 + rad) / cell, 0, ny - 1))
        s = chm_open[j0:j1 + 1, i0:i1 + 1]
        return float(s.max()) if s.size else 0.0

    def in_building(x, y):
        return bool(braw[int(np.clip((y - gy0) / cell, 0, ny - 1)),
                         int(np.clip((x - gx0) / cell, 0, nx - 1))])

    IX = np.array([[t['x'], t['y']] for t in inv], 'f8')
    CX = np.array([[c['x'], c['y']] for c in crowns], 'f8')
    OX = np.array([[o['x'], o['y']] for o in O], 'f8')

    # ---- 1. inventory point -> lidar crown -------------------------------- #
    krad = np.clip([max(CROWN_MATCH_MIN, c['crown_radius_m']) for c in crowns],
                   CROWN_MATCH_MIN, CROWN_MATCH_MAX)
    inv2crown, crown2inv = greedy_pairs(IX, CX, lambda i: CROWN_MATCH_MAX)
    # re-test each pair against that crown's own radius allowance
    for i in list(inv2crown):
        j, d = inv2crown[i]
        if d > krad[j]:
            del inv2crown[i]; del crown2inv[j]

    # ---- 2. inventory point -> OSM natural=tree node ---------------------- #
    inv2osm, osm2inv = greedy_pairs(IX, OX, lambda i: OSM_MATCH)

    # ---- 3. allometry fitted on this area's own matched pairs ------------- #
    fit = [(inv[i]['dbh_in'], crowns[j]['height_m'], crowns[j]['crown_radius_m'])
           for i, (j, d) in inv2crown.items() if inv[i]['dbh_in']]
    D = np.array([r[0] for r in fit], 'f8')
    H = np.array([r[1] for r in fit], 'f8')
    Rr = np.array([r[2] for r in fit], 'f8')
    # an area with no city inventory (s03b writes an empty one) or too few matched pairs
    # cannot fit its own allometry.  It uses a fixed prior allometry (the constants
    # below, a street-tree DBH/height/crown fit used as the default) and the lidar
    # crowns' own medians for the area-median fallbacks.
    if len(fit) < MIN_FIT_PAIRS:
        h_a, h_b, r_m, r_c, d_a, d_b = 3.571, 0.303, 0.1042, 1.429, 1.149, 0.888
        ch = np.array([c['height_m'] for c in crowns] or [8.0], 'f8')
        cr = np.array([c['crown_radius_m'] for c in crowns] or [2.5], 'f8')
        allom = dict(n_pairs=len(fit), prior="fixed prior allometry (constants in s05b_inventory_join.py)",
                     height_m="%.3f * dbh_in ** %.3f" % (h_a, h_b), crown_radius_m="%.4f * dbh_in + %.3f" % (r_m, r_c),
                     dbh_in_from_height="%.3f * height_m ** %.3f" % (d_a, d_b),
                     note="fewer than %d inventory/crown pairs (no city inventory here): prior, not fitted; "
                          "area medians are the lidar crowns' own" % MIN_FIT_PAIRS)
        print("allometry: %d pairs < %d, using the prior: %s" % (len(fit), MIN_FIT_PAIRS, allom))
        med_h = float(np.median(ch)); med_r = float(np.median(cr))
    else:
        lf = np.polyfit(np.log(D), np.log(H), 1)
        h_a, h_b = float(math.exp(lf[1])), float(lf[0])          # height = a * dbh^b
        rl = np.polyfit(D, Rr, 1)
        r_m, r_c = float(rl[0]), float(rl[1])                    # radius = m*dbh + c
        # and the other direction, for the trees the inventory has no record of: a
        # trunk from the lidar height.  A fixed 6 % of height is about six times too
        # fat for a street tree.
        df = np.polyfit(np.log(H), np.log(D), 1)
        d_a, d_b = float(math.exp(df[1])), float(df[0])          # dbh = a * height^b
        allom = dict(n_pairs=len(fit),
                     height_m="%.3f * dbh_in ** %.3f" % (h_a, h_b),
                     height_r=round(float(np.corrcoef(np.log(D), np.log(H))[0, 1]), 3),
                     crown_radius_m="%.4f * dbh_in + %.3f" % (r_m, r_c),
                     radius_r=round(float(np.corrcoef(D, Rr)[0, 1]), 3),
                     dbh_in_from_height="%.3f * height_m ** %.3f" % (d_a, d_b),
                     dbh_r=round(float(np.corrcoef(np.log(H), np.log(D))[0, 1]), 3),
                     fitted_over_height_m=[round(float(H.min()), 2), round(float(H.max()), 2)],
                     note="fitted on this area's own inventory-to-crown pairs; heights above "
                          "the fitted range are extrapolated and the trunk is capped at 0.45 m")
        print("allometry fitted on %d inventory/crown pairs: %s" % (len(fit), allom))

        med_h = float(np.median(H)); med_r = float(np.median(Rr))

    def h_from_dbh(d): return float(np.clip(h_a * d ** h_b, 1.5, 40.0))
    def r_from_dbh(d): return float(np.clip(r_m * d + r_c, 0.8, 8.0))
    def dbh_from_h(h): return float(np.clip(d_a * max(h, 0.5) ** d_b, 1.0, 40.0))
    def trunk_from_h(h): return round(float(np.clip(dbh_from_h(h) * 0.0254 / 2.0,
                                                    0.04, 0.45)), 3)

    out = []

    # ---- 4. one placement per inventory point ----------------------------- #
    tiers = collections.Counter()
    for i, t in enumerate(inv):
        x, y = t['x'], t['y']
        dbh = t['dbh_in']
        cj = inv2crown.get(i)
        oj = inv2osm.get(i)
        o = O[oj[0]] if oj else None
        chmw = chm_max(x, y)
        if cj:
            c = crowns[cj[0]]
            tier = 'lidar_crown'
            h = c['height_m']; cb = c['crown_base_m']; rad = c['crown_radius_m']
            area = c['crown_area_m2']; nret = c['lidar_returns']; por = c['porosity']
            gf = c['green_frac']
        else:
            nret = 0; por = None; gf = None
            if chmw >= MIN_CANOPY:
                tier = 'chm_window'; h = round(chmw, 2)
            elif t['height_ft']:
                tier = 'field_height'; h = round(t['height_ft'] * 0.3048, 2)
            elif dbh:
                tier = 'dbh_allometry'; h = round(h_from_dbh(dbh), 2)
            else:
                tier = 'area_median'; h = round(med_h, 2)
            rad = round(r_from_dbh(dbh) if dbh else med_r, 2)
            cb = round(min(2.0, 0.35 * h), 2)
            area = round(math.pi * rad * rad, 1)
        tiers[tier] += 1
        leaf, cycle = leaf_of(t['species'], t.get('leaf_type'))
        trunk = (round(float(np.clip(dbh * 0.0254 / 2.0, 0.04, 0.9)), 3) if dbh
                 else trunk_from_h(h))
        src_name = 'inventory' + ('+lidar' if cj else '') + ('+osm' if o else '')
        conf = 'high' if cj else ('medium' if tier in ('chm_window', 'field_height') else 'low')
        why = {
          'lidar_crown': "city tree inventory point matched to a lidar crown %.1f m away" %
                         (cj[1] if cj else 0),
          'chm_window': "city tree inventory point; no separate crown, height from the "
                        "lidar canopy height model in an 11 x 11 cell window",
          'field_height': "city tree inventory point; the lidar has no canopy here, "
                          "height is the inventory's own field measurement",
          'dbh_allometry': "city tree inventory point; the lidar has no canopy here "
                           "(planted since, or too thin to return), size from trunk "
                           "diameter through an allometry fitted on this area",
          'area_median': "city tree inventory point with no diameter and no lidar canopy",
        }[tier]
        out.append(dict(
            id=None, source=src_name, confidence=conf, evidence=why, height_tier=tier,
            x_enu_m=round(x, 2), y_enu_m=round(y, 2),
            ground_z_enu_m=round(dtm_at(x, y), 2),
            height_m=round(h, 2), crown_base_m=round(cb, 2),
            crown_radius_m=round(rad, 2), crown_area_m2=area,
            trunk_radius_m=trunk,
            inv_id=t['inv_id'], inv_dept=t['dept'], dbh_in=dbh,
            dbh_source=('inventory' if dbh else 'allometry_from_lidar_height'),
            field_height_m=(round(t['height_ft'] * 0.3048, 2) if t['height_ft'] else None),
            chm_window_max_m=round(chmw, 2),
            planted_year=t['planted_year'], condition=t['condition'],
            inside_building_footprint=in_building(x, y),
            osm_id=(o['osm_id'] if o else None),
            osm_dist_m=(round(oj[1], 2) if oj else None),
            species=t['species'] or (o.get('species') if o else None),
            common_name=t['common_name'],
            leaf_type=leaf or (o.get('leaf_type') if o else None),
            leaf_cycle=cycle,
            lidar_returns=nret, porosity=por, green_frac=gf, over_water=False,
            lat=t['lat'], lon=t['lon']))

    # ---- 5. lidar crowns the inventory did not claim ---------------------- #
    free_c = [j for j in range(len(crowns)) if j not in crown2inv]
    free_o = [j for j in range(len(O)) if j not in osm2inv]
    kept_c, rejected = [], []
    for j in free_c:
        conf, why = classify(crowns[j])
        (kept_c if conf else rejected).append((j, conf, why))
    c2o, o2c = greedy_pairs(np.array([[crowns[j]['x'], crowns[j]['y']] for j, _, _ in kept_c], 'f8')
                            if kept_c else np.zeros((0, 2)),
                            OX[free_o] if free_o else np.zeros((0, 2)),
                            lambda i: max(OSM_MATCH, crowns[kept_c[i][0]]['crown_radius_m']))
    for k, (j, conf, why) in enumerate(kept_c):
        c = crowns[j]
        pj = c2o.get(k)
        o = O[free_o[pj[0]]] if pj else None
        h = c['height_m']
        lat, lon, _ = C.enu_to_lla(c['x'], c['y'], 0.0)
        leaf, cycle = leaf_of(o.get('species') if o else None, None)
        out.append(dict(
            id=None, source='lidar' + ('+osm' if o else ''), confidence=conf,
            evidence=why + "; not in the city tree inventory", height_tier='lidar_crown',
            x_enu_m=round(c['x'], 2), y_enu_m=round(c['y'], 2),
            ground_z_enu_m=round(c['z_base'], 2),
            height_m=round(h, 2), crown_base_m=round(c['crown_base_m'], 2),
            crown_radius_m=round(c['crown_radius_m'], 2), crown_area_m2=c['crown_area_m2'],
            trunk_radius_m=trunk_from_h(h),
            inv_id=None, inv_dept=None, dbh_in=round(dbh_from_h(h), 1),
            dbh_source='allometry_from_lidar_height',
            field_height_m=None, chm_window_max_m=round(chm_max(c['x'], c['y']), 2),
            planted_year=None, condition=None,
            inside_building_footprint=in_building(c['x'], c['y']),
            osm_id=(o['osm_id'] if o else None),
            osm_dist_m=(round(pj[1], 2) if pj else None),
            species=(o.get('species') if o else None), common_name=None,
            leaf_type=leaf or (o.get('leaf_type') if o else None), leaf_cycle=cycle,
            lidar_returns=c['lidar_returns'], porosity=c['porosity'],
            green_frac=c['green_frac'], over_water=c['over_water'],
            lat=round(float(lat), 8), lon=round(float(lon), 8)))

    # ---- 6. OSM nodes claimed by neither inventory nor crown -------------- #
    n_merged = n_none = 0
    for k, j in enumerate(free_o):
        if k in o2c: continue
        o = O[j]
        top = chm_max(o['x'], o['y'], 4.0)
        if top >= MIN_CANOPY:
            n_merged += 1; srcn = 'osm+lidar_canopy'; conf = 'medium'; tier = 'chm_window'
            why = ("OSM maps a tree here, the city inventory does not, and the lidar canopy "
                   "merges into a neighbouring crown")
            h = round(top, 2); rad = round(min(med_r, 3.0), 2)
        else:
            n_none += 1; srcn = 'osm_only'; conf = 'low'; tier = 'area_median'
            why = ("OSM maps a tree here; neither the city inventory nor the lidar "
                   "has anything above 2.5 m within 4 m")
            h = round(med_h, 2); rad = round(med_r, 2)
        try: h = round(float(str(o.get('height_tag')).split()[0]), 2)
        except Exception: pass
        leaf, cycle = leaf_of(o.get('species'), None)
        out.append(dict(
            id=None, source=srcn, confidence=conf, evidence=why, height_tier=tier,
            x_enu_m=round(o['x'], 2), y_enu_m=round(o['y'], 2),
            ground_z_enu_m=round(dtm_at(o['x'], o['y']), 2),
            height_m=h, crown_base_m=round(min(2.0, 0.35 * h), 2),
            crown_radius_m=rad, crown_area_m2=round(math.pi * rad * rad, 1),
            trunk_radius_m=trunk_from_h(h),
            inv_id=None, inv_dept=None, dbh_in=round(dbh_from_h(h), 1),
            dbh_source='allometry_from_lidar_height',
            field_height_m=None, chm_window_max_m=round(top, 2),
            planted_year=None, condition=None,
            inside_building_footprint=in_building(o['x'], o['y']),
            osm_id=o['osm_id'], osm_dist_m=0.0,
            species=o.get('species'), common_name=None,
            leaf_type=leaf or o.get('leaf_type'), leaf_cycle=cycle,
            lidar_returns=0, porosity=None, green_frac=None, over_water=False,
            lat=o['lat'], lon=o['lon']))

    out.sort(key=lambda t: (t['y_enu_m'], t['x_enu_m']))
    for k, t in enumerate(out, 1):
        t['id'] = f"tree_{k:04d}"

    # ---- 7. difference against the crown-only layer ----------------------- #
    NEW = np.array([[t['x_enu_m'], t['y_enu_m']] for t in out], 'f8')  # noqa: E501
    added = removed = 0
    if prev:
        PV = np.array([[t['x_enu_m'], t['y_enu_m']] for t in prev], 'f8')
        p2n, n2p = greedy_pairs(PV, NEW, lambda i: 2.5)
        added = len(out) - len(n2p)
        moved = [d for _, d in p2n.values()]
        move_med = round(float(np.median(moved)), 2) if moved else None
        # a previous placement with no new placement within 2.5 m is only really
        # gone if its OSM node is gone too; otherwise the inventory point simply
        # moved it further than the match radius
        by_osm = {}
        for t in out:
            if t['osm_id']: by_osm.setdefault(t['osm_id'], []).append(t)
        absorbed = removed = 0
        for i, t in enumerate(prev):
            if i in p2n: continue
            if t.get('osm_id') and t['osm_id'] in by_osm: absorbed += 1
            else: removed += 1
    else:
        move_med = None; added = len(out); absorbed = removed = 0
    rescued = sum(1 for t in out if t['source'].startswith('inventory')
                  and t['height_tier'] != 'lidar_crown' and t['chm_window_max_m'] < MIN_CANOPY)
    # crowns the crown-only classifier rejected but the inventory confirms
    revived = sum(1 for j in crown2inv if classify(crowns[j])[0] is None)
    # the previous layer's map-only trees (OSM node, no lidar canopy) that the
    # inventory can now name and size
    prev_map_only = [t for t in prev if t.get('source') == 'osm_only']
    named = 0
    if prev_map_only:
        PM = np.array([[t['x_enu_m'], t['y_enu_m']] for t in prev_map_only], 'f8')
        pm2n, _ = greedy_pairs(PM, NEW, lambda i: 3.0)
        kept_map_only = len(pm2n)
        named = sum(1 for _, (j, _d) in pm2n.items()
                    if out[j]['inv_id'] and out[j]['species'])
    else:
        kept_map_only = 0

    # ---- 8. join QA: match distances, height agreement and match counts --- #
    dm = np.array([v[1] for v in inv2crown.values()]) if inv2crown else np.zeros(0)
    do_ = np.array([v[1] for v in inv2osm.values()]) if inv2osm else np.zeros(0)
    fh = [(inv[i]['height_ft'] * 0.3048, crowns[j]['height_m'])
          for i, (j, _d) in inv2crown.items() if inv[i]['height_ft']]
    join_qa = dict(
      inventory_to_crown=dict(n=len(dm),
        median_m=round(float(np.median(dm)), 2) if len(dm) else None,
        p90_m=round(float(np.percentile(dm, 90)), 2) if len(dm) else None,
        pct_within_3m=round(float((dm < 3).mean() * 100), 1) if len(dm) else None,
        pct_within_1m=round(float((dm < 1).mean() * 100), 1) if len(dm) else None),
      inventory_to_osm_node=dict(n=len(do_),
        median_m=round(float(np.median(do_)), 2) if len(do_) else None,
        p90_m=round(float(np.percentile(do_, 90)), 2) if len(do_) else None),
      field_height_vs_lidar_crown_height=dict(n=len(fh),
        median_residual_m=round(float(np.median([b - a for a, b in fh])), 2) if fh else None,
        mae_m=round(float(np.mean([abs(b - a) for a, b in fh])), 2) if fh else None,
        pct_within_3m=round(float(np.mean([abs(b - a) < 3 for a, b in fh]) * 100), 1) if fh else None))
    print("join QA:", json.dumps(join_qa))

    rej = [dict(x_enu_m=round(crowns[j]['x'], 2), y_enu_m=round(crowns[j]['y'], 2),
                height_m=crowns[j]['height_m'],
                crown_radius_m=crowns[j]['crown_radius_m'],
                green_frac=crowns[j]['green_frac'], porosity=crowns[j]['porosity'],
                over_water=crowns[j]['over_water'], reason=why_)
           for j, _, why_ in rejected]

    stats = dict(
        inventory_points_in_bbox=src['counts']['combined_in_bbox'],
        inventory_points_in_model_extent=len(inv),
        inventory_matched_to_a_lidar_crown=len(inv2crown),
        inventory_matched_to_an_osm_node=len(inv2osm),
        inventory_with_no_lidar_canopy=rescued,
        inventory_revived_crowns_the_classifier_had_rejected=revived,
        previous_map_only_trees=len(prev_map_only),
        previous_map_only_still_placed=kept_map_only,
        previous_map_only_now_named_by_the_inventory=named,
        inventory_with_species=sum(1 for t in out if t['inv_id'] and t['species']),
        inventory_with_dbh=sum(1 for t in out if t['dbh_source'] == 'inventory'),
        trunk_from_fitted_allometry=sum(1 for t in out
                                        if t['dbh_source'] == 'allometry_from_lidar_height'),
        height_tiers=dict(tiers),
        crowns_detected=len(crowns),
        crowns_claimed_by_inventory=len(crown2inv),
        crowns_unclaimed_accepted_as_trees=len(kept_c),
        crowns_unclaimed_rejected_as_not_a_tree=len(rejected),
        osm_trees_in_bbox=len(O),
        osm_matched_to_inventory=len(inv2osm),
        osm_matched_to_an_unclaimed_crown=len(o2c),
        osm_unmatched_but_canopy_present=n_merged,
        osm_unmatched_no_lidar_canopy=n_none,
        total_trees=len(out),
        previous_total_trees=len(prev),
        new_trees_not_previously_placed=added,
        trees_removed_as_not_a_tree=removed,
        previous_placements_absorbed_by_an_inventory_point=absorbed,
        median_shift_of_kept_trees_m=move_med,
        confidence=dict(high=sum(1 for t in out if t['confidence'] == 'high'),
                        medium=sum(1 for t in out if t['confidence'] == 'medium'),
                        low=sum(1 for t in out if t['confidence'] == 'low')),
        height_m=dict(median=round(float(np.median([t['height_m'] for t in out])), 2),
                      p90=round(float(np.percentile([t['height_m'] for t in out], 90)), 2),
                      max=round(float(max(t['height_m'] for t in out)), 2)),
        crown_radius_m=dict(median=round(float(np.median([t['crown_radius_m'] for t in out])), 2),
                            max=round(float(max(t['crown_radius_m'] for t in out)), 2)),
        trunk_radius_m=dict(median=round(float(np.median([t['trunk_radius_m'] for t in out])), 3),
                            max=round(float(max(t['trunk_radius_m'] for t in out)), 3)),
        species_distinct=len({t['species'] for t in out if t['species']}),
        placed_inside_a_building_footprint=sum(1 for t in out
                                               if t['inside_building_footprint']),
        join_qa=join_qa)

    # an area keeps only trees whose trunk is inside its own bbox: crowns in the lidar
    # pad would otherwise be placed beyond the ground mesh.
    if hasattr(C, '_areacfg') and C._areacfg.build_area_cfg(C) is not None:
        _x0, _y0, _x1, _y1 = C._areacfg.enu_ext(C)
        _n0 = len(out)
        out = [t for t in out if _x0 <= t['x_enu_m'] <= _x1 and _y0 <= t['y_enu_m'] <= _y1]
        stats['clipped_outside_area_bbox'] = _n0 - len(out)
    doc = dict(
        what="Tree placements for the area's 3D model. Positions, species and "
             "trunk diameters come from the city's public tree inventory where "
             "it has a record; heights and crown spreads come from 3DEP lidar where it has "
             "a crown. The trees themselves are not scanned: swap the placeholder geometry "
             "in model.glb (nodes named tree_NNNN under LAYER_trees) for real assets using "
             "these records.",
        frame=dict(type="ENU metres", origin_lat_lon_alt=list(C.ORIGIN),
                   note="x=East, y=North, z=Up in metres about the origin. In the GLB the "
                        "same point is (x, z, -y) because glTF is Y-up; 1 unit = 1 metre."),
        method=dict(
          recipe="place from a published point inventory, join lidar for height, never "
                 "segment the canopy for position",
          anchor=(src.get('what') or "no city inventory"),
          matching=dict(inventory_to_crown_m=[CROWN_MATCH_MIN, CROWN_MATCH_MAX],
                        inventory_to_osm_node_m=OSM_MATCH,
                        note="one-to-one, shortest distance first"),
          height_tiers=["lidar_crown",
                        "chm_window (11 x 11 cells at 0.5 m, building and open-water "
                        "cells blanked first so the window cannot read a roof)",
                        "field_height (inventory feet to metres)", "dbh_allometry",
                        "area_median"],
          allometry=allom,
          trunk="inventory DBH in inches, halved and converted to metres; that is a measured "
                "diameter at breast height, not a rule of thumb",
          detection="0.5 m canopy height model from 3DEP lidar, crowns grown to local maxima "
                    "with a height-scaled cap (s03), classified on foliage colour, return "
                    "porosity and OSM agreement (s05) only where the inventory has no record",
          licences=dict(lidar=(C._areacfg.lidar_licence(C) if hasattr(C, '_areacfg') else "USGS 3DEP, public domain"),
                        inventory=src.get('licence'),
                        osm="(c) OpenStreetMap contributors, ODbL 1.0",
                        imagery="Mapillary, CC BY-SA 4.0")),
        sources=src['endpoints'],
        stats=stats, trees=out, rejected_crowns=rej)
    json.dump(doc, open(C.W + '/tree_inventory.json', 'w'), indent=1)

    cols = ["id", "source", "confidence", "height_tier", "x_enu_m", "y_enu_m",
            "ground_z_enu_m", "lat", "lon", "height_m", "crown_base_m", "crown_radius_m",
            "trunk_radius_m", "dbh_in", "species", "common_name", "leaf_type", "leaf_cycle",
            "inv_id", "inv_dept", "planted_year", "condition", "osm_id", "osm_dist_m",
            "chm_window_max_m", "field_height_m", "green_frac", "porosity", "evidence"]
    with open(C.W + '/tree_inventory.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        wr.writeheader()
        for t in out: wr.writerow(t)
    print(json.dumps(stats, indent=1))
    C.note('trees', stats)
    C.mark('tree_inventory_join')
    print("%.0fs" % (time.time() - t0))


if __name__ == '__main__':
    main()
