"""Final tree inventory: classify crowns, merge with OSM, write JSON + CSV."""
import sys, os, json, math, csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
from scipy.spatial import cKDTree
import common as C

GREEN_T = 0.50
POR_T   = 0.45

def classify(r):
    g = r['green_frac']; dos = r['osm_dist_m'] if r['osm_dist_m'] is not None else 999
    if r['over_water'] and dos > 6:
        return None, "over water, no mapped tree: mast, piling or moored boat"
    if g is not None and g >= GREEN_T:
        if dos <= 4: return "high", "photo sees foliage colour and OSM maps a tree here"
        return "high" if r['porosity'] >= POR_T else "medium", \
               "photo sees foliage colour in the crown volume"
    if g is None and dos <= 4:
        return "medium", "OSM maps a tree and the lidar has a crown here; no camera reached it"
    if g is None and not r['over_water'] and r['porosity'] >= POR_T and r['height_m'] >= 3.0:
        return "low", "lidar crown is porous like foliage; no photo colour, no OSM tree"
    if g is not None and g < GREEN_T:
        return None, "photo sees the crown and it is not foliage-coloured"
    return None, "no foliage evidence"

def main():
    R = json.load(open(C.W+'/crowns_scored.json'))
    O = json.load(open(C.W+'/trees_raw.json'))['osm_trees']
    Z = np.load(C.W+'/geom/chm.npz')
    chm = Z['chm']; gx0,gy0,cell,nx,ny = Z['grid']; nx=int(nx); ny=int(ny)
    S = np.load(C.FW+'/surfaces.npz')
    dx0,dy0,dc,dnx,dny = [float(v) for v in S['grid']]; dnx=int(dnx); dny=int(dny)
    dtm = S['dtm']
    def dtm_at(x,y):
        return float(dtm[int(np.clip((y-dy0)/dc,0,dny-1)), int(np.clip((x-dx0)/dc,0,dnx-1))])
    def chm_near(x,y,rad):
        i0=int(np.clip((x-gx0-rad)/cell,0,nx-1)); i1=int(np.clip((x-gx0+rad)/cell,0,nx-1))
        j0=int(np.clip((y-gy0-rad)/cell,0,ny-1)); j1=int(np.clip((y-gy0+rad)/cell,0,ny-1))
        sub = chm[j0:j1+1, i0:i1+1]
        return float(sub.max()) if sub.size else 0.0

    trees=[]; rejected=[]
    for r in R:
        conf, why = classify(r)
        (trees if conf else rejected).append((r, conf, why))
    print(f"crowns {len(R)}: {len(trees)} accepted as trees, {len(rejected)} rejected")

    # ---- one-to-one match of accepted crowns to OSM tree nodes ----------- #
    out=[]
    if O:
        ko = cKDTree(np.array([[t['x'],t['y']] for t in O]))
        pairs=[]
        for i,(r,conf,why) in enumerate(trees):
            for j in ko.query_ball_point([r['x'],r['y']], max(4.0, r['crown_radius_m'])):
                d = math.dist((r['x'],r['y']),(O[j]['x'],O[j]['y']))
                pairs.append((d,i,j))
        pairs.sort()
        ci={}; oj={}
        for d,i,j in pairs:
            if i in ci or j in oj: continue
            ci[i]=j; oj[j]=i
    else:
        ci={}; oj={}
    for i,(r,conf,why) in enumerate(trees):
        j = ci.get(i); o = O[j] if j is not None else None
        h = r['height_m']; cb = r['crown_base_m']; rad = r['crown_radius_m']
        out.append(dict(
            id=None, source=("lidar+osm" if o else "lidar"),
            confidence=conf, evidence=why,
            x_enu_m=round(r['x'],2), y_enu_m=round(r['y'],2),
            ground_z_enu_m=round(r['z_base'],2),
            height_m=round(h,2), crown_base_m=round(cb,2),
            crown_radius_m=round(rad,2), crown_area_m2=r['crown_area_m2'],
            trunk_radius_m=round(float(np.clip(0.06*h, 0.08, 0.45)),2),
            osm_id=(o['osm_id'] if o else None),
            osm_dist_m=(round(math.dist((r['x'],r['y']),(o['x'],o['y'])),2) if o else None),
            species=(o.get('species') if o else None),
            leaf_type=(o.get('leaf_type') if o else None),
            lidar_returns=r['lidar_returns'], porosity=r['porosity'],
            green_frac=r['green_frac'], over_water=r['over_water']))
    # ---- OSM trees the lidar did not give a separate crown --------------- #
    med_h = float(np.median([t['height_m'] for t in out])) if out else 7.0
    med_r = float(np.median([t['crown_radius_m'] for t in out])) if out else 2.5
    n_merged=n_none=0
    for j,o in enumerate(O):
        if j in oj: continue
        top = chm_near(o['x'],o['y'],4.0)
        if top >= 2.5:
            n_merged+=1; src="osm+lidar_canopy"; conf="medium"
            why="OSM maps a tree and the lidar has canopy here, but it merges into a neighbouring crown"
            h=round(top,2); rad=round(min(med_r, 3.0),2)
        else:
            n_none+=1; src="osm_only"; conf="low"
            why="OSM maps a tree; the lidar has no canopy above 2.5 m within 4 m (young, removed, or planted after the survey)"
            h=round(med_h,2); rad=round(med_r,2)
        try: ht=float(str(o.get('height_tag')).split()[0]); h=round(ht,2)
        except Exception: pass
        out.append(dict(id=None, source=src, confidence=conf, evidence=why,
            x_enu_m=round(o['x'],2), y_enu_m=round(o['y'],2),
            ground_z_enu_m=round(dtm_at(o['x'],o['y']),2),
            height_m=h, crown_base_m=round(min(2.0, 0.35*h),2),
            crown_radius_m=rad, crown_area_m2=round(math.pi*rad*rad,1),
            trunk_radius_m=round(float(np.clip(0.06*h,0.08,0.45)),2),
            osm_id=o['osm_id'], osm_dist_m=0.0,
            species=o.get('species'), leaf_type=o.get('leaf_type'),
            lidar_returns=0, porosity=None, green_frac=None, over_water=False))
    out.sort(key=lambda t:(t['y_enu_m'], t['x_enu_m']))
    for k,t in enumerate(out,1):
        t['id']=f"tree_{k:04d}"
        lat,lon,_ = C.enu_to_lla(t['x_enu_m'], t['y_enu_m'], 0.0)
        t['lat']=round(float(lat),8); t['lon']=round(float(lon),8)
    # rejected crowns are still useful: they are the masts, poles and roof
    # clutter the canopy filter caught, and they say what was NOT called a tree
    rej=[dict(x_enu_m=round(r['x'],2), y_enu_m=round(r['y'],2),
              height_m=r['height_m'], crown_radius_m=r['crown_radius_m'],
              green_frac=r['green_frac'], porosity=r['porosity'],
              over_water=r['over_water'], reason=why) for r,_,why in rejected]
    stats=dict(
        crowns_detected=len(R), crowns_accepted_as_trees=len(trees),
        crowns_rejected=len(rejected),
        osm_trees_in_bbox=len(O), osm_matched_to_a_lidar_crown=len(oj),
        osm_unmatched_but_canopy_present=n_merged,
        osm_unmatched_no_lidar_canopy=n_none,
        lidar_only_not_in_osm=sum(1 for t in out if t['source']=='lidar'),
        total_trees=len(out),
        confidence=dict(high=sum(1 for t in out if t['confidence']=='high'),
                        medium=sum(1 for t in out if t['confidence']=='medium'),
                        low=sum(1 for t in out if t['confidence']=='low')),
        height_m=dict(median=round(float(np.median([t['height_m'] for t in out])),2),
                      p90=round(float(np.percentile([t['height_m'] for t in out],90)),2),
                      max=round(float(max(t['height_m'] for t in out)),2)),
        crown_radius_m=dict(median=round(float(np.median([t['crown_radius_m'] for t in out])),2),
                            max=round(float(max(t['crown_radius_m'] for t in out)),2)))
    # an area keeps only trees whose trunk is inside its own bbox: crowns in the lidar
    # pad would otherwise be placed beyond the ground mesh.
    if hasattr(C, '_areacfg') and C._areacfg.build_area_cfg(C) is not None:
        _x0, _y0, _x1, _y1 = C._areacfg.enu_ext(C)
        _n0 = len(out)
        out = [t for t in out if _x0 <= t['x_enu_m'] <= _x1 and _y0 <= t['y_enu_m'] <= _y1]
        stats['clipped_outside_area_bbox'] = _n0 - len(out)
    doc=dict(
        what="Tree placements for the area's 3D model. Positions and sizes "
             "are measured; the trees themselves are not scanned. Swap the placeholder "
             "geometry in model.glb (nodes named tree_NNNN under LAYER_trees) for real "
             "assets using these records.",
        frame=dict(type="ENU metres", origin_lat_lon_alt=list(C.ORIGIN),
                   note="x=East, y=North, z=Up in metres about the origin. In the GLB the "
                        "same point is (x, z, -y) because glTF is Y-up; 1 unit = 1 metre."),
        method=dict(
          detection="0.5 m canopy height model from 3DEP lidar (height above the class-2 "
                    "ground surface), pinholes closed, local maxima with a window that grows "
                    "with height, crowns grown to the nearest apex with a height-scaled cap",
          classification="the lidar carries no reliable vegetation class, so each crown is "
                         "judged on evidence: foliage colour in the photographs "
                         "(excess green > 0.06 on more than half the crown points), return "
                         "porosity through the crown, and whether OSM maps a tree there",
          thresholds=dict(min_height_m=2.5, min_crown_area_m2=2.0,
                          green_fraction=GREEN_T, porosity=POR_T),
          licences=dict(lidar=(C._areacfg.lidar_licence(C) if hasattr(C, '_areacfg') else "USGS 3DEP, public domain"),
                        osm="(c) OpenStreetMap contributors, ODbL 1.0",
                        imagery="Mapillary, CC BY-SA 4.0")),
        stats=stats, trees=out, rejected_crowns=rej)
    json.dump(doc, open(C.W+'/tree_inventory.json','w'), indent=1)
    cols=["id","source","confidence","x_enu_m","y_enu_m","ground_z_enu_m","lat","lon",
          "height_m","crown_base_m","crown_radius_m","trunk_radius_m","osm_id",
          "osm_dist_m","species","leaf_type","green_frac","porosity","evidence"]
    with open(C.W+'/tree_inventory.csv','w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); wr.writeheader()
        for t in out: wr.writerow(t)
    print(json.dumps(stats,indent=1))

if __name__=='__main__': main()
