"""Decide which lidar crowns are trees, merge with the OSM tree nodes, and
write the inventory.

The lidar may carry no vegetation class, so 'tree' has to be earned from
evidence: how the returns are distributed through the crown (foliage is porous,
a roof is a single surface), whether the photo reconstruction sees green there,
whether OSM mapped a tree at the spot, and where it stands."""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
from scipy.spatial import cKDTree
import common as C

def main():
    raw = json.load(open(C.W+'/trees_raw.json'))
    crowns = raw['crowns']; osm = raw['osm_trees']
    Z = np.load(C.W+'/geom/chm.npz')
    gx0,gy0,cell,nx,ny = Z['grid']; nx=int(nx); ny=int(ny)
    d = np.load(C.FW+'/loaded.npz')
    cls = d['lidar_cls']; prgb = d['photo_rgb']; photo = d['photo']
    xyz = np.load(C.FW+'/lidar_reg.npy')
    S = np.load(C.FW+'/surfaces.npz')
    dx0,dy0,dc,dnx,dny = [float(v) for v in S['grid']]; dnx=int(dnx); dny=int(dny)
    dtm = S['dtm']
    def dtm_at(x,y):
        ix=np.clip(((x-dx0)/dc).astype(int),0,dnx-1); iy=np.clip(((y-dy0)/dc).astype(int),0,dny-1)
        return dtm[iy,ix]
    hag_l = xyz[:,2]-dtm_at(xyz[:,0],xyz[:,1])
    good = (cls!=7)&(cls!=18)&(cls!=9)
    L = xyz[good]; HL = hag_l[good]
    hag_p = photo[:,2]-dtm_at(photo[:,0],photo[:,1])
    klid = cKDTree(L[:,:2]); kpho = cKDTree(photo[:,:2])
    # measured water surface from the combined build (lidar class 9 + OSM)
    wm = S['m_water']
    def water_at(x,y):
        ix=np.clip(int((x-dx0)/dc),0,dnx-1); iy=np.clip(int((y-dy0)/dc),0,dny-1)
        return bool(wm[iy,ix])

    ok = cKDTree(np.array([[t['x'],t['y']] for t in osm])) if osm else None
    recs=[]
    for i,c in enumerate(crowns):
        r = c['crown_radius_m']; x,y = c['x'],c['y']
        # vertical structure of the lidar returns through the crown column
        idx = klid.query_ball_point([x,y], r)
        h = HL[idx] if idx else np.zeros(0)
        top = c['height_m']
        inside = h[h > 0.5]
        if len(inside) >= 4:
            porosity = float(((inside < top-1.0) & (inside > 1.0)).sum()/len(inside))
            vspread  = float(np.std(inside))
            nret = len(inside)
        else:
            porosity, vspread, nret = 0.0, 0.0, len(inside)
        # photo colour in the crown volume
        pidx = kpho.query_ball_point([x,y], max(r,1.5))
        gfrac = None; npx = 0
        if pidx:
            pi = np.array(pidx)
            sel = pi[(hag_p[pi] > max(1.5, c['crown_base_m']-1.0)) & (hag_p[pi] < top+2.0)]
            if len(sel) >= 30:
                rgb = prgb[sel].astype('f4')
                s = rgb.sum(1)+1e-6
                exg = (2*rgb[:,1]-rgb[:,0]-rgb[:,2])/s
                gfrac = float((exg > 0.06).mean()); npx = int(len(sel))
        dos = float('inf'); osm_id=None
        if ok is not None:
            dd, jj = ok.query([x,y], k=1)
            dos = float(dd); osm_id = osm[int(jj)]['osm_id']
        c2 = dict(c)
        c2.update(porosity=round(porosity,3), vspread_m=round(vspread,2),
                  lidar_returns=nret, green_frac=(None if gfrac is None else round(gfrac,3)),
                  photo_points=npx, osm_dist_m=round(dos,2) if math.isfinite(dos) else None,
                  osm_nearest_id=osm_id, over_water=water_at(x,y))
        recs.append(c2)
    json.dump(recs, open(C.W+'/crowns_scored.json','w'))
    # ---- quick separation report ---------------------------------------- #
    A=np.array([[r['porosity'],r['vspread_m'],(r['green_frac'] if r['green_frac'] is not None else -1),
                 r['osm_dist_m'] if r['osm_dist_m'] is not None else 999,
                 r['height_m'],r['crown_radius_m'],r['slenderness'],r['top_roughness_m']] for r in recs])
    near = A[:,3] < 4.0
    print(f"{len(recs)} crowns; {near.sum()} within 4 m of an OSM tree node")
    for nm,k in (("porosity",0),("vspread",1),("green_frac",2),("height",4),
                 ("radius",5),("slender",6),("roughness",7)):
        a=A[near,k]; b=A[~near,k]
        print(f"  {nm:>10}: osm-matched med {np.median(a):7.2f}  unmatched med {np.median(b):7.2f}")
    print("over water:",sum(r['over_water'] for r in recs),
          "near building:",sum(r['near_building'] for r in recs),
          "in grass:",sum(r['in_grass'] for r in recs),
          "no photo colour:",sum(r['green_frac'] is None for r in recs))

if __name__=='__main__':
    main()
