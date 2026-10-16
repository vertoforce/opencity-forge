"""Tree inventory: crowns segmented out of the lidar canopy height model,
cross-checked against individually mapped OSM trees.

Nothing here is a guess about what a tree looks like: a crown is a local
maximum in the lidar height-above-ground surface that is not inside a building
footprint, not over water, and tall and compact enough to be a tree rather than
a kiosk roof or a mast.  Every record carries where the evidence came from."""
import sys, os, json, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import common as C
import memguard

CELL = 0.5           # CHM raster, metres
MIN_H = 2.5          # crown must stand this far above ground
MAX_H = 45.0
MIN_AREA = 2.0       # m2 of canopy
MIN_R, MAX_R = 0.8, 14.0

def load_lidar():
    d = np.load(C.FW+'/loaded.npz')
    cls = d['lidar_cls']
    xyz = np.load(C.FW+'/lidar_reg.npy')      # already in the photo ENU frame
    assert len(xyz) == len(cls)
    return xyz, cls

def dtm_lookup(xy):
    S = np.load(C.FW+'/surfaces.npz')
    x0,y0,c,nx,ny = [float(v) for v in S['grid']]; nx=int(nx); ny=int(ny)
    dtm = S['dtm']
    ix = np.clip(((xy[:,0]-x0)/c).astype(int),0,nx-1)
    iy = np.clip(((xy[:,1]-y0)/c).astype(int),0,ny-1)
    return dtm[iy,ix], (x0,y0,c,nx,ny), S

def rasterise_polys(polys, x0, y0, cell, nx, ny):
    """Even-odd scanline fill, no external deps."""
    m = np.zeros((ny,nx), bool)
    for P in polys:
        if len(P) < 3: continue
        px = (P[:,0]-x0)/cell; py = (P[:,1]-y0)/cell
        j0 = max(0,int(np.floor(py.min()))); j1 = min(ny-1,int(np.ceil(py.max())))
        for j in range(j0, j1+1):
            yc = j+0.5
            xs = []
            for k in range(len(px)-1):
                ya, yb = py[k], py[k+1]
                if (ya <= yc < yb) or (yb <= yc < ya):
                    t = (yc-ya)/(yb-ya)
                    xs.append(px[k] + t*(px[k+1]-px[k]))
            if not xs: continue
            xs.sort()
            for a,b in zip(xs[0::2], xs[1::2]):
                ia = max(0,int(np.ceil(a-0.5))); ib = min(nx-1,int(np.floor(b-0.5)))
                if ib >= ia: m[j, ia:ib+1] = True
    return m

def closed(P):
    return P if np.allclose(P[0],P[-1]) else np.vstack([P,P[:1]])

def osm_layers():
    els = C.osm_elements()
    trees, builds, water, piers, grass = [], [], [], [], []
    for e in els:
        t = e.get('tags') or {}
        if e['type']=='node' and t.get('natural')=='tree':
            p = C.lla_to_enu(e['lat'], e['lon'], C.ORIGIN[2])
            trees.append(dict(osm_id=e['id'], x=float(p[0]), y=float(p[1]),
                              lat=e['lat'], lon=e['lon'],
                              species=t.get('species') or t.get('genus'),
                              leaf_type=t.get('leaf_type'),
                              height_tag=t.get('height'),
                              circumference=t.get('circumference')))
        if e['type']=='way' and e.get('geometry'):
            P = C.way_xy(e)
            if P is None or len(P)<3: continue
            if t.get('building') or t.get('building:part'):
                builds.append((closed(P), t, e['id']))
            if t.get('natural')=='water':
                water.append(closed(P))
            if t.get('man_made')=='pier':
                piers.append(closed(P))
            if t.get('landuse')=='grass' or t.get('leisure')=='park':
                grass.append(closed(P))
    return trees, builds, water, piers, grass

def variable_maxima(chm, cell):
    """Local maxima with a window that grows with crown height: small trees are
    a few metres across, mature ones over ten, so one fixed window either
    merges neighbours or shatters big crowns."""
    peaks = np.zeros(chm.shape, bool)
    bands = [(2.5, 6.0, 1.5), (6.0, 11.0, 2.5), (11.0, 18.0, 3.5), (18.0, 99.0, 4.5)]
    for lo, hi, rad in bands:
        k = max(3, int(round(2*rad/cell))|1)
        mx = ndimage.maximum_filter(chm, size=k, mode='nearest')
        band = (chm >= lo) & (chm < hi) & (chm >= mx - 1e-6) & (chm > 0)
        peaks |= band
    return peaks

def main():
    if not hasattr(C, '_areacfg') or C._areacfg.build_area_cfg(C) is None:
        sys.exit('s03_trees: set OPENCITY_AREA to an area with paths.scratch_dir')
    t0 = time.time()
    memguard.require("trees", 4.0, need_disk_gb=2.0)
    xyz, cls = load_lidar()
    trees_osm, builds, water, piers, grass = osm_layers()
    print(f"osm: {len(trees_osm)} mapped trees, {len(builds)} building polys, "
          f"{len(water)} water, {len(piers)} piers", flush=True)

    # ---- canopy height model -------------------------------------------- #
    zg, grid, S = dtm_lookup(xyz[:,:2])
    hag = xyz[:,2] - zg
    keep = (cls != 7) & (cls != 18) & (cls != 9) & (hag > 0.3) & (hag < MAX_H)
    # the CHM covers the area's own bbox + 6 m (a crown at the edge still finds its
    # apex); points beyond are dropped, not clipped onto the edge cells.  s05/s05b
    # then keep trunks inside the bbox only.
    _x0, _y0, _x1, _y1 = C._areacfg.enu_ext(C, pad=6.0)
    gx0, gy0 = _x0, _y0
    nx = int(math.ceil((_x1-gx0)/CELL)); ny = int(math.ceil((_y1-gy0)/CELL))
    _in = (xyz[:,0] >= gx0) & (xyz[:,0] < gx0+nx*CELL) & (xyz[:,1] >= gy0) & (xyz[:,1] < gy0+ny*CELL)
    keep &= _in
    P = xyz[keep]; H = hag[keep]
    ix = np.clip(((P[:,0]-gx0)/CELL).astype(int),0,nx-1)
    iy = np.clip(((P[:,1]-gy0)/CELL).astype(int),0,ny-1)
    lin = iy*nx+ix
    chm = np.zeros(nx*ny,'f4'); np.maximum.at(chm, lin, H.astype('f4'))
    chm = chm.reshape(ny,nx)
    # lidar is ~7 pts/m2 so a 0.5 m cell is often empty inside a real crown;
    # close those pinholes before looking for maxima.
    chm_f = ndimage.grey_closing(chm, size=3)
    chm_f = ndimage.gaussian_filter(chm_f, 1.0)

    bmask = rasterise_polys([b[0] for b in builds], gx0, gy0, CELL, nx, ny)
    bmask = ndimage.binary_dilation(bmask, iterations=int(round(1.5/CELL)))
    wmask = rasterise_polys(water, gx0, gy0, CELL, nx, ny)
    wmask = ndimage.binary_erosion(wmask, iterations=int(round(3.0/CELL)))
    pmask = rasterise_polys(piers, gx0, gy0, CELL, nx, ny)
    gmask = rasterise_polys(grass, gx0, gy0, CELL, nx, ny)
    print(f"masks: building {bmask.sum()*CELL**2:,.0f} m2, water {wmask.sum()*CELL**2:,.0f} m2, "
          f"grass/park {gmask.sum()*CELL**2:,.0f} m2", flush=True)

    canopy = (chm_f >= MIN_H) & ~bmask & ~(wmask & ~pmask)
    canopy = ndimage.binary_opening(canopy, structure=np.ones((3,3)))
    print(f"canopy area {canopy.sum()*CELL**2:,.0f} m2", flush=True)

    peaks = variable_maxima(np.where(canopy, chm_f, 0.0), CELL) & canopy
    pl, npk = ndimage.label(peaks, structure=np.ones((3,3)))
    cen = np.array(ndimage.center_of_mass(peaks, pl, range(1,npk+1))) if npk else np.zeros((0,2))
    print(f"{npk} candidate crown apexes", flush=True)

    # ---- grow crowns: nearest apex, distance capped by apex height ------- #
    cy, cx = np.nonzero(canopy)
    apex_h = np.array([chm_f[int(round(r)), int(round(c))] for r,c in cen]) if npk else np.zeros(0)
    tree_cap = np.clip(1.0 + 0.30*apex_h, 2.0, MAX_R)       # metres
    kt = cKDTree(np.stack([cen[:,1]*CELL+gx0+CELL/2, cen[:,0]*CELL+gy0+CELL/2],1))
    qxy = np.stack([cx*CELL+gx0+CELL/2, cy*CELL+gy0+CELL/2],1)
    dist, who = kt.query(qxy, k=1)
    ok = dist <= tree_cap[who]
    lab = np.full(canopy.shape, -1, 'i4'); lab[cy[ok], cx[ok]] = who[ok]

    # ---- per-crown measurements ------------------------------------------ #
    recs = []
    order = np.argsort(who[ok], kind='stable')
    w_s = who[ok][order]; qx = qxy[ok][order]
    hh = chm_f[cy[ok], cx[ok]][order]
    starts = np.searchsorted(w_s, np.arange(npk)); ends = np.searchsorted(w_s, np.arange(npk)+1)
    dtm = S['dtm']; gx0d, gy0d, cd, nxd, nyd = [float(v) for v in S['grid']]
    for i in range(npk):
        a, b = starts[i], ends[i]
        n = b-a
        if n == 0: continue
        area = n*CELL*CELL
        if area < MIN_AREA: continue
        pts = qx[a:b]; hs = hh[a:b]
        # apex-weighted centre: the top of the crown, not the drip line
        wgt = np.maximum(hs - MIN_H, 0.05)**2
        cxy = (pts*wgt[:,None]).sum(0)/wgt.sum()
        r_eq = math.sqrt(area/math.pi)
        r_max = float(np.percentile(np.linalg.norm(pts-cxy,axis=1), 90))
        r = float(np.clip(0.5*(r_eq+r_max), MIN_R, MAX_R))
        h_top = float(hs.max())
        if not (MIN_H <= h_top <= MAX_H): continue
        # ground height under the crown centre
        jx = int(np.clip((cxy[0]-gx0d)/cd,0,nxd-1)); jy = int(np.clip((cxy[1]-gy0d)/cd,0,nyd-1))
        zbase = float(dtm[jy,jx])
        # crown base: lowest canopy return in the column, floored sensibly
        col = (np.abs(P[:,0]-cxy[0]) < r) & (np.abs(P[:,1]-cxy[1]) < r)
        hcol = H[col]
        hcrown = float(np.percentile(hcol[hcol>=MIN_H],5)) if (hcol>=MIN_H).any() else MIN_H
        crown_base = float(np.clip(hcrown, 1.0, max(1.2, h_top-1.0)))
        # shape test: a tree crown is roughly as tall as it is wide and has a
        # ragged top; a kiosk or container roof is flat and much wider than tall
        slender = h_top / max(2*r, 1e-6)
        rough = float(np.std(hs))
        ipx = int(np.clip((cxy[0]-gx0)/CELL,0,nx-1)); ipy = int(np.clip((cxy[1]-gy0)/CELL,0,ny-1))
        recs.append(dict(x=float(cxy[0]), y=float(cxy[1]), z_base=zbase,
                         height_m=round(h_top,2), crown_base_m=round(crown_base,2),
                         crown_radius_m=round(r,2), crown_area_m2=round(area,1),
                         slenderness=round(slender,3), top_roughness_m=round(rough,2),
                         lidar_points=int(n),
                         in_grass=bool(gmask[ipy,ipx]), on_pier=bool(pmask[ipy,ipx]),
                         near_building=bool(bmask[ipy,ipx])))
    print(f"{len(recs)} crowns after size filter, {time.time()-t0:.0f}s", flush=True)
    json.dump(dict(cell=CELL, grid=[gx0,gy0,CELL,nx,ny], crowns=recs,
                   osm_trees=trees_osm),
              open(C.W+'/trees_raw.json','w'))
    np.savez_compressed(C.W+'/geom/chm.npz', chm=chm_f.astype('f4'),
                        canopy=canopy, grid=np.array([gx0,gy0,CELL,nx,ny],'f8'),
                        bmask=bmask, wmask=wmask, pmask=pmask, gmask=gmask)
    print("wrote trees_raw.json and geom/chm.npz")

if __name__ == '__main__':
    main()
