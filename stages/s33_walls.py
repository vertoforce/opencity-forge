"""Stage 33 -- the wall inventory.

Every building mass in layers/buildings.npz is an extruded OSM footprint, so
its side faces are exact planar rectangles.  This groups the side triangles
into those rectangles, gives each one an in-plane basis (u along the wall,
v straight up), and records the OSM evidence that will decide its material when
no camera saw it: building:material, building:colour, building:levels, name.

Output: facades/walls.npz + facades/walls.json.  Nothing here reads imagery.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import numpy as np
import common as C
import texlib as T

FAC = T.SCR + '/facades'
os.makedirs(FAC, exist_ok=True)
UP_WALL = 0.30          # |nz| below this is a wall


def footprint_segments(pos, idx, faces):
    """The part's footprint edges in plan, each edge ONCE.

    A footprint edge is extruded into one to four triangles (one per height
    step of its neighbours), and each of them spans the edge in plan.  Taking
    one base segment per triangle would count an edge once per triangle, so the
    even-odd crossing count below would read the parity of the triangle count,
    not of the footprint.  So each triangle gives its two distinct plan points,
    and the segments are de-duplicated on their end points.

    A ring can also be open: where two masses abut, the shared wall is not
    extruded (a mass can have no east or north wall at all), and an
    open ring has no inside.  Each end point used by an odd number of edges is
    a loose end; loose ends are joined nearest-first, which closes the ring
    along the missing wall."""
    seen = set(); out = []
    for fi in faces:
        v = pos[idx[fi]][:, :2]
        d = [np.hypot(*(v[a] - v[b])) for a, b in ((0, 1), (0, 2), (1, 2))]
        a, b = ((0, 1), (0, 2), (1, 2))[int(np.argmax(d))]
        if max(d) < 1e-3:
            continue
        k = tuple(sorted((tuple(np.round(v[a], 3)), tuple(np.round(v[b], 3)))))
        if k in seen:
            continue
        seen.add(k)
        out.append((np.array(k[0]), np.array(k[1])))
    deg = {}
    for k in seen:
        for e in k:
            deg[e] = deg.get(e, 0) + 1
    loose = [np.array(e) for e, c in sorted(deg.items()) if c % 2]
    while len(loose) >= 2:
        a = loose.pop(0)
        j = int(np.argmin([np.hypot(*(a - b)) for b in loose]))
        out.append((a, loose.pop(j)))
    return out


def main():
    t0 = time.time()
    global DSM
    DSM = T.DSM()
    z = np.load(C.W + '/layers/buildings.npz')
    meta = json.load(open(C.W + '/layers/buildings.json'))
    pos = z['pos'].astype('f8'); idx = z['idx'].astype('i8')
    tri = pos[idx]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    nu = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    cen = tri.mean(1)
    is_wall = (np.abs(nu[:, 2]) < UP_WALL) & (area > 1e-4)

    els = C.osm_elements()
    bytag = {e['id']: e.get('tags', {}) for e in els}

    walls = []
    part_segs = {}
    wall_of_face = np.full(len(idx), -1, 'i4')
    for prt in meta['parts']:
        a0, b0 = prt['face_start'], prt['face_end']
        f = np.arange(a0, b0)
        # an open shelter's posts (the part's last post_faces faces,
        # s10a_layers._posts) are 0.5 m columns, not walls, but their coplanar
        # outer faces would merge into one plane group wider than the 0.4 m minimum
        # and take a facade photo strip, so they are excluded; only the fascia
        # band stays a wall.
        osh = (prt.get('extras') or {}).get('open_shelter')
        if osh:
            f = f[f < b0 - int(osh.get('post_faces') or 0)]
        f = f[is_wall[f]]
        if len(f) == 0:
            continue
        # outward: the side of the plane that is outside this part's footprint.
        # The footprint is the set of wall base segments, so an even-odd
        # crossing count settles it for a ring or a multipolygon too, where a
        # centroid test gets the courtyard walls backwards.
        segs = footprint_segments(pos, idx, f)
        part_segs[prt['index']] = segs
        nn = nu[f].copy()
        for i, fi in enumerate(f):
            q = cen[fi][:2] + nn[i][:2] * 0.35
            cross = 0
            for (p0, p1) in segs:
                if (p0[1] > q[1]) != (p1[1] > q[1]):
                    xx = p0[0] + (q[1] - p0[1]) * (p1[0] - p0[0]) / (p1[1] - p0[1] + 1e-12)
                    if xx > q[0]:
                        cross += 1
            if cross % 2 == 1:                      # the test point is inside
                nn[i] *= -1.0
        # group by plane: normal direction (1 deg) and signed offset (0.15 m)
        ang = np.round(np.degrees(np.arctan2(nn[:, 1], nn[:, 0])) / 1.0).astype(np.int64)
        off = (nn * cen[f]).sum(1)
        key = {}
        for i, fi in enumerate(f):
            k = (int(ang[i] % 360), int(round(off[i] / 0.15)))
            key.setdefault(k, []).append(i)
        # merge neighbouring offset bins of the same direction
        merged = {}
        for (aa, oo), v in sorted(key.items()):
            tgt = None
            for (a2, o2) in merged:
                if a2 == aa and abs(o2 - oo) <= 1:
                    tgt = (a2, o2); break
            if tgt is None:
                merged[(aa, oo)] = list(v)
            else:
                merged[tgt].extend(v)
        tg = bytag.get(prt['extras']['osm_id'], {})
        for (aa, oo), v in merged.items():
            v = np.array(v)
            fids = f[v]
            N = nn[v].mean(0); N /= np.linalg.norm(N)
            N[2] = 0.0; N /= max(np.linalg.norm(N), 1e-9)      # force vertical wall
            U = np.array([-N[1], N[0], 0.0])
            V = np.array([0.0, 0.0, 1.0])
            vp = np.unique(idx[fids].ravel())
            P = pos[vp]
            O = P.mean(0)
            uu = (P - O) @ U; vv = (P - O) @ V
            u0, u1 = float(uu.min()), float(uu.max())
            v0, v1 = float(vv.min()), float(vv.max())
            w = u1 - u0; h = v1 - v0
            if w < 0.4 or h < 0.4:
                continue
            org = O + U * u0 + V * v0                      # bottom-left corner
            mid = org + U * (w / 2) + V * (h / 2)
            zf = float(DSM.sample(np.array([mid[0] + N[0] * 2.0], 'f4'),
                                  np.array([mid[1] + N[1] * 2.0], 'f4'))[0])
            wid = len(walls)
            wall_of_face[fids] = wid
            walls.append(dict(
                id=wid, part=prt['index'], part_name=prt['name'],
                osm_id=prt['extras']['osm_id'],
                faces=[int(x) for x in fids],
                origin=[round(float(x), 3) for x in org],
                normal=[round(float(x), 5) for x in N],
                u=[round(float(x), 5) for x in U],
                width_m=round(w, 3), height_m=round(h, 3),
                area_m2=round(float(area[fids].sum()), 2),
                base_z_m=round(float(org[2]), 3),
                roof_z_m=round(float(org[2] + h), 3),
                centre=[round(float(x), 2) for x in (org + U * (w / 2) + V * (h / 2))],
                osm=dict(name=tg.get('name'), building=tg.get('building'),
                         levels=tg.get('building:levels'),
                         material=tg.get('building:material'),
                         colour=tg.get('building:colour'),
                         height=tg.get('height'),
                         amenity=tg.get('amenity'), tourism=tg.get('tourism')),
                buried=False, dsm_front_z_m=round(zf, 2),
                height_source=prt['extras'].get('height_source')))
    # ---- buried walls ----------------------------------------------------- #
    # A wall is buried when another building mass, spanning 60 % of this
    # wall's height, covers the ground 2 m in front of it: no camera can ever see it and
    # it only needs a cheap fallback texture.  Testing against the footprints
    # rather than the lidar DSM matters -- the DSM carries the tree canopy, so
    # testing against the DSM would call walls buried that are merely shaded by
    # a street tree.
    segs = {p: v for p, v in part_segs.items() if any(w['part'] == p for w in walls)}
    roof_z = {p['index']: p['extras']['roof_z_m'] for p in meta['parts']}
    # a raised mass (a skybridge, an upper floor on posts: base_z_m well above
    # the ground) hides only the band it spans, so what counts is the height
    # the two share, not the other mass's roof alone
    base_z = {p['index']: float(p['extras'].get('base_z_m') if p['extras'].get('base_z_m')
                                is not None else -1e9) for p in meta['parts']}

    def shared_h(p, wl):
        return (min(roof_z.get(p, -1e9), wl['roof_z_m'])
                - max(base_z.get(p, -1e9), wl['base_z_m']))

    def inside(part, q):
        cr = 0
        for p0, p1 in segs.get(part, ()):
            if (p0[1] > q[1]) != (p1[1] > q[1]):
                xx = p0[0] + (q[1] - p0[1]) * (p1[0] - p0[0]) / (p1[1] - p0[1] + 1e-12)
                if xx > q[0]:
                    cr += 1
        return cr % 2 == 1

    for wl in walls:
        O = np.array(wl['origin']); U = np.array(wl['u']); N = np.array(wl['normal'])
        q = (O + U * wl['width_m'] / 2 + N * 2.0)[:2]
        wl['buried'] = bool(any(
            p != wl['part'] and shared_h(p, wl) > 0.6 * wl['height_m']
            and inside(p, q) for p in segs))

    against_winding = sum(1 for w in walls if float(np.dot(
        nu[w['faces']].mean(0)[:2], np.array(w['normal'][:2]))) < 0)
    A = np.array([w['area_m2'] for w in walls])
    doc = dict(n_walls=len(walls), wall_faces=int(is_wall.sum()),
               wall_area_m2=round(float(A.sum()), 1),
               roof_faces=int((~is_wall).sum()),
               width_m=dict(median=round(float(np.median([w['width_m'] for w in walls])), 2),
                            max=round(float(max(w['width_m'] for w in walls)), 2)),
               height_m=dict(median=round(float(np.median([w['height_m'] for w in walls])), 2),
                             max=round(float(max(w['height_m'] for w in walls)), 2)),
               faces_unassigned=int((is_wall & (wall_of_face < 0)).sum()),
               buried_walls=int(sum(1 for w in walls if w['buried'])),
               normal_against_winding=int(against_winding),
               buried_area_m2=round(float(sum(w['area_m2'] for w in walls if w['buried'])), 1),
               seconds=round(time.time() - t0, 1),
               walls=walls)
    json.dump(doc, open(FAC + '/walls.json', 'w'), indent=1)
    np.savez_compressed(FAC + '/walls.npz', wall_of_face=wall_of_face,
                        is_wall=is_wall, area=area.astype('f4'),
                        nrm=nu.astype('f4'), cen=cen.astype('f4'))
    print(json.dumps({k: v for k, v in doc.items() if k != 'walls'}, indent=1))
    # per-part summary
    from collections import Counter
    c = Counter(w['part_name'] for w in walls)
    for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:<44} {v:3d} walls")


if __name__ == '__main__':
    main()
