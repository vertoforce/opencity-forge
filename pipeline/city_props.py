"""City asset inventories -> street-furniture rows for s80.

An area's props come from OSM, lidar poles and Mapillary detections (s80).  A city
that publishes its street and park assets as open data usually lists far more
benches, tables and poles than OSM maps.  This module fetches those layers for the
area bbox only and writes <props>/city_props.json in s80's OSM row shape (source
'city').  s80 fuse merges them: a city row within MATCH_M of an OSM row of the same
class becomes evidence on it, otherwise it is a new prop.  An area outside every
listed city gets no file and stays OSM + lidar + photo.

  usage: OPENCITY_AREA=<name> python pipeline/city_props.py [--city NAME]

The city comes from area.yaml sources.city_props, else from the bbox (inside a
CITIES envelope).  One query per layer, bbox only, cached raw under
<props>/city_raw/.  Each entry carries its layers' open-data licence.

A CITIES entry (none ship by default):

    'example_city': dict(
        env=(W, S, E, N),                       # envelope the city's layers cover
        base='https://<host>/arcgis/rest/services',   # ArcGIS REST services root
        layers=[('Park_Bench', 'bench', 'GLOBALID', 'park benches'),
                ('Street_Lights', lamp_rule, 'ASSET_ID', 'street-light poles')],
        bench_length_ft_field='LENGTH',         # optional
        licence='<the city open-data terms>'),

A layer is (FeatureServer service name, s80 class or fn(properties) -> (class,
height_m), id field, note).
"""
import os, sys, json, time, argparse, urllib.parse, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
if not os.environ.get('OPENCITY_AREA', '').strip():
    sys.exit('city_props: OPENCITY_AREA must name an area')
import numpy as np                                    # noqa: E402
import common as C                                    # noqa: E402

AP = C._areacfg.module_paths(C)
if AP is None:
    sys.exit('city_props: the area config has no paths.scratch_dir')
OUT = AP['PROPS']


def lamp_rule(p, height_field='HEIGHT', lit_field='HasStreetlight'):
    """An example layer rule: a pole with a street light is a lamp post, a bare
    utility pole is a mast (s80 draws masts as flagpole-type poles).  Height in feet."""
    h = p.get(height_field)
    h = float(h) * 0.3048 if h not in (None, '', 0) else None
    lit = str(p.get(lit_field, '')).strip().upper() in ('Y', 'YES', '1', 'TRUE')
    return ('lamp_post' if lit else 'mast'), h


# city -> dict(env=(W, S, E, N), base=ArcGIS REST root, layers=[...], licence=...)
CITIES = {}


def pick_city(cfg):
    c = ((cfg.get('sources') or {}).get('city_props'))
    if c:
        return None if c in ('none', 'osm_only') else c
    w, s, e, n = C.BBOX_WSEN
    for name, d in CITIES.items():
        env = d['env']
        if env[0] <= w and env[1] <= s and e <= env[2] and n <= env[3]:
            return name
    return None


def fetch(base, service, timeout=120):
    q = dict(where='1=1', geometry=','.join(str(v) for v in C.BBOX_WSEN),
             geometryType='esriGeometryEnvelope', inSR='4326',
             spatialRel='esriSpatialRelIntersects', outFields='*',
             outSR='4326', returnGeometry='true', f='geojson')
    url = f"{base}/{urllib.parse.quote(service)}/FeatureServer/0/query?" + urllib.parse.urlencode(q)
    last = None
    for k in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            if 'error' in d:
                raise RuntimeError(str(d['error'])[:200])
            return d
        except Exception as e:                       # retry, then report
            last = e; time.sleep(10 * (k + 1))
    raise RuntimeError(f'{service}: {last}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', default=None)
    a = ap.parse_args()
    city = a.city or pick_city(C._AREA_CFG)
    os.makedirs(OUT, exist_ok=True)
    fp = OUT + '/city_props.json'
    if city is None:
        if os.path.exists(fp):
            os.remove(fp)
        print('city_props: no city inventory covers this area; s80 stays OSM + lidar + photo')
        return
    if city not in CITIES:
        sys.exit(f'city_props: no CITIES entry {city!r}')
    d0 = CITIES[city]; layers = d0['layers']
    w, s, e, n = C.BBOX_WSEN
    os.makedirs(OUT + '/city_raw', exist_ok=True)
    rows, counts = [], {}
    for service, rule, idf, note in layers:
        d = fetch(d0['base'], service)
        json.dump(d, open(f"{OUT}/city_raw/{service.replace('(', '').replace(')', '')}.json", 'w'))
        k = 0
        for f in d.get('features') or []:
            g = f.get('geometry') or {}
            if g.get('type') != 'Point':
                continue
            lon, lat = g['coordinates'][:2]
            if not (w <= lon <= e and s <= lat <= n):
                continue
            p = f.get('properties') or {}
            cls, h = rule(p) if callable(rule) else (rule, None)
            P = C.lla_to_enu(np.asarray([lat], 'f8'), np.asarray([lon], 'f8'), np.asarray([C.ORIGIN[2]], 'f8'))
            r = dict(cls=cls, source='city', city=city, city_layer=service, city_id=str(p.get(idf)),
                     x=float(P[0, 0]), y=float(P[0, 1]), confidence=1.0, height_m=h, name=None)
            lf = d0.get('bench_length_ft_field')            # optional: a bench length field, in feet
            if cls == 'bench' and lf and p.get(lf):
                try:
                    r['length_m'] = round(float(p[lf]) * 0.3048, 2)
                except (TypeError, ValueError):
                    pass
            rows.append(r); k += 1
        counts[service] = k
        print(f'  {service}: {k} points ({note})', flush=True)
    json.dump(dict(city=city, licence=d0.get('licence'), counts=counts, rows=rows), open(fp, 'w'), indent=1)
    print(f'city_props: {len(rows)} rows from {city} -> {fp}')


if __name__ == '__main__':
    main()
