"""City street-tree inventories -> s03b's inventory rows.

A city that publishes a point tree layer goes in CITIES as one entry: an envelope
(W, S, E, N), a fetch function that asks for the area bbox only, and a row mapper to
s03b's row shape (inv_id, x, y, lat, lon, species, common_name, dbh_in, height_ft, ...).
s03b calls build(); outside every envelope it returns None and s03b writes its empty
inventory (lidar crowns + OSM only).  area.yaml sources.trees.city names a city
explicitly, or `none` to skip the lookup.

  San Francisco: DataSF "Street Tree List" (tkzw-k3nq, Socrata SODA API, bbox query,
  no key).  Licence: Open Data Commons PDDL.  mapdbh is the DBH in inches; species is
  "Scientific :: Common".  Rows whose planttype is not Tree (landscaping) are dropped.
"""
import json, os, time, urllib.parse, urllib.request
import numpy as np

UA = {'User-Agent': os.environ.get('OPENCITY_USER_AGENT') or 'opencity-forge/1.0 (python-requests)'}   # = opencity_paths.USER_AGENT


def _get(url, timeout=120, tries=3):
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:                      # retry, then report
            last = e; time.sleep(10 * (k + 1))
    raise RuntimeError(f'{url[:80]}: {last}')


def _sf_fetch(bbox):
    w, s, e, n = bbox
    q = {'$where': f'latitude between {s} and {n} and longitude between {w} and {e}',
         '$select': 'treeid,species,mapdbh,planttype,legalstatus,siteinfo,planteddate,latitude,longitude',
         '$limit': '50000'}
    return _get('https://data.sf.gov/resource/tkzw-k3nq.json?' + urllib.parse.urlencode(q))


def _num(v):
    try:
        v = float(v)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sf_row(r):
    if (r.get('planttype') or 'Tree').strip().lower() != 'tree':
        return None
    sp = (r.get('species') or '').split('::')
    sci = sp[0].strip() or None
    if sci and sci.lower().startswith('tree('):   # "Tree(s) ::" = species not recorded
        sci = None
    yr = None
    if r.get('planteddate'):
        try:
            yr = int(str(r['planteddate'])[:4])
        except ValueError:
            pass
    return dict(inv_id='SF-' + str(r.get('treeid')), lat=_f(r.get('latitude')), lon=_f(r.get('longitude')),
                species=sci, common_name=(sp[1].strip() if len(sp) > 1 and sp[1].strip() else None),
                dbh_in=_num(r.get('mapdbh')), height_ft=None, dept=r.get('legalstatus'), ownership=None,
                leaf_type=None, park=None, condition=None, planted_year=yr, address=r.get('siteinfo'))


CITIES = {
    'san_francisco': dict(env=(-122.52, 37.70, -122.35, 37.84), fetch=_sf_fetch, row=_sf_row,
                          portal='https://data.sfgov.org/d/tkzw-k3nq',
                          licence='DataSF Street Tree List, Open Data Commons PDDL (public domain)'),
}


def pick(bbox, cfg=None):
    c = (((cfg or {}).get('sources') or {}).get('trees') or {}).get('city')
    if c:
        return None if c in ('none', 'osm_only') else c
    w, s, e, n = bbox
    for name, d in CITIES.items():
        x0, y0, x1, y1 = d['env']
        if x0 <= w and y0 <= s and e <= x1 and n <= y1:
            return name
    return None


def build(C, extent, cfg=None):
    """s03b's inventory doc for the city covering C.BBOX_WSEN, or None outside every city."""
    city = pick(C.BBOX_WSEN, cfg)
    if city is None:
        return None
    d = CITIES[city]
    raw = d['fetch'](C.BBOX_WSEN)
    rows = [r for r in (d['row'](x) for x in raw) if r and r['lat'] is not None and r['lon'] is not None]
    x0, x1, y0, y1 = extent
    out = []
    if rows:
        P = C.lla_to_enu(np.array([r['lat'] for r in rows]), np.array([r['lon'] for r in rows]), C.ORIGIN[2])
        for r, p in zip(rows, P):
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                out.append(dict(inv_id=r['inv_id'], x=round(float(p[0]), 2), y=round(float(p[1]), 2),
                                lat=round(r['lat'], 8), lon=round(r['lon'], 8),
                                **{k: r[k] for k in ('dept', 'ownership', 'species', 'common_name', 'dbh_in',
                                                     'height_ft', 'leaf_type', 'park', 'condition',
                                                     'planted_year', 'address')}))
    return dict(what=f'{city} public street-tree inventory points inside the area extent (pipeline/city_trees.py)',
                city=city, fetched=time.strftime('%Y-%m-%dT%H:%M:%S'), bbox_wsen=list(C.BBOX_WSEN),
                extent_enu_m=list(extent), frame=dict(type='ENU metres', origin_lat_lon_alt=list(C.ORIGIN)),
                endpoints={city: dict(portal=d['portal'])}, licence=d['licence'],
                units=dict(dbh_in='inches, diameter at breast height', height_ft='not published'),
                counts=dict(combined_in_bbox=len(raw), combined_in_extent=len(out),
                            with_species=sum(1 for t in out if t['species']),
                            with_dbh=sum(1 for t in out if t['dbh_in']), with_field_height=0,
                            by_dept={k: sum(1 for t in out if t['dept'] == k)
                                     for k in sorted({t['dept'] for t in out if t['dept']})}),
                trees=out)
