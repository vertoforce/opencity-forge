import sys, os, json, time, urllib.request, urllib.parse, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
from common import BBOX_WSEN, W, mark, note, done, OCP
import osm_rules as R

def fetch():
    w,s,e,n = BBOX_WSEN
    QL = f"""[out:json][timeout:180];
(
  node({s},{w},{n},{e});
  way({s},{w},{n},{e});
  relation({s},{w},{n},{e})["type"="multipolygon"];
);
out body geom;"""
    MIRRORS=["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
    t0=time.time(); raw=None; last=None
    for attempt in range(6):
        url=MIRRORS[attempt % len(MIRRORS)]
        req = urllib.request.Request(url,
            data=("data="+urllib.parse.quote(QL)).encode(),
            headers={"User-Agent": OCP.USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                raw = r.read()
            print("ok from",url); break
        except Exception as ex:
            last=ex; print("fail",url,ex,flush=True); time.sleep(8)
    if raw is None: raise SystemExit("overpass unreachable: %s"%last)
    d = json.loads(raw)
    open(W+'/osm/osm_raw.json','wb').write(raw)
    els = d['elements']
    c = collections.Counter()
    for el in els:
        t = el.get('tags') or {}
        if t.get('natural')=='tree': c['tree']+=1
        if t.get('building'): c['building']+=1
        if t.get('highway') in ('footway','path','pedestrian','steps','cycleway'): c['footway']+=1
        if t.get('man_made')=='pier': c['pier']+=1
        if t.get('bridge'): c['bridge']+=1
        if t.get('natural')=='water' or t.get('waterway'): c['water']+=1
        if t.get('landuse')=='grass' or t.get('leisure')=='park': c['grass_park']+=1
    print("elements",len(els),dict(c),"in %.1fs"%(time.time()-t0))
    note('osm_counts', dict(c)); note('osm_elements', len(els))
    mark('osm', {"elements":len(els),"counts":dict(c),"bbox":list(BBOX_WSEN)})


def classify_all():
    """Apply the OSM decision table (osm_rules) to the whole fetch.

    Idempotent and cheap: no geometry, only tags.  Writes the per-element
    decision so s10a_3dfier and the vector stages all read one table instead of
    each carrying its own tag guesses.
    """
    els = json.load(open(W+'/osm/osm_raw.json'))['elements']
    out = {}
    per_layer = collections.Counter()
    per_kind = collections.Counter()
    per_surface = collections.Counter()
    widths = collections.Counter()
    unclassified = collections.Counter()
    for el in els:
        if el['type'] == 'node':
            continue
        t = el.get('tags') or {}
        if not t:
            continue
        nodes = el.get('nodes') or []
        closed = bool(nodes and nodes[0] == nodes[-1] and len(el.get('geometry') or []) >= 4)
        if el['type'] == 'relation':
            closed = (t.get('type') == 'multipolygon')
        d = R.classify(t, closed=closed)
        if d is None:
            for k in ('highway','landuse','natural','leisure','man_made','amenity',
                      'waterway','building','barrier','railway','place'):
                if t.get(k):
                    unclassified[f'{k}={t[k]}'] += 1
            continue
        key = f"{el['type']}/{el['id']}"
        out[key] = {k: v for k, v in d.items() if k != 'rule'}
        per_layer[d['layer']] += 1
        per_kind[d['kind']] += 1
        if d.get('surface'):
            per_surface[d['surface']] += 1
        if d.get('width_m'):
            widths[f"{d['kind']}:{d['width_m']:.1f}m"] += 1
    json.dump(dict(decisions=out,
                   layers=dict(per_layer), kinds=dict(per_kind),
                   surfaces=dict(per_surface), widths=dict(widths),
                   unclassified_tags=dict(unclassified.most_common(40)),
                   table='osm_rules.py, ported by hand from louis-e/arnis '
                         'src/element_processing/ with per-rule citations'),
              open(W+'/osm/osm_layers.json','w'), indent=1)
    summary = dict(classified=len(out), by_layer=dict(per_layer),
                   by_surface=dict(per_surface),
                   path_widths={k: v for k, v in widths.items()
                                if k.startswith('highway')})
    note('osm_rules', summary)
    print("osm_rules:", json.dumps(summary, indent=1))
    return summary


if __name__ == '__main__':
    if done('osm'):
        print("osm already fetched")
    else:
        fetch()
    classify_all()
