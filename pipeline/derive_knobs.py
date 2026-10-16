#!/usr/bin/env python3
"""Derive the 'derivable' knobs from an area's own data.

    derive_knobs.py AREA

Reads the area's fuse_context work dir (FW: loaded.npz, lidar_reg.npy,
surfaces.npz), its OSM (CTX/osm_raw.json), its Mapillary metadata
(PANO/metadata/images.jsonl) and the shared CC0 class tiles.  It runs on the CPU:
numpy over files already on disk.

  derive: true  -> <scratch_dir>/derived.yaml.  areacfg.load() merges it as knob
                   defaults; an explicit knobs: entry wins.
  derive: false -> derived_report.yaml next to derived.yaml, report only.
                   The hand values stay; nothing reads the report.

Each entry records value, the hand value it would replace, the recipe and its
inputs.  A knob whose data is missing is written with value: null and why, so a
stage that needs it fails loudly instead of falling back to a placeholder default.
"""
import ast, datetime, json, os, re, sys
try:
    import opencity_paths as OCP                # host roots, env-overridable
except ImportError:                            # imported with lib/ off sys.path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
    import opencity_paths as OCP
if __name__ == '__main__':                       # run as ./derive_knobs.py under /usr/bin/env python3
    OCP.reexec_in_venv()
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import areacfg as A                                         # noqa: E402

CC0 = OCP.SHARED + '/texture/cc0'
# recipe constants (not per area; each is a physical or code-derived number)
SHORE_K = 3.0          # shore band = 3 x median lidar point spacing around the water edge
LANE_W_M = 3.3         # typical US urban travel lane (10-11 ft)
SIDEWALK_M = 1.8       # common US minimum sidewalk width (6 ft), when OSM has no sidewalk:width
LANES_DEFAULT = dict(motorway=4, trunk=4, primary=4, secondary=2, tertiary=2,
                     residential=2, unclassified=2, living_street=1, service=1)
DETAIL_BAND_M = (0.25, 1.0)
DETAIL_REF_AMP = 0.20  # s22 CLASS_DETAIL amp, tuned on the grain-free Concrete034


def _hand_s22(name):
    """Read a module constant's literal from s22_composite.py without importing it."""
    src = open(os.path.join(OCP.STAGES, 's22_composite.py')).read()
    m = re.search(rf'^{name} = (dict\(.*?\))\s*$', src, re.M | re.S)
    node = ast.parse(m.group(1), mode='eval').body
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def water_level(fw):
    d = np.load(f'{fw}/loaded.npz'); cls = d['lidar_cls']
    L = np.load(f'{fw}/lidar_reg.npy')
    wz = L[cls == 9][:, 2]
    if len(wz) < 200:
        return None, len(wz), None
    return float(np.median(wz)), int(len(wz)), float(np.percentile(wz, 90) - np.percentile(wz, 10))


def point_spacing(fw):
    from scipy.spatial import cKDTree
    d = np.load(f'{fw}/loaded.npz'); cls = d['lidar_cls']
    L = np.load(f'{fw}/lidar_reg.npy')
    out = {}
    for c in (2, 9):
        P = L[cls == c][:, :2]
        if len(P) < 1000:
            continue
        rng = np.random.default_rng(0)
        q = P[rng.choice(len(P), min(len(P), 50000), replace=False)]
        dist, _ = cKDTree(P).query(q, k=2)
        out[c] = float(np.median(dist[:, 1]))
    return out


def path_half_width(fw):
    """p90 half-width of OSM/lidar paved paths from the surfaces raster (distance transform)."""
    from scipy import ndimage
    S = np.load(f'{fw}/surfaces.npz')
    c = float(S['grid'][2])
    m = S['m_path'].astype(bool)
    if m.sum() < 20:
        return None, c
    dt = ndimage.distance_transform_edt(m) * c
    # ridge cells: local maxima of the distance transform = the path centre line
    ridge = m & (dt >= ndimage.maximum_filter(dt, size=3) - 1e-6)
    return float(np.percentile(dt[ridge], 90)), c


def street_edge(osm_paths, bbox):
    ps = [p for p in osm_paths if p and os.path.exists(p)]
    if not ps:
        return None, 0
    p = ps[0]
    els = json.load(open(p)).get('elements', [])
    nodes = {e['id']: (e['lon'], e['lat']) for e in els if e['type'] == 'node' and 'lon' in e}
    w, s, e_, n = bbox
    vals = []
    for e in els:
        t = e.get('tags') or {}
        hw = t.get('highway')
        if e['type'] != 'way' or hw not in LANES_DEFAULT:
            continue
        pts = ([(g['lon'], g['lat']) for g in e['geometry']] if e.get('geometry')
               else [nodes[i] for i in e.get('nodes', []) if i in nodes])
        if not any(w <= x <= e_ and s <= y <= n for x, y in pts):
            continue
        try:
            width = float(str(t.get('width', '')).split()[0])
        except (ValueError, IndexError):
            try:
                width = int(str(t.get('lanes', '')).split(';')[0]) * LANE_W_M
            except (ValueError, IndexError):
                width = LANES_DEFAULT[hw] * LANE_W_M
        sw = SIDEWALK_M
        for k in ('sidewalk:width', 'sidewalk:both:width', 'sidewalk:right:width'):
            try:
                sw = float(str(t[k]).split()[0]); break
            except (KeyError, ValueError, IndexError):
                pass
        if hw != 'service':
            vals.append(width / 2 + sw)
    if not vals:
        return None, 0
    return float(np.median(vals)), len(vals)


def utc_offset(pano, tz):
    p = f'{pano}/metadata/images.jsonl'
    if not os.path.exists(p):
        return None, 0, None
    from zoneinfo import ZoneInfo
    z = ZoneInfo(tz)
    offs = []
    for line in open(p):
        r = json.loads(line)
        if r.get('is_pano') is False or not r.get('captured_at'):
            continue
        t = datetime.datetime.fromtimestamp(r['captured_at'] / 1000, datetime.timezone.utc)
        offs.append(int(t.astimezone(z).utcoffset().total_seconds()))
    if not offs:
        return None, 0, None
    vals, cnt = np.unique(offs, return_counts=True)
    return int(vals[np.argmax(cnt)]), len(offs), {int(v): int(c) for v, c in zip(vals, cnt)}


def band_sd(logl, ppm, band=DETAIL_BAND_M):
    from scipy import ndimage
    lo = ndimage.gaussian_filter(logl, 0.5 * band[0] * ppm, mode='wrap')
    hi = ndimage.gaussian_filter(logl, 0.5 * band[1] * ppm, mode='wrap')
    return float((lo - hi).std())


def detail_amp():
    """amp = max(0, REF_AMP - own_sd / field_sd): the tile's own 0.25-1 m grain counts
    against the synthetic field, so a grain-free tile keeps the tuned 0.20 and a
    tile that already carries that much grain gets 0."""
    from PIL import Image
    tile_m = _hand_s22('TILE_M')
    cd = _hand_s22('CLASS_DETAIL')
    # the field itself, at unit amplitude, measured in the same band.  Built by
    # s22's own class_detail() when importable, else skipped (value null).
    try:
        import importlib
        s22 = importlib.import_module('s22_composite')
        field = lambda h, w, tm: np.log(s22.class_detail(h, w, tm, cfg=dict(amp=1.0)).astype('f8'))  # noqa: E731
    except Exception as e:                                   # noqa: BLE001
        return {'error': f's22_composite import failed: {e}'}
    idx = json.load(open(f'{CC0}/index.json'))['materials']
    out = {}
    for mat in ('concrete', 'asphalt', 'paving', 'gravel', 'lawn', 'sand'):
        if mat not in idx or mat not in tile_m:
            continue
        col = idx[mat]['files']['color']
        im = np.asarray(Image.open(col).convert('RGB').resize((1024, 1024), Image.BILINEAR), 'f8') / 255.0
        lin = np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)
        lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
        ppm = 1024 / float(tile_m[mat])
        own = band_sd(np.log(np.clip(lum, 1e-4, None)), ppm)
        fsd = band_sd(field(1024, 1024, float(tile_m[mat])), ppm)
        amp = max(0.0, DETAIL_REF_AMP - own / fsd) if fsd > 0 else None
        out[mat] = dict(value=None if amp is None else round(amp, 3), own_sd=round(own, 4),
                        field_sd_at_amp1=round(fsd, 4), tile=os.path.basename(col), tile_m=tile_m[mat])
    out['_ref'] = dict(ref_amp=DETAIL_REF_AMP, band_m=list(DETAIL_BAND_M), class_detail=cd)
    return out


def main():
    name = sys.argv[1]
    cfg = A.load(name)
    fr = A.frame_values(cfg)
    p = cfg['paths']
    fw, ctx = p.get('FW', fr.get('FW')), p.get('CTX')
    pano = p.get('PANO')
    tz = cfg['area'].get('timezone') or 'UTC'
    knobs, stage, rejected = {}, {}, {}
    hand = (cfg.get('knobs') or {})
    hstage = (cfg.get('stage_knobs') or {})

    def hv(d, k):
        return (d.get(k) or {}).get('value')

    have_fw = fw and os.path.exists(f'{fw}/loaded.npz')
    if have_fw:
        wl, n9, spread = water_level(fw)
        stage['WATER_Z'] = dict(value=None if wl is None else round(wl, 4), class_='derived',
                                hand=hv(hstage, 'WATER_Z'),
                                reason=(f'median z of {n9:,} lidar class-9 returns (p10-p90 spread {spread:.3f} m)'
                                        if wl is not None else f'only {n9} class-9 returns: no water body; '
                                        'every WATER_* knob does not apply'))
        sp = point_spacing(fw)
        if sp:
            spacing = float(np.median(list(sp.values())))
            stage['SHORE_BAND_M'] = dict(value=round(SHORE_K * spacing, 2), class_='derived',
                                         hand=hv(hstage, 'SHORE_BAND_M'),
                                         reason=f'{SHORE_K:g} x median lidar NN spacing '
                                                f'({", ".join(f"class {c}: {v:.3f} m" for c, v in sp.items())})'
                                                + ('' if stage['WATER_Z']['value'] is not None else '; no water, unused'))
        hw, cell = path_half_width(fw)
        stage['HARDSCAPE_SNAP_M'] = dict(value=None if hw is None else round(hw, 2), class_='derived',
                                         hand=hv(hstage, 'HARDSCAPE_SNAP_M'),
                                         reason=f'p90 paved-path half-width, distance transform of surfaces m_path '
                                                f'on its {cell:g} m grid (coarse: +-{cell/2:g} m)')
    else:
        stage['_missing'] = f'no fuse_context work dir at {fw}; run opencity build --steps fuse first'
    se, nse = street_edge([f"{fr.get('W', '')}/osm/osm_raw.json", f'{ctx}/osm_raw.json'],
                          cfg['area']['bbox_wsen'])
    stage['STREET_EDGE_M'] = dict(value=None if se is None else round(se, 2), class_='derived',
                                  hand=hv(hstage, 'STREET_EDGE_M'),
                                  reason=f'median over {nse} OSM street ways of width/2 (or lanes x {LANE_W_M} m / 2) '
                                         f'+ sidewalk ({SIDEWALK_M} m unless tagged)')
    off, npano, hist = utc_offset(pano, tz) if pano else (None, 0, None)
    stage['UTC_OFFSET_S'] = dict(value=off, class_='derived', hand=hv(hstage, 'UTC_OFFSET_S'),
                                 reason=f'modal {tz} UTC offset over {npano} panorama capture times {hist}')
    da = detail_amp()
    if 'error' not in da:
        # Reported, not applied: on the shared grained CC0 tiles any synthetic detail field draws
        # blotches, so GROUND_DETAIL_AMP stays 0.0; the recipe value is recorded under rejected:.
        rejected['GROUND_DETAIL_AMP'] = dict(
            value={m: v['value'] for m, v in da.items() if not m.startswith('_') and v['value'] is not None},
            class_='derived', hand=hv(hand, 'GROUND_DETAIL_AMP'),
            reason=f'per CC0 tile: max(0, {DETAIL_REF_AMP} - own/field 0.25-1 m log-luma sd)',
            detail={m: v for m, v in da.items() if not m.startswith('_')})
    else:
        knobs['_GROUND_DETAIL_AMP_error'] = da['error']
    # derived.yaml uses areacfg's knob schema: value, class, reason
    for d in (knobs, stage, rejected):
        for k, v in d.items():
            if isinstance(v, dict) and 'class_' in v:
                v['class'] = v.pop('class_')
    derive = bool(cfg.get('derive'))
    outp = A.derived_path(cfg) if derive else os.path.join(cfg['_dir'], 'derived_report.yaml')
    if derive and (cfg.get('paths') or {}).get('scratch_dir'):
        outp = os.path.join(cfg['paths']['scratch_dir'], 'derived.yaml')   # never next to area.yaml
        os.makedirs(os.path.dirname(outp), exist_ok=True)
    y = A._yaml()
    with open(outp, 'w') as f:
        f.write(f'# written by pipeline/derive_knobs.py for {name}.\n'
                + ('# derive: true -> these are this area\'s knob defaults (areacfg.load merges knobs:).\n'
                   if derive else '# REPORT ONLY (derive: false): the hand values in area.yaml stay authoritative.\n'))
        y.safe_dump(dict(knobs=knobs, stage_knobs=stage, rejected=rejected), f, sort_keys=False, default_flow_style=None, width=110)
    print(f'wrote {outp}')
    for sec in (knobs, stage, rejected):
        for k, v in sec.items():
            if isinstance(v, dict):
                print(f'  {k:18s} derived={v.get("value")!s:40.40s} hand={v.get("hand")!s:.50s}')
            else:
                print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
