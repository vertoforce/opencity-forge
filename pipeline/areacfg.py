"""Per-area config: areas/<name>/area.yaml, read by common.py and the opencity CLI.

The active area is the env var OPENCITY_AREA (a name under areas/, or a path to
an area.yaml).  When OPENCITY_AREA is unset common.py does not import this module
at all, and every stage sees the placeholder values written in common.py.

An area.yaml has these sections (see areas/tucson_jacome/area.yaml):

  area:      name, title, bbox_wsen, origin {lat, lon, alt_m}, crs_epsg, timezone
  paths:     work_dir, scratch_dir, publish_name, and optional common.py path
             overrides (W, CMB, G, FW, CTX, SRCM, PANO, OUT, KEEP)
  sources:   mapillary {token_file, max_panos, pano_spacing_m}, lidar, osm, ortho, trees,
             city_props
  registration: {yaw_deg, t: [x, y, z]}  (lidar -> photo, from fuse_context)
  knobs:     NAME: {value, class, reason}.  NAME must be a common.py global.
             class is data | hand | derivable | derived.  A dict value is merged
             key by key into the common.py default; a list becomes a tuple or an
             array when the default is one.
  stage_knobs: knobs that live in stage modules (WATER_Z, SHORE_BAND_M,
             FACADE_PHOTO_QUALITY, ...); the stage that owns a knob reads it from here.
  derive:    true: `opencity derive` writes <scratch_dir>/derived.yaml and those values
             become the area's knob defaults.  false: only the hand values in knobs: and
             the common.py defaults apply.

Frame values (paths, ORIGIN, REG_*, BBOX_WSEN) are applied before common.py
defines lla_to_enu, whose default argument captures ORIGIN; knobs are applied
at the end of common.py.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                  # the repo root (areas/, lib/)
AREAS = os.path.join(ROOT, 'areas')

FRAME_PATHS = ('W', 'CMB', 'G', 'FW', 'CTX', 'SRCM', 'PANO', 'OUT', 'KEEP')
FRAME_KEYS = FRAME_PATHS + ('ORIGIN', 'REG_YAW_DEG', 'REG_T', 'BBOX_WSEN')
KNOB_CLASSES = ('data', 'hand', 'derivable', 'derived')


class AreaConfigError(Exception):
    pass


# Knob defaults every area with paths.scratch_dir gets on top of the common.py values.
# Same shape as knobs:.
AREA_KNOB_DEFAULTS = {
    'FACADE_GLASS_L_MAX': dict(value=0.11, **{'class': 'hand'}, reason=(
        "s35's absolute dark-glass cut: a curtain wall carrying sky and cloud reflections measures "
        "L 0.09-0.10 (hue 201-206, band 0.47-1.24); the cut must sit above that band, or the wall "
        "ships the reflection instead of glass")),
    'FACADE_GLASS_SIBLINGS': dict(value=False, **{'class': 'hand'}, reason=(
        "one photographed glass bay must not turn its whole building, photographed front "
        "included, into synthetic curtain wall")),
}


def area_file(name):
    if name.endswith(('.yaml', '.yml')) or os.sep in name:
        return os.path.abspath(name)
    return os.path.join(AREAS, name, 'area.yaml')


def _yaml():
    try:
        import yaml
    except ImportError:
        raise AreaConfigError('PyYAML missing; use the pipeline venv python ($OPENCITY_VENV/bin/python)')
    return yaml


def derived_path(cfg):
    """Where `opencity derive` writes an area's derived.yaml: its scratch dir (under
    OPENCITY_WORK), so a build never writes into the repo.  A derived.yaml next to
    area.yaml is read when the scratch dir has none."""
    sd = (cfg.get('paths') or {}).get('scratch_dir')
    beside = os.path.join(cfg['_dir'], 'derived.yaml')
    if not sd:
        return beside
    p = os.path.join(sd, 'derived.yaml')
    return beside if not os.path.exists(p) and os.path.exists(beside) else p


def load(name):
    p = area_file(name)
    if not os.path.exists(p):
        raise AreaConfigError(f'no area config {p}')
    with open(p) as f:
        cfg = _yaml().safe_load(f) or {}
    cfg['_file'] = p
    cfg['_dir'] = os.path.dirname(p)
    # ${DATA}/... style roots (opencity_paths.expand) in paths: and the token file's name,
    # so a shipped area.yaml names no host path
    try:
        import opencity_paths as _OCP
    except ImportError:
        import sys; sys.path.insert(0, os.path.join(ROOT, 'lib')); import opencity_paths as _OCP
    for k, v in list((cfg.get('paths') or {}).items()):
        cfg['paths'][k] = _OCP.expand(v)
    mly = (cfg.get('sources') or {}).get('mapillary')
    if isinstance(mly, dict) and 'token_file' in mly:
        mly['token_file'] = _OCP.expand(mly['token_file'])
    for sec in ('area', 'paths'):
        if sec not in cfg:
            raise AreaConfigError(f'{p}: missing section {sec}')
    for k in ('name', 'bbox_wsen'):
        if k not in cfg['area']:
            raise AreaConfigError(f'{p}: missing area.{k}')
    # an area's derived knobs (opencity derive) are defaults only: an
    # explicit entry in knobs: wins.
    dp = derived_path(cfg)
    if cfg.get('derive') and os.path.exists(dp):
        with open(dp) as f:
            der = _yaml().safe_load(f) or {}
        for k, v in (der.get('knobs') or {}).items():
            cfg.setdefault('knobs', {}).setdefault(k, v)
        for k, v in (der.get('stage_knobs') or {}).items():
            cfg.setdefault('stage_knobs', {}).setdefault(k, v)
    # generic area defaults, for an area with paths.scratch_dir; an explicit knobs:
    # entry or a derived value wins.
    if (cfg.get('paths') or {}).get('scratch_dir'):
        for k, v in AREA_KNOB_DEFAULTS.items():
            cfg.setdefault('knobs', {}).setdefault(k, dict(v))
    for k, v in (cfg.get('knobs') or {}).items():
        if not isinstance(v, dict) or 'value' not in v:
            raise AreaConfigError(f'{p}: knobs.{k} needs value: and reason:')
        if not v.get('reason'):
            raise AreaConfigError(f'{p}: knobs.{k} has no reason')
        if v.get('class', 'hand') not in KNOB_CLASSES:
            raise AreaConfigError(f'{p}: knobs.{k} class {v.get("class")!r}')
    return cfg


def _coerce(default, new):
    import numpy as np
    if isinstance(default, np.ndarray):
        return np.array(new, dtype=default.dtype)
    if isinstance(default, dict) and isinstance(new, dict):
        out = dict(default)
        for k, v in new.items():
            out[k] = _coerce(default[k], v) if k in default else v
        return out
    if isinstance(default, tuple) and isinstance(new, (list, tuple)):
        if len(default) == len(new):
            return tuple(_coerce(d, n) for d, n in zip(default, new))
        return tuple(new)
    if isinstance(default, float) and isinstance(new, int) and not isinstance(new, bool):
        return float(new)
    return new


def frame_values(cfg):
    """common.py frame overrides: paths, ORIGIN, REG_*, BBOX_WSEN."""
    import numpy as np
    a, p, out = cfg['area'], cfg['paths'], {}
    for k in FRAME_PATHS:
        if k in p:
            out[k] = str(p[k])
    if 'W' not in out and 'work_dir' in p:
        out['W'] = str(p['work_dir'])
    o = a.get('origin')
    if o:
        out['ORIGIN'] = (float(o['lat']), float(o['lon']), float(o['alt_m']))
    out['BBOX_WSEN'] = tuple(float(v) for v in a['bbox_wsen'])
    r = cfg.get('registration') or {}
    if 'yaw_deg' in r:
        out['REG_YAW_DEG'] = float(r['yaw_deg'])
    if 't' in r:
        out['REG_T'] = np.array([float(v) for v in r['t']])
    return out


def apply_frame(g, cfg):
    vals = frame_values(cfg)
    g.update(vals)
    return sorted(vals)


def apply_knobs(g, cfg):
    done = []
    for k, spec in (cfg.get('knobs') or {}).items():
        if k in FRAME_KEYS:
            raise AreaConfigError(f'knobs.{k}: frame values go in area:/paths:/registration:')
        if k not in g:
            raise AreaConfigError(f'knobs.{k} is not a common.py global; put it in stage_knobs:')
        g[k] = _coerce(g[k], spec['value'])
        done.append(k)
    return done


def active():
    """Name (or path) of the active area, '' when none."""
    return os.environ.get('OPENCITY_AREA', '').strip()


# ---- helpers the shared modules call for an area ---------
# An area build is OPENCITY_AREA set AND its area.yaml has paths.scratch_dir.  With no
# area, every caller keeps its module-level placeholder.
def build_area_cfg(C):
    """The loaded area config if this process builds an area, else None."""
    if not os.environ.get('OPENCITY_AREA', '').strip():
        return None
    cfg = getattr(C, '_AREA_CFG', None) or {}
    return cfg if (cfg.get('paths') or {}).get('scratch_dir') else None


def enu_ext(C, pad=0.0):
    """The area bbox (common.BBOX_WSEN) in ENU metres, whole metres outward, +pad.
    The same rectangle s10a_3dfier.EXT uses for an area build."""
    import math
    w, s, e, n = C.BBOX_WSEN
    P = C.lla_to_enu([s, s, n, n], [w, e, e, w], [C.ORIGIN[2]] * 4)
    return (float(math.floor(P[:, 0].min())) - pad, float(math.floor(P[:, 1].min())) - pad,
            float(math.ceil(P[:, 0].max())) + pad, float(math.ceil(P[:, 1].max())) + pad)


def module_paths(C):
    """Per-area replacements for the module-level path defaults in texlib, wpcommon,
    alcommon, polish_lib and objcommon.  None with no area."""
    cfg = build_area_cfg(C)
    if cfg is None:
        return None
    sd = cfg['paths']['scratch_dir']
    return dict(W=C.W, SCR=sd + '/texture', PSCR=sd + '/pano_pose', SRC=sd + '/src',
                IMD=C.PANO + '/images_pano', META=C.PANO + '/metadata/images.jsonl',
                DET=sd + '/detections', FW=C.FW,
                PROPS=sd + '/props', OBJ=sd + '/objects', TREES=sd + '/trees')


def lidar_licence(C, default='USGS 3DEP, public domain'):
    """'USGS 3DEP <dataset>, public domain' for an area, read from its CTX/lidar_stats.json
    (fetch_context.py); no area or a missing file keeps the default."""
    import json
    if build_area_cfg(C) is None:
        return default
    p = os.path.join(C.CTX, 'lidar_stats.json')
    ds = json.load(open(p)).get('dataset') if os.path.exists(p) else None
    return 'USGS 3DEP %s, public domain' % ds if ds else default
