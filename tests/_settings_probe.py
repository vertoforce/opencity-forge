"""Print, as JSON, the module-level settings the per-area stage_knobs drive, for the
area in OPENCITY_AREA (or none).  Run in a subprocess by test_area_knobs.py so every
area gets a fresh common.py.  An item that raises is recorded as 'ERROR: ...'."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pipeline')); sys.path.insert(0, os.path.join(ROOT, 'lib'))
import opencity_paths  # noqa: E402,F401  puts lib/ and stages/ on sys.path
out = {}


def item(key, fn):
    try:
        v = fn()
        out[key] = sorted(v) if isinstance(v, set) else v
    except Exception as e:                       # report, do not hide
        out[key] = f'ERROR: {type(e).__name__}: {e}'[:200]


def mod(name):
    import importlib
    return importlib.import_module(name)


item('s22.RAW_CLIP_MIN', lambda: mod('s22_composite').RAW_CLIP_MIN)
item('s22.TILE_CONTRAST', lambda: mod('s22_composite').TILE_CONTRAST)
item('s22.SAT_CAP', lambda: mod('s22_composite').SAT_CAP)
item('s22._GROUND_SAT', lambda: mod('s22_composite')._GROUND_SAT)
item('s34.import', lambda: bool(mod('s34_wallproj')))
item('s34.MIN_COS', lambda: mod('s34_wallproj').MIN_COS)
item('s35._data_colour_knob', lambda: mod('s35_facadetex')._data_colour_knob())
item('s85._PLACEHOLDER_HIDE', lambda: mod('s85_object_place')._PLACEHOLDER_HIDE)
item('facade_quality.knob', lambda: mod('facade_quality').knob())
item('nomesh_ground.GROUND_COLOUR_TRANSFER', lambda: mod('nomesh_ground')._stage_knob(
    'GROUND_COLOUR_TRANSFER', mod('nomesh_ground').GROUND_COLOUR_TRANSFER))
item('s10a_layers._water_max_rise', lambda: mod('s10a_layers')._water_max_rise())
print(json.dumps(out, sort_keys=True, default=repr))
