"""areacfg / area.yaml: knobs apply only with OPENCITY_AREA set; with no area the code
defaults hold and none of the per-area stage_knobs apply."""
import functools, json, os, subprocess
import pytest
from conftest import PY, ROOT

import areacfg as A

PROBE = os.path.join(ROOT, 'tests', '_settings_probe.py')

# Code defaults (OPENCITY_AREA unset) of every setting the per-area stage_knobs drive (s22 ground,
# s34/s35/facade_quality, s85 placeholders, nomesh_ground colour, s10a water rise).
NO_AREA_DEFAULTS = {
    's22.RAW_CLIP_MIN': 0.03,
    's22.TILE_CONTRAST': {},
    's22.SAT_CAP': dict(lawn=0.62, paving=0.30, concrete=0.26, asphalt=0.24, gravel=0.50,
                        sand=0.40, wood_deck=0.55, metal=0.30, brick=0.85, water=0.45),
    's22._GROUND_SAT': None,
    's34.import': True,
    's34.MIN_COS': 0.2,
    's35._data_colour_knob': None,
    's85._PLACEHOLDER_HIDE': None,
    'facade_quality.knob': None,
    'nomesh_ground.GROUND_COLOUR_TRANSFER': dict(
        enabled=True, haze_pct=0.5, haze_keep_min=0.5, shadow_sigma_m=6.0, shadow_gain_max=16.0,
        min_support=0.05, luma_pct=[10, 95], skip_classes=['water']),
    # deliberately ON as the code default (no area)
    's10a_layers._water_max_rise': 3.0,
}


def _env(area):
    env = {k: v for k, v in os.environ.items() if k != 'OPENCITY_AREA'}
    if area:
        env['OPENCITY_AREA'] = area
    return env


@functools.lru_cache(maxsize=None)
def probe(area):
    r = subprocess.run([PY, PROBE], env=_env(area), cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---- area.yaml loading ---------------------------------------------------------------
def test_active_area_comes_only_from_env(monkeypatch):
    monkeypatch.delenv('OPENCITY_AREA', raising=False)
    assert A.active() == ''
    monkeypatch.setenv('OPENCITY_AREA', '  tucson_jacome ')
    assert A.active() == 'tucson_jacome'


def test_common_loads_no_area_config_without_env():
    import common as C
    assert C._AREA_CFG is None


def test_area_gets_area_knob_defaults(area_yaml):
    cfg = A.load(area_yaml)
    assert cfg['paths']['scratch_dir']
    for k, v in A.AREA_KNOB_DEFAULTS.items():
        assert cfg['knobs'][k]['value'] == v['value']


def test_knob_without_reason_is_rejected(tmp_path):
    p = tmp_path / 'area.yaml'
    p.write_text("area: {name: x, bbox_wsen: [0, 0, 1, 1]}\npaths: {}\nknobs: {WATER_Z: {value: 1}}\n")
    with pytest.raises(A.AreaConfigError, match='reason'):
        A.load(str(p))


def test_apply_knobs_refuses_unknown_global():
    cfg = {'knobs': {'NOT_A_GLOBAL': {'value': 1, 'reason': 'x'}}}
    with pytest.raises(A.AreaConfigError, match='stage_knobs'):
        A.apply_knobs({}, cfg)


# ---- module-level settings, in fresh subprocesses ------------------------------------
def _cases():
    for key in NO_AREA_DEFAULTS:
        yield pytest.param('', key, id=f'no_area-{key}')


@pytest.mark.parametrize('area,key', list(_cases()))
def test_code_default_holds_without_area(area, key):
    assert probe(area)[key] == NO_AREA_DEFAULTS[key]


def test_area_knobs_reach_the_stage_modules(area_yaml):
    got = probe(area_yaml)
    assert got['s22.RAW_CLIP_MIN'] == 0.07
    assert got['s85._PLACEHOLDER_HIDE'] == []                     # True -> hide every kind
    assert got['s35._data_colour_knob'] == {'enabled': True, 'k_parts': 5, 'r_m': 150.0}
    assert got['facade_quality.knob'] == {'seg_veto': True, 'min_cos': 0.5, 'measure': True}
    gct = got['nomesh_ground.GROUND_COLOUR_TRANSFER']
    assert gct['enabled'] is False and gct['haze_pct'] == 1.5 and gct['shadow_sigma_m'] == 6.0
    assert got['s10a_layers._water_max_rise'] == 2.5
