"""pipeline/area_cli.py: licence profile, time zone lookup, the new-area template,
`opencity build --from` and the panorama restore.  Every path is a pytest tmp dir; no
stage is run."""
import datetime, json, os, sys, time, types, zoneinfo
import pytest
import yaml

import areacfg as A
import area_cli as CLI

SF = '-122.4010,37.7920,-122.4000,37.7928'
TUCSON = '-110.9750,32.2205,-110.9730,32.2220'


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """new-area writes into tmp, not areas/ or the work dir."""
    monkeypatch.setattr(A, 'AREAS', str(tmp_path / 'areas'))
    monkeypatch.setattr(CLI, 'SCRATCH_ROOT', str(tmp_path / 'scratch'))
    monkeypatch.setattr(CLI, 'PUBLISH_ROOT', str(tmp_path / 'publish'))
    return tmp_path


def make_area(name, bbox, *extra):
    CLI.main(['new-area', name, f'--bbox={bbox}', '--timezone', 'America/Los_Angeles', *extra])
    return yaml.safe_load(open(A.area_file(name)))


# ---- licence profile ------------------------------------------------------------------
def test_new_area_ortho_is_naip(sandbox):
    for lic in ('private', 'public'):
        cfg = make_area('t_' + lic, SF, '--license', lic)
        assert cfg['license'] == lic
        assert cfg['sources']['ortho']['primary'] == 'naip'
        assert not cfg['sources']['ortho'].get('restricted')


def _restricted_area(sandbox):
    """A built area whose ortho came from a local source declared restricted."""
    sd = sandbox / 'scratch' / 'local_built'
    (sd / 'ortho').mkdir(parents=True); (sd / 'publish').mkdir()
    (sd / 'ortho' / 'source.json').write_text(json.dumps({'source': 'city_ortho', 'credit': 'test credit'}))
    (sd / 'publish' / 'model.glb').write_bytes(b'glTF')
    return {'area': {'name': 'local_built'}, 'license': 'private', 'build_mode': 'poses',
            'sources': {'ortho': {'primary': 'city_ortho', 'restricted': True}},
            'paths': {'scratch_dir': str(sd), 'W': str(sd / 'work'), 'CTX': str(sd / 'ctx'),
                      'OUT': str(sd / 'publish'), 'publish_name': 'opencity_local_built'}}


def test_publish_public_refuses_restricted_source(sandbox, monkeypatch):
    cfg = _restricted_area(sandbox)
    monkeypatch.setattr(CLI, 'cfg_of', lambda n: cfg)
    with pytest.raises(SystemExit, match='refusing to publish as public'):
        CLI.main(['publish', 'local_built', '--license', 'public'])
    assert not os.path.exists(sandbox / 'publish')


def test_publish_private_allows_restricted_source(sandbox, monkeypatch, capsys):
    cfg = _restricted_area(sandbox)
    monkeypatch.setattr(CLI, 'cfg_of', lambda n: cfg)
    CLI.main(['publish', 'local_built', '--dry-run'])
    assert 'would copy' in capsys.readouterr().out


def test_fetch_public_falls_back_to_naip(sandbox, monkeypatch):
    sd = sandbox / 'scratch' / 'local_fetch'
    cfg = {'area': {'name': 'local_fetch', 'bbox_wsen': [-122.401, 37.792, -122.400, 37.7928], 'crs_epsg': 32610},
           'license': 'private', 'build_mode': 'poses', 'paths': {'scratch_dir': str(sd)},
           'sources': {'ortho': {'primary': 'city_ortho', 'restricted': True}, 'mapillary': {}}}
    calls = []
    monkeypatch.setattr(CLI, 'cfg_of', lambda n: cfg)
    monkeypatch.setattr(CLI, 'run', lambda cmd, **k: calls.append(cmd))
    CLI.main(['fetch', 'local_fetch', '--skip-mapillary', '--skip-context', '--license', 'public'])
    (orth,) = [c for c in calls if c[1].endswith('fetch_ortho.py')]
    assert orth[orth.index('--source') + 1] == 'naip'
    calls.clear()
    CLI.main(['fetch', 'local_fetch', '--skip-mapillary', '--skip-context'])    # private: local kept, not fetched
    assert not [c for c in calls if c[1].endswith('fetch_ortho.py')]


# ---- time zone (offline) ----------------------------------------------------------------
ZONES = [('tucson', -110.974, 32.221, 'America/Phoenix'),
         ('boston', -71.070, 42.354, 'America/New_York'),
         ('san_francisco', -122.402, 37.793, 'America/Los_Angeles')]


@pytest.mark.parametrize('lookup', ['timezonefinder', 'longitude_fallback'])
@pytest.mark.parametrize('city,lon,lat,zone', ZONES, ids=[z[0] for z in ZONES])
def test_timezone_lookup(city, lon, lat, zone, lookup, monkeypatch):
    if lookup == 'longitude_fallback':
        monkeypatch.setitem(sys.modules, 'timezonefinder', None)     # import -> ImportError
    assert CLI.guess_us_timezone(lon, lat) == zone


def test_tucson_zone_keeps_no_daylight_time():
    z = zoneinfo.ZoneInfo(CLI.guess_us_timezone(-110.974, 32.221))
    jan, jul = (datetime.datetime(2026, m, 1, 12, tzinfo=z).utcoffset() for m in (1, 7))
    assert jan == jul == datetime.timedelta(hours=-7)


def test_new_area_writes_looked_up_timezone(sandbox):
    CLI.main(['new-area', 't_tz', f'--bbox={TUCSON}'])
    assert yaml.safe_load(open(A.area_file('t_tz')))['area']['timezone'] == 'America/Phoenix'


# ---- new-area template --------------------------------------------------------------------
def test_new_area_template_poses_and_facade_knobs(sandbox):
    cfg = make_area('t_tpl', TUCSON)
    assert cfg['build_mode'] == 'poses'
    sk = cfg['stage_knobs']
    assert sk['FACADE_PHOTO_QUALITY']['value'] == {'seg_veto': True, 'min_cos': 0.5, 'measure': True}
    assert sk['FACADE_DATA_COLOUR']['value'] == {'enabled': True, 'k_parts': 5, 'r_m': 150.0}
    assert sk['OBJECT_PLACEHOLDER_HIDE']['value'] is True
    assert sk['WATER_MAX_RISE_M']['value'] == 3.0               # default on
    assert all(v.get('reason') for v in sk.values())
    loaded = A.load(A.area_file('t_tpl'))           # the written file passes the loader
    assert loaded['paths']['scratch_dir'].startswith(str(sandbox / 'scratch'))


# ---- opencity build --from -------------------------------------------------------------------
def test_build_from_clears_markers_since_stage_start(tmp_path, monkeypatch):
    W = tmp_path / 'work'; md = W / 'markers'; md.mkdir(parents=True)
    cfg = {'area': {'name': 't_from'}, 'paths': {'scratch_dir': str(tmp_path), 'W': str(W)}}
    t0 = time.time() - 1000
    stamp = lambda p, t: (p.write_text('x'), os.utime(p, (t, t)))
    stamp(md / '_start_s10a_3dfier.py', t0)
    stamp(md / '_start_s02_osm.py', t0 - 500)
    old = ['s02_osm.done', 's03_trees.done']
    new = ['3dfier_layers.done', '3dfier_run.done', 's19_delight.done']
    for f in old:
        stamp(md / f, t0 - 100)
    for i, f in enumerate(new):
        stamp(md / f, t0 + 10 * i)                       # the first one is exactly at t0
    ran = []
    monkeypatch.setattr(CLI, 'ensure_work', lambda c: None)
    monkeypatch.setattr(CLI, 'cfg_of', lambda n: cfg)
    monkeypatch.setattr(CLI, 'make_web', lambda c: None)
    monkeypatch.setattr(CLI, '_sandbox_cmd', lambda cmd, c: cmd)
    monkeypatch.setattr(CLI.subprocess, 'run', lambda cmd, **k: (ran.append(os.path.basename(cmd[1])),
                                                                 types.SimpleNamespace(returncode=0))[1])
    CLI.build_poses(cfg, ['stages'], from_stage='s10a_3dfier.py')
    left = set(os.listdir(md))
    assert left.isdisjoint(new), 'markers written since the stage start must be cleared'
    assert set(old) <= left, 'markers older than the stage start must stay'
    assert {'_start_s02_osm.py', '_start_s10a_3dfier.py'} <= left
    stages = [os.path.basename(s) for s, _ in CLI.POSES_STAGES]
    assert ran == stages[stages.index('s10a_3dfier.py'):]


def test_build_from_unknown_stage_dies(tmp_path, monkeypatch):
    cfg = {'area': {'name': 't_from'}, 'paths': {'scratch_dir': str(tmp_path), 'W': str(tmp_path)}}
    monkeypatch.setattr(CLI, 'ensure_work', lambda c: None)
    monkeypatch.setattr(CLI, 'cfg_of', lambda n: cfg)
    with pytest.raises(SystemExit, match='not a poses-mode stage'):
        CLI.build_poses(cfg, ['stages'], from_stage='s99_nope.py')


# ---- deleted panoramas are restored by id before the stages ------------------------
def test_restore_panos_fetches_only_missing_posed_ids(tmp_path, monkeypatch):
    W = tmp_path / 'work'; P = tmp_path / 'src'; (P / 'images_pano').mkdir(parents=True); W.mkdir()
    (W / 'pano_poses.json').write_text('{"poses": {"111": {}, "222": {}}}')
    (P / 'images_pano' / '111.jpg').write_text('x')
    cfg = {'paths': {'W': str(W), 'PANO': str(P)}, 'area': {'bbox_wsen': [0, 0, 1, 1]}}
    calls = []
    def fake_run(cmd, **k):
        calls.append(cmd); (P / 'images_pano' / '222.jpg').write_text('x')
    monkeypatch.setattr(CLI, 'run', fake_run); monkeypatch.setattr(CLI, 'bbox_arg', lambda c: '0,0,1,1')
    CLI.restore_panos(cfg)
    assert len(calls) == 1 and '--ids-file' in calls[0]
    assert (P / 'metadata' / 'restore_ids.txt').read_text().split() == ['222']
    CLI.restore_panos(cfg)                      # nothing missing: no fetch
    assert len(calls) == 1
