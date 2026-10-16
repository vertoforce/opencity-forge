"""s10a: a dry box (no measured water) must not crash the 3dfier layer collection, and
the no-data cut is the area's lowest ground - 10 m for a configured area, and a fixed -50.0 when no area is set."""
import numpy as np

import common as C
import s10a_3dfier as S
import s10a_layers as SL


def _surfaces(d, water, dtm):
    ny, nx = water.shape
    np.savez(d / 'surfaces.npz', grid=np.array([-5.0, -5.0, 1.0, nx, ny]), m_water=water, dtm=dtm)


def test_empty_water_mask_does_not_crash(tmp_path, monkeypatch):
    _surfaces(tmp_path, np.zeros((10, 10), bool), np.zeros((10, 10)))
    monkeypatch.setattr(C, 'FW', str(tmp_path))
    monkeypatch.setattr(C, 'osm_elements', lambda: [])
    poly, cells, area = S._water_mask_polygon()
    assert cells == 0 and area == 0.0
    out, tally, info = S.collect_layers()
    assert out['water'] == [] and info['water_mask_cells'] == 0
    S.partition(out)                           # must accept an empty water layer


def test_water_mask_present_adds_measured_water(tmp_path, monkeypatch):
    w = np.zeros((10, 10), bool); w[2:6, 2:6] = True
    _surfaces(tmp_path, w, np.zeros((10, 10)))
    monkeypatch.setattr(C, 'FW', str(tmp_path))
    monkeypatch.setattr(C, 'osm_elements', lambda: [])
    out, tally, info = S.collect_layers()
    assert info['water_mask_cells'] == 16
    assert [d['kind'] for d in out['water']] == ['water_measured']


def _verts():
    # ground at 5 m, a 3dfier no-data vertex at -99.99, a vertex just above the fixed -50.0 cut at -49.99, and
    # two low vertices either side of an area build's cut (lowest DTM 2.0 -> cut -8.0)
    V = np.array([[0, 0, 5.0], [1, 0, 5.0], [0, 1, 5.0], [1, 1, -99.99], [2, 0, -49.99],
                  [2, 1, -7.0], [2, 2, -9.0]], float)
    return V, np.array([[0, 1, 2]])


def test_nodata_cut_fixed_without_area():
    assert not hasattr(C, '_areacfg')
    V, F = _verts()
    V2, n = SL.repair_nodata(V.copy(), F)
    assert n == 1 and V2[3, 2] == 5.0          # only below -50.0
    assert V2[4, 2] == -49.99 and V2[6, 2] == -9.0


def test_nodata_cut_area_lowest_ground_minus_10(built_area, tmp_path, monkeypatch):
    dtm = np.full((10, 10), 7.0); dtm[3, 3] = 2.0; dtm[0, 0] = np.nan
    _surfaces(tmp_path, np.zeros((10, 10), bool), dtm)
    monkeypatch.setattr(C, 'FW', str(tmp_path))
    V, F = _verts()
    V2, n = SL.repair_nodata(V.copy(), F)
    assert n == 3                              # -99.99, -49.99, -9.0 (below 2.0 - 10)
    assert V2[2 + 2, 2] == 5.0 and V2[5, 2] == -7.0 and (V2[[3, 4, 6], 2] > -8.0).all()


# ---- WATER_MAX_RISE_M drops 3dfier WaterWall faces that climb a building ----
def _water_fixture():
    """A 10 x 10 m lake at z 0 (two flat faces), a 1 m quay wall on its south edge (kept),
    a 12 m wall on its east edge up a building side (dropped), and a pond at z 5 with a
    0.5 m rim, 40 m away (kept: its local level is the pond, not the larger water body's median)."""
    V = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],          # lake 0-3
                  [0, 0, 1.0], [10, 0, 1.0],                               # quay top 4-5
                  [10, 0, 12.0], [10, 10, 12.0],                           # building wall top 6-7
                  [40, 0, 5], [42, 0, 5], [42, 2, 5],                      # pond 8-10
                  [40, 0, 5.5], [42, 0, 5.5]], float)                      # pond rim top 11-12
    F = np.array([[0, 1, 2], [0, 2, 3],                  # lake
                  [0, 1, 5], [0, 5, 4],                  # quay (rise 1 m)
                  [1, 2, 7], [1, 7, 6],                  # wall (rise 12 m)
                  [8, 9, 10],                            # pond
                  [8, 9, 12], [8, 12, 11]])              # pond rim (rise 0.5 m)
    return V, F


def test_water_rise_rule_drops_only_the_building_wall():
    V, F = _water_fixture()
    drop, rise = SL._water_rise_drop(V, F.astype('u4'), 3.0, 0.0)
    assert drop.tolist() == [False, False, False, False, True, True, False, False, False]
    assert np.allclose(rise[[2, 3, 7, 8]], [1.0, 1.0, 0.5, 0.5]) and np.allclose(rise[[4, 5]], 12.0)
    off, _ = SL._water_rise_drop(V, F.astype('u4'), None, 0.0)
    assert not off.any()


def test_water_layer_records_dropped_faces(tmp_path, monkeypatch):
    (tmp_path / 'layers').mkdir()
    monkeypatch.setattr(C, 'W', str(tmp_path))
    monkeypatch.setattr(C, 'note', lambda *a, **k: None)
    V, F = _water_fixture()
    r = SL._water(V, F, None)
    assert r['triangles'] == 7 and r['water_rise_faces_dropped'] == 2
    d = np.load(tmp_path / 'layers' / 'water.npz')
    assert len(d['idx']) == 7 and len(d['pos']) == 11 and d['pos'][:, 2].max() == 5.5
    doc = __import__('json').load(open(tmp_path / 'layers' / 'water.json'))
    wr = doc['parts'][0]['extras']['water_rise']
    assert wr['faces_dropped'] == 2 and wr['area_dropped_m2'] == 120.0 and wr['water_max_rise_m'] == 3.0
    assert doc['water_rise']['faces_dropped'] == 2


def test_water_max_rise_knob(built_area, monkeypatch):
    assert SL._water_max_rise() == 2.5                       # fixture area.yaml value
    monkeypatch.setitem(built_area['stage_knobs'], 'WATER_MAX_RISE_M', {'value': None})
    assert SL._water_max_rise() is None                      # null = off
    monkeypatch.setattr(C, '_AREA_CFG', None)
    assert SL._water_max_rise() == SL.WATER_MAX_RISE_M == 3.0  # code default with no area set


# ---- the water plane and the clip box come from the area, not from fixed numbers ----
def test_water_plane_is_the_area_knob(built_area, monkeypatch):
    import s2x_lib as L
    assert L.water_z() is None                               # fixture area: no WATER_Z
    monkeypatch.setitem(built_area['stage_knobs'], 'WATER_Z', {'value': 1.25})
    assert L.water_z() == 1.25
    monkeypatch.setattr(C, '_AREA_CFG', None)
    assert L.water_z() is None                               # no area: no water plane


def test_clip_box_is_the_area_bbox_plus_margin(built_area, monkeypatch):
    import shapely
    import s2x_lib as L
    monkeypatch.setattr(C, 'BBOX_WSEN', (-0.001, -0.001, 0.001, 0.001))   # about the 0/0 origin
    x0, y0, x1, y1 = L.clip_box().bounds
    assert x0 == -112.0 - L.CLIP_PAD_M and x1 == 112.0 + L.CLIP_PAD_M
    far = shapely.box(1000.0, 1000.0, 1010.0, 1010.0)
    assert L.clip(far) is None                               # outside the box: dropped
    monkeypatch.setattr(C, 'BBOX_WSEN', (0.0, 0.0, 0.0, 0.0))
    assert L.clip_box() is None and L.clip(far).area == 100.0  # no area: geometry kept whole
