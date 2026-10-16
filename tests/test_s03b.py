"""s03b: an optional city tree source that fails (network, schema) falls back to the
empty inventory (lidar crowns + OSM) instead of failing the build."""
import json
import pytest

import common as C
import city_trees
import s03b_city_trees as S03B


@pytest.fixture
def s03b_sandbox(built_area, tmp_path, monkeypatch):
    inv = tmp_path / 'inventory'
    notes, marks = {}, []
    monkeypatch.setattr(S03B, 'INV', str(inv))
    monkeypatch.setattr(C, 'note', lambda k, v: notes.__setitem__(k, v))
    monkeypatch.setattr(C, 'mark', lambda n, info=None: marks.append(n))
    monkeypatch.setattr(C, 'BBOX_WSEN', tuple(built_area['area']['bbox_wsen']))
    return inv, notes, marks


def test_city_tree_source_failure_falls_back(s03b_sandbox, monkeypatch):
    inv, notes, marks = s03b_sandbox
    assert city_trees.pick(C.BBOX_WSEN, C._AREA_CFG) == 'san_francisco'

    def boom(bbox):
        raise RuntimeError('DataSF unreachable (test)')
    monkeypatch.setitem(city_trees.CITIES['san_francisco'], 'fetch', boom)
    S03B.main()                                          # must not raise
    doc = json.load(open(inv / 'city_trees.json'))
    assert doc['trees'] == [] and doc['counts']['combined_in_extent'] == 0
    assert 'DataSF unreachable' in notes['tree_inventory_city_error']
    assert marks == ['city_trees']


def test_city_tree_source_bad_rows_fall_back(s03b_sandbox, monkeypatch):
    inv, notes, marks = s03b_sandbox
    monkeypatch.setitem(city_trees.CITIES['san_francisco'], 'fetch', lambda bbox: [{'junk': 1}, None])
    S03B.main()
    assert json.load(open(inv / 'city_trees.json'))['trees'] == []
    assert marks == ['city_trees']
