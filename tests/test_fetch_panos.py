"""pipeline/fetch_panos.py (disk-safe poses-mode fetch): 3 m thinning before any
download, and a hard error when a selected id is not on disk.  fetch_area.py is
never run: subprocess.run is replaced, so no Mapillary call and no token read."""
import json, math, os, runpy, sys
import pytest
from conftest import PIPE

SCRIPT = os.path.join(PIPE, 'fetch_panos.py')
BBOX = (-122.401, 37.792, -122.399, 37.794)
LAT0, LON0 = (BBOX[1] + BBOX[3]) / 2, (BBOX[0] + BBOX[2]) / 2
MX, MY = 111320.0 * math.cos(math.radians(LAT0)), 110540.0


def rec(pid, seq, t, x_m, y_m=0.0, **kw):
    d = dict(id=pid, sequence=seq, captured_at=t, camera_type='spherical', computed_rotation=[0, 0, 0],
             computed_geometry={'coordinates': [LON0 + x_m / MX, LAT0 + y_m / MY]})
    d.update(kw)
    return d


def run_fetch(tmp_path, monkeypatch, recs, drop=(), args=()):
    out = tmp_path / 'src'
    (out / 'metadata').mkdir(parents=True)
    with open(out / 'metadata' / 'images_all.jsonl', 'w') as f:   # metadata step already done
        f.writelines(json.dumps(r) + '\n' for r in recs)
    calls = []

    def fake_run(cmd, check=False, **k):
        calls.append(cmd)
        assert cmd[1].endswith('fetch_area.py') and '--ids-file' in cmd, cmd
        ids = open(cmd[cmd.index('--ids-file') + 1]).read().split()
        (out / 'images_pano').mkdir(exist_ok=True)
        for i in ids:
            if i not in drop:
                (out / 'images_pano' / f'{i}.jpg').write_bytes(b'')

    monkeypatch.setattr('subprocess.run', fake_run)
    monkeypatch.setattr(sys, 'argv', [SCRIPT, '--bbox=' + ','.join(map(str, BBOX)), '--out', str(out), *args])
    runpy.run_path(SCRIPT, run_name='__main__')
    return out, calls


def test_thinning_keeps_panos_3m_apart(tmp_path, monkeypatch):
    recs = [rec(f'a{i}', 's1', 100 + i, float(i)) for i in range(11)]          # 1 m apart
    out, calls = run_fetch(tmp_path, monkeypatch, recs)
    kept = (out / 'metadata' / 'selected_ids.txt').read_text().split()
    assert kept == ['a0', 'a3', 'a6', 'a9']
    sel = json.load(open(out / 'metadata' / 'selection.json'))
    assert sel['thinned'] == 4 and sel['spacing_m'] == 3.0 and sel['panos_with_metadata'] == 11
    assert len(calls) == 1                         # one download call, selection only


def test_thinning_skips_unposed_and_flat_images(tmp_path, monkeypatch):
    recs = [rec('ok', 's1', 1, 0.0), rec('flat', 's1', 2, 10.0, camera_type='perspective'),
            rec('norot', 's1', 3, 20.0, computed_rotation=None), rec('nogeo', 's1', 4, 30.0, computed_geometry=None)]
    out, _ = run_fetch(tmp_path, monkeypatch, recs)
    assert (out / 'metadata' / 'selected_ids.txt').read_text().split() == ['ok']


def test_newest_sequence_wins_a_shared_spot(tmp_path, monkeypatch):
    recs = [rec('old', 's_old', 10, 0.0), rec('new', 's_new', 99, 1.0)]
    out, _ = run_fetch(tmp_path, monkeypatch, recs)
    assert (out / 'metadata' / 'selected_ids.txt').read_text().split() == ['new']


def test_max_panos_caps_evenly(tmp_path, monkeypatch):
    recs = [rec(f'a{i}', 's1', i, 3.0 * i) for i in range(10)]
    out, _ = run_fetch(tmp_path, monkeypatch, recs, args=('--max-panos', '5'))
    assert (out / 'metadata' / 'selected_ids.txt').read_text().split() == ['a0', 'a2', 'a4', 'a6', 'a8']


def test_missing_download_is_an_error(tmp_path, monkeypatch):
    recs = [rec(f'a{i}', 's1', i, 3.0 * i) for i in range(4)]
    with pytest.raises(SystemExit, match='1 of 4 selected panoramas not downloaded'):
        run_fetch(tmp_path, monkeypatch, recs, drop={'a2'})
    assert not (tmp_path / 'src' / 'metadata' / 'selection.json').exists()
