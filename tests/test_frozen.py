"""The publish freeze: a published dir holding FROZEN.md refuses `opencity publish`
unless OPENCITY_UNFREEZE=1."""
import os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import opencity_paths as OCP  # noqa: E402


def test_guard_refuses_frozen_dir(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENCITY_UNFREEZE', raising=False)
    OCP.refuse_if_frozen(str(tmp_path))                     # no FROZEN.md: no-op
    (tmp_path / 'FROZEN.md').write_text('frozen\n')
    with pytest.raises(SystemExit) as e:
        OCP.refuse_if_frozen(str(tmp_path), 'test')
    assert 'frozen' in str(e.value) and 'OPENCITY_UNFREEZE=1' in str(e.value)
    monkeypatch.setenv('OPENCITY_UNFREEZE', '0')
    with pytest.raises(SystemExit):
        OCP.refuse_if_frozen(str(tmp_path))
    monkeypatch.setenv('OPENCITY_UNFREEZE', '1')
    OCP.refuse_if_frozen(str(tmp_path))                     # explicit unfreeze: no-op
