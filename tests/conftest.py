"""Shared fixtures for the pipeline regression tests (python -m pytest tests).

Rules every test here keeps: CPU only, no network (sockets are blocked), no reads of
area build data, and nothing written outside pytest's tmp dirs."""
import os, socket, sys
# Child processes the tests start use the venv pytest runs in, unless OPENCITY_VENV names one.
os.environ.setdefault('OPENCITY_VENV', sys.prefix)
try:
    import opencity_paths as OCP                # host roots, env-overridable
except ImportError:                            # imported with lib/ off sys.path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
    import opencity_paths as OCP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, 'pipeline')
FIX = os.path.join(ROOT, 'tests', 'fixtures')
PY = OCP.PY
if PIPE not in sys.path:                      # lib/ and stages/ come from opencity_paths
    sys.path.insert(0, PIPE)
# The test process runs with no area set: common.py must load without one.
os.environ.pop('OPENCITY_AREA', None)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise OSError('network disabled in tests')
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)


@pytest.fixture
def area_yaml(tmp_path):
    """The fixture area written to tmp, scratch_dir = tmp/scratch.  Returns its path."""
    sd = tmp_path / 'scratch'
    sd.mkdir()
    text = open(os.path.join(FIX, 'area', 'area.yaml')).read().replace('@SCRATCH@', str(sd))
    p = tmp_path / 'area.yaml'
    p.write_text(text)
    return str(p)


@pytest.fixture
def built_area(area_yaml, monkeypatch):
    """Make the in-process common module look like a process with an area set (OPENCITY_AREA set,
    C._areacfg / C._AREA_CFG present), restored after the test."""
    import areacfg as A
    import common as C
    cfg = A.load(area_yaml)
    monkeypatch.setenv('OPENCITY_AREA', area_yaml)
    monkeypatch.setattr(C, '_areacfg', A, raising=False)
    monkeypatch.setattr(C, '_AREA_CFG', cfg)
    return cfg
