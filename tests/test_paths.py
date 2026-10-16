"""Host-path hygiene.

Every filesystem root the pipeline uses lives in opencity_paths.py as an OPENCITY_* env
root.  These tests (1) fail on an absolute host-path literal in the files the pipeline
runs, and (2) copy those files to a temp dir, point every root there, and check that
what the modules resolve at import stays inside it."""
import ast, glob, json, os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import opencity_paths as OCP  # noqa: E402

FETCH = OCP.FETCH
HOST = re.compile(r'(?<![\w.$}~-])/(root|mnt|scratch|home|srv|opt)/')
# The defaults module, and the fetch tools' own `os.environ.get("OPENCITY_X") or "<default>"`
# lines (fetch/*.py run standalone, without importing opencity_paths).
ALLOW_FILES = {os.path.join(ROOT, 'lib', 'opencity_paths.py')}
ALLOW_LINE = re.compile(r'os\.environ\.get\(["\']OPENCITY_[A-Z_]+["\']\)')
# Imported for the relocation probe: everything in scope except scripts that do work
# at import (fetch_panos/fetch_ortho parse argv).
NOT_IMPORTABLE = re.compile(r'^pipeline/fetch_(panos|ortho)\.py$')
# The dirs code imports from (lib/ and stages/ are on sys.path via opencity_paths).
CODE_DIRS = ('lib', 'stages', 'pipeline')
PACKAGES = ('align', 'objects')          # imported as `from objects import ...`
# These exit at import unless OPENCITY_AREA names an area (by design).
AREA_REQUIRED = {'city_props', 'nomesh_facades', 'nomesh_swaps'}


def _resolve(mod, here):
    parts = mod.split('.')
    for base in (os.path.dirname(here), ROOT + '/lib', ROOT + '/stages', FETCH, ROOT + '/pipeline',
                 ROOT + '/lib/wallphoto'):
        for p in (os.path.join(base, *parts) + '.py', os.path.join(base, *parts, '__init__.py')):
            if os.path.isfile(p):
                return p
    return None


def in_scope():
    """The files the pipeline runs: every module under lib/, stages/, pipeline/ and
    fetch/, and every local module they import."""
    seeds = sorted(glob.glob(ROOT + '/lib/**/*.py', recursive=True) + glob.glob(ROOT + '/stages/*.py')
                   + glob.glob(ROOT + '/pipeline/*.py')
                   + glob.glob(FETCH + '/*.py'))
    seen, todo = set(), [s for s in seeds if os.path.isfile(s)]
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        if not p.endswith('.py'):
            continue
        for n in ast.walk(ast.parse(open(p).read())):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                [n.module] + [n.module + '.' + a.name for a in n.names] \
                if isinstance(n, ast.ImportFrom) and n.module and not n.level else []
            todo += [r for r in (_resolve(m, p) for m in mods) if r and r not in seen]
    return sorted(seen)


def test_scope_is_the_pipeline():
    files = in_scope()
    rel = {os.path.relpath(f, ROOT) for f in files if f.startswith(ROOT)}
    assert {'lib/common.py', 'pipeline/cli.py', 'pipeline/area_cli.py', 'stages/s22_composite.py',
            'lib/texlib.py', 'lib/wallphoto/wpcommon.py', 'lib/opencity_paths.py'} <= rel
    assert {os.path.join(FETCH, f) for f in ('fetch_area.py', 'fetch_context.py', 'fuse_context.py')} <= set(files)
    assert all(r.split('/')[0] in ('lib', 'stages', 'pipeline', 'fetch') for r in rel), rel


def test_no_host_path_literals():
    """An absolute host-path literal belongs in opencity_paths.py as a root."""
    bad = []
    for f in in_scope():
        if f in ALLOW_FILES:
            continue
        for i, line in enumerate(open(f, errors='replace'), 1):
            if HOST.search(line) and not ALLOW_LINE.search(line):
                bad.append('%s:%d: %s' % (os.path.relpath(f, ROOT), i, line.strip()[:120]))
    assert not bad, 'host paths outside opencity_paths.py:\n' + '\n'.join(bad)


def test_defaults_are_the_in_repo_layout(monkeypatch):
    """Unset, every root falls back to the layout inside the repo."""
    for k in list(os.environ):
        if k.startswith('OPENCITY_') and k != 'OPENCITY_AREA':
            monkeypatch.delenv(k)
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'lib', 'opencity_paths.py'), '--sh'],
                         capture_output=True, text=True, check=True).stdout
    got = dict(l.split('=', 1) for l in out.split())
    assert got['OPENCITY_CODE'] == "'%s'" % os.path.realpath(ROOT)
    assert got['OPENCITY_PUBLISH'] == got['OPENCITY_DATA']
    assert got['OPENCITY_PY'] == got['OPENCITY_VENV'][:-1] + "/bin/python'"


def test_fetch_tools_share_the_defaults():
    """The fetch tools inline their defaults; they must match opencity_paths.py
    (DATA = <repo>/data, BUILD = <repo>/work/build, VENV = <repo>/.venv, token in the home dir)."""
    src = {f: open(os.path.join(FETCH, f)).read() for f in ('fetch_context.py', 'fuse_context.py',
                                                             'fetch_area.py')}
    want = {'OPENCITY_DATA': 'Path(__file__).resolve().parents[1] / "data"',
            'OPENCITY_BUILD': 'Path(__file__).resolve().parents[1] / "work" / "build"',
            'OPENCITY_TOKEN_FILE': 'Path.home() / ".mapillary_token"',
            'OPENCITY_VENV': 'HERE.parent / ".venv"'}
    seen = set()
    for f, text in src.items():
        for k in set(re.findall(r'os\.environ\.get\("(OPENCITY_[A-Z_]+)"\) or ', text)):
            seen.add(k)
            if k in want:
                assert 'os.environ.get("%s") or %s)' % (k, want[k]) in text, (f, k)
    gp = open(os.path.join(ROOT, 'lib', 'opencity_paths.py')).read()
    assert "DATA = _env('OPENCITY_DATA', CODE + '/data')" in gp
    assert "WORK = _env('OPENCITY_WORK', CODE + '/work')" in gp and "BUILD = _env('OPENCITY_BUILD', WORK + '/build')" in gp
    assert "expanduser('~/.mapillary_token')" in gp
    assert "VENV = _env('OPENCITY_VENV', CODE + '/.venv')" in gp
    assert seen == {'OPENCITY_DATA', 'OPENCITY_BUILD', 'OPENCITY_TOKEN_FILE', 'OPENCITY_USER_AGENT'}


def test_roots_relocate_to_env(tmp_path):
    """Copy the in-scope files to tmp, point every OPENCITY_* root under tmp, import every
    in-scope module there: no resolved path may point back at a host root."""
    code, fetch = tmp_path / 'code', tmp_path / 'fetch'
    mods = []
    for f in in_scope():
        if f.startswith(FETCH + '/'):
            dst, rel = fetch / os.path.relpath(f, FETCH), None
            mods.append(os.path.basename(f)[:-3])
        else:
            rel = os.path.relpath(f, ROOT)
            dst = code / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        if rel and rel.endswith('.py') and not NOT_IMPORTABLE.search(rel):
            sub = rel[:-3].split('/', 1)[1] if rel.split('/')[0] in CODE_DIRS else rel[:-3]
            d, _, b = sub.rpartition('/')
            mods.append(sub.replace('/', '.').replace('.__init__', '') if d in PACKAGES else b)
    shutil.copytree(os.path.join(ROOT, 'areas'), code / 'areas')
    for f in glob.glob(ROOT + '/pipeline/*.json'):
        shutil.copy2(f, code / 'pipeline')
    (code / 'tests').mkdir()
    shutil.copy2(os.path.join(ROOT, 'tests', '_paths_probe.py'), code / 'tests' / '_paths_probe.py')
    env = {k: v for k, v in os.environ.items() if not k.startswith('OPENCITY_')}
    env.update({'OPENCITY_%s' % k: str(tmp_path / k.lower()) for k in
                ('DATA', 'PUBLISH', 'WORK', 'COMBINED', 'VENV', 'BUILD', 'TOKEN_FILE')})
    env.update(OPENCITY_CODE=str(code), OPENCITY_FETCH=str(fetch))
    r = subprocess.run([OCP.PY if os.path.exists(OCP.PY) else sys.executable,
                        str(code / 'tests' / '_paths_probe.py')] + sorted(set(mods)),
                       capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=280)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout)
    errors = {m: v for m, v in res.items() if isinstance(v, str) and m not in AREA_REQUIRED}
    assert not errors, errors
    leaks = ['%s.%s = %s' % (m, k, p) for m, d in res.items() if isinstance(d, dict)
             for k, p in d.items() if HOST.match(p)]
    assert not leaks, 'paths not under the OPENCITY_* roots:\n' + '\n'.join(leaks)
    t = str(tmp_path)
    assert res['common']['W'] == str(code) and res['common']['OUT'] == t + '/data/model'
    assert res['texlib']['SCR'] == t + '/data/_shared/texture'
    assert res['area_cli']['PY'] == t + '/venv/bin/python'
    assert res['area_cli']['SCRATCH_ROOT'] == t + '/work/areas'
    assert res['area_cli']['PUBLISH_ROOT'] == t + '/publish'
    assert res['fetch_area']['_TOKEN_FILE'] == t + '/token_file'
    assert res['fuse_context']['GLTF_DIR'] == t + '/build/gltf'
    assert len([m for m in res if isinstance(res[m], dict)]) >= 45
