"""Print, as JSON, every absolute path a module holds at import time: module-level
str/Path values, nested in dicts/lists/tuples, and function defaults.  Run in a
subprocess by test_paths.py with the OPENCITY_* roots pointed at a temp dir.
usage: _paths_probe.py MODULE [MODULE ...]   (an import that raises -> 'ERROR: ...')"""
import contextlib, importlib, json, os, sys, types
from pathlib import PurePath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FETCH = os.environ.get('OPENCITY_FETCH', '')              # the fetch tools (fetch/); set by test_paths.py
for p in [FETCH, ROOT + '/lib/wallphoto', ROOT + '/pipeline', ROOT + '/stages', ROOT + '/lib']:
    if p:
        sys.path.insert(0, p)
mods, sys.argv = sys.argv[1:], ['probe']


def walk(v, key, out, depth=0):
    if isinstance(v, PurePath):
        v = str(v)
    if isinstance(v, str):
        if v.startswith('/') and len(v) < 400:
            out[key] = v
    elif isinstance(v, dict) and depth < 5:
        for k, x in v.items():
            walk(x, '%s[%r]' % (key, k), out, depth + 1)
    elif isinstance(v, (list, tuple, set, frozenset)) and depth < 5:
        for i, x in enumerate(v):
            walk(x, '%s[%d]' % (key, i), out, depth + 1)


res = {}
for name in mods:
    out = {}
    try:
        with contextlib.redirect_stdout(sys.stderr):        # some modules print on import
            m = importlib.import_module(name)
    except BaseException as e:                    # report, do not hide
        res[name] = 'ERROR: %s: %s' % (type(e).__name__, str(e)[:160]); continue
    for k, v in vars(m).items():
        if k.startswith('__') or isinstance(v, (types.ModuleType, type)):
            continue
        if isinstance(v, types.FunctionType):
            if v.__module__ == m.__name__:
                walk(list(v.__defaults__ or ()) + list((v.__kwdefaults__ or {}).values()), k + '()', out)
            continue
        walk(v, k, out)
    res[name] = out
print(json.dumps(res, sort_keys=True))
