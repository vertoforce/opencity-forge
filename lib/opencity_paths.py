"""Named filesystem roots for the pipeline: the one place paths are configured.

Every root reads an environment variable and falls back to a directory inside this
repo.  Pipeline code builds paths from these names (OCP.DATA + '/_source', ...),
never from literals; tests/test_paths.py fails on a new absolute host-path literal.

    OPENCITY_CODE        this repo (stages/, lib/, pipeline/); default: the parent of this file's lib/ dir
    OPENCITY_DATA        model data: _source/ (fetched), _shared/ (assets every area reads), <area>/ models; default CODE/data
    OPENCITY_PUBLISH     where `opencity publish` copies a model; default DATA
    OPENCITY_WORK        bulky build data: areas/<name>/; default CODE/work
    OPENCITY_COMBINED    default fusion dir (geom/, fusework/) when no area overrides it; default WORK/combined
    OPENCITY_FETCH       the fetch tools: fetch_area.py, fetch_context.py, fuse_context.py; default CODE/fetch
    OPENCITY_VENV        Python venv with the pipeline's dependencies; default CODE/.venv
    OPENCITY_BUILD       tool installs and fetch work dirs: gltf/ (gltf-transform), areas/; default WORK/build
    OPENCITY_TOKEN_FILE  Mapillary API token file (its path only; the pipeline reads it when fetching); default ~/.mapillary_token
    OPENCITY_USER_AGENT  HTTP User-Agent of every fetch (not a path; default a neutral opencity-forge/1.0)

`python3 opencity_paths.py --sh` prints the resolved roots as shell assignments.
"""
import os
import sys


def _env(name, default):
    v = os.environ.get(name, '').strip()
    return v.rstrip('/') if v else default


CODE = _env('OPENCITY_CODE', os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA = _env('OPENCITY_DATA', CODE + '/data')
PUBLISH = _env('OPENCITY_PUBLISH', DATA)
WORK = _env('OPENCITY_WORK', CODE + '/work')
COMBINED = _env('OPENCITY_COMBINED', WORK + '/combined')
FETCH = _env('OPENCITY_FETCH', CODE + '/fetch')
VENV = _env('OPENCITY_VENV', CODE + '/.venv')
BUILD = _env('OPENCITY_BUILD', WORK + '/build')
TOKEN_FILE = _env('OPENCITY_TOKEN_FILE', os.path.expanduser('~/.mapillary_token'))
# HTTP User-Agent for every fetch (OSM, Overpass, ortho, tree inventories); set one that
# names you or your project, as the Overpass and Nominatim usage policies ask.
USER_AGENT = _env('OPENCITY_USER_AGENT', 'opencity-forge/1.0 (python-requests)')

# derived (not separately configurable)
LIB = CODE + '/lib'                                   # shared modules and helper packages
STAGES = CODE + '/stages'                             # the sNN_*.py stage scripts (also imported as modules)
PY = VENV + '/bin/python'
SHARED = DATA + '/_shared'                           # assets every area reads (CC0 tiles, tree and prop scratch)
SOURCE = DATA + '/_source'                            # fetched inputs (panos, lidar, ortho)
GLTF_DIR = BUILD + '/gltf'                            # npm prefix holding gltf-transform
GLTF_TRANSFORM = GLTF_DIR + '/node_modules/.bin/gltf-transform'

ROOTS = dict(CODE=CODE, LIB=LIB, STAGES=STAGES, DATA=DATA, PUBLISH=PUBLISH, WORK=WORK, COMBINED=COMBINED, FETCH=FETCH,
             VENV=VENV, BUILD=BUILD, TOKEN_FILE=TOKEN_FILE, PY=PY, SHARED=SHARED,
             SOURCE=SOURCE, GLTF_DIR=GLTF_DIR, GLTF_TRANSFORM=GLTF_TRANSFORM)



def _code_dirs_on_path():
    """lib/ and stages/ import each other (a stage imports common, a lib module imports a
    stage's constants), so both go on sys.path, and on PYTHONPATH for child processes."""
    here = os.path.dirname(os.path.realpath(__file__))
    dirs = [here, os.path.join(os.path.dirname(here), 'stages')]
    for d in reversed(dirs):
        if d not in sys.path:
            sys.path.insert(0, d)
    pp = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
    os.environ['PYTHONPATH'] = os.pathsep.join(dirs + [p for p in pp if p not in dirs])


_code_dirs_on_path()


def expand(s):
    """'${DATA}/_source' -> DATA + '/_source' (config files such as area.yaml)."""
    if not isinstance(s, str) or '${' not in s:
        return s
    for k, v in ROOTS.items():
        s = s.replace('${%s}' % k, v)
    return s


def reexec_in_venv():
    """Re-run this script under VENV's python when started by another interpreter
    (a `#!/usr/bin/env python3` script run as ./name).  No-op inside the venv."""
    if os.path.realpath(sys.prefix) != os.path.realpath(VENV) and os.path.exists(PY) \
            and not os.environ.get('OPENCITY_NO_REEXEC'):
        os.execv(PY, [PY] + sys.argv)


def refuse_if_frozen(out_dir, what='publish'):
    """A published dir holding FROZEN.md is a frozen release: exit unless
    OPENCITY_UNFREEZE=1.  Called by the publish paths only (s62, opencity publish)."""
    f = os.path.join(out_dir, 'FROZEN.md')
    if os.path.exists(f) and os.environ.get('OPENCITY_UNFREEZE', '').strip() != '1':
        sys.exit('%s refused: %s is frozen (see %s); set OPENCITY_UNFREEZE=1 to publish there'
                 % (what, out_dir, f))


if __name__ == '__main__' and sys.argv[1:] == ['--sh']:
    for k, v in ROOTS.items():
        print("OPENCITY_%s='%s'" % (k, v.replace("'", "'\\''")))
