#!/usr/bin/env python3
"""Download the shared CC0 material tiles every area reads (ambientCG, Poly Haven).

    fetch/fetch_cc0.py            fetch what is missing into $OPENCITY_SHARED/texture/cc0

assets/cc0_index.json lists each material with its download URL and the sha256 of
its archive or colour map.  The tiles land in <cc0>/<material>/ and <cc0>/index.json is written
with their local paths, which is what the stages read.  `opencity fetch` runs this
when index.json is missing.  A sha256 that differs stops the fetch.
"""
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import opencity_paths as OCP  # noqa: E402

INDEX = os.path.join(ROOT, 'assets', 'cc0_index.json')
DEST = OCP.SHARED + '/texture/cc0'


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': OCP.USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    src = json.load(open(INDEX))
    out = {'materials': {}}
    for name, m in src['materials'].items():
        d = os.path.join(DEST, name)
        files = {role: os.path.join(d, fn) for role, fn in m['files'].items()}
        if not all(os.path.exists(p) for p in files.values()):
            print('cc0: %s <- %s' % (name, m['download_url']), flush=True)
            blob = get(m['download_url'])
            os.makedirs(d, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                # the recorded sha256 is of the archive or of its colour map
                shas = {hashlib.sha256(blob).hexdigest(),
                        hashlib.sha256(z.read(m['files']['color'])).hexdigest()}
                if m.get('sha256') and m['sha256'] not in shas:
                    sys.exit('cc0: %s sha256 %s, expected %s' % (name, sorted(shas), m['sha256']))
                for fn in m['files'].values():
                    with open(os.path.join(d, fn), 'wb') as f:
                        f.write(z.read(fn))
        out['materials'][name] = dict(m, files=files)
    for name, e in src.get('extra', {}).items():
        d = os.path.join(DEST, name)
        os.makedirs(d, exist_ok=True)
        for fn in e['files']:
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                print('cc0: %s <- %s' % (fn, e['base_url'] + fn), flush=True)
                blob = get(e['base_url'] + fn)
                with open(p + '.part', 'wb') as f:
                    f.write(blob)
                os.replace(p + '.part', p)
    with open(os.path.join(DEST, 'index.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('cc0: %d materials in %s' % (len(out['materials']), DEST))


if __name__ == '__main__':
    main()
