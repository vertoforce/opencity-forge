"""Panorama fetch (`opencity fetch`): metadata first, select, then download only the selection.
Downloading every panorama in the box before thinning fills the disk on a well-covered
box (Lombard St: 6,899 panos, ~43 GB at original quality).  Here:
 1. fetch_area --no-images  (tile discovery + batched metadata for every pano in the box)
 2. spatial thinning: round-robin over sequences, newest sequence first, accept a pano only when
    it is >= --spacing m from every accepted one; cap at --max-panos, evenly spaced
 3. fetch_area --ids-file  (download only those; images.jsonl then holds only the selection)
usage: fetch_panos.py --bbox=W,S,E,N --out DIR [--max-panos 300] [--spacing 3.0]"""
import argparse, json, math, subprocess, sys, os
try:
    import opencity_paths as OCP                # host roots, env-overridable
except ImportError:                            # imported with lib/ off sys.path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
    import opencity_paths as OCP
from pathlib import Path
FETCH_DIR = OCP.FETCH; PY = OCP.PY
ap = argparse.ArgumentParser(); ap.add_argument('--bbox', required=True); ap.add_argument('--out', required=True, type=Path)
ap.add_argument('--max-panos', type=int, default=300); ap.add_argument('--spacing', type=float, default=3.0)
ap.add_argument('--quality', default='original')
a = ap.parse_args()
b = [float(v) for v in a.bbox.split(',')]
base = [PY, f'{FETCH_DIR}/fetch_area.py', f'--bbox={a.bbox}', '--out', str(a.out), '--max-images', '0',
        '--pano', 'only', '--no-clusters', '--workers', '8']
meta = a.out / 'metadata' / 'images.jsonl'
allm = a.out / 'metadata' / 'images_all.jsonl'
if not allm.exists():
    subprocess.run(base + ['--no-images'], check=True)
    os.replace(meta, allm)
recs = [json.loads(l) for l in open(allm) if l.strip()]
lat0, lon0 = (b[1] + b[3]) / 2, (b[0] + b[2]) / 2
mx, my = 111320.0 * math.cos(math.radians(lat0)), 110540.0
seqs = {}
for d in recs:
    g = (d.get('computed_geometry') or {}).get('coordinates')
    if not g or not d.get('computed_rotation') or d.get('camera_type') != 'spherical':
        continue
    seqs.setdefault(str(d.get('sequence')), []).append((d.get('captured_at') or 0, str(d['id']), (g[0] - lon0) * mx, (g[1] - lat0) * my))
lists = sorted(seqs.values(), key=lambda v: -max(t for t, *_ in v))     # newest sequence first
for v in lists: v.sort()
order = [v[i] for i in range(max((len(v) for v in lists), default=0)) for v in lists if i < len(v)]
kept, pts = [], []
for t, pid, x, y in order:
    if any((x - px) ** 2 + (y - py) ** 2 < a.spacing ** 2 for px, py in pts):
        continue
    kept.append(pid); pts.append((x, y))
n_thin = len(kept)
if a.max_panos and len(kept) > a.max_panos:
    st = len(kept) / a.max_panos; kept = [kept[int(i * st)] for i in range(a.max_panos)]
ids = a.out / 'metadata' / 'selected_ids.txt'; ids.write_text('\n'.join(kept) + '\n')
print(f'fetch_panos: {len(recs)} panos with metadata, {len(order)} posed in {len(seqs)} sequences, '
      f'{n_thin} after {a.spacing} m thinning, {len(kept)} selected', flush=True)
subprocess.run(base + ['--ids-file', str(ids), '--quality', a.quality], check=True)
# every selected id must be on disk, or a later stage dies on a missing image
have = {f[:-4] for f in os.listdir(a.out / 'images_pano')} if (a.out / 'images_pano').is_dir() else set()
lost = [i for i in kept if i not in have]
if lost:
    sys.exit(f'fetch_panos: {len(lost)} of {len(kept)} selected panoramas not downloaded: {lost[:5]}')
json.dump(dict(panos_with_metadata=len(recs), posed=len(order), sequences=len(seqs), thinned=n_thin,
               selected=len(kept), spacing_m=a.spacing, max_panos=a.max_panos),
          open(a.out / 'metadata' / 'selection.json', 'w'), indent=1)
