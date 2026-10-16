"""Shared paths and small helpers for the s84 object inventory.

Frame is the photo-build ENU of common.py (x=East, y=North, z=Up, metres).
"""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import opencity_paths as OCP                    # noqa: E402  host roots, env-overridable
import numpy as np
import common as C

SCR = OCP.SHARED + '/objects'
DET = OCP.SOURCE + '/detections'
PUB = OCP.DATA + '/model'
# per-area paths when an area is set (pipeline/areacfg.module_paths);
# otherwise the defaults above.
_AP = C._areacfg.module_paths(C) if hasattr(C, '_areacfg') else None
if _AP:
    SCR = _AP['OBJ']; DET = _AP['DET']; PUB = C.OUT
for d in (SCR, SCR + '/work', SCR + '/reports', SCR + '/logs'):
    os.makedirs(d, exist_ok=True)


def det_path(pid):
    return '%s/%s/%s.json.gz' % (DET, str(pid)[-2:], pid)


def seeded(key, salt=''):
    h = hashlib.sha256((str(key) + '|' + salt).encode()).digest()
    return int.from_bytes(h[:8], 'little')


def hash_unit(key, salt=''):
    return ((seeded(key, salt) >> 8) & 0xFFFF) / 65535.0


def load_poses(inbox_only=True, margin=20.0):
    """-> dict pid -> pose, panoramas with a usable pose and a cached detection."""
    P = json.load(open(C.W + '/pano_poses.json'))['poses']
    out = {}
    for k, v in P.items():
        if v.get('confidence') not in ('high', 'usable'):
            continue
        if not os.path.exists(det_path(k)):
            continue
        out[k] = v
    return out


def enu_of(lat, lon):
    p = C.lla_to_enu(np.asarray([lat], 'f8'), np.asarray([lon], 'f8'),
                     np.asarray([0.0], 'f8'))[0]
    return float(p[0]), float(p[1])
