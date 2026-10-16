"""align/ -- shared transient-mask and detection-cache helpers.

Used by s40_masks (which writes the masks), the wallphoto facade passes and
pipeline/facade_quality (which read them): the per-area paths, the transient
class test, the run log and step markers, the panorama id list, and the
detection and mask file layout with a mask sampler for equirect pixel coords.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import opencity_paths as OCP                    # noqa: E402  host roots, env-overridable

W    = OCP.CODE
SCR  = OCP.SHARED + '/texture'
AL   = SCR + '/align'
MK   = AL + '/markers'
LOGS = W + '/logs'
SRC  = OCP.SOURCE
IMD  = SRC + '/panos/images_pano'
META = SRC + '/panos/metadata/images.jsonl'
DET  = SRC + '/detections'
DETM = DET + '/masks'

# per-area paths when an area is set (OPENCITY_AREA set and its area.yaml has
# paths.scratch_dir; pipeline/areacfg.module_paths); otherwise the defaults above.
_AP = None
if os.environ.get('OPENCITY_AREA', '').strip():
    import common as _C
    _AP = _C._areacfg.module_paths(_C) if hasattr(_C, '_areacfg') else None
if _AP:
    W = _AP['W']; SCR = _AP['SCR']; AL = SCR + '/align'; MK = AL + '/markers'
    LOGS = W + '/logs'; SRC = _AP['SRC']; IMD = _AP['IMD']; META = _AP['META']
    DET = _AP['DET']; DETM = DET + '/masks'

for d in (AL, MK, DET, DETM, LOGS):
    os.makedirs(d, exist_ok=True)

TRANSIENT_PREFIX = ('human--', 'object--vehicle--', 'animal--')
TRANSIENT_EXACT = {'void--dynamic', 'void--ego-vehicle'}

MASK_W, MASK_H = 2048, 1024


def is_transient(v):
    return v in TRANSIENT_EXACT or any(v.startswith(p) for p in TRANSIENT_PREFIX)


def log(msg, name='align'):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(f'{LOGS}/{name}.log', 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass


def marker(name):
    return f'{MK}/{name}.done'


def mark(name, info=None):
    with open(marker(name), 'w') as f:
        json.dump(info or {'at': time.strftime('%Y-%m-%dT%H:%M:%S')}, f)


def have(name):
    return os.path.exists(marker(name))


# ---- ids ------------------------------------------------------------------ #
def image_ids():
    out = []
    for l in open(META):
        out.append(str(json.loads(l)['id']))
    return out


# ---- detections cache ----------------------------------------------------- #
def det_path(pid):
    return f'{DET}/{str(pid)[-2:]}/{pid}.json.gz'


def mask_path(pid):
    return f'{DETM}/{str(pid)[-2:]}/{pid}.png'


def load_mask(pid):
    """-> bool (MASK_H, MASK_W), True where a transient class covers the pixel."""
    p = mask_path(pid)
    if not os.path.exists(p):
        return None
    from PIL import Image
    return np.asarray(Image.open(p), 'u1') > 0


def mask_lookup(mask, u, v, Wpx):
    """Sample a transient mask at equirect pixel coords of a Wpx-wide image."""
    if mask is None:
        return np.zeros(np.shape(u), bool)
    sx = MASK_W / float(Wpx)
    sy = MASK_H / float(Wpx // 2)
    mj = np.clip((u * sx).astype(np.int32), 0, MASK_W - 1)
    mi = np.clip((v * sy).astype(np.int32), 0, MASK_H - 1)
    return mask[mi, mj]
