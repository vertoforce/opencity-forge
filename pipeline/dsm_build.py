#!/usr/bin/env python3
"""DSM helpers for the panorama poses, used by pipeline/nomesh.py.

  build()            max-Z DSM, class-2 min DTM and class-9 water on a grid
  stage_fetch()      a coarse-LOD USGS 3DEP EPT fetch (PDAL in Docker) around the origin,
                     for the 3 x 3 km wide DSM; nomesh.py sets OUT, EPT and register first
  ply_to_registered  the fetched PLY in the area's ENU frame

Resumable: a fetch skips itself if its output already exists.
"""
import os, sys, json, math, time, subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import opencity_paths as OCP  # noqa: E402
from common import lla_to_enu, enu_to_lla, REG_YAW_DEG, REG_T  # noqa: E402

OUT   = (OCP.SHARED + '/pano_pose/dsm')
FW    = (OCP.COMBINED + '/fusework')
PDAL  = 'pdal/pdal:2.9.0'
EPT   = None                           # the area's EPT url; pipeline/nomesh.py sets it
A     = 6378137.0                      # EPSG:3857 sphere radius (WGS84 semi-major axis)

WIDE_HALF = 1500.0
WIDE_CELL = 2.0
EPT_RES   = 3.0                        # EPT (3857) units; measured: ~0.58 m ground spacing
NOISE     = (7, 18)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- frame ---- #
def register(enu):
    """lidar ENU -> photo frame: R(yaw)+t from REG_YAW_DEG / REG_T (the fuse_context registration)."""
    th = math.radians(REG_YAW_DEG)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return enu @ R.T + REG_T


def to3857(lat, lon):
    x = A * np.radians(lon)
    y = A * np.log(np.tan(math.pi / 4 + np.radians(lat) / 2))
    return x, y


# ------------------------------------------------------------ rasterise ---- #
def cellkey(xy, x0, y0, cell, nx, ny):
    ix = np.floor((xy[:, 0] - x0) / cell).astype(np.int64)
    iy = np.floor((xy[:, 1] - y0) / cell).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return iy[ok] * nx + ix[ok], ok


def agg(key, z, n, how):
    """Per-cell max or min of z. NaN where the cell has no point."""
    out = np.full(n, np.nan)
    if len(key) == 0:
        return out
    order = np.argsort(key, kind='stable')
    k = key[order]; zz = z[order]
    b = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    vals = (np.maximum.reduceat(zz, b) if how == 'max'
            else np.minimum.reduceat(zz, b))
    out[k[b]] = vals
    return out


def build(xyz, cls, x0, y0, cell, nx, ny, want_water=False):
    keep = ~np.isin(cls, NOISE)
    p, c = xyz[keep], cls[keep]
    n = nx * ny
    key, ok = cellkey(p, x0, y0, cell, nx, ny)
    z = p[ok, 2].astype('f8'); ck = c[ok]
    dsm = agg(key, z, n, 'max')
    cnt = np.bincount(key, minlength=n).astype('i4')
    g = ck == 2
    dtm = agg(key[g], z[g], n, 'min')
    res = dict(dsm=dsm.reshape(ny, nx).astype('f4'),
               dtm=dtm.reshape(ny, nx).astype('f4'),
               cnt=cnt.reshape(ny, nx),
               grid=np.array([x0, y0, cell, nx, ny], 'f8'))
    if want_water:
        w = np.bincount(key[ck == 9], minlength=n)
        res['water'] = ((w * 2 > cnt) & (cnt > 0)).reshape(ny, nx)
    return res


# ---------------------------------------------------------------- ply ------ #
PLY_NP = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
          "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
          "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
          "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}


def read_ply(path):
    props, n, off = [], 0, 0
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'ply'
        fmt = None
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('format'):
                fmt = line.split()[1]
            elif line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                t, nm = line.split()[1], line.split()[-1]
                props.append((nm, PLY_NP[t]))
            elif line == 'end_header':
                off = f.tell(); break
        assert fmt == 'binary_little_endian', fmt
    arr = np.memmap(path, dtype=np.dtype(props), mode='r',
                    offset=off, shape=(n,))
    alias = {'x': 'X', 'y': 'Y', 'z': 'Z', 'classification': 'Classification'}
    arr.dtype.names = tuple(alias.get(nm.lower(), nm) for nm in arr.dtype.names)
    return arr


# ---------------------------------------------------------------- fetch ---- #
def ept_bounds():
    """EPT bounds (EPSG:3857) covering the wide ENU box with slack."""
    h = WIDE_HALF + 80.0
    ex = np.array([-h, h, h, -h]); ny_ = np.array([-h, -h, h, h])
    lat, lon, _ = enu_to_lla(ex, ny_, 0.0)
    x, y = to3857(lat, lon)
    return (float(x.min()), float(y.min()), float(x.max()), float(y.max()),
            float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))


def net_rx():
    tot = 0
    for line in open('/proc/net/dev').read().splitlines()[2:]:
        name, rest = line.split(':', 1)
        if name.strip() == 'lo':
            continue
        tot += int(rest.split()[0])
    return tot


def stage_fetch():
    ply = f'{OUT}/wide.ply'
    if os.path.exists(ply) and os.path.getsize(ply) > 1000:
        log('fetch: already have', ply, f'{os.path.getsize(ply)/1e6:.0f} MB')
        return ply
    xmn, ymn, xmx, ymx, lomn, lamn, lomx, lamx = ept_bounds()
    pad = 0.0015   # ~150 m of degrees of slack on the 4326 crop
    pipe = {"pipeline": [
        {"type": "readers.ept", "filename": EPT,
         "bounds": f"([{xmn},{xmx}],[{ymn},{ymx}])",
         "resolution": EPT_RES, "threads": 8},
        {"type": "filters.reprojection",
         "in_srs": "EPSG:3857", "out_srs": "EPSG:4326"},
        {"type": "filters.crop",
         "bounds": f"([{lomn-pad},{lomx+pad}],[{lamn-pad},{lamx+pad}])"},
        {"type": "writers.ply", "filename": "/out/wide.ply.tmp",
         "storage_mode": "little endian", "dims": "X,Y,Z,Classification"},
    ]}
    json.dump(pipe, open(f'{OUT}/pipeline_wide.json', 'w'), indent=1)
    log('fetch: 3857 bounds', [round(v, 1) for v in (xmn, ymn, xmx, ymx)])
    cmd = ['docker', 'run', '--rm', '--network', 'host', '--cpus', '8',
           '--memory', '12g', '-v', f'{OUT}:/out', PDAL, 'pdal', 'pipeline',
           '/out/pipeline_wide.json', '--metadata', '/out/metadata_wide.json']
    t0 = time.time()
    rx0 = net_rx()
    p = subprocess.run(cmd, capture_output=True, text=True)
    rx = net_rx() - rx0
    json.dump({"net_rx_bytes": rx, "secs": time.time() - t0},
              open(f'{OUT}/fetch_net.json', 'w'))
    log(f'fetch: host RX during pdal run = {rx/1e6:.1f} MB')
    if p.returncode != 0:
        raise SystemExit('pdal failed: ' + (p.stderr or p.stdout)[-2000:])
    os.rename(f'{OUT}/wide.ply.tmp', ply)
    log(f'fetch: ok in {time.time()-t0:.0f}s, {os.path.getsize(ply)/1e6:.1f} MB')
    return ply


def ply_to_registered(ply):
    a = read_ply(ply)
    log('fetch: points', len(a), 'dims', a.dtype.names)
    lon = np.asarray(a['X'], 'f8'); lat = np.asarray(a['Y'], 'f8')
    z = np.asarray(a['Z'], 'f8')
    cls = np.asarray(a['Classification']).astype('i2')
    enu = lla_to_enu(lat, lon, z)
    return register(enu), cls
