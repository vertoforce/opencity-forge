#!/usr/bin/env python3
"""
fetch_area.py -- download Mapillary street-level imagery + metadata + sparse SfM
point clouds for a bounding box, into a directory ready to feed a reconstruction
(COLMAP / OpenSfM / nerfstudio).

Discovery goes through the VECTOR TILE endpoint (tiles.mapillary.com), not the
Graph API `/images?bbox=` search -- that search endpoint returns HTTP 500
("Please reduce the amount of data you're asking for") on a box of any size.  The tile
endpoint is also far cheaper: one z14 tile returns every image point in ~2.4 km
of ground for a single request.

Per-image metadata then comes from the batched Graph endpoint
`/images?image_ids=a,b,c`, which works fine.

Usage:
  .venv/bin/python fetch_area.py --bbox=W,S,E,N --out DIR [options]

NOTE the `=`: a western longitude starts with '-', which argparse would
otherwise read as a flag.  Always write --bbox=... not --bbox ...

Example (a block of downtown San Francisco):
  .venv/bin/python fetch_area.py \
      --bbox=-122.4010,37.7920,-122.4000,37.7928 \
      --out $OPENCITY_DATA/sf_block --max-images 400
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import sys
import threading
import time
import zlib
from pathlib import Path

# Host roots: env-overridable, same names and defaults as opencity_paths.py in the
# pipeline (OPENCITY_TOKEN_FILE).
_TOKEN_FILE = Path(os.environ.get("OPENCITY_TOKEN_FILE") or Path.home() / ".mapillary_token")
from typing import Any, Iterable

import mapbox_vector_tile
import mercantile
import requests

TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
GRAPH = "https://graph.mapillary.com"
TILE_ZOOM = 14          # the only zoom that carries the per-image point layer
TILE_EXTENT = 4096

# Everything useful for reconstruction.  `camera_parameters` is
# [normalized_focal, k1, k2] for perspective/fisheye and [] for spherical.
IMAGE_FIELDS = ",".join([
    "id", "camera_type", "camera_parameters", "exif_orientation",
    "width", "height", "is_pano",
    "geometry", "computed_geometry", "altitude", "computed_altitude",
    "compass_angle", "computed_compass_angle", "computed_rotation",
    "atomic_scale", "captured_at", "sequence", "creator",
    "thumb_original_url", "thumb_2048_url", "thumb_1024_url",
    "sfm_cluster", "mesh",
])

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


# --------------------------------------------------------------------------- #
# geodesy: OpenSfM reconstructions are in a local ENU frame anchored at
# reference_lla.  These two functions move between that frame and WGS84.
# --------------------------------------------------------------------------- #

def lla_to_ecef(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    la, lo = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(la) ** 2)
    return (
        (n + alt) * math.cos(la) * math.cos(lo),
        (n + alt) * math.cos(la) * math.sin(lo),
        (n * (1 - WGS84_E2) + alt) * math.sin(la),
    )


def enu_to_lla(e: float, n: float, u: float,
               lat0: float, lon0: float, alt0: float) -> tuple[float, float, float]:
    """Local East/North/Up metres -> WGS84 lat, lon, alt."""
    la, lo = math.radians(lat0), math.radians(lon0)
    sla, cla, slo, clo = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    x0, y0, z0 = lla_to_ecef(lat0, lon0, alt0)
    x = x0 + (-slo * e - sla * clo * n + cla * clo * u)
    y = y0 + (clo * e - sla * slo * n + cla * slo * u)
    z = z0 + (cla * n + sla * u)

    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(8):  # converges in ~4
        nn = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - nn
        lat = math.atan2(z, p * (1 - WGS84_E2 * nn / (nn + alt)))
    nn = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - nn
    return math.degrees(lat), math.degrees(lon), alt


def rodrigues(r: Iterable[float]) -> list[list[float]]:
    """Axis-angle 3-vector -> 3x3 rotation matrix (OpenSfM shot['rotation'])."""
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    th = math.sqrt(rx * rx + ry * ry + rz * rz)
    if th < 1e-12:
        return [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    kx, ky, kz = rx / th, ry / th, rz / th
    c, s, v = math.cos(th), math.sin(th), 1 - math.cos(th)
    return [
        [c + kx * kx * v,      kx * ky * v - kz * s, kx * kz * v + ky * s],
        [ky * kx * v + kz * s, c + ky * ky * v,      ky * kz * v - kx * s],
        [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v],
    ]


def shot_origin(shot: dict) -> tuple[float, float, float]:
    """Camera centre in the reconstruction's ENU frame.

    OpenSfM stores the WORLD->CAMERA transform, so shot['translation'] is NOT
    the camera position.  origin = -R^T t   (verified against the shot's own
    gps_position: median residual 16.7 m with gps_dop 15, vs 113 m for raw t).
    """
    r = rodrigues(shot["rotation"])
    t = shot["translation"]
    return tuple(-(r[0][i] * t[0] + r[1][i] * t[1] + r[2][i] * t[2]) for i in range(3))


# --------------------------------------------------------------------------- #
# http
# --------------------------------------------------------------------------- #

class Client:
    def __init__(self, token: str, workers: int = 8):
        self.token = token
        self.local = threading.local()
        self.workers = workers

    @property
    def s(self) -> requests.Session:
        s = getattr(self.local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = "fetch_area.py/1.0"
            a = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
            s.mount("https://", a)
            self.local.s = s
        return s

    def get(self, url: str, *, params: dict | None = None, tries: int = 5,
            timeout: int = 120) -> requests.Response:
        last = None
        for i in range(tries):
            try:
                r = self.s.get(url, params=params, timeout=timeout)
                if r.status_code == 200:
                    return r
                # 429/5xx are worth retrying; 4xx other than 429 are not
                if r.status_code < 500 and r.status_code != 429:
                    r.raise_for_status()
                last = requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
            except requests.RequestException as e:
                last = e
            time.sleep(min(30, (2 ** i) + random.random()))
        raise RuntimeError(f"GET failed after {tries}: {url} :: {last}")


# --------------------------------------------------------------------------- #
# stage 1: discover image ids from vector tiles
# --------------------------------------------------------------------------- #

def discover(cl: Client, bbox: tuple[float, float, float, float],
             raw_tile_dir: Path | None) -> list[dict]:
    w, s, e, n = bbox
    tiles = list(mercantile.tiles(w, s, e, n, TILE_ZOOM))
    out: list[dict] = []
    for t in tiles:
        r = cl.get(TILE_URL.format(z=t.z, x=t.x, y=t.y),
                   params={"access_token": cl.token})
        if raw_tile_dir:
            (raw_tile_dir / f"{t.z}_{t.x}_{t.y}.mvt").write_bytes(r.content)
        b = mercantile.bounds(t)

        def tf(x, y, _b=b):
            return (_b.west + (x / TILE_EXTENT) * (_b.east - _b.west),
                    _b.south + (y / TILE_EXTENT) * (_b.north - _b.south))

        dec = mapbox_vector_tile.decode(r.content, transformer=tf)
        layer = dec.get("image", {}).get("features", [])
        hit = 0
        for f in layer:
            lon, lat = f["geometry"]["coordinates"]
            if not (w <= lon <= e and s <= lat <= n):
                continue
            p = dict(f["properties"])
            p["_lon"], p["_lat"] = lon, lat
            out.append(p)
            hit += 1
        print(f"  tile {t.z}/{t.x}/{t.y}  {len(r.content)/1e6:.1f} MB  "
              f"{len(layer)} pts  {hit} in bbox", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# stage 2: batched metadata
# --------------------------------------------------------------------------- #

def fetch_metadata(cl: Client, ids: list[str], chunk: int = 50) -> list[dict]:
    recs: list[dict] = []
    chunks = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]

    def one(c: list[str]) -> list[dict]:
        try:
            r = cl.get(f"{GRAPH}/images", params={
                "access_token": cl.token,
                "image_ids": ",".join(c),
                "fields": IMAGE_FIELDS,
            }, timeout=90)
            return r.json().get("data", [])
        except Exception as ex:
            if len(c) == 1:
                print(f"  ! metadata failed for {c[0]}: {ex}", file=sys.stderr)
                return []
            h = len(c) // 2                      # halve and retry
            return one(c[:h]) + one(c[h:])

    with cf.ThreadPoolExecutor(cl.workers) as ex:
        for got in ex.map(one, chunks):
            recs.extend(got)
    return recs


# --------------------------------------------------------------------------- #
# stage 3: images
# --------------------------------------------------------------------------- #

def download_images(cl: Client, recs: list[dict], root: Path, quality: str) -> dict:
    key = {"original": "thumb_original_url",
           "2048": "thumb_2048_url",
           "1024": "thumb_1024_url"}[quality]
    flat, pano = root / "images", root / "images_pano"
    flat.mkdir(parents=True, exist_ok=True)
    pano.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "skip": 0, "fail": 0, "bytes": 0}
    lock = threading.Lock()

    def one(rec: dict) -> None:
        url = rec.get(key) or rec.get("thumb_2048_url") or rec.get("thumb_original_url")
        if not url:
            with lock:
                stats["fail"] += 1
            return
        d = pano if rec.get("is_pano") else flat
        p = d / f"{rec['id']}.jpg"
        if p.exists() and p.stat().st_size > 0:
            with lock:
                stats["skip"] += 1
                stats["bytes"] += p.stat().st_size
            return
        try:
            r = cl.get(url, timeout=180)
            tmp = p.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(p)
            with lock:
                stats["ok"] += 1
                stats["bytes"] += len(r.content)
        except Exception as ex:
            print(f"  ! image {rec['id']}: {ex}", file=sys.stderr)
            with lock:
                stats["fail"] += 1

    with cf.ThreadPoolExecutor(cl.workers) as ex:
        list(ex.map(one, recs))
    return stats


# --------------------------------------------------------------------------- #
# stage 4: sfm_cluster point clouds (deduped -- many images share one cluster)
# --------------------------------------------------------------------------- #

def download_clusters(cl: Client, recs: list[dict], root: Path) -> dict:
    d = root / "sfm"
    d.mkdir(parents=True, exist_ok=True)
    want: dict[str, str] = {}
    no_url = 0
    for r in recs:
        sc = r.get("sfm_cluster") or {}
        if sc.get("url") and sc.get("id"):
            want.setdefault(str(sc["id"]), sc["url"])
        elif sc.get("id"):
            no_url += 1
    stats = {"clusters": 0, "skip": 0, "fail": 0, "bytes": 0,
             "points": 0, "shots": 0, "images_without_cluster_url": no_url}
    lock = threading.Lock()

    def one(item: tuple[str, str]) -> None:
        cid, url = item
        p = d / f"{cid}.json"
        if p.exists() and p.stat().st_size > 0:
            with lock:
                stats["skip"] += 1
            return
        try:
            raw = cl.get(url, timeout=300).content
            try:
                blob = zlib.decompress(raw)
            except zlib.error:
                blob = zlib.decompress(raw, 16 + zlib.MAX_WBITS)  # gzip wrapper
            recons = json.loads(blob)
            p.write_bytes(blob)
            npts = sum(len(x.get("points", {})) for x in recons)
            nsh = sum(len(x.get("shots", {})) for x in recons)
            with lock:
                stats["clusters"] += 1
                stats["bytes"] += len(blob)
                stats["points"] += npts
                stats["shots"] += nsh
        except Exception as ex:
            print(f"  ! cluster {cid}: {ex}", file=sys.stderr)
            with lock:
                stats["fail"] += 1

    with cf.ThreadPoolExecutor(min(cl.workers, 6)) as ex:
        list(ex.map(one, want.items()))
    return stats


# --------------------------------------------------------------------------- #
# stage 5: derived, reconstruction-ready artefacts
# --------------------------------------------------------------------------- #

def write_cluster_ply(root: Path, max_radius: float = 250.0,
                      epsg_out: int | None = None) -> dict:
    """Merge every downloaded cluster's sparse points into one ENU-anchored PLY.

    The PLY is in metres, East/North/Up, anchored at `anchor` in
    sparse_cloud_anchor.json so it can be lifted straight back to WGS84.

    Outlier rejection matters here.  OpenSfM triangulates some features from
    near-parallel rays and parks them at effectively infinity: a cluster whose real points
    stay within a few hundred metres of its camera path can also hold points
    thousands of km out, enough to stretch a merged cloud across thousands of km.
    Points further than `max_radius` from
    their own cluster's camera-path median are dropped.
    """
    d = root / "sfm"
    files = sorted(d.glob("*.json"))
    if not files:
        return {"points": 0}
    anchor = None
    rows: list[str] = []
    cams: list[str] = []
    dropped = 0
    for f in files:
        for rec in json.loads(f.read_text()):
            ref = rec.get("reference_lla")
            if not ref:
                continue
            if anchor is None:
                anchor = (ref["latitude"], ref["longitude"], ref.get("altitude", 0.0))
            # shift this cluster's ENU into the common anchor's ENU
            dx, dy, dz = 0.0, 0.0, 0.0
            if (ref["latitude"], ref["longitude"]) != (anchor[0], anchor[1]):
                x0, y0, z0 = lla_to_ecef(*anchor)
                x1, y1, z1 = lla_to_ecef(ref["latitude"], ref["longitude"],
                                         ref.get("altitude", 0.0))
                la, lo = math.radians(anchor[0]), math.radians(anchor[1])
                sla, cla, slo, clo = (math.sin(la), math.cos(la),
                                      math.sin(lo), math.cos(lo))
                ex_, ey_, ez_ = x1 - x0, y1 - y0, z1 - z0
                dx = -slo * ex_ + clo * ey_
                dy = -sla * clo * ex_ - sla * slo * ey_ + cla * ez_
                dz = cla * clo * ex_ + cla * slo * ey_ + sla * ez_
            origins = [shot_origin(sh) for sh in rec.get("shots", {}).values()]
            if origins:
                mx = sorted(o[0] for o in origins)[len(origins) // 2]
                my = sorted(o[1] for o in origins)[len(origins) // 2]
                mz = sorted(o[2] for o in origins)[len(origins) // 2]
            else:
                mx = my = mz = 0.0
            for p in rec.get("points", {}).values():
                x, y, z = p["coordinates"]
                if max_radius and math.dist((x, y, z), (mx, my, mz)) > max_radius:
                    dropped += 1
                    continue
                c = p.get("color", [200, 200, 200])
                rows.append(f"{x+dx:.4f} {y+dy:.4f} {z+dz:.4f} "
                            f"{int(c[0])} {int(c[1])} {int(c[2])}")
            for sid, sh in rec.get("shots", {}).items():
                ox, oy, oz = shot_origin(sh)
                lat, lon, alt = enu_to_lla(ox + dx, oy + dy, oz + dz, *anchor)
                cams.append(json.dumps({
                    "shot_id": sid, "cluster": f.stem, "camera": sh.get("camera"),
                    "enu": [ox + dx, oy + dy, oz + dz],
                    "lat": lat, "lon": lon, "alt": alt,
                    "capture_time": sh.get("capture_time"),
                    "gps_dop": sh.get("gps_dop"),
                }))
    ply = root / "sparse_cloud.ply"
    with ply.open("w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(rows)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        fh.write("\n".join(rows))
        fh.write("\n")
    # Second copy in a real projected CRS, so it drops straight into GIS /
    # py3dtiles / CesiumJS without a hand-built transform.  Written with
    # `double` coordinates on purpose: UTM northings run to several million metres
    # (4.2e6 to 8.4e6 over most of the northern mid-latitudes), and float32
    # only resolves 0.5 to 1 m at that size, which would quantise the whole cloud.
    epsg = None
    if epsg_out != 0:
        try:
            from pyproj import CRS, Transformer
            if epsg_out is None:
                zone = int((anchor[1] + 180) / 6) + 1
                epsg = (32600 if anchor[0] >= 0 else 32700) + zone
            else:
                epsg = int(epsg_out)
            tr = Transformer.from_crs(CRS.from_epsg(4979), CRS.from_epsg(epsg),
                                      always_xy=True)
            pp = root / f"sparse_cloud_epsg{epsg}.ply"
            with pp.open("w") as fh:
                fh.write("ply\nformat ascii 1.0\n")
                fh.write(f"element vertex {len(rows)}\n")
                fh.write("property double x\nproperty double y\n"
                         "property double z\n")
                fh.write("property uchar red\nproperty uchar green\n"
                         "property uchar blue\n")
                fh.write("end_header\n")
                for r in rows:
                    f6 = r.split()
                    lat, lon, alt = enu_to_lla(float(f6[0]), float(f6[1]),
                                               float(f6[2]), *anchor)
                    x, y, z = tr.transform(lon, lat, alt)
                    fh.write(f"{x:.4f} {y:.4f} {z:.4f} "
                             f"{f6[3]} {f6[4]} {f6[5]}\n")
            proj_bytes = pp.stat().st_size
        except ImportError:
            print("      (pyproj not installed -- skipping projected PLY)",
                  file=sys.stderr)
            epsg, proj_bytes = None, 0
    else:
        proj_bytes = 0

    (root / "sparse_cloud_anchor.json").write_text(json.dumps({
        "frame": "ENU metres (x=East, y=North, z=Up)",
        "anchor_lla": {"latitude": anchor[0], "longitude": anchor[1],
                       "altitude": anchor[2]},
        "note": "lat,lon,alt = enu_to_lla(x, y, z, **anchor_lla)",
        "projected_ply": (f"sparse_cloud_epsg{epsg}.ply" if epsg else None),
        "projected_epsg": epsg,
    }, indent=2))
    (root / "sfm_shot_origins.jsonl").write_text("\n".join(cams) + "\n")
    return {"points": len(rows), "outliers_dropped": dropped,
            "max_radius_m": max_radius, "shots": len(cams),
            "anchor": anchor, "ply_bytes": ply.stat().st_size,
            "projected_epsg": epsg, "projected_ply_bytes": proj_bytes}


def _actual_size(path: Path) -> tuple[int, int] | None:
    """Pixel size of a downloaded JPEG, read from its SOF marker (no Pillow)."""
    try:
        with path.open("rb") as fh:
            if fh.read(2) != b"\xff\xd8":
                return None
            while True:
                b = fh.read(1)
                if not b:
                    return None
                if b != b"\xff":
                    continue
                while b == b"\xff":
                    b = fh.read(1)
                m = b[0]
                if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
                    continue
                ln = int.from_bytes(fh.read(2), "big")
                if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                    fh.read(1)
                    h = int.from_bytes(fh.read(2), "big")
                    w = int.from_bytes(fh.read(2), "big")
                    return w, h
                fh.seek(ln - 2, 1)
    except Exception:
        return None


def write_recon_inputs(root: Path, recs: list[dict]) -> dict:
    """COLMAP geo-reference list + OpenSfM overrides + a camera summary.

    Camera width/height come from the JPEG actually on disk, not from the
    Graph API's `width`/`height` -- those describe the ORIGINAL upload, and
    --quality 2048 gives you a resized copy (e.g. a 4096x2048 pano arrives as
    2048x1024).  OpenSfM's focal is normalised by max(w, h) so it survives the
    resize untouched; the pixel dimensions do not.
    """
    geo = root / "colmap_geo_ref.txt"          # for: colmap model_aligner
    lines, exif_over, cam_over = [], {}, {}
    ntypes: dict[str, int] = {}
    resized = 0
    for r in recs:
        sub = "images_pano" if r.get("is_pano") else "images"
        name = f"{r['id']}.jpg"
        g = (r.get("computed_geometry") or r.get("geometry") or {}).get("coordinates")
        if not g:
            continue
        lon, lat = g
        alt = r.get("computed_altitude")
        if alt is None:
            alt = r.get("altitude") or 0.0
        # model_aligner --ref_is_gps 1 expects: IMAGE_NAME LAT LON ALT
        lines.append(f"{name} {lat:.9f} {lon:.9f} {alt:.3f}")

        exif_over[name] = {
            "gps": {"latitude": lat, "longitude": lon, "altitude": float(alt),
                    "dop": 5.0},
            "capture_time": (r.get("captured_at") or 0) / 1000.0,
            "compass": {"angle": r.get("computed_compass_angle")
                        if r.get("computed_compass_angle") is not None
                        else r.get("compass_angle", 0.0)},
        }
        ct = r.get("camera_type") or "unknown"
        ntypes[ct] = ntypes.get(ct, 0) + 1
        w, h = r.get("width"), r.get("height")
        real = _actual_size(root / sub / name)
        if real:
            if real != (w, h):
                resized += 1
            w, h = real
        cp = r.get("camera_parameters") or []
        if not (w and h):
            continue
        if ct == "spherical" or r.get("is_pano"):
            cam_over[name] = {"projection_type": "spherical", "width": w, "height": h}
        elif len(cp) >= 3:
            # OpenSfM focal is normalised by max(width, height), so it is
            # resolution independent; multiply by max(w, h) for pixels.
            cam_over[name] = {
                "projection_type": "brown" if ct == "perspective" else "fisheye",
                "width": w, "height": h,
                "focal_x": cp[0], "focal_y": cp[0],
                "c_x": 0.0, "c_y": 0.0,
                "k1": cp[1], "k2": cp[2], "k3": 0.0, "p1": 0.0, "p2": 0.0,
            }
    geo.write_text("\n".join(lines) + "\n")
    # Mapillary's [focal, k1, k2] is bit-for-bit COLMAP's RADIAL model
    # (params "f, cx, cy, k1, k2", distortion k1*r^2 + k2*r^4 on normalised
    # coords), so no reparameterisation is needed -- just denormalise focal.
    # Panos get EQUIRECTANGULAR, native in COLMAP since 4.1.0, whose only
    # params are w,h.
    crows = ["# COLMAP cameras.txt -- one camera per image.",
             "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"]
    for i, (name, cm) in enumerate(cam_over.items(), start=1):
        w_, h_ = cm["width"], cm["height"]
        if cm["projection_type"] == "spherical":
            crows.append(f"{i} EQUIRECTANGULAR {w_} {h_} {w_} {h_}  # {name}")
        else:
            f_ = cm["focal_x"] * max(w_, h_)
            crows.append(f"{i} RADIAL {w_} {h_} {f_:.4f} {w_/2:.1f} {h_/2:.1f} "
                         f"{cm['k1']:.9f} {cm['k2']:.9f}  # {name}")
    (root / "colmap_cameras.txt").write_text("\n".join(crows) + "\n")
    od = root / "opensfm"
    od.mkdir(exist_ok=True)
    (od / "exif_overrides.json").write_text(json.dumps(exif_over, indent=1))
    (od / "camera_models_overrides.json").write_text(json.dumps(cam_over, indent=1))
    return {"geo_ref_rows": len(lines), "camera_types": ntypes,
            "cameras_rescaled_to_download_size": resized}


# --------------------------------------------------------------------------- #
# stage 6 (optional): cut equirectangular panoramas into perspective slices
# --------------------------------------------------------------------------- #

def slice_panos(root: Path, recs: list[dict], n: int, size: int,
                hfov: float, pitches: list[float], workers: int) -> dict:
    """Equirect -> N perspective views per pano, via ffmpeg's v360 filter.

    Older COLMAP releases (before 4.1.0) have no equirectangular camera model, so
    panoramas are resampled to pinhole views for them.  Each slice is a perfect
    pinhole: principal point at the centre, zero distortion, and

        focal_px = (size / 2) / tan(hfov / 2)

    which is written to colmap_pano_cameras.txt in COLMAP's SIMPLE_PINHOLE
    form so you can use it with --ImageReader.camera_model.

    ffmpeg gotcha: v360's `yaw` is clamped to [-180, 180], so a 270-degree
    heading must be written as -90.
    """
    src = root / "images_pano"
    if not src.exists():
        return {"skipped": "no panoramas"}
    out = root / "images_pano_slices"
    out.mkdir(exist_ok=True)
    heading = {str(r["id"]): (r.get("computed_compass_angle")
                              if r.get("computed_compass_angle") is not None
                              else r.get("compass_angle") or 0.0)
               for r in recs}
    yaws = [(i * 360.0 / n) for i in range(n)]
    jobs = []
    sizes: set[int] = set()
    # Slice only the panoramas in `recs`, so --ids-file restricts slicing and
    # every sliced panorama has a heading entry.
    wanted = {str(r["id"]) for r in recs}
    candidates = [q for q in sorted(src.glob("*.jpg")) if q.stem in wanted]
    for img in candidates:
        sz = size
        if sz <= 0:  # auto: native sampling, as pycolmap's panorama example does
            wh = _actual_size(img)
            sz = max(64, int(wh[0] * hfov / 360.0)) if wh else 1024
        sizes.add(sz)
        for pitch in pitches:
            for k, y in enumerate(yaws):
                # v360 yaw must land in [-180, 180]
                yy = ((y + 180.0) % 360.0) - 180.0
                dst = out / f"{img.stem}_p{int(pitch):+03d}_y{k:02d}_{sz}.jpg"
                jobs.append((img, dst, yy, pitch, sz))
    stats = {"panos": len(candidates), "slices": 0,
             "skip": 0, "fail": 0, "bytes": 0}
    lock = threading.Lock()

    def one(j) -> None:
        img, dst, yy, pitch, sz = j
        if dst.exists() and dst.stat().st_size > 0:
            with lock:
                stats["skip"] += 1
                stats["bytes"] += dst.stat().st_size
            return
        vf = (f"v360=equirect:flat:h_fov={hfov}:v_fov={hfov}"
              f":yaw={yy}:pitch={pitch}:w={sz}:h={sz}:interp=lanczos")
        rc = os.system(f'ffmpeg -loglevel error -y -i "{img}" -vf "{vf}" '
                       f'-q:v 2 "{dst}" 2>/dev/null')
        with lock:
            if rc == 0 and dst.exists():
                stats["slices"] += 1
                stats["bytes"] += dst.stat().st_size
            else:
                stats["fail"] += 1

    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))

    def fpx(sz: int) -> float:
        return (sz / 2.0) / math.tan(math.radians(hfov) / 2.0)

    lines = ["# COLMAP cameras.txt for the pano slices.",
             "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[f, cx, cy]"]
    cam_id = {}
    for i, sz in enumerate(sorted(sizes), start=1):
        cam_id[sz] = i
        lines.append(f"{i} SIMPLE_PINHOLE {sz} {sz} "
                     f"{fpx(sz):.6f} {sz/2:.1f} {sz/2:.1f}")
    (root / "colmap_pano_cameras.txt").write_text("\n".join(lines) + "\n")
    # per-slice absolute heading, handy for seeding pose priors
    rows = []
    for img, dst, yy, pitch, sz in jobs:
        base = heading.get(img.stem, 0.0)
        rows.append(json.dumps({"slice": dst.name, "pano_image_id": img.stem,
                                "rel_yaw_deg": yy, "pitch_deg": pitch,
                                "abs_heading_deg": (base + yy) % 360.0,
                                "focal_px": fpx(sz), "width": sz, "height": sz,
                                "colmap_camera_id": cam_id[sz]}))
    (root / "pano_slice_index.jsonl").write_text("\n".join(rows) + "\n")
    stats["slice_sizes"] = sorted(sizes)
    stats["focal_px"] = {sz: round(fpx(sz), 2) for sz in sorted(sizes)}
    stats["hfov_deg"] = hfov
    return stats


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", required=True,
                    help="W,S,E,N in WGS84 degrees; use --bbox=-122.41,37.79,-122.40,37.80 "
                         "(the = is required for negative longitudes)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--token-file", type=Path, default=_TOKEN_FILE)
    ap.add_argument("--max-images", type=int, default=500,
                    help="cap after filtering; 0 = no cap")
    ap.add_argument("--quality", choices=["original", "2048", "1024"],
                    default="2048")
    ap.add_argument("--pano", choices=["both", "only", "exclude"], default="both")
    ap.add_argument("--min-quality-score", type=float, default=0.0)
    ap.add_argument("--since", type=str, default=None,
                    help="only images captured on/after YYYY-MM-DD")
    ap.add_argument("--sequence", action="append", default=None,
                    help="restrict to these sequence ids (repeatable)")
    ap.add_argument("--ids-file", type=Path, default=None,
                    help="restrict to the image ids listed in this file, one "
                         "per line. Lets you download everything once and then "
                         "slice/reconstruct chosen subsets without refetching.")
    ap.add_argument("--creator", action="append", default=None,
                    help="restrict to these creator ids (repeatable)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--no-clusters", action="store_true")
    ap.add_argument("--keep-tiles", action="store_true")
    ap.add_argument("--slice-panos", type=int, default=0, metavar="N",
                    help="cut each equirect pano into N perspective views "
                         "(needs ffmpeg). 4-8 is typical; 0 = off")
    ap.add_argument("--slice-size", type=int, default=0,
                    help="pixels per square slice; 0 = auto "
                         "(pano_width * hfov/360, i.e. native sampling)")
    ap.add_argument("--slice-hfov", type=float, default=90.0)
    ap.add_argument("--slice-pitch", type=float, action="append", default=None,
                    help="pitch in degrees, repeatable (default: just 0)")
    ap.add_argument("--ply-epsg", type=int, default=None,
                    help="EPSG for the projected copy of the sparse cloud; "
                         "default auto-picks the WGS84 UTM zone, 0 disables")
    ap.add_argument("--cloud-max-radius", type=float, default=250.0,
                    help="drop sparse points further than this many metres "
                         "from their cluster's camera path; 0 disables")
    a = ap.parse_args()

    bbox = tuple(float(v) for v in a.bbox.split(","))
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        ap.error("--bbox must be W,S,E,N with W<E and S<N")

    raw = a.token_file.read_text().strip()
    token = raw.split("=")[-1].strip().strip('"\'')
    if not token.startswith("MLY"):
        print("warning: token does not look like a Mapillary client token",
              file=sys.stderr)

    root: Path = a.out
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    cl = Client(token, a.workers)
    t0 = time.time()

    print(f"[1/6] discovering images in {bbox} via z{TILE_ZOOM} vector tiles",
          file=sys.stderr)
    tile_dir = root / "metadata" / "tiles" if a.keep_tiles else None
    if tile_dir:
        tile_dir.mkdir(parents=True, exist_ok=True)
    pts = discover(cl, bbox, tile_dir)
    print(f"      {len(pts)} image points in bbox", file=sys.stderr)

    since_ms = None
    if a.since:
        since_ms = time.mktime(time.strptime(a.since, "%Y-%m-%d")) * 1000

    want_ids = None
    if a.ids_file:
        want_ids = {l.strip() for l in a.ids_file.read_text().splitlines() if l.strip()}
        print(f"      restricting to {len(want_ids)} ids from {a.ids_file}",
              file=sys.stderr)

    def keep(p: dict) -> bool:
        if want_ids is not None and str(p.get("id")) not in want_ids:
            return False
        if a.pano == "only" and not p.get("is_pano"):
            return False
        if a.pano == "exclude" and p.get("is_pano"):
            return False
        if (p.get("quality_score") or 0) < a.min_quality_score:
            return False
        if since_ms and (p.get("captured_at") or 0) < since_ms:
            return False
        if a.sequence and str(p.get("sequence_id")) not in set(a.sequence):
            return False
        if a.creator and str(p.get("creator_id")) not in set(a.creator):
            return False
        return True

    pts = [p for p in pts if keep(p)]
    print(f"      {len(pts)} after filters", file=sys.stderr)
    if want_ids is not None:
        # an id asked for by name is fetched by id even when the z14 coverage
        # tile does not list it: the tile is a changing snapshot and can omit ids
        # the Graph API still serves, and a missing image stops s60.
        got = {str(p.get("id")) for p in pts}
        miss = sorted(want_ids - got)
        if miss:
            print(f"      {len(miss)} requested ids not in the coverage tile; "
                  f"fetching them by id", file=sys.stderr)
            pts += [{"id": i, "sequence_id": "", "captured_at": 0} for i in miss]
    # Prefer long runs: sort by sequence then time, so a cap keeps contiguous
    # tracks (which is what SfM needs) instead of a scatter of orphan frames.
    pts.sort(key=lambda p: (str(p.get("sequence_id")), p.get("captured_at") or 0))
    if a.max_images and len(pts) > a.max_images:
        pts = pts[:a.max_images]
        print(f"      capped to {len(pts)}", file=sys.stderr)
    if not pts:
        print("nothing to fetch", file=sys.stderr)
        return 1

    print(f"[2/6] fetching metadata for {len(pts)} images", file=sys.stderr)
    recs = fetch_metadata(cl, [str(p["id"]) for p in pts])
    print(f"      got {len(recs)} records", file=sys.stderr)
    mp = root / "metadata" / "images.jsonl"
    mp.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (root / "metadata" / "tile_points.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pts) + "\n")

    istats = {"skipped": True}
    if not a.no_images:
        print(f"[3/6] downloading {a.quality} images", file=sys.stderr)
        istats = download_images(cl, recs, root, a.quality)
        print(f"      {istats}", file=sys.stderr)

    cstats = {"skipped": True}
    plystats: dict[str, Any] = {}
    if not a.no_clusters:
        print("[4/6] downloading sfm_cluster point clouds", file=sys.stderr)
        cstats = download_clusters(cl, recs, root)
        print(f"      {cstats}", file=sys.stderr)
        plystats = write_cluster_ply(root, a.cloud_max_radius, a.ply_epsg)
        print(f"      merged PLY: {plystats}", file=sys.stderr)

    print("[5/6] writing reconstruction inputs", file=sys.stderr)
    rstats = write_recon_inputs(root, recs)
    print(f"      {rstats}", file=sys.stderr)

    sstats = {"skipped": True}
    if a.slice_panos:
        print(f"[6/6] slicing panoramas into {a.slice_panos} views each",
              file=sys.stderr)
        sstats = slice_panos(root, recs, a.slice_panos, a.slice_size,
                             a.slice_hfov, a.slice_pitch or [0.0], a.workers)
        print(f"      {sstats}", file=sys.stderr)

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bbox": bbox,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(a).items() if k != "token_file"},
        "counts": {
            "tile_points_in_bbox": len(pts),
            "metadata_records": len(recs),
            "panoramas": sum(1 for r in recs if r.get("is_pano")),
            "flat": sum(1 for r in recs if not r.get("is_pano")),
        },
        "images": istats,
        "clusters": cstats,
        "sparse_cloud": plystats,
        "recon_inputs": rstats,
        "pano_slices": sstats,
        "elapsed_s": round(time.time() - t0, 1),
        "license": "Mapillary imagery is CC BY-SA 4.0 -- "
                   "attribute creators (see metadata/images.jsonl .creator) "
                   "and share derivatives alike.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
