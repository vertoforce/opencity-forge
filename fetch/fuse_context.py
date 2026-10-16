#!/usr/bin/env python3
"""
fuse_context.py -- fuse public lidar + OSM context with a photo reconstruction.

    .venv/bin/python fuse_context.py sf_block --no-photo \
        --photo-dir <dir holding manifest.json>

That is the form `opencity build` runs, with the manifest that
`pipeline/nomesh.py manifest` writes.  The photo mode drops --no-photo and points
--photo-dir at a photogrammetry output dir holding points_enu.ply (ENU photo
points), model.glb (photo mesh, glTF Y-up) and manifest.json with
origin_lat_lon_alt.

Street-level photos and airborne lidar fail in opposite directions.  Photos see
facades, kerbs, benches and path texture, and see nothing of roofs, tree tops or
water, and nothing at all where nobody walked.  Airborne lidar sees every roof,
tree top and lawn interior from above, at about 7 points per square metre, and
almost nothing of a vertical wall.  This tool puts the two in one frame and
lets each one own what it is good at:

  * the photo mesh stays the authority below ~4 m wherever it has surface
  * lidar owns everything overhead, plus the ground where no photo reached
  * OSM footprints + level counts give building massing where one facade was
    photographed and the other three were not
  * water is a flat plane at the lidar-measured surface level, replacing the
    noise photogrammetry always produces over water

Nothing is assumed about the georeferencing: the lidar is registered to the
photo cloud and the residual is measured and recorded.

Conventions follow fetch_context.py: resumable stages, verified
outputs, loud failure, preflight, provenance in the manifest.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fetch_context import (                                      # noqa: E402
    Fatal, now, free_disk_gb, mem_available_gb, sh, read_ply_points,
    read_lidar_ply, write_ply_points)

# Host roots: env-overridable, same names and defaults as opencity_paths.py in the
# pipeline (OPENCITY_DATA, OPENCITY_BUILD).
_DATA = Path(os.environ.get("OPENCITY_DATA") or Path(__file__).resolve().parents[1] / "data")
_BUILD = Path(os.environ.get("OPENCITY_BUILD") or Path(__file__).resolve().parents[1] / "work" / "build")
DEFAULT_WORK_ROOT = _BUILD / "fusion"
DEFAULT_OUT_ROOT = _DATA
DEFAULT_SRC_ROOT = _DATA / "_source"
GLTF_DIR = _BUILD / "gltf"
GLTF_BIN = GLTF_DIR / "node_modules" / ".bin" / "gltf-transform"

STAGES = ["load", "register", "surfaces", "layers", "assemble", "render",
          "publish"]

RAM_HEADROOM_GB = 12.0
DISK_MARGIN_GB = 3.0
MAX_ATLAS_W = 8000          # many WebGL viewers drop textures wider than this (max texture size)

# ASPRS classes treated specially.  Many 3DEP tiles classify only ground, water and
# noise, not vegetation or buildings, so trees and roofs are separated by height
# above ground rather than by class.
CLS_GROUND, CLS_NOISE_LOW, CLS_WATER, CLS_NOISE_HIGH = 2, 7, 9, 18
CLS_IGNORED_GROUND = 20

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


# --------------------------------------------------------------------------- #
# geodesy                                                                      #
# --------------------------------------------------------------------------- #

def lla_to_ecef(lat, lon, alt):
    import numpy as np
    lat = np.radians(np.asarray(lat, dtype="f8"))
    lon = np.radians(np.asarray(lon, dtype="f8"))
    alt = np.asarray(alt, dtype="f8")
    s = np.sin(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * s * s)
    return ((n + alt) * np.cos(lat) * np.cos(lon),
            (n + alt) * np.cos(lat) * np.sin(lon),
            (n * (1 - WGS84_E2) + alt) * s)


def lla_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """WGS84 lat/lon/alt -> local East-North-Up metres about an origin."""
    import numpy as np
    x, y, z = lla_to_ecef(lat, lon, alt)
    x0, y0, z0 = lla_to_ecef(lat0, lon0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0
    la, lo = math.radians(lat0), math.radians(lon0)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    return np.stack([e, n, u], axis=-1)


def yaw_translation(src, dst):
    """Least-squares rigid transform restricted to a yaw rotation about Z plus
    a translation.  Full 3D rotation is not wanted: both clouds are already
    levelled by their own georeferencing, and letting the fit tilt would trade
    a real offset for a fake tilt."""
    import numpy as np
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    p, q = src - cs, dst - cd
    num = float(np.sum(p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]))
    den = float(np.sum(p[:, 0] * q[:, 0] + p[:, 1] * q[:, 1]))
    th = math.atan2(num, den)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = cd - R @ cs
    return R, t, th


def apply_rt(pts, R, t):
    return pts @ R.T + t


# --------------------------------------------------------------------------- #

class Fuse:
    def __init__(self, name: str, a: argparse.Namespace):
        self.name = name
        self.a = a
        self.work = Path(a.work_root) / name
        self.out = Path(a.out_root) / name
        self.ctx = (Path(a.context_dir) if a.context_dir
                    else Path(a.src_root) / name / "context")
        self.photo = Path(a.photo_dir)
        self.markers = self.work / ".stages"
        self.logfile = self.work / "fuse.log"
        self.statusfile = self.work / "status.json"
        self.status: dict[str, Any] = {
            "area": name, "pid": os.getpid(), "started": now(),
            "updated": now(), "state": "starting", "stage": None,
            "stages": {}, "metrics": {}, "error": None}
        if self.statusfile.exists():
            try:
                old = json.loads(self.statusfile.read_text())
                self.status["metrics"] = old.get("metrics", {})
                self.status["stages"] = old.get("stages", {})
            except (json.JSONDecodeError, OSError):
                pass
        self.d: dict[str, Any] = {}        # in-memory handoff between stages

    # ---- plumbing --------------------------------------------------------- #

    def log(self, *msg: Any) -> None:
        line = f"[{now()}] [{self.name}] " + " ".join(str(m) for m in msg)
        print(line, flush=True)
        try:
            self.work.mkdir(parents=True, exist_ok=True)
            with open(self.logfile, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def save_status(self) -> None:
        self.status["updated"] = now()
        try:
            self.work.mkdir(parents=True, exist_ok=True)
            self.statusfile.write_text(json.dumps(self.status, indent=2))
        except OSError:
            pass

    def marker(self, stage: str) -> Path:
        return self.markers / f"{stage}.done"

    def mark_done(self, stage: str, secs: float) -> None:
        self.markers.mkdir(parents=True, exist_ok=True)
        self.marker(stage).write_text(now())
        self.status["stages"][stage] = {"done": now(), "seconds": round(secs, 1)}
        self.save_status()

    def met(self, stage: str) -> dict:
        return self.status["metrics"].setdefault(stage, {})

    # ---- preflight -------------------------------------------------------- #

    def preflight(self) -> None:
        need = [self.ctx / "lidar_points.ply", self.ctx / "osm_features.geojson",
                self.ctx / "lidar_stats.json", self.ctx / "context_manifest.json",
                self.photo / "points_enu.ply", self.photo / "model.glb",
                self.photo / "manifest.json"]
        if getattr(self.a, "no_photo", False):
            # --no-photo: the photo dir holds only manifest.json (ENU origin,
            # attribution), written by `opencity build` (pipeline/nomesh.py manifest)
            need = [p for p in need if p.name == "manifest.json" or p.parent != self.photo]
        # name the tool that writes each missing input
        photo_writer = ("pipeline/nomesh.py manifest, run by `opencity build`"
                        if getattr(self.a, "no_photo", False)
                        else "a photogrammetry build, external to this repository")
        missing = [f"{p}  (written by "
                   f"{photo_writer if p.parent == self.photo else 'fetch_context.py'})"
                   for p in need if not p.exists()]
        if missing:
            raise Fatal("inputs missing:\n  " + "\n  ".join(missing))
        have = free_disk_gb(Path(self.a.work_root))
        if have < DISK_MARGIN_GB:
            raise Fatal(f"only {have:.1f} GB free on {self.a.work_root}, need "
                        f"{DISK_MARGIN_GB} GB")
        ram = mem_available_gb()
        if ram < RAM_HEADROOM_GB:
            raise Fatal(f"only {ram:.1f} GB RAM available; need "
                        f"{RAM_HEADROOM_GB:.0f} GB free")
        self.out.mkdir(parents=True, exist_ok=True)
        self.log(f"preflight ok: disk {have:.1f} GB, ram {ram:.1f} GB, "
                 f"context {self.ctx}, photo {self.photo}")
        self.met("preflight").update({"disk_free_gb": round(have, 1),
                                      "ram_available_gb": round(ram, 1)})

    # ---- stage: load ------------------------------------------------------ #

    def st_load(self) -> None:
        import numpy as np
        from pyproj import Transformer

        pman = json.loads((self.photo / "manifest.json").read_text())
        origin = pman["crs"]["origin_lat_lon_alt"]
        if not origin or len(origin) != 3:
            raise Fatal(f"{self.photo}/manifest.json has no ENU origin")
        self.log(f"photo frame: ENU about {origin} "
                 f"(georeference error {pman.get('georeference_error_m')})")

        if getattr(self.a, "no_photo", False):
            photo, prgb = np.zeros((0, 3), "f8"), None
        else:
            pts = read_ply_points(self.photo / "points_enu.ply")
            photo = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype("f8")
            prgb = (np.stack([pts["red"], pts["green"], pts["blue"]], axis=1)
                    if "red" in pts.dtype.names else None)

        lid = read_lidar_ply(self.ctx / "lidar_points.ply")
        cman = json.loads((self.ctx / "context_manifest.json").read_text())
        epsg = int(cman["metric_crs"].split(":")[1])
        tr = Transformer.from_crs(epsg, 4326, always_xy=True)
        lon, lat = tr.transform(lid["X"].astype("f8"), lid["Y"].astype("f8"))
        lz = lid["Z"].astype("f8")
        # Lidar Z is an orthometric height (NAVD88), the photo frame's up comes
        # from GPS altitudes.  The two differ by the geoid, a constant over a
        # box a few hundred metres across; it is measured and removed in the
        # register stage, not modelled here.
        enu = lla_to_enu(lat, lon, lz, *origin)
        cls = lid["Classification"].astype("i2")

        np.savez_compressed(
            self.work / "loaded.npz", photo=photo.astype("f4"),
            photo_rgb=(prgb.astype("u1") if prgb is not None
                       else np.zeros((0, 3), "u1")),
            lidar=enu.astype("f4"), lidar_cls=cls,
            origin=np.array(origin, dtype="f8"),
            lidar_z_raw=lz.astype("f4"))
        self.d.update(photo=photo, photo_rgb=prgb, lidar=enu, lidar_cls=cls,
                      origin=origin, lidar_z_raw=lz)

        keep = ~np.isin(cls, [CLS_NOISE_LOW, CLS_NOISE_HIGH])
        self.met("load").update({
            "photo_points": int(len(photo)),
            "photo_extent_m": [round(float(v), 1) for v in
                               ((photo.max(axis=0) - photo.min(axis=0)) if len(photo)
                                else np.zeros(3))],
            "lidar_points": int(len(enu)),
            "lidar_points_after_noise_drop": int(keep.sum()),
            "lidar_extent_m": [round(float(v), 1) for v in
                               (enu[keep].max(axis=0) - enu[keep].min(axis=0))],
            "enu_origin": origin,
            "lidar_source_crs": cman["metric_crs"],
            "lidar_dataset": cman["lidar"]["dataset"],
            "lidar_survey_year": cman["lidar"]["survey_year"],
        })
        self.log(f"load: {len(photo):,} photo points, {len(enu):,} lidar points "
                 f"({int(keep.sum()):,} after dropping noise classes); "
                 f"photo extent {self.met('load')['photo_extent_m']} m, "
                 f"lidar extent {self.met('load')['lidar_extent_m']} m")

    def _loaded(self) -> None:
        """Bring the load stage's arrays back after a resume."""
        import numpy as np
        if "photo" in self.d:
            return
        f = self.work / "loaded.npz"
        if not f.exists():
            raise Fatal("loaded.npz missing; rerun with --force-from load")
        z = np.load(f)
        self.d.update(photo=z["photo"].astype("f8"),
                      photo_rgb=(z["photo_rgb"] if len(z["photo_rgb"]) else None),
                      lidar=z["lidar"].astype("f8"),
                      lidar_cls=z["lidar_cls"],
                      origin=[float(v) for v in z["origin"]],
                      lidar_z_raw=z["lidar_z_raw"].astype("f8"))

    # ---- stage: register -------------------------------------------------- #

    def _overlap_masks(self, photo, lid, cell=2.0, min_photo=5):
        """Cells where the photo cloud actually has geometry.  The photo point
        cloud has a long tail of far-field junk (its bounding box can be nearly
        twice the mesh's), so a bounding-box overlap
        test would call a lawn the photos never saw "shared" and the residual
        would measure nothing but that mistake."""
        import numpy as np
        lo = photo.min(axis=0)[:2]
        nx = int(np.ceil((photo[:, 0].max() - lo[0]) / cell)) + 1
        ny = int(np.ceil((photo[:, 1].max() - lo[1]) / cell)) + 1
        pi = np.floor((photo[:, :2] - lo) / cell).astype("i8")
        cnt = np.zeros((ny, nx), "i4")
        np.add.at(cnt, (pi[:, 1], pi[:, 0]), 1)
        occ = cnt >= min_photo
        li = np.floor((lid[:, :2] - lo) / cell).astype("i8")
        inb = ((li[:, 0] >= 0) & (li[:, 0] < nx) & (li[:, 1] >= 0) & (li[:, 1] < ny))
        lsel = np.zeros(len(lid), bool)
        lsel[inb] = occ[li[inb, 1], li[inb, 0]]
        psel = occ[pi[:, 1], pi[:, 0]]
        return lsel, psel, occ, lo, cell, cnt

    def _icp(self, L, tree, P, R0, t0, gate: float):
        """Trimmed ICP, yaw + translation only.  Trimmed because half of each
        cloud is genuinely absent from the other: roofs and canopy tops have no
        photo surface, and photographed shopfront detail has no lidar return."""
        import numpy as np
        R, t = R0.copy(), t0.copy()
        Lw = apply_rt(L, R, t)
        hist = []
        for it in range(self.a.icp_iters):
            dist, j = tree.query(Lw, k=1, workers=self.a.cpus)
            ok = dist < gate
            if ok.sum() < 500:
                raise Fatal(f"ICP lost correspondence at iteration {it}: "
                            f"{int(ok.sum())} pairs within {gate} m")
            cut = np.percentile(dist[ok], self.a.icp_trim_pct)
            sel = ok.copy()
            sel[ok] = dist[ok] <= cut
            dR, dt, th = yaw_translation(Lw[sel], P[j[sel]])
            Lw = apply_rt(Lw, dR, dt)
            R, t = dR @ R, dR @ t + dt
            hist.append({"iter": it,
                         "pairs_within_gate": int(ok.sum()),
                         "trimmed_median_m": round(float(np.median(dist[sel])), 4),
                         "yaw_deg": round(math.degrees(th), 5),
                         "step_m": round(float(np.linalg.norm(dt)), 5)})
            if float(np.linalg.norm(dt)) + abs(th) * 50.0 < 5e-4:
                break
        return R, t, hist, Lw

    def _resid(self, Lw, tree, gate: float) -> dict:
        import numpy as np
        dist, _ = tree.query(Lw, k=1, workers=self.a.cpus)
        ok = dist < gate
        return {"points": int(len(dist)),
                "with_photo_within_gate": int(ok.sum()),
                "gate_m": gate,
                "overlap_pct": round(100 * float(ok.mean()), 1),
                "median_m": round(float(np.median(dist[ok])), 4) if ok.any() else None,
                "mean_m": round(float(dist[ok].mean()), 4) if ok.any() else None,
                "p90_m": round(float(np.percentile(dist[ok], 90)), 4) if ok.any() else None,
                "median_all_m": round(float(np.median(dist)), 4)}

    def st_register(self) -> None:
        import numpy as np
        from scipy.spatial import cKDTree
        self._loaded()
        if getattr(self.a, "no_photo", False):
            # No photo reconstruction to register to: the frame IS the lidar,
            # ENU about the manifest origin with the lidar's NAVD88 heights.
            # Panorama poses are registered to this frame instead (pano heights
            # from the lidar ground, see pipeline/nomesh.py poses).
            R, t = np.eye(3), np.zeros(3)
            reg = {"method": "none (--no-photo): lidar defines the frame",
                   "applied_transform": {"yaw_deg": 0.0, "translation_m": [0.0, 0.0, 0.0],
                                         "horizontal_shift_m": 0.0, "vertical_shift_m": 0.0},
                   "residual": None}
            (self.work / "registration.json").write_text(json.dumps(reg, indent=2))
            self.met("register").update(reg)
            self.d["R"], self.d["t"] = R, t
            self.d["lidar_reg"] = self.d["lidar"].copy()
            np.save(self.work / "lidar_reg.npy", self.d["lidar_reg"].astype("f4"))
            self.log("register: skipped (--no-photo); lidar frame used as is")
            return
        photo = self.d["photo"]
        lidar = self.d["lidar"]
        cls = self.d["lidar_cls"]
        keep = ~np.isin(cls, [CLS_NOISE_LOW, CLS_NOISE_HIGH])
        lid, lcls = lidar[keep], cls[keep]

        lsel, psel, occ, lo, cell, pcnt = self._overlap_masks(
            photo, lid, self.a.reg_cell, self.a.reg_min_photo)
        L, LC, P = lid[lsel], lcls[lsel], photo[psel]
        self.log(f"register: {int(occ.sum())} cells of {cell:.0f} m carry at "
                 f"least {self.a.reg_min_photo} photo points "
                 f"({occ.sum()*cell*cell:,.0f} m2); {len(L):,} lidar and "
                 f"{len(P):,} photo points fall in them")
        if len(L) < 5000 or len(P) < 5000:
            raise Fatal(f"not enough real overlap to register: {len(L)} lidar "
                        f"and {len(P)} photo points share occupied cells")

        # 1. Coarse height offset, measured on the one surface both sensors
        #    definitely see: the ground.
        def _groups(keys, vals):
            """Sort once, then hand back (key, values) per cell.  The obvious
            `vals[keys == k]` inside a loop is O(cells x points): ~2.7e11
            comparisons at 8.5M photo points."""
            order = np.argsort(keys, kind="stable")
            k_s, v_s = keys[order], vals[order]
            starts = np.flatnonzero(np.r_[True, k_s[1:] != k_s[:-1]])
            ends = np.r_[starts[1:], len(k_s)]
            return k_s[starts], starts, ends, v_s

        def ground_medians(Lpts, LCpts):
            """Lidar ground height per cell, keyed by cell."""
            g = Lpts[LCpts == CLS_GROUND]
            gi = np.floor((g[:, :2] - lo) / cell).astype("i8")
            gk = gi[:, 0] * 100000 + gi[:, 1]
            ks, st, en, zs = _groups(gk, g[:, 2])
            out = {}
            for k, a, b in zip(ks, st, en):
                if b - a >= 2:
                    out[int(k)] = float(np.median(zs[a:b]))
            return out

        def ground_cells(Lpts, LCpts, Ppts):
            gmed = ground_medians(Lpts, LCpts)
            pi = np.floor((Ppts[:, :2] - lo) / cell).astype("i8")
            pk = pi[:, 0] * 100000 + pi[:, 1]
            ks, st, en, zs = _groups(pk, Ppts[:, 2])
            out = []
            for k, a, b in zip(ks, st, en):
                if b - a < self.a.reg_ground_min_photo:
                    continue
                gm = gmed.get(int(k))
                if gm is None:
                    continue
                out.append(gm - float(np.percentile(zs[a:b], 10)))
            return np.array(out)

        diffs = ground_cells(L, LC, P)
        if len(diffs) < 10:
            raise Fatal("could not find enough shared ground cells for a "
                        "coarse height offset")
        dz0 = float(np.median(diffs))
        self.log(f"register: coarse height offset from {len(diffs)} shared "
                 f"ground cells: {dz0:+.3f} m (16-84 pct "
                 f"{np.percentile(diffs,16):+.2f} .. {np.percentile(diffs,84):+.2f})")

        rng = np.random.default_rng(0)
        idx = rng.choice(len(L), size=min(self.a.icp_points, len(L)),
                         replace=False)
        Ls = L[idx]
        tree = cKDTree(P)
        gate = self.a.icp_max_pair_m
        R0, t0 = np.eye(3), np.array([0.0, 0.0, -dz0])
        before_raw = self._resid(L, tree, gate)
        before = self._resid(apply_rt(Ls, R0, t0), tree, gate)
        R, t, hist, _ = self._icp(Ls, tree, P, R0, t0, gate)
        after = self._resid(apply_rt(Ls, R, t), tree, gate)

        # 2. Does the fit actually pull from where it started, or is it just
        #    sitting where it was put?  Restart from four deliberately wrong
        #    offsets and see whether it lands back in the same place.  The
        #    spread of those endpoints is the honest uncertainty.
        probes = []
        for dx, dy, dz in ((1.0, 0, 0), (-1.0, 0, 0), (0, 1.0, 0), (0, 0, 0.5)):
            try:
                Rp, tp, _, _ = self._icp(
                    Ls, tree, P, R0, t0 + np.array([dx, dy, dz]), gate)
                probes.append({"start_offset_m": [dx, dy, dz],
                               "translation_m": [round(float(v), 4) for v in tp],
                               "yaw_deg": round(math.degrees(
                                   math.atan2(Rp[1, 0], Rp[0, 0])), 4),
                               "delta_from_main_m": round(float(
                                   np.linalg.norm(tp - t)), 4)})
            except Fatal as e:
                probes.append({"start_offset_m": [dx, dy, dz], "failed": str(e)})
        spread = [p["delta_from_main_m"] for p in probes if "delta_from_main_m" in p]
        repeatability = round(float(max(spread)), 4) if spread else None

        # 3. A check that does not reuse the ICP's own metric: vertical
        #    agreement on open ground, after the fit.
        Lw_all = apply_rt(L, R, t)
        resid_g = ground_cells(Lw_all, LC, P)
        ground_resid = {
            "cells": int(len(resid_g)),
            "median_m": round(float(np.median(resid_g)), 4),
            "mean_abs_m": round(float(np.mean(np.abs(resid_g))), 4),
            "p90_abs_m": round(float(np.percentile(np.abs(resid_g), 90)), 4),
            "within_0_5m_pct": round(100 * float((np.abs(resid_g) < 0.5).mean()), 1),
            "note": ("lidar ground height minus the photo cloud's 10th "
                     "percentile height, per shared cell"),
        }
        # 3b. The same check against the photo MESH rather than the point
        #     cloud.  The cloud has a tail of junk metres below the real
        #     surface, so a percentile of it flatters or punishes the fit for
        #     reasons that have nothing to do with registration; the mesh is
        #     the surface anyone will actually see.
        gmed = ground_medians(Lw_all, LC)
        pm = self._photo_mesh(trim_water=False)
        fc = pm.triangles_center
        fi = np.floor((fc[:, :2] - lo) / cell).astype("i8")
        fk = fi[:, 0] * 100000 + fi[:, 1]
        mesh_resid = []
        for k in np.unique(fk):
            if int(k) not in gmed:
                continue
            v = fc[fk == k, 2]
            if len(v) >= self.a.reg_mesh_min_faces:
                mesh_resid.append(gmed[int(k)] - float(np.percentile(v, 20)))
        mesh_ground = ({
            "cells": len(mesh_resid),
            "median_m": round(float(np.median(mesh_resid)), 4),
            "mean_abs_m": round(float(np.mean(np.abs(mesh_resid))), 4),
            "p90_abs_m": round(float(np.percentile(np.abs(mesh_resid), 90)), 4),
            "within_0_5m_pct": round(100 * float(
                (np.abs(np.array(mesh_resid)) < 0.5).mean()), 1),
            "note": ("lidar ground height minus the 20th percentile height of "
                     "the photo MESH faces in the same cell"),
        } if mesh_resid else None)

        yaw_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        reg = {
            "method": ("overlap restricted to cells where the photo cloud "
                       "really has geometry, coarse ground-height offset, then "
                       "trimmed ICP (yaw + translation) of lidar onto photo"),
            "overlap": {"cell_m": cell, "occupied_cells": int(occ.sum()),
                        "occupied_area_m2": round(float(occ.sum() * cell * cell)),
                        "lidar_points": int(len(L)), "photo_points": int(len(P))},
            "icp": {"points": int(len(idx)), "iterations": len(hist),
                    "trim_pct": self.a.icp_trim_pct, "gate_m": gate,
                    "history": hist[-6:]},
            "applied_transform": {
                "yaw_deg": round(yaw_deg, 4),
                "translation_m": [round(float(v), 4) for v in t],
                "horizontal_shift_m": round(float(np.hypot(t[0], t[1])), 4),
                "vertical_shift_m": round(float(t[2]), 4),
                "note": ("lidar_photo_frame = R(yaw) @ lidar_enu + t; lidar_enu "
                         "is the context manifest's metric_crs (UTM) + NAVD88 heights converted to the photo "
                         "build's ENU origin"),
            },
            "residual": {"before_any_correction": before_raw,
                         "after_coarse_height_only": before,
                         "after_icp": after,
                         "note": ("distance from each lidar point to the "
                                  "nearest photo point, over the shared cells "
                                  "only; the photo cloud is ~15 points/m2 so a "
                                  "perfect fit still leaves a few tenths of a "
                                  "metre, and roofs and canopy have no photo "
                                  "counterpart at all")},
            "restart_probes": probes,
            "repeatability_m": repeatability,
            "ground_height_residual_point_cloud": ground_resid,
            "ground_height_residual_photo_mesh": mesh_ground,
            "coarse_dz_m": round(dz0, 4),
        }
        (self.work / "registration.json").write_text(json.dumps(reg, indent=2))
        self.met("register").update(reg)
        self.d["R"], self.d["t"] = R, t
        self.d["lidar_reg"] = apply_rt(self.d["lidar"], R, t)
        np.save(self.work / "lidar_reg.npy", self.d["lidar_reg"].astype("f4"))
        self.log(f"register: yaw {yaw_deg:+.4f} deg, translation "
                 f"[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] m "
                 f"(horizontal {np.hypot(t[0], t[1]):.3f} m, vertical "
                 f"{t[2]:+.3f} m)")
        self.log(f"register: residual median {before_raw['median_m']} -> "
                 f"{after['median_m']} m, mean {before_raw['mean_m']} -> "
                 f"{after['mean_m']} m, overlap within {gate} m "
                 f"{before_raw['overlap_pct']}% -> {after['overlap_pct']}%")
        if mesh_ground:
            self.log(f"register: ground residual against the photo MESH over "
                     f"{mesh_ground['cells']} cells: median "
                     f"{mesh_ground['median_m']:+.3f} m, mean abs "
                     f"{mesh_ground['mean_abs_m']:.3f} m, "
                     f"{mesh_ground['within_0_5m_pct']}% within 0.5 m")
        self.log(f"register: ground height residual (point cloud) over {ground_resid['cells']} "
                 f"cells: median {ground_resid['median_m']:+.3f} m, mean abs "
                 f"{ground_resid['mean_abs_m']:.3f} m, "
                 f"{ground_resid['within_0_5m_pct']}% within 0.5 m")
        self.log(f"register: restarts from 1 m away land within "
                 f"{repeatability} m of the main solution")
        if after["median_m"] and before["median_m"] and \
                after["median_m"] > before["median_m"] * 1.02:
            raise Fatal("ICP made the fit worse; refusing to ship a "
                        "registration that is worse than doing nothing")

    def _registered(self):
        import numpy as np
        self._loaded()
        if "lidar_reg" in self.d:
            return self.d["lidar_reg"]
        f = self.work / "lidar_reg.npy"
        if not f.exists():
            raise Fatal("lidar_reg.npy missing; rerun with --force-from register")
        self.d["lidar_reg"] = np.load(f).astype("f8")
        return self.d["lidar_reg"]

    # ---- OSM in the photo frame ------------------------------------------- #

    def _osm(self) -> dict:
        """OSM features, reprojected into the photo cloud's ENU frame and
        grouped by layer as shapely geometries."""
        if "osm" in self.d:
            return self.d["osm"]
        self._loaded()
        from shapely.geometry import shape
        gj = json.loads((self.ctx / "osm_features.geojson").read_text())
        lat0, lon0, alt0 = self.d["origin"]

        def conv(c):
            if isinstance(c[0], (int, float)):
                e = lla_to_enu(c[1], c[0], alt0, lat0, lon0, alt0)
                return [float(e[0]), float(e[1])]
            return [conv(x) for x in c]

        groups: dict[str, list] = {}
        for f in gj["features"]:
            g = shape({"type": f["geometry"]["type"],
                       "coordinates": conv(f["geometry"]["coordinates"])})
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty:
                continue
            groups.setdefault(f["properties"]["layer"], []).append(
                (g, f["properties"]))
        self.d["osm"] = groups
        self.log("osm in photo frame: "
                 + ", ".join(f"{k}={len(v)}" for k, v in sorted(groups.items())))
        return groups

    def _union(self, layer: str, buffer_m: float = 0.0):
        from shapely.ops import unary_union
        geoms = [g for g, _ in self._osm().get(layer, [])
                 if g.geom_type in ("Polygon", "MultiPolygon")]
        if not geoms:
            return None
        u = unary_union(geoms)
        return u.buffer(buffer_m) if buffer_m else u

    def _photo_mesh(self, trim_water: bool = True):
        """The published photo mesh, rotated back out of glTF Y-up into ENU,
        with the water noise cut out.  Loaded here rather than in two places so
        the coverage mask and the exported scene can never disagree about what
        the photos actually cover."""
        import numpy as np
        import shapely
        import trimesh
        key = f"photo_mesh_{trim_water}"
        if key in self.d:
            return self.d[key]
        if getattr(self.a, "no_photo", False):
            m = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), "i8"), process=False)
            self.d[key] = m
            self.d["photo_faces_removed_over_water"] = 0
            return m
        m = trimesh.load(self.photo / "model.glb", force="mesh", process=False)
        back = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                        dtype=float)
        m.apply_transform(back)
        removed = 0
        if trim_water:
            wz = self.d.get("water_z")
            if wz is None:
                s = self._surfaces()
                wz = s["water_z"]
            wu = self._union("water")
            if wu is not None and np.isfinite(wz):
                c = m.triangles_center
                drop = (shapely.contains_xy(wu, c[:, 0], c[:, 1])
                        & (c[:, 2] < wz + self.a.water_trim_above_m))
                if drop.any():
                    m.update_faces(~drop)
                    m.remove_unreferenced_vertices()
                    removed = int(drop.sum())
        self.d[key] = m
        self.d["photo_faces_removed_over_water"] = removed
        return m

    # ---- stage: surfaces -------------------------------------------------- #

    def st_surfaces(self) -> None:
        """Every raster the layers are cut from: lidar terrain, lidar surface,
        where the photos actually have geometry, and where the map says water,
        buildings, grass and paths are."""
        import numpy as np
        import shapely
        from scipy import ndimage
        from scipy.stats import binned_statistic_2d
        lid = self._registered()
        cls = self.d["lidar_cls"]
        photo = self.d["photo"]
        cell = self.a.cell

        keep = ~np.isin(cls, [CLS_NOISE_LOW, CLS_NOISE_HIGH])
        L, LC = lid[keep], cls[keep]
        x0, y0 = np.floor(L[:, :2].min(axis=0) / cell) * cell
        x1, y1 = np.ceil(L[:, :2].max(axis=0) / cell) * cell
        nx, ny = int((x1 - x0) / cell), int((y1 - y0) / cell)
        xe = x0 + np.arange(nx + 1) * cell
        ye = y0 + np.arange(ny + 1) * cell
        self.log(f"surfaces: {nx} x {ny} cells of {cell} m "
                 f"({(x1-x0):.0f} x {(y1-y0):.0f} m)")

        def raster(pts, stat, vals=None):
            if len(pts) == 0:
                return np.full((ny, nx), np.nan)
            r = binned_statistic_2d(
                pts[:, 1], pts[:, 0],
                values=(pts[:, 2] if vals is None else vals),
                statistic=stat, bins=[ye, xe])
            return r.statistic

        ground = L[(LC == CLS_GROUND) | (LC == CLS_IGNORED_GROUND)]
        water_pts = L[LC == CLS_WATER]
        dtm_raw = raster(ground, "median")
        n_ground = raster(ground, "count")
        have = np.isfinite(dtm_raw)
        if have.sum() < 100:
            raise Fatal("almost no classified ground: cannot build a terrain")
        # Nearest-neighbour fill, and remember how far each filled cell is from
        # real data so invented terrain can be excluded later.
        dist, ind = ndimage.distance_transform_edt(
            ~have, return_distances=True, return_indices=True)
        dtm = dtm_raw[tuple(ind)]
        fill_dist_m = dist * cell
        dtm_s = ndimage.uniform_filter(dtm, size=3, mode="nearest")

        # Surface model: everything that is not ground, water or noise (trees, roofs,
        # cars, people) when the tile has no vegetation/building classes.
        above = L[~np.isin(LC, [CLS_GROUND, CLS_IGNORED_GROUND, CLS_WATER])]
        dsm = raster(above, "max")
        n_above = raster(above, "count")
        hag = dsm - dtm_s

        # Water level, straight off the lidar rather than guessed.
        if len(water_pts) >= 100:
            water_z = float(np.median(water_pts[:, 2]))
            water_std = float(np.std(water_pts[:, 2]))
        else:
            water_z, water_std = float("nan"), float("nan")

        # Where do the photos actually have geometry, below 4 m?  Measured on
        # the MESH, not on the point cloud: the cloud is wider and denser than
        # the surface that was actually reconstructed, and a mask built from it
        # switches the lidar terrain off in cells the mesh then fails to fill,
        # which is exactly how a fused model ends up with holes in it.
        pi = np.floor((photo[:, :2] - [x0, y0]) / cell).astype("i8")
        inb = ((pi[:, 0] >= 0) & (pi[:, 0] < nx) & (pi[:, 1] >= 0) & (pi[:, 1] < ny))
        pc = np.zeros((ny, nx), "i4")
        np.add.at(pc, (pi[inb, 1], pi[inb, 0]), 1)
        gz = dtm_s[pi[inb, 1], pi[inb, 0]]
        low = photo[inb, 2] < gz + self.a.photo_authority_m
        pc_low = np.zeros((ny, nx), "i4")
        np.add.at(pc_low, (pi[inb, 1][low], pi[inb, 0][low]), 1)
        cloud_cover = pc_low >= self.a.photo_min_points

        self.d["water_z"] = water_z
        pm = self._photo_mesh()
        fc = pm.triangles_center
        fa = pm.area_faces
        fi = np.floor((fc[:, :2] - [x0, y0]) / cell).astype("i8")
        finb = ((fi[:, 0] >= 0) & (fi[:, 0] < nx) & (fi[:, 1] >= 0) & (fi[:, 1] < ny))
        fz = dtm_s[fi[finb, 1], fi[finb, 0]]
        flow = fc[finb, 2] < fz + self.a.photo_authority_m
        face_area = np.zeros((ny, nx), "f8")
        np.add.at(face_area, (fi[finb, 1][flow], fi[finb, 0][flow]), fa[finb][flow])
        photo_cover = face_area >= self.a.photo_min_area_frac * cell * cell

        # Map masks, sampled at cell centres.
        cx = (xe[:-1] + cell / 2)[None, :].repeat(ny, axis=0)
        cy = (ye[:-1] + cell / 2)[:, None].repeat(nx, axis=1)

        def mask_of(layer, buf=0.0):
            u = self._union(layer, buf)
            if u is None:
                return np.zeros((ny, nx), bool)
            return shapely.contains_xy(u, cx.ravel(), cy.ravel()).reshape(ny, nx)

        # Water: OSM draws water bodies, but the lidar knows where water actually
        # returned.  Cells with class-9 returns outside every OSM water polygon
        # are water too, or they come out as holes.
        m_water_osm = mask_of("water")
        n_water = raster(water_pts, "count")
        m_water_lidar = np.nan_to_num(n_water, nan=0.0) >= self.a.water_min_points
        # OSM water polygons follow the legal shoreline, not the surveyed
        # waterline, and can cover solid ground above the waterline.  Believe
        # the lidar ground there, or the flat water plane gets drawn over real
        # ground and the photo mesh under it is cut.
        solid_ground = np.isfinite(dtm_raw) & (dtm_raw > water_z + 0.5)
        m_water = (m_water_osm | m_water_lidar) & ~solid_ground
        # Floating structures and the gaps between them return neither ground
        # nor water, so the raw mask has pinholes.  Close it, but only over
        # cells that have no ground return at all -- never over land.
        no_ground = ~np.isfinite(dtm_raw)
        closed = (ndimage.binary_fill_holes(m_water)
                  | ndimage.binary_closing(m_water, np.ones((5, 5), bool)))
        m_water_closed = m_water | (closed & no_ground)
        n_closed = int(m_water_closed.sum() - m_water.sum())
        m_water = m_water_closed
        # Two footprint masks: the plain one decides where the terrain stops
        # (a buffered one leaves a ring of holes the building mesh does not
        # cover), the buffered one keeps canopy patches off roof edges.
        m_build = mask_of("building")
        m_build_buf = mask_of("building", self.a.building_buffer_m)
        m_grass = mask_of("grass")
        m_beach = mask_of("beach")
        m_pier = mask_of("pier")
        paths = [g for g, _ in self._osm().get("path", [])]
        m_path = np.zeros((ny, nx), bool)
        if paths:
            from shapely.ops import unary_union
            pu = unary_union(paths).buffer(self.a.path_width_m / 2)
            m_path = shapely.contains_xy(pu, cx.ravel(), cy.ravel()).reshape(ny, nx)

        trees = [g for g, _ in self._osm().get("tree", [])]
        m_tree = np.zeros((ny, nx), bool)
        if trees:
            from shapely.ops import unary_union
            tu = unary_union([g.buffer(self.a.tree_radius_m) for g in trees])
            m_tree = shapely.contains_xy(tu, cx.ravel(), cy.ravel()).reshape(ny, nx)

        np.savez_compressed(
            self.work / "surfaces.npz", dtm=dtm_s.astype("f4"),
            dtm_raw=dtm_raw.astype("f4"), fill_dist_m=fill_dist_m.astype("f4"),
            dsm=dsm.astype("f4"), hag=hag.astype("f4"),
            n_ground=n_ground.astype("f4"), n_above=n_above.astype("f4"),
            photo_count=pc, photo_low_count=pc_low, photo_cover=photo_cover,
            photo_face_area=face_area.astype("f4"), cloud_cover=cloud_cover,
            m_water=m_water, m_water_osm=m_water_osm,
            m_water_lidar=m_water_lidar,
            m_build=m_build, m_build_buf=m_build_buf,
            m_grass=m_grass, m_path=m_path,
            m_tree=m_tree, m_beach=m_beach, m_pier=m_pier,
            grid=np.array([x0, y0, cell, nx, ny], dtype="f8"),
            water=np.array([water_z, water_std], dtype="f8"))

        area_cell = cell * cell
        self.met("surfaces").update({
            "cell_m": cell, "nx": nx, "ny": ny,
            "extent_m": [round(float(x1 - x0), 1), round(float(y1 - y0), 1)],
            "ground_cells_measured": int(have.sum()),
            "ground_cells_filled": int((~have).sum()),
            "ground_fill_max_m": round(float(fill_dist_m.max()), 1),
            "photo_cover_cells": int(photo_cover.sum()),
            "photo_cover_rule": (f"mesh face area >= "
                                 f"{self.a.photo_min_area_frac:.2f} of a cell, "
                                 f"below {self.a.photo_authority_m:.0f} m above "
                                 f"terrain"),
            "cloud_cover_cells": int(cloud_cover.sum()),
            "cells_cloud_but_no_mesh": int((cloud_cover & ~photo_cover).sum()),
            "photo_mesh_faces_after_water_trim": int(len(pm.faces)),
            "photo_mesh_surface_area_m2": round(float(pm.area)),
            "photo_mesh_footprint_cells": int((face_area > 0).sum()),
            "photo_cells_only_partly_filled": int(
                ((face_area >= self.a.photo_min_area_frac * cell * cell)
                 & (face_area < cell * cell)).sum()),
            "photo_faces_removed_over_water":
                int(self.d.get("photo_faces_removed_over_water", 0)),
            "photo_cover_pct_of_box": round(100 * photo_cover.sum() / (nx * ny), 1),
            "photo_cover_area_m2": round(float(photo_cover.sum() * area_cell), 0),
            "cells_water": int(m_water.sum()),
            "cells_water_osm_polygon": int(m_water_osm.sum()),
            "cells_water_lidar_returns": int(m_water_lidar.sum()),
            "cells_water_lidar_only": int((m_water_lidar & ~m_water_osm).sum()),
            "cells_water_dropped_solid_ground":
                int(((m_water_osm | m_water_lidar) & solid_ground).sum()),
            "cells_water_closed_over_no_return": n_closed,
            "cells_building": int(m_build.sum()),
            "cells_grass": int(m_grass.sum()), "cells_path": int(m_path.sum()),
            "water_level_m_enu": round(water_z, 3),
            "water_level_std_m": round(water_std, 3),
            "water_points": int(len(water_pts)),
            "hag_p99_m": round(float(np.nanpercentile(hag, 99)), 2),
        })
        self.log(f"surfaces: ground measured in {int(have.sum())} cells, "
                 f"{int((~have).sum())} filled (max {fill_dist_m.max():.0f} m "
                 f"from data); photos cover {photo_cover.sum()} cells "
                 f"({100*photo_cover.sum()/(nx*ny):.1f}% of the box); water "
                 f"level {water_z:.3f} m ENU from {len(water_pts):,} points")

    def _surfaces(self) -> dict:
        import numpy as np
        if "surf" in self.d:
            return self.d["surf"]
        f = self.work / "surfaces.npz"
        if not f.exists():
            raise Fatal("surfaces.npz missing; rerun with --force-from surfaces")
        z = dict(np.load(f))
        x0, y0, cell, nx, ny = z["grid"]
        z["x0"], z["y0"], z["cell"] = float(x0), float(y0), float(cell)
        z["nx"], z["ny"] = int(nx), int(ny)
        z["water_z"], z["water_std"] = float(z["water"][0]), float(z["water"][1])
        self.d["surf"] = z
        return z

    # ---- mesh helpers ----------------------------------------------------- #

    @staticmethod
    def _corners(cellvals):
        """Cell-centred heights -> corner heights, ignoring gaps."""
        import numpy as np
        ny, nx = cellvals.shape
        pad = np.full((ny + 2, nx + 2), np.nan)
        pad[1:-1, 1:-1] = cellvals
        st = np.stack([pad[:-1, :-1], pad[:-1, 1:], pad[1:, :-1], pad[1:, 1:]])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(st, axis=0)

    def _grid_mesh(self, mask, cellvals, colors, skirt_to=None):
        """Turn selected cells of a raster into a mesh.  Vertices are not
        shared between cells, so each cell keeps its own colour and a skirt can
        be dropped from any edge that borders an unselected cell."""
        import numpy as np
        s = self._surfaces()
        x0, y0, cell = s["x0"], s["y0"], s["cell"]
        cz = self._corners(cellvals)
        ii, jj = np.nonzero(mask)
        if len(ii) == 0:
            return None
        z00 = cz[ii, jj]; z10 = cz[ii, jj + 1]
        z11 = cz[ii + 1, jj + 1]; z01 = cz[ii + 1, jj]
        good = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z11) & np.isfinite(z01)
        ii, jj = ii[good], jj[good]
        z00, z10, z11, z01 = z00[good], z10[good], z11[good], z01[good]
        if len(ii) == 0:
            return None
        xl = x0 + jj * cell; xr = xl + cell
        yb = y0 + ii * cell; yt = yb + cell
        V = np.empty((len(ii) * 4, 3))
        V[0::4] = np.stack([xl, yb, z00], axis=1)
        V[1::4] = np.stack([xr, yb, z10], axis=1)
        V[2::4] = np.stack([xr, yt, z11], axis=1)
        V[3::4] = np.stack([xl, yt, z01], axis=1)
        b = np.arange(len(ii)) * 4
        F = np.concatenate([np.stack([b, b + 1, b + 2], axis=1),
                            np.stack([b, b + 2, b + 3], axis=1)])
        C = np.repeat(colors[ii, jj], 4, axis=0) if colors.ndim == 3 else \
            np.repeat(colors[None, :], len(ii) * 4, axis=0)

        if skirt_to is not None:
            m = np.zeros_like(mask)
            m[ii, jj] = True
            skv, skf, skc = [], [], []
            base = len(V)
            sz = self._corners(skirt_to)
            edges = [(-1, 0, (0, 1)), (1, 0, (3, 2)), (0, -1, (0, 3)), (0, 1, (1, 2))]
            for di, dj, (ca, cb) in edges:
                ni, nj = ii + di, jj + dj
                inb = (ni >= 0) & (ni < m.shape[0]) & (nj >= 0) & (nj < m.shape[1])
                open_edge = ~inb
                open_edge[inb] = ~m[ni[inb], nj[inb]]
                if not open_edge.any():
                    continue
                k = np.nonzero(open_edge)[0]
                top = V[np.stack([b[k] + ca, b[k] + cb], axis=1)]        # (n,2,3)
                bot = top.copy()
                ci = [(ii[k], jj[k]), (ii[k], jj[k] + 1), (ii[k] + 1, jj[k] + 1),
                      (ii[k] + 1, jj[k])]
                bot[:, 0, 2] = sz[ci[ca][0], ci[ca][1]]
                bot[:, 1, 2] = sz[ci[cb][0], ci[cb][1]]
                bad = ~np.isfinite(bot[:, :, 2]).all(axis=1)
                bot[bad, :, 2] = np.minimum(top[bad, :, 2], 0.0)
                start = base + sum(len(x) for x in skv)
                quad = np.concatenate([top, bot[:, ::-1]], axis=1).reshape(-1, 3)
                q = start + np.arange(len(k)) * 4
                skv.append(quad)
                skf.append(np.concatenate([np.stack([q, q + 1, q + 2], axis=1),
                                           np.stack([q, q + 2, q + 3], axis=1)]))
                skc.append(np.repeat(
                    (colors[ii[k], jj[k]] if colors.ndim == 3
                     else np.repeat(colors[None, :], len(k), axis=0)), 4, axis=0))
            if skv:
                V = np.concatenate([V] + skv)
                F = np.concatenate([F] + skf)
                C = np.concatenate([C] + skc)
        return V, F, C.astype("u1")

    @staticmethod
    def _mesh(V, F, C, name):
        import numpy as np
        import trimesh
        m = trimesh.Trimesh(vertices=np.asarray(V, dtype="f8"),
                            faces=np.asarray(F, dtype="i8"),
                            vertex_colors=np.asarray(C, dtype="u1"),
                            process=False)
        m.metadata["name"] = name
        return m

    # ---- stage: layers ---------------------------------------------------- #

    COL = {"grass": (86, 118, 68), "path": (146, 142, 136),
           "beach": (196, 181, 146), "pier": (120, 100, 78),
           "bare": (112, 104, 90), "water": (44, 72, 96),
           "canopy": (58, 92, 52), "structure": (132, 124, 112),
           "roof": (168, 163, 155), "wall": (186, 181, 172)}

    def st_layers(self) -> None:
        import numpy as np
        import shapely
        import trimesh
        s = self._surfaces()
        lid = self._registered()
        cls = self.d["lidar_cls"]
        photo = self.d["photo"]
        cell, x0, y0 = s["cell"], s["x0"], s["y0"]
        dtm, dsm, hag = s["dtm"], s["dsm"], s["hag"]
        ny, nx = dtm.shape
        ldir = self.work / "layers"
        ldir.mkdir(exist_ok=True)
        made = {}

        # ---------- terrain: ground where the photos gave nothing ----------- #
        terrain_mask = (~s["photo_cover"] & ~s["m_water"] & ~s["m_build"]
                        & (s["fill_dist_m"] <= self.a.max_ground_fill_m))
        cols = np.zeros((ny, nx, 3), "u1")
        cols[:] = self.COL["bare"]
        cols[s["m_grass"]] = self.COL["grass"]
        cols[s["m_beach"]] = self.COL["beach"]
        cols[s["m_pier"]] = self.COL["pier"]
        cols[s["m_path"]] = self.COL["path"]
        got = self._grid_mesh(terrain_mask, dtm, cols)
        if got is None:
            raise Fatal("terrain layer came out empty; the photo coverage mask "
                        "or the ground classification is wrong")
        tm = self._mesh(*got, "terrain_lidar_ground")
        tm.export(ldir / "terrain.ply")
        made["terrain_lidar_ground"] = {
            "cells": int(terrain_mask.sum()),
            "area_m2": round(float(terrain_mask.sum() * cell * cell), 0),
            "faces": int(len(tm.faces)),
            "source": "USGS 3DEP lidar, ASPRS class 2 ground",
            "method": (f"median ground height per {cell:.0f} m cell, gaps filled "
                       f"from the nearest measured cell up to "
                       f"{self.a.max_ground_fill_m:.0f} m, 3x3 smoothing; only "
                       f"emitted where the photo cloud has fewer than "
                       f"{self.a.photo_min_points} points below "
                       f"{self.a.photo_authority_m:.0f} m"),
            "colour": "inferred from OSM grass / path / beach / pier polygons",
            "licence": "3DEP public domain; colours from OSM, ODbL",
        }

        # ---------- water: flat, at the measured level ---------------------- #
        wz = s["water_z"]
        if np.isfinite(wz) and s["m_water"].any():
            wvals = np.full((ny, nx), np.nan)
            wvals[s["m_water"]] = wz
            got = self._grid_mesh(s["m_water"], wvals,
                                  np.array(self.COL["water"], "u1"))
            wm = self._mesh(*got, "water_lidar_level")
            wm.export(ldir / "water.ply")
            made["water_lidar_level"] = {
                "cells": int(s["m_water"].sum()),
                "area_m2": round(float(s["m_water"].sum() * cell * cell), 0),
                "faces": int(len(wm.faces)),
                "level_m_enu": round(float(wz), 3),
                "level_std_m": round(float(s["water_std"]), 3),
                "points_measured": int((cls == CLS_WATER).sum()),
                "source": "USGS 3DEP lidar class 9 water returns, plus OSM "
                          "water polygons where the lidar returned nothing",
                "cells_from_lidar_returns":
                    int(s["m_water_lidar"].sum()),
                "cells_from_osm_only":
                    int((s["m_water_osm"] & ~s["m_water_lidar"]).sum()),
                "method": ("flat plane at the median lidar water height, cut to "
                           "the OSM water polygon; photo faces inside that "
                           "polygon below the waterline are removed in assemble"),
                "licence": "3DEP public domain; outline from OSM, ODbL",
            }
        else:
            self.log("no water in this box (no class 9 points or no OSM water)")

        # ---------- canopy and roofs: what photos never see ----------------- #
        canopy_mask = ((hag >= self.a.canopy_min_m) & (s["n_above"] >= 2)
                       & ~s["m_build_buf"] & ~s["m_water"] & np.isfinite(dsm))
        ccol = np.zeros((ny, nx, 3), "u1")
        ccol[:] = self.COL["structure"]
        veg = s["m_tree"] | s["m_grass"]
        ccol[veg] = self.COL["canopy"]
        got = self._grid_mesh(canopy_mask, dsm, ccol, skirt_to=dtm)
        if got is None:
            raise Fatal("canopy/roof layer came out empty")
        cm = self._mesh(*got, "canopy_roofs_lidar")
        cm.export(ldir / "canopy.ply")
        heights = hag[canopy_mask]
        made["canopy_roofs_lidar"] = {
            "cells": int(canopy_mask.sum()),
            "area_m2": round(float(canopy_mask.sum() * cell * cell), 0),
            "faces": int(len(cm.faces)),
            "height_above_ground_m": {
                "median": round(float(np.median(heights)), 2),
                "p90": round(float(np.percentile(heights, 90)), 2),
                "max": round(float(heights.max()), 2)},
            "near_mapped_tree_cells": int((canopy_mask & s["m_tree"]).sum()),
            "source": "USGS 3DEP lidar, unclassified returns above ground",
            "method": (f"highest return per {cell:.0f} m cell where it stands "
                       f"{self.a.canopy_min_m:.1f} m or more above the terrain "
                       f"and is outside every OSM building footprint, with a "
                       f"skirt dropped to the terrain; trees and unmapped "
                       f"structures are separated by height, not by class label"),
            "colour": "green where OSM maps a tree or grass, grey otherwise "
                      "(inferred, not measured)",
            "licence": "3DEP public domain",
        }

        # ---------- buildings: OSM footprints, lidar roof heights ----------- #
        bmeshes, brecs = [], []
        skipped = {"no_height": 0, "photo_already_covers": 0, "tiny": 0}
        for g, props in self._osm().get("building", []):
            polys = ([g] if g.geom_type == "Polygon"
                     else list(g.geoms) if g.geom_type == "MultiPolygon" else [])
            for poly in polys:
                if poly.area < 8.0:
                    skipped["tiny"] += 1
                    continue
                xs, ys = np.array(poly.exterior.coords).T
                if (xs.max() < x0 or xs.min() > x0 + nx * cell
                        or ys.max() < y0 or ys.min() > y0 + ny * cell):
                    continue
                inside = shapely.contains_xy(poly, lid[:, 0], lid[:, 1])
                cin = shapely.contains_xy(
                    poly, (x0 + (np.arange(nx) + .5) * cell)[None, :].repeat(ny, 0).ravel(),
                    (y0 + (np.arange(ny) + .5) * cell)[:, None].repeat(nx, 1).ravel()
                ).reshape(ny, nx)
                base = (float(np.nanmedian(dtm[cin])) if cin.any()
                        else float(np.nanmedian(dtm)))
                lpts = lid[inside]
                roof_src, roof_z = None, None
                if len(lpts) >= self.a.building_min_lidar:
                    hi = lpts[lpts[:, 2] > base + 2.0]
                    if len(hi) >= self.a.building_min_lidar:
                        roof_z = float(np.percentile(hi[:, 2], 75))
                        roof_src = "lidar roof returns (75th percentile)"
                if roof_z is None and props.get("height_m_est"):
                    roof_z = base + float(props["height_m_est"])
                    roof_src = props.get("height_source") or "osm height estimate"
                if roof_z is None or roof_z - base < 2.0:
                    skipped["no_height"] += 1
                    continue
                # photos own the bottom: start the massing where they stop
                pin = shapely.contains_xy(poly, photo[:, 0], photo[:, 1])
                ptop = None
                if pin.sum() >= self.a.building_photo_points:
                    ptop = float(np.percentile(photo[pin, 2], 90))
                bottom = base
                photo_cov = False
                if ptop is not None and ptop > base + 1.0:
                    bottom = max(base, min(ptop - 1.0, roof_z - 1.0))
                    photo_cov = True
                if roof_z - bottom < 0.75:
                    skipped["photo_already_covers"] += 1
                    continue
                V, F, C = [], [], []
                ring = np.array(poly.exterior.coords)[:-1]
                n = len(ring)
                wall = np.array(self.COL["wall"], "u1")
                for k in range(n):
                    p, q = ring[k], ring[(k + 1) % n]
                    b0 = len(V)
                    V += [[p[0], p[1], bottom], [q[0], q[1], bottom],
                          [q[0], q[1], roof_z], [p[0], p[1], roof_z]]
                    F += [[b0, b0 + 1, b0 + 2], [b0, b0 + 2, b0 + 3]]
                    C += [wall] * 4
                cv, cf = trimesh.creation.triangulate_polygon(poly, engine="earcut")
                b0 = len(V)
                V += [[v[0], v[1], roof_z] for v in cv]
                F += (np.asarray(cf) + b0).tolist()
                C += [np.array(self.COL["roof"], "u1")] * len(cv)
                bm = self._mesh(V, F, C, "b")
                bmeshes.append(bm)
                brecs.append({
                    "osm_id": props.get("osm_id"), "name": props.get("name"),
                    "footprint_m2": round(float(poly.area), 1),
                    "base_m": round(base, 2), "roof_m": round(roof_z, 2),
                    "height_m": round(roof_z - base, 2),
                    "roof_source": roof_src,
                    "osm_levels": props.get("levels"),
                    "osm_height_tag": props.get("height_tag"),
                    "massing_starts_at_m": round(bottom, 2),
                    "photo_covers_base": photo_cov,
                    "lidar_points_inside": int(len(lpts)),
                })
        if bmeshes:
            bmesh = trimesh.util.concatenate(bmeshes)
            bmesh.metadata["name"] = "buildings_osm_lidar"
            bmesh.export(ldir / "buildings.ply")
            (self.work / "buildings.json").write_text(json.dumps(brecs, indent=2))
            from_lidar = sum(1 for b in brecs if "lidar" in (b["roof_source"] or ""))
            made["buildings_osm_lidar"] = {
                "buildings": len(brecs), "faces": int(len(bmesh.faces)),
                "roof_height_from_lidar": from_lidar,
                "roof_height_from_osm_levels": len(brecs) - from_lidar,
                "with_photographed_base": sum(1 for b in brecs
                                              if b["photo_covers_base"]),
                "skipped": skipped,
                "median_height_m": round(float(np.median(
                    [b["height_m"] for b in brecs])), 2),
                "source": "OSM building footprints (ODbL); heights measured "
                          "from 3DEP lidar returns inside each footprint, "
                          "falling back to OSM building:levels x 3.2 m + 1 m",
                "method": ("each footprint extruded from the point the photo "
                           "cloud stops (or the terrain, where no photo saw it) "
                           "up to the roof height, with a flat roof cap; "
                           "photographed facades below that point are left to "
                           "the photo mesh"),
                "licence": "OSM ODbL for the footprints and level counts; "
                           "3DEP public domain for the heights",
            }
        else:
            self.log("WARNING: no building massing produced")

        (self.work / "layers.json").write_text(json.dumps(made, indent=2))
        self.met("layers").update(made)
        for k, v in made.items():
            self.log(f"layer {k}: " + json.dumps(
                {kk: vv for kk, vv in v.items()
                 if kk in ("cells", "faces", "area_m2", "buildings",
                           "roof_height_from_lidar", "level_m_enu")}))

    # ---- stage: assemble -------------------------------------------------- #

    def st_assemble(self) -> None:
        import numpy as np
        import trimesh
        from PIL import Image
        self._surfaces()
        ldir = self.work / "layers"

        # photogrammetry over water is noise, so the water faces are cut
        # and the measured flat surface takes over (done in _photo_mesh, the
        # same call the coverage mask used)
        photo_mesh = self._photo_mesh()
        removed = int(self.d.get("photo_faces_removed_over_water", 0))
        self.log(f"assemble: photo mesh {len(photo_mesh.faces):,} faces after "
                 f"removing {removed:,} over water")

        scene = trimesh.Scene()
        scene.add_geometry(photo_mesh, geom_name="photo_mapillary_mvs",
                           node_name="photo_mapillary_mvs")
        parts = {"photo_mapillary_mvs": int(len(photo_mesh.faces))}
        for fn, nm in (("terrain.ply", "terrain_lidar_ground"),
                       ("water.ply", "water_lidar_level"),
                       ("canopy.ply", "canopy_roofs_lidar"),
                       ("buildings.ply", "buildings_osm_lidar")):
            p = ldir / fn
            if not p.exists():
                continue
            m = trimesh.load(p, process=False)
            scene.add_geometry(m, geom_name=nm, node_name=nm)
            parts[nm] = int(len(m.faces))

        # glTF is Y-up: (x, y, z) -> (x, z, -y)
        yup = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                       dtype=float)
        for g in scene.geometry.values():
            g.apply_transform(yup)

        glb = self.work / "model.glb"
        scene.export(glb)
        self.log(f"assemble: model.glb {glb.stat().st_size/1e6:.1f} MB, parts "
                 + json.dumps(parts))

        # ---- browser-sized copy: atlas capped, geometry decimated ---------- #
        atlas_before = None
        vis = getattr(photo_mesh, "visual", None)
        img = getattr(getattr(vis, "material", None), "baseColorTexture", None)
        if img is not None:
            atlas_before = list(img.size)
            if img.size[0] > MAX_ATLAS_W:
                h = max(1, int(round(img.size[1] * MAX_ATLAS_W / img.size[0])))
                small = img.resize((MAX_ATLAS_W, h), Image.LANCZOS)
                photo_mesh.visual.material.baseColorTexture = small
                self.log(f"web: atlas {img.size[0]}x{img.size[1]} -> "
                         f"{MAX_ATLAS_W}x{h} (wider than {MAX_ATLAS_W} px "
                         f"is over many WebGL viewers' max texture size)")
        web = self.work / "model_web.glb"
        scene.export(web)
        if GLTF_BIN.exists():
            tmp = self.work / "web_tmp.glb"
            r = sh([str(GLTF_BIN), "prune", str(web), str(tmp)], cwd=str(GLTF_DIR))
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1000:
                tmp.replace(web)
            else:
                self.log(f"gltf-transform prune skipped: {r.stderr.strip()[:200]}")

        # ---- verify what was actually written ------------------------------ #
        checks = {}
        for tag, f in (("model", glb), ("model_web", web)):
            sc = trimesh.load(f)
            geoms = {k: int(len(v.faces)) for k, v in sc.geometry.items()}
            tex = None
            for v in sc.geometry.values():
                im = getattr(getattr(getattr(v, "visual", None), "material", None),
                             "baseColorTexture", None)
                if im is not None:
                    tex = list(im.size)
                    break
            ext = [round(float(x), 2) for x in sc.extents]
            checks[tag] = {"bytes": f.stat().st_size, "geometries": geoms,
                           "triangles": int(sum(geoms.values())),
                           "atlas_px": tex, "extents_m": ext}
            if sum(geoms.values()) < 1000:
                raise Fatal(f"{f.name} has only {sum(geoms.values())} triangles")
            if max(ext) > 2000 or max(ext) < 50:
                raise Fatal(f"{f.name} extents {ext} m are outside the 50-2000 m scene range")
        if checks["model"]["atlas_px"] is None and not getattr(self.a, "no_photo", False):
            raise Fatal("the fused model lost the photo texture atlas")
        if checks["model_web"]["atlas_px"] and checks["model_web"]["atlas_px"][0] > MAX_ATLAS_W:
            raise Fatal(f"model_web.glb atlas is "
                        f"{checks['model_web']['atlas_px'][0]} px wide; many "
                        f"WebGL viewers silently drop textures above {MAX_ATLAS_W} px")
        checks["atlas_before_web_resize_px"] = atlas_before
        checks["photo_faces_removed_over_water"] = removed
        self.met("assemble").update({"parts": parts, **checks})

    # ---- stage: render ---------------------------------------------------- #

    def st_render(self) -> None:
        import numpy as np
        import trimesh
        from PIL import Image
        rdir = self.work / "renders"
        rdir.mkdir(exist_ok=True)
        self._coverage_png(rdir.parent / "coverage_map.png")
        made = []
        try:
            import pyrender
            sc = trimesh.load(self.work / "model.glb")
            bounds = sc.bounds
            centre = bounds.mean(axis=0)
            radius = float(np.linalg.norm(bounds[1] - bounds[0]) / 2.0)
            scene = pyrender.Scene(bg_color=[0.09, 0.10, 0.12, 1.0],
                                   ambient_light=[1.0, 1.0, 1.0])
            for g in sc.geometry.values():
                g = g.copy()
                g.apply_translation(-centre)
                scene.add(pyrender.Mesh.from_trimesh(g, smooth=False))
            cam = pyrender.PerspectiveCamera(yfov=math.radians(50))
            w, h = self.a.render_size, int(self.a.render_size * 0.75)
            r = pyrender.OffscreenRenderer(w, h)

            def pose(az, el, dist):
                az, el = math.radians(az), math.radians(el)
                eye = np.array([dist * math.cos(el) * math.sin(az),
                                dist * math.sin(el),
                                dist * math.cos(el) * math.cos(az)])
                fwd = -eye / np.linalg.norm(eye)
                up = np.array([0.0, 1.0, 0.0])
                if abs(float(np.dot(fwd, up))) > 0.99:
                    up = np.array([0.0, 0.0, 1.0])
                right = np.cross(fwd, up); right /= np.linalg.norm(right)
                tu = np.cross(right, fwd)
                m = np.eye(4)
                m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, tu, -fwd, eye
                return m

            # A close view of the block the photos actually cover, which is
            # the only place the two sources meet and therefore the only place
            # a misregistration would show.
            pm = self._photo_mesh()
            pc = pm.bounds.mean(axis=0)
            pc_yup = np.array([pc[0], pc[2], -pc[1]]) - centre
            prad = float(np.linalg.norm(pm.extents) / 2.0)
            for tag, az, el, dist, focus in (
                    ("closeup_photo_block", 35, 45, prad * 1.1, pc_yup),
                    ("closeup_photo_top", 0, 70, prad * 1.3, pc_yup)):
                p = pose(az, el, dist)
                p[:3, 3] += focus
                nc = scene.add(cam, pose=p)
                nl = scene.add(pyrender.DirectionalLight(color=[1, 1, 1],
                                                         intensity=1.5), pose=p)
                colour, _ = r.render(scene)
                scene.remove_node(nc); scene.remove_node(nl)
                Image.fromarray(colour).save(rdir / f"{tag}.png")
                made.append(f"{tag}.png")

            for tag, az, el, d in (("orbit_000", 0, 25, 2.1),
                                   ("orbit_090", 90, 25, 2.1),
                                   ("orbit_180", 180, 25, 2.1),
                                   ("orbit_270", 270, 25, 2.1),
                                   ("high_045", 45, 65, 2.0),
                                   ("low_135", 135, 8, 2.1),
                                   ("top_down", 0, 89, 1.6)):
                p = pose(az, el, radius * d)
                nc = scene.add(cam, pose=p)
                nl = scene.add(pyrender.DirectionalLight(color=[1, 1, 1],
                                                         intensity=1.5), pose=p)
                colour, _ = r.render(scene)
                scene.remove_node(nc); scene.remove_node(nl)
                Image.fromarray(colour).save(rdir / f"{tag}.png")
                made.append(f"{tag}.png")
            r.delete()
        except Exception as e:                                   # noqa: BLE001
            self.log(f"WARNING: 3D renders failed ({type(e).__name__}: {e}); "
                     f"the coverage map is still written, but this build has "
                     f"not been visually checked in 3D")
            self.status.setdefault("degradations", []).append(
                f"renders failed: {type(e).__name__}: {e}")
        if made:
            self.log(f"render: {len(made)} views at {self.a.render_size}px")
        self.met("render").update({"views": made})

    def _coverage_png(self, path: Path) -> None:
        """Top-down provenance map: which square metre came from what."""
        import numpy as np
        from PIL import Image
        s = self._surfaces()
        ny, nx = s["dtm"].shape
        img = np.zeros((ny, nx, 3), "u1")
        img[:] = (24, 24, 28)
        terrain = (~s["photo_cover"] & ~s["m_water"] & ~s["m_build"]
                   & (s["fill_dist_m"] <= self.a.max_ground_fill_m))
        canopy = ((s["hag"] >= self.a.canopy_min_m) & (s["n_above"] >= 2)
                  & ~s["m_build_buf"] & ~s["m_water"] & np.isfinite(s["dsm"]))
        img[terrain] = (196, 128, 48)          # lidar ground
        img[s["m_water"]] = (44, 92, 140)      # measured water plane
        img[canopy] = (48, 124, 64)            # lidar canopy / roofs
        img[s["m_build"]] = (150, 150, 158)    # osm building massing
        img[s["photo_cover"]] = (64, 196, 180)  # photographed
        img[s["photo_cover"] & canopy] = (120, 210, 120)
        img = np.flipud(img)
        scale = max(1, int(900 / max(nx, ny)))
        im = Image.fromarray(img).resize((nx * scale, ny * scale), Image.NEAREST)
        im.save(path)
        self.log(f"coverage map: {path.name} "
                 f"(teal photographed, orange lidar ground, green canopy, "
                 f"grey OSM massing, blue water)")

    # ---- stage: publish --------------------------------------------------- #

    def st_publish(self) -> None:
        import numpy as np
        self._surfaces()
        lid = self._registered()
        cls = self.d["lidar_cls"]
        self.out.mkdir(parents=True, exist_ok=True)
        for f in ("model.glb", "model_web.glb", "coverage_map.png"):
            p = self.work / f
            if p.exists():
                shutil.copy2(p, self.out / f)
        rdir = self.out / "renders"
        rdir.mkdir(exist_ok=True)
        for p in sorted((self.work / "renders").glob("*.png")):
            shutil.copy2(p, rdir / p.name)
        for f in ("registration.json", "buildings.json", "fuse.log"):
            p = self.work / f
            if p.exists():
                shutil.copy2(p, self.out / f)

        # the registered lidar, in the photo frame, so the next tool does not
        # have to redo any of this
        arr = np.zeros(len(lid), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                        ("classification", "u1")])
        arr["x"], arr["y"], arr["z"] = lid[:, 0], lid[:, 1], lid[:, 2]
        arr["classification"] = cls.astype("u1")
        write_ply_points(self.out / "points_lidar_enu.ply", arr, comments=[
            "USGS 3DEP lidar registered into the photo reconstruction's ENU frame",
            "classification is ASPRS: 2 ground, 7 low noise, 9 water, 18 high noise",
        ])

        cman = json.loads((self.ctx / "context_manifest.json").read_text())
        pman = json.loads((self.photo / "manifest.json").read_text())
        reg = json.loads((self.work / "registration.json").read_text())
        layers = json.loads((self.work / "layers.json").read_text())
        surf = self.met("surfaces")
        cellA = surf["cell_m"] ** 2
        total_cells = surf["nx"] * surf["ny"]
        photo_cells = surf["photo_cover_cells"]
        manifest = {
            "area": self.name,
            "generated": now(),
            "tool": "fuse_context.py",
            "what_this_is": (
                "Street-level photos of the area (a reconstruction, or "
                "their poses only) fused with public airborne lidar and OpenStreetMap. Photos own everything they "
                "saw below about 4 m; lidar owns the terrain they missed, the "
                "tree tops and roofs they could never see, and the water "
                "surface; OSM footprints give building massing where only one "
                "facade was photographed."),
            "crs": {
                "type": "ENU metres",
                "origin_lat_lon_alt": pman["crs"]["origin_lat_lon_alt"],
                "note": ("same local East-North-Up frame as the photo build; "
                         "model.glb is rotated to glTF Y-up (x, z, -y), the "
                         "point cloud stays in ENU"),
            },
            "registration": reg,
            "inputs": {
                "photo_reconstruction": {
                    "dir": str(self.photo), "tool": pman.get("tool"),
                    "points": self.met("load")["photo_points"],
                    "georeference_error_m": pman.get("georeference_error_m"),
                    "licence": "Mapillary imagery CC BY-SA 4.0",
                    "attribution": pman.get("attribution"),
                },
                "lidar": {
                    "dataset": cman["lidar"]["dataset"],
                    "survey_year": cman["lidar"]["survey_year"],
                    "points_in_box": cman["lidar"]["points"],
                    "density_pts_per_m2": cman["lidar"]["density_pts_per_m2"],
                    "source_crs": cman["lidar"]["source_crs"],
                    "classes": cman["lidar"]["classes"],
                    "licence": "USGS 3DEP: public domain, no use restrictions",
                },
                "osm": {**{k: v for k, v in cman["osm"].items() if k != "files"},
                        "licence": "(c) OpenStreetMap contributors, ODbL 1.0"},
            },
            "provenance": {
                "rule": ("photo mesh is authoritative below "
                         f"{self.a.photo_authority_m:.0f} m wherever it has "
                         "geometry; lidar is authoritative overhead and "
                         "wherever the photos have none; OSM is used for "
                         "outlines, building massing and for colouring "
                         "lidar-only ground"),
                "layers": layers,
                "coverage_cells": {
                    "cell_m": surf["cell_m"], "total": total_cells,
                    "photographed": photo_cells,
                    "photographed_pct": round(100 * photo_cells / total_cells, 1),
                    "photographed_area_m2": round(photo_cells * cellA),
                    "lidar_only_ground_cells":
                        layers.get("terrain_lidar_ground", {}).get("cells", 0),
                    "lidar_canopy_roof_cells":
                        layers.get("canopy_roofs_lidar", {}).get("cells", 0),
                    "osm_massing_cells": surf["cells_building"],
                    "water_cells": surf["cells_water"],
                },
                "coverage_map": "coverage_map.png",
                "map_legend": {
                    "teal": "photographed (photo mesh)",
                    "orange": "lidar ground, no photo coverage",
                    "green": "lidar canopy / roof tops",
                    "grey": "OSM building massing",
                    "blue": "water plane at the lidar-measured level",
                },
            },
            "metrics": self.status["metrics"],
            "files": {
                "model": "model.glb",
                "model_web": "model_web.glb",
                "lidar_points": "points_lidar_enu.ply",
                "coverage_map": "coverage_map.png",
                "registration": "registration.json",
                "buildings": "buildings.json",
                "renders": sorted(p.name for p in rdir.glob("*.png")),
            },
            "licences": {
                "lidar": "USGS 3DEP -- public domain, no use restrictions",
                "map_data": "(c) OpenStreetMap contributors, ODbL 1.0",
                "imagery": "Mapillary street-level imagery, CC BY-SA 4.0 -- "
                           "derivatives must credit the creators and share alike",
            },
        }
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        missing = [f for f in ("model.glb", "model_web.glb", "manifest.json",
                               "points_lidar_enu.ply", "coverage_map.png")
                   if not (self.out / f).exists()]
        if missing:
            raise Fatal(f"publish left files missing: {missing}")
        self.log(f"published to {self.out}: "
                 + ", ".join(sorted(p.name for p in self.out.iterdir())))
        self.status["outputs"] = {"dir": str(self.out)}

    # ---- driver ----------------------------------------------------------- #

    def run(self) -> bool:
        self.work.mkdir(parents=True, exist_ok=True)
        self.log(f"=== fuse {self.name} ===")
        try:
            self.preflight()
            if self.a.force_from:
                if self.a.force_from not in STAGES:
                    raise Fatal(f"--force-from {self.a.force_from} is not a stage")
                for st in STAGES[STAGES.index(self.a.force_from):]:
                    self.marker(st).unlink(missing_ok=True)
            for stage in STAGES:
                if self.marker(stage).exists():
                    self.log(f"skip {stage} (done)")
                    continue
                self.status["stage"] = stage
                self.status["state"] = "running"
                self.save_status()
                t0 = time.time()
                self.log(f"--- {stage} ---")
                getattr(self, f"st_{stage}")()
                self.mark_done(stage, time.time() - t0)
                self.log(f"--- {stage} done in {time.time()-t0:.0f}s ---")
                if self.a.stop_after == stage:
                    self.log(f"stopping after {stage} as asked")
                    break
        except Fatal as e:
            self.status["state"] = "failed"
            self.status["error"] = str(e)
            self.save_status()
            self.log(f"FATAL: {e}")
            return False
        except Exception as e:                                   # noqa: BLE001
            import traceback
            self.status["state"] = "failed"
            self.status["error"] = f"{type(e).__name__}: {e}"
            self.save_status()
            self.log("FATAL (unexpected):\n" + traceback.format_exc())
            return False
        self.status["state"] = "done"
        self.save_status()
        try:
            shutil.copy2(self.statusfile, self.out / "status.json")
        except OSError:
            pass
        self.log("=== done ===")
        return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fuse public lidar + OSM with a photo reconstruction.")
    ap.add_argument("name", help="output area name, e.g. sf_block")
    ap.add_argument("--photo-dir", required=True,
                    help="a photogrammetry output dir holding points_enu.ply, model.glb and "
                         "manifest.json (external; no tool here writes it); with --no-photo, "
                         "a dir holding only manifest.json")
    ap.add_argument("--no-photo", action="store_true",
                    help="no photo reconstruction: --photo-dir holds only manifest.json "
                         "(ENU origin); lidar + OSM own every surface")
    ap.add_argument("--context-dir", default=None,
                    help="fetch_context.py output (default _source/<name>/context)")
    ap.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--src-root", default=str(DEFAULT_SRC_ROOT))
    ap.add_argument("--cell", type=float, default=2.0,
                    help="raster cell size in metres (default 2.0)")
    ap.add_argument("--photo-authority-m", type=float, default=4.0,
                    help="height below which photo geometry wins (default 4)")
    ap.add_argument("--photo-min-points", type=int, default=3,
                    help="photo points in a cell before the cloud counts as there")
    ap.add_argument("--photo-min-area-frac", type=float, default=0.25,
                    help="fraction of a cell the photo MESH must fill for the "
                         "photo to own that cell (default 0.25)")
    ap.add_argument("--canopy-min-m", type=float, default=2.5,
                    help="height above ground for a cell to be canopy/roof")
    ap.add_argument("--max-ground-fill-m", type=float, default=8.0,
                    help="how far lidar ground may be extrapolated")
    ap.add_argument("--water-min-points", type=int, default=2,
                    help="class-9 lidar returns in a cell to call it water")
    ap.add_argument("--water-trim-above-m", type=float, default=2.0,
                    help="photo faces this far above the waterline are cut too")
    ap.add_argument("--building-buffer-m", type=float, default=1.0)
    ap.add_argument("--building-min-lidar", type=int, default=25)
    ap.add_argument("--building-photo-points", type=int, default=200)
    ap.add_argument("--path-width-m", type=float, default=3.0)
    ap.add_argument("--tree-radius-m", type=float, default=4.0)
    ap.add_argument("--overlap-inset", type=float, default=5.0)
    ap.add_argument("--reg-cell", type=float, default=2.0,
                    help="cell size for the photo-occupancy overlap mask")
    ap.add_argument("--reg-min-photo", type=int, default=5,
                    help="photo points in a cell before it counts as shared")
    ap.add_argument("--reg-ground-min-photo", type=int, default=20)
    ap.add_argument("--reg-mesh-min-faces", type=int, default=20)
    ap.add_argument("--icp-points", type=int, default=150000)
    ap.add_argument("--icp-iters", type=int, default=40)
    ap.add_argument("--icp-trim-pct", type=float, default=60.0)
    ap.add_argument("--icp-max-pair-m", type=float, default=3.0)
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--render-size", type=int, default=1200)
    ap.add_argument("--force-from", default=None)
    ap.add_argument("--stop-after", default=None)
    a = ap.parse_args()
    return 0 if Fuse(a.name, a).run() else 1


if __name__ == "__main__":
    sys.exit(main())
