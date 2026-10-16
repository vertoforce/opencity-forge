#!/usr/bin/env python3
"""
fetch_context.py -- public lidar + OpenStreetMap context for a bounding box.

Sibling of fetch_area.py.  Where that one fetches Mapillary
street-level photos, this one pulls the two public datasets that
fill the holes photos always leave: airborne lidar (roofs, tree tops, water,
lawn interiors) and OSM vector features (building footprints with level
counts, trees, grass, water, paths).

    .venv/bin/python fetch_context.py sf_block=-122.4010,37.7920,-122.4000,37.7928

No region is hardcoded.  The lidar dataset is discovered by
asking OpenTopography's catalog which point-cloud surveys cover the box, and
the USGS 3DEP public EPT index as a fallback, then picking the newest one that
actually has a live EPT endpoint covering the box.

Conventions:
  * resumable stages, one marker file per finished stage
  * loud failure: a stage that cannot do its job raises, it never degrades
  * every stage output is verified before its marker is written
  * source data under OPENCITY_DATA, heavy scratch under OPENCITY_BUILD
  * preflight checks disk, RAM, docker image and network before doing work
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent

PDAL_IMAGE = ("pdal/pdal:2.9.0@sha256:"
              "eb636adfd471721339d1f7793bd5dd78e267fed505c50c7d58b193817817f975")

# Host roots: env-overridable, same names and defaults as opencity_paths.py in the
# pipeline (OPENCITY_DATA, OPENCITY_BUILD).
_DATA = Path(os.environ.get("OPENCITY_DATA") or Path(__file__).resolve().parents[1] / "data")
_BUILD = Path(os.environ.get("OPENCITY_BUILD") or Path(__file__).resolve().parents[1] / "work" / "build")
DEFAULT_WORK_ROOT = _BUILD / "context"
DEFAULT_SRC_ROOT = _DATA / "_source"

# Public service endpoints.  All are open APIs whose terms permit
# programmatic access; none of them is scraped out of an HTML page.
OT_CATALOG = "https://portal.opentopography.org/API/otCatalog"
USGS_EPT_BASE = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
USGS_EPT_INDEX = ("https://raw.githubusercontent.com/hobuinc/usgs-lidar/"
                  "master/boundaries/resources.geojson")
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = ("https://overpass.private.coffee/api/interpreter",
                    "https://maps.mail.ru/osm/tools/overpass/api/interpreter")

# A real user agent, so the operators can identify this tool and throttle or
# block it specifically rather than blocking python-requests as a class.
# OPENCITY_USER_AGENT overrides it (opencity_paths.USER_AGENT); name yourself or your project.
UA = os.environ.get("OPENCITY_USER_AGENT") or "opencity-forge/1.0 (python-requests)"

# Politeness: minimum seconds between requests to the same host (1 s for a host
# not listed).  Overpass asks for gaps between queries; this module sends one
# query per area per run, plus one per mirror it falls back to.
MIN_GAP_S = {"overpass-api.de": 4.0, "portal.opentopography.org": 2.0,
             "raw.githubusercontent.com": 1.0, "s3-us-west-2.amazonaws.com": 0.5}
RETRY_STATUS = (429, 502, 503, 504)

STAGES = ["discover", "lidar", "osm", "verify"]

DISK_MARGIN_GB = 2.0
RAM_HEADROOM_GB = 12.0

# Classification codes used downstream (ASPRS LAS 1.4).
ASPRS = {0: "never classified", 1: "unclassified", 2: "ground",
         3: "low vegetation", 4: "medium vegetation", 5: "high vegetation",
         6: "building", 7: "low point (noise)", 9: "water",
         10: "rail", 11: "road surface", 13: "wire guard", 14: "wire conductor",
         15: "transmission tower", 17: "bridge deck", 18: "high noise",
         20: "ignored ground"}


class Fatal(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# small helpers, shared with fuse_context.py                                   #
# --------------------------------------------------------------------------- #

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def free_disk_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 1e9


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def utm_epsg(lon: float, lat: float) -> int:
    """The metric CRS to measure in.  Web Mercator (what most public EPT
    indexes are published in) is inflated by 1/cos(lat) (1.41x at 45 deg latitude)
    so every measurement has to happen somewhere else."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


PLY_NP = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
          "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
          "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
          "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}


def ply_header(path: Path) -> tuple[dict, int]:
    """Parse a PLY header.  Returns (header dict, byte offset of the data)."""
    elements: list[dict] = []
    fmt = None
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise Fatal(f"{path} is not a PLY file")
        while True:
            raw = f.readline()
            if not raw:
                raise Fatal(f"{path}: PLY header never ended")
            line = raw.decode("ascii", "replace").strip()
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                elements.append({"name": tok[1], "count": int(tok[2]), "props": []})
            elif tok[0] == "property":
                if tok[1] == "list":
                    elements[-1]["props"].append(
                        {"name": tok[4], "list": True, "count_type": tok[2],
                         "type": tok[3]})
                else:
                    elements[-1]["props"].append(
                        {"name": tok[2], "list": False, "type": tok[1]})
            elif tok[0] == "end_header":
                return {"format": fmt, "elements": elements}, f.tell()


def read_ply_points(path: Path):
    """Read the vertex element of a PLY into a numpy structured array.
    Handles binary_little_endian and ascii, vertex-only files."""
    import numpy as np
    hdr, off = ply_header(path)
    vert = next((e for e in hdr["elements"] if e["name"] == "vertex"), None)
    if vert is None:
        raise Fatal(f"{path}: no vertex element")
    if any(p["list"] for p in vert["props"]):
        raise Fatal(f"{path}: list properties in vertex element are not supported")
    dtype = np.dtype([(p["name"], PLY_NP[p["type"]]) for p in vert["props"]])
    if hdr["format"] == "binary_little_endian":
        arr = np.fromfile(path, dtype=dtype, count=vert["count"], offset=off)
    elif hdr["format"] == "ascii":
        with open(path) as f:
            f.seek(off)
            arr = np.loadtxt(f, dtype=dtype, max_rows=vert["count"])
    else:
        raise Fatal(f"{path}: unsupported PLY format {hdr['format']}")
    if len(arr) != vert["count"]:
        raise Fatal(f"{path}: header says {vert['count']} vertices, read {len(arr)}")
    return arr


# PDAL writes PLY property names lowercased ("classification"), while PDAL
# pipelines, LAS and everything downstream use the ASPRS CamelCase names.
# Normalise on read so the rest of the code only ever sees one spelling.
LIDAR_ALIASES = {"x": "X", "y": "Y", "z": "Z", "intensity": "Intensity",
                 "returnnumber": "ReturnNumber",
                 "numberofreturns": "NumberOfReturns",
                 "classification": "Classification",
                 "red": "Red", "green": "Green", "blue": "Blue"}


def read_lidar_ply(path: Path):
    arr = read_ply_points(path)
    arr.dtype.names = tuple(LIDAR_ALIASES.get(n.lower(), n)
                            for n in arr.dtype.names)
    for need in ("X", "Y", "Z"):
        if need not in arr.dtype.names:
            raise Fatal(f"{path}: no {need} column (has {arr.dtype.names})")
    return arr


def write_ply_points(path: Path, arr, comments: Iterable[str] = ()) -> None:
    import numpy as np
    inv = {np.dtype(v).str[1:]: k for k, v in PLY_NP.items()}
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {c}" for c in comments]
    lines.append(f"element vertex {len(arr)}")
    for name in arr.dtype.names:
        lines.append(f"property {inv[arr.dtype[name].str[1:]]} {name}")
    lines.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        arr.tofile(f)


# --------------------------------------------------------------------------- #
# polite HTTP                                                                  #
# --------------------------------------------------------------------------- #

class Http:
    """One session, a real user agent, a per-host gap and backoff on the
    status codes that mean 'you are asking too fast'."""

    def __init__(self, log=print):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.last: dict[str, float] = {}
        self.log = log
        self.count = 0

    def _wait(self, url: str) -> None:
        host = url.split("/")[2]
        gap = MIN_GAP_S.get(host, 1.0)
        prev = self.last.get(host)
        if prev is not None:
            sleep = gap - (time.time() - prev)
            if sleep > 0:
                time.sleep(sleep)
        self.last[host] = time.time()

    def get(self, url: str, params: dict | None = None, timeout: int = 120,
            tries: int = 4, data: dict | None = None):
        for attempt in range(1, tries + 1):
            self._wait(url)
            self.count += 1
            try:
                if data is not None:
                    r = self.s.post(url, data=data, timeout=timeout)
                else:
                    r = self.s.get(url, params=params, timeout=timeout)
            except Exception as e:                       # network-level failure
                if attempt == tries:
                    raise Fatal(f"GET {url} failed after {tries} tries: {e}")
                self.log(f"  http error ({e}); retry {attempt}/{tries}")
                time.sleep(5 * attempt)
                continue
            if r.status_code in RETRY_STATUS:
                if attempt == tries:
                    raise Fatal(f"{url} returned {r.status_code} on every try; "
                                f"the service is rate limiting or down -- stop "
                                f"and try later rather than hammering it")
                back = 15 * attempt
                self.log(f"  {r.status_code} from {url.split('/')[2]}; "
                         f"backing off {back}s ({attempt}/{tries})")
                time.sleep(back)
                continue
            if r.status_code != 200:
                raise Fatal(f"{url} returned HTTP {r.status_code}")
            return r
        raise Fatal("unreachable")


# --------------------------------------------------------------------------- #
# the fetcher                                                                  #
# --------------------------------------------------------------------------- #

class ContextFetch:
    def __init__(self, name: str, bbox: tuple[float, float, float, float],
                 args: argparse.Namespace):
        self.name = name
        self.bbox = bbox                       # w, s, e, n in WGS84 degrees
        self.a = args
        self.work = Path(args.work_root) / name
        self.src = Path(args.src_root) / name / "context"
        self.markers = self.work / ".stages"
        self.logfile = self.work / "fetch_context.log"
        self.statusfile = self.work / "status.json"
        self.status: dict[str, Any] = {
            "area": name, "bbox": list(bbox), "pid": os.getpid(),
            "started": now(), "updated": now(), "state": "starting",
            "stage": None, "stages": {}, "metrics": {}, "error": None}
        if self.statusfile.exists():
            try:
                old = json.loads(self.statusfile.read_text())
                self.status["metrics"] = old.get("metrics", {})
                self.status["stages"] = old.get("stages", {})
            except (json.JSONDecodeError, OSError):
                pass
        self.epsg = args.epsg or utm_epsg((bbox[0] + bbox[2]) / 2,
                                          (bbox[1] + bbox[3]) / 2)
        self.http: Http | None = None

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

    def is_done(self, stage: str) -> bool:
        return self.marker(stage).exists()

    def mark_done(self, stage: str, secs: float, **extra) -> None:
        self.markers.mkdir(parents=True, exist_ok=True)
        self.marker(stage).write_text(now())
        self.status["stages"][stage] = {"done": now(), "seconds": round(secs, 1),
                                        **extra}
        self.save_status()

    def metric_bbox_m(self) -> tuple[float, float, float, float]:
        """The bbox in the metric CRS, as (xmin, ymin, xmax, ymax)."""
        from pyproj import Transformer
        tr = Transformer.from_crs(4326, self.epsg, always_xy=True)
        w, s, e, n = self.bbox
        xs, ys = [], []
        for lon in (w, (w + e) / 2, e):
            for lat in (s, (s + n) / 2, n):
                x, y = tr.transform(lon, lat)
                xs.append(x)
                ys.append(y)
        return min(xs), min(ys), max(xs), max(ys)

    def area_m2(self) -> float:
        x0, y0, x1, y1 = self.metric_bbox_m()
        return (x1 - x0) * (y1 - y0)

    # ---- preflight -------------------------------------------------------- #

    def preflight(self) -> None:
        if sh(["docker", "image", "inspect", PDAL_IMAGE]).returncode != 0:
            self.log(f"pulling {PDAL_IMAGE.split('@')[0]} (pinned by digest)")
            p = sh(["docker", "pull", PDAL_IMAGE])
            if p.returncode != 0:
                raise Fatal(f"docker image {PDAL_IMAGE} not present and pull "
                            f"failed: {p.stderr.strip()[:300]}")
        have = free_disk_gb(Path(self.a.work_root))
        if have < DISK_MARGIN_GB:
            raise Fatal(f"only {have:.1f} GB free on {self.a.work_root}, "
                        f"need {DISK_MARGIN_GB} GB of headroom")
        ram = mem_available_gb()
        if ram < RAM_HEADROOM_GB:
            raise Fatal(f"only {ram:.1f} GB RAM available; this tool needs "
                        f"{RAM_HEADROOM_GB:.0f} GB free before it starts")
        try:
            self.src.mkdir(parents=True, exist_ok=True)
            probe = self.src / ".writetest"
            probe.write_bytes(b"x")
            probe.unlink()
        except OSError as e:
            raise Fatal(f"source root {self.src} is not writable: {e}")
        w, s, e, n = self.bbox
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise Fatal(f"bbox {self.bbox} is not a valid w,s,e,n box")
        self.http = Http(log=self.log)
        self.log(f"preflight ok: disk {have:.1f} GB free, ram {ram:.1f} GB "
                 f"available, metric CRS EPSG:{self.epsg}, "
                 f"bbox area {self.area_m2()/1e6:.3f} km2")
        self.status["metrics"]["preflight"] = {
            "disk_free_gb": round(have, 1), "ram_available_gb": round(ram, 1),
            "metric_epsg": self.epsg, "bbox_area_km2": round(self.area_m2()/1e6, 4)}

    # ---- stage: discover -------------------------------------------------- #

    def _ot_catalog(self) -> list[dict]:
        """Ask OpenTopography which point-cloud datasets cover the box."""
        w, s, e, n = self.bbox
        cache = self.src / "opentopography_catalog.json"
        if cache.exists() and not self.a.refresh:
            doc = json.loads(cache.read_text())
            self.log(f"catalog: cached ({len(doc.get('Datasets', []))} datasets)")
        else:
            r = self.http.get(OT_CATALOG, params={
                "productFormat": "PointCloud", "minx": w, "miny": s,
                "maxx": e, "maxy": n, "detail": "false",
                "outputFormat": "json", "include_federated": "true"})
            doc = r.json()
            cache.write_text(json.dumps(doc, indent=1))
            self.log(f"catalog: {len(doc.get('Datasets', []))} datasets cover "
                     f"this box (OpenTopography)")
        out = []
        for d in doc.get("Datasets", []):
            ds = d.get("Dataset", {})
            ident = ds.get("identifier", {}) or {}
            out.append({
                "name": ds.get("name"),
                "id_kind": ident.get("propertyID"),
                "id": ident.get("value"),
                "date_created": ds.get("dateCreated"),
                "temporal": ds.get("temporalCoverage"),
                "source": "opentopography",
            })
        return out

    def _usgs_index(self) -> list[dict]:
        """Fallback / complement: the USGS 3DEP EPT resource index, which is a
        geojson of every public EPT with its footprint.  Used when the catalog
        returns nothing usable, so this tool still works where OpenTopography
        has no record."""
        cache = self.src / "usgs_ept_index.geojson"
        if not cache.exists() or self.a.refresh:
            r = self.http.get(USGS_EPT_INDEX, timeout=180)
            cache.write_bytes(r.content)
        from shapely.geometry import shape, box
        doc = json.loads(cache.read_text())
        want = box(*self.bbox)
        hits = []
        for feat in doc.get("features", []):
            try:
                geom = shape(feat["geometry"])
            except Exception:
                continue
            if not geom.intersects(want):
                continue
            props = feat.get("properties", {})
            hits.append({"name": props.get("name"), "id_kind": "USGS_3DEP_ID",
                         "id": props.get("name"), "date_created": None,
                         "temporal": None, "source": "usgs_ept_index",
                         "index_points": props.get("count"),
                         "url": props.get("url")})
        self.log(f"usgs ept index: {len(hits)} datasets intersect this box")
        return hits

    @staticmethod
    def _year(ds: dict) -> int:
        """Newest-first ranking key.  Survey dates beat publication dates; a
        name like <STATE>_<Project>_<n>_<year> is the last resort."""
        def years_in(text: str) -> list[int]:
            digits = "".join(c if c.isdigit() else " " for c in str(text))
            return [int(t) for t in digits.split()
                    if len(t) == 4 and 1990 <= int(t) <= 2100]

        for field in ("temporal", "date_created"):
            if ds.get(field):
                got = years_in(ds[field])
                if got:
                    return max(got)
        got = years_in(ds.get("id") or ds.get("name") or "")
        return max(got) if got else 0

    def _probe_ept(self, ds: dict) -> dict | None:
        """Confirm the dataset has a live EPT index that really covers the box."""
        url = ds.get("url") or f"{USGS_EPT_BASE}/{ds['id']}/ept.json"
        try:
            r = self.http.get(url, timeout=60, tries=2)
            ept = r.json()
        except Fatal:
            return None
        srs = (ept.get("srs") or {})
        code = srs.get("horizontal") or ""
        auth = srs.get("authority") or "EPSG"
        if not code:
            wkt = srs.get("wkt", "")
            code = "3857" if "Pseudo-Mercator" in wkt else ""
        if not code:
            return None
        ept_epsg = f"{auth}:{code}"
        # does its footprint really contain the requested box?
        from pyproj import Transformer
        try:
            tr = Transformer.from_crs(4326, int(code), always_xy=True)
        except (ValueError, TypeError):
            return None
        w, s, e, n = self.bbox
        corners = [tr.transform(lon, lat) for lon in (w, e) for lat in (s, n)]
        bc = ept.get("boundsConforming") or ept.get("bounds")
        inside = all(bc[0] <= x <= bc[3] and bc[1] <= y <= bc[4]
                     for x, y in corners)
        if not inside:
            return None
        return {**ds, "ept_url": url, "ept_epsg": ept_epsg,
                "ept_points": ept.get("points"),
                "ept_dims": [s_["name"] for s_ in ept.get("schema", [])],
                "year": self._year(ds)}

    def st_discover(self) -> None:
        cands = self._ot_catalog()
        usable = [c for c in cands if c.get("id_kind") == "USGS_3DEP_ID" and c.get("id")]
        self.log(f"discover: {len(usable)} of {len(cands)} catalog datasets are "
                 f"USGS 3DEP (which is the family with public EPT endpoints)")
        # Both indexes, always.  OpenTopography carries the survey dates but is
        # not complete; the USGS EPT index is complete but dateless.  Merging
        # them means "newest" is chosen over everything that exists, and the
        # tool still works where one of the two has no record.
        by_id = {c["id"]: c for c in self._usgs_index() if c.get("id")}
        for c in usable:
            prev = by_id.get(c["id"], {})
            by_id[c["id"]] = {**prev, **{k: v for k, v in c.items() if v is not None}}
        usable = list(by_id.values())
        self.log(f"discover: {len(usable)} distinct datasets after merging the "
                 f"OpenTopography catalog with the USGS EPT index")
        if not usable:
            raise Fatal(
                f"no public point-cloud dataset covers {self.bbox}. "
                f"OpenTopography returned {len(cands)} datasets and the USGS "
                f"EPT index returned none that intersect.")
        usable.sort(key=self._year, reverse=True)
        probed, chosen = [], None
        for ds in usable[: self.a.max_probe]:
            got = self._probe_ept(ds)
            state = "live, covers box" if got else "no live EPT covering the box"
            self.log(f"  {self._year(ds)}  {ds['id']:<28} {state}")
            if got:
                probed.append(got)
                if chosen is None:
                    chosen = got
        if chosen is None:
            raise Fatal(
                f"{len(usable)} datasets are listed over this box but none has "
                f"a live EPT index covering it; checked the newest "
                f"{min(len(usable), self.a.max_probe)}.")
        if "Classification" not in (chosen.get("ept_dims") or []):
            self.log("WARNING: chosen dataset carries no Classification "
                     "dimension; ground/water separation will be geometric only")
        doc = {"bbox_wsen": list(self.bbox), "metric_epsg": self.epsg,
               "queried": now(), "candidates": usable, "probed": probed,
               "chosen": chosen,
               "rule": "newest survey year with a live EPT index covering the box"}
        (self.src / "lidar_catalog.json").write_text(json.dumps(doc, indent=2))
        self.status["metrics"]["discover"] = {
            "candidates": len(usable), "probed": len(probed),
            "chosen": chosen["id"], "chosen_year": chosen["year"],
            "ept_epsg": chosen["ept_epsg"], "ept_points": chosen["ept_points"]}
        self.log(f"discover: chose {chosen['id']} ({chosen['year']}), "
                 f"{chosen['ept_points']:,} points, {chosen['ept_epsg']}")

    # ---- stage: lidar ----------------------------------------------------- #

    def _chosen(self) -> dict:
        f = self.src / "lidar_catalog.json"
        if not f.exists():
            raise Fatal("lidar_catalog.json missing; run the discover stage")
        return json.loads(f.read_text())["chosen"]

    def pdal(self, pipeline: dict, tag: str, extra_mounts: list[str] | None = None,
             timeout: int = 3600) -> dict:
        """Run one PDAL pipeline in the pinned container, CPU only, with the
        metadata written out so the result can be verified rather than assumed."""
        self.work.mkdir(parents=True, exist_ok=True)
        pfile = self.work / f"pipeline_{tag}.json"
        mfile = self.work / f"metadata_{tag}.json"
        pfile.write_text(json.dumps(pipeline, indent=1))
        if mfile.exists():
            mfile.unlink()
        cmd = ["docker", "run", "--rm", "--network", "host",
               "--cpus", str(self.a.cpus), "--memory", f"{self.a.mem_gb}g",
               "-v", f"{self.work}:/work",
               "-v", f"{self.src}:/src",
               *(extra_mounts or []),
               PDAL_IMAGE, "pdal", "pipeline", f"/work/{pfile.name}",
               "--metadata", f"/work/{mfile.name}"]
        t0 = time.time()
        self.log(f"pdal[{tag}]: running (cpus={self.a.cpus}, "
                 f"mem={self.a.mem_gb}g)")
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise Fatal(f"pdal pipeline '{tag}' failed ({p.returncode}): "
                        f"{(p.stderr or p.stdout).strip()[-800:]}")
        self.log(f"pdal[{tag}]: ok in {time.time()-t0:.0f}s")
        if not mfile.exists():
            raise Fatal(f"pdal pipeline '{tag}' wrote no metadata")
        return json.loads(mfile.read_text())

    def st_lidar(self) -> None:
        import numpy as np
        from pyproj import Transformer
        chosen = self._chosen()
        ept_code = int(chosen["ept_epsg"].split(":")[1])
        w, s, e, n = self.bbox

        # readers.ept wants the box in the EPT's own CRS ...
        tr_in = Transformer.from_crs(4326, ept_code, always_xy=True)
        pts = [tr_in.transform(lon, lat) for lon in (w, e) for lat in (s, n)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = self.a.pad_m
        ept_bounds = (f"([{min(xs)-pad*2},{max(xs)+pad*2}],"
                      f"[{min(ys)-pad*2},{max(ys)+pad*2}])")
        # ... and the crop happens again after reprojection, in metres, because
        # Web Mercator is 1/cos(lat) too large (about 1.4x at mid-US latitudes) so a box that looks
        # right in 3857 is not the box you asked for.
        x0, y0, x1, y1 = self.metric_bbox_m()
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad

        dims = "X,Y,Z,Intensity,ReturnNumber,NumberOfReturns,Classification"
        pipeline = {"pipeline": [
            {"type": "readers.ept", "filename": chosen["ept_url"],
             "bounds": ept_bounds, "threads": max(2, self.a.cpus)},
            {"type": "filters.reprojection",
             "in_srs": chosen["ept_epsg"], "out_srs": f"EPSG:{self.epsg}"},
            {"type": "filters.crop",
             "bounds": f"([{x0},{x1}],[{y0},{y1}])"},
            {"type": "filters.stats", "dimensions": "X,Y,Z,Classification",
             "enumerate": "Classification"},
            {"type": "writers.las", "filename": "/src/lidar_points.laz",
             "compression": "true", "a_srs": f"EPSG:{self.epsg}",
             "minor_version": "4"},
            {"type": "writers.ply", "filename": "/src/lidar_points.ply",
             "storage_mode": "little endian", "dims": dims},
        ]}
        meta = self.pdal(pipeline, "crop", timeout=self.a.lidar_timeout)

        ply = self.src / "lidar_points.ply"
        laz = self.src / "lidar_points.laz"
        for f in (ply, laz):
            if not f.exists() or f.stat().st_size < 1000:
                raise Fatal(f"lidar stage produced no usable {f.name}")
        arr = read_lidar_ply(ply)
        if len(arr) < 100:
            raise Fatal(f"only {len(arr)} lidar points in this box; the dataset "
                        f"is listed as covering it but has no data here")
        area = (x1 - x0) * (y1 - y0)
        cls, counts = np.unique(arr["Classification"], return_counts=True)
        by_class = {ASPRS.get(int(c), f"class {int(c)}"): int(k)
                    for c, k in zip(cls, counts)}
        z = arr["Z"].astype("f8")
        ground = arr[arr["Classification"] == 2]
        water = arr[arr["Classification"] == 9]
        stats = {
            "dataset": chosen["id"], "survey_year": chosen["year"],
            "ept_epsg": chosen["ept_epsg"], "metric_epsg": self.epsg,
            "points": int(len(arr)),
            "crop_bounds_m": [x0, y0, x1, y1],
            "area_m2": round(area, 1),
            "density_pts_per_m2": round(len(arr) / area, 3),
            "z_min": round(float(z.min()), 3), "z_max": round(float(z.max()), 3),
            "classes": by_class,
            "class_fraction_pct": {k: round(100 * v / len(arr), 1)
                                   for k, v in by_class.items()},
            "ground": ({"n": int(len(ground)),
                        "z_min": round(float(ground["Z"].min()), 3),
                        "z_max": round(float(ground["Z"].max()), 3),
                        "z_mean": round(float(ground["Z"].mean()), 3),
                        "z_std": round(float(ground["Z"].std()), 3)}
                       if len(ground) else None),
            "water": ({"n": int(len(water)),
                       "z_mean": round(float(water["Z"].mean()), 3),
                       "z_std": round(float(water["Z"].std()), 3),
                       "z_median": round(float(np.median(water["Z"])), 3)}
                      if len(water) else None),
            "has_vegetation_class": bool({3, 4, 5} & set(int(c) for c in cls)),
            "has_building_class": bool(6 in set(int(c) for c in cls)),
            "pdal_stats": meta.get("stages", {}).get("filters.stats", {}),
        }
        (self.src / "lidar_stats.json").write_text(json.dumps(stats, indent=2))
        self.status["metrics"]["lidar"] = {
            k: stats[k] for k in ("points", "density_pts_per_m2", "classes",
                                  "ground", "water", "has_vegetation_class",
                                  "has_building_class")}
        self.log(f"lidar: {len(arr):,} points, "
                 f"{stats['density_pts_per_m2']} pts/m2, classes "
                 f"{json.dumps(stats['class_fraction_pct'])}")
        if not stats["has_vegetation_class"] or not stats["has_building_class"]:
            self.log("note: this tile does not classify "
                     + (" or ".join(x for x, ok in
                                    (("vegetation", stats["has_vegetation_class"]),
                                     ("buildings", stats["has_building_class"]))
                                    if not ok))
                     + "; separate them with height-above-ground plus the OSM "
                       "footprints instead of trusting Classification")

    # ---- stage: osm ------------------------------------------------------- #

    OVERPASS_QL = """[out:json][timeout:{timeout}];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
  way["building:part"]({s},{w},{n},{e});
  node["natural"="tree"]({s},{w},{n},{e});
  way["natural"="tree_row"]({s},{w},{n},{e});
  way["landuse"="grass"]({s},{w},{n},{e});
  way["leisure"="park"]({s},{w},{n},{e});
  way["leisure"="pitch"]({s},{w},{n},{e});
  way["leisure"="playground"]({s},{w},{n},{e});
  way["natural"="water"]({s},{w},{n},{e});
  relation["natural"="water"]({s},{w},{n},{e});
  way["natural"="beach"]({s},{w},{n},{e});
  way["natural"="scrub"]({s},{w},{n},{e});
  way["waterway"="dock"]({s},{w},{n},{e});
  way["man_made"="pier"]({s},{w},{n},{e});
  way["highway"~"^(footway|path|steps|pedestrian|cycleway|living_street|service|residential)$"]({s},{w},{n},{e});
);
out body geom;
"""

    # which tag decides the layer a feature lands in, in priority order
    LAYERS = [
        ("building", lambda t: "building" in t or "building:part" in t),
        ("tree", lambda t: t.get("natural") == "tree"),
        ("tree_row", lambda t: t.get("natural") == "tree_row"),
        ("water", lambda t: t.get("natural") == "water"
                            or t.get("waterway") == "dock"),
        ("beach", lambda t: t.get("natural") == "beach"),
        ("grass", lambda t: t.get("landuse") == "grass"
                            or t.get("leisure") in ("park", "pitch")
                            or t.get("natural") == "scrub"),
        ("playground", lambda t: t.get("leisure") == "playground"),
        ("pier", lambda t: t.get("man_made") == "pier"),
        ("path", lambda t: "highway" in t),
    ]

    @staticmethod
    def _levels(tags: dict) -> float | None:
        """Metres of building massing, preferring a surveyed height."""
        for key, scale in (("height", 1.0), ("building:height", 1.0)):
            v = tags.get(key)
            if v:
                try:
                    return float(str(v).split()[0])
                except ValueError:
                    pass
        for key in ("building:levels", "levels"):
            v = tags.get(key)
            if v:
                try:
                    lv = float(str(v).split(";")[0])
                except ValueError:
                    continue
                # 3.2 m per storey plus a 1.0 m parapet/roof allowance: the
                # usual OSM rule of thumb, and it is an estimate, flagged as
                # such in the manifest.
                return lv * 3.2 + 1.0
        return None

    def st_osm(self) -> None:
        w, s, e, n = self.bbox
        raw_f = self.src / "osm_raw.json"
        if raw_f.exists() and not self.a.refresh:
            doc = json.loads(raw_f.read_text())
            self.log(f"osm: cached ({len(doc.get('elements', []))} elements); "
                     f"--refresh to refetch")
        else:
            ql = self.OVERPASS_QL.format(timeout=self.a.overpass_timeout,
                                         s=s, w=w, n=n, e=e)
            self.log("osm: one Overpass query for the whole box "
                     "(buildings, trees, grass, water, paths)")
            # public mirrors after the main instance: overpass-api.de can time out or
            # answer non-JSON under load.
            doc, last = None, None
            for url in (OVERPASS,) + OVERPASS_MIRRORS:
                try:
                    r = self.http.get(url, data={"data": ql},
                                      timeout=self.a.overpass_timeout + 60)
                    doc = r.json()
                    break
                except Exception as ex:          # Fatal after its own retries, or bad JSON
                    last = ex
                    self.log(f"osm: {url.split('/')[2]} failed ({str(ex)[:120]}); next mirror")
            if doc is None:
                raise Fatal(f"Overpass failed on every mirror: {last}")
            raw_f.write_text(json.dumps(doc, indent=1))
        els = doc.get("elements", [])
        if not els:
            raise Fatal("Overpass returned zero elements for this box; either "
                        "OSM has no features here or the query is wrong")

        feats, counts = [], {}
        for el in els:
            tags = el.get("tags") or {}
            if not tags:
                continue
            layer = next((nm for nm, test in self.LAYERS if test(tags)), None)
            if layer is None:
                continue
            geom = None
            if el["type"] == "node":
                geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
            elif el["type"] == "way" and el.get("geometry"):
                ring = [[p["lon"], p["lat"]] for p in el["geometry"]]
                closed = len(ring) > 3 and ring[0] == ring[-1]
                geom = ({"type": "Polygon", "coordinates": [ring]} if closed
                        else {"type": "LineString", "coordinates": ring})
            elif el["type"] == "relation":
                polys = []
                for m in el.get("members", []):
                    if m.get("role") not in ("outer", "") or not m.get("geometry"):
                        continue
                    ring = [[p["lon"], p["lat"]] for p in m["geometry"]]
                    if len(ring) > 3:
                        if ring[0] != ring[-1]:
                            ring.append(ring[0])
                        polys.append([ring])
                if polys:
                    geom = {"type": "MultiPolygon", "coordinates": polys}
            if geom is None:
                continue
            props = {"osm_type": el["type"], "osm_id": el["id"],
                     "layer": layer, "name": tags.get("name"),
                     "tags": tags}
            if layer == "building":
                props["levels"] = tags.get("building:levels")
                props["height_tag"] = tags.get("height")
                props["height_m_est"] = self._levels(tags)
                props["height_source"] = (
                    "osm height tag" if tags.get("height") else
                    "osm building:levels x 3.2 m + 1.0 m" if tags.get("building:levels")
                    else None)
            feats.append({"type": "Feature", "geometry": geom, "properties": props})
            counts[layer] = counts.get(layer, 0) + 1

        if not feats:
            raise Fatal("no OSM feature in the response matched any layer")
        gj = {"type": "FeatureCollection",
              "crs_note": "WGS84 lon/lat (EPSG:4326)",
              "attribution": "(c) OpenStreetMap contributors, ODbL 1.0",
              "queried": now(), "bbox_wsen": list(self.bbox),
              "features": feats}
        (self.src / "osm_features.geojson").write_text(json.dumps(gj))

        # a projected copy in the same metric CRS as the lidar, so downstream
        # code never has to think about degrees
        from pyproj import Transformer
        tr = Transformer.from_crs(4326, self.epsg, always_xy=True)

        def proj(c):
            if isinstance(c[0], (int, float)):
                x, y = tr.transform(c[0], c[1])
                return [round(x, 3), round(y, 3)]
            return [proj(x) for x in c]

        pfeats = [{"type": "Feature", "properties": f["properties"],
                   "geometry": {"type": f["geometry"]["type"],
                                "coordinates": proj(f["geometry"]["coordinates"])}}
                  for f in feats]
        (self.src / f"osm_features_epsg{self.epsg}.geojson").write_text(
            json.dumps({**gj, "crs_note": f"EPSG:{self.epsg} metres",
                        "features": pfeats}))

        with_levels = sum(1 for f in feats
                          if f["properties"].get("height_m_est") is not None)
        named = sum(1 for f in feats if f["properties"].get("name"))
        self.status["metrics"]["osm"] = {
            "elements": len(els), "features": len(feats), "by_layer": counts,
            "buildings_with_height": with_levels, "named": named}
        self.log(f"osm: {len(feats)} features "
                 f"({json.dumps(counts)}), {with_levels} buildings with a "
                 f"height or level count, {named} named")

    # ---- stage: verify ---------------------------------------------------- #

    def st_verify(self) -> None:
        need = ["lidar_catalog.json", "lidar_stats.json", "lidar_points.laz",
                "lidar_points.ply", "osm_raw.json", "osm_features.geojson",
                f"osm_features_epsg{self.epsg}.geojson"]
        missing = [f for f in need if not (self.src / f).exists()]
        if missing:
            raise Fatal(f"outputs missing after a 'successful' run: {missing}")
        arr = read_lidar_ply(self.src / "lidar_points.ply")
        stats = json.loads((self.src / "lidar_stats.json").read_text())
        if len(arr) != stats["points"]:
            raise Fatal(f"lidar_points.ply has {len(arr)} points but "
                        f"lidar_stats.json claims {stats['points']}")
        x0, y0, x1, y1 = stats["crop_bounds_m"]
        xs, ys = arr["X"].astype("f8"), arr["Y"].astype("f8")
        if xs.min() < x0 - 1 or xs.max() > x1 + 1 or ys.min() < y0 - 1 or ys.max() > y1 + 1:
            raise Fatal("cropped points fall outside the crop bounds; the "
                        "reprojection or the crop is wrong")
        # the Web Mercator trap, caught rather than assumed: a box measured in
        # the metric CRS must match the requested box, in metres
        want_w = x1 - x0
        got_w = xs.max() - xs.min()
        if got_w > want_w * 1.05:
            raise Fatal(f"point extent {got_w:.1f} m exceeds the {want_w:.1f} m "
                        f"crop: coordinates are not in EPSG:{self.epsg}")
        gj = json.loads((self.src / "osm_features.geojson").read_text())
        if len(gj["features"]) < 1:
            raise Fatal("osm_features.geojson has no features")
        manifest = {
            "area": self.name, "bbox_wsen": list(self.bbox),
            "generated": now(), "tool": "fetch_context.py",
            "metric_crs": f"EPSG:{self.epsg}",
            "lidar": {
                "dataset": stats["dataset"], "survey_year": stats["survey_year"],
                "source_crs": stats["ept_epsg"],
                "points": stats["points"],
                "density_pts_per_m2": stats["density_pts_per_m2"],
                "classes": stats["classes"],
                "files": ["lidar_points.laz", "lidar_points.ply",
                          "lidar_stats.json", "lidar_catalog.json"],
                "licence": "USGS 3DEP: public domain, no use restrictions",
            },
            "osm": {
                **self.status["metrics"].get("osm", {}),
                "files": ["osm_raw.json", "osm_features.geojson",
                          f"osm_features_epsg{self.epsg}.geojson"],
                "licence": "(c) OpenStreetMap contributors, ODbL 1.0",
            },
            "http_requests_made": self.http.count if self.http else None,
        }
        (self.src / "context_manifest.json").write_text(json.dumps(manifest, indent=2))
        self.status["metrics"]["verify"] = {"ok": True, "files": need}
        self.log(f"verify: ok -- {len(arr):,} lidar points and "
                 f"{len(gj['features'])} OSM features in {self.src}")

    # ---- driver ----------------------------------------------------------- #

    def run(self) -> bool:
        self.work.mkdir(parents=True, exist_ok=True)
        self.src.mkdir(parents=True, exist_ok=True)
        self.log(f"=== context fetch {self.name} {self.bbox} ===")
        try:
            self.preflight()
            if self.a.force_from:
                if self.a.force_from not in STAGES:
                    raise Fatal(f"--force-from {self.a.force_from} is not a stage")
                for st in STAGES[STAGES.index(self.a.force_from):]:
                    self.marker(st).unlink(missing_ok=True)
            for stage in STAGES:
                if self.is_done(stage):
                    self.log(f"skip {stage} (done)")
                    continue
                self.status["stage"] = stage
                self.status["state"] = "running"
                self.save_status()
                t0 = time.time()
                self.log(f"--- {stage} ---")
                getattr(self, f"st_{stage}")()
                self.mark_done(stage, time.time() - t0)
                if self.a.stop_after == stage:
                    self.log(f"stopping after {stage} as asked")
                    break
        except Fatal as e:
            self.status["state"] = "failed"
            self.status["error"] = str(e)
            self.save_status()
            self.log(f"FATAL: {e}")
            return False
        except Exception as e:                       # noqa: BLE001 -- loud
            import traceback
            self.status["state"] = "failed"
            self.status["error"] = f"{type(e).__name__}: {e}"
            self.save_status()
            self.log("FATAL (unexpected):\n" + traceback.format_exc())
            return False
        self.status["state"] = "done"
        self.status["stage"] = None
        self.save_status()
        try:
            shutil.copy2(self.statusfile, self.src / "fetch_status.json")
            shutil.copy2(self.logfile, self.src / "fetch_context.log")
        except OSError:
            pass
        self.log(f"=== done: {self.src} ===")
        return True


def parse_area(tok: str) -> tuple[str, tuple[float, float, float, float]]:
    if "=" not in tok:
        raise SystemExit(f"expected NAME=w,s,e,n, got {tok!r}")
    name, box = tok.split("=", 1)
    parts = [float(v) for v in box.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"bbox needs 4 numbers w,s,e,n, got {box!r}")
    return name, (parts[0], parts[1], parts[2], parts[3])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch public lidar + OSM context for a bounding box.")
    ap.add_argument("areas", nargs="+", metavar="NAME=w,s,e,n")
    ap.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    ap.add_argument("--src-root", default=str(DEFAULT_SRC_ROOT))
    ap.add_argument("--epsg", type=int, default=0,
                    help="metric CRS to measure in (default: the local UTM zone)")
    ap.add_argument("--pad-m", type=float, default=25.0,
                    help="metres of margin around the bbox (default 25)")
    ap.add_argument("--max-probe", type=int, default=8,
                    help="how many catalog datasets to probe for a live EPT")
    ap.add_argument("--cpus", type=int, default=4, help="CPUs for PDAL")
    ap.add_argument("--mem-gb", type=int, default=6, help="RAM cap for PDAL")
    ap.add_argument("--lidar-timeout", type=int, default=3600)
    ap.add_argument("--overpass-timeout", type=int, default=180)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached API responses and refetch")
    ap.add_argument("--force-from", default=None)
    ap.add_argument("--stop-after", default=None)
    a = ap.parse_args()

    ok = True
    for tok in a.areas:
        name, bbox = parse_area(tok)
        ok = ContextFetch(name, bbox, a).run() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
