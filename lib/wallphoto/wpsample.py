"""Projection of one panorama onto one rect, with every admissibility test the
ground pass uses: range gate, incidence gate, lidar DSM line of sight, the
panorama's own measured sky boundary, Mapillary's transient detections and a
foliage veto for the trees the DSM missed.
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import texlib as T
from align import alcommon as A
from wallphoto import wpcommon as H

BACK_OFF = 1.25          # m stepped back off the wall before the DSM test (stage 34's value)
_FQ = None
if H._AP:
    import facade_quality as _FQ     # wpcommon put pipeline/ on the path


class ViewCtx:
    """One panorama, ready to paint rects with.  The image is decoded lazily and
    at the width the caller asks for, so the cover and chip passes never pay
    for a 60 megapixel decode."""

    def __init__(self, pid, pose, dsm, r_max=None, with_mask=True):
        # the occluder is built once per panorama and reused for every rect it
        # touches, so it is sized for the longest gate of any rect
        r_max = H.R_MAX_FAR if r_max is None else r_max
        self.pid = str(pid)
        self.pose = pose
        self.C = np.array([pose['x'], pose['y'], pose['z']], 'f8')
        self.R = np.asarray(pose['R'], 'f8')
        self.occ = T.Occluder(dsm, self.C, r_max=r_max + 15.0, back_off_m=BACK_OFF)
        self.sl, self.sv = T.skyline(self.pid)
        self.mask = A.load_mask(self.pid) if with_mask else None
        # Mapillary-segmentation 'not wall' mask, for an area that sets
        # stage_knobs.FACADE_PHOTO_QUALITY (None otherwise)
        self.seg = _FQ.load(self.pid) if (with_mask and _FQ is not None) else None
        self._img = None
        self._imgw = 0
        self.gain = None

    @property
    def native_width(self):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(f'{H.IMD}/{self.pid}.jpg') as im:
            return min(im.width, H.NATIVE_MAX)

    def img(self, width):
        if self._img is None or self._imgw != width:
            self._img = T.pano_rgb(self.pid, width)
            self._imgw = width
        return self._img

    def free(self):
        self._img = None
        self._imgw = 0

    # ---- the test ---------------------------------------------------------- #
    def look(self, P, nrm, r_max=None):
        """World points (n,3) and the surface normal -> (ok, lon, lat, rng, cos).

        ok is the geometric admissibility: range, incidence, DSM line of sight
        and the panorama's own sky boundary.  No image is decoded.  `r_max` is
        the rect's own range gate (H.rect_rmax)."""
        rmx = H.R_MAX if r_max is None else float(r_max)
        d = (P - self.C[None, :]).astype('f4')
        rng = np.linalg.norm(d, axis=1)
        dn = d / np.maximum(rng[:, None], 1e-6)
        cos = -(dn @ np.asarray(nrm, 'f4'))
        ok = (cos > H.MIN_COS) & (rng > H.R_MIN) & (rng < rmx)
        lon = np.zeros(len(P), 'f4'); lat = np.zeros(len(P), 'f4')
        if not ok.any():
            return ok, lon, lat, rng, cos
        sel = np.nonzero(ok)[0]
        vis = self.occ.visible(P[sel].astype('f4'))
        ok[sel[~vis]] = False
        sel = sel[vis]
        if len(sel):
            lo, la, _ = T.project(P[sel].astype('f4'), self.C, self.R)
            lon[sel] = lo; lat[sel] = la
            if self.sl is not None:
                col = np.clip(((lo / (2 * math.pi) + 0.5) * len(self.sl)
                               ).astype(np.int32), 0, len(self.sl) - 1)
                sky = self.sv[col] & (la > self.sl[col] - 0.02)
                ok[sel[sky]] = False
        return ok, lon, lat, rng, cos

    def masked(self, lon, lat):
        """True where a Mapillary transient detection covers that pixel."""
        if self.mask is None:
            return np.zeros(np.shape(lon), bool)
        u = (lon / (2 * math.pi) + 0.5) * A.MASK_W - 0.5
        v = (0.25 - lat / (2 * math.pi)) * A.MASK_W - 0.5
        return A.mask_lookup(self.mask, u, v, A.MASK_W)

    def colour(self, lon, lat, width):
        return T.sample_equirect(self.img(width), lon, lat)


def sample_rect(ctx, rect, uu, vv, width=None, offset=(0.0, 0.0), vscale=1.0,
                rgb=True, veto_green=True, chunk_rows=None):
    """Paint the (len(vv), len(uu)) rect lattice from one panorama.

    `offset` is the measured 2D shift of the PHOTOGRAPH relative to the model
    in rect metres: the texel at (u, v) is taken from the world point
    (u + du, v * vscale + dv), which is where the pose error actually put that
    piece of wall in this panorama.  `vscale` carries the rect's roofline
    correction: when the silhouette says the real building is taller than the
    OSM extrusion, the texture is STRETCHED from the base rather than slid, so
    the photograph's ground line stays on the model's ground line and its
    roofline still lands on the model's roofline.

    -> dict(valid, rgb, rng, cos, masked_frac, green_frac)
    """
    nu, nv = len(uu), len(vv)
    arr = hasattr(offset[0], 'shape') and np.ndim(offset[0]) == 2
    if arr:
        DU = np.asarray(offset[0], 'f8'); DV = np.asarray(offset[1], 'f8')
        du = dv = 0.0
    else:
        du, dv = float(offset[0]), float(offset[1])
    if chunk_rows is None:
        chunk_rows = max(1, int(2.0e6 // max(nu, 1)))
    VAL = np.zeros((nv, nu), bool)
    RGB = np.zeros((nv, nu, 3), 'u1') if rgb else None
    RNG = np.zeros((nv, nu), 'f4')
    COS = np.zeros((nv, nu), 'f4')
    nmask = 0; ngreen = 0; nlook = 0
    SEG = np.zeros((nv, nu), bool) if ctx.seg is not None else None
    seg_veto = SEG is not None and _FQ.vetoes()
    for r0 in range(0, nv, chunk_rows):
        r1 = min(r0 + chunk_rows, nv)
        UU, VV = np.meshgrid(uu, vv[r0:r1])
        if arr:
            UU = UU + DU[r0:r1]; VV = VV * vscale + DV[r0:r1]
        else:
            UU = UU + du; VV = VV * vscale + dv
        P = rect.points(UU.ravel(), VV.ravel())
        ok, lon, lat, rng, cos = ctx.look(P, rect.n, r_max=H.rect_rmax(rect))
        RNG[r0:r1] = rng.reshape(r1 - r0, nu)
        COS[r0:r1] = cos.reshape(r1 - r0, nu)
        if not ok.any():
            continue
        sel = np.nonzero(ok)[0]
        nlook += len(sel)
        vet = ctx.masked(lon[sel], lat[sel])
        nmask += int(vet.sum())
        sel = sel[~vet]
        if SEG is not None and len(sel):
            sv = _FQ.lookup(ctx.seg, lon[sel], lat[sel])
            blk = SEG[r0:r1].reshape(-1); blk[sel[sv]] = True
            SEG[r0:r1] = blk.reshape(r1 - r0, nu)
            if seg_veto:
                sel = sel[~sv]
        if len(sel) == 0:
            continue
        if rgb or veto_green:
            col = np.clip(ctx.colour(lon[sel], lat[sel], width), 0, 255).astype('u1')
            if veto_green:
                gv = H.green_veto(col)
                ngreen += int(gv.sum())
                sel = sel[~gv]; col = col[~gv]
            if rgb and len(sel):
                blk = RGB[r0:r1].reshape(-1, 3)
                blk[sel] = col
                RGB[r0:r1] = blk.reshape(r1 - r0, nu, 3)
        blk = VAL[r0:r1].reshape(-1)
        blk[sel] = True
        VAL[r0:r1] = blk.reshape(r1 - r0, nu)
    out = dict(valid=VAL, rgb=RGB, rng=RNG, cos=COS,
                masked=nmask, green=ngreen, looked=nlook)
    if SEG is not None:
        out['seg'] = SEG
    return out


def candidates(rects, poses, pid_list, r_max=None):
    """-> per rect, the panorama indices that could see it at all (in front,
    inside r_max of the rect's own surface, above the incidence gate for at
    least the nearest corner)."""
    Cx = np.array([[poses[p]['x'], poses[p]['y'], poses[p]['z']] for p in pid_list])
    out = {}
    for rc in rects:
        ctr = rc.centre
        d = Cx - ctr[None, :]
        du = d @ rc.u; dv = d[:, 2]; dn = d @ rc.n
        cu = np.clip(du, -rc.width / 2, rc.width / 2)
        cv = np.clip(dv, -rc.height / 2, rc.height / 2)
        r = np.sqrt((du - cu) ** 2 + (dv - cv) ** 2 + dn ** 2)
        rmx = H.rect_rmax(rc) if r_max is None else float(r_max)
        ok = (dn > 1.0) & (r < rmx) & (r > H.R_MIN)
        # incidence at the nearest point of the rect
        cosn = dn / np.maximum(r, 1e-6)
        ok &= cosn > H.MIN_COS * 0.8
        out[rc.id] = (np.nonzero(ok)[0].astype('i4'), r[ok].astype('f4'))
    return out
