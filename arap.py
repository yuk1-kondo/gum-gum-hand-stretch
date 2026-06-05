"""
As-Rigid-As-Possible 2D shape manipulation (Igarashi, Moscovich, Hughes,
SIGGRAPH 2005) — the image-warping method with the two-step closed-form solve.

Used by Gum-Gum to deform the real finger photo like an elastic material:

  1. build_finger_mesh()  — triangulate the masked finger crop (boundary +
     interior), texture-mapped 1:1 to the source pixels (UV == rest coords).
  2. ARAPSolver           — pre-factorize the two linear systems once per grab
     (step 1: free-similarity fit; step 2: scale-adjusted rigid fit). Per frame
     is just two cheap back-substitutions given the live handle positions.
  3. render_mesh()        — piecewise-affine texture-map each triangle from rest
     -> deformed, composited with a feathered alpha. No stroke / outline.

All coordinates are full-frame pixels. `uv` indexes the cached grab-time crop.
"""

import cv2
import numpy as np
from scipy.spatial import Delaunay
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import lil_matrix


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------
def _resample_polyline(poly, n):
    """Resample a closed polyline to n points evenly spaced by arc length."""
    pts = np.vstack([poly, poly[:1]])
    seg = np.hypot(*(np.diff(pts, axis=0).T))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return poly[:1].repeat(n, 0)
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = []
    for t in targets:
        i = np.searchsorted(cum, t) - 1
        i = max(0, min(i, len(seg) - 1))
        f = (t - cum[i]) / max(seg[i], 1e-9)
        out.append(pts[i] + f * (pts[i + 1] - pts[i]))
    return np.array(out)


def build_finger_mesh(mask, n_boundary=36, interior_step=None):
    """
    Triangulate the masked finger region. Returns (verts Nx2 float, tris Mx3)
    in full-frame pixel coords, or None. verts double as UV (rest == source).
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(cnt) < 3:
        return None
    bnd = _resample_polyline(cnt, n_boundary)

    x, y, w, h = cv2.boundingRect(mask)
    step = interior_step or max(9, int(0.22 * min(w, h)))
    interior = [(xx, yy)
                for yy in range(y + step // 2, y + h, step)
                for xx in range(x + step // 2, x + w, step)
                if mask[yy, xx] > 0]
    pts = np.vstack([bnd, interior]) if interior else bnd
    pts = np.unique(np.round(pts, 1), axis=0)
    if len(pts) < 4:
        return None

    tri = Delaunay(pts)
    keep = []
    for t in tri.simplices:
        cx, cy = pts[t].mean(0)
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]:
            keep.append(t)
    if not keep:
        return None
    return pts.astype(np.float64), np.array(keep, np.int32)


def pick_handles(verts, knuckle, tip, anchor_radius):
    """Anchor vertices = near the MCP knuckle (pinned); tip vertex = nearest tip."""
    dk = np.hypot(verts[:, 0] - knuckle[0], verts[:, 1] - knuckle[1])
    dt = np.hypot(verts[:, 0] - tip[0], verts[:, 1] - tip[1])
    tip_v = int(dt.argmin())
    anchors = [int(i) for i in np.where(dk < anchor_radius)[0] if i != tip_v]
    if not anchors:
        anchors = [int(dk.argmin())]
    return anchors, tip_v


# ---------------------------------------------------------------------------
# ARAP two-step closed-form solver
# ---------------------------------------------------------------------------
class ARAPSolver:
    def __init__(self, verts, tris, handle_idx, weight=1000.0):
        self.verts = np.asarray(verts, np.float64)
        self.tris = np.asarray(tris, np.int32)
        self.handles = list(handle_idx)
        self.w = float(weight)
        n = len(self.verts)
        self.n = n

        # ---- Step 1: free-similarity fit. Unknown u = [x0,y0,x1,y1,...] (2n).
        A0 = np.zeros((2 * n, 2 * n))
        self._ginv = []          # per-tri (4x6): extracts (a,b,tx,ty) from V_t
        self._tdofs = []         # per-tri interleaved dof indices (len 6)
        for t in self.tris:
            rest = self.verts[t]                       # 3x2
            G = np.zeros((6, 4))
            for r, (px, py) in enumerate(rest):
                G[2 * r] = [px, -py, 1.0, 0.0]
                G[2 * r + 1] = [py, px, 0.0, 1.0]
            ginv = np.linalg.solve(G.T @ G, G.T)        # 4x6
            F = np.eye(6) - G @ ginv                    # residual (I - H)
            ftf = F.T @ F
            dofs = [d for v in t for d in (2 * v, 2 * v + 1)]
            for a in range(6):
                for b in range(6):
                    A0[dofs[a], dofs[b]] += ftf[a, b]
            self._ginv.append(ginv)
            self._tdofs.append(dofs)
        for hi in self.handles:
            A0[2 * hi, 2 * hi] += self.w
            A0[2 * hi + 1, 2 * hi + 1] += self.w
        A0 += 1e-8 * np.eye(2 * n)
        self._A1 = cho_factor(A0)

        # ---- Step 2: rigid (scale-adjusted) fit. x and y decoupled, same A2.
        edges = []               # (i, j) per directed triangle edge
        self._tri_edge_rows = [] # per-tri row indices into `edges`
        self._rest_edges = []    # per-tri 3 rest edge vectors
        for t in self.tris:
            i, j, k = int(t[0]), int(t[1]), int(t[2])
            rows = []
            re = []
            for a, b in ((i, j), (j, k), (k, i)):
                rows.append(len(edges))
                edges.append((a, b))
                re.append(self.verts[b] - self.verts[a])
            self._tri_edge_rows.append(rows)
            self._rest_edges.append(np.array(re))
        self._edges = edges
        E = len(edges)
        B2 = lil_matrix((E, n))
        for r, (a, b) in enumerate(edges):
            B2[r, b] += 1.0
            B2[r, a] -= 1.0
        self._B2 = B2.tocsr()
        A2 = (self._B2.T @ self._B2).toarray()
        for hi in self.handles:
            A2[hi, hi] += self.w
        A2 += 1e-8 * np.eye(n)
        self._A2 = cho_factor(A2)

    def solve(self, targets):
        """targets: {vertex_index: (x, y)}. Returns deformed verts (n, 2)."""
        n = self.n
        # Step 1 — free similarity fit.
        b1 = np.zeros(2 * n)
        for hi, (hx, hy) in targets.items():
            b1[2 * hi] = self.w * hx
            b1[2 * hi + 1] = self.w * hy
        u = cho_solve(self._A1, b1)

        # Per-triangle rotation (normalize the fitted similarity to unit scale).
        E = len(self._edges)
        tx = np.zeros(E)
        ty = np.zeros(E)
        for ti in range(len(self.tris)):
            V = u[self._tdofs[ti]]                      # 6
            ab = self._ginv[ti] @ V                     # (a, b, tx, ty)
            a, b = ab[0], ab[1]
            s = np.hypot(a, b) or 1.0
            R = np.array([[a / s, -b / s], [b / s, a / s]])
            for row, e0 in zip(self._tri_edge_rows[ti], self._rest_edges[ti]):
                te = R @ e0
                tx[row] = te[0]
                ty[row] = te[1]

        # Step 2 — fit rotated rest-length edges (x, y decoupled).
        bx = self._B2.T @ tx
        by = self._B2.T @ ty
        for hi, (hx, hy) in targets.items():
            bx[hi] += self.w * hx
            by[hi] += self.w * hy
        x = cho_solve(self._A2, bx)
        y = cho_solve(self._A2, by)
        return np.stack([x, y], axis=1)


# ---------------------------------------------------------------------------
# Deformation helpers + rendering
# ---------------------------------------------------------------------------
def thin_toward_axis(deformed, anchor, tip, free_mask, amount):
    """
    Volume-conservation thinning: pull non-handle vertices toward the
    anchor->tip centerline by `amount` (0..~0.4). Returns a new array.
    """
    if amount <= 0:
        return deformed
    a = np.asarray(anchor, float)
    d = np.asarray(tip, float) - a
    L2 = float(d @ d)
    if L2 < 1e-6:
        return deformed
    out = deformed.copy()
    rel = deformed - a
    t = (rel @ d) / L2                       # projection param along axis
    proj = a + np.outer(t, d)                # foot of perpendicular
    out[free_mask] = (deformed[free_mask]
                      + (proj[free_mask] - deformed[free_mask]) * amount)
    return out


def render_mesh(texture, uv, deformed, tris, frame, feather=5,
                spec=0.0, anchor=None, tip=None, spec_offset=0.30,
                finger_width=None, spec_max=0.25):
    """
    Piecewise-affine texture-map each triangle (rest UV -> deformed) into a limb
    layer, then composite over `frame` with a feathered alpha. No stroke.

    Optional sheen (`spec`): a thin, feathered, off-center band along anchor->tip
    that BRIGHTENS the real skin tone (multiplicative == lifting only V in HSV,
    leaving H/S) — capped at `spec_max` of the per-pixel headroom so it can never
    clip to white. The rest of the limb passes through as raw deformed skin.
    Returns the modified ROI box or None.
    """
    H, W = frame.shape[:2]
    xs, ys = deformed[:, 0], deformed[:, 1]
    x0 = max(0, int(xs.min()) - 2)
    y0 = max(0, int(ys.min()) - 2)
    x1 = min(W, int(xs.max()) + 3)
    y1 = min(H, int(ys.max()) + 3)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    rw, rh = x1 - x0, y1 - y0
    layer = np.zeros((rh, rw, 3), np.float32)
    alpha = np.zeros((rh, rw), np.uint8)
    off = np.array([x0, y0], np.float32)
    # A 4-channel texture carries a per-pixel silhouette alpha (the tight finger
    # mask). It is warped WITH the triangle, so anything outside the finger
    # outline renders with alpha 0 — no background ever travels with the limb.
    has_sil = texture.ndim == 3 and texture.shape[2] == 4

    for t in tris:
        src = uv[t].astype(np.float32)
        dst = (deformed[t] - off).astype(np.float32)
        bx, by, bw, bh = cv2.boundingRect(dst)
        if bw <= 0 or bh <= 0:
            continue
        ax0, ay0 = max(0, bx), max(0, by)
        ax1, ay1 = min(rw, bx + bw), min(rh, by + bh)
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        dst_local = (dst - np.float32([bx, by]))
        M = cv2.getAffineTransform(src, dst_local)
        # Transparent (zero) border, NOT reflect/replicate — reflecting smears
        # edge pixels (and the silhouette alpha) out to the triangle bbox edge,
        # which is what leaves a constant-color rectangular seam.
        warp = cv2.warpAffine(texture, M, (bw, bh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        tmask = np.zeros((bh, bw), np.uint8)
        # Binary (LINE_8) coverage — anti-aliased triangle edges would blend the
        # transparent-black border in along every internal seam. The smooth
        # OUTER edge comes from the inward-feathered silhouette alpha at the end.
        cv2.fillConvexPoly(tmask, dst_local.astype(np.int32), 255, cv2.LINE_8)
        if has_sil:                       # gate triangle by warped silhouette
            tmask = np.minimum(tmask, warp[:, :, 3])
            warp = warp[:, :, :3]
        sy0, sx0 = ay0 - by, ax0 - bx
        sub_m = tmask[sy0:sy0 + (ay1 - ay0), sx0:sx0 + (ax1 - ax0)]
        sub_w = warp[sy0:sy0 + (ay1 - ay0), sx0:sx0 + (ax1 - ax0)]
        sel = sub_m > 0
        dst_layer = layer[ay0:ay1, ax0:ax1]
        dst_alpha = alpha[ay0:ay1, ax0:ax1]
        dst_layer[sel] = sub_w[sel]
        np.maximum(dst_alpha, sub_m, out=dst_alpha)

    # Subtle hue-preserving sheen: a thin off-center band that brightens the
    # real skin (never paints white). Capped so it plateaus well below clipping.
    if spec > 0.01 and anchor is not None and tip is not None:
        k = min(spec_max, max(0.0, spec))               # low ceiling, plateaus
        a = np.array(anchor, float) - off
        b = np.array(tip, float) - off
        d = b - a
        nlen = np.hypot(*d) or 1.0
        nrm = np.array([-d[1], d[0]]) / nlen            # perpendicular
        if finger_width:
            bw = max(2.0, finger_width * 0.28)          # band ~28% of width
            offs = finger_width * 0.5 * spec_offset     # off-center
        else:
            ext = max(np.ptp(deformed[:, 0]), np.ptp(deformed[:, 1]))
            bw = max(2.0, ext * 0.06)
            offs = bw * spec_offset
        p1 = (a + nrm * offs).astype(np.int32)
        p2 = (b + nrm * offs).astype(np.int32)
        glow = np.zeros((rh, rw), np.uint8)
        cv2.line(glow, tuple(p1), tuple(p2), 255, int(bw), cv2.LINE_AA)
        kk = max(3, int(bw) | 1)
        band = cv2.GaussianBlur(glow, (kk, kk), 0).astype(np.float32) / 255.0
        band *= alpha.astype(np.float32) / 255.0        # only on the limb
        # Multiplicative brighten preserves H & S; clamp so no channel clips.
        f = 1.0 + k * band[:, :, None]
        maxc = np.maximum(layer.max(axis=2, keepdims=True), 1e-3)
        np.minimum(f, 255.0 / maxc, out=f)
        layer *= f

    # Feather INWARD: erode first so the blur fades finger->transparent INSIDE
    # the silhouette (where the layer has real color), never out into the black
    # ROI margin — that outward bleed is what drew a dark edge/box seam.
    if feather >= 3 and feather % 2 == 1:
        alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                           (3, 3)))
        alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    # Hard guarantee: never composite where the layer is unpainted (black). The
    # blur can spread alpha a few px past the painted skin into the transparent
    # border; gating by painted-pixels kills that dark halo entirely.
    a *= (layer.sum(axis=2, keepdims=True) > 0).astype(np.float32)
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1.0 - a) + layer * a).astype(np.uint8)
    return (x0, y0, x1, y1)
