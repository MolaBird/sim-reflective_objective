"""
Model of the Thorlabs LMM40X-P01 Schwarzschild reflective objective and
vectorial focusing of a Gaussian beam through it.

Prescription: Thorlabs Zemax file LMM40X-P01-Zemax-ZAR/265_LMM40X-P01.zmx
(units mm, object at infinity, ENPD 5.1 mm, stop at surface 8).

Global frame: z along the optical axis, collimated light enters travelling +z.

  z =  0.000  S1  1st mechanical surface (back aperture side, RMS thread)
  z = 13.400  S3  plane of the primary mirror vertex; light passes through
                  the central hole of the primary
  z = 23.371  S4  convex secondary,  R = +6.1750 mm, clear semi-dia 2.60
  z = 13.400  S5  concave primary,   R = +16.180 mm, reflective annulus
                  4.0 mm < r < 10.75 mm (CLAP 4 10.75)
  z = 24.671  S6  curved 3-vane spider (user obscuration, UDA file)
  z = 24.671  S7  secondary housing: annular aperture, blocks r < 2.5 mm
  z = 24.671  S8  stop (semi-dia 5.875)
  z = 26.901  S9  last mechanical surface (front of objective)
  z = 34.660  S10 image (focal) plane -> working distance 7.76 mm

Method
------
1. Exact real-ray trace (sphere intersections, specular reflection) from a
   plane at the entrance to the last mirror.  This gives for every entrance
   pupil point (x0, y0): the output direction s, the optical path length
   and whether the ray is vignetted (hole, secondary, housing, spiders).
2. Debye-Wolf (Richards-Wolf) vector diffraction integral over the output
   direction cosines, with the pupil-to-angle mapping and energy
   apodisation taken numerically from the trace (sqrt of the Jacobian
   dOmega/dA), the residual wavefront error from the real OPL, and the
   polarisation rotated with the s/p basis of each ray.
   Valid because the Fresnel number a^2/(lambda f) is ~10^3.
"""
import numpy as np

# ----------------------------------------------------------------------------
# Prescription (from the .zmx file)
# ----------------------------------------------------------------------------
Z_ENT = 0.0
Z_PRI = 4.7 + 8.7                    # 13.4
Z_SEC = Z_PRI + 9.97122              # 23.37122
C_SEC = 1.619425104308223895e-01     # 1/mm
C_PRI = 6.180355123205379803e-02     # 1/mm
D_SEC_TO_S6 = 11.271224725550001
Z_S6 = Z_SEC - 9.9712247255449995 + D_SEC_TO_S6   # spider / housing / stop
Z_IMG = Z_S6 + 2.23 + 7.758446638763
ENPD = 5.1                           # entrance pupil diameter, mm
R_SEC_CLEAR = 2.6
R_PRI_IN, R_PRI_OUT = 4.0, 10.75
R_HOUSING = 2.5                      # S7 CLAP inner radius
SPIDER_FILE = "LMM40X-P01-Zemax-ZAR/curved-spiders-LMM40X.uda"
SPIDER_SCALE = 1e-2                  # UDAD 0 scale factor in the .zmx


def _intersect_sphere(p, d, z_v, c):
    """Intersection of rays p + t d with the sphere of curvature c, vertex z_v.
    Returns points and unit normals (pointing towards the centre side)."""
    R = 1.0 / c
    ctr = np.array([0.0, 0.0, z_v + R])
    oc = p - ctr
    b = np.einsum("ij,ij->i", oc, d)
    cc = np.einsum("ij,ij->i", oc, oc) - R * R
    disc = np.sqrt(b * b - cc)
    t1, t2 = -b - disc, -b + disc
    q1 = p + t1[:, None] * d
    q2 = p + t2[:, None] * d
    # the physical cap is the one close to the vertex
    use1 = np.abs(q1[:, 2] - z_v) < np.abs(q2[:, 2] - z_v)
    t = np.where(use1, t1, t2)
    q = np.where(use1[:, None], q1, q2)
    n = (ctr - q) / R
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return q, n, t


def _reflect(d, n):
    return d - 2 * np.einsum("ij,ij->i", d, n)[:, None] * n


def load_spiders(path=SPIDER_FILE, scale=SPIDER_SCALE):
    """Parse a Zemax UDA file into a list of polygons (N,2) in mm.
    ARC xc yc angle n : arc about (xc,yc) from the current point; a positive
    angle is clockwise (checked against the UDAD points stored in the .zmx)."""
    polys, cur = [], []
    with open(path, encoding="utf-16") as f:
        for line in f:
            tok = line.split()
            if not tok or tok[0].startswith("!"):
                continue
            key = tok[0].upper()
            if key == "LIN":
                cur.append((float(tok[1]), float(tok[2])))
            elif key == "ARC":
                xc, yc, ang, n = map(float, tok[1:5])
                x, y = cur[-1]
                r = np.hypot(x - xc, y - yc)
                a0 = np.arctan2(y - yc, x - xc)
                a = a0 - np.deg2rad(ang) * np.arange(1, int(n) + 1) / n
                cur.extend(zip(xc + r * np.cos(a), yc + r * np.sin(a)))
            elif key == "BRK":
                polys.append(np.array(cur) * scale)
                cur = []
    if cur:
        polys.append(np.array(cur) * scale)
    return polys


def _in_polygon(x, y, poly):
    """Even-odd point-in-polygon test, vectorised over points."""
    inside = np.zeros(x.shape, bool)
    xp, yp = poly[:, 0], poly[:, 1]
    xq, yq = np.roll(xp, 1), np.roll(yp, 1)
    for x1, y1, x2, y2 in zip(xp, yp, xq, yq):
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xi)
    return inside


def trace(x0, y0, spiders=True, polys=None):
    """Trace collimated rays entering at (x0, y0) [mm] parallel to +z.

    Returns dict with
      s      : (N,3) unit output directions
      P      : (N,3) point on the primary mirror (last optical surface)
      L      : (N,)  optical path length from the plane z=Z_ENT to P [mm]
      ok     : (N,)  bool, ray reaches the focus (not vignetted)
      r_pri, r_sec, xy6 : hit radii / position at the spider plane
    """
    x0 = np.atleast_1d(x0).astype(float).ravel()
    y0 = np.atleast_1d(y0).astype(float).ravel()
    N = x0.size
    p = np.stack([x0, y0, np.full(N, Z_ENT)], 1)
    d = np.tile([0.0, 0.0, 1.0], (N, 1))
    ok = np.hypot(x0, y0) <= ENPD / 2 + 1e-12

    # secondary (convex)
    q, n, t = _intersect_sphere(p, d, Z_SEC, C_SEC)
    L = t.copy()
    r_sec = np.hypot(q[:, 0], q[:, 1])
    ok &= r_sec <= R_SEC_CLEAR
    d = _reflect(d, n)
    # primary (concave)
    q2, n2, t2 = _intersect_sphere(q, d, Z_PRI, C_PRI)
    L += t2
    r_pri = np.hypot(q2[:, 0], q2[:, 1])
    ok &= (r_pri >= R_PRI_IN) & (r_pri <= R_PRI_OUT)
    d = _reflect(d, n2)
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    # obscurations in the plane z = Z_S6
    t6 = (Z_S6 - q2[:, 2]) / d[:, 2]
    p6 = q2 + t6[:, None] * d
    r6 = np.hypot(p6[:, 0], p6[:, 1])
    ok &= r6 >= R_HOUSING
    if spiders:
        if polys is None:
            polys = load_spiders()
        for poly in polys:
            ok &= ~_in_polygon(p6[:, 0], p6[:, 1], poly)
    return dict(s=d, P=q2, L=L, ok=ok, r_pri=r_pri, r_sec=r_sec,
                xy6=p6[:, :2])


def paraxial_properties(h=1e-4):
    """EFL, back focal position (paraxial) from a near-axis ray.
    The near-axis ray is vignetted physically but the geometry is valid."""
    tr = trace([h], [0.0], spiders=False)
    s, P = tr["s"][0], tr["P"][0]
    u = s[0] / s[2]                      # slope (negative: converging)
    efl = -h / np.sin(np.arctan(u))
    z_focus = P[2] - P[0] / u
    return efl, z_focus


def best_focus(ref_z=None, nr=400):
    """Axial focus position minimising the rms wavefront error of the
    unobscured annulus (uniform weighting) -> 'best focus' plane."""
    efl, zf = paraxial_properties()
    r = np.linspace(0, ENPD / 2, nr)
    tr = trace(r, 0 * r, spiders=False)
    ok = tr["ok"]
    zs = np.linspace(zf - 0.01, zf + 0.01, 401)
    rms = []
    for z in zs:
        W = _wavefront(tr, np.array([0, 0, z]))[ok]
        w = r[ok]                        # area weight ~ r dr
        W = W - np.sum(W * w) / np.sum(w)
        rms.append(np.sqrt(np.sum(W ** 2 * w) / np.sum(w)))
    i = int(np.argmin(rms))
    return zs[i], rms[i], zf


def _wavefront(tr, F):
    """Optical path of each ray to the reference point F, measured along the
    plane wave it represents: L + s.(F - P).  Constant for a perfect focus."""
    return tr["L"] + np.einsum("ij,j->i", tr["s"], F) - np.einsum(
        "ij,ij->i", tr["s"], tr["P"])


# ----------------------------------------------------------------------------
# Pupil sampling and the Debye-Wolf integral
# ----------------------------------------------------------------------------
class Pupil:
    """Ray-traced pupil sampled on a Cartesian grid of the entrance pupil.
    Geometry only (wavelength independent: mirrors are achromatic)."""

    def __init__(self, n=401, spiders=True, focus=None):
        self.n = n
        a = ENPD / 2
        g = np.linspace(-a, a, n)
        self.dA = (g[1] - g[0]) ** 2
        X, Y = np.meshgrid(g, g)
        m = np.hypot(X, Y) <= a
        x0, y0 = X[m], Y[m]
        polys = load_spiders() if spiders else None
        tr = trace(x0, y0, spiders=spiders, polys=polys)
        ok = tr["ok"]
        if focus is None:
            focus = best_focus()[0]
        self.z_focus = focus
        F = np.array([0.0, 0.0, focus])
        self.x0, self.y0 = x0[ok], y0[ok]
        self.s = tr["s"][ok]
        self.W = _wavefront(tr, F)[ok]
        self.W -= self.W.mean()          # piston irrelevant (mm)
        # energy apodisation sqrt(dOmega/dA) from a finite-difference Jacobian
        h = 1e-4
        sxp = trace(self.x0 + h, self.y0, spiders=False)["s"]
        sxm = trace(self.x0 - h, self.y0, spiders=False)["s"]
        syp = trace(self.x0, self.y0 + h, spiders=False)["s"]
        sym = trace(self.x0, self.y0 - h, spiders=False)["s"]
        dsx_dx = (sxp[:, 0] - sxm[:, 0]) / (2 * h)
        dsy_dx = (sxp[:, 1] - sxm[:, 1]) / (2 * h)
        dsx_dy = (syp[:, 0] - sym[:, 0]) / (2 * h)
        dsy_dy = (syp[:, 1] - sym[:, 1]) / (2 * h)
        J = np.abs(dsx_dx * dsy_dy - dsx_dy * dsy_dx) / self.s[:, 2]
        self.apod = np.sqrt(J)
        # full-aperture mask (for reporting) on the grid
        self.grid = g
        self.mask_full = np.zeros(X.shape, bool)
        idx = np.flatnonzero(m.ravel())[ok]
        self.mask_full.ravel()[idx] = True

    def polarisation(self, pol=(1.0, 0.0)):
        """Output field vectors (N,3) for an input Jones vector pol=(Ex,Ey):
        the radial (p) component rotates with the ray, the azimuthal (s)
        component is unchanged (ideal mirrors, Richards-Wolf)."""
        phi = np.arctan2(self.y0, self.x0)
        ephi = np.stack([-np.sin(phi), np.cos(phi), 0 * phi], 1)
        erho = np.stack([np.cos(phi), np.sin(phi), 0 * phi], 1)
        ep_out = np.cross(ephi, self.s)
        ep_out /= np.linalg.norm(ep_out, axis=1, keepdims=True)
        Ex, Ey = pol
        a_p = erho[:, 0] * Ex + erho[:, 1] * Ey
        a_s = ephi[:, 0] * Ex + ephi[:, 1] * Ey
        return a_p[:, None] * ep_out + a_s[:, None] * ephi


def gaussian_input(pupil, w_in, x_off=0.0, y_off=0.0):
    """Collimated TEM00 amplitude, w_in = 1/e^2 intensity radius [mm]."""
    r2 = (pupil.x0 - x_off) ** 2 + (pupil.y0 - y_off) ** 2
    return np.exp(-r2 / w_in ** 2)


def gaussian_power_fractions(w_in, pupil):
    """Fraction of the input beam power transmitted through the objective
    (relative to the full, unclipped Gaussian beam)."""
    P_in = np.pi * w_in ** 2 / 2                     # int exp(-2r^2/w^2) dA
    P_t = np.sum(gaussian_input(pupil, w_in) ** 2) * pupil.dA
    return P_t / P_in


def focal_field(pupil, amp, wl, x, y, z=0.0, pol=(1.0, 0.0), vector=True,
                chunk=20000):
    """Field E(x, y, z) [3, ny, nx] near focus (coordinates in mm, relative
    to the best-focus point).  amp: input amplitude at each pupil sample.

    E(r) = (-i/wl) sum_j amp_j sqrt(J_j) e_j exp(ik[W_j + s_j.r]) dA
    Normalised so that sum |E|^2 dx dy (energy flux through the plane,
    small-angle part) equals the transmitted power for a uniform grid."""
    k = 2 * np.pi / wl
    c = amp * pupil.apod * np.exp(1j * k * pupil.W) * pupil.dA
    c = c * np.exp(1j * k * pupil.s[:, 2] * z)
    if vector:
        vec = pupil.polarisation(pol)
    else:
        vec = np.ones((c.size, 1))
    x = np.asarray(x)
    y = np.asarray(y)
    E = np.zeros((vec.shape[1], y.size, x.size), complex)
    for i0 in range(0, c.size, chunk):
        sl = slice(i0, i0 + chunk)
        X = np.exp(1j * k * np.outer(x, pupil.s[sl, 0]))   # (nx, Nj)
        Y = np.exp(1j * k * np.outer(y, pupil.s[sl, 1]))   # (ny, Nj)
        for m in range(vec.shape[1]):
            E[m] += (Y * (c[sl] * vec[sl, m])) @ X.T
    return E * (-1j / wl)
