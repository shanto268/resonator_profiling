import matplotlib.pyplot as plt
import numpy as np

try:
    import skrf as rf
    HAVE_SKRF = True
except Exception:
    HAVE_SKRF = False


# ------------------------- Utility & checks -------------------------

def _assert_1d_freqs(freqs):
    f = np.asarray(freqs, dtype=float)
    if f.ndim != 1 or f.size == 0:
        raise ValueError("freqs must be a non-empty 1D array in Hz.")
    return f

def _tile_abcd(A, B, C, D):
    """Stack 2x2 ABCD parts into shape (N,2,2)."""
    N = A.shape[0]
    out = np.empty((N, 2, 2), dtype=complex)
    out[:, 0, 0] = A; out[:, 0, 1] = B
    out[:, 1, 0] = C; out[:, 1, 1] = D
    return out


# ------------------------- CPW per-unit-length (simple, fast) -------------------------

def cpw_char(width, gap, eps_r):
    """
    Basic CPW characteristics from conformal mapping.
    Returns (Z0 [ohm], eps_eff [-], v_p [m/s], L' [H/m], C' [F/m]).
    """
    if width <= 0 or gap <= 0:
        raise ValueError("width and gap must be > 0.")
    k = width / (width + 2.0*gap)
    if not (0.0 < k < 1.0):
        raise ValueError("width/(width+2*gap) must lie in (0,1).")
    # Complete elliptic integrals via mpmath gives best accuracy, but to keep
    # this self-contained we’ll use a good approximation for K(k')/K(k):
    # If you prefer exactness, swap in mpmath.ellipk as in previous snippets.
    import math
    kp = math.sqrt(1.0 - k*k)
    # Heuristic approx for K(k')/K(k); OK for design. Replace if you need tighter error.
    Kk_over = (1 + 0.25*k**2 + 9.0/64.0*k**4)
    Kkp_over = (1 + 0.25*kp**2 + 9.0/64.0*kp**4)
    Kkp_over_Kk = Kkp_over / Kk_over

    eps_eff = 0.5*(eps_r + 1.0)
    Z0 = (30.0*np.pi/np.sqrt(eps_eff)) * Kkp_over_Kk
    c0 = 299_792_458.0
    vp = c0/np.sqrt(eps_eff)
    C_per_m = 1.0 / (Z0 * vp)
    L_per_m = (Z0**2) * C_per_m
    return float(Z0), float(eps_eff), float(vp), float(L_per_m), float(C_per_m)


def cpw_rlgc_from_geom(freqs, width, gap, eps_r, tan_delta_sub=0.0, R_override=None):
    """
    RLGC'(f) for a uniform CPW from geometry.

    Parameters
    ----------
    freqs : array[Hz], shape (N,)
    width, gap : [m]
    eps_r : [-]
    tan_delta_sub : [-]   dielectric loss tangent (effective)
    R_override : None or array[ohm/m], shape (N,) for conductor loss

    Returns
    -------
    R, L, G, C : arrays (N,)
    Z0, eps_eff, vp : floats
    """
    f = _assert_1d_freqs(freqs)
    Z0, eps_eff, vp, Lp, Cp = cpw_char(width, gap, eps_r)
    L = np.full_like(f, Lp, dtype=float)
    C = np.full_like(f, Cp, dtype=float)
    w = 2.0*np.pi*f
    G = w * C * float(tan_delta_sub)
    if R_override is not None:
        R = np.asarray(R_override, dtype=float)
        if R.shape != f.shape:
            raise ValueError("R_override must match freqs shape.")
    else:
        R = np.zeros_like(f, dtype=float)
    return R, L, G, C, Z0, eps_eff, vp


# ------------------------- ABCD primitives (for S construction) -------------------------

def abcd_line_from_rlgc(freqs, R, L, G, C, length):
    """ABCD of a lossy uniform line of length 'length' using RLGC'(f)."""
    if length <= 0:
        raise ValueError("length must be > 0.")
    f = _assert_1d_freqs(freqs)
    jw = 1j*2.0*np.pi*f
    Zp = R + jw*L
    Yp = G + jw*C
    gamma = np.sqrt(Zp*Yp)        # (N,)
    Zc = np.sqrt(Zp/Yp)           # (N,)
    gl = gamma*length
    cosh_gl = np.cosh(gl)
    sinh_gl = np.sinh(gl)
    A = cosh_gl
    B = Zc*sinh_gl
    Cm = sinh_gl/Zc
    D = cosh_gl
    return _tile_abcd(A, B, Cm, D)


def abcd_shunt_admittance(freqs, Ysh):
    """ABCD of a 2-port with a shunt admittance at the node: [[1,0],[Y,1]]."""
    f = _assert_1d_freqs(freqs)
    Y = np.asarray(Ysh, dtype=complex)
    if Y.size == 1:
        Y = np.full(f.size, complex(Y), dtype=complex)
    if Y.shape != f.shape:
        raise ValueError("Ysh must be scalar or shape (Nfreq,).")
    A = np.ones_like(Y, dtype=complex)
    B = np.zeros_like(Y, dtype=complex)
    C = Y
    D = np.ones_like(Y, dtype=complex)
    return _tile_abcd(A, B, C, D)


# ------------------------- S<->ABCD and blocks as S -------------------------

def abcd2s_vectorized(abcd, z0_ref):
    """
    Manual ABCD→S conversion. Shape abcd: (N,2,2).
    """
    z0 = complex(z0_ref)
    A = abcd[:,0,0]; B = abcd[:,0,1]
    C = abcd[:,1,0]; D = abcd[:,1,1]
    den = A + B/z0 + C*z0 + D
    S11 = (A + B/z0 - C*z0 - D)/den
    S21 = 2.0/den
    S12 = 2.0*(A*D - B*C)/den
    S22 = (-A + B/z0 - C*z0 + D)/den
    S = np.empty_like(abcd, dtype=complex)
    S[:,0,0] = S11; S[:,0,1] = S12
    S[:,1,0] = S21; S[:,1,1] = S22
    return S

def s_line_from_geom(freqs, length, width, gap, eps_r, tan_delta_sub=0.0, R_override=None, z0_ref=None):
    """2-port S of a CPW straight section from geometry (via ABCD)."""
    R, L, G, C, Z0_line, _, _ = cpw_rlgc_from_geom(freqs, width, gap, eps_r, tan_delta_sub, R_override)
    abcd = abcd_line_from_rlgc(freqs, R, L, G, C, length)
    z0 = Z0_line if z0_ref is None else float(z0_ref)
    return abcd2s_vectorized(abcd, z0), Z0_line

def s_bend_as_line(freqs, radius, angle_deg, width, gap, eps_r, tan_delta_sub=0.0, R_override=None, z0_ref=None):
    """2-port S of a CPW bend approximated by arc length line."""
    arc = np.pi*float(radius)*float(angle_deg)/180.0
    return s_line_from_geom(freqs, arc, width, gap, eps_r, tan_delta_sub, R_override, z0_ref)

def s_shunt_open_stub(freqs, stub_length, width, gap, eps_r, tan_delta_sub=0.0, R_override=None, z0_ref=None):
    """
    2-port S of a *shunt* open-circuited CPW stub of physical length.
    We form the shunt via ABCD=[[1,0],[Y_in,1]] with Y_in = 1/Z_in, Z_in = Zc*coth(gamma L).
    """
    R, L, G, C, Z0_line, _, _ = cpw_rlgc_from_geom(freqs, width, gap, eps_r, tan_delta_sub, R_override)
    f = _assert_1d_freqs(freqs)
    jw = 1j*2.0*np.pi*f
    Zp = R + jw*L
    Yp = G + jw*C
    gamma = np.sqrt(Zp*Yp)
    Zc = np.sqrt(Zp/Yp)
    Zin = Zc / np.tanh(gamma*stub_length)    # open end ⇒ Z_in = Zc * coth(γL)
    Yin = 1.0 / Zin
    abcd = abcd_shunt_admittance(freqs, Yin)
    z0 = Z0_line if z0_ref is None else float(z0_ref)
    return abcd2s_vectorized(abcd, z0), Z0_line


# ------------------------- Redheffer star product (2-port, vectorized) -------------------------

def redheffer_star_2port(SA, SB, eps=1e-14):
    """
    Redheffer star product for cascading 2-port S-matrices:
      connect port 2 of A to port 1 of B ⇒ overall 2-port {A1,B2}.

    For each frequency (scalar 2-port case), the closed-form is:
      S11 = S11a + S12a * (I - S11b*S22a)^-1 * S11b * S21a
      S12 = S12a * (I - S11b*S22a)^-1 * S12b
      S21 = S21b * (I - S22a*S11b)^-1 * S21a
      S22 = S22b + S21b * (I - S22a*S11b)^-1 * S22a * S12b
    (Same denominator in scalar case; this is the classic formula.)  ← Rumpf (2011) notes.
    """
    SA = np.asarray(SA, dtype=complex)
    SB = np.asarray(SB, dtype=complex)
    if SA.ndim != 3 or SB.ndim != 3 or SA.shape[1:] != (2,2) or SB.shape[1:] != (2,2):
        raise ValueError("SA and SB must have shape (N,2,2).")
    if SA.shape[0] != SB.shape[0]:
        raise ValueError("SA and SB must share the same frequency axis.")

    S11a, S12a, S21a, S22a = SA[:,0,0], SA[:,0,1], SA[:,1,0], SA[:,1,1]
    S11b, S12b, S21b, S22b = SB[:,0,0], SB[:,0,1], SB[:,1,0], SB[:,1,1]

    den = 1.0 - S11b*S22a
    # regularize near singularities (very strong resonances)
    den = np.where(np.abs(den) < eps, den + eps, den)

    S11 = S11a + S12a * (S11b/den) * S21a
    S12 = S12a * (S12b/den)
    S21 = S21b * (S21a/den)
    S22 = S22b + S21b * (S22a/den) * S12b

    S = np.empty_like(SA)
    S[:,0,0] = S11; S[:,0,1] = S12
    S[:,1,0] = S21; S[:,1,1] = S22
    return S


def star_chain(S_list):
    """Cascade a list of 2-port S-matrices with the Redheffer star product."""
    if not S_list:
        raise ValueError("S_list must be non-empty.")
    out = np.asarray(S_list[0], dtype=complex)
    for S in S_list[1:]:
        out = redheffer_star_2port(out, np.asarray(S, dtype=complex))
    return out


# ------------------------- High-level: build your component-wise chain -------------------------

def build_cpw_chain_with_stub_star(freqs,
                                   components,
                                   stub_insert_index,
                                   stub_length=None,
                                   f0_target=None,
                                   width=10e-6, gap=6e-6, eps_r=11.45,
                                   tan_delta_sub=0.0, R_override=None,
                                   z0_s_param=None):
    """
    Build S of a CPW feedline with an *open* λ/4 stub in shunt at a chosen position,
    using component-wise S-blocks combined by the Redheffer star product.

    Parameters
    ----------
    freqs : array[Hz]
    components : list of dicts in order from Port1 to Port2, each either:
        {"type":"line", "length": L}
        {"type":"bend", "radius": R, "angle_deg": ang}
    stub_insert_index : int
        Insert the shunt stub *after* this component index (use -1 for very beginning).
    stub_length : float [m] or None
        Physical stub length. If None, you must provide f0_target.
    f0_target : float [Hz] or None
        If provided (and stub_length is None), stub length is set to vp/(4 f0_target).
    width, gap, eps_r, tan_delta_sub, R_override : CPW params
    z0_s_param : float or None
        S-parameter reference impedance. If None, uses the CPW Z0.

    Returns
    -------
    net_or_tuple :
        If scikit-rf is installed: skrf.Network (2-port) with S on the chosen reference.
        Else: (freqs, S(2x2), z0_used, stub_length_used)
    """
    f = _assert_1d_freqs(freqs)
    if not components:
        raise ValueError("components must be a non-empty list.")
    if stub_insert_index < -1 or stub_insert_index >= len(components):
        raise ValueError("stub_insert_index out of range.")

    # CPW reference Z0 for S
    Z0_line, _, vp, _, _ = cpw_char(width, gap, eps_r)
    z0 = Z0_line if z0_s_param is None else float(z0_s_param)

    # Build S-blocks up to the insertion point
    Sblocks = []
    for i, comp in enumerate(components):
        if comp["type"] == "line":
            S, _ = s_line_from_geom(f, comp["length"], width, gap, eps_r, tan_delta_sub, R_override, z0_ref=z0)
        elif comp["type"] == "bend":
            S, _ = s_bend_as_line(f, comp["radius"], comp["angle_deg"], width, gap, eps_r, tan_delta_sub, R_override, z0_ref=z0)
        else:
            raise ValueError(f"Unknown component type: {comp['type']}")
        Sblocks.append(S)
        if i == stub_insert_index:
            break

    # Determine stub length
    if stub_length is None and f0_target is None:
        raise ValueError("Provide either stub_length or f0_target.")
    if stub_length is None:
        if f0_target <= 0:
            raise ValueError("f0_target must be > 0.")
        stub_length = vp/(4.0*float(f0_target))

    # Make the shunt stub as a 2-port S (via ABCD shunt → S)
    Sstub, _ = s_shunt_open_stub(f, stub_length, width, gap, eps_r, tan_delta_sub, R_override, z0_ref=z0)
    Sblocks.append(Sstub)

    # Remaining blocks after stub
    for j in range(stub_insert_index+1, len(components)):
        comp = components[j]
        if comp["type"] == "line":
            S, _ = s_line_from_geom(f, comp["length"], width, gap, eps_r, tan_delta_sub, R_override, z0_ref=z0)
        elif comp["type"] == "bend":
            S, _ = s_bend_as_line(f, comp["radius"], comp["angle_deg"], width, gap, eps_r, tan_delta_sub, R_override, z0_ref=z0)
        Sblocks.append(S)

    # Stitch with Redheffer star
    S_total = star_chain(Sblocks)

    if HAVE_SKRF:
        freq = rf.Frequency.from_f(f, unit="Hz")
        ntw = rf.Network(frequency=freq, s=S_total)
        ntw.comment = f"Redheffer-star stitched CPW chain; stub L={stub_length*1e6:.2f} µm; z0={z0:.2f} Ω"
        return ntw
    return f, S_total, z0, stub_length


# Frequency axis
f = np.linspace(3e9, 8e9, 3001)

# CPW geometry/materials
width, gap, eps_r = 10e-6, 6e-6, 11.45
tan_delta = 2e-6

# Your chain: Port1 → line1 → bend → line2 → (stub) → line3 → Port2
components = [
    {"type": "line", "length": 200e-6},
    {"type": "bend", "radius": 50e-6, "angle_deg": 90},
    {"type": "line", "length": 200e-6},
    {"type": "line", "length": 200e-6},
]

# Put the stub right after the bend (index 1). Target ~5 GHz (λ/4 from geometry).
net = build_cpw_chain_with_stub_star(
    freqs=f,
    components=components,
    stub_insert_index=1,
    stub_length=None,
    f0_target=5.0e9,               # or give stub_length directly
    width=width, gap=gap, eps_r=eps_r,
    tan_delta_sub=tan_delta,
    z0_s_param=None                # None ⇒ reference = CPW Z0
)

# If skrf is installed, 'net' is a Network:
net.plot_s_db(m=1, n=0, label="S21 (star-stitched)")
plt.show()
net.plot_s_deg(m=1, n=0)
plt.show()
# print(net.comment)
