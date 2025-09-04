"""
Component-wise ABCD CPW toolkit with λ/4 stub length from target frequency.

You can:
  1) Build a feedline from explicit components: lines and bends.
  2) Insert a shunt open-circuited λ/4 stub at a chosen index, where the stub
     length is automatically computed from a target frequency f0 using vp from geometry.
  3) Convert ABCD to S at a chosen reference impedance (default CPW Z0).
  4) Build a one-port λ/4 resonator from f0_target as a reference.

Units:
  Frequencies in Hz. Lengths in meters. Permittivities dimensionless.
  Loss tangents dimensionless. Impedances in ohms.
"""

import numpy as np
from mpmath import ellipk

try:
    import skrf as rf
    _HAVE_SKRF = True
except Exception:
    _HAVE_SKRF = False


# ------------------------- CPW core -------------------------

def _cpw_char(width, gap, eps_r):
    """
    Return CPW characteristic values from conformal mapping.

    Parameters
    ----------
    width : float [m]  CPW center trace width
    gap   : float [m]  CPW slot to ground
    eps_r : float [-]  substrate relative permittivity

    Returns
    -------
    Z0 : float [ohm]
    eps_eff : float [-]
    vp : float [m/s]
    L_per_m : float [H/m]
    C_per_m : float [F/m]
    """
    if width <= 0 or gap <= 0:
        raise ValueError("width and gap must be positive.")
    k = width / (width + 2.0 * gap)
    if not (0.0 < k < 1.0):
        raise ValueError("Invalid CPW geometry: width/(width+2*gap) must lie in (0,1).")
    kp = np.sqrt(1.0 - k**2)
    Kk = float(ellipk(k**2))
    Kkp = float(ellipk(kp**2))
    eps_eff = 0.5 * (eps_r + 1.0)
    Z0 = (30.0 * np.pi / np.sqrt(eps_eff)) * (Kkp / Kk)
    c0 = 299_792_458.0
    vp = c0 / np.sqrt(eps_eff)
    C_per_m = 1.0 / (Z0 * vp)
    L_per_m = Z0**2 * C_per_m
    return Z0, eps_eff, vp, L_per_m, C_per_m


def cpw_rlgc_from_geom(freqs, width, gap, eps_r, tan_delta_sub=0.0, R_override=None):
    """
    Per-unit-length RLGC arrays for a uniform CPW section from geometry.

    Parameters
    ----------
    freqs : array_like [Hz]
    width : float [m]
    gap   : float [m]
    eps_r : float [-]
    tan_delta_sub : float [-]  dielectric loss tangent (effective)
    R_override : array_like or None [ohm/m]
        If provided, must match freqs shape. Use for superconducting conductor loss.

    Returns
    -------
    R, L, G, C : arrays shaped like freqs
    Z0, eps_eff, vp : floats
    """
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("freqs must be a non-empty 1D array in Hz.")
    Z0, eps_eff, vp, Lp, Cp = _cpw_char(width, gap, eps_r)
    L = np.full_like(freqs, Lp, dtype=float)
    C = np.full_like(freqs, Cp, dtype=float)
    omega = 2.0 * np.pi * freqs
    G = omega * C * float(tan_delta_sub)
    if R_override is not None:
        R = np.asarray(R_override, dtype=float)
        if R.shape != freqs.shape:
            raise ValueError("R_override must match freqs shape.")
    else:
        R = np.zeros_like(freqs, dtype=float)
    return R, L, G, C, Z0, eps_eff, vp


# ------------------------- ABCD primitives -------------------------

def abcd_line_from_rlgc(freqs, R, L, G, C, length):
    """
    ABCD of a uniform distributed line with given RLGC arrays and length.

    gamma = sqrt((R + jωL)(G + jωC))
    Zc    = sqrt((R + jωL)/(G + jωC))
    ABCD  = [[cosh(γl), Zc sinh(γl)],
             [sinh(γl)/Zc, cosh(γl)]]
    """
    if length <= 0:
        raise ValueError("length must be positive.")
    freqs = np.asarray(freqs, dtype=float)
    jw = 1j * 2.0 * np.pi * freqs
    Zp = R + jw * L
    Yp = G + jw * C
    gamma = np.sqrt(Zp * Yp)
    Zc = np.sqrt(Zp / Yp)
    gl = gamma * length
    cosh_gl = np.cosh(gl)
    sinh_gl = np.sinh(gl)
    A = cosh_gl
    B = Zc * sinh_gl
    Cmat = sinh_gl / Zc
    D = cosh_gl
    abcd = np.empty((freqs.size, 2, 2), dtype=complex)
    abcd[:, 0, 0] = A
    abcd[:, 0, 1] = B
    abcd[:, 1, 0] = Cmat
    abcd[:, 1, 1] = D
    return abcd


def abcd_shunt_open_stub(freqs, R, L, G, C, length):
    """
    ABCD of a shunt open-circuited stub (attached at a node).

    Zin_open = Zc * coth(γ L),    Ysh = 1 / Zin_open
    ABCD_shunt = [[1, 0], [Ysh, 1]]
    """
    freqs = np.asarray(freqs, dtype=float)
    jw = 1j * 2.0 * np.pi * freqs
    Zp = R + jw * L
    Yp = G + jw * C
    gamma = np.sqrt(Zp * Yp)
    Zc = np.sqrt(Zp / Yp)
    Zin = Zc / np.tanh(gamma * length)  # coth = 1/tanh
    Ysh = 1.0 / Zin
    abcd = np.zeros((freqs.size, 2, 2), dtype=complex)
    abcd[:, 0, 0] = 1.0
    abcd[:, 0, 1] = 0.0
    abcd[:, 1, 0] = Ysh
    abcd[:, 1, 1] = 1.0
    return abcd


def abcd_cascade(*abcd_list):
    """
    Vectorized cascade: out = A @ B @ C ... for each frequency.
    """
    if not abcd_list:
        raise ValueError("provide at least one ABCD matrix")
    out = np.asarray(abcd_list[0], dtype=complex)
    for nxt in abcd_list[1:]:
        B = np.asarray(nxt, dtype=complex)
        A11 = out[:, 0, 0]*B[:, 0, 0] + out[:, 0, 1]*B[:, 1, 0]
        A12 = out[:, 0, 0]*B[:, 0, 1] + out[:, 0, 1]*B[:, 1, 1]
        A21 = out[:, 1, 0]*B[:, 0, 0] + out[:, 1, 1]*B[:, 1, 0]
        A22 = out[:, 1, 0]*B[:, 0, 1] + out[:, 1, 1]*B[:, 1, 1]
        out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = A11, A12, A21, A22
    return out


def abcd_to_s(abcd, z0_ref):
    """
    Convert ABCD (N,2,2) to S-parameters (N,2,2) at scalar reference z0_ref.
    """
    z0 = complex(z0_ref)
    A = abcd[:, 0, 0]
    B = abcd[:, 0, 1]
    C = abcd[:, 1, 0]
    D = abcd[:, 1, 1]
    den = A + B / z0 + C * z0 + D
    S = np.empty_like(abcd, dtype=complex)
    S[:, 0, 0] = (A + B / z0 - C * z0 - D) / den
    S[:, 1, 0] = 2.0 / den
    S[:, 0, 1] = 2.0 * (A * D - B * C) / den
    S[:, 1, 1] = (-A + B / z0 - C * z0 + D) / den
    return S


def to_skrf_network(freqs, S, z0):
    """
    Wrap S into a scikit-rf Network with reference z0.
    """
    if not _HAVE_SKRF:
        raise RuntimeError("scikit-rf not available.")
    f = rf.Frequency.from_f(np.asarray(freqs, dtype=float), unit="Hz")
    return rf.Network(frequency=f, s=S)


# ------------------------- Component-wise builders -------------------------

def abcd_cpw_line(freqs, length, width, gap, eps_r,
                  tan_delta_sub=0.0, R_override=None):
    """
    ABCD for a CPW straight section from geometry.
    Returns (ABCD, Z0_line).
    """
    R, L, G, C, Z0_line, _, _ = cpw_rlgc_from_geom(freqs, width, gap, eps_r,
                                                   tan_delta_sub, R_override)
    return abcd_line_from_rlgc(freqs, R, L, G, C, length), Z0_line


def abcd_cpw_bend_as_line(freqs, radius, angle_deg, width, gap, eps_r,
                          tan_delta_sub=0.0, R_override=None):
    """
    ABCD for a CPW bend approximated by arc length line.
    Returns (ABCD, Z0_line).
    """
    arc_len = np.pi * float(radius) * float(angle_deg) / 180.0
    return abcd_cpw_line(freqs, arc_len, width, gap, eps_r, tan_delta_sub, R_override)


def abcd_open_stub_from_f0(freqs, f0_target, width, gap, eps_r,
                           tan_delta_sub=0.0, R_override=None):
    """
    ABCD for a shunt open-circuited CPW stub whose length is computed
    from a target resonance frequency f0_target as L = vp / (4 f0).

    Returns (ABCD_shunt, Z0_line, L_stub_used)
    """
    if f0_target <= 0:
        raise ValueError("f0_target must be positive.")
    # get vp and RLGC once
    R, L, G, C, Z0_line, _, vp = cpw_rlgc_from_geom(freqs, width, gap, eps_r,
                                                    tan_delta_sub, R_override)
    L_stub = vp / (4.0 * float(f0_target))
    abcd = abcd_shunt_open_stub(freqs, R, L, G, C, L_stub)
    return abcd, Z0_line, L_stub


def build_chain_with_stub_from_f0(freqs,
                                  components,
                                  stub_insert_index,
                                  f0_target,
                                  width, gap, eps_r,
                                  tan_delta_sub=0.0, R_override=None,
                                  z0_s_param=None):
    """
    Build a 2-port feedline from explicit components and insert a shunt
    open-circuited λ/4 stub computed from f0_target at position stub_insert_index.

    Parameters
    ----------
    freqs : array_like [Hz]
    components : list of dict
        Each dict is one element in order from Port1 to Port2.
        Supported types:
          {"type":"line", "length": L}
          {"type":"bend", "radius": R, "angle_deg": ang}
    stub_insert_index : int
        Insert the shunt stub AFTER this component index.
        Example: index 1 means after components[1].
        Use -1 to insert at the very beginning.
    f0_target : float [Hz]  target resonance for λ/4
    width, gap, eps_r, tan_delta_sub, R_override : CPW parameters
    z0_s_param : float or None
        Reference impedance for the S-parameters. If None the CPW Z0 is used.

    Returns
    -------
    net_or_tuple :
        If scikit-rf is available, returns skrf.Network (2-port).
        Otherwise returns (freqs, S(2x2), Z0_line, L_stub_used).
    """
    if not components:
        raise ValueError("components must be a non-empty list.")

    # Build ABCD up to insertion point
    abcd_total = None
    Z0_line = None

    def add_block(abcd_block):
        nonlocal abcd_total
        abcd_total = abcd_block if abcd_total is None else abcd_cascade(abcd_total, abcd_block)

    n = len(components)
    if stub_insert_index < -1 or stub_insert_index >= n:
        raise ValueError("stub_insert_index out of range.")

    # left side up to insertion point
    last_z0 = None
    for i, comp in enumerate(components):
        if comp["type"] == "line":
            ab, Z0 = abcd_cpw_line(freqs, comp["length"], width, gap, eps_r,
                                   tan_delta_sub, R_override)
        elif comp["type"] == "bend":
            ab, Z0 = abcd_cpw_bend_as_line(freqs, comp["radius"], comp["angle_deg"],
                                           width, gap, eps_r, tan_delta_sub, R_override)
        else:
            raise ValueError(f"Unknown component type: {comp['type']}")
        add_block(ab)
        last_z0 = Z0
        if i == stub_insert_index:
            break

    if last_z0 is None:
        # inserting at the very beginning (index -1)
        _, last_z0, _ = abcd_open_stub_from_f0(freqs, max(f0_target, 1.0), width, gap, eps_r,
                                               tan_delta_sub, R_override)

    # insert the shunt stub computed from f0
    abcd_stub, Z0_line, L_stub = abcd_open_stub_from_f0(freqs, f0_target, width, gap, eps_r,
                                                        tan_delta_sub, R_override)
    add_block(abcd_stub)

    # right side after insertion point
    for j in range(stub_insert_index + 1, n):
        comp = components[j]
        if comp["type"] == "line":
            ab, _ = abcd_cpw_line(freqs, comp["length"], width, gap, eps_r,
                                  tan_delta_sub, R_override)
        elif comp["type"] == "bend":
            ab, _ = abcd_cpw_bend_as_line(freqs, comp["radius"], comp["angle_deg"],
                                          width, gap, eps_r, tan_delta_sub, R_override)
        add_block(ab)

    # choose S reference
    z0 = Z0_line if z0_s_param is None else float(z0_s_param)
    S = abcd_to_s(abcd_total, z0)

    if _HAVE_SKRF:
        net = to_skrf_network(freqs, S, z0)
        # add metadata for your later inspection
        net.z0 = z0
        net.comment = f"λ/4 stub length from f0={f0_target/1e9:.3f} GHz; L_stub={L_stub*1e6:.2f} µm"
        return net
    return np.asarray(freqs), S, Z0_line, L_stub


# ------------------------- One-port from target f0 -------------------------

def one_port_qw_from_f0(freqs,
                        f0_target,
                        width, gap, eps_r,
                        termination='open',
                        tan_delta_sub=0.0,
                        R_override=None,
                        z0_s_param=None):
    """
    One-port quarter-wave resonator from target frequency.

    Parameters
    ----------
    freqs : array_like [Hz]
    f0_target : float [Hz]  target resonance
    width, gap, eps_r : CPW geometry
    termination : {'open','short'}
        'open'  means far end is open; input is short at resonance.
        'short' means far end is short; input is open at resonance.
    tan_delta_sub, R_override : losses
    z0_s_param : float or None reference for S11

    Returns
    -------
    net_or_tuple :
        If scikit-rf available, returns 1-port Network with S11.
        Else returns (freqs, S11 array, Z0_line, L_stub_used).
    """
    if f0_target <= 0:
        raise ValueError("f0_target must be positive.")
    R, L, G, C, Z0_line, _, vp = cpw_rlgc_from_geom(freqs, width, gap, eps_r,
                                                    tan_delta_sub, R_override)
    Lq = vp / (4.0 * float(f0_target))
    # two-port ABCD of line
    abcd = abcd_line_from_rlgc(freqs, R, L, G, C, Lq)
    A = abcd[:, 0, 0]
    B = abcd[:, 0, 1]
    Cmat = abcd[:, 1, 0]
    D = abcd[:, 1, 1]
    if termination == 'open':
        # ZL = ∞: S11 = (-A + D)/(A + D)
        S11 = (-A + D) / (A + D)
    elif termination == 'short':
        # ZL = 0: S11 = (A - D)/(A + D)
        S11 = (A - D) / (A + D)
    else:
        raise ValueError("termination must be 'open' or 'short'.")

    S = np.zeros((len(freqs), 1, 1), dtype=complex)
    S[:, 0, 0] = S11
    z0 = Z0_line if z0_s_param is None else float(z0_s_param)
    if _HAVE_SKRF:
        return to_skrf_network(freqs, S, z0)
    return np.asarray(freqs), S11, Z0_line, Lq


# frequency grid
f = np.linspace(3e9, 8e9, 3001)

# CPW geometry and materials
width, gap, eps_r = 10e-6, 6e-6, 11.45
tan_delta = 2e-6

# feed components in order from Port1 to Port2
components = [
    {"type": "line", "length": 200e-6},
    {"type": "bend", "radius": 50e-6, "angle_deg": 90},
    {"type": "line", "length": 200e-6},
    {"type": "line", "length": 200e-6},
]

# place the shunt stub after components[1] (after the bend)
stub_after_index = 1
target_f0 = 5.0e9

net = build_chain_with_stub_from_f0(
    f,
    components,
    stub_insert_index=stub_after_index,
    f0_target=target_f0,
    width=width, gap=gap, eps_r=eps_r,
    tan_delta_sub=tan_delta,
    z0_s_param=None  # None means use CPW Z0 as S reference
)

import matplotlib.pyplot as plt
# Example plotting (if scikit-rf present):
net.plot_s_db(m=1, n=0, label="S21")
plt.show()
net.plot_s_deg(m=1, n=0)
plt.show()
