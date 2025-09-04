"""
Quarter-wave CPW resonator builder (component-wise, ABCD), with shorted far end.

Flow
----
- For each component (straight or bend), compute RLGC'(f) from geometry/materials
- Build 2-port ABCD for that component
- Cascade ABCDs in order (Port1 -> ... -> Port2)
- Apply short termination at Port2 to form a quarter-wave resonator
- Return a one-port skrf.Network with S11 of the resonator
- Optionally also return the un-terminated 2-port (through) for debugging

Units
-----
- Frequencies: Hz
- Lengths: meters
- Permittivities: dimensionless
- Loss tangent: dimensionless
- Impedances: ohms

Dependencies
------------
- numpy, scipy, scikit-rf (>=1.0), all commonly available in conda envs.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.special import ellipk

import skrf as rf


# ------------------------------ CPW physics core ------------------------------

def _cpw_char(width: float, gap: float, eps_r: float) -> Tuple[float, float, float, float, float]:
    """
    Characteristic values for CPW (simple conformal mapping model).

    Parameters
    ----------
    width : float
        CPW center conductor width [m]
    gap : float
        CPW slot-to-ground spacing [m]
    eps_r : float
        Substrate relative permittivity [-]

    Returns
    -------
    Z0 : float [ohm]
    eps_eff : float [-]
    v_p : float [m/s]
    L_per_m : float [H/m]
    C_per_m : float [F/m]

    Notes
    -----
    Z0 ≈ (30π/√eps_eff) * K(k')/K(k), with k = w/(w+2g),
    K(m) = complete elliptic integral of the first kind with parameter m=k^2.
    eps_eff ≈ (eps_r + 1)/2 (first-order CPW approximation).
    """
    if width <= 0 or gap <= 0:
        raise ValueError("width and gap must be > 0.")
    k = width / (width + 2.0 * gap)
    if not (0.0 < k < 1.0):
        raise ValueError("Invalid CPW geometry: width/(width+2*gap) must lie in (0,1).")

    Kk = float(ellipk(k**2))
    Kkp = float(ellipk(1.0 - k**2))

    eps_eff = 0.5 * (eps_r + 1.0)
    Z0 = (30.0 * np.pi / np.sqrt(eps_eff)) * (Kkp / Kk)

    c0 = 299_792_458.0
    v_p = c0 / np.sqrt(eps_eff)

    C_per_m = 1.0 / (Z0 * v_p)
    L_per_m = Z0**2 * C_per_m
    return Z0, eps_eff, v_p, L_per_m, C_per_m


def cpw_rlgc_from_geom(
    freqs: np.ndarray,
    width: float,
    gap: float,
    eps_r: float,
    tan_delta_sub: float = 0.0,
    R_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Per-unit-length RLGC for a uniform CPW derived from geometry/materials.

    Parameters
    ----------
    freqs : array_like
        Frequency axis [Hz], shape (N,)
    width, gap, eps_r : see _cpw_char()
    tan_delta_sub : float, optional
        Effective dielectric loss tangent for the substrate [-]
    R_override : array_like or None, optional
        If provided, overrides conductor loss per meter R'(f) [ohm/m].
        Shape must match freqs.

    Returns
    -------
    R, L, G, C : ndarray
        Per-unit-length arrays shaped like freqs
    Z0, eps_eff, v_p : floats
        Characteristic impedance, effective permittivity, phase velocity
    """
    f = np.asarray(freqs, dtype=float)
    if f.ndim != 1 or f.size == 0:
        raise ValueError("freqs must be a non-empty 1D array of Hz.")

    Z0, eps_eff, v_p, Lp, Cp = _cpw_char(width, gap, eps_r)

    L = np.full_like(f, Lp)
    C = np.full_like(f, Cp)
    omega = 2.0 * np.pi * f
    G = omega * C * float(tan_delta_sub)

    if R_override is not None:
        R = np.asarray(R_override, dtype=float)
        if R.shape != f.shape:
            raise ValueError("R_override must have same shape as freqs.")
    else:
        # Default PEC (superconducting ideal) → R' = 0
        R = np.zeros_like(f)

    return R, L, G, C, Z0, eps_eff, v_p


# ------------------------------ ABCD primitives ------------------------------

def abcd_line_from_rlgc(
    freqs: np.ndarray,
    R: np.ndarray,
    L: np.ndarray,
    G: np.ndarray,
    C: np.ndarray,
    length: float,
) -> np.ndarray:
    """
    ABCD of a lossy uniform line of length 'length' for given RLGC'(f).

    ABCD = [[cosh(γl),   Zc sinh(γl)],
            [sinh(γl)/Zc, cosh(γl)]]

    where γ = sqrt((R + jωL)(G + jωC)), Zc = sqrt((R + jωL)/(G + jωC)).
    """
    if length <= 0:
        raise ValueError("length must be > 0.")
    f = np.asarray(freqs, dtype=float)
    jw = 1j * 2.0 * np.pi * f

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

    abcd = np.empty((f.size, 2, 2), dtype=complex)
    abcd[:, 0, 0] = A
    abcd[:, 0, 1] = B
    abcd[:, 1, 0] = Cmat
    abcd[:, 1, 1] = D
    return abcd


def abcd_cascade(*abcd_list: np.ndarray) -> np.ndarray:
    """
    Vectorized 2x2 matrix cascade over frequency axis.
    out = A @ B @ C ... for each frequency.

    Each input has shape (N,2,2).
    """
    if not abcd_list:
        raise ValueError("provide at least one ABCD matrix")
    out = np.asarray(abcd_list[0], dtype=complex)
    for nxt in abcd_list[1:]:
        B = np.asarray(nxt, dtype=complex)
        A11 = out[:, 0, 0] * B[:, 0, 0] + out[:, 0, 1] * B[:, 1, 0]
        A12 = out[:, 0, 0] * B[:, 0, 1] + out[:, 0, 1] * B[:, 1, 1]
        A21 = out[:, 1, 0] * B[:, 0, 0] + out[:, 1, 1] * B[:, 1, 0]
        A22 = out[:, 1, 0] * B[:, 0, 1] + out[:, 1, 1] * B[:, 1, 1]
        out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = A11, A12, A21, A22
    return out


# ------------------------------ S conversion helpers ------------------------------

def s2p_from_abcd(abcd: np.ndarray, z0_ref: float) -> np.ndarray:
    """
    Manual ABCD→S conversion (version-independent; no rf.abcd2s dependency).

    Parameters
    ----------
    abcd : ndarray
        Shape (N,2,2)
    z0_ref : float
        Scalar reference impedance for both ports [ohm]

    Returns
    -------
    S : ndarray
        Shape (N,2,2)
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


def s11_from_abcd_with_short(abcd: np.ndarray) -> np.ndarray:
    """
    Input reflection at port-1 when port-2 is shorted (ZL=0).

    For a 2-port with ABCD = [[A,B],[C,D]]:
      S11_in|ZL=0 = (A - D) / (A + D)
    (obtained from the standard terminated-2port relations).

    Returns
    -------
    S11 : ndarray, shape (N,)
    """
    A = abcd[:, 0, 0]
    D = abcd[:, 1, 1]
    return (A - D) / (A + D)


# ------------------------------ Component-wise builders ------------------------------

def _bend_arc_length(radius: float, angle_deg: float) -> float:
    if radius <= 0:
        raise ValueError("bend radius must be > 0.")
    return np.pi * float(radius) * float(angle_deg) / 180.0


def build_qw_resonator_chain(
    freqs: np.ndarray,
    components: List[Dict],
    *,
    width: float,
    gap: float,
    eps_r: float,
    tan_delta_sub: float = 0.0,
    R_override: Optional[np.ndarray] = None,
    z0_s_param: Optional[float] = None,
    return_through_twoport: bool = False,
) -> Tuple[rf.Network, Optional[rf.Network]]:
    """
    Build a **quarter-wave CPW resonator** from component list and return a 1-port Network (S11).

    The device is a series chain of CPW elements (lines/bends) that ends in a **short to ground**
    at the far end (Port-2). The result is a 1-port resonator seen from Port-1.

    Parameters
    ----------
    freqs : array_like
        Frequency axis [Hz], shape (N,)
    components : list of dict
        In order from Port1 to Port2. Each dict is one of:
          {"type": "line", "length": <float meters>}
          {"type": "bend", "radius": <float meters>, "angle_deg": <float degrees>}
        You can include any number of lines and bends.
    width, gap, eps_r : floats
        CPW cross-section and substrate permittivity
    tan_delta_sub : float, optional
        Effective dielectric loss tangent
    R_override : array_like or None, optional
        Conductor loss per meter R'(f) [ohm/m]. If provided, shape must match freqs.
        Use this to inject superconducting surface-resistance models.
    z0_s_param : float or None, optional
        Reference impedance for resulting Networks. If None, uses line CPW Z0.
    return_through_twoport : bool, optional
        If True, also return the un-terminated 2-port (through) network for debugging.

    Returns
    -------
    net_res : skrf.Network
        1-port Network (S11) of the quarter-wave resonator (far end shorted)
    net_thru : skrf.Network or None
        2-port through line (no termination) if return_through_twoport=True, else None
    """
    if not components:
        raise ValueError("components must be a non-empty list of dicts.")
    f = np.asarray(freqs, dtype=float)
    # RLGC once for this geometry
    R, L, G, C, Z0_line, _, _ = cpw_rlgc_from_geom(
        f, width, gap, eps_r, tan_delta_sub, R_override
    )
    z0 = Z0_line if z0_s_param is None else float(z0_s_param)

    # ABCD per component
    abcd_blocks = []
    for i, comp in enumerate(components):
        ctype = comp.get("type", "").lower()
        if ctype == "line":
            length = float(comp["length"])
            ab = abcd_line_from_rlgc(f, R, L, G, C, length)
        elif ctype == "bend":
            arc = _bend_arc_length(float(comp["radius"]), float(comp["angle_deg"]))
            ab = abcd_line_from_rlgc(f, R, L, G, C, arc)
        else:
            raise ValueError(f"Unknown component type at index {i}: {comp}")
        abcd_blocks.append(ab)

    # Cascade all components
    abcd_total = abcd_cascade(*abcd_blocks)

    # 2-port through (for debugging / comparison)
    S_through = s2p_from_abcd(abcd_total, z0)
    freq_obj_2p = rf.Frequency.from_f(f, unit="Hz")
    net_through = rf.Network(frequency=freq_obj_2p, s=S_through) if return_through_twoport else None

    # Apply short at Port-2 → 1-port resonator S11
    S11 = s11_from_abcd_with_short(abcd_total)
    S1 = np.zeros((f.size, 1, 1), dtype=complex)
    S1[:, 0, 0] = S11
    freq_obj_1p = rf.Frequency.from_f(f, unit="Hz")
    net_res = rf.Network(frequency=freq_obj_1p, s=S1)

    # annotate
    net_res.comment = "Quarter-wave CPW resonator (far end shorted); component-wise ABCD cascade"
    if net_through is not None:
        net_through.comment = "CPW chain (through), no termination"

    return net_res, net_through

# Frequency axis
f = np.linspace(3e9, 8e9, 3001)

# CPW cross-section and materials (all lengths in meters)
width = 10e-6
gap   = 6e-6
eps_r = 11.45
tan_delta = 2e-6   # effective dielectric loss tangent
Rprime = None      # or an array R'(f) to include conductor loss (e.g., superconducting surface resistance)

# Your component-wise geometry (add as many straights/bends as you like)
components = [
    {"type": "line", "length": 200e-6},
    {"type": "bend", "radius": 50e-6, "angle_deg": 90},
    {"type": "line", "length": 200e-6},
    {"type": "bend", "radius": 50e-6, "angle_deg": 90},
    {"type": "line", "length": 300e-6},
]

# Build the 1-port quarter-wave resonator (far end shorted).
# Also get the 2-port "through" for sanity checks.
res, thru = build_qw_resonator_chain(
    f, components,
    width=width, gap=gap, eps_r=eps_r,
    tan_delta_sub=tan_delta, R_override=Rprime,
    z0_s_param=None,                 # None → use CPW Z0 as S reference
    return_through_twoport=True
)

# Plot (requires matplotlib):
res.plot_s_db(m=0,n=0, label="Resonator S11")
res.plot_s_deg(m=0,n=0)
plt.show()
# if thru: thru.plot_s_db(m=1,n=0, label="Through S21")

# Find approximate resonance (phase flip or |S11| extremum)
idx = np.argmin(np.abs(res.s[:,0,0]))  # dip if losses present (else use phase)
print(f"~Resonance near {res.f[idx]/1e9:.3f} GHz")
