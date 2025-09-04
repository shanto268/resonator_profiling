import numpy as np
import skrf as rf
from skrf.media.distributedCircuit import DistributedCircuit
from mpmath import ellipk

def cpw_media(freqs, width, gap, thickness, h_sub,
              eps_r, tan_delta=0.0, metal='PEC'):
    freqs = np.asarray(freqs)
    f = rf.Frequency.from_f(freqs, unit='Hz')
    c0 = 3e8

    k = width / (width + 2 * gap)
    k_p = np.sqrt(1 - k**2)
    Kk = float(ellipk(k**2))
    Kkp = float(ellipk(k_p**2))

    eps_eff = (eps_r + 1) / 2
    Z0 = (30 * np.pi / np.sqrt(eps_eff)) * (Kkp / Kk)

    v_p = c0 / np.sqrt(eps_eff)
    C_per_m = 1 / (Z0 * v_p)
    L_per_m = Z0**2 * C_per_m

    omega = 2 * np.pi * freqs
    G_per_m = omega * C_per_m * tan_delta
    R_per_m = np.zeros_like(freqs) if metal.upper() == 'PEC' else np.ones_like(freqs)

    dc = DistributedCircuit(
        frequency=f,
        R=R_per_m,
        L=L_per_m,
        G=G_per_m,
        C=C_per_m,
        z0=Z0
    )
    return dc


def cpw_quarter_wave_resonator(freqs, target_freq, width, gap, thickness, h_sub,
                               eps_r, tan_delta=0.0, Q_loaded=5000):
    """
    One-port quarter-wave CPW resonator terminated in a short,
    with finite Q so that S11 shows a dip.

    Parameters
    ----------
    freqs : array
        Frequency sweep [Hz].
    target_freq : float
        Desired resonance frequency [Hz] (fundamental).
    width, gap, thickness, h_sub, eps_r, tan_delta : geometry and material.
    Q_loaded : float
        Loaded quality factor to set resonance linewidth.

    Returns
    -------
    skrf.Network
    """

    # Build media
    dc = cpw_media(freqs, width, gap, thickness, h_sub, eps_r, tan_delta)

    # Effective permittivity -> phase velocity
    eps_eff = (eps_r + 1)/2
    v_p = 3e8 / np.sqrt(eps_eff)

    # Quarter-wave length for target frequency
    L_qw = v_p / (4 * target_freq)

    # Line terminated in short
    short = dc.short()
    ideal_res = dc.line(L_qw, unit='m') ** short

    # Add loaded Q (simple notch filter model)
    s11 = ideal_res.s[:,0,0]
    f0 = target_freq
    df = f0 / Q_loaded
    lorentz = 1 - (df / (1j*(freqs - f0) + df))
    s11_with_Q = s11 * lorentz

    # Wrap back into a Network
    ntw = rf.Network(frequency=dc.frequency, s=s11_with_Q[:,None,None])
    return ntw


freqs = np.linspace(1e9, 10e9, 2001)

res = cpw_quarter_wave_resonator(freqs,
                                 target_freq=5e9,
                                 width=10e-6, gap=6e-6, thickness=0.2e-6,
                                 h_sub=500e-6, eps_r=11.45,
                                 tan_delta=1e-6, Q_loaded=10000)

import matplotlib.pyplot as plt
res.plot_s_db(m=0, n=0)
# res.plot_s_db(m=0, n=1)
plt.show()

res.plot_s_deg(m=0, n=0)
# res.plot_s_deg(m=0, n=1)
plt.show()
