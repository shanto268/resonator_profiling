import numpy as np
import skrf as rf
from skrf.media.distributedCircuit import DistributedCircuit
from mpmath import ellipk
import matplotlib.pyplot as plt

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

def cpw_line(freqs, length, **kwargs):
    dc = cpw_media(freqs, **kwargs)
    return dc.line(length, unit='m')

def cpw_bend(freqs, radius, angle_deg, **kwargs):
    arc_length = (np.pi * radius * angle_deg) / 180.0
    return cpw_line(freqs, arc_length, **kwargs)

def cpw_quarterwave_stub(freqs, target_freq, **kwargs):
    """
    Quarter-wave CPW stub resonator (1-port).
    """
    dc = cpw_media(freqs, **kwargs)
    eps_eff = (kwargs["eps_r"] + 1)/2
    vp = 3e8 / np.sqrt(eps_eff)
    L_qw = vp / (4 * target_freq)
    return dc.line(L_qw, unit="m") ** dc.short()

def chain_cpw(components):
    net = components[0]
    for nxt in components[1:]:
        net = rf.cascade(net, nxt)
    return net

freqs = np.linspace(1e9, 10e9, 2001)
params = dict(width=10e-6, gap=6e-6, thickness=0.2e-6,
              h_sub=500e-6, eps_r=11.45, tan_delta=1e-6, metal='PEC')

# Media
dc = cpw_media(freqs, **params)

# Feedline: straight + bend + straight
feedline = chain_cpw([
    cpw_line(freqs, length=200e-6, **params),
    cpw_bend(freqs, radius=50e-6, angle_deg=90, **params),
    cpw_line(freqs, length=200e-6, **params)
])

# Quarter-wave resonator stub
stub = cpw_quarterwave_stub(freqs, target_freq=5e9, **params)

# Couple stub to feedline (hanger resonator geometry)
# Port mapping: connect stub port 0 to feedline node
full_net = rf.connect(feedline, 1, stub, 0)

full_net.plot_s_db()
plt.show()

full_net.plot_s_deg()
plt.show()

