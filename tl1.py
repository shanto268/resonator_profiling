import numpy as np
import skrf as rf
from mpmath import ellipk
from skrf.media.distributedCircuit import DistributedCircuit


def cpw_network(freqs, length, width, gap, thickness, h_sub,
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

    G_per_m = 2 * np.pi * freqs * C_per_m * tan_delta
    R_per_m = np.zeros_like(freqs) if metal.upper() == 'PEC' else np.ones_like(freqs)

    dc = DistributedCircuit(
        frequency=f,
        R=R_per_m,
        L=L_per_m,
        G=G_per_m,
        C=C_per_m,
        z0=Z0
    )

    return dc.line(length, unit='m')



def cpw_bend_network(freqs, radius, angle_deg,
                     width, gap, thickness, h_sub,
                     eps_r, tan_delta=0.0, metal='PEC'):
    """
    Analytical CPW bend approximated as a straight line of arc length.
    
    Parameters
    ----------
    freqs : array_like
        Frequency points in Hz.
    radius : float
        Bend radius (m).
    angle_deg : float
        Bend angle in degrees.
    Other parameters same as cpw_network.
    
    Returns
    -------
    skrf.Network
    """
    arc_length = (np.pi * radius * angle_deg) / 180.0
    return cpw_network(freqs, arc_length,
                       width, gap, thickness, h_sub,
                       eps_r, tan_delta, metal)


def chain_cpw(components):
    """
    Cascade a list of Networks into a single Network.
    
    Parameters
    ----------
    components : list of skrf.Network
        Networks to cascade in order.
    
    Returns
    -------
    skrf.Network
    """
    net = components[0]
    for nxt in components[1:]:
        net = rf.cascade(net, nxt)
    return net

freqs = np.linspace(1e9, 10e9, 201)

# Straight section 1 (100 µm long)
s1 = cpw_network(freqs, length=100e-6,
                 width=10e-6, gap=6e-6, thickness=0.5e-6,
                 h_sub=300e-6, eps_r=11.45, tan_delta=0.001)

# 90° bend, radius 50 µm
b1 = cpw_bend_network(freqs, radius=50e-6, angle_deg=90,
                      width=10e-6, gap=6e-6, thickness=0.5e-6,
                      h_sub=300e-6, eps_r=11.45, tan_delta=0.001)

# Straight section 2 (200 µm long)
s2 = cpw_network(freqs, length=200e-6,
                 width=10e-6, gap=6e-6, thickness=0.5e-6,
                 h_sub=300e-6, eps_r=11.45, tan_delta=0.001)

# Cascade into one equivalent network
cpw_total = chain_cpw([s1, b1, s2])

print(cpw_total)
print(cpw_total)
