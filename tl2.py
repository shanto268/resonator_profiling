
import numpy as np
import skrf as rf
from mpmath import ellipk

def transmission_line_network(freqs, length, line_type,
                              w=None, g=None, h=None, d=None, r=None,
                              eps_r=1.0, tan_delta=0.0,
                              metal='PEC'):
    """
    Generalized analytical transmission line model -> scikit-rf Network.
    
    Parameters
    ----------
    freqs : array_like
        Frequency points in Hz.
    length : float
        Line length in meters.
    line_type : str
        One of ['cpw', 'microstrip', 'stripline', 'coax'].
    w : float, optional
        Trace width (for cpw, microstrip, stripline) in meters.
    g : float, optional
        Gap (for CPW) in meters.
    h : float, optional
        Substrate height in meters.
    d : float, optional
        Ground separation (for stripline) in meters.
    r : tuple, optional
        Inner and outer radius (for coax), e.g. (a, b).
    eps_r : float
        Substrate relative permittivity.
    tan_delta : float
        Loss tangent of substrate.
    metal : str
        'PEC' (lossless) or 'finite' (adds conductor loss placeholder).
    
    Returns
    -------
    skrf.Network
    """
    freqs = np.asarray(freqs)
    f = rf.Frequency.from_f(freqs, unit='Hz')
    c0 = 3e8
    
    if line_type.lower() == 'cpw':
        if w is None or g is None or h is None:
            raise ValueError("CPW requires w, g, h")
        k = w / (w + 2*g)
        k_p = np.sqrt(1 - k**2)
        Kk = float(ellipk(k**2))
        Kkp = float(ellipk(k_p**2))
        eps_eff = (eps_r + 1)/2  # simple Hammerstad approx
        Z0 = (30*np.pi / np.sqrt(eps_eff)) * (Kkp / Kk)
        
    elif line_type.lower() == 'microstrip':
        if w is None or h is None:
            raise ValueError("Microstrip requires w, h")
        # Hammerstad & Jensen
        u = w/h
        eps_eff = (eps_r + 1)/2 + (eps_r - 1)/(2*np.sqrt(1+12/u))
        if u <= 1:
            Z0 = (60/np.sqrt(eps_eff)) * np.log(8/u + 0.25*u)
        else:
            Z0 = (120*np.pi) / (np.sqrt(eps_eff) * (u + 1.393 + 0.667*np.log(u+1.444)))
            
    elif line_type.lower() == 'stripline':
        if w is None or d is None:
            raise ValueError("Stripline requires w, d")
        # Wheeler's stripline approximation
        eps_eff = eps_r
        Z0 = (30*np.pi/np.sqrt(eps_r)) * (1/(w/d + 0.441))
        
    elif line_type.lower() == 'coax':
        if r is None:
            raise ValueError("Coax requires r=(a,b)")
        a, b = r
        eps_eff = eps_r
        Z0 = (60/np.sqrt(eps_r)) * np.log(b/a)
        
    else:
        raise ValueError(f"Unsupported line_type: {line_type}")
    
    # Velocity, capacitance and inductance
    v_p = c0 / np.sqrt(eps_eff)
    C_per_m = 1 / (Z0 * v_p)
    L_per_m = Z0**2 * C_per_m
    
    # Losses
    G_per_m = 2*np.pi*freqs * C_per_m * tan_delta
    if metal.upper() == 'PEC':
        R_per_m = np.zeros_like(freqs)
    else:
        # crude skin-effect scaling ~ sqrt(f), user should refine
        sigma = 5.8e7  # copper
        delta = np.sqrt(1/(np.pi*freqs*4e-7*np.pi*sigma))
        R_per_m = 1/(w*delta) if w else np.zeros_like(freqs)
    
    # Build distributed circuit
    dc = rf.media.DistributedCircuit(
        frequency=f,
        R=R_per_m,
        L=L_per_m,
        G=G_per_m,
        C=C_per_m,
        z0=Z0
    )
    
    return dc.line(length, unit='m')

if __name__ == "__main__":
    freqs = np.linspace(1e9, 10e9, 201)
    cpw_net = transmission_line_network(freqs, length=5e-3,
                                        line_type='cpw',
                                        w=10e-6, g=6e-6, h=300e-6,
                                        eps_r=11.45, tan_delta=0.001)
    print(cpw_net)

    ms_net = transmission_line_network(freqs, length=10e-3,
                                    line_type='microstrip',
                                    w=2e-3, h=1.6e-3,
                                    eps_r=4.4, tan_delta=0.02)

    coax_net = transmission_line_network(freqs, length=1.0,
                                        line_type='coax',
                                        r=(0.5e-3, 2.95e-3),  # RG405
                                        eps_r=2.1, tan_delta=0.0002)
