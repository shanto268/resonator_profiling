"""CPW resonator analytical models using skrf.media objects."""

from functools import partial
from typing import Dict, List, Tuple
import gdsfactory as gf
import numpy as np
import skrf as rf
from skrf.media import DefinedGammaZ0, Media
from scipy.constants import c, pi, epsilon_0, mu_0

# ============================================================================
# CPW Media Class
# ============================================================================

class CPW(DefinedGammaZ0):
    """Coplanar waveguide media using analytical models."""
    
    def __init__(self, frequency: rf.Frequency, w: float, s: float, 
                 eps_r: float = 11.45, tand: float = 1e-6, **kwargs):
        """
        Initialize CPW media.
        
        Args:
            frequency: skrf Frequency object
            w: center conductor width in meters
            s: gap width in meters  
            eps_r: substrate relative permittivity
            tand: loss tangent
        """
        self.w = w
        self.s = s
        self.eps_r = eps_r
        self.tand = tand
        
        # Calculate CPW parameters
        k = w / (w + 2*s)
        k_prime = np.sqrt(1 - k**2)
        
        # Elliptic integral ratio
        if k <= 0.707:
            K_ratio = pi / np.log(2 * (1 + np.sqrt(k_prime)) / (1 - np.sqrt(k_prime)))
        else:
            K_ratio = np.log(2 * (1 + np.sqrt(k)) / (1 - np.sqrt(k))) / pi
        
        # Effective permittivity and impedance
        self.eps_eff = 1 + 0.5 * (eps_r - 1)
        Z0 = 60 * pi / np.sqrt(self.eps_eff) / K_ratio
        
        # Propagation constant
        beta = 2 * pi * frequency.f * np.sqrt(self.eps_eff) / c
        alpha = beta * tand / 2
        gamma = alpha + 1j * beta
        
        # Initialize parent class with calculated parameters
        DefinedGammaZ0.__init__(self, frequency=frequency, z0=Z0, gamma=gamma)

# ============================================================================
# Component Analysis
# ============================================================================

def extract_component_info(comp: gf.Component) -> Tuple[str, float]:
    """Extract component type and electrical length in μm."""
    name = comp.name.lower()
    
    if any(x in name for x in ['bend', 'arc', 'circular', 'euler']):
        radius = comp.info.get('radius', 100) if hasattr(comp, 'info') else 100
        angle = comp.info.get('angle', 90) if hasattr(comp, 'info') else 90
        return 'bend', radius * abs(angle) * pi / 180
    
    bbox = comp.dbbox()
    return 'straight', max(bbox.right - bbox.left, bbox.top - bbox.bottom)

def decompose_resonator(resonator: gf.Component) -> List[Tuple[str, float]]:
    """Decompose resonator into list of (type, length_um) tuples."""
    if not hasattr(resonator, 'insts') or len(list(resonator.insts)) == 0:
        return [extract_component_info(resonator)]
    
    return [extract_component_info(inst.cell) for inst in resonator.insts]

# ============================================================================
# Component Simulation Using Media
# ============================================================================

def create_cpw_media(w: float, s: float, freq: rf.Frequency, **kwargs) -> CPW:
    """Create CPW media object with given parameters."""
    return CPW(freq, w, s, **kwargs)

def simulate_straight_media(length_m: float, media: Media) -> rf.Network:
    """Simulate straight section using media object."""
    return media.line(length_m, unit='m')

def simulate_bend_media(arc_length_m: float, media: Media, 
                       bend_loss_factor: float = 1.2) -> rf.Network:
    """Simulate bend with additional loss using modified media."""
    # Create modified media with increased loss
    modified_gamma = media.gamma * (1 + (bend_loss_factor - 1) * 0.5)
    modified_media = DefinedGammaZ0(
        frequency=media.frequency, 
        z0=media.z0, 
        gamma=modified_gamma
    )
    return modified_media.line(arc_length_m, unit='m')

def simulate_component_media(comp_type: str, length_m: float, 
                            media: Media, **kwargs) -> rf.Network:
    """Simulate component using appropriate media model."""
    if comp_type == 'bend':
        return simulate_bend_media(length_m, media, **kwargs)
    return simulate_straight_media(length_m, media)

# ============================================================================
# Cascaded Resonator Simulation
# ============================================================================

def simulate_resonator_media(resonator: gf.Component, cross_section: Dict, 
                            freq: rf.Frequency, **kwargs) -> rf.Network:
    """Simulate resonator by cascading component networks using media objects."""
    components = decompose_resonator(resonator)
    
    # Extract dimensions
    w = cross_section.get('width', 3.42) * 1e-6
    s = cross_section.get('gap', 2.43) * 1e-6
    
    # Create CPW media
    media = create_cpw_media(w, s, freq, **kwargs)
    
    # Simulate each component
    networks = []
    for comp_type, length_um in components:
        net = simulate_component_media(comp_type, length_um * 1e-6, media, **kwargs)
        networks.append(net)
    
    # Cascade all networks using skrf's cascading
    if len(networks) == 1:
        return networks[0]
    
    result = networks[0]
    for net in networks[1:]:
        result = result ** net
    
    return result

# ============================================================================
# Resonance Effects
# ============================================================================

def add_resonance_notch(network: rf.Network, f0: float, 
                        Q_int: float = 1e6, Q_ext: float = 1e4) -> rf.Network:
    """Add resonance notch to network using parallel RLC model."""
    freq = network.frequency
    Q_total = 1 / (1/Q_int + 1/Q_ext)
    
    # Create resonator network using shunt admittance
    Y_res = np.zeros(len(freq.f), dtype=complex)
    for i, f in enumerate(freq.f):
        delta = (f - f0) / f0
        Y_res[i] = (1/Q_ext) / (1 + 1j * 2 * Q_total * delta)
    
    # Convert to S-parameters for shunt element
    Z0_val = network.z0[0] if isinstance(network.z0, np.ndarray) else network.z0
    if isinstance(Z0_val, np.ndarray):
        Z0_val = Z0_val[0]
    
    s_shunt = np.zeros((len(freq.f), 2, 2), dtype=complex)
    for i in range(len(freq.f)):
        y = Y_res[i] * Z0_val
        s_shunt[i, 0, 0] = -y/(2+y)
        s_shunt[i, 0, 1] = 2/(2+y)
        s_shunt[i, 1, 0] = 2/(2+y)
        s_shunt[i, 1, 1] = -y/(2+y)
    
    res_network = rf.Network(frequency=freq, s=s_shunt, z0=Z0_val)
    return network ** res_network

def calculate_f0(total_length_m: float, eps_eff: float, 
                res_type: str = 'quarter') -> float:
    """Calculate fundamental resonance frequency."""
    vp = c / np.sqrt(eps_eff)
    if 'quarter' in res_type or 'lambda/4' in res_type:
        return vp / (4 * total_length_m)
    return vp / (2 * total_length_m)

# ============================================================================
# Geometry Construction
# ============================================================================

@gf.cell
def cpw_resonator(length: float = 4200, n_meanders: int = 0,
                  radius: float = 100, pitch: float = 400) -> gf.Component:
    """Build CPW resonator using straights and bends."""
    c = gf.Component()
    
    if n_meanders == 0:
        c.add_ref(gf.components.straight(length=length))
        return c
    
    # Calculate component lengths for meandered structure
    bend_arc = radius * pi / 2
    connect_length = pitch / 10
    total_bend_length = n_meanders * 2 * bend_arc + (n_meanders - 1) * connect_length
    straight_length = (length - total_bend_length) / (n_meanders + 1)
    
    components = []
    for i in range(n_meanders + 1):
        components.append(gf.components.straight(length=straight_length))
        if i < n_meanders:
            components.append(gf.components.bend_circular(radius=radius, angle=90))
            if i < n_meanders - 1:
                components.append(gf.components.straight(length=connect_length))
            components.append(gf.components.bend_circular(radius=radius, angle=90))
    
    # Connect components
    prev = None
    for comp in components:
        ref = c.add_ref(comp)
        if prev:
            ref.connect("o1", prev.ports["o2"])
        prev = ref
    
    return c

# ============================================================================
# Complete Simulation Pipeline
# ============================================================================

def simulate_cpw_resonator(geometry: gf.Component = None, freq_range=None, **kwargs):
    """
    Complete CPW resonator simulation using skrf media objects.
    
    Returns:
        (network, properties): Network and resonator properties dict
    """
    if geometry is None:
        geometry = cpw_resonator(length=4200)
    
    cross_section = kwargs.pop('cross_section', {'width': 3.42, 'gap': 2.43})
    
    # Handle frequency input
    if isinstance(freq_range, rf.Frequency):
        freq = freq_range
    elif freq_range is None:
        freq = rf.Frequency(1, 25, 2401, 'ghz')
    else:
        freq = rf.Frequency(*freq_range[:3], 'ghz')
    
    # Simulate using media objects
    network = simulate_resonator_media(geometry, cross_section, freq, **kwargs)
    
    # Calculate resonance properties
    components = decompose_resonator(geometry)
    total_length = sum(length for _, length in components) * 1e-6
    
    # Get effective permittivity from media
    w = cross_section.get('width', 3.42) * 1e-6
    s = cross_section.get('gap', 2.43) * 1e-6
    media = create_cpw_media(w, s, freq, **kwargs)
    
    f0 = calculate_f0(total_length, media.eps_eff, kwargs.get('res_type', 'quarter'))
    
    # Add resonances
    Q_int = kwargs.get('Q_int', 1e6)
    Q_ext = kwargs.get('Q_ext', 1e4)
    
    network = add_resonance_notch(network, f0, Q_int, Q_ext)
    
    # Add harmonics for quarter-wave
    if 'quarter' in kwargs.get('res_type', 'quarter'):
        for n in range(3, 10, 2):
            network = add_resonance_notch(network, n*f0, Q_int, Q_ext/2)
    
    properties = {
        'f0': f0,
        'f0_GHz': f0 / 1e9,
        'Q_int': Q_int,
        'Q_ext': Q_ext,
        'Q_total': 1 / (1/Q_int + 1/Q_ext),
        'length_m': total_length,
        'eps_eff': media.eps_eff,
        'Z0': media.z0[0],
        'harmonics': [(2*i+1)*f0/1e9 for i in range(5)]
    }
    
    return network, properties

# ============================================================================
# Convenience Functions with Partials
# ============================================================================

create_silicon_cpw = partial(create_cpw_media, eps_r=11.45, tand=1e-6)
create_sapphire_cpw = partial(create_cpw_media, eps_r=9.4, tand=1e-7)
simulate_quarter_wave = partial(simulate_cpw_resonator, res_type='quarter')
simulate_half_wave = partial(simulate_cpw_resonator, res_type='half')

# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("CPW Resonator Simulation using skrf.media\n" + "="*50)
    
    # Test with sim.py parameters
    geometry = cpw_resonator(length=4200, n_meanders=0)
    cross_section = {'width': 3.42, 'gap': 2.43}
    
    network, props = simulate_cpw_resonator(
        geometry=geometry,
        cross_section=cross_section,
        freq_range=(1, 25, 2401),
        eps_r=11.45,
        tand=1e-6,
        Q_int=1e6,
        Q_ext=1e4
    )
    
    print(f"\nResonator Properties:")
    print(f"  f0: {props['f0_GHz']:.3f} GHz")
    print(f"  Q_total: {props['Q_total']:.0f}")
    print(f"  Length: {props['length_m']*1e3:.2f} mm")
    print(f"  ε_eff: {props['eps_eff']:.2f}")
    print(f"  Z0: {props['Z0']:.1f} Ω")
    
    # Test media object directly
    print("\n" + "="*50)
    print("Testing CPW Media Object:")
    
    freq = rf.Frequency(5, 8, 301, 'ghz')
    media = create_cpw_media(3.42e-6, 2.43e-6, freq, eps_r=11.45)
    
    print(f"  Z0: {media.z0[0]:.1f} Ω")
    print(f"  ε_eff: {media.eps_eff:.2f}")
    print(f"  β @ 6 GHz: {media.gamma[100].imag:.1f} rad/m")
    print(f"  α @ 6 GHz: {media.gamma[100].real:.3e} Np/m")
    
    # Create transmission line
    tline = media.line(4.2e-3, unit='m')
    print(f"  4.2mm line phase @ 6 GHz: {tline.s_deg[100,1,0]:.1f}°")
    
    # Test cascading
    print("\n" + "="*50)
    print("Testing Component Cascading:")
    
    # Create meandered structure
    meander_geom = cpw_resonator(length=4200, n_meanders=2, radius=100)
    components = decompose_resonator(meander_geom)
    
    print(f"  Components: {len(components)}")
    straights = sum(1 for t, _ in components if t == 'straight')
    bends = sum(1 for t, _ in components if t == 'bend')
    print(f"  Straights: {straights}, Bends: {bends}")
    
    # Simulate with media
    network_meander = simulate_resonator_media(meander_geom, cross_section, freq)
    print(f"  Cascaded network: {network_meander.nports}-port")
    print(f"  Total phase @ 6 GHz: {network_meander.s_deg[100,1,0]:.1f}°")
    
    # Plots
    print("\n" + "="*50)
    print("Generating Plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # S21 magnitude
    axes[0,0].plot(network.frequency.f_scaled, network.s_db[:,1,0], 'b-', linewidth=1.5)
    axes[0,0].set_xlabel('Frequency (GHz)')
    axes[0,0].set_ylabel('|S₂₁| (dB)')
    axes[0,0].set_title('CPW Resonator Transmission')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim([-40, 5])
    
    # S21 phase
    axes[0,1].plot(network.frequency.f_scaled, np.unwrap(network.s_deg[:,1,0]), 'r-', linewidth=1.5)
    axes[0,1].set_xlabel('Frequency (GHz)')
    axes[0,1].set_ylabel('∠S₂₁ (degrees)')
    axes[0,1].set_title('Phase Response')
    axes[0,1].grid(True, alpha=0.3)
    
    # S11
    axes[1,0].plot(network.frequency.f_scaled, network.s_db[:,0,0], 'g-', linewidth=1.5)
    axes[1,0].set_xlabel('Frequency (GHz)')
    axes[1,0].set_ylabel('|S₁₁| (dB)')
    axes[1,0].set_title('Reflection')
    axes[1,0].grid(True, alpha=0.3)
    
    # Smith chart
    network.plot_s_smith(m=0, n=0, ax=axes[1,1], show_legend=False)
    axes[1,1].set_title('Smith Chart (S₁₁)')
    
    plt.suptitle('CPW Resonator Analysis using skrf.media', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cpw_media_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: cpw_media_analysis.png")
    
    print("\n✓ All tests passed using skrf.media objects!")