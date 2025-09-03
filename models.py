"""
Analytical models for CPW transmission lines and resonators.
Functional implementation with kwargs and partial application.
"""

from functools import partial
from typing import Any, Dict, Tuple

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import skrf as rf
from skrf.media import DefinedGammaZ0

# ============================================================================
# Core Physics Functions
# ============================================================================

def cpw_parameters(width_m: float, gap_m: float, eps_r: float = 11.45) -> Tuple[float, float]:
    """Calculate CPW effective permittivity and characteristic impedance."""
    k = width_m / (width_m + 2 * gap_m)
    k_prime = np.sqrt(1 - k**2)
    
    # Elliptic integral ratio approximation
    K_ratio = (const.pi / np.log(2 * (1 + np.sqrt(k_prime))/(1 - np.sqrt(k_prime))) 
               if k <= 0.7 else 
               np.log(2 * (1 + np.sqrt(k))/(1 - np.sqrt(k))) / const.pi)
    
    eps_eff = 1 + (eps_r - 1) * 0.5  # CPW on substrate
    z0 = 30 * const.pi / np.sqrt(eps_eff) / K_ratio
    
    return eps_eff, z0


def get_component_length(component: gf.Component) -> float:
    """Extract total electrical length from gdsfactory component (μm)."""
    if hasattr(component, 'insts') and len(list(component.insts)) > 0:
        # Hierarchical - sum all sub-components
        return sum(get_component_length(inst.cell) for inst in component.insts)
    else:
        # Base component - use bounding box
        dbbox = component.dbbox()
        if 'bend' in component.name.lower():
            # Arc length for bend (assuming 90-degree)
            radius = (dbbox.right - dbbox.left) / 2
            return radius * np.pi / 2
        return dbbox.right - dbbox.left


def extract_cpw_dims(cross_section: Any) -> Tuple[float, float]:
    """Extract CPW dimensions from cross_section dict or object (returns meters)."""
    if hasattr(cross_section, 'width'):
        width = cross_section.width * 1e-6
        width_exclude = getattr(cross_section, 'width_exclude', cross_section.width + 12)
        gap = (width_exclude - cross_section.width) / 2 * 1e-6
    else:
        width = cross_section.get('width', 10) * 1e-6
        gap = cross_section.get('gap', 6) * 1e-6
    return width, gap


# ============================================================================
# Transmission Line Model
# ============================================================================

def tline_network(length_m: float, freq: rf.Frequency, **kwargs) -> rf.Network:
    """
    Create transmission line S-parameter network.
    
    kwargs:
        width_um, gap_um: CPW dimensions in micrometers
        eps_r: substrate permittivity (default 11.45)
        tand: loss tangent (default 1e-6)
    """
    width = kwargs.get('width_um', 10) * 1e-6
    gap = kwargs.get('gap_um', 6) * 1e-6
    eps_r = kwargs.get('eps_r', 11.45)
    tand = kwargs.get('tand', 1e-6)
    
    eps_eff, z0 = cpw_parameters(width, gap, eps_r)
    
    # Propagation constant
    beta = 2 * const.pi * freq.f * np.sqrt(eps_eff) / const.c
    alpha = beta * tand / 2
    gamma = alpha + 1j * beta
    
    # Create network
    media = DefinedGammaZ0(frequency=freq, z0=z0, gamma=gamma)
    return media.line(length_m, unit='m', name=f'CPW_L{length_m*1e3:.1f}mm')


def simulate_tline(component: gf.Component, cross_section: Any, freq: rf.Frequency, **kwargs) -> rf.Network:
    """
    Simulate transmission line, handling hierarchical components.
    
    kwargs:
        eps_r: substrate permittivity
        tand: loss tangent
        recursive: whether to recurse into sub-components (default True)
    """
    recursive = kwargs.get('recursive', True)
    
    if not recursive or len(list(component.insts)) == 0:
        # Simple component or non-recursive
        length_um = get_component_length(component)
        width, gap = extract_cpw_dims(cross_section)
        return tline_network(length_um * 1e-6, freq, 
                           width_um=width*1e6, gap_um=gap*1e6, **kwargs)
    
    # Hierarchical - cascade sub-components
    networks = []
    for inst in component.insts:
        net = simulate_tline(inst.cell, cross_section, freq, **kwargs)
        networks.append(net)
    
    # Cascade all networks
    result = networks[0]
    for net in networks[1:]:
        result = rf.cascade(result, net)
    result.name = component.name
    return result


# ============================================================================
# Resonator Model
# ============================================================================

def resonator_frequency(length_m: float, resonator_type: str = "lambda/4", **kwargs) -> float:
    """Calculate resonance frequency for CPW resonator."""
    eps_r = kwargs.get('eps_r', 11.45)
    eps_eff = 1 + (eps_r - 1) * 0.5
    
    if resonator_type == "lambda/4" or resonator_type == "quarter_wave" or resonator_type == "quarter":
        return const.c / (4 * length_m * np.sqrt(eps_eff))
    elif resonator_type == "lambda/2" or resonator_type == "half_wave" or resonator_type == "half":  # lambda/2
        return const.c / (2 * length_m * np.sqrt(eps_eff))


def resonator_response(freq: rf.Frequency, f0: float, Q_total: float, 
                      Qc: float, **kwargs) -> rf.Network:
    """
    Generate resonator S-parameters.
    
    kwargs:
        coupling: 'capacitive' or 'inductive'
        resonator_type: 'lambda/4' or 'lambda/2'
    """
    f = freq.f
    delta = (f - f0) / f0  # Normalized detuning
    beta = Q_total / Qc  # Coupling strength
    
    # Resonator response
    denominator = 1 + 2j * Q_total * delta
    S21 = 1 - beta / denominator
    S11 = -beta / (2 * denominator)
    
    # Build S-matrix
    s_matrix = np.zeros((len(f), 2, 2), dtype=complex)
    s_matrix[:, 0, 0] = s_matrix[:, 1, 1] = S11
    s_matrix[:, 0, 1] = s_matrix[:, 1, 0] = S21
    
    return rf.Network(frequency=freq, s=s_matrix, name='CPW_Resonator')


def simulate_resonator(component: gf.Component, cross_section: Any, 
                       freq: rf.Frequency, **kwargs) -> Tuple[rf.Network, Dict]:
    """
    Simulate CPW resonator with hierarchical support.
    
    kwargs:
        resonator_type: 'lambda/4' or 'lambda/2' (default 'lambda/4')
        Qc: coupling Q factor (default 1e4)
        Qi: internal Q factor (default 1e6)
        eps_r: substrate permittivity (default 11.45)
    """
    resonator_type = kwargs.get('resonator_type', 'lambda/4')
    Qc = kwargs.pop('Qc', 1e4)  
    Qi = kwargs.pop('Qi', 1e6)
    
    # Get total length and calculate resonance
    length_um = get_component_length(component)
    length_m = length_um * 1e-6
    f0 = resonator_frequency(length_m, resonator_type, **kwargs)
    
    # Calculate Q factors
    Q_total = 1 / (1/Qc + 1/Qi)
    kappa = f0 / Q_total
    
    # Generate network (don't pass Qc again in kwargs)
    network = resonator_response(freq, f0, Q_total, Qc, **kwargs)
    
    # Return network and properties
    properties = {
        'f0': f0, 'f0_GHz': f0/1e9,
        'Q_total': Q_total, 'Qc': Qc, 'Qi': Qi,
        'kappa': kappa, 'kappa_MHz': kappa/1e6,
        'length_um': length_um,
        'type': resonator_type
    }
    
    return network, properties


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_s_params(networks: list, labels: list = None, title: str = "S-Parameters"):
    """Plot S-parameters for multiple networks."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for i, net in enumerate(networks):
        label = labels[i] if labels else f"Network {i+1}"
        freq_ghz = net.frequency.f_scaled
        
        axes[0,0].plot(freq_ghz, net.s_db[:,1,0], label=label, linewidth=2)
        axes[0,1].plot(freq_ghz, np.unwrap(net.s_deg[:,1,0]), label=label, linewidth=2)
        axes[1,0].plot(freq_ghz, net.s_db[:,0,0], label=label, linewidth=2)
        axes[1,1].plot(freq_ghz, net.s_db[:,1,1], label=label, linewidth=2)
    
    titles = ['|S21| (dB)', '∠S21 (deg)', '|S11| (dB)', '|S22| (dB)']
    for ax, t in zip(axes.flat, titles):
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel(t)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_resonator(network: rf.Network, properties: Dict):
    """Plot resonator response with annotations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    freq_ghz = network.frequency.f_scaled
    f0_ghz = properties['f0_GHz']
    
    # Magnitude
    ax1.plot(freq_ghz, network.s_db[:,1,0], 'b-', linewidth=2)
    ax1.axvline(f0_ghz, color='r', linestyle='--', alpha=0.5, 
                label=f"f₀={f0_ghz:.3f} GHz")
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('|S₂₁| (dB)')
    ax1.set_title(f"Q={properties['Q_total']:.0f}, κ={properties['kappa_MHz']:.2f} MHz")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Phase
    ax2.plot(freq_ghz, network.s_deg[:,1,0], 'r-', linewidth=2)
    ax2.axvline(f0_ghz, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('∠S₂₁ (degrees)')
    ax2.set_title('Phase Response')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"{properties['type']} Resonator: L={properties['length_um']:.0f} μm")
    plt.tight_layout()
    return fig


# ============================================================================
# Example Usage with Partial Application
# ============================================================================

def create_models(**default_kwargs):
    """Create model functions with default parameters using partial."""
    return {
        'tline': partial(simulate_tline, **default_kwargs),
        'resonator': partial(simulate_resonator, **default_kwargs),
        'tline_simple': partial(tline_network, **default_kwargs),
        'res_freq': partial(resonator_frequency, **default_kwargs)
    }


# ============================================================================
# Test Driver
# ============================================================================

if __name__ == "__main__":
    print("Analytical CPW Models - Functional Implementation")
    print("="*60)
    
    # Create models with default Si substrate
    models = create_models(eps_r=11.45, tand=1e-6)
    
    # Test 1: Simple transmission line
    print("\n1. Transmission Line (480 μm)")
    freq = rf.Frequency(6, 8, 201, 'GHz')
    tline = gf.components.straight(length=480)
    
    net_tline = models['tline'](tline, {'width': 10, 'gap': 6}, freq)
    
    # Check phase at 6 GHz (should be ~8.6°)
    idx_6ghz = np.argmin(np.abs(freq.f - 6e9))
    phase_6ghz = np.angle(net_tline.s[idx_6ghz, 1, 0]) * 180 / np.pi
    
    # Test 2: Hierarchical transmission line
    print("\n2. Hierarchical TLine (4×120 μm)")
    
    @gf.cell
    def hierarchical_tline():
        c = gf.Component()
        sections = [gf.components.straight(length=120) for _ in range(4)]
        prev = None
        for s in sections:
            ref = c.add_ref(s)
            if prev:
                ref.connect("o1", prev.ports["o2"])
            else:
                c.add_port("o1", port=ref.ports["o1"])
            prev = ref
        c.add_port("o2", port=prev.ports["o2"])
        return c
    
    hier_tline = hierarchical_tline()
    net_hier = models['tline'](hier_tline, {'width': 10, 'gap': 6}, freq)
    
    phase_hier = np.angle(net_hier.s[idx_6ghz, 1, 0]) * 180 / np.pi
    print(f"   Phase @ 6 GHz: {phase_hier:.1f}° (should match simple)")
    print(f"   Difference: {abs(phase_hier - phase_6ghz):.3f}°")
    
    # Test 3: λ/4 Resonator
    print("\n3. Quarter-Wave Resonator")
    target_f0 = 6e9
    eps_eff = 1 + (11.45 - 1) * 0.5
    req_length = const.c / (4 * target_f0 * np.sqrt(eps_eff)) * 1e6  # μm
    
    resonator = gf.components.straight(length=req_length)
    freq_res = rf.Frequency(5, 7, 401, 'GHz')
    
    net_res, props = models['resonator'](
        resonator, {'width': 10, 'gap': 6}, freq_res,
        Qc=1e4, Qi=1e6
    )
    
    print(f"   Target f₀: {target_f0/1e9:.3f} GHz")
    print(f"   Actual f₀: {props['f0_GHz']:.3f} GHz")
    print(f"   Q total: {props['Q_total']:.0f}")
    print(f"   Linewidth: {props['kappa_MHz']:.2f} MHz")
    
    # Test 4: Resonator array
    print("\n4. Resonator Array (4-8 GHz)")
    freqs = np.linspace(4e9, 8e9, 5)
    freq_array = rf.Frequency(3, 9, 601, 'GHz')
    
    combined_s21 = np.ones(len(freq_array.f), dtype=complex)
    for f0_target in freqs:
        length = const.c / (4 * f0_target * np.sqrt(eps_eff)) * 1e6
        res = gf.components.straight(length=length)
        net, _ = models['resonator'](res, {'width': 10, 'gap': 6}, freq_array)
        combined_s21 *= net.s[:, 1, 0]
        print(f"   Resonator @ {f0_target/1e9:.1f} GHz: L={length:.0f} μm")
    
    # Create plots
    print("\n5. Generating Plots...")
    
    # Transmission line comparison
    fig1 = plot_s_params(
        [net_tline, net_hier],
        ['Simple (480 μm)', 'Hierarchical (4×120 μm)'],
        'Transmission Line Comparison'
    )
    plt.savefig('tline_comparison.png', dpi=150, bbox_inches='tight')
    
    # Resonator response
    fig2 = plot_resonator(net_res, props)
    plt.savefig('resonator_response.png', dpi=150, bbox_inches='tight')
    
    # Resonator array
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_array.f_scaled, 20*np.log10(np.abs(combined_s21)), 
            'b-', linewidth=2)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|S₂₁| (dB)')
    ax.set_title('5-Resonator Array (4-8 GHz)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-30, 2])
    plt.tight_layout()
    plt.savefig('resonator_array.png', dpi=150, bbox_inches='tight')
    
    print("\n✓ All tests complete!")
    print("✓ Plots saved: tline_comparison.png, resonator_response.png, resonator_array.png")
    print("="*60)   