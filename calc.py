import numpy as np
from scipy.special import ellipk # Import the elliptic integral function
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def extract_q_factors(frequency, s21_complex, f_resonance=None):
    """
    Extract Q_int and Q_ext from S21 measurement data.
    
    Method 1: Circle fit method (most accurate)
    Method 2: 3dB bandwidth method (simpler)
    """
    
    # Convert to magnitude and phase
    s21_mag = np.abs(s21_complex)
    s21_db = 20 * np.log10(s21_mag)
    
    # Find resonance if not provided
    if f_resonance is None:
        min_idx = np.argmin(s21_db)
        f_resonance = frequency[min_idx]
    else:
        min_idx = np.argmin(np.abs(frequency - f_resonance))
    
    # Method 1: Lorentzian fit
    def lorentzian(f, f0, Q_total, beta, offset):
        """S21 for a notch resonator"""
        denominator = 1 + 2j * Q_total * (f - f0) / f0
        s21 = offset * (1 + 2j * Q_total * (f - f0) / f0 - beta) / denominator
        return np.abs(s21)
    
    # Fit region around resonance (±10 linewidths)
    estimated_Q = 10000  # Initial guess
    fit_width = 10 * f_resonance / estimated_Q
    fit_mask = np.abs(frequency - f_resonance) < fit_width
    
    try:
        # Fit the magnitude response
        popt, pcov = curve_fit(
            lorentzian, 
            frequency[fit_mask], 
            s21_mag[fit_mask],
            p0=[f_resonance, estimated_Q, 0.99, 1.0],
            bounds=([f_resonance*0.99, 100, 0, 0.9], 
                    [f_resonance*1.01, 1e7, 1, 1.1])
        )
        
        f0_fit, Q_total_fit, beta_fit, offset = popt
        
        # Extract Q_ext and Q_int from beta
        Q_ext = Q_total_fit / beta_fit
        Q_int = 1 / (1/Q_total_fit - 1/Q_ext)
        
        print(f"Lorentzian fit results:")
        print(f"  f0 = {f0_fit/1e9:.4f} GHz")
        print(f"  Q_total = {Q_total_fit:.0f}")
        print(f"  Q_ext = {Q_ext:.0f}")
        print(f"  Q_int = {Q_int:.0f}")
        print(f"  Coupling β = {beta_fit:.3f}")
        
    except:
        print("Lorentzian fit failed, using 3dB method")
        Q_total_fit = None
    
    # Method 2: Simple 3dB bandwidth
    min_val = s21_db[min_idx]
    threshold_3db = min_val + 3
    
    # Find 3dB points
    left_idx = min_idx
    right_idx = min_idx
    
    while left_idx > 0 and s21_db[left_idx] < threshold_3db:
        left_idx -= 1
    while right_idx < len(s21_db)-1 and s21_db[right_idx] < threshold_3db:
        right_idx += 1
    
    if left_idx != min_idx and right_idx != min_idx:
        f_low = frequency[left_idx]
        f_high = frequency[right_idx]
        bandwidth_3db = f_high - f_low
        Q_loaded_3db = f_resonance / bandwidth_3db
        
        print(f"\n3dB bandwidth method:")
        print(f"  f0 = {f_resonance/1e9:.4f} GHz")
        print(f"  3dB bandwidth = {bandwidth_3db/1e6:.2f} MHz")
        print(f"  Q_loaded = {Q_loaded_3db:.0f}")
        
        # Estimate Q_ext from insertion loss
        insertion_loss_db = -min_val
        if insertion_loss_db > 20:  # Nearly critical coupling
            Q_ext_3db = Q_loaded_3db
            Q_int_3db = Q_loaded_3db
        else:
            # From S21_min = -20*log10(Q_loaded/Q_ext)
            Q_ext_3db = Q_loaded_3db / (10**(-insertion_loss_db/20))
            Q_int_3db = 1 / (1/Q_loaded_3db - 1/Q_ext_3db)
        
        print(f"  Q_ext ≈ {Q_ext_3db:.0f}")
        print(f"  Q_int ≈ {Q_int_3db:.0f}")
    
    return f0_fit if Q_total_fit else f_resonance, Q_total_fit, Q_ext, Q_int


def calculate_phase_velocity(center_width, gap_width, epsilon_r, 
                            thickness=100e-9, london_depth=16e-9, 
                            temperature=0.02, critical_temp=1.2):
    """
    Calculate phase velocity in CPW including kinetic inductance.
    
    Three methods:
    1. From measured resonance frequency
    2. From geometry and material parameters
    3. From S21 phase slope
    """
    
    # Constants
    c = 2.998e8
    mu0 = 4 * np.pi * 1e-7
    epsilon0 = 8.854e-12
    
    # Method 1: From resonance measurement
    def vp_from_resonance(f_resonance, length, mode='quarter'):
        """Extract v_p from measured resonance"""
        if mode == 'quarter':
            vp = 4 * length * f_resonance
        elif mode == 'half':
            vp = 2 * length * f_resonance
        return vp
    
    # Method 2: Calculate from geometry
    # CPW geometry factor
    k = center_width / (center_width + 2 * gap_width)
    k_prime = np.sqrt(1 - k**2)
    
    # Elliptic integrals
    if k <= 0.707:
        K_k = np.pi / np.log(2 * (1 + np.sqrt(k_prime)) / (1 - np.sqrt(k_prime)))
        K_k_prime = np.log(2 * (1 + np.sqrt(k)) / (1 - np.sqrt(k))) / np.pi
    else:
        K_k = np.log(2 * (1 + np.sqrt(k_prime)) / (1 - np.sqrt(k_prime))) / np.pi
        K_k_prime = np.pi / np.log(2 * (1 + np.sqrt(k)) / (1 - np.sqrt(k)))
    
    # Effective permittivity (quasi-static approximation)
    epsilon_eff = 1 + 0.5 * (epsilon_r - 1) * (K_k_prime / K_k) / (K_k_prime / K_k + 1)
    
    # Without kinetic inductance
    vp_no_kinetic = c / np.sqrt(epsilon_eff)
    
    # With kinetic inductance
    lambda_t = london_depth / np.sqrt(1 - (temperature / critical_temp)**4)
    
    # Geometric inductance
    L_g = (mu0 / 4) * (K_k_prime / K_k)
    
    # Kinetic inductance
    if thickness < lambda_t:
        L_k = (mu0 * lambda_t / thickness) * (K_k_prime / K_k) / (4 * K_k * K_k_prime)
    else:
        L_k = (mu0 / 2) * (K_k_prime / K_k) / (4 * K_k * K_k_prime)
    
    # Capacitance per unit length
    C = 4 * epsilon0 * epsilon_eff * (K_k / K_k_prime)
    
    # Phase velocity with kinetic inductance
    vp_with_kinetic = 1 / np.sqrt((L_g + L_k) * C)
    
    # Kinetic inductance fraction
    alpha = L_k / (L_g + L_k)
    
    print(f"Phase velocity calculation:")
    print(f"  Geometry: w={center_width*1e6:.2f}μm, g={gap_width*1e6:.2f}μm")
    print(f"  k = {k:.3f}")
    print(f"  ε_eff = {epsilon_eff:.2f}")
    print(f"  v_p (no kinetic) = {vp_no_kinetic/c:.3f} * c = {vp_no_kinetic/1e8:.2f} × 10^8 m/s")
    print(f"  Kinetic inductance fraction α = {alpha:.3f}")
    print(f"  v_p (with kinetic) = {vp_with_kinetic/c:.3f} * c = {vp_with_kinetic/1e8:.2f} × 10^8 m/s")
    
    return vp_with_kinetic, epsilon_eff, alpha


def vp_from_phase_slope(frequency, s21_phase, length):
    """
    Extract phase velocity from S21 phase vs frequency.
    Away from resonance: phase = -2π * f * length / v_p
    """
    # Unwrap phase
    phase_unwrapped = np.unwrap(s21_phase)
    
    # Linear fit to phase vs frequency
    coeffs = np.polyfit(frequency, phase_unwrapped, 1)
    phase_slope = coeffs[0]  # radians/Hz
    
    # v_p = -2π * length / phase_slope
    vp = -2 * np.pi * length / phase_slope
    
    print(f"\nPhase velocity from S21 phase slope:")
    print(f"  Phase slope = {phase_slope:.3e} rad/Hz")
    print(f"  v_p = {vp/c:.3f} * c = {vp/1e8:.2f} × 10^8 m/s")
    
    return vp


# Example usage
if __name__ == "__main__":
    # Your resonator parameters
    length = 4.2e-3
    center_width = 3.42e-6
    gap_width = 2.43e-6
    
    # Calculate phase velocity
    vp, eps_eff, kinetic_fraction = calculate_phase_velocity(
        center_width, gap_width, 
        epsilon_r=11.45,  # Silicon
        thickness=100e-9,  # 100nm Al
        london_depth=16e-9,  # Al
        temperature=0.02,  # 20mK
        critical_temp=1.2  # Al
    )
    
    # Expected resonance
    f0_expected = vp / (4 * length)
    print(f"\nExpected λ/4 resonance: {f0_expected/1e9:.3f} GHz")
    
    # Simulate measurement data
    freq = np.linspace(6.5e9, 7.5e9, 1001)
    Q_int_actual = 5e5
    Q_ext_actual = 1e4
    Q_total = 1/(1/Q_int_actual + 1/Q_ext_actual)
    beta = Q_total/Q_ext_actual
    
    # Generate synthetic S21 data
    s21 = np.zeros(len(freq), dtype=complex)
    for i, f in enumerate(freq):
        denom = 1 + 2j * Q_total * (f - f0_expected) / f0_expected
        s21[i] = (1 + 2j * Q_total * (f - f0_expected) / f0_expected - beta) / denom
    
    # Add small noise
    s21 += (np.random.randn(len(freq)) + 1j*np.random.randn(len(freq))) * 0.001
    
    # Extract Q factors
    print("\n" + "="*50)
    print("Extracting Q factors from simulated data:")
    f0_fit, Q_tot, Q_e, Q_i = extract_q_factors(freq, s21)
    
    # Verify phase velocity from resonance
    vp_measured = 4 * length * f0_fit
    print(f"\nPhase velocity from resonance:")
    print(f"  v_p = {vp_measured/c:.3f} * c (should match calculated value)")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(freq/1e9, 20*np.log10(np.abs(s21)))
    axes[0].set_xlabel('Frequency (GHz)')
    axes[0].set_ylabel('|S21| (dB)')
    axes[0].set_title('Resonance for Q extraction')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=-3, color='r', linestyle='--', alpha=0.5, label='3dB line')
    axes[0].legend()
    
    axes[1].plot(freq/1e9, np.angle(s21, deg=True))
    axes[1].set_xlabel('Frequency (GHz)')
    axes[1].set_ylabel('Phase (deg)')
    axes[1].set_title('Phase response')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_eps_eff_cpw(eps_r, w, s):
    """
    Calculates the effective relative permittivity (eps_eff) for a
    coplanar waveguide (CPW) on an infinitely thick substrate.

    This uses a standard analytical approximation formula.

    Args:
        eps_r (float): Relative permittivity of the substrate material
                       (e.g., ~11.7 for silicon).
        w (float): Width of the center conductor trace (in any consistent unit).
        s (float): Width of the gap from the center trace to the ground planes
                   (in the same unit as w).

    Returns:
        float: The calculated effective relative permittivity.
    """
    # 1. Calculate the geometric factor 'k'
    k = w / (w + 2 * s)

    # 2. Calculate the complementary modulus k'
    # k_prime must be handled carefully to avoid domain errors for k=1
    if k == 1.0:
        k_prime = 0.0
    else:
        k_prime = np.sqrt(1 - k**2)

    # 3. Calculate the ratio of complete elliptic integrals of the first kind
    # K(k) is ellipk(k**2) in SciPy's convention (it takes m = k^2 as input)
    if k == 0.0:
        # Avoid division by zero when w=0, K(0) = pi/2
        ratio = 0 
    else:
        K_k = ellipk(k**2)
        K_k_prime = ellipk(k_prime**2)
        ratio = K_k_prime / K_k

    # 4. Calculate the final effective permittivity
    eps_eff = 1 + (eps_r - 1) / 2 * ratio
    
    return eps_eff

# --- Example Calculation ---
# Let's use typical values for a CPW on a silicon chip
substrate_eps_r = 11.7  # Relative permittivity of silicon
trace_width_w = 10e-6   # 10 micrometers
gap_width_s = 6e-6      # 6 micrometers

# Calculate the effective permittivity
effective_permittivity = calculate_eps_eff_cpw(substrate_eps_r, trace_width_w, gap_width_s)

print(f"Substrate ε_r: {substrate_eps_r}")
print(f"Trace Width (w): {trace_width_w*1e6:.1f} um")
print(f"Gap Width (s): {gap_width_s*1e6:.1f} um")
print("-" * 30)
print(f"Calculated ε_eff: {effective_permittivity:.4f}")

# Now, let's calculate the phase velocity (vp) using this value
c = 3e8 # Speed of light in vacuum (m/s)
vp = c / np.sqrt(effective_permittivity)
print(f"Resulting Phase Velocity (vp): {vp:.2e} m/s")
