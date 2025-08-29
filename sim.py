    # network.write_touchstone('cpw_line.s2p')
import numpy as np
import skrf as rf

def calculate_cpw_scattering_matrix(
    length: float,
    frequency: float,
    center_width: float = 10e-6,
    gap_width: float = 6e-6,
    thickness: float = 200e-9,
    epsilon_r: float = 11.7,
    london_depth: float = 150e-9,
    temperature: float = 1.5,
    critical_temp: float = 9.2,
    z0_port: float = 50.0,
    loss_tangent: float = 1e-6,
    Q_factor: float = 1e6
):
    """
    Calculate the scattering matrix for a superconducting CPW transmission line.
    
    Parameters:
    -----------
    length : float
        Length of the transmission line (m)
    frequency : float or array-like
        Operating frequency (Hz) - can be scalar or array
    center_width : float
        Width of center conductor (m)
    gap_width : float
        Gap width between center and ground (m)
    thickness : float
        Superconductor film thickness (m)
    epsilon_r : float
        Relative permittivity of substrate
    london_depth : float
        London penetration depth at T=0 (m)
    temperature : float
        Operating temperature (K)
    critical_temp : float
        Critical temperature of superconductor (K)
    z0_port : float
        Port impedance (Ohms)
    loss_tangent : float
        Dielectric loss tangent
    Q_factor : float
        Quality factor for conductor losses
    
    Returns:
    --------
    skrf.Network
        Network object containing S-parameters
    """
    
    # Physical constants
    c = 2.998e8  # Speed of light (m/s)
    mu0 = 4 * np.pi * 1e-7  # Permeability of free space
    epsilon0 = 8.854e-12  # Permittivity of free space
    
    # Convert frequency to array if scalar
    freq_array = np.atleast_1d(frequency)
    num_freqs = len(freq_array)
    
    # Initialize S-parameter array
    s_matrix = np.zeros((num_freqs, 2, 2), dtype=complex)
    
    # Calculate CPW geometry factors
    k = center_width / (center_width + 2 * gap_width)
    k_prime = np.sqrt(1 - k**2)
    
    # Complete elliptic integrals approximation
    if k <= 0.707:
        K_k = np.pi / np.log(2 * (1 + np.sqrt(k_prime)) / (1 - np.sqrt(k_prime)))
        K_k_prime = np.log(2 * (1 + np.sqrt(k)) / (1 - np.sqrt(k))) / np.pi
    else:
        K_k = np.log(2 * (1 + np.sqrt(k_prime)) / (1 - np.sqrt(k_prime))) / np.pi
        K_k_prime = np.pi / np.log(2 * (1 + np.sqrt(k)) / (1 - np.sqrt(k)))
    
    # Effective permittivity for CPW on substrate
    # Using weighted average based on field distribution
    epsilon_eff = 1 + 0.5 * (epsilon_r - 1) * (K_k_prime / K_k) / (K_k_prime / K_k + 1)
    
    # Temperature-dependent London penetration depth
    if temperature < critical_temp:
        lambda_t = london_depth / np.sqrt(1 - (temperature / critical_temp)**4)
    else:
        lambda_t = london_depth * 10  # Non-superconducting state
    
    # Geometric inductance per unit length for CPW
    L_geometric = (mu0 / 4) * (K_k_prime / K_k)
    
    # Kinetic inductance per unit length
    # For thin film superconductor in CPW geometry
    if thickness < lambda_t:
        # Thin film limit - kinetic inductance is significant
        L_kinetic = (mu0 * lambda_t / thickness) * (K_k_prime / K_k) / (4 * K_k * K_k_prime)
    else:
        # Thick film limit
        L_kinetic = (mu0 / 2) * (K_k_prime / K_k) / (4 * K_k * K_k_prime)
    
    # Total inductance per unit length
    L_total = L_geometric + L_kinetic
    
    # Capacitance per unit length for CPW
    C = 4 * epsilon0 * epsilon_eff * (K_k / K_k_prime)
    
    # Characteristic impedance (lossless approximation)
    Z0 = np.sqrt(L_total / C)
    
    # Phase velocity
    vp = 1 / np.sqrt(L_total * C)
    
    # Calculate S-parameters for each frequency
    for i, freq in enumerate(freq_array):
        omega = 2 * np.pi * freq
        
        # Wavelength in the transmission line
        wavelength = vp / freq
        beta = 2 * np.pi / wavelength  # Phase constant
        
        # Attenuation constant (very small for superconductors)
        # Conductor losses through Q factor
        alpha_c = beta / (2 * Q_factor)
        
        # Dielectric losses
        alpha_d = (beta * loss_tangent) / 2
        
        # Total attenuation
        alpha = alpha_c + alpha_d
        
        # Propagation constant
        gamma = alpha + 1j * beta
        
        # Calculate transmission line parameters
        gamma_l = gamma * length
        
        # For numerical stability, check the magnitude
        if np.abs(gamma_l.real) > 20:
            # Very high loss - use approximation
            S11 = 1.0 + 0j
            S12 = 0.0 + 0j
            S21 = 0.0 + 0j
            S22 = 1.0 + 0j
        else:
            # Calculate S-parameters using impedance mismatch formulation
            # Reflection coefficient at the ports
            Gamma = (Z0 - z0_port) / (Z0 + z0_port)
            
            # Transmission coefficient
            T = np.exp(-gamma_l)
            
            # S-parameters for matched or nearly matched line
            denominator = 1 - Gamma**2 * T**2
            
            S11 = Gamma * (1 - T**2) / denominator
            S21 = T * (1 - Gamma**2) / denominator
            S12 = S21  # Reciprocal network
            S22 = S11  # Symmetric network
        
        # Store in S-matrix
        s_matrix[i] = np.array([[S11, S12],
                                [S21, S22]])
    
    # Create frequency object for skrf (convert to GHz)
    freq_ghz = rf.Frequency.from_f(freq_array / 1e9, unit='ghz')
    
    # Create and return Network object
    network = rf.Network(frequency=freq_ghz, s=s_matrix, z0=z0_port)
    
    return network


def calculate_cpw_network(
    length: float,
    freq_start: float,
    freq_stop: float,
    num_points: int = 201,
    **kwargs
):
    """
    Calculate S-parameters over a frequency range and return as skrf Network.
    
    Parameters:
    -----------
    length : float
        Length of transmission line (m)
    freq_start : float
        Start frequency (Hz)
    freq_stop : float
        Stop frequency (Hz)
    num_points : int
        Number of frequency points
    **kwargs : additional parameters passed to calculate_cpw_scattering_matrix
    
    Returns:
    --------
    skrf.Network
        Network object with S-parameters over frequency range
    """
    
    frequencies = np.linspace(freq_start, freq_stop, num_points)
    return calculate_cpw_scattering_matrix(length, frequencies, **kwargs)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example 1: Single frequency
    length = 10e-3  # 10 mm
    frequency = 5e9  # 5 GHz
    
    network_single = calculate_cpw_scattering_matrix(
        length=length, 
        frequency=frequency, 
        temperature=1.5,
        Q_factor=1e6  # High Q for superconductor
    )
    
    print("Single frequency network:")
    print(f"Frequency: {network_single.frequency.f_scaled[0]:.2f} GHz")
    print(f"S11 = {network_single.s[0,0,0]:.4f}")
    print(f"S21 = {network_single.s[0,1,0]:.4f}")
    print(f"|S11| = {20*np.log10(np.abs(network_single.s[0,0,0])):.2f} dB")
    print(f"|S21| = {20*np.log10(np.abs(network_single.s[0,1,0])):.2f} dB")
    
    # Example 2: Frequency sweep
    frequencies = np.linspace(1e9, 10e9, 101)  # 1-10 GHz
    
    network = calculate_cpw_scattering_matrix(
        length=10e-3,
        frequency=frequencies,
        temperature=1.5,
        center_width=10e-6,
        gap_width=6e-6,
        epsilon_r=11.7,
        Q_factor=1e6,
        z0_port=50.0
    )
    
    print(f"\nFrequency sweep network:")
    print(f"Frequency range: {network.frequency.f_scaled[0]:.2f} - {network.frequency.f_scaled[-1]:.2f} GHz")
    print(f"Number of points: {len(network.frequency)}")
    
    # Calculate characteristic impedance
    L_total = 4.5e-7  # Approximate value (H/m)
    C = 1.5e-10  # Approximate value (F/m)
    Z0_calc = np.sqrt(L_total / C)
    print(f"Characteristic impedance: ~{Z0_calc:.1f} Ohms")
    
    # Example 3: Using the convenience function
    network2 = calculate_cpw_network(
        length=5e-3,
        freq_start=2e9,
        freq_stop=8e9,
        num_points=51,
        temperature=1.5,
        Q_factor=5e5
    )
    
    print(f"\nConvenience function network:")
    print(f"Frequency range: {network2.frequency.f_scaled[0]:.2f} - {network2.frequency.f_scaled[-1]:.2f} GHz")
    
    # Check S-parameter magnitudes
    s21_db = 20 * np.log10(np.abs(network.s[:, 1, 0]))
    s11_db = 20 * np.log10(np.abs(network.s[:, 0, 0]))
    print(f"\nS21 range: {s21_db.min():.2f} to {s21_db.max():.2f} dB")
    print(f"S11 range: {s11_db.min():.2f} to {s11_db.max():.2f} dB")
    
    # Plotting example (commented out, uncomment to use)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    network.plot_s_db(m=0, n=0, ax=axes[0,0])  # S11 magnitude
    axes[0,0].set_ylim(-50, 0)
    network.plot_s_deg(m=0, n=0, ax=axes[0,1])  # S11 phase
    
    network.plot_s_db(m=1, n=0, ax=axes[1,0])  # S21 magnitude  
    axes[1,0].set_ylim(-3, 0)
    network.plot_s_deg(m=1, n=0, ax=axes[1,1])  # S21 phase
    
    plt.tight_layout()
    plt.show()
    
    # Export to touchstone file
