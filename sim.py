import numpy as np
import skrf as rf

def calculate_cpw_resonator_scattering_matrix(
    length: float,
    frequency: float,
    center_width: float = 3.42e-6,
    gap_width: float = 2.43e-6,
    thickness: float = 100e-9,
    epsilon_r: float = 11.45,
    london_depth: float = 16e-9,
    temperature: float = 0.02,
    critical_temp: float = 1.2,
    z0_port: float = 50.0,
    resonator_type: str = 'quarter_wave',
    Q_int: float = 1e6,
    Q_ext: float = 1e4
):
    # Physical constants
    c = 2.998e8
    mu0 = 4 * np.pi * 1e-7
    epsilon0 = 8.854e-12
    
    # Convert frequency to array
    freq_array = np.atleast_1d(frequency)
    num_freqs = len(freq_array)
    
    # Initialize S-parameter array
    s_matrix = np.zeros((num_freqs, 2, 2), dtype=complex)
    
    # CPW geometry factors
    k = center_width / (center_width + 2 * gap_width)
    k_prime = np.sqrt(1 - k**2)
    
    # Elliptic integrals
    if k <= 0.707:
        K_k = np.pi / np.log(2 * (1 + np.sqrt(k_prime)) / (1 - np.sqrt(k_prime)))
        K_k_prime = np.log(2 * (1 + np.sqrt(k)) / (1 - np.sqrt(k))) / np.pi
    else:
        K_k = np.log(2 * (1 + np.sqrt(k_prime)) / (1 - np.sqrt(k_prime))) / np.pi
        K_k_prime = np.pi / np.log(2 * (1 + np.sqrt(k)) / (1 - np.sqrt(k)))
    
    # Effective permittivity
    epsilon_eff = 1 + 0.5 * (epsilon_r - 1) * (K_k_prime / K_k) / (K_k_prime / K_k + 1)
    
    # Temperature-dependent London depth
    lambda_t = london_depth / np.sqrt(1 - (temperature / critical_temp)**4)
    
    # Inductance per unit length
    L_geometric = (mu0 / 4) * (K_k_prime / K_k)
    if thickness < lambda_t:
        L_kinetic = (mu0 * lambda_t / thickness) * (K_k_prime / K_k) / (4 * K_k * K_k_prime)
    else:
        L_kinetic = (mu0 / 2) * (K_k_prime / K_k) / (4 * K_k * K_k_prime)
    L_total = L_geometric + L_kinetic
    
    # Capacitance per unit length
    C = 4 * epsilon0 * epsilon_eff * (K_k / K_k_prime)
    
    # Phase velocity
    vp = 1 / np.sqrt(L_total * C)
    
    # Calculate resonant frequencies
    if resonator_type == 'quarter_wave':
        f0 = vp / (4 * length)
        harmonics = [(2*n - 1) * f0 for n in range(1, 6)]  # f0, 3f0, 5f0...
    else:
        f0 = vp / (2 * length)
        harmonics = [n * f0 for n in range(1, 6)]  # f0, 2f0, 3f0...
    
    # Calculate S-parameters for each frequency
    for i, freq in enumerate(freq_array):
        # Start with unity transmission
        S21 = 1.0 + 0j
        S11 = 0.0 + 0j
        
        # Add resonance dips for each harmonic
        for f_res in harmonics:
            # Total Q and coupling parameter
            Q_total = 1 / (1/Q_int + 1/Q_ext)
            
            # Lorentzian resonance response for notch filter
            denominator = 1 + 2j * Q_total * (freq - f_res) / f_res
            
            # Coupling factor (beta = Q_total/Q_ext)
            beta = Q_total / Q_ext
            
            # Notch filter transmission (dip at resonance)
            # For critically coupled (beta = 1): complete dip
            # For undercoupled (beta < 1): partial dip
            S21_resonance = (1 + 2j * Q_total * (freq - f_res) / f_res - beta) / denominator
            
            # Multiply by this resonance response
            S21 = S21 * S21_resonance
            
            # Small reflection at resonance
            S11 = S11 - beta / denominator * 0.05
        
        # Ensure reciprocity
        S12 = S21
        S22 = S11
        
        # Store in S-matrix
        s_matrix[i] = np.array([[S11, S12],
                                [S21, S22]])
    
    # Create frequency object for skrf
    freq_ghz = rf.Frequency.from_f(freq_array / 1e9, unit='ghz')
    
    # Create Network object
    network = rf.Network(frequency=freq_ghz, s=s_matrix, z0=z0_port)
    network.f0 = f0 / 1e9
    network.harmonics = [f/1e9 for f in harmonics]
    
    return network


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Your resonator dimensions
    length = 4.2e-3
    center_width = 3.42e-6
    gap_width = 2.43e-6
    
    # Wide frequency sweep
    frequencies = np.linspace(1e9, 25e9, 2401)
    
    network = calculate_cpw_resonator_scattering_matrix(
        length=length,
        frequency=frequencies,
        center_width=center_width,
        gap_width=gap_width,
        resonator_type='quarter_wave',
        Q_int=1e6,
        Q_ext=1e4
    )
    
    print(f"Fundamental resonance: {network.f0:.3f} GHz")
    print(f"First 3 harmonics: {[f'{f:.2f}' for f in network.harmonics[:3]]} GHz")
    
    # Find actual resonance dips
    s21_db = 20 * np.log10(np.abs(network.s[:, 1, 0]))
    minima_indices = []
    for i in range(1, len(s21_db)-1):
        if s21_db[i] < s21_db[i-1] and s21_db[i] < s21_db[i+1] and s21_db[i] < -3:
            minima_indices.append(i)
    
    print(f"\nResonance dips found at:")
    for idx in minima_indices[:3]:
        f_res = network.frequency.f_scaled[idx]
        depth = s21_db[idx]
        print(f"  {f_res:.3f} GHz ({depth:.1f} dB)")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # S21 magnitude
    axes[0,0].plot(network.frequency.f_scaled, s21_db)
    axes[0,0].set_xlabel('Frequency (GHz)')
    axes[0,0].set_ylabel('|S21| (dB)')
    axes[0,0].set_title('Transmission - Multiple Resonances')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(-30, 5)
    
    # S21 phase
    axes[0,1].plot(network.frequency.f_scaled, np.unwrap(np.angle(network.s[:, 1, 0])) * 180/np.pi)
    axes[0,1].set_xlabel('Frequency (GHz)')
    axes[0,1].set_ylabel('Phase S21 (deg)')
    axes[0,1].set_title('Phase Response')
    axes[0,1].grid(True, alpha=0.3)
    
    # Zoom on fundamental
    f0_idx = np.argmin(np.abs(network.frequency.f_scaled - network.f0))
    zoom_range = 200
    zoom_indices = slice(max(0, f0_idx-zoom_range), min(len(frequencies), f0_idx+zoom_range))
    
    axes[1,0].plot(network.frequency.f_scaled[zoom_indices], s21_db[zoom_indices])
    axes[1,0].set_xlabel('Frequency (GHz)')
    axes[1,0].set_ylabel('|S21| (dB)')
    axes[1,0].set_title(f'Zoom on Fundamental (~{network.f0:.2f} GHz)')
    axes[1,0].grid(True, alpha=0.3)
    
    # S11 magnitude
    s11_db = 20 * np.log10(np.abs(network.s[:, 0, 0]))
    axes[1,1].plot(network.frequency.f_scaled, s11_db)
    axes[1,1].set_xlabel('Frequency (GHz)')
    axes[1,1].set_ylabel('|S11| (dB)')
    axes[1,1].set_title('Reflection')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
