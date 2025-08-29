
# main.py
# Description: A complete script to simulate microwave circuits using ABCD matrices,
# based on the functions transcribed from the YouTube video.
# This script defines the necessary functions, runs a simulation for a sample
# circuit, and plots the results.

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. ABCD Matrix Definitions for Basic Circuit Components
# =============================================================================

def series_inductance(freq_array, L):
    """
    Calculates the ABCD matrix for a series inductor.
    Args:
        freq_array (np.ndarray): Array of frequencies in Hz.
        L (float): Inductance in Henrys.
    Returns:
        np.ndarray: A 3D array of shape (len(freq_array), 2, 2) representing
                    the ABCD matrix at each frequency.
    """
    omega = 2 * np.pi * freq_array
    # Create an identity matrix for each frequency point
    abcd = np.zeros((len(freq_array), 2, 2), dtype=complex)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    # Set the B element (impedance)
    abcd[:, 0, 1] = 1j * omega * L
    return abcd

def series_capacitance(freq_array, C):
    """
    Calculates the ABCD matrix for a series capacitor.
    Args:
        freq_array (np.ndarray): Array of frequencies in Hz.
        C (float): Capacitance in Farads.
    Returns:
        np.ndarray: A 3D array of shape (len(freq_array), 2, 2) representing
                    the ABCD matrix at each frequency.
    """
    omega = 2 * np.pi * freq_array
    abcd = np.zeros((len(freq_array), 2, 2), dtype=complex)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    # Set the B element (impedance)
    abcd[:, 0, 1] = 1 / (1j * omega * C)
    return abcd

def parallel_inductance(freq_array, L):
    """
    Calculates the ABCD matrix for a parallel (shunt) inductor.
    Args:
        freq_array (np.ndarray): Array of frequencies in Hz.
        L (float): Inductance in Henrys.
    Returns:
        np.ndarray: A 3D array of shape (len(freq_array), 2, 2) representing
                    the ABCD matrix at each frequency.
    """
    omega = 2 * np.pi * freq_array
    abcd = np.zeros((len(freq_array), 2, 2), dtype=complex)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    # Set the C element (admittance)
    abcd[:, 1, 0] = 1 / (1j * omega * L)
    return abcd

def parallel_capacitance(freq_array, C):
    """
    Calculates the ABCD matrix for a parallel (shunt) capacitor.
    Args:
        freq_array (np.ndarray): Array of frequencies in Hz.
        C (float): Capacitance in Farads.
    Returns:
        np.ndarray: A 3D array of shape (len(freq_array), 2, 2) representing
                    the ABCD matrix at each frequency.
    """
    omega = 2 * np.pi * freq_array
    abcd = np.zeros((len(freq_array), 2, 2), dtype=complex)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    # Set the C element (admittance)
    abcd[:, 1, 0] = 1j * omega * C
    return abcd

def transmission_line(freq_array, length, z0, vp):
    """
    Calculates the ABCD matrix for a lossless transmission line.
    Args:
        freq_array (np.ndarray): Array of frequencies in Hz.
        length (float): Physical length of the line in meters.
        z0 (float): Characteristic impedance in Ohms.
        vp (float): Phase velocity in m/s.
    Returns:
        np.ndarray: A 3D array of shape (len(freq_array), 2, 2) representing
                    the ABCD matrix at each frequency.
    """
    omega = 2 * np.pi * freq_array
    beta = omega / vp
    cos_bl = np.cos(beta * length)
    sin_bl = np.sin(beta * length)

    abcd = np.zeros((len(freq_array), 2, 2), dtype=complex)
    abcd[:, 0, 0] = cos_bl
    abcd[:, 0, 1] = 1j * z0 * sin_bl
    abcd[:, 1, 0] = 1j * (1 / z0) * sin_bl
    abcd[:, 1, 1] = cos_bl
    return abcd

# =============================================================================
# 2. Conversion and Transformation Functions
# =============================================================================

def abcd_to_s21(abcd_array, z0):
    """
    Converts an ABCD matrix array to an S21 transmission parameter array.
    Args:
        abcd_array (np.ndarray): 3D array of ABCD matrices.
        z0 (float): Port impedance in Ohms.
    Returns:
        np.ndarray: 1D array of complex S21 values.
    """
    A = abcd_array[:, 0, 0]
    B = abcd_array[:, 0, 1]
    C = abcd_array[:, 1, 0]
    D = abcd_array[:, 1, 1]
    return 2 / (A + B / z0 + C * z0 + D)

def multiply_abcd_matrices(abcd_list):
    """
    Multiplies a list of ABCD matrix arrays together in sequence.
    This is done element-wise for each frequency.
    Args:
        abcd_list (list): A list of 3D ABCD matrix arrays.
    Returns:
        np.ndarray: The resulting total 3D ABCD matrix array.
    """
    # Start with an identity matrix for each frequency point
    total_abcd = np.zeros_like(abcd_list[0])
    total_abcd[:, 0, 0] = 1
    total_abcd[:, 1, 1] = 1

    # Sequentially multiply the matrices
    for abcd in abcd_list:
        total_abcd = np.einsum('ijk,ikl->ijl', total_abcd, abcd)
    return total_abcd

def hanger_transform(abcd_array):
    """
    Performs a hanger transform on an ABCD matrix array. This models shunting
    the 2-port network to ground at one end.
    Args:
        abcd_array (np.ndarray): 3D array of ABCD matrices to transform.
    Returns:
        np.ndarray: The transformed 3D ABCD matrix array.
    """
    D = abcd_array[:, 1, 1]
    B = abcd_array[:, 0, 1]
    
    hanger_abcd = np.zeros_like(abcd_array)
    hanger_abcd[:, 0, 0] = 1
    hanger_abcd[:, 1, 1] = 1
    # The new admittance is D/B
    hanger_abcd[:, 1, 0] = D / B
    return hanger_abcd

# =============================================================================
# 3. Fitting Function
# =============================================================================

def s21_lorentzian(freq_array, f0, kappa, phi):
    """
    Generates an ideal S21 Lorentzian lineshape.
    Args:
        freq_array (np.ndarray): Array of frequencies in Hz.
        f0 (float): Resonance frequency in Hz.
        kappa (float): Total linewidth in Hz.
        phi (float): Asymmetry phase factor in radians.
    Returns:
        np.ndarray: 1D array of complex S21 values for the Lorentzian.
    """
    return 1 - (kappa / 2) / (1j * (freq_array - f0) + kappa / 2) * np.exp(1j * phi)

# =============================================================================
# 4. Main Simulation and Plotting
# =============================================================================

def main():
    """
    Main function to run the circuit simulation and plot the results.
    """
    # --- Simulation Parameters ---
    z0 = 50.0  # System impedance (Ohms)
    vp = 1.2e8   # Phase velocity (m/s), typical for CPW on silicon
    
    # --- Frequency Range ---
    freq_array = np.linspace(1e9, 10e9, 1001) # 1 to 10 GHz

    # --- Circuit Component Values ---
    C_series = 30e-15  # 30 fF series capacitor
    L_parallel = 1e-9    # 1 nH parallel inductor
    tline_length = 4e-3  # 4 mm transmission line

    # --- Step 1: Create ABCD matrices for each component ---
    abcd_cap = series_capacitance(freq_array, C_series)
    abcd_tline = transmission_line(freq_array, tline_length, z0, vp)
    abcd_ind = parallel_inductance(freq_array, L_parallel)

    # --- Step 2: Cascade the components by multiplying their matrices ---
    # The order in the list matters: from input port to output port.
    circuit_elements = [abcd_cap, abcd_tline, abcd_ind]
    total_abcd_circuit = multiply_abcd_matrices(circuit_elements)
    
    # --- Step 3: Convert the total ABCD matrix to S21 transmission ---
    s21_circuit = abcd_to_s21(total_abcd_circuit, z0)
    
    # --- Step 4: Apply the hanger transform to the original circuit ---
    # This creates a new circuit configuration.
    total_abcd_hanger = hanger_transform(total_abcd_circuit)
    s21_hanger = abcd_to_s21(total_abcd_hanger, z0)

    # --- Step 5: Generate an ideal Lorentzian for comparison ---
    f0_lorentz = 7.1e9 # Resonance frequency for the plot
    kappa_lorentz = 20e6 # Linewidth for the plot
    s21_ideal = s21_lorentzian(freq_array, f0_lorentz, kappa_lorentz, phi=0)

    # --- Step 6: Plot the results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('ABCD Matrix Circuit Simulation', fontsize=16)

    # Plot magnitude in dB
    ax1.plot(freq_array / 1e9, 20 * np.log10(np.abs(s21_circuit)), label='Original Circuit (Through)', color='royalblue')
    ax1.plot(freq_array / 1e9, 20 * np.log10(np.abs(s21_hanger)), label='Hanger Transformed Circuit', color='darkorange', linestyle='--')
    ax1.plot(freq_array / 1e9, 20 * np.log10(np.abs(s21_ideal)), label='Ideal Lorentzian (f0=7.1 GHz)', color='green', linestyle=':')
    ax1.set_ylabel('S21 Magnitude (dB)')
    ax1.legend()
    ax1.grid(True)

    # Plot phase in degrees
    ax2.plot(freq_array / 1e9, np.angle(s21_circuit, deg=True), label='Original Circuit (Through)', color='royalblue')
    ax2.plot(freq_array / 1e9, np.angle(s21_hanger, deg=True), label='Hanger Transformed Circuit', color='darkorange', linestyle='--')
    ax2.plot(freq_array / 1e9, np.angle(s21_ideal, deg=True), label='Ideal Lorentzian (f0=7.1 GHz)', color='green', linestyle=':')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('S21 Phase (degrees)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
