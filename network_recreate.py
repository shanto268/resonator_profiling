import numpy as np
import skrf as rf
from skrf.vectorFitting import VectorFitting
import matplotlib.pyplot as plt

# Load your HFSS output (Touchstone file)
# e.g., HFSS -> Right-click Solution -> Export -> Touchstone
net = rf.Network("device.s2p")

# Inspect frequencies and S-parameters
print(f"Loaded {net.nports}-port network with {len(net.f)} frequency points")
print("Frequency range: %.2f to %.2f GHz" % (net.f[0]/1e9, net.f[-1]/1e9))

# Choose number of poles for fitting
n_poles_real = 6
n_poles_cmplx = 12

# Run vector fitting
vf = VectorFitting(net.s, net.f,
                   n_poles_real=n_poles_real,
                   n_poles_cmplx=n_poles_cmplx)

print("Running vector fit...")
vf.vector_fit()

# Evaluate fitted model on a dense grid
f_dense = np.linspace(net.f[0], net.f[-1], 2001)
s_dense = vf.get_model_response(f_dense)

# Wrap as a scikit-rf Network
net_dense = rf.Network(f=f_dense, s=s_dense, z0=net.z0)

# Write dense, fitted Touchstone
net_dense.write_touchstone("device_fitted.s2p")
print("Wrote dense fitted S-parameters to device_fitted.s2p")

# Optional: plot comparison for one element, e.g. S21
plt.figure()
plt.plot(net.f/1e9, 20*np.log10(np.abs(net.s[:,1,0])), 'o-', label="HFSS sparse")
plt.plot(f_dense/1e9, 20*np.log10(np.abs(s_dense[:,1,0])), '-', label="VectorFit dense")
plt.xlabel("Frequency [GHz]")
plt.ylabel("|S21| [dB]")
plt.legend()
plt.title("HFSS vs Vector Fitting")
plt.grid(True)
plt.show()
