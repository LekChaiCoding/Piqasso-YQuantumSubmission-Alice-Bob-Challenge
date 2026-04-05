from Scratch import compute_vals
import matplotlib.pyplot as plt
import numpy as np

k_deltas = [0.0, 0.05, 0.1,0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Tz = [59.4658, 58.7897, 57.1178,55.4080, 54.1970,52.8338, 45.3459, 22.2865, 57.7398, 75.5854, 72.9526, 89.3430, 77.0109, 67.7573]
Tx = [0.177867, 0.176443, 0.171746,0.165209, 0.163844,0.159623, 0.141018, 0.109825, 0.164784, 0.233201, 0.231987, 0.246722,  0.245323, 0.206027]

plt.scatter(k_deltas,Tz)
plt.title(r"$\langle Z \rangle$ Lifetime vs. Kerr Nonlinearity Intensity")
plt.ylabel(r"$T_z$")
plt.axvline(0.4,ls=":",color = "red")
plt.xlabel(r"$\delta$")
plt.show()

plt.scatter(k_deltas,Tx)
plt.title(r"$\langle X \rangle$ Lifetime vs. Kerr Nonlinearity Intensity")
plt.ylabel(r"$T_x$")
plt.axvline(0.4,ls=":",color = "red")
plt.xlabel(r"$\delta$")
plt.show()