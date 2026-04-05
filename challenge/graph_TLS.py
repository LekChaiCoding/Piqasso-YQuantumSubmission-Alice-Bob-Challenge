import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp

d_lts_nm = np.asarray([0.25,0.45,0.5,0.75,1.0])
d_lts_m = np.asarray([0.25,0.45,0.5,0.55,0.75,1.0])
Tz_nm = [46.8863, 45.0595, 44.8096, 41.5399, 39.1777]
Tz_m = [56.2788, 52.1843, 52.0726, 51.5466, 47.8494, 44.0028]
Tx_nm = [0.143624, 0.137191, 0.136905, 0.127128, 0.120485]
Tx_m = [0.174459, 0.162165, 0.159888, 0.158163, 0.147041, 0.134694]
deltas_m = [-0.2038, -0.1666, -0.1580, -0.1748, -0.1139, -0.0746]

plt.scatter(d_lts_nm, Tz_nm)
plt.scatter(d_lts_m, Tz_m, c="red")
plt.xlabel(r"$\delta$")
plt.ylabel(r"$T_z$")
plt.legend(["Circular", "Moon"])
plt.title(r"$\langle Z\rangle$ Lifetime vs. TLS Interaction Strength, $\delta$")
plt.show()

plt.scatter(d_lts_nm, Tx_nm)
plt.scatter(d_lts_m, Tx_m, c="red")
plt.xlabel(r"$\delta$")
plt.ylabel(r"$T_x$")
plt.legend(["Circular", "Moon"])
plt.title(r"$\langle X\rangle$ Lifetime vs. TLS Interaction Strength, $\delta$")
plt.show()

plt.scatter(d_lts_m,deltas_m)
plt.ylabel(r"$\lambda$")
plt.xlabel(r"$\delta$")
plt.title("Moon Cat State Squeezing Parameter vs. TLS Interaction Strength")
res = sp.linregress(d_lts_m, deltas_m)
xs = np.linspace(0.0,1.0001,100)
plt.plot(xs,res[0]*xs+res[1],c = "red",ls = ":")
plt.show()