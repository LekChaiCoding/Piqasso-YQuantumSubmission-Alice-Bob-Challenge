from Scratch import compute_vals
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp

deltas = [0.5,0.75,1.0,1.25,1.5]
dds = [-0.5528,-0.7478,-1.0133,-1.2383,-1.46]
res = sp.linregress(np.asarray(deltas),np.asarray(dds))
print(res)
xs = np.arange(0.5,1.505,0.05)

plt.ylabel(r"$\Delta_d$")
plt.scatter(deltas,dds)
plt.plot(xs,res[0]*xs+res[1],ls = ":", c="red")
plt.text(1.3,-0.6, r"$r = ${pr}".format(pr = np.round(res[2],4)), fontsize=12)
plt.xlabel(r"$\Delta$")
plt.title("Counteracting Coefficient vs. Detuning Intensity")
plt.show()