#MODIFIED FROM 1-challenge.ipynb

import dynamiqs as dq
import jax.numpy as jnp
from matplotlib import pyplot as plt

from jax import vmap, jit
from cmaes import SepCMA

from scipy.optimize import curve_fit
from scipy.optimize import least_squares


def evolve_state(initial_state, tfinal, eps_d_real: float, eps_d_im: float, g2_re: float, g2_im: float):
    na = 15 # Hilbert space dimension
    nb = 5
    a = dq.tensor(dq.destroy(na), dq.eye(nb)) # annihilaiton operator
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    kappa_b = 50 # MHz
    eps_d = eps_d_real + 1j*eps_d_im
    g_2 = g2_re + 1j*g2_im # MHz
    kappa_a = 1 # MHz

    eps_2 = 2 * g_2 * eps_d / kappa_b
    kappa_2 = 4 * jnp.abs(g_2)**2/kappa_b
    alpha_estimate = jnp.sqrt(2/kappa_2 * (eps_2 - kappa_a/4))

    H = jnp.conj(g_2) * a @ a @ b.dag() + g_2 * a.dag() @ a.dag() @ b - eps_d * b.dag() - jnp.conj(eps_d) * b 

    loss_b = jnp.sqrt(kappa_b) * b
    loss_a = jnp.sqrt(kappa_a) * a

    tsave = jnp.linspace(0, tfinal, 40)

    g_state = dq.coherent(na, alpha_estimate)
    e_state = dq.coherent(na, -alpha_estimate)

    basis = {
        "+z": g_state,
        "-z": e_state,
        "+x": (g_state + e_state) / jnp.sqrt(2),
        "-x": (g_state - e_state) / jnp.sqrt(2),
        "+y": (g_state + 1j*e_state) / jnp.sqrt(2),
        "-y": (g_state - 1j*e_state) / jnp.sqrt(2),
    }

    sx = (1j * jnp.pi * a.dag() @ a).expm()
    a2 = dq.powm(a,2)

    psi0 = dq.tensor(basis[initial_state], dq.fock(nb,0)) # initial state

    res = dq.mesolve(
        H, 
        [loss_b, loss_a], 
        psi0, 
        tsave, 
        options=dq.Options(progress_meter=False),
        exp_ops=[sx, a2, a, a.dag(), a.dag()@a]
    )

    return res

# model: y = A * exp(-t/tau) + C
def model(p, t):
    A, tau, C = p
    return A * jnp.exp(-t/tau) + C

def residuals(p, x, y):
    return model(p, x) - y


def robust_exp_fit(x, y):
    # smart initialization
    A0 = y.max() - y.min()
    C0 = y.min()
    tau0 = (x.max() - x.min())
    p0 = [A0, tau0, C0]

    # robust fit (soft_l1 or huber are key)
    res = least_squares(
        residuals,
        p0,
        args=(x, y),
        bounds=([0, 0, -jnp.inf], [jnp.inf, jnp.inf, jnp.inf]),
        loss="soft_l1",   # try "huber" too
        f_scale=0.1       # tune based on noise level
    )

    A, tau, C = res.x
    y_fit = model(res.x, x)

    return {
        "popt": res.x,
        "y_fit": y_fit,
    }

def compute_z_lifetime(ev_res):
    a2_exp = ev_res.expects[1,:]
    a_exp = ev_res.expects[2,:]
    adag_exp = ev_res.expects[3,:]
    num_exp = ev_res.expects[4,:].real
    phi = jnp.angle(a2_exp)/2
    Xphi = 0.5*(jnp.exp(1j*phi)*adag_exp+jnp.exp(-1j*phi)*a_exp)/jnp.sqrt(num_exp)
    szt = jnp.real(Xphi)
    ts = ev_res.tsave
    
    y = szt 
    x = ts 
    fit_z = robust_exp_fit(x, y)
    Tz = fit_z["popt"][1] #Z Lifetime

    return Tz

def compute_x_lifetime(ev_res):

    sxt = ev_res.expects[0,:].real
    ts = ev_res.tsave

    y = sxt 
    x = ts 
    fit = robust_exp_fit(x, y)
    Tx = fit["popt"][1]

    return Tx

def compute_vals(eps_d_real: float, eps_d_im: float, g2_re: float, g2_im: float, delta_d: float):
    ev_res_z = evolve_state("+z",50, eps_d_real, eps_d_im, g2_re, g2_im, delta_d)
    Tz = compute_z_lifetime(ev_res_z)
    ev_res_x = evolve_state("+x",0.25, eps_d_real, eps_d_im, g2_re, g2_im, delta_d)
    Tx = compute_x_lifetime(ev_res_x)
    return Tz,Tx

print(compute_vals(4,0,1,0,-1.0))

