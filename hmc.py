import numpy as np
from plotter import Plotter


def hmc(w, epsilon, T, H, H_grad, plotter):
    w_old = w
    p = np.random.normal(size=len(w))
    h_old = np.sum(p ** 2) / 2 + H(w_old)
    for _ in range(T):
        p_middle = p - epsilon / 2.0 * H_grad(w)
        w = w + epsilon * p_middle
        p = p_middle - epsilon / 2.0 * H_grad(w)

        plotter.propose(w[0], w[1])

    h = np.sum(p ** 2) / 2 + H(w)
    delta_h = h - h_old
    P = min(1, np.exp(-delta_h))
    if np.random.uniform(0,1) > P:
        w = w_old
        plotter.reject()
    else:
        plotter.accept()

    return w

epsilon = 0.5
T = 10
step = 100

normal_dens = lambda x, mean: (1.0 / (2 * np.pi) ** (2 * 0.5) * np.exp(-np.sum((x - mean) ** 2) / 2.0))
H = lambda w: -np.log(0.3 * normal_dens(w, np.array([-1,-1])) + 0.7 * normal_dens(w, np.array([2,2])))
H_grad = lambda w: (0.3 * normal_dens(w, np.array([-1,-1])) * (w - np.array([-1,-1])) + 0.7 * normal_dens(w, np.array([2,2])) * (w - np.array([2,2])) ) / (0.3 * normal_dens(w, np.array([-1,-1])) + 0.7 * normal_dens(w, np.array([2,2])))

w = np.random.normal(size=2)
plotter = Plotter(T * step, H)

for i in range(step):
    w = hmc(w, epsilon, T, H, H_grad, plotter)

plotter.flush('hmc.gif')
