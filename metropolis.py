import numpy as np
from plotter import Plotter


def metropolis(w, H, plotter):
    w_old = np.copy(w)
    w += np.random.normal(size=len(w))
    plotter.propose(w[0], w[1])

    delta_h = H(w) - H(w_old)
    P = min(1, np.exp(-delta_h))
    if np.random.uniform(0,1) > P:
        w = w_old
        plotter.reject()
    else:
        plotter.accept()

    return w

step = 1000

normal_dens = lambda x, mean: (1.0 / (2 * np.pi) ** (2 * 0.5) * np.exp(-np.sum((x - mean) ** 2) / 2.0))
H = lambda w: -np.log(0.3 * normal_dens(w, np.array([-1,-1])) + 0.7 * normal_dens(w, np.array([2,2])))

w = np.random.normal(size=2)
plotter = Plotter(step, H)

for i in range(step):
    print(i)
    w = metropolis(w, H, plotter)

plotter.flush('metro.gif')
