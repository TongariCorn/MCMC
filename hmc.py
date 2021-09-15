import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation


def hmc(w, epsilon, T, H, H_grad):
    w_old = w
    p = np.random.normal(size=len(w))
    h_old = 0.5 * np.sum(p ** 2) + H(w_old)
    for _ in range(T):
        p_middle = p - epsilon / 2.0 * H_grad(w)
        w = w + epsilon * p_middle
        p = p_middle - epsilon / 2.0 * H_grad(w)

    h = 0.5 * np.sum(p ** 2) + H(w)
    delta_h = h - h_old
    P = min(1, np.exp(-delta_h))
    if np.random.uniform(0,1) > P:
        w = w_old
    return w

epsilon = 0.5
T = 1
#H = lambda w: np.sum(w ** 2) / 2.0
normal_dens = lambda x, mean: (1.0 / (2 * np.pi) ** (2 * 0.5) * np.exp(-np.sum((x - mean) ** 2) / 2.0))
H = lambda w: -np.log(0.3 * normal_dens(w, np.array([-1,-1])) + 0.7 * normal_dens(w, np.array([2,2])))
#H_grad = lambda w: w
H_grad = lambda w: -(0.3 * normal_dens(w, np.array([-1,-1])) * (w - np.array([-1,-1])) + 0.7 * normal_dens(w, np.array([2,2])) * (w - np.array([2,2])) ) / (0.3 * normal_dens(w, np.array([-1,-1])) + 0.7 * normal_dens(w, np.array([2,2])))

X = np.arange(-4, 4, 0.1)
Y = np.arange(-4, 4, 0.1)
X, Y = np.meshgrid(X, Y)
Z = [[ np.exp(-H([x, y])) for x in np.arange(-4, 4, 0.1) ] for y in np.arange(-4, 4, 0.1)]
#cont = plt.contour(X, Y, Z, levels=[0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16])

w = np.random.normal(size=2)
x = []
y = []
t = []
fig = plt.figure()
ims = []
for i in range(1000):
    w = hmc(w, epsilon, T, H, H_grad)
    #print(w[0], w[1])
    x.append(w[0])
    y.append(w[1])
    t.append(i)
    cont = plt.contour(X, Y, Z, levels=70, alpha=0.3)
    im = plt.scatter(x, y, c=t, cmap=cm.jet, marker='.', lw=0, vmin=0, vmax=1000)
    ims.append(cont.collections + [im])

ani = animation.ArtistAnimation(fig, ims, interval=0.01, repeat_delay=1000)
ani.save('hmc.gif', writer='imagemagick')
plt.xlim(-4,4)
plt.ylim(-4,4)


plt.savefig('fig.png')
