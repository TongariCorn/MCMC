import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import animation

class Plotter:
    def __init__(self, scale, H):
        self.scale = scale
        self.x = []
        self.y = []
        self.t = [0]
        
        self.proposed_x = []
        self.proposed_y = []
        self.proposed_t = []

        self.ims = []
        X = np.arange(-4, 6, 0.1)
        Y = np.arange(-4, 6, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = [[ np.exp(-H([x, y])) for x in np.arange(-4, 6, 0.1) ] for y in np.arange(-4, 6, 0.1)]

        self.fig = plt.figure()
        self.cont = plt.contour(X, Y, Z, levels=70, alpha=0.3)

    def propose(self, x, y):
        self.proposed_x.append(x)
        self.proposed_y.append(y)
        if not self.proposed_t:
            self.proposed_t.append(self.t[-1]+1)
        else:
            self.proposed_t.append(self.proposed_t[-1]+1)

        self.plot()

    def accept(self):
        self.x.extend(self.proposed_x)
        self.y.extend(self.proposed_y)
        self.t.extend(self.proposed_t)
        self.reject()

    def reject(self):
        self.proposed_x.clear()
        self.proposed_y.clear()
        self.proposed_t.clear()

    def plot(self):
        im = [ plt.scatter(self.x, self.y, c=self.t[1:], cmap=cm.jet, marker='.', lw=0, vmin=0, vmax=self.scale) ]
        if self.proposed_t:
            im.append(plt.scatter(self.proposed_x, self.proposed_y, c=self.proposed_t, cmap=cm.jet, marker='o', lw=0, vmin=0, vmax=self.scale, alpha=0.8))

        self.ims.append(self.cont.collections + im)

    def flush(self, filename):
        ani = animation.ArtistAnimation(self.fig, self.ims, interval=0.01, repeat_delay=1000)
        ani.save(filename, writer='imagemagick')
        self.ims.clear()
