# Generate movie of geodesics to compare with result
# of gradient flow

import numpy as np
from function_bank import dvAfunc,dlBfunc
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

## Generates the movie file for the local optimization flow

# metadata = dict(title='Movie', artist='codinglikemad')
# writer = PillowWriter(fps=15, metadata=metadata)
# writer = FFMpegWriter(fps=15, metadata=metadata)
writer = FFMpegWriter(fps=15)

fig,ax=plt.subplots(1,1,figsize=(6,5))

plt.xlim(9.5, 10.5)
plt.ylim(2, 3)
plt.xlabel('l')
plt.ylabel('v')
plt.title('Control Space Curvature')

data1 = np.loadtxt('local_optimization_data/pathdata/t_248000.txt')

xlist = np.linspace(9.5, 10.5, 1000)
ylist = np.linspace(2.0, 3.0, 1000)
X, Y = np.meshgrid(xlist, ylist)
Z = dvAfunc(Y,5.) - dlBfunc(X)

cp = ax.contourf(X, Y, Z, 100)
fig.colorbar(cp) # Add a colorbar to a plot


with writer.saving(fig, "test.mp4", 100):
    for a in range(0,41,1):
        print(a)
        data2 = []
        old_data = np.loadtxt('local_optimization_data/geodesics/lam_{}.txt'.format(a))
        for b in old_data:
            data2.append([b[0],b[1]])
        data2 = np.array(data2)
        cp = ax.contourf(X, Y, Z, 100)
        ax.plot(data1[:,0],data1[:,1],'r')
        ax.plot(data2[:,0],data2[:,1],'k')
        plt.xlim(9.5, 10.5)
        plt.ylim(2, 3)
        plt.xlabel('l')
        plt.ylabel('v')
        plt.title('Control Space Curvature')
        writer.grab_frame()
        plt.cla()







