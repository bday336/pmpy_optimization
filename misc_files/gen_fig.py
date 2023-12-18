import numpy as np
from function_bank import dvAfunc,dlBfunc
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

# data1 = np.loadtxt('local_optimization_data/pathdata/t_0.txt')
# for a in range(0,249000,2000):
#     data2 = np.loadtxt('local_optimization_data/pathdata/t_{}.txt'.format(a))


#     xlist = np.linspace(9.5, 10.5, 1000)
#     ylist = np.linspace(2.0, 3.0, 1000)
#     X, Y = np.meshgrid(xlist, ylist)
#     Z = dvAfunc(Y,5.) - dlBfunc(X)


#     fig,ax=plt.subplots(1,1,figsize=(6,5))
#     cp = ax.contourf(X, Y, Z, 100)
#     ax.plot(data1[:,0],data1[:,1],'r')
#     ax.plot(data2[:,0],data2[:,1],'k')
#     fig.colorbar(cp) # Add a colorbar to a plot
#     ax.set_title('Control Space Curvature')
#     ax.set_xlabel('l')
#     ax.set_ylabel('v')
#     ax.set_xlim(9.5,10.5) 
#     ax.set_ylim(2,3) 
#     #plt.savefig('local_optimization_data/figs/loop_t{}.pdf'.format(a), bbox_inches='tight')
#     plt.close()
#     # plt.show()

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

data1 = np.loadtxt('local_optimization_data/constant_E/pathdata/t_0.txt')
# data1 = np.loadtxt('local_optimization_data/constant_H/pathdata/t_0.txt')

xlist = np.linspace(9.5, 10.5, 1000)
ylist = np.linspace(2.0, 3.0, 1000)
X, Y = np.meshgrid(xlist, ylist)
Z = dvAfunc(Y,5.) - dlBfunc(X)

cp = ax.contourf(X, Y, Z, 100)
fig.colorbar(cp) # Add a colorbar to a plot


with writer.saving(fig, "eloop.mp4", 100):
    for a in range(0,1000000+1000,1000):
        print(a)
        data2 = np.loadtxt('local_optimization_data/constant_E/pathdata/t_{}.txt'.format(a))
        # data2 = np.loadtxt('local_optimization_data/constant_H/pathdata/t_{}.txt'.format(a))
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

