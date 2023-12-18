# Generic Simulation Script

import sys
import numpy as np
import matplotlib.pyplot as plt
from integrator_bank import gausss1, gausss2, gausss3, rads2, rads3
from function_bank import dvAfunc,dlBfunc
from pmpyfunc import pmpyfunc
from pmpyjac import pmpyjac

# Solver Setup
# Command Line Input
# 1 - Lambda Value
# 2 - Angle of Initial Velocity (Multiple of pi/8)
# 3 - Total Simulation Time t_max
# 4 - Time Step dt
# 5 - Naming Options for File Name

# Time array based on time step
dt = float(sys.argv[4])         # Number of steps
t_max = float(sys.argv[3])      # Total simulation time
t_arr = np.arange(0.,t_max+dt,dt)

# Time array based on number of steps
# nump = 10000    # Number of steps
# t_max = 10      # Total simulation time
# t_arr, dt= np.linspace(0.,t_max,nump,retstep=True)

# Simulation data container

# Sim Data
gs3simdatalist = np.zeros((t_arr.shape[0],4))


# Initial Data
lam = float(sys.argv[1])        # Constant lambda due to group being abelian (steve default 88.05)
vt = .001                        # Total volume of both spheres in m^3 (steve default 5)
mu = .95                        # Absolute (Dynamic) viscosity of medium in N*s/m^2 (set to gylcerine)
params = [lam,vt,mu]

# Initial Conditions
ang = float(sys.argv[2])*np.pi/8.
# Start in reference position
# startvec = np.array([.2, vt*.5, np.cos(ang)*1e-2,np.sin(ang)*1e-2])
startvec = np.array([1, vt*.5, np.cos(ang)*1e-4,np.sin(ang)*1e-6])   # Using the correct limit r1+r2<<l


# Sim add initial conditions
gs3simdatalist[0] = startvec.copy()

# First Step
step = 1

# Sim first step
gs3simdatalist[step] = gausss3(startvec=startvec,params=params,dynfunc=pmpyfunc,dynjac=pmpyjac,dt=dt) 
print(gs3simdatalist[step])

startvecgs3sim = gs3simdatalist[step]

step += 1

while (step <= t_max/dt):
    # print(step)
    # Sim step
    gs3simdatalist[step] = gausss3(startvec=startvecgs3sim,params=params,dynfunc=pmpyfunc,dynjac=pmpyjac,dt=dt) 
    print(gs3simdatalist[step])
    startvecgs3sim = gs3simdatalist[step]

    if step%1000==0:
            print(step)
    step += 1

# Save data
# if str(sys.argv[6])=="p":
#     np.save("purcell2d_lxp{}_lyp{}_lzp{}_ang{}_tmax{}_dt001".format(str(lx).split('.')[-1],str(ly).split('.')[-1],str(lz).split('.')[-1],str(int(sys.argv[4])),int(t_max)),gs3simdatalist)
# For lambda of 1
if str(sys.argv[5])=="p":
    np.save("pmpy_lamp{}_ang{}_tmax{}_dt{}".format(str(lam).split('.')[-1],str(int(sys.argv[2])),int(t_max),str(dt).split('.')[-1]),gs3simdatalist)
# For lambda of 1
elif str(sys.argv[5])=="i":
    np.save("pmpy_lam{}_ang{}_tmax{}_dt{}".format(str(int(sys.argv[1])),str(int(sys.argv[2])),str(int(sys.argv[3])),str(dt).split('.')[-1]),gs3simdatalist)
# # For lambda of 1 with offset config
# elif str(sys.argv[6])=="o":
#     np.save("purcell2doff_lx{}_ly{}_lz{}_ang{}_tmax{}_dt001".format(str(int(sys.argv[1])),str(int(sys.argv[2])),str(int(sys.argv[3])),str(int(sys.argv[4])),int(t_max)),gs3simdatalist)


# # Load data
# data3 = gs3simdatalist
# # data3 = np.load("pmpy_lam{}_ang{}_tmax{}_dt001.npy".format(str(int(sys.argv[1])),str(int(sys.argv[2])),str(int(sys.argv[3]))))

# xlist = np.linspace(.95, .105, 1000)
# ylist = np.linspace(0.00001, .00099, 1000)
# X, Y = np.meshgrid(xlist, ylist)
# Z = dvAfunc(Y,vt) - dlBfunc(X)

# # phase plot of angles
# fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))

# # cp = ax1.contourf(X, Y, Z, 1000)

# ax1.plot(data3[:,0],data3[:,1],'r')
# ax2.plot(data3[:,2],data3[:,3],'k')


# # fig.colorbar(cp) # Add a colorbar to a plot
# # ax.set_title('Control Space Curvature')
# ax1.set_xlabel('l')
# ax1.set_ylabel('v')
# # ax1.set_xlim(.95,1.05) 
# ax1.set_ylim(0,.001)
# # ax.set_xlabel('t')
# # ax.set_ylabel('l')
# plt.show()







