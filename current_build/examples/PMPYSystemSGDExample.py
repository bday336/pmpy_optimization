# Allow for package import
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.getcwd())+"/src")

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from src.function_bank import Afunc, Bfunc, dvAfunc, dlBfunc, efunc, hfunc, egradfunc, hgradfunc
from src.GradientDescent import GradientDescent

### Sample Script
### Example simulation of a finding optimal deformation to minimize energy expenditure for PMPY microswimmer (gradient descent)




## System Data (length in units of micrometers)
cell_width  = 9.2                               # Cell Width (𝜇m) of Euglena (Rossi et al. Kinematics of flagellar swimming in Euglena gracilis: Helical trajectories and flagellar shapes 2017)
cell_length = 50.8                              # Cell Length (𝜇m) of Euglena (Rossi et al. Kinematics of flagellar swimming in Euglena gracilis: Helical trajectories and flagellar shapes 2017)
vt  = 2.*(4./3.)*np.pi*(cell_width/2.)**3.      # Total volume of both spheres in 𝜇m^3
mu  = .95/(1e6)                                 # Absolute (Dynamic) viscosity of medium in N*s/m^2 (set to gylcerine) - (1e6) for conversion to 𝜇m

## Trial solution
initpath = []
nump = 1000
t_arr, dt= np.linspace(0.,2.*np.pi,nump,retstep=True)

for a in t_arr:
    initpath.append([10.*np.cos(a)+(cell_length - 2*cell_width), 100.*np.sin(a) + vt*.75])


# Optimization parameters
learning_rate = 1e1
tolerance = 1e-15
maximum_iterations = int(1e6)  # Default 1e6
params = [vt,mu,dt,nump]


## Initialize Simulation Object and Run Simulation
opt_test = GradientDescent([efunc, hfunc], [egradfunc, hgradfunc], initpath, params, learning_rate, tolerance, maximum_iterations)
opt_test.run()


## Read-In Data File for Data Analysis and Visualization
pdata = np.load("pmpy_gdopt_100_pdat.npy")
edata = np.load("pmpy_gdopt_100_edat.npy")
hdata = np.load("pmpy_gdopt_100_hdat.npy")

# --------------------------------------------------------------------
### Plot trajectory in the Poincare disk model with distance plots ###
# --------------------------------------------------------------------

## Plot Space Optimal Trajectory and Corresponding Holonomy
fig = plt.figure(figsize=(18,6))

## Plot the Optimal Trajectory in Control Space
ax1=fig.add_subplot(1,3,1)

# Generate the colormap of the curvature in control space
xscale, yscale = [ 2e-2 , 20**(.25) ]
xlist = np.linspace(2*cell_width, cell_length, 1000)
ylist = np.linspace(1e0, vt, 1000)
X, Y = np.meshgrid(xlist, ylist)
Z = dvAfunc(Y,vt) - dlBfunc(X)
cp = ax1.contourf(X, Y, Z, 1000)
fig.colorbar(cp) # Add a colorbar to a plot

# Draw Optimal Trajectory
ax1.plot(pdata[0][:,0],pdata[0][:,1],'r', linewidth=2,label = 'Trial Solution')
ax1.plot(pdata[-1][:,0],pdata[-1][:,1],'k', linewidth=2,label = 'Optimal Solution')



## Plot the Energy of Optimization
ax2=fig.add_subplot(1,3,2)

ax2.plot(edata,'r', linewidth=2)
ax2.legend()
ax2.set_title('Simulation Data')
ax2.set_xlabel('t')
ax2.set_ylabel('x')

## Plot the Holonomy of Optimization
ax2=fig.add_subplot(1,3,3)

ax2.plot(hdata,'r', linewidth=2)
ax2.legend()
ax2.set_title('Simulation Data')
ax2.set_xlabel('t')
ax2.set_ylabel('x')

fig.tight_layout()	

plt.show()







