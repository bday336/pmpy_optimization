# Allow for package import
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.getcwd())+"/src")

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from src.function_bank import Afunc,Bfunc,dvAfunc,dlBfunc
from src.integrator_files.integrator_bank import gausss1, gausss2, gausss3, rads2, rads3
from src.test_system_simulations.test_system_bank import pmpyfunc, pmpyjac
from src.test_system_simulations.PMPYSystem import PMPYSystem

### Sample Script
### Example simulation of a finding optimal deformation to minimize energy expenditure for PMPY microswimmer



## Time array based on time step
dt = .001          # Time step size
t_max = 10.       # Total simulation time

## System Data (length in units of micrometers)
cell_width  = 9.2                               # Cell Width (𝜇m) of Euglena (Rossi et al. Kinematics of flagellar swimming in Euglena gracilis: Helical trajectories and flagellar shapes 2017)
cell_length = 50.8                              # Cell Length (𝜇m) of Euglena (Rossi et al. Kinematics of flagellar swimming in Euglena gracilis: Helical trajectories and flagellar shapes 2017)
lam = 1e-1                                        # Constant lambda due to group being abelian (steve default 88.05)
vt  = 2.*(4./3.)*np.pi*(cell_width/2.)**3.      # Total volume of both spheres in 𝜇m^3
mu  = .95/(1e6)                                 # Absolute (Dynamic) viscosity of medium in N*s/m^2 (set to gylcerine) - (1e6) for conversion to 𝜇m
params = [lam,vt,mu]

## Initial Data for System
ang = 2.*np.pi/8.  # Angle of initial velocity vector in v-l space
startvec = np.array([
    cell_length - 2*cell_width,        # Initial Length of Connecting Rod
    vt*.75,                          # Initial Volume of Left Bladder
    np.cos(ang)*1e2,                # Initial Velocity of Rod Extension
    np.sin(ang)*1e1                 # Initial Velocity of Left Bladder Expansion
    ])


## Solver 
solver_id = "gs3"   # Here using Gauss collocation method with 1 internal step

## Initialize Simulation Object and Run Simulation
sim_test = PMPYSystem(pmpyfunc, pmpyjac, params, dt, t_max, solver_id)
sim_test.set_initial_conditions(startvec)
sim_test.run()
sim_test.output_data()


## Read-In Data File for Data Analysis and Visualization
data1 = np.load("pmpy_{}_sim_tmax{}_dt{}.npy".format(solver_id, str(t_max), str(dt)))

# --------------------------------------------------------------------
### Plot trajectory in the Poincare disk model with distance plots ###
# --------------------------------------------------------------------

## Plot Space Optimal Trajectory and Corresponding Holonomy
fig = plt.figure(figsize=(18,6))

## Plot the Optimal Trajectory in Control Space
ax1=fig.add_subplot(1,2,1)

# Generate the colormap of the curvature in control space
xscale, yscale = [ 2e-2 , 20**(.25) ]
xlist = np.linspace(2*cell_width, cell_length, 1000)
ylist = np.linspace(1e0, vt, 1000)
X, Y = np.meshgrid(xlist, ylist)
Z = dvAfunc(Y,vt) - dlBfunc(X)
cp = ax1.contourf(X, Y, Z, 1000)
fig.colorbar(cp) # Add a colorbar to a plot

# Draw Optimal Trajectory
ax1.plot(data1[:,0],data1[:,1],'r')



## Plot the Corresponding Holonomy of Optimal Trajectory
ax2=fig.add_subplot(1,2,2)

# Calculate the motion along fiber

link2veldat0 = []
caydat0 = []
fulldat0 = []

for a in range(data1.shape[0]):
    link2veldat0.append(Afunc(data1[a,1],vt)*data1[a,2] + Bfunc(data1[a,0])*data1[a,3])
    caydat0.append((Afunc(data1[a,1],vt)*data1[a,2] + Bfunc(data1[a,0])*data1[a,3])*sim_test.dt)

fulldat0.append(0)
for c in range(data1.shape[0]):
    fulldat0.append(fulldat0[-1] + caydat0[c])


ax2.plot(np.append(sim_test.t_arr,sim_test.tmax+sim_test.dt),np.asarray(fulldat0)[:])
ax2.legend()
ax2.set_title('Simulation Data')
ax2.set_xlabel('t')
ax2.set_ylabel('x')

fig.tight_layout()	

plt.show()







