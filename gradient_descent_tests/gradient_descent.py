# This is the base script for running gradient descent
# Designed for use with the PMPY swimmer

import numpy as np
import os
import matplotlib.pyplot as plt
from function_bank import efunc, hfunc, egradfunc, hgradfunc

# System parameters
vt = .001                        # Total volume of both spheres in m^3 (steve default 5)
mu = .95                        # Absolute (Dynamic) viscosity of medium in N*s/m^2 (set to gylcerine)

# Generate the initial trial path
initpath = []
nump = 500
t_arr, dt= np.linspace(0.,2.*np.pi,nump,retstep=True)

for a in t_arr:
    # initpath.append([.05*np.cos(a)+1., .00025*np.sin(a) + vt*.5])
    # initpath.append([.005*np.cos(a)+1., .0001*np.sin(a) + vt*.6])
    # initpath.append([.02*np.cos(a)+1., .0001*np.sin(a) + vt*.6])
    # initpath.append([.02*np.cos(a)+1., .0001*np.sin(a) + vt*.6])
    # initpath.append([.005*np.cos(a)+.2, .0002*np.sin(a) + vt*.7])
    initpath.append([.02*np.cos(a)+.5, .0003*np.sin(a) + vt*.55])

# Restart flow
# initpath = np.load('local_optimization_data/constant_H/pathdata/data2.npy')[-1]

# Optimization parameters
counter = 0
learnrate = 1e-9
max_iterations = 1000000  # Default 1e6
data_Save = True

elist = []
hlist = []
pathlist = []


# Initialize first step (Constant_H)

pathlist.append(initpath)
elist.append(efunc(pathlist[-1],vt,mu,dt))
hlist.append(hfunc(pathlist[-1],vt,dt))
# if data_Save:
#     np.savetxt("local_optimization_data/constant_H/pathdata/t_0.txt", np.array(pathlist[0]))
#     np.savetxt("local_optimization_data/constant_H/edata/t_0.txt", np.array([elist[0]]))
#     np.savetxt("local_optimization_data/constant_H/hdata/t_0.txt", np.array([hlist[0]]))

egrad = egradfunc(pathlist[-1], vt, mu, dt)
hgrad = hgradfunc(pathlist[-1], vt, dt)
modgrad = egrad - (np.dot(egrad.flatten(),hgrad.flatten()))/(np.dot(hgrad.flatten(),hgrad.flatten()))*hgrad

pathpart = list(pathlist[-1][0:nump-1] - learnrate*modgrad)
pathpart.append(pathpart[0])

pathlist.append(pathpart)
elist.append(efunc(pathlist[-1],vt,mu,dt))
hlist.append(hfunc(pathlist[-1],vt,dt))

# Optimization
while counter<=max_iterations and abs(elist[-1]-elist[-2]) >= 1e-12:

    egrad = egradfunc(pathlist[-1], vt,mu, dt)
    hgrad = hgradfunc(pathlist[-1], vt, dt)
    modgrad = egrad - (np.dot(egrad.flatten(),hgrad.flatten()))/(np.dot(hgrad.flatten(),hgrad.flatten()))*hgrad

    pathpart = list(pathlist[-1][0:nump-1] - learnrate*modgrad)
    pathpart.append(pathpart[0])

    pathlist.append(pathpart)
    elist.append(efunc(pathlist[-1],vt,mu,dt))
    hlist.append(hfunc(pathlist[-1],vt,dt))

    counter += 1

    if counter%100==0:
        # if data_Save:
        #     np.savetxt("local_optimization_data/constant_H/pathdata/t_{}.txt".format(counter), np.array(pathlist[counter]))
        #     np.savetxt("local_optimization_data/constant_H/edata/t_{}.txt".format(counter), np.array([elist[counter]]))
        #     np.savetxt("local_optimization_data/constant_H/hdata/t_{}.txt".format(counter), np.array([hlist[counter]]))
        print(counter)
        print(elist[-1]-elist[-2])


if data_Save:
    np.save("local_optimization_data/constant_H/pathdata/data3", np.array(pathlist))
    np.save("local_optimization_data/constant_H/edata/data3", np.array([elist]))
    np.save("local_optimization_data/constant_H/hdata/data3", np.array([hlist]))


# Initialize first step (Constant_E)

# pathlist.append(initpath)
# elist.append(efunc(pathlist[-1],vt,mu,dt))
# hlist.append(hfunc(pathlist[-1],vt,dt))
# if data_Save:
#     np.savetxt("local_optimization_data/constant_E/pathdata/t_0.txt", np.array(pathlist[0]))
#     np.savetxt("local_optimization_data/constant_E/edata/t_0.txt", np.array([elist[0]]))
#     np.savetxt("local_optimization_data/constant_E/hdata/t_0.txt", np.array([hlist[0]]))

# egrad = egradfunc(pathlist[-1], vt,mu, dt)
# hgrad = hgradfunc(pathlist[-1], vt, dt)
# modgrad = hgrad - (np.dot(egrad.flatten(),hgrad.flatten()))/(np.dot(egrad.flatten(),egrad.flatten()))*egrad

# pathpart = list(pathlist[-1][0:nump-1] + learnrate*modgrad)
# pathpart.append(pathpart[0])

# pathlist.append(pathpart)
# elist.append(efunc(pathlist[-1],vt,mu,dt))
# hlist.append(hfunc(pathlist[-1],vt,dt))

# # Optimization
# while counter<=max_iterations and abs(hlist[-1]-hlist[-2]) >= 1e-12:

#     egrad = egradfunc(pathlist[-1], vt,mu, dt)
#     hgrad = hgradfunc(pathlist[-1], vt, dt)
#     modgrad = hgrad - (np.dot(egrad.flatten(),hgrad.flatten()))/(np.dot(egrad.flatten(),egrad.flatten()))*egrad

#     pathpart = list(pathlist[-1][0:nump-1] + learnrate*modgrad)
#     pathpart.append(pathpart[0])

#     pathlist.append(pathpart)
#     elist.append(efunc(pathlist[-1],vt,mu,dt))
#     hlist.append(hfunc(pathlist[-1],vt,dt))

#     counter += 1

#     if counter%100==0:
#         # if data_Save:
#         #     np.savetxt("local_optimization_data/constant_E/pathdata/t_{}.txt".format(counter), np.array(pathlist[counter]))
#         #     np.savetxt("local_optimization_data/constant_E/edata/t_{}.txt".format(counter), np.array([elist[counter]]))
#         #     np.savetxt("local_optimization_data/constant_E/hdata/t_{}.txt".format(counter), np.array([hlist[counter]]))
#         print(counter)
#         print(hlist[-1])
#         print(abs(hlist[-1]-hlist[-2]))
#         print(elist[-1])
#         print(abs(elist[-1]-elist[-2]))

# if data_Save:
#     np.save("local_optimization_data/constant_E/pathdata/data", np.array(pathlist))
#     np.save("local_optimization_data/constant_E/edata/data", np.array([elist]))
#     np.save("local_optimization_data/constant_E/hdata/data", np.array([hlist]))


# fig,ax=plt.subplots(1,1,figsize=(5,5))
# ax.plot(np.array(pathlist)[0][:,0],np.array(pathlist)[0][:,1],'r')
# ax.plot(np.array(pathlist)[-1][:,0],np.array(pathlist)[-1][:,1],'k')
# ax.set_title('Control Space Curvature')
# ax.set_xlabel('l')
# ax.set_ylabel('v')
# ax.set_xlim(9.5,10.5) 
# ax.set_ylim(2,3) 
# plt.show()


