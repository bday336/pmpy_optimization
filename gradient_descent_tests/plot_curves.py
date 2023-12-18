import numpy as np
from matplotlib import rc,rcParams
import matplotlib.pyplot as plt
from function_bank import dvAfunc,dlBfunc

# conval = "H"

edata = np.load('local_optimization_data/constant_H/edata/data.npy')
hdata = np.load('local_optimization_data/constant_H/hdata/data.npy')
pdata = np.load('local_optimization_data/constant_H/pathdata/data.npy')

# edata2 = np.load('local_optimization_data/constant_H/edata/data2.npy')
# hdata2 = np.load('local_optimization_data/constant_H/hdata/data2.npy')
# pdata2 = np.load('local_optimization_data/constant_H/pathdata/data2.npy')

# etot=np.array([edata,edata2]).flatten()
# htot=np.array([hdata,hdata2]).flatten()

# data1 = np.loadtxt('local_optimization_data/constant_{}/pathdata/t_0.txt'.format(conval))
# data2 = np.loadtxt('local_optimization_data/constant_{}/pathdata/t_852400.txt'.format("H"))
# data3 = np.loadtxt('local_optimization_data/constant_{}/pathdata/t_90000.txt'.format("E"))

# activate latex text rendering
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

xlist = np.linspace(.45,.55, 1000)
ylist = np.linspace(0.00001, .00099, 1000)
# xlist = np.linspace(.95, 1.05, 1000)
# ylist = np.linspace(0.00001, .00099, 1000)
X, Y = np.meshgrid(xlist, ylist)
Z = dvAfunc(Y,.001) - dlBfunc(X)


# Plot control space
# fig,ax=plt.subplots(1,1,figsize=(6,5))
# cp = ax.contourf(X, Y, Z, 100)
# ax.plot(data1[:,0],data1[:,1],'r')
# ax.plot(data2[:,0],data2[:,1],'k')
# ax.plot(data3[:,0],data3[:,1],'b')
# fig.colorbar(cp) # Add a colorbar to a plot
# ax.set_title('Control Space Curvature')
# ax.set_xlabel('l')
# ax.set_ylabel('v')
# ax.set_xlim(.19,.21) 
# # ax.set_xlim(.195,.205) 
# ax.set_ylim(0,.001) 
# plt.show()

# fig,ax=plt.subplots(1,1)
# # fig.canvas.draw()

# # cp = ax.contourf(X, Y, Z, 100)
# ax.plot(pdata[0][:,0],pdata[0][:,1],'r', linewidth=2,label = r'Trial Solution')
# ax.plot(pdata[-1][:,0],pdata[-1][:,1],'k', linewidth=2,label = r'Optimal Solution')

# # xlabels = [item.get_text() for item in ax.get_xticklabels()]
# # xlabels[1] = r'\textbf{0.0}'
# # xlabels[2] = r'\textbf{.25}'
# # xlabels[3] = r'\textbf{.5}'
# # xlabels[4] = r'\textbf{.75}'
# # xlabels[5] = r'\textbf{1.0}'

# # # for h data
# # ylabels = [item.get_text() for item in ax.get_yticklabels()]
# # ylabels[0] = r'\textbf{-4}'
# # ylabels[1] = r'\textbf{-3}'
# # ylabels[2] = r'\textbf{-2}'
# # ylabels[3] = r'\textbf{-1}'
# # ylabels[4] = r'\textbf{0}'
# # ylabels[5] = r'\textbf{1}'

# # For e data
# # ylabels = [item.get_text() for item in ax.get_yticklabels()]
# # ylabels[1] = r'\textbf{2.129}'
# # ylabels[2] = r'\textbf{2.130}'
# # ylabels[3] = r'\textbf{2.131}'
# # ylabels[4] = r'\textbf{2.132}'
# # ylabels[5] = r'\textbf{2.133}'

# # ax.set_xticklabels(xlabels)
# # ax.set_yticklabels(ylabels)
# ax.legend(fontsize="10",loc="lower left")
# plt.title(r'\textbf{Closed Loops in Shape Space}', fontsize=20)
# plt.ylabel(r'\textbf{v ($m^3$)}', fontsize=20, labelpad = 20)
# plt.xlabel(r'\textbf{l ($m$)}', fontsize=20, labelpad = 10)
# ax.xaxis.set_tick_params(labelsize=20)
# ax.yaxis.set_tick_params(labelsize=20)

# # fig.colorbar(cp) # Add a colorbar to a plot

# ax.set_xlim(.46,.54)
# ax.set_ylim(0,.001) 

# plt.tight_layout()
# plt.show()



###### E and H data plots

fig,ax=plt.subplots(1,1)
fig.canvas.draw()
ax.plot(edata[0],'r', linewidth=2)
# ax.plot(htot-htot[0],'r', linewidth=2)
# ax.plot(hdata[0]-hdata[0,0],'r', linewidth=2)
# ax.plot(hdata[0],'r', linewidth=2)

# xlabels = [item.get_text() for item in ax.get_xticklabels()]
# xlabels[1] = r'\textbf{0.0}'
# xlabels[2] = r'\textbf{.25}'
# xlabels[3] = r'\textbf{.5}'
# xlabels[4] = r'\textbf{.75}'
# xlabels[5] = r'\textbf{1.0}'

# # for h data
# ylabels = [item.get_text() for item in ax.get_yticklabels()]
# ylabels[0] = r'\textbf{-4}'
# ylabels[1] = r'\textbf{-3}'
# ylabels[2] = r'\textbf{-2}'
# ylabels[3] = r'\textbf{-1}'
# ylabels[4] = r'\textbf{0}'
# ylabels[5] = r'\textbf{1}'

# For e data
# ylabels = [item.get_text() for item in ax.get_yticklabels()]
# ylabels[1] = r'\textbf{2.129}'
# ylabels[2] = r'\textbf{2.130}'
# ylabels[3] = r'\textbf{2.131}'
# ylabels[4] = r'\textbf{2.132}'
# ylabels[5] = r'\textbf{2.133}'

# ax.set_xticklabels(xlabels)
# ax.set_yticklabels(ylabels)
plt.title(r'\textbf{Change in Holonomy of Loop}', fontsize=20)
plt.ylabel(r'\textbf{Energy Expenditure ($\times 10^{-13}$m)}', fontsize=20, labelpad = 20)
# plt.ylabel(r'\textbf{$\Delta$ Holonomy ($\times 10^{-13}$m)}', fontsize=20, labelpad = 20)
plt.xlabel(r'\textbf{Iterations ($\times 10^{6}$)}', fontsize=20, labelpad = 10)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.tight_layout()
plt.show()
