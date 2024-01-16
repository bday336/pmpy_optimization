# Generic Simulation Script

import sys
import numpy as np
import matplotlib.pyplot as plt
from function_bank import Afunc,Bfunc

# Command Line Input
# 1 - Lambda Value
# 2 - Angle of Initial Velocity (Multiple of pi/8)
# 3 - Total Simulation Time t_max
# 4 - Time Step dt
# 5 - Naming Options for File Name

# Load data
data0 = np.load( "lam{}/pmpy_lam{}_ang{}_tmax{}_dt{}.npy".format( sys.argv[1],sys.argv[1],sys.argv[2],sys.argv[3],str(sys.argv[4]).split('.')[-1]))

# data1 = np.load( "lam{}/pmpy_lam{}_ang1_tmax{}_dt{}.npy".format( sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data2 = np.load( "lam{}/pmpy_lam{}_ang2_tmax{}_dt{}.npy".format( sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data3 = np.load( "lam{}/pmpy_lam{}_ang3_tmax{}_dt{}.npy".format( sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data4 = np.load( "lam{}/pmpy_lam{}_ang4_tmax{}_dt{}.npy".format( sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data5 = np.load( "lam{}/pmpy_lam{}_ang5_tmax{}_dt{}.npy".format( sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data6 = np.load( "lam{}/pmpy_lam{}_ang6_tmax{}_dt{}.npy".format( sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data7 = np.load( "lam{}/pmpy_lam{}_ang7_tmax{}_dt{}.npy".format( sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data8 = np.load( "lam{}/pmpy_lam{}_ang8_tmax{}_dt{}.npy".format( sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data9 = np.load( "lam{}/pmpy_lam{}_ang9_tmax{}_dt{}.npy".format( sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data10 = np.load("lam{}/pmpy_lam{}_ang10_tmax{}_dt{}.npy".format(sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data11 = np.load("lam{}/pmpy_lam{}_ang11_tmax{}_dt{}.npy".format(sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data12 = np.load("lam{}/pmpy_lam{}_ang12_tmax{}_dt{}.npy".format(sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data13 = np.load("lam{}/pmpy_lam{}_ang13_tmax{}_dt{}.npy".format(sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data14 = np.load("lam{}/pmpy_lam{}_ang14_tmax{}_dt{}.npy".format(sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))
# data15 = np.load("lam{}/pmpy_lam{}_ang15_tmax{}_dt{}.npy".format(sys.argv[1],sys.argv[1],sys.argv[2],str(sys.argv[3]).split('.')[-1]))

# Calculate the motion along fiber

# Data containers
link2veldat0 = []

caydat0 = []

fulldat0 = []

# Populate data containers
for a in range(data0.shape[0]):
    link2veldat0.append(Afunc(data0[a,1],.001)*data0[a,2] + Bfunc(data0[a,0])*data0[a,3])

for b in range(data0.shape[0]):
    caydat0.append(link2veldat0[b]*float(sys.argv[4]))

fulldat0.append(0)

for c in range(data0.shape[0]):
    fulldat0.append(fulldat0[-1] + caydat0[c])

# phase plot of angles
fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,figsize=(14,7))

ax1.set_title("l vs. t")
ax1.scatter(np.arange(0,float(sys.argv[3])+float(sys.argv[4]),float(sys.argv[4])),data0[:,0],10,label="l")
ax2.set_title("v vs. t")
ax2.scatter(np.arange(0,float(sys.argv[3])+float(sys.argv[4]),float(sys.argv[4])),data0[:,1],10,label="v")

# ax3.text(0.5, 0.5, 'l{} Ang {} tmax {} dt {}'.format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]), horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize='x-large')
fig.suptitle('l{} Ang {} tmax {} dt {}'.format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
ax4.set_title("ld vs. t")
ax4.scatter(np.arange(0,float(sys.argv[3])+float(sys.argv[4]),float(sys.argv[4])),data0[:,2],10,label="dl")
ax5.set_title("vd vs. t")
ax5.scatter(np.arange(0,float(sys.argv[3])+float(sys.argv[4]),float(sys.argv[4])),data0[:,3],10,label="dv")


ax3.set_title("xdot vs. time")
ax3.plot(np.arange(0,float(sys.argv[3])+float(sys.argv[4]),float(sys.argv[4])),np.asarray(link2veldat0)[:])

ax6.set_title("x vs. time")
ax6.plot(np.arange(0,float(sys.argv[3])+2.*float(sys.argv[4]),float(sys.argv[4])),np.asarray(fulldat0)[:])

plt.savefig('recon_data_l{}_ang{}_tmax{}_dt{}.png'.format(sys.argv[1],sys.argv[2],sys.argv[3],str(sys.argv[4]).split('.')[-1]))

# ax1.plot(data0[:,0],data0[:,1],label="ang0")
# ax1.plot(data1[:,0],data1[:,1],label="ang1")
# ax1.plot(data2[:,0],data2[:,1],label="ang2")
# ax1.plot(data3[:,0],data3[:,1],label="ang3")
# ax1.plot(data4[:,0],data4[:,1],label="ang4")
# ax1.plot(data5[:,0],data5[:,1],label="ang5")
# ax1.plot(data6[:,0],data6[:,1],label="ang6")
# ax1.plot(data7[:,0],data7[:,1],label="ang7")
# ax1.plot(data8[:,0],data8[:,1],label="ang8")
# ax1.plot(data9[:,0],data9[:,1],label="ang9")
# ax1.plot(data10[:,0],data10[:,1],label="ang10")
# ax1.plot(data11[:,0],data11[:,1],label="ang11")
# ax1.plot(data12[:,0],data12[:,1],label="ang12")
# ax1.plot(data13[:,0],data13[:,1],label="ang13")
# ax1.plot(data14[:,0],data14[:,1],label="ang14")
# ax1.plot(data15[:,0],data15[:,1],label="ang15")

# ax2.plot(data0[:,2],data0[:,3],label="ang0")
# ax2.plot(data1[:,2],data1[:,3],label="ang1")
# ax2.plot(data2[:,2],data2[:,3],label="ang2")
# ax2.plot(data3[:,2],data3[:,3],label="ang3")
# ax2.plot(data4[:,2],data4[:,3],label="ang4")
# ax2.plot(data5[:,2],data5[:,3],label="ang5")
# ax2.plot(data6[:,2],data6[:,3],label="ang6")
# ax2.plot(data7[:,2],data7[:,3],label="ang7")
# ax2.plot(data8[:,2],data8[:,3],label="ang8")
# ax2.plot(data9[:,2],data9[:,3],label="ang9")
# ax2.plot(data10[:,2],data10[:,3],label="ang10")
# ax2.plot(data11[:,2],data11[:,3],label="ang11")
# ax2.plot(data12[:,2],data12[:,3],label="ang12")
# ax2.plot(data13[:,2],data13[:,3],label="ang13")
# ax2.plot(data14[:,2],data14[:,3],label="ang14")
# ax2.plot(data15[:,2],data15[:,3],label="ang15")
# ax.set_xlabel('t')
# ax.set_ylabel('l')
# ax1.legend()
# ax2.legend()
plt.show()