# Sub-Riemannian Geodesic Script
# Generate array of geodesics to compare with result
# of gradient flow

import numpy as np
import matplotlib.pyplot as plt
from function_bank import dvAfunc,dlBfunc
from function_bank import dynfunc, dynjac, difffunc3s


# Solver Setup
nump = 10000
t_max = 10
t_arr, dt= np.linspace(0.,t_max,nump,retstep=True)
datalist = np.zeros((500,4))

a11,a12,a13 = [5./36., 2./9. - np.sqrt(15.)/15., 5./36. - np.sqrt(15.)/30.]
a21,a22,a23 = [5./36. + np.sqrt(15.)/24., 2./9., 5./36. - np.sqrt(15.)/24.]
a31,a32,a33 = [5./36. + np.sqrt(15.)/30., 2./9. + np.sqrt(15.)/15., 5./36.]

bs1,bs2,bs3 = [5./18., 4./9., 5./18.]

# Initial Data
vt = 5.
lam = 28.9
localt = .00101
local_opt_data = np.loadtxt('local_optimization_data/pathdata/t_248000.txt')
startp = local_opt_data[0]
startv = (local_opt_data[0] - local_opt_data[1])/localt
startvec = np.array([startp[0], startp[1], startv[0], startv[1]])

for lam in range(0,41,1):
    datalist = np.zeros((t_arr.shape[0],4))
    startvec = np.array([startp[0], startp[1], startv[0], startv[1]])
    datalist[0] = startvec.copy()


    # First Step
    step = 1

    while (step <= 500):
        # Initial Guess - Explicit Euler
        k = dynfunc(startvec,lam,vt)
        x1guess = startvec + (1./2. - np.sqrt(15.)/10.)*dt*k
        x2guess = startvec + (1./2.)*dt*k
        x3guess = startvec + (1./2. + np.sqrt(15.)/10.)*dt*k
        k1 = dynfunc(x1guess,lam,vt)
        k2 = dynfunc(x2guess,lam,vt)
        k3 = dynfunc(x3guess,lam,vt)

        # Check Error Before iterations
        er = difffunc3s(startvec, lam, vt, k1, k2, k3, dt)

        # Begin Iterations
        counter = 0
        while (np.linalg.norm(er) >= 1e-10 and counter <= 100):
            j1 = dynjac(startvec + (a11*k1 + a12*k2 + a13*k3)*dt,lam,vt)
            j2 = dynjac(startvec + (a21*k1 + a22*k2 + a23*k3)*dt,lam,vt)
            j3 = dynjac(startvec + (a31*k1 + a32*k2 + a33*k3)*dt,lam,vt)
            
            fullj = np.block([
                [np.eye(k.shape[0]) - dt*a11*j1, -(dt*a12*j1), -(dt*a13*j1)],
                [-(dt*a21*j2), np.eye(k.shape[0]) - dt*a22*j2, -(dt*a23*j2)],
                [-(dt*a31*j3), -(dt*a32*j3), np.eye(k.shape[0]) - dt*a33*j3]
            ])

            linsolve = np.linalg.solve(fullj,-er)

            k1 = k1 + linsolve[0:k.shape[0]]
            k2 = k2 + linsolve[k.shape[0]:2*k.shape[0]]
            k3 = k3 + linsolve[2*k.shape[0]:3*k.shape[0]]

            er = difffunc3s(startvec, lam, vt, k1, k2, k3, dt)

            counter += 1

        startvec = startvec + dt*(bs1*k1 + bs2*k2 + bs3*k3)
        datalist[step] = startvec.copy()
        if step%100==0:
                print(step)
        step += 1

    print(lam)
    np.savetxt("local_optimization_data/geodesics/lam_{}.txt".format(lam), np.array(datalist.copy()))


# # First Step
# datalist[0] = startvec.copy()
# step = 1

# while (step < 500):
#     # Initial Guess - Explicit Euler
#     k = dynfunc(startvec,lam,vt)
#     x1guess = startvec + (1./2. - np.sqrt(15.)/10.)*dt*k
#     x2guess = startvec + (1./2.)*dt*k
#     x3guess = startvec + (1./2. + np.sqrt(15.)/10.)*dt*k
#     k1 = dynfunc(x1guess,lam,vt)
#     k2 = dynfunc(x2guess,lam,vt)
#     k3 = dynfunc(x3guess,lam,vt)

#     # Check Error Before iterations
#     er = difffunc3s(startvec, lam, vt, k1, k2, k3, dt)

#     # Begin Iterations
#     counter = 0
#     while (np.linalg.norm(er) >= 1e-10 and counter <= 100):
#         j1 = dynjac(startvec + (a11*k1 + a12*k2 + a13*k3)*dt,lam,vt)
#         j2 = dynjac(startvec + (a21*k1 + a22*k2 + a23*k3)*dt,lam,vt)
#         j3 = dynjac(startvec + (a31*k1 + a32*k2 + a33*k3)*dt,lam,vt)
        
#         fullj = np.block([
#             [np.eye(k.shape[0]) - dt*a11*j1, -(dt*a12*j1), -(dt*a13*j1)],
#             [-(dt*a21*j2), np.eye(k.shape[0]) - dt*a22*j2, -(dt*a23*j2)],
#             [-(dt*a31*j3), -(dt*a32*j3), np.eye(k.shape[0]) - dt*a33*j3]
#         ])

#         linsolve = np.linalg.solve(fullj,-er)

#         k1 = k1 + linsolve[0:k.shape[0]]
#         k2 = k2 + linsolve[k.shape[0]:2*k.shape[0]]
#         k3 = k3 + linsolve[2*k.shape[0]:3*k.shape[0]]

#         er = difffunc3s(startvec, lam, vt, k1, k2, k3, dt)

#         counter += 1

#     startvec = startvec + dt*(bs1*k1 + bs2*k2 + bs3*k3)
#     datalist[step] = startvec.copy()
#     if step%100==0:
#             print(step)
#     step += 1



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

data2 = []
old_data = datalist
# old_data = np.loadtxt('local_optimization_data/geodesics/lam_29.txt')
for b in old_data:
    data2.append([b[0],b[1]])
data2 = np.array(data2)
cp = ax.contourf(X, Y, Z, 100)
ax.plot(data1[:,0],data1[:,1],'r')
ax.plot(data2[:,0],data2[:,1],'k')
plt.show()





