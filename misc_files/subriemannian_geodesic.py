# Sub-Riemannian Geodesic Script

import numpy as np
import matplotlib.pyplot as plt
from function_bank import dynfunc, dynjac, difffunc3s


# Solver Setup
nump = 10000
t_max = 10
t_arr, dt= np.linspace(0.,t_max,nump,retstep=True)
datalist = np.zeros((t_arr.shape[0],4))

a11,a12,a13 = [5./36., 2./9. - np.sqrt(15.)/15., 5./36. - np.sqrt(15.)/30.]
a21,a22,a23 = [5./36. + np.sqrt(15.)/24., 2./9., 5./36. - np.sqrt(15.)/24.]
a31,a32,a33 = [5./36. + np.sqrt(15.)/30., 2./9. + np.sqrt(15.)/15., 5./36.]

bs1,bs2,bs3 = [5./18., 4./9., 5./18.]

# Initial Data
amp = .5
theta = np.pi
vt = 5.
lam = 88.05
startvec = np.array([10, vt*.5, amp*np.cos(theta), amp*np.sin(theta)])

datalist[0] = startvec.copy()

# First Step
step = 1

while (step <= t_max/dt):
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


fig,ax=plt.subplots(1,1)
ax.plot(datalist[0:,0],datalist[0:,1],'r')
ax.set_title('Control Space Curvature')
ax.set_xlabel('l')
ax.set_ylabel('v')
plt.xlim(9.5, 10.5)
plt.ylim(2, 3)
plt.show()







