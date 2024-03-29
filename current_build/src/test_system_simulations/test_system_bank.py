##########################
# System Setup Functions #
##########################

import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arccos,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,linalg,add

def pmpyfunc(statevec,params):
    l,v,ld,vd = statevec
    lam,vt,mu = params
     
    ldd = (vd*(3*lam*v**(2/3)*(-2*v**(4/3) + 2*v**(1/3)*vt - v*(-v + vt)**(1/3) + vt*(-v + vt)**(1/3) + v**(2/3)*(-v + vt)**(2/3)) + 2*l**3*pi*(lam*vt + 6*6**(1/3)*ld*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)))/(36*6**(1/3)*l**3*pi**(5/3)*v*(-v + vt)*(v**(1/3) + (-v + vt)**(1/3))*mu)
    vdd = (v*(-v + vt)*((-3*lam*ld)/(l**3*pi) - (2*lam*ld*vt)/(-2*v**2 + 2*v*vt - v**(5/3)*(-v + vt)**(1/3) + v**(2/3)*vt*(-v + vt)**(1/3) + v**(4/3)*(-v + vt)**(2/3)) + (8*vd**2*vt*(-2*v + vt)*mu)/(v**2*(-v + vt)**2) - (6*6**(1/3)*ld**2*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)/(v**(2/3)*(v**(1/3) + (-v + vt)**(1/3))*(-v + vt + v**(1/3)*(-v + vt)**(2/3)))))/(16*vt*mu)
 
    return array([
        ld, vd, 
        ldd, vdd])

def pmpyjac(statevec,params):
    l,v,ld,vd = statevec
    lam,vt,mu = params
    return array([
        [0, 0, 1, 0], 
        [0, 0, 0, 1], 
        [(vd*(lam*vt + 6*6**(1/3)*ld*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu))/(6*6**(1/3)*l*pi**(2/3)*v*(-v + vt)*(v**(1/3) + (-v + vt)**(1/3))*mu) - (vd*(3*lam*v**(2/3)*(-2*v**(4/3) + 2*v**(1/3)*vt - v*(-v + vt)**(1/3) + vt*(-v + vt)**(1/3) + v**(2/3)*(-v + vt)**(2/3)) + 2*l**3*pi*(lam*vt + 6*6**(1/3)*ld*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)))/(12*6**(1/3)*l**4*pi**(5/3)*v*(-v + vt)*(v**(1/3) + (-v + vt)**(1/3))*mu), (vd*(3*lam*v**(2/3)*((-8*v**(1/3))/3 + (2*vt)/(3*v**(2/3)) + v/(3*(-v + vt)**(2/3)) - vt/(3*(-v + vt)**(2/3)) - (2*v**(2/3))/(3*(-v + vt)**(1/3)) - (-v + vt)**(1/3) + (2*(-v + vt)**(2/3))/(3*v**(1/3))) + (2*lam*(-2*v**(4/3) + 2*v**(1/3)*vt - v*(-v + vt)**(1/3) + vt*(-v + vt)**(1/3) + v**(2/3)*(-v + vt)**(2/3)))/v**(1/3) + 12*6**(1/3)*l**3*ld*pi**(5/3)*((4*v**(1/3))/3 - v/(3*(-v + vt)**(2/3)) + vt/(3*(-v + vt)**(2/3)) + (-v + vt)**(1/3))*mu))/(36*6**(1/3)*l**3*pi**(5/3)*v*(-v + vt)*(v**(1/3) + (-v + vt)**(1/3))*mu) - (vd*(1/(3*v**(2/3)) - 1/(3*(-v + vt)**(2/3)))*(3*lam*v**(2/3)*(-2*v**(4/3) + 2*v**(1/3)*vt - v*(-v + vt)**(1/3) + vt*(-v + vt)**(1/3) + v**(2/3)*(-v + vt)**(2/3)) + 2*l**3*pi*(lam*vt + 6*6**(1/3)*ld*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)))/(36*6**(1/3)*l**3*pi**(5/3)*v*(-v + vt)*(v**(1/3) + (-v + vt)**(1/3))**2*mu) + (vd*(3*lam*v**(2/3)*(-2*v**(4/3) + 2*v**(1/3)*vt - v*(-v + vt)**(1/3) + vt*(-v + vt)**(1/3) + v**(2/3)*(-v + vt)**(2/3)) + 2*l**3*pi*(lam*vt + 6*6**(1/3)*ld*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)))/(36*6**(1/3)*l**3*pi**(5/3)*v*(-v + vt)**2*(v**(1/3) + (-v + vt)**(1/3))*mu) - (vd*(3*lam*v**(2/3)*(-2*v**(4/3) + 2*v**(1/3)*vt - v*(-v + vt)**(1/3) + vt*(-v + vt)**(1/3) + v**(2/3)*(-v + vt)**(2/3)) + 2*l**3*pi*(lam*vt + 6*6**(1/3)*ld*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)))/(36*6**(1/3)*l**3*pi**(5/3)*v**2*(-v + vt)*(v**(1/3) + (-v + vt)**(1/3))*mu), (vd*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3)))/(3*v*(-v + vt)*(v**(1/3) + (-v + vt)**(1/3))), (3*lam*v**(2/3)*(-2*v**(4/3) + 2*v**(1/3)*vt - v*(-v + vt)**(1/3) + vt*(-v + vt)**(1/3) + v**(2/3)*(-v + vt)**(2/3)) + 2*l**3*pi*(lam*vt + 6*6**(1/3)*ld*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu))/(36*6**(1/3)*l**3*pi**(5/3)*v*(-v + vt)*(v**(1/3) + (-v + vt)**(1/3))*mu)], 
        [(9*lam*ld*v*(-v + vt))/(16*l**4*pi*vt*mu), (v*(-v + vt)*((2*lam*ld*vt*(-4*v + 2*vt + v**(5/3)/(3*(-v + vt)**(2/3)) - (v**(2/3)*vt)/(3*(-v + vt)**(2/3)) - (2*v**(4/3))/(3*(-v + vt)**(1/3)) - (5*v**(2/3)*(-v + vt)**(1/3))/3 + (2*vt*(-v + vt)**(1/3))/(3*v**(1/3)) + (4*v**(1/3)*(-v + vt)**(2/3))/3))/(-2*v**2 + 2*v*vt - v**(5/3)*(-v + vt)**(1/3) + v**(2/3)*vt*(-v + vt)**(1/3) + v**(4/3)*(-v + vt)**(2/3))**2 + (16*vd**2*vt*(-2*v + vt)*mu)/(v**2*(-v + vt)**3) - (16*vd**2*vt*mu)/(v**2*(-v + vt)**2) - (16*vd**2*vt*(-2*v + vt)*mu)/(v**3*(-v + vt)**2) + (6*6**(1/3)*ld**2*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*(-1 - (2*v**(1/3))/(3*(-v + vt)**(1/3)) + (-v + vt)**(2/3)/(3*v**(2/3)))*mu)/(v**(2/3)*(v**(1/3) + (-v + vt)**(1/3))*(-v + vt + v**(1/3)*(-v + vt)**(2/3))**2) - (6*6**(1/3)*ld**2*pi**(2/3)*((4*v**(1/3))/3 - v/(3*(-v + vt)**(2/3)) + vt/(3*(-v + vt)**(2/3)) + (-v + vt)**(1/3))*mu)/(v**(2/3)*(v**(1/3) + (-v + vt)**(1/3))*(-v + vt + v**(1/3)*(-v + vt)**(2/3))) + (6*6**(1/3)*ld**2*pi**(2/3)*(1/(3*v**(2/3)) - 1/(3*(-v + vt)**(2/3)))*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)/(v**(2/3)*(v**(1/3) + (-v + vt)**(1/3))**2*(-v + vt + v**(1/3)*(-v + vt)**(2/3))) + (4*6**(1/3)*ld**2*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)/(v**(5/3)*(v**(1/3) + (-v + vt)**(1/3))*(-v + vt + v**(1/3)*(-v + vt)**(2/3)))))/(16*vt*mu) - (v*((-3*lam*ld)/(l**3*pi) - (2*lam*ld*vt)/(-2*v**2 + 2*v*vt - v**(5/3)*(-v + vt)**(1/3) + v**(2/3)*vt*(-v + vt)**(1/3) + v**(4/3)*(-v + vt)**(2/3)) + (8*vd**2*vt*(-2*v + vt)*mu)/(v**2*(-v + vt)**2) - (6*6**(1/3)*ld**2*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)/(v**(2/3)*(v**(1/3) + (-v + vt)**(1/3))*(-v + vt + v**(1/3)*(-v + vt)**(2/3)))))/(16*vt*mu) + ((-v + vt)*((-3*lam*ld)/(l**3*pi) - (2*lam*ld*vt)/(-2*v**2 + 2*v*vt - v**(5/3)*(-v + vt)**(1/3) + v**(2/3)*vt*(-v + vt)**(1/3) + v**(4/3)*(-v + vt)**(2/3)) + (8*vd**2*vt*(-2*v + vt)*mu)/(v**2*(-v + vt)**2) - (6*6**(1/3)*ld**2*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)/(v**(2/3)*(v**(1/3) + (-v + vt)**(1/3))*(-v + vt + v**(1/3)*(-v + vt)**(2/3)))))/(16*vt*mu), (v*(-v + vt)*((-3*lam)/(l**3*pi) - (2*lam*vt)/(-2*v**2 + 2*v*vt - v**(5/3)*(-v + vt)**(1/3) + v**(2/3)*vt*(-v + vt)**(1/3) + v**(4/3)*(-v + vt)**(2/3)) - (12*6**(1/3)*ld*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)/(v**(2/3)*(v**(1/3) + (-v + vt)**(1/3))*(-v + vt + v**(1/3)*(-v + vt)**(2/3)))))/(16*vt*mu), (vd*(-2*v + vt))/(v*(-v + vt))]])







