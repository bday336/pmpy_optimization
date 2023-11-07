from numpy import cos,sin,pi,array

def pmpyfunc(statevec,params):
	l,v,ld,vd = statevec
	lam,vt,mu = params
	return array([ld, vd, (vd*(3*lam*v**(2/3)*(-2*v**(4/3) + 2*v**(1/3)*vt - v*(-v + vt)**(1/3) + vt*(-v + vt)**(1/3) + v**(2/3)*(-v + vt)**(2/3)) + 2*l**3*pi*(lam*vt + 6*6**(1/3)*ld*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)))/(36*6**(1/3)*l**3*pi**(5/3)*v*(-v + vt)*(v**(1/3) + (-v + vt)**(1/3))*mu), (v*(-v + vt)*((-3*lam*ld)/(l**3*pi) - (2*lam*ld*vt)/(-2*v**2 + 2*v*vt - v**(5/3)*(-v + vt)**(1/3) + v**(2/3)*vt*(-v + vt)**(1/3) + v**(4/3)*(-v + vt)**(2/3)) + (8*vd**2*vt*(-2*v + vt)*mu)/(v**2*(-v + vt)**2) - (6*6**(1/3)*ld**2*pi**(2/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))*mu)/(v**(2/3)*(v**(1/3) + (-v + vt)**(1/3))*(-v + vt + v**(1/3)*(-v + vt)**(2/3)))))/(16*vt*mu)])