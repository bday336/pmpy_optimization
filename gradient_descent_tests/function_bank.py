# Function Bank

import numpy as np

# Set to run Gauss 3-stage method by default
a11,a12,a13 = [5./36., 2./9. - np.sqrt(15.)/15., 5./36. - np.sqrt(15.)/30.]
a21,a22,a23 = [5./36. + np.sqrt(15.)/24., 2./9., 5./36. - np.sqrt(15.)/24.]
a31,a32,a33 = [5./36. + np.sqrt(15.)/30., 2./9. + np.sqrt(15.)/15., 5./36.]

# Metric coefficients and their derivatives
def gllfunc(v, vt, mu):
    return 6*np.pi*mu*(3./(4.*np.pi))**(1./3.) * ( (v*(vt - v))**(1./3.)/(v**(1./3.) + (vt - v)**(1./3.)) )

def gvvfunc(v, vt, mu):
    return 6*np.pi*mu*(2./(9.*np.pi))*( 1./v + 1./(vt - v) )

def dvgllfunc(v, vt, mu):
    return 6*np.pi*mu*(1./v**(4./3.) - 1./(vt-v)**(4./3.))/( 6.**(2./3.)*np.pi**(1./3.) * (1./v**(1./3.) + 1./(vt-v)**(1./3.))**2 )

def dvgvvfunc(v, vt, mu):
    return -6*np.pi*mu*(2.*vt*(vt - 2.*v))/(9*np.pi*(vt-v)**2*v**2)

# Horizontal distribution coefficients and their derivatives
def Afunc(v, vt):
    return .5*(v**(1./3.) - (vt - v)**(1./3.))/(v**(1./3.) + (vt - v)**(1./3.))

def Bfunc(l):
    return 1./(4.*np.pi*l*l)

def dvAfunc(v, vt):
    return (1. - (vt - v)**(1./3.)/v**(1./3.) + (vt - v)**(2./3.)/v**(2./3.))/(3.*(vt - v + v**(1./3.)*(vt - v)**(2./3.)))

def dlBfunc(l):
    return -1./(2.*np.pi*l*l*l)


# Helper functions for the numerical integration
def discretep(ln, ln1, vn, vn1, vt, mu):
    lterm = 6*np.pi*mu*(3./(4.*np.pi))**(1./3.) * ( (vn*(vt - vn))**(1./3.)/(vn**(1./3.) + (vt - vn)**(1./3.)) )*(ln1 - ln)**2
    vterm = 6*np.pi*mu*(2./(9.*np.pi))*( 1./vn + 1./(vt - vn) )*(vn1 - vn)**2
    return (lterm + vterm)

def discreteh(ln, ln1, vn, vn1, vt):
    lterm = .5*(vn**(1./3.) - (vt - vn)**(1./3.))/(vn**(1./3.) + (vt - vn)**(1./3.))*(ln1 - ln)
    vterm = 1./(4.*np.pi*ln*ln)*(vn1 - vn)
    return (lterm + vterm)



## Closed Loop Functions
# Function to return the total energy expenditure of a path
def efunc( path, vt, mu, dt ):
    total = 0
    arr = np.array(path)
    for a in range(arr.shape[0]-2):
        ln,vn = arr[a]
        ln1,vn1 = arr[a+1]
        ln2,vn2 = arr[a+2]
        p1 = discretep(ln, ln1, vn, vn1, vt, mu)
        p2 = discretep(ln1, ln2, vn1, vn2, vt, mu)
        total += (p1 + p2)/(2.*dt)
    
    # Include last interval
    ln,vn = arr[-2]
    ln1,vn1 = arr[-1]
    ln2,vn2 = arr[1]
    p1 = discretep(ln, ln1, vn, vn1, vt, mu)
    p2 = discretep(ln1, ln2, vn1, vn2, vt, mu)
    total += (p1 + p2)/(2.*dt)
    return total

# Function to return the gradient of total energy expenditure of a path
def egradfunc( path, vt, mu, dt ):
    gradlist = []
    arr = np.array(path)

    # Step 1
    lin1,vin1 = arr[-2]
    li,vi = arr[0]
    li1,vi1 = arr[1]
    lterm = (2.*gllfunc(vin1,vt, mu)*(li - lin1) - 2.*gllfunc(vi,vt, mu)*(li1 - li))/dt
    vterm = (2.*gvvfunc(vin1,vt, mu)*(vi - vin1) + dvgllfunc(vi,vt, mu)*(li1 - li)**2 + dvgvvfunc(vi,vt, mu)*(vi1 - vi)**2 - 2.*gvvfunc(vi,vt, mu)*(vi1 - vi))/dt
    gradlist.append([lterm,vterm])

    # Steps 2 - (nump-2)
    for a in range(1,arr.shape[0]-2):
        lin1,vin1 = arr[a-1]
        li,vi = arr[a]
        li1,vi1 = arr[a+1]
        lterm = (2.*gllfunc(vin1,vt, mu)*(li - lin1) - 2.*gllfunc(vi,vt, mu)*(li1 - li))/dt
        vterm = (2.*gvvfunc(vin1,vt, mu)*(vi - vin1) + dvgllfunc(vi,vt, mu)*(li1 - li)**2 + dvgvvfunc(vi,vt, mu)*(vi1 - vi)**2 - 2.*gvvfunc(vi,vt, mu)*(vi1 - vi))/dt
        gradlist.append([lterm,vterm])

    # Step nump-1
    lin1,vin1 = arr[-3]
    li,vi = arr[-2]
    li1,vi1 = arr[0]
    lterm = (2.*gllfunc(vin1,vt, mu)*(li - lin1) - 2.*gllfunc(vi,vt, mu)*(li1 - li))/dt
    vterm = (2.*gvvfunc(vin1,vt, mu)*(vi - vin1) + dvgllfunc(vi,vt, mu)*(li1 - li)**2 + dvgvvfunc(vi,vt, mu)*(vi1 - vi)**2 - 2.*gvvfunc(vi,vt, mu)*(vi1 - vi))/dt
    gradlist.append([lterm,vterm])
    return np.array(gradlist)


# Function to return the total holonomy of a path
def hfunc( path, vt, dt ):
    total = 0
    arr = np.array(path)
    for a in range(arr.shape[0]-2):
        ln,vn = arr[a]
        ln1,vn1 = arr[a+1]
        ln2,vn2 = arr[a+2]
        h1 = discreteh(ln, ln1, vn, vn1, vt)
        h2 = discreteh(ln1, ln2, vn1, vn2, vt)
        total += (h1 + h2)/2
    
    # Include second to last and last contributions
    ln,vn = arr[-2]
    ln1,vn1 = arr[-1]
    ln2,vn2 = arr[0]
    h1 = discreteh(ln, ln1, vn, vn1, vt)
    h2 = discreteh(ln1, ln2, vn1, vn2, vt)
    total += (h1 + h2)/2

    ln,vn = arr[-1]
    ln1,vn1 = arr[0]
    ln2,vn2 = arr[1]
    h1 = discreteh(ln, ln1, vn, vn1, vt)
    h2 = discreteh(ln1, ln2, vn1, vn2, vt)
    total += (h1 + h2)/2

    return total

# Function to return the gradient of holonomy of a path
def hgradfunc( path, vt, dt ):
    gradlist = []
    arr = np.array(path)

    # Step 1
    lin1,vin1 = arr[-2]
    li,vi = arr[0]
    li1,vi1 = arr[1]
    lterm = Afunc(vin1, vt) + dlBfunc(li)*(vi1 - vi) - Afunc(vi, vt)
    vterm = Bfunc(lin1) + dvAfunc(vi, vt)*(li1 - li) - Bfunc(li)
    gradlist.append([lterm,vterm])

    # Steps 2 - (nump-2)
    for a in range(1,arr.shape[0]-2):
        lin1,vin1 = arr[a-1]
        li,vi = arr[a]
        li1,vi1 = arr[a+1]
        lterm = Afunc(vin1, vt) + dlBfunc(li)*(vi1 - vi) - Afunc(vi, vt)
        vterm = Bfunc(lin1) + dvAfunc(vi, vt)*(li1 - li) - Bfunc(li)
        gradlist.append([lterm,vterm])

    # Step nump-1
    lin1,vin1 = arr[-3]
    li,vi = arr[-2]
    li1,vi1 = arr[0]
    lterm = Afunc(vin1, vt) + dlBfunc(li)*(vi1 - vi) - Afunc(vi, vt)
    vterm = Bfunc(lin1) + dvAfunc(vi, vt)*(li1 - li) - Bfunc(li)
    gradlist.append([lterm,vterm])
    return np.array(gradlist)



# Sub-Riemannian Geodesic Functions
# state_vec = [l,v,dl,dv]
def ddl(state_vec,lam,vt):
    l,v,dl,dv = state_vec
    gll = gllfunc(v,vt)
    return -dvgllfunc(v,vt)/gll*dl*dv - lam*(dvAfunc(v,vt) - dlBfunc(l))*dv/gll

def ddv(state_vec,lam,vt):
    l,v,dl,dv = state_vec
    gvv = gvvfunc(v,vt)
    return .5*dvgllfunc(v,vt)/gvv*dl**2 - .5*dvgvvfunc(v,vt)/gvv*dv**2 + lam*(dvAfunc(v,vt) - dlBfunc(l))*dl/gvv

def dynfunc(state_vec,lam,vt):
    return np.array([state_vec[2],state_vec[3],ddl(state_vec,lam,vt),ddv(state_vec,lam,vt)])

def dynjac(state_vec,lam,vt):
    l,v,dl,dv = state_vec
    root_num = -1.650963624447313
    return np.array([
        [0.,0.,1.,0.],
        [0.,0.,0.,1.],
        [
            (dv*lam*(1./v**(1./3.) + 1./(-v + vt)**(1./3.))*root_num)/(l**4.*np.pi**(2./3.)),
            (dv*(-6.*dl*l**3.*np.pi*(-3.*v**3. + 9.*v**2.*vt - 9.*v*vt**2. + 3.*vt**3. + 3.*v**(8./3.)*(-v + vt)**(1./3.) + 6.*v**(7./3.)*(-v + vt)**(2./3.) - 6.*v**(4./3.)*vt*(-v + vt)**(2./3.) + 4.*v**(1./3.)*vt**2.*(-v + vt)**(2./3.)) + 6.**(2./3.)*lam*np.pi**(1./3.)*(6.*v**(8./3.)*vt - 9.*v**(5./3.)*vt**2. + 3.*v**(2./3.)*vt**3. + 9.*v**(10./3.)*(-v + vt)**(1./3.) - 12.*v**(7./3.)*vt*(-v + vt)**(1./3.) + 8.*l**3.*np.pi*v**(1./3.)*vt**2.*(-v + vt)**(1./3.) + 9.*v**3.*(-v + vt)**(2./3.) - 15.*v**2.*vt*(-v + vt)**(2./3.) + 6.*l**3.*np.pi*vt**2.*(-v + vt)**(2./3.) + v**(4./3.)*vt*(-v + vt)**(1./3.)*(-14.*l**3.*np.pi + 3.*vt) + 2.*v*vt*(-v + vt)**(2./3.)*(-7.*l**3.*np.pi + 3.*vt))))/(54.*l**3.*np.pi*v**2.*(v - vt)*(-v + vt + v**(1./3.)*(-v + vt)**(2./3.))**2.),
            -((dv*(1./v**(4./3.) - 1./(-v + vt)**(4./3.)))/(3.*(1./v**(1./3.) + 1./(-v + vt)**(1./3.)))), 
            (lam/(6.**(1./3.)*np.pi**(2./3.)*v**(1./3.)) + lam/(6.**(1./3.)*np.pi**(2./3.)*(-v + vt)**(1./3.)))/l**3. + (-3.*dl*(v - vt)**2. + 3.*dl*v**(4./3.)*(-v + vt)**(2./3.) + 6.**(2./3.)*lam*np.pi**(1./3.)*vt*(-v + vt)**(2./3.))/(9.*v*(v - vt)*(v - vt - v**(1./3.)*(-v + vt)**(2./3.)))
        ],
        [
            (27.*dl*lam*v*(-v + vt))/(4.*l**4.*vt),
            (6.**(1./3.)*dl**2.*l**3.*np.pi**(2./3.)*v**(4./3.)*(v - vt)*(4.*v**3. - 9.*v**2.*vt + 6.*v*vt**2. - vt**3. - 4.*v**(8./3.)*(-v + vt)**(1./3.) + 3.*v**(5./3.)*vt*(-v + vt)**(1./3.) - 8.*v**(7./3.)*(-v + vt)**(2./3.) + 8.*v**(4./3.)*vt*(-v + vt)**(2./3.) + v**(1./3.)*vt**2.*(-v + vt)**(2./3.)) + 2.*dl*lam*v**(4./3.)*(v - vt)*(18.*v**(8./3.)*vt - 27.*v**(5./3.)*vt**2. + 9.*v**(2./3.)*vt**3. + 54.*v**(10./3.)*(-v + vt)**(1./3.) - 81.*v**(7./3.)*vt*(-v + vt)**(1./3.) - 2.*l**3.*np.pi*v**(1./3.)*vt**2.*(-v + vt)**(1./3.) + 27.*v**(4./3.)*vt**2.*(-v + vt)**(1./3.) + 54.*v**3.*(-v + vt)**(2./3.) - 81.*v**2.*vt*(-v + vt)**(2./3.) + 2.*l**3.*np.pi*vt**2.*(-v + vt)**(2./3.) + 27.*v*vt**2.*(-v + vt)**(2./3.)) - 4.*dv**2.*l**3.*vt*(2.*v**2. - 2.*v*vt + vt**2.)*(vt + 3.*v**(1./3.)*(-v + vt)**(1./3.)*(v**(1./3.) + (-v + vt)**(1./3.))))/(8.*l**3.*v**2.*vt*(-v + vt)**(2./3.)*(v**(1./3.) + (-v + vt)**(1./3.))*(-v + vt + v**(1./3.)*(-v + vt)**(2./3.))**2.),
            (dl*(-9.*np.pi*v**2. + 9.*np.pi*v*vt)*(1./v**(4./3.) - 1./(-v + vt)**(4./3.)))/(2.*6.**(2./3.)*np.pi**(1./3.)*vt*(1./v**(1./3.) + 1./(-v + vt)*(1./3.))**2.) - (lam*(-9.*np.pi*v**2. + 9.*np.pi*v*vt)*(1./(2.*l**3.*np.pi) + (1. - (-v + vt)**(1./3.)/v**(1./3.) + (-v + vt)**(2./3.)/v**(2./3.))/(-3.*v + 3.*vt + 3.*v**(1./3.)*(-v + vt)**(2./3.))))/(2.*vt),
            dv*(1./v + 1./(v - vt))
        ]
    ])


## Using the connection given by Avron et al.
def pmpyfunc(state_vec,params):
    l,v,dl,dv = state_vec
    lam,vt = params
    return np.array([dl,dv,
                     (dv*(np.pi/6.)**(1/3)*(v**(1/3) + (-v + vt)**(1/3))*((3.*lam)/(l**3.*np.pi) + (2.*(lam*vt + dl*(6/np.pi)**(1/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3))))/(-2.*v**2. + 2.*v*vt - v^(5/3)*(-v + vt)**(1/3) + v**(2/3)*vt*(-v + vt)**(1/3) + v**(4/3)*(-v + vt)**(2/3))))/(6.*v**(1/3)*(-v + vt)**(1/3)),
                     (v*(-v + vt)*(-((3.*dl*lam)/(l**3.*np.pi)) + (6.*dv**2.*vt*(-2.*v + vt))/(v**2.*(-v + vt)**2) - (dl**2.*(6./np.pi)**(1/3)*(v**(4/3) + v*(-v + vt)**(1/3) - vt*(-v + vt)**(1/3)))/(v**(2/3)*(v**(1/3) + (-v + vt)**(1/3))*(-v + vt + v**(1/3)*(-v + vt)**(2/3))) - (2.*dl*lam*vt)/(-2.*v**2 + 2.*v*vt - v**(5/3)*(-v + vt)**(1/3) + v**(2/3)*vt*(-v + vt)**(1/3) + v**(4/3)*(-v + vt)**(2/3))))/(12.*vt)
                     ])

def pmpyjac(state_vec,params):
    l,v,dl,dv = state_vec
    lam,vt = params
    root_num = -1.650963624447313
    return np.array([
        [0.,0.,1.,0.],
        [0.,0.,0.,1.],
        [
            (dv*lam*(1./v**(1./3.) + 1./(-v + vt)**(1./3.))*root_num)/(l**4.*np.pi**(2./3.)),
            (dv*(-6.*dl*l**3.*np.pi*(-3.*v**3. + 9.*v**2.*vt - 9.*v*vt**2. + 3.*vt**3. + 3.*v**(8./3.)*(-v + vt)**(1./3.) + 6.*v**(7./3.)*(-v + vt)**(2./3.) - 6.*v**(4./3.)*vt*(-v + vt)**(2./3.) + 4.*v**(1./3.)*vt**2.*(-v + vt)**(2./3.)) + 6.**(2./3.)*lam*np.pi**(1./3.)*(6.*v**(8./3.)*vt - 9.*v**(5./3.)*vt**2. + 3.*v**(2./3.)*vt**3. + 9.*v**(10./3.)*(-v + vt)**(1./3.) - 12.*v**(7./3.)*vt*(-v + vt)**(1./3.) + 8.*l**3.*np.pi*v**(1./3.)*vt**2.*(-v + vt)**(1./3.) + 9.*v**3.*(-v + vt)**(2./3.) - 15.*v**2.*vt*(-v + vt)**(2./3.) + 6.*l**3.*np.pi*vt**2.*(-v + vt)**(2./3.) + v**(4./3.)*vt*(-v + vt)**(1./3.)*(-14.*l**3.*np.pi + 3.*vt) + 2.*v*vt*(-v + vt)**(2./3.)*(-7.*l**3.*np.pi + 3.*vt))))/(54.*l**3.*np.pi*v**2.*(v - vt)*(-v + vt + v**(1./3.)*(-v + vt)**(2./3.))**2.),
            -((dv*(1./v**(4./3.) - 1./(-v + vt)**(4./3.)))/(3.*(1./v**(1./3.) + 1./(-v + vt)**(1./3.)))), 
            (lam/(6.**(1./3.)*np.pi**(2./3.)*v**(1./3.)) + lam/(6.**(1./3.)*np.pi**(2./3.)*(-v + vt)**(1./3.)))/l**3. + (-3.*dl*(v - vt)**2. + 3.*dl*v**(4./3.)*(-v + vt)**(2./3.) + 6.**(2./3.)*lam*np.pi**(1./3.)*vt*(-v + vt)**(2./3.))/(9.*v*(v - vt)*(v - vt - v**(1./3.)*(-v + vt)**(2./3.)))
        ],
        [
            (27.*dl*lam*v*(-v + vt))/(4.*l**4.*vt),
            (6.**(1./3.)*dl**2.*l**3.*np.pi**(2./3.)*v**(4./3.)*(v - vt)*(4.*v**3. - 9.*v**2.*vt + 6.*v*vt**2. - vt**3. - 4.*v**(8./3.)*(-v + vt)**(1./3.) + 3.*v**(5./3.)*vt*(-v + vt)**(1./3.) - 8.*v**(7./3.)*(-v + vt)**(2./3.) + 8.*v**(4./3.)*vt*(-v + vt)**(2./3.) + v**(1./3.)*vt**2.*(-v + vt)**(2./3.)) + 2.*dl*lam*v**(4./3.)*(v - vt)*(18.*v**(8./3.)*vt - 27.*v**(5./3.)*vt**2. + 9.*v**(2./3.)*vt**3. + 54.*v**(10./3.)*(-v + vt)**(1./3.) - 81.*v**(7./3.)*vt*(-v + vt)**(1./3.) - 2.*l**3.*np.pi*v**(1./3.)*vt**2.*(-v + vt)**(1./3.) + 27.*v**(4./3.)*vt**2.*(-v + vt)**(1./3.) + 54.*v**3.*(-v + vt)**(2./3.) - 81.*v**2.*vt*(-v + vt)**(2./3.) + 2.*l**3.*np.pi*vt**2.*(-v + vt)**(2./3.) + 27.*v*vt**2.*(-v + vt)**(2./3.)) - 4.*dv**2.*l**3.*vt*(2.*v**2. - 2.*v*vt + vt**2.)*(vt + 3.*v**(1./3.)*(-v + vt)**(1./3.)*(v**(1./3.) + (-v + vt)**(1./3.))))/(8.*l**3.*v**2.*vt*(-v + vt)**(2./3.)*(v**(1./3.) + (-v + vt)**(1./3.))*(-v + vt + v**(1./3.)*(-v + vt)**(2./3.))**2.),
            (dl*(-9.*np.pi*v**2. + 9.*np.pi*v*vt)*(1./v**(4./3.) - 1./(-v + vt)**(4./3.)))/(2.*6.**(2./3.)*np.pi**(1./3.)*vt*(1./v**(1./3.) + 1./(-v + vt)*(1./3.))**2.) - (lam*(-9.*np.pi*v**2. + 9.*np.pi*v*vt)*(1./(2.*l**3.*np.pi) + (1. - (-v + vt)**(1./3.)/v**(1./3.) + (-v + vt)**(2./3.)/v**(2./3.))/(-3.*v + 3.*vt + 3.*v**(1./3.)*(-v + vt)**(2./3.))))/(2.*vt),
            dv*(1./v + 1./(v - vt))
        ]
    ])

