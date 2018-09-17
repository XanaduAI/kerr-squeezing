import numpy as np
from scipy.linalg import expm

#Pulse Shapes
def gaussian(z):
    return np.exp(-z**2/2.)

def sech(z):
    return 1./(np.cosh(z))

def rect(z):
    return np.where(abs(z)<=np.sqrt(2*np.log(2)), 1, 0)

#Helper For Determining Mean-Field Widths
def FWHM(X,Y):
    half_max = np.max(Y) / 2.
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    return X[right_idx] - X[left_idx]

#Fourier Transform Functions
def myfft(z,dz):
    return np.fft.fftshift(np.fft.fft(z)*dz/np.sqrt(2.*np.pi))

def myifft(k,dk,n):
    return np.fft.ifftshift(np.fft.ifft(k)*dk*n/np.sqrt(2.*np.pi))
    
#Split-Step Fourier Operators For Mean-Field Evolution
def opD(u,TD,G,kk,dt):
    k=np.fft.fft(u)    
    return np.fft.ifft(np.exp(dt/2.*(1j*kk**2/(2.*TD)))*k)*np.exp(dt/2.*(-G/2.))

def opN(u,TN,ui,dt):
    return np.exp(dt*1j/TN*np.abs(ui)**2)*u

#Mean-Field Evolution
def P_mean_field(u,TD,TN,G,zz,dz,kk,ks,tf,dt,Dcheck="False"):
    if Dcheck=="True":
        FWHM1=FWHM(zz,abs(u)**2)
    for i in range(tf):
        ui=u
        u=opD(u,TD,G,kk,dt) 
        u=opN(u,TN,ui,dt)
        u=opD(u,TD,G,kk,dt)
    if Dcheck=="True":
    #Check output field width
        FWHM2=FWHM(zz,abs(u)**2)
        print(FWHM2/FWHM1)
        print(np.sqrt(1.+(dz*tf/TD)**2))
    return u

#Matrices For Fluctuation Evolution
def s(u,TN,dz):
    return myfft(u**2,dz)/TN

def m(u,TN,dz):
    return myfft(np.abs(u)**2,dz)/TN

def A(u,TD,TN,dz,kk,dk,i,n):
    mk=m(u,TN,dz)
    D=np.diag(np.full(n,kk**2/(2.*TD)))
    return D+2.*dk*mk[i]/np.sqrt(2.*np.pi)

def B(u,TN,dz,dk,i):
    sk=s(u,TN,dz)
    return dk*sk[i]/np.sqrt(2.*np.pi)

def Q(u,TD,TN,dz,kk,dk,im,ip,n,check="False"):
    a=A(u,TD,TN,dz,kk,dk,im,n)
    b=B(u,TN,dz,dk,ip)
    if check=="True":
        print(np.linalg.norm(a-a.conj().T)) #check properties of A and B
        print(np.linalg.norm(b-b.T))
    return np.block([[a,b],[-b.conj().T,-a.conj()]])

#Lossless Propagation
def P_no_loss(u,TD,TN,dz,kk,ks,dk,im,ip,tf,dt,n,UWcheck="False",MNcheck="False"):
    M=np.zeros(n)
    N=np.zeros(n)
    K=np.identity(2*n)
    for i in range(tf):
        ui=u
        u=opD(u,TD,0,kk,dt) 
        u=opN(u,TN,ui,dt)
        u=opD(u,TD,0,kk,dt)
        K=expm(1j*dt*Q(u,TD,TN,dz,ks,dk,im,ip,n))@K
    U=K[0:n,0:n]
    W=K[0:n,n:2*n]
    if UWcheck=="True":
    #Check properties of U and W
        print(np.linalg.norm(U@(U.conj().T)-W@(W.conj().T)-np.identity(n)))
        print(np.linalg.norm(U@(W.T)-W@(U.T)))
    M=U@W.T
    N=W.conj()@W.T
    if MNcheck=="True":
    #Check properties of N and M
        l1,v1=np.linalg.eigh(N)
        l1=np.sort(l1)
        v2,l2,w2=np.linalg.svd(M)
        l2=np.sort(l2)
        print(np.linalg.norm(l2*l2-l1*(l1+1)))
    return u,M,N

#Lossy Propagation
def P_loss(u,TD,TN,G,dz,kk,ks,dk,im,ip,tf,dt,n,UWcheck="False",MNcheck="False"):
    M=np.zeros(n)
    N=np.zeros(n)
    for i in range(tf):
        ui=u
        u=opD(u,TD,G,kk,dt) 
        u=opN(u,TN,ui,dt)
        u=opD(u,TD,G,kk,dt)
        K=expm(1j*dt*Q(u,TD,TN,dz,ks,dk,im,ip,n))
        U=K[0:n,0:n]
        W=K[0:n,n:2*n]
        if UWcheck=="True":
        #Check properties of U and W
            print(np.linalg.norm(U@(U.conj().T)-W@(W.conj().T)-np.identity(n)))
            print(np.linalg.norm(U@(W.T)-W@(U.T)))
        M= U@M@(U.T) + W@(M.conj())@(W.T) + W@N@(U.T) + U@(N.T)@(W.T) + U@(W.T)
        N= W.conj()@M@(U.T) + U.conj()@(M.conj())@(W.T) + U.conj()@N@(U.T) + W.conj()@(N.T)@(W.T) + W.conj()@(W.T)
        M=(1-G*dt)*M
        N=(1-G*dt)*N
        if MNcheck=="True":
        #Check properties of N and M
            l1,v1=np.linalg.eigh(N)
            l1=np.sort(l1)
            v2,l2,w2=np.linalg.svd(M)
            l2=np.sort(l2)
            print(np.linalg.norm(l2*l2-l1*(l1+1)))
    return u,M,N

#Nico Propagation
def P_Nico(u,TD,TN,G,dz,kk,ks,dk,im,ip,tf,dt,n,UWcheck="False",MNcheck="False"):
    M=np.zeros(n)
    N=np.zeros(n)
    K=np.identity(2*n)
    for i in range(tf):
        ui=u
        u=opD(u,TD,G,kk,dt) 
        u=opN(u,TN,ui,dt)
        u=opD(u,TD,G,kk,dt)
        K=expm(1j*dt*Q(u,TD,TN,dz,ks,dk,im,ip,n))@K
    U=K[0:n,0:n]
    W=K[0:n,n:2*n]
    if UWcheck=="True":
    #Check properties of U and W
        print(np.linalg.norm(U@(U.conj().T)-W@(W.conj().T)-np.identity(n)))
        print(np.linalg.norm(U@(W.T)-W@(U.T)))
    M=U@W.T
    N=W.conj()@W.T
    M=np.exp(-G*dt*tf)*M
    N=np.exp(-G*dt*tf)*N
    if MNcheck=="True":
    #Check properties of N and M
        l1,v1=np.linalg.eigh(N)
        l1=np.sort(l1)
        v2,l2,w2=np.linalg.svd(M)
        l2=np.sort(l2)
        print(np.linalg.norm(l2*l2-l1*(l1+1)))
    return u,M,N

#Verification Functions
def norm_check(u,dz,dk):
    print(u@u.conj().T*dz)
    a=myfft(u,dz)
    print(a@a.conj().T*dk)
    
def Q_check(u,dz,ks,dk,im,ip,n):
    atemp=np.identity(n)
    R=np.block([[atemp,atemp*0],[atemp*0,-atemp]])
    Qtemp=Q(u,dz,ks,dk,im,ip)
    print(np.linalg.norm(Qtemp@R-R@Qtemp.conj().T))
