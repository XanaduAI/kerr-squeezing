import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


TN=2.33   #nonlinear time
TD=1600  #dispersion time
G=0.01 #loss rate

zf=8   #end points (-zf,+zf) of real-space array
n=101  #number of points in real-space array
tf=13 #number of points in time (final time=dt*tf)

#Pulse Shapes
def gaussian(z):
    return np.exp(-z**2/2.)/np.sqrt(np.sqrt(np.pi))

def sech(z):
    return 1./(np.cosh(z)*np.sqrt(2.))

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

def myifft(k,dk):
    return np.fft.ifftshift(np.fft.ifft(k)*dk*n/np.sqrt(2.*np.pi))
    
#Split-Step Fourier Operators For Mean-Field Evolution
def opD(u,kk,dt):
    k=np.fft.fft(u)    
    return np.exp(dt/2.*(-G/2.))*np.fft.ifft(np.exp(dt/2.*(1j*kk**2/(2.*TD)))*k)

def opN(u,ui,dt):
    return np.exp(dt*1j/TN*np.abs(ui)**2)*u

#Matrices For Fluctuation Evolution
def s(u,dz):
    return myfft(u**2,dz)/TN

def m(u,dz):
    return myfft(np.abs(u)**2,dz)/TN

def A(u,dz,kk,dk,i):
    mk=m(u,dz)
    D=np.diag(np.full(n,kk**2/(2.*TD)))
    return D+2.*dk*mk[i]/np.sqrt(2.*np.pi)

def B(u,dz,dk,i):
    sk=s(u,dz)
    return dk*sk[i]/np.sqrt(2.*np.pi)

def Q(u,dz,kk,dk,im,ip,check="False"):
    a=A(u,dz,kk,dk,im)
    b=B(u,dz,dk,ip)
    if check=="True":
        print(np.linalg.norm(a-a.conj().T)) #check properties of A and B
        print(np.linalg.norm(b-b.T))
    return np.block([[a,b],[-b.conj().T,-a.conj()]])

#Lossless Propagation
def P_no_loss(u,dz,kk,ks,dk,im,ip,dt,UWcheck="False",MNcheck="False"):
    M=np.zeros(n)
    N=np.zeros(n)
    K=np.identity(2*n)
    for i in range(tf):
        ui=u
        u=opD(u,kk,dt) 
        u=opN(u,ui,dt)
        u=opD(u,kk,dt)
        K=expm(1j*dt*Q(u,dz,ks,dk,im,ip))@K
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
def P_loss(u,dz,kk,ks,dk,im,ip,dt,UWcheck="False",MNcheck="False"):
    M=np.zeros(n)
    N=np.zeros(n)
    for i in range(tf):
        ui=u
        u=opD(u,kk,dt) 
        u=opN(u,ui,dt)
        u=opD(u,kk,dt)
        K=expm(1j*dt*Q(u,dz,ks,dk,im,ip))
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
def P_Nico(u,dz,kk,ks,dk,im,ip,dt,UWcheck="False",MNcheck="False"):
    M=np.zeros(n)
    N=np.zeros(n)
    K=np.identity(2*n)
    for i in range(tf):
        ui=u
        u=opD(u,kk,dt) 
        u=opN(u,ui,dt)
        u=opD(u,kk,dt)
        K=expm(1j*dt*Q(u,dz,ks,dk,im,ip))@K
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
    
def Q_check(u,dz,ks,dk,im,ip):
    atemp=np.identity(n)
    R=np.block([[atemp,atemp*0],[atemp*0,-atemp]])
    Qtemp=Q(u,dz,ks,dk,im,ip)
    print(np.linalg.norm(Qtemp@R-R@Qtemp.conj().T))

#Set up z- and k-space arrays
zz=np.linspace(-zf,zf,n)
dz=zz[1]-zz[0]
kk=np.fft.fftfreq(n, d=dz)*(2.*np.pi)
ks=np.fft.fftshift(kk)
dk=kk[1]-kk[0]

#Define mean-field in z-space
u=gaussian(zz)

#Plot in initial mean-field in z- and k-space
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(zz,np.abs(u)**2)
plt.xlim(-1.5*zf,1.5*zf)
ax2.plot(ks,np.abs(myfft(u,dz))**2)

#Determine initial z-space FWHM
#FWHM1=FWHM(zz,abs(U)**2)

#Set up k-space grid
xx,yy=np.meshgrid(ks,ks)/dk
im=np.rint(xx-yy)+(n-1)/2
ip=np.rint(xx+yy)+(n-1)/2
im=np.clip(im,0,n-1).astype(int)
ip=np.clip(ip,0,n-1).astype(int)

#Perform Evolution
dt=dz #using dz as dt
u,M,N=P_loss(u,dz,kk,ks,dk,im,ip,dt)

#Plot final mean-field in z- and k-space
ax1.plot(zz,np.abs(u)**2)
ax2.plot(ks,np.abs(myfft(u,dz))**2)

#Plot final expectation value of "number" moment in k-space
ax2.plot(ks,np.real_if_close(np.diag(N)))
plt.matshow(np.abs(M)**2,origin="lower")

#Plot shot-noise-subtracted quadtrature variance
fig2, ax3 = plt.subplots()
pp=np.linspace(0,2*np.pi,n)
f=myfft(u,dz)
fnorm=f@f.conj().T*dk

p1=f.conj()@M@f.conj().T
p2=f.conj()@N@f.T
p3=np.exp(1j*2.*pp)

ax3.plot(pp,(dk**4*(2.*np.real(p3*p1)+2.*p2)/fnorm)+1.)

#Determine final z-space FWHM
#FWHM2=FWHM(zz,np.abs(U)**2)

#Compare ratio of final and initial widths to that expected from TN>>TD theory
#print(FWHM2/FWHM1)
#print(np.sqrt(1.+(hz*tf/TD)**2))
