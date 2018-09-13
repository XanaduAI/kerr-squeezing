import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


TN=10   #nonlinear time
TD=500   #dispersion time
#TN=0.9578 #0.5pi for t=1000, zf=8, n=6001
#TN=0.4789 #1.0pi for t=1000, zf=8, n=6001
#TN=0.3193 #1.5pi for t=1000, zf=8, n=6001
#TN=0.1916 #2.5pi for t=1000, zf=8, n=6001
#TN=0.1368 #3.5pi for t=1000, zf=8, n=6001

zf=8   #end points (-zf,+zf) of real-space array
n=101 # number of points in real-space array

#Pulse Shapes
def gaussian(z):
    return np.exp(-z**2/2.)/np.sqrt(np.sqrt(np.pi))

def sech(z):
    return 1./(np.cosh(z)*np.sqrt(2.))

#Fourier Transform Functions
def myfft(z,dz):
    return np.fft.fftshift(np.fft.fft(z)*dz/np.sqrt(2.*np.pi))

def myifft(k,dk):
    return np.fft.ifftshift(np.fft.ifft(k)*dk*n/np.sqrt(2.*np.pi))
    
#Split-Step Fourier Operators For Mean-Field Evolution
def opD(z,kk,dt):
    k=np.fft.fft(z)    
    return np.fft.ifft(np.exp(dt/2.*(1j*kk**2/(2.*TD)))*k)

def opN(z,zi,dt):
    return np.exp(dt*1j/TN*np.abs(zi)**2)*z

#Matrices For Fluctuation Evolution
def S(z,dz):
    return myfft(z**2,dz)/TN

def M(z,dz):
    return myfft(np.abs(z)**2,dz)/TN

def A(z,dz,kk,dk,i):
    Mk=M(z,dz)
    D=np.diag(np.full(len(kk),kk**2/(2.*TD)))
    return D+2.*dk*Mk[i]/np.sqrt(2.*np.pi)

def B(z,dz,dk,i):
    Sk=S(z,dz)
    return dk*Sk[i]/np.sqrt(2.*np.pi)

def Q(z,dz,kk,dk,im,ip):
    a=A(z,dz,kk,dk,im)
    b=B(z,dz,dk,ip)
#    print(np.linalg.norm(a-a.conj().T)) #check properties of A and B
#    print(np.linalg.norm(b-b.T))
    return np.block([[a,b],[-b.conj().T,-a.conj()]])

#Helper For Determining Mean-Field Widths
def FWHM(X,Y):
    half_max = np.max(Y) / 2.
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    return X[right_idx] - X[left_idx]

#Set up z- and k-space arrays
zz=np.linspace(-zf,zf,n)
dz=zz[1]-zz[0]
kk=np.fft.fftfreq(n, d=dz)*(2.*np.pi)
ks=np.fft.fftshift(kk)
dk=kk[1]-kk[0]

#Define mean-field in z-space
U=gaussian(zz)

#Plot in initial mean-field in z- and k-space
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(zz,np.abs(U)**2)
plt.xlim(-1.5*zf,1.5*zf)
ax2.plot(ks,np.abs(myfft(U,dz))**2)

#Determine initial z-space FWHM
#FWHM1=FWHM(zz,abs(U)**2)

#Check initial z- and k-space normalizations
#print(U@U.conj().T*dz)
#a=myfft(U,dz)
#print(a@a.conj().T*dk)

#Set up k-space grid
xx,yy=np.meshgrid(ks,ks)/dk
im=np.rint(xx-yy)+(n-1)/2
ip=np.rint(xx+yy)+(n-1)/2
im=np.clip(im,0,n-1).astype(int)
ip=np.clip(ip,0,n-1).astype(int)

#Initialize K
K=np.identity(2*len(im))

#Check properties of Q
#atemp=np.identity(len(im))
#R=np.block([[atemp,atemp*0],[atemp*0,-atemp]])
#Qtemp=Q(U,dz,ks,dk,im,ip)
#print(np.linalg.norm(Qtemp@R-R@Qtemp.conj().T))

#Perform Evolution
for i in range(100):
  Ui=U
  U=opD(U,kk,dz) #using dz as dt
  U=opN(U,Ui,dz)
  U=opD(U,kk,dz)
  K=expm(1j*dz*Q(U,dz,ks,dk,im,ip))@K
  
X=K[0:n,0:n]
W=K[0:n,n:2*n]
W2=K[n:2*n,0:n]
X2=K[n:2*n,n:2*n]

#Check properties of X and W
print(np.linalg.norm(X@X.conj().T-W@W.conj().T-np.identity(len(im))))
print(np.linalg.norm(X@W.T-W@X.T))

#Plot final mean-field in z- and k-space
ax1.plot(zz,np.abs(U)**2)
ax2.plot(ks,np.abs(myfft(U,dz))**2)

#Plot final expectation value of "number" moment in k-space
ax2.plot(ks,np.real_if_close(np.diag(W.conj()@W.T)))

#Determine final z-space FWHM
#FWHM2=FWHM(zz,np.abs(U)**2)

#Check final z- and k-space normalizations
#print(U@U.conj().T*dz)
#a=myfft(U,dz)
#print(a@a.conj().T*dk)

#Compare ratio of final and initial widths to that expected from TN>>TD theory
#print(FWHM2/FWHM1)
#print(np.sqrt(1.+(hz*1000/TD)**2))
