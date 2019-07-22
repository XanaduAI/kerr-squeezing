import numpy as np
import matplotlib.pyplot as plt
from kerrlib import P_loss, P_no_loss, gaussian, rect, sech, lorentzian, myfft, FWHM

c=299792458     #speed of light [m/s]
L=0.03          #length of waveguide [m]
T0=0.17*10**-12 #pulse FWHM [s]               
T0=T0/(2*np.sqrt(1+np.log(np.sqrt(2))))  #pulse 1/e width [s]           
v=c/2.1         #group velocity
b2=1.7*10**-25  #group velocity dispersion [s^2/m]
P0=20           #input power [W]
g=1             #nonlinear parameter [/m/W]

Z0=T0*v

TN=1/(g*P0*v)*10**10      #scaled nonlinear time
TN=8.40
TD=Z0**2/(b2*v**3)*10**10 #scaled dispersion time
TD=5000

G=0.01                    #scaled loss rate

zf=1.5   #end points (-zf,+zf) of real-space array
n=101  #number of points in real-space array

#Set up z- and k-space arrays
zz=np.linspace(-zf,zf,n)
dz=zz[1]-zz[0]
kk=np.fft.fftfreq(n, d=dz)*(2.*np.pi)
ks=np.fft.fftshift(kk)
dk=kk[1]-kk[0]
dt=dz
tf=np.rint(L/(v*dt)*10**10).astype(int) #number of points in time (final time=dt*tf)

#Define mean-field in z-space
u=gaussian(zz)

#Plot in initial mean-field in z- and k-space
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xlim(-zf,zf)
ax1.plot(zz,np.abs(u)**2)
ax2.set_xlim(-zf,zf)
ax2.plot(ks,np.abs(myfft(u,dz))**2)

#Set up k-space grid
xx,yy=np.meshgrid(ks,ks)/dk
im=np.rint(xx-yy)+(n-1)/2
ip=np.rint(xx+yy)+(n-1)/2
im=np.clip(im,0,n-1).astype(int)
ip=np.clip(ip,0,n-1).astype(int)

#Perform Evolution
#u=P_mean_field(u,TD,TN,G,zz,dz,kk,ks,tf,dt)
u,M,N=P_no_loss(u,TD,TN,dz,kk,ks,dk,im,ip,tf,dt,n)
#u,M,N=P_loss(u,TD,TN,G,dz,kk,ks,dk,im,ip,tf,dt,n)

#Plot final mean-field in z- and k-space
ax1.plot(zz,np.abs(u)**2)
ax2.plot(ks,np.abs(myfft(u,dz))**2)

#Plot final expectation value of "number" moment in k-space
ax2.plot(ks,np.real_if_close(np.diag(N)))
plt.matshow(np.abs(M)**2,origin="lower")

#Plot quadtrature variance
fig2, ax3 = plt.subplots()
phi=np.linspace(0,np.pi,100)
f=myfft(u,dz)
f/=np.linalg.norm(f)

p1=f.conj()@M@f.conj().T
p2=f@N@f.conj().T

q1=np.exp(2j*np.linspace(0,np.pi,100))*p1
q2=p2
tp=q1+q1.conj()+2*q2+1
ax3.plot(phi,10*np.log10((q1+q1.conj()+2*q2+1).real))
plt.xlabel(r"$\phi$")
plt.ylabel(r"Squeezing in dB")
plt.show()
