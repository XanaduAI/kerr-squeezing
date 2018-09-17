import numpy as np
import matplotlib.pyplot as plt
from kerrlib import P_loss, gaussian, myfft

c=299792458     #speed of light [m/s]
L=0.03          #length of waveguide [m]
T0=2.0*10**-12  #pulse width [s]               
v=c/2.1         #group velocity
b2=1.75*10**-25 #group velocity dispersion [s^2/m]
P0=30           #input power [W]
g=1             #nonlinear parameter [/m/W]

Z0=T0*v

TN=1/(g*P0*v)*10**10      #nonlinear time
TD=Z0**2/(b2*v**3)*10**10 #dispersion time

G=0.01                    #loss rate

zf=8   #end points (-zf,+zf) of real-space array
n=101  #number of points in real-space array

#Set up z- and k-space arrays
zz=np.linspace(-zf,zf,n)
dz=zz[1]-zz[0]
kk=np.fft.fftfreq(n, d=dz)*(2.*np.pi)
ks=np.fft.fftshift(kk)
dk=kk[1]-kk[0]
tf=np.rint(L/(v*dz)*10**10).astype(int) #number of points in time (final time=dt*tf)


#Define mean-field in z-space
u=gaussian(zz)

#Plot in initial mean-field in z- and k-space
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(zz,np.abs(u)**2)
plt.xlim(-4*zf,4*zf)
ax2.plot(ks,np.abs(myfft(u,dz))**2)

#Set up k-space grid
xx,yy=np.meshgrid(ks,ks)/dk
im=np.rint(xx-yy)+(n-1)/2
ip=np.rint(xx+yy)+(n-1)/2
im=np.clip(im,0,n-1).astype(int)
ip=np.clip(ip,0,n-1).astype(int)

#Perform Evolution
dt=dz #using dz as dt
#u=P_mean_field(u,TD,TN,0,zz,dz,kk,ks,tf,dt)
u,M,N=P_loss(u,TD,TN,G,dz,kk,ks,dk,im,ip,tf,dt,n)

#Plot final mean-field in z- and k-space
ax1.plot(zz,np.abs(u)**2)
ax2.plot(ks,np.abs(myfft(u,dz))**2)

#Plot final expectation value of "number" moment in k-space
ax2.plot(ks,np.real_if_close(np.diag(N)))
plt.matshow(np.abs(M)**2,origin="lower")

#Plot shot-noise-subtracted quadtrature variance
fig2, ax3 = plt.subplots()
phi=np.linspace(0,np.pi,100)
f=myfft(u,dz)
f/=np.linalg.norm(f)

p1=f.conj()@M@f.conj().T
p2=f.conj()@N@f.T

q1=np.exp(2j*np.linspace(0,np.pi,100))*p1
q2=p2
tp=q1+q1.conj()+2*q2+1
ax3.plot(phi,10*np.log10((q1+q1.conj()+2*q2+1).real))
plt.xlabel(r"$\phi$")
plt.ylabel(r"Squeezing in dB")
