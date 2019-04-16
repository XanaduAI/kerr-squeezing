import numpy as np
from dispersions import ks, kp1, kp2
import scipy.integrate as integrate

def sinc(x):
    return np.sinc(x/np.pi)

def pmf(wc,wcp,l,wm):
    wp = wc+wcp
    deltaK = kp1(0.5*(wp+wm))+kp2(0.5*(wp-wm))-ks(wc)-ks(wcp)
    return sinc(0.5*l*deltaK)

def gaussian(x,sigma=1.0):
    return np.exp(-0.5*(x/sigma)**2)

def tophat(x, scale=1.0):
        return np.where(abs(x/scale)<=0.5, 1, 0)

def to_int(wc,wcp,l,wm,alpha,beta):
    wp = wc+wcp
    return pmf(wc,wcp,l,wm)*alpha(0.5*(wp+wm))*beta(0.5*(wp-wm))


def jsa(wc1,wcp1, l, alpha, beta):
    toint1 = lambda x: to_int(wc1, wcp1, l, x, alpha, beta)
    return integrate.quad(toint1,-np.inf,np.inf)
