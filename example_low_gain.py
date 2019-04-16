from low_gain import jsa, gaussian
import numpy as np
import sys

scale1 = float(sys.argv[1])
scale2 = float(sys.argv[2])



wc1 = 0.0
wcp1 = 0.0
l = 14000.0
sigma1 = 1.25042*scale1
sigma2 = 1.2608*scale2

alpha  = lambda x: gaussian(x, sigma1)
beta = lambda x: gaussian(x, sigma2)

def jsa1(wc1, wcp1):
    return jsa(wc1, wcp1, l, alpha, beta)
    
wcs = np.arange(-30,30,0.1)

jsa_mat = np.array([[np.array(jsa1(wc1,wcp1)) for wc1 in wcs] for wcp1 in wcs])

np.save("jsa_mat"+sys.argv[1]+"_"+sys.argv[2], jsa_mat)
