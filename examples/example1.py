import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pickle

import sys
sys.path.insert(0, '../')

from gp_latent import MFGPLatent
from _posterior import MFPosterior
from _mcmc import GeodesicMC
from _stiefel_sampling import *
from pyDOE import lhs

np.random.seed(1234)

#W = np.random.randn((10))

#W = W / np.linalg.norm(W)

W = np.array([ 0.14042183, -0.35474441, 0.42674656, -0.0931266, -0.21463479, 0.26425064,
     0.25603728, -0.1895951, 0.00467533, -0.66800687])[np.newaxis, :]

print(W)

#W = np.array([0.14])

def low_fid(x):
     z = np.dot(W, x.T)
     noise = 0.5 * st.norm.rvs(size=z.shape)
     return (0.5 * (8 * z - 2) * np.sin(5 * z - 4) - 10*(z-0.5)).T + noise.T, z.T

def mid_fid(x):
     z = np.dot(W, x.T)
     noise = 3 * st.norm.rvs(size=z.shape)
     return 2 * low_fid(x)[0] - 20 * z.T + 20 + noise.T, z.T

def high_fid(x):
     z = np.dot(W, x.T)
     noise = 5 * st.norm.rvs(size=z.shape)
     return 1.5 * mid_fid(x)[0] + 30 * (z**2).T + noise.T, z.T


if __name__ == "__main__":
     
     n1 = 300
     n2 = 200
     n3 = 10
     D = 10

     x_lf = 4 * lhs(n1, D).T - 2
     x_mf = x_lf[-n2:, :]
     x_hf = x_mf[-n3:, :]

 
     y_lf, z_lf = low_fid(x_lf)
     y_mf, z_mf = mid_fid(x_mf)
     y_hf, z_hf = high_fid(x_hf)

     data = {'X_0': x_lf, 'X_1': x_mf, 'X_2': x_hf, 'Y_0': y_lf, 'Y_1': y_mf, 'Y_2': y_hf}
     #np.save('../data/data_mf_ex1_'+str(n1)+'l'+str(n2)+'m'+str(n3)+'h/data.npy', data)
     np.save('../data/data_mf_ex1.npy', data)     

     Y = np.vstack([data['Y_0'], data['Y_1'], data['Y_2']])

     W0 = VectorLangevin(D, 0.01).sample(1)
     print('initial projection', W0)

     # Initialize MFGP model, Prior and Posterior objects
     mfgp = MFGPLatent(data, W0.T, levels=3, nested_data=True)
     VL = MatrixLangevin(np.ones((D, 1)) / np.sqrt(D), np.array([0.1]))
     posterior = MFPosterior(mfgp, VL)

     eps = 0.01  # W1 works with eps = 0.01
     T = 10
     HMC = GeodesicMC(posterior, eps=eps, T=T)
     
     hamilt = [-np.inf]
     incr = 1e+5
     #for i in range(5):

     i = 0
     while incr > 0.001:
          HMC._target._mfgp.train()

          chain, hamiltonian = HMC.run_chain(1, HMC._target._mfgp._projW.T)
          hamilt.append(hamiltonian)

          #print(chain[-1].shape)

          incr = np.abs(hamilt[-1] - hamilt[-2])
          print(incr)

          HMC._target._mfgp._projW = chain[-1].T
          HMC._target._mfgp.update_latent_data()

          print('Iteration # ' + str(i))

          i = i + 1

          HMC._target._prior._G = HMC._target._mfgp._projW.T
          HMC._target._prior._k = np.array([.1])
          np.save('results_mf_ex1_'+str(n1)+'l'+str(n2)+'m'+str(n3)+'h/hamiltonian.npy', hamilt)
          pickle.dump(HMC._target._mfgp._hyp, open('results_mf_ex1_'+str(n1)+'l'+str(n2)+'m'+str(n3)+'h/gp_params_'+str(i)+'.p', 'wb'))
          np.save('results_mf_ex1_'+str(n1)+'l'+str(n2)+'m'+str(n3)+'h/_projW_'+str(i)+'.npy', chain[-1].T)
          HMC._target._mfgp.init_params()
          
     print(hamilt)
     print(HMC._target._prior._G)


