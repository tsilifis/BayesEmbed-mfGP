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
from utilities import add_basis_element

#W = np.random.randn((10))

#W = W / np.linalg.norm(W)

np.random.seed(12345)

W = np.array([[ 0.28490595,  0.34201611],# 
				[-0.21608236,  0.19310992],#
                [-0.46249355,  0.36223432],# 
                [-0.15187821, -0.05088952],#
                [-0.16601913,  0.51910281],#
                [ 0.70297967,  0.23900268],#
                [-0.16004973,  0.23084574],#
                [ 0.06096373, -0.48747067],#
                [ 0.23763697,  0.26276529],#
                [-0.16620214, -0.15930184]]).T#


def low_fid(x):
	z = np.dot(W, x.T)
	noise = 0.1 * st.norm.rvs(size=(z.shape[1], 1))
	return np.sin(z[0, :]).reshape(-1, 1) + noise, z.T

def mid_fid(x):
	z = np.dot(W, x.T)
	noise = 0.1 * st.norm.rvs(size=(z.shape[1], 1))
	return low_fid(x)[0] - 7 * np.sin(z[1, :]).reshape(-1, 1)**2 + noise, z.T

def high_fid(x):
	z = np.dot(W, x.T)
	noise = 0.05 * st.norm.rvs(size=(z.shape[1], 1))
	return 1.5 * mid_fid(x)[0] + 5 * z[1,:].reshape(-1, 1)**2 * np.sin(z[0,:]).reshape(-1, 1) + noise, z.T


if __name__ == "__main__":

	n1 = 200
	n2 = 100
	n3 = 25
	D = 10
	
	x_lf = st.uniform.rvs(size=(n1, D))
	x_mf = x_lf[-n2:, :]
	x_hf = x_mf[-n3:, :]

	y_lf, z_lf = low_fid(x_lf)
	y_mf, z_mf = mid_fid(x_mf)
	y_hf, z_hf = high_fid(x_hf)

	data = {'X_0': x_lf, 'X_1': x_mf, 'X_2': x_hf, 'Y_0': y_lf, 'Y_1': y_mf, 'Y_2': y_hf}

	#np.save('results_mf_ishi_'+str(n1)+'l'+str(n2)+'m'+str(n3)+'h/data.npy', data)
	np.save('../data/data_mf_ex2.npy', data)
	Y = np.vstack([data['Y_0'], data['Y_1'], data['Y_2']])
	w_ = VectorLangevin(D, 0.01).sample(1)
	print(w_.shape)
	W0 = add_basis_element(w_)
	print(W0.shape)

	VL = MatrixLangevin(W0, np.array([.1, .1]))
	print('initial projection', W0)

	# Initialize MFGP model, Prior and Posterior objects
	mfgp = MFGPLatent(data, W0.T, levels=3, nested_data=True)
	#VL = MatrixLangevin(np.ones((D, 1)) / np.sqrt(D), np.array([0.1]))
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
		HMC._target._prior._k = np.array([.1, .1])
		np.save('results_mf_ishi_'+str(n1)+'l'+str(n2)+'m'+str(n3)+'h/hamiltonian.npy', hamilt)
		pickle.dump(HMC._target._mfgp._hyp, open('results_mf_ishi_'+str(n1)+'l'+str(n2)+'m'+str(n3)+'h/gp_params_'+str(i)+'.p', 'wb'))
		np.save('results_mf_ishi_'+str(n1)+'l'+str(n2)+'m'+str(n3)+'h/_projW_'+str(i)+'.npy', chain[-1].T)
		HMC._target._mfgp.init_params()

	print(hamilt)
	print(HMC._target._prior._G)

