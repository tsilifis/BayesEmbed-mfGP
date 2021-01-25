"""
Date: 29/06/2020
Author: Panagiotis Tsilifis
"""

import numpy as np 
import GPy
import scipy.stats as st 
from scipy.optimize import minimize
from collections import OrderedDict


class RestrictedLikelihood(object):


	_data = None
	_kern = None
	_fidelity = None


	def __init__(self, data, prior_params):

		assert data['X_lf'].shape[0] == data['Z_lf'].shape[0], 'Input and output data set must have the same number of points.'
		assert data['X_hf'].shape[0] == data['Z_hf'].shape[0], 'Input and output data set must have the same number of points.'

		d = data['X_lf'].shape[1]
		self._data = data
		self._prior_params = prior_params

		self._kern = GPy.kern.RBF(input_dim=d, variance=1., ARD=True)


	def eval(self, l, level='low'):
		self._kern.lengthscale[:] = l
		if level == 'low':
			n = self._data['X_lf'].shape[0] + 1
			K = self._kern.K(self._data['X_lf'])
			Q = self.Q1(l)
			alpha = 0.5 * n + self._prior_params['alpha1']
		else:
			n = self._data['X_hf'].shape[0]
			K = self._kern.K(self._data['X_hf'])
			Q = self.Q2(l)
			alpha = 0.5 * n + self._prior_params['alpha2']

		L = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))

		return 2 * np.log(np.diag(L)).sum() + (n - 1) * np.log(Q / (2*alpha))

	def __call__(self, l, level='low'):
		return self.eval(l, level)

	def Q1(self, l):
		self._kern.lengthscale[:] = l
		nl = self._data['X_lf'].shape[0]
		R1 = self._kern.K(self._data['X_lf'])
		L = np.linalg.cholesky(R1 + 1e-6*np.eye(R1.shape[0]))
		LQ = np.linalg.solve(L.T, np.linalg.solve(L, np.ones((nl, 1))))
		Q1_II = np.dot(self._data['Z_lf'].T, np.dot(np.linalg.solve(L.T, np.linalg.solve(L, np.eye(nl))) - np.dot(LQ, LQ.T) / np.dot(np.ones((1, nl)), LQ)[0,0] , self._data['Z_lf']))[0, 0]
		Q1 = self._prior_params['gamma1'] + Q1_II
		return Q1

	def Q2(self, l):
		self._kern.lengthscale[:] = l
		nl = self._data['X_lf'].shape[0]
		nh = self._data['X_hf'].shape[0]

		R2 = self._kern.K(self._data['X_hf'])
		L = np.linalg.cholesky(R2 + 1e-6 * np.eye(R2.shape[0]))
		R2_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(L.shape[0])))
		z1_2 = self._data['Z_lf'][-nh:].reshape(-1, 1)
		G = np.hstack([z1_2, np.ones((nh, 1))])
		L1 = np.linalg.solve(L.T, np.linalg.solve(L, z1_2)) # R_inv * z_2
		LG = np.linalg.solve(L.T, np.linalg.solve(L, G)) # R_inv * G
		GLG = np.dot(G.T, LG)
		GLG_chol = np.linalg.cholesky(GLG + 1e-6*np.eye(GLG.shape[0]))
		lam = np.linalg.solve(GLG_chol.T, np.linalg.solve(GLG_chol, np.dot(G.T, L1)) )
		QQ = R2_inv - np.dot(LG, np.linalg.solve(GLG_chol.T, np.linalg.solve(GLG_chol, LG.T)) )
		Q_II = np.dot(self._data['Z_hf'].T, np.dot(QQ, self._data['Z_hf']))[0, 0]
		V_lam = np.array([[self._prior_params['V_rho'], 0], [0, 0]])
		VV = V_lam + np.linalg.solve(GLG_chol.T, np.linalg.solve(GLG_chol, np.eye(GLG.shape[0])))
		VV_chol = np.linalg.cholesky(VV + 1e-6*np.eye(VV.shape[0]))
		b_lam = np.array([[self._prior_params['b2']], [0]])
		Q2 = self._prior_params['gamma2'] + np.dot((b_lam - lam).T, np.linalg.solve(VV_chol.T, np.linalg.solve(VV_chol, b_lam - lam)))[0, 0] + Q_II
		return Q2



class GP(object):

	_dim = None
	_prior_params = None
	_likelihood = None
	_data = None
	_posterior_params = OrderedDict()

	def __init__(self, data, prior_params):

		self._data = data
		self._prior_params = prior_params
		self._likelihood = RestrictedLikelihood(data, prior_params)
		self._dim = data['X_lf'].shape[1]
		self._posterior_params.update({'alpha1': data['X_lf'].shape[0] / 2 + prior_params['alpha1']})
		self._posterior_params.update({'alpha2': data['X_hf'].shape[0] / 2 + prior_params['alpha2']})


	def maximizeRL(self):

		res_l = minimize(self._likelihood, 0.3 * np.ones(self._dim), args=('low'), method='BFGS' )
		res_h = minimize(self._likelihood, 0.5 * np.ones(self._dim), args=('high'), method='BFGS')		
		"""
		We have to deal with minimization issues later (convergence to a local minimum)
		"""
		return res_l, res_h

	def update_params(self):

		res_l, res_h = self.maximizeRL()
		self._posterior_params.update({ 'theta1': res_l.x})
		self._posterior_params.update({ 'theta2': res_h.x})
		self._posterior_params.update({ 'Q1': self._likelihood.Q1(res_l.x) })
		self._posterior_params.update({ 'Q2': self._likelihood.Q2(res_h.x) })
		self._posterior_params.update({ 'sigma1' : self._posterior_params['Q1'] / (2 * self._posterior_params['alpha1'])})
		self._posterior_params.update({ 'sigma2' : self._posterior_params['Q2'] / (2 * self._posterior_params['alpha2'])})
		rho_mean, rho_var = self.update_rho()
		self._posterior_params.update({ 'rho_mean': rho_mean, 'rho_var': rho_var})
		print(self._posterior_params)


	def update_rho(self):

		nh = self._data['X_hf'].shape[0]
		kern = GPy.kern.RBF(input_dim=self._dim, ARD=True, variance=1.)
		kern.lengthscale[:] = self._posterior_params['theta2']
		R2 = kern.K(self._data['X_hf'])
		L = np.linalg.cholesky(R2 + 1e-6*np.eye(R2.shape[0]))
		z1_2 = self._data['Z_lf'][-nh:].reshape(-1, 1)
		I = np.ones(z1_2.shape)
		L1 = np.linalg.solve(L.T, np.linalg.solve(L, z1_2))
		L2 = np.linalg.solve(L.T, np.linalg.solve(L, self._data['Z_hf']))
		L3 = np.linalg.solve(L.T, np.linalg.solve(L, I))
		zRz = np.dot(z1_2.T, L1)[0, 0]
		iRi = np.dot(I.T, L3)[0, 0]
		z1Rz2 = np.dot(z1_2.T, L2)[0, 0]
		IRz1 = np.dot(I.T, L1)[0, 0]
		IRz2 = np.dot(I.T, L2)[0, 0]
		denom = zRz * iRi - IRz1**2 + iRi / self._prior_params['V_rho'] 
		rho_mean = (z1Rz2 * iRi + iRi * self._prior_params['b2'] / self._prior_params['V_rho'] + IRz1 * IRz2) / denom
		rho_var = iRi / denom
		#A_rho = 1. / ( (np.dot(z1_2.T, L1)[0, 0] - 1./self._prior_params['V_rho']) / self._posterior_params['sigma2'] )
		#nu_rho = (np.dot(z1_2.T, L2)[0, 0] + self._prior_params['b2']/self._prior_params['V_rho']) / self._posterior_params['sigma2']
		#return A_rho, nu_rho
		return rho_mean, rho_var


	def posteriorV(self):

		kern = GPy.kern.RBF(input_dim=self._dim, ARD=True, variance=1)
		kern.lengthscale[:] = self._posterior_params['theta1']
		V11 = self._posterior_params['sigma1'] * kern.K(self._data['X_lf'])
		#V12 = (self._posterior_params['A_rho'] * self._posterior_params['nu_rho']) * self._posterior_params['sigma1'] * kern.K(self._data['X_lf'], self._data['X_hf'])
		V12 = self._posterior_params['rho_mean'] * self._posterior_params['sigma1'] * kern.K(self._data['X_lf'], self._data['X_hf'])
		#V22 = self._posterior_params['sigma1'] * (self._posterior_params['A_rho'] * self._posterior_params['nu_rho'])**2 * kern.K(self._data['X_hf'])
		V22 = self._posterior_params['rho_mean']**2 * self._posterior_params['sigma1'] * kern.K(self._data['X_hf'])
		kern.lengthscale[:] = self._posterior_params['theta2']
		V22 += self._posterior_params['sigma2'] * kern.K(self._data['X_hf'])
		return np.vstack([np.hstack([V11, V12]), np.hstack([V12.T, V22])])

	def predict(self, z):

		z_all = np.vstack([self._data['Z_lf'], self._data['Z_hf']])
		V = self.posteriorV()
		Lv = np.linalg.cholesky(V + 1e-6*np.eye(V.shape[0]))
		zz = np.linalg.solve(Lv.T, np.linalg.solve(Lv, z_all))
		rho_mean = self._posterior_params['rho_mean']
		kern = GPy.kern.RBF(input_dim=self._dim, ARD=True, variance=1)
		kern.lengthscale[:] = self._posterior_params['theta1']
		t1_1 = rho_mean * self._posterior_params['sigma1'] * kern.K(z, self._data['X_lf'])
		t1_2 = rho_mean * self._posterior_params['sigma1'] * kern.K(z, self._data['X_hf'])
		R1_z = kern.K(z)
		kern.lengthscale[:] = self._posterior_params['theta2']
		t2 = rho_mean * t1_2 + self._posterior_params['sigma2'] * kern.K(z, self._data['X_hf'])
		R2_z = kern.K(z)

		tt = np.hstack([t1_1, t2])
		Vt = np.linalg.solve(Lv.T, np.linalg.solve(Lv, tt.T))
		
		m_pred = np.dot(tt, zz)
		S_pred = rho_mean**2 * self._posterior_params['sigma1'] * R1_z + self._posterior_params['sigma2'] * R2_z - np.dot(tt, Vt)

		return m_pred, S_pred
