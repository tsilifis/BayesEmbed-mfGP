"""
Date: 07/05/2020
Author: Panagiotis Tsilifis
"""

__all__ = ['Metropolis', 'GeodesicMC']


import numpy as np
import scipy.stats as st
import scipy.linalg as lng
from _stiefel_sampling import MatrixLangevin


class Metropolis(object):

	_proposal = None

	_like = None

	_chain = None

	def __init__(self, prop, like):
		self._proposal = prop
		self._like = like

	def acceptance(self, W):
		return np.min([1., np.exp(self._like.eval(W) - self._like.eval(self._chain[-1]))])

	def run_chain(self, N=1000):
		self._chain = [self._proposal.sample(1).T]
		i = 1
		while i < N:
			W = self._proposal.sample(1).T
			if st.uniform.rvs() < self.acceptance(W):
				print(self.acceptance(W))
				print(W)
				self._chain += [W]
				i = i + 1
				print('Chain current length : ' + str(len(self._chain)))


class GeodesicMC(object):

	_target = None

	_eps = None

	_T = None

	def __init__(self, target, eps=0.01, T=5):
		"""
		Initializes the object
		"""
		assert isinstance(T, int)
		self._target = target
		self._eps = eps
		self._T = T

	def run_chain(self, M, W0):
		X = W0
		chain = [X.copy()]
		n = 0
		d = W0.shape[0]
		p = W0.shape[1]
		n_rejects = 0
		if p > 1:
			post_grad = self._target.eval_grad_logp(X.T)[:, :p-1].reshape(d, p-1)
			while n < M:
				u = st.norm.rvs(size=(d, p))
				u_proj = u - 0.5 * np.dot(X, np.dot(X.T, u) + np.dot(u.T, X))

				H = self._target.eval_logp(X.T) # - 0.5 * np.linalg.norm(u_proj.flatten()) ** 2
				print('Hamiltonian :' + ' '*10 + str(H))
				x_star = X.copy()
				for i in range(self._T):
					#u_new = u_proj + self._eps * self._target.eval_grad_logp(x_star.T, complement = post_grad, fixcols = True) / 2.
					u_new = u_proj + 0.5 * self._eps * self._target.eval_grad_logp(x_star.T)
					u_new_proj = u_new - np.dot(x_star, np.dot(x_star.T, u_new) + np.dot(u_new.T, x_star)) / 2.
					A = np.dot(x_star.T, u_new_proj)
					S_0 = np.dot(u_new_proj.T, u_new_proj)
					V_0 = np.hstack([x_star, u_new_proj])
					exp1 = np.vstack([np.hstack([A, -S_0]), np.hstack([np.eye(X.shape[1]), A]) ])
					exp2 = np.vstack([np.hstack([lng.expm(-self._eps*A), np.zeros((p,p)) ]), np.hstack([np.zeros((p,p)), lng.expm(-self._eps*A)]) ])
	
					V_eps = np.dot(V_0, np.dot(lng.expm(self._eps * exp1), exp2))
					x_star = V_eps[:,:p].reshape(d,p).copy()
					u_new = V_eps[:,p:].reshape(d,p).copy()
					#u_new = u_new + self._eps * self._target.eval_grad_logp(x_star.T, complement = post_grad, fixcols = True) / 2.
					u_new = u_new + 0.5 * self._eps * self._target.eval_grad_logp(x_star.T)
					u_new_proj = u_new - 0.5 * np.dot(x_star, np.dot(x_star.T, u_new) + np.dot(u_new.T, x_star) )
		

				H_new = self._target.eval_logp(x_star.T) # - 0.5 * np.linalg.norm(u_new_proj.flatten()) ** 2
				print('Proposed Hamiltonian :' + str(H_new))
				if st.uniform.rvs() < np.exp(H_new - H):
					X = x_star.copy()
					chain += [X]
					n = n + 1
					print(n)
				else:
					print('Sample rejected. Total: ' + str(n_rejects))
					n_rejects += 1

				if n_rejects > 20:
					self._eps = self._eps / 1.2
					self._T = self._T + 1
					print('Decreasing time step size. New step size: ' + str(self._eps))
					n_rejects = 0

		elif p == 1:
			#chain = X.copy()
			while n < M:
				u = st.norm.rvs(size=(d, 1))
				u_proj = np.dot((np.eye(d) - np.dot(X, X.T)), u)

				H = self._target.eval_logp(X.T) #- 0.5 * np.linalg.norm(u_proj)
				print('Hamiltonian' +' ' * 10 + ':' + str(H))
				x_star = X.copy()
				for i in range(self._T):
					#print(self._target.eval_grad_logp(x_star.T))
					u_new = u_proj + 0.5 * self._eps * self._target.eval_grad_logp(x_star.T)
					# Projection on the velocity should always be on tangent space 
					# of the current spatial location, not on the initial point. The notation in
					# Byrne & Girolami is wrong !
					u_new_proj = np.dot(np.eye(d) - np.dot(x_star, x_star.T), u_new)
					alpha = np.linalg.norm(u_new_proj)
					V_0 = np.hstack([x_star, u_new_proj])
					a = np.array([1., alpha])
					rot = np.array([[np.cos(alpha*self._eps), - np.sin(alpha*self._eps)], [np.sin(alpha*self._eps), np.cos(alpha*self._eps)]])
					V_eps = np.dot(V_0, np.dot(np.diag(1./a), np.dot(rot, np.diag(a)) ))
					x_star = V_eps[:, 0].reshape(d, 1).copy()
					u_new = V_eps[:, 1].reshape(d, 1).copy()
					u_new = u_new + 0.5 * self._eps * self._target.eval_grad_logp(x_star.T)
					u_new_proj = np.dot(np.eye(d) - np.dot(x_star, x_star.T), u_new)
					u_proj = u_new_proj.copy()

				print('Splitting the Hamiltonian: ')
				print(self._target.eval_logp(x_star.T), 0.5 * np.linalg.norm(u_new_proj))
				print('-' * 50)
				H_new = self._target.eval_logp(x_star.T) #- 0.5 * np.linalg.norm(u_new_proj)
				print('Proposed Hamiltonian :' + str(H_new))
				if st.uniform.rvs() < np.exp(H_new - H):
					X = x_star.copy()
					chain += [X]
					n = n + 1
					print(n)
					# Reset n_rejects
					n_rejects = 0
				else:
					print('Sample rejected. Total: ' + str(n_rejects))
					n_rejects += 1

				if n_rejects > 20:
					self._eps = self._eps / 1.2
					self._T = self._T + 1
					print('Decreasing time step size. New step size: ' + str(self._eps))
					n_rejects = 0
					# Train the GP
					#self._target._likelihood.tune_gp_params(X.T)

		return chain, H_new

