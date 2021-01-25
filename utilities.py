from datetime import datetime 
import numpy as np 


def add_basis_element(W):
	"""
	Given a D x d orthonormal matrix W (with d << D), it computes a new vector v that 
	is orthogonal to all d columns of W and add it as an additional column.
	Return : D x (d+1) orthonormal matrix [W v]
	"""

	dim = W.shape[1]
	d = W.shape[0]
	v = np.random.randn(d)
	v = v / np.linalg.norm(v)
	u = np.zeros(v.shape)
	for i in range(dim):
		u = u - np.sum(W[:, i] * v) * W[:, i]
	v = (v - u).reshape(-1, 1)
	v = v / np.linalg.norm(v)
	return np.hstack([W, v])


def compact_timestamp():
	return '{:%Y%m%d_%H%M%S}'.format(datetime.now())

