"""
Date: 06/05/2020
Author: Panagiotis Tsilifis
"""

import numpy as np
import GPy


class Likelihood(object):
    """
    Class representing the likelihood for GP regression with built-in
    dimensionality reduction
    """

    _data = None

    _K = None

    _inp_dim = None

    _gp_params = None

    def __init__(self, data, gp_params):
        """
        :param data: dictionary (or dataframe ?) that contains the training data for the GP model.
        :param gp_params: Parameters (and hyperparameters) of the GP model.
        """

        assert data['Y'].shape[0] == data['X'].shape[0]
        self._data = data
        self._K = data['Y'].shape[0]
        self._inp_dim = data['X'].shape[1]
        self._gp_params = gp_params

    def tune_gp_params(self, w, n_samples=500, burn_in=100):
        z = np.dot(w, self._data['X'].T).T
        ker = GPy.kern.RBF(input_dim=w.shape[0], ARD=True)
        gp = GPy.models.GPRegression(z, self._data['Y'], ker)
        gp.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1., 50.))
        gp.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1., 50.))
        gp.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1., 50.))
        hmc = GPy.inference.mcmc.HMC(gp, stepsize=5e-2)
        # s = hmc.sample(num_samples=burn_in)
        s = hmc.sample(num_samples=n_samples)[burn_in:]
        self._gp_params = {'var': s[:, 0].mean(), 'lengthscale': s[:, 1].mean(), 'var_like': s[:, 2].mean()}

    def eval(self, w):
        assert w.shape[1] == self._inp_dim
        z = np.dot(w, self._data['X'].T).T
        ker = GPy.kern.RBF(input_dim=w.shape[0], ARD=True)
        gp_model = GPy.models.GPRegression(z, self._data['Y'], ker)
        gp_model.kern.variance[:] = self._gp_params['var']
        gp_model.kern.lengthscale[:] = self._gp_params['lengthscale']
        gp_model.likelihood.variance[:] = self._gp_params['var_like']
        cov = gp_model.kern.K(z, z)

        L_s = np.linalg.cholesky(cov + self._gp_params['var_like'] * np.eye(z.shape[0]))
        alpha_s = np.linalg.solve(L_s.T, np.linalg.solve(L_s, self._data['Y']))
        log_L_marginal = - 0.5 * np.dot(self._data['Y'].T, alpha_s) - np.log(np.diag(L_s)).sum() + 0.5 * z.shape[0] * np.log(2 * np.pi)
        return log_L_marginal[0]

    def eval_grad_W(self, w):
        assert w.shape[1] == self._inp_dim
        z = np.dot(w, self._data['X'].T).T
        ker = GPy.kern.RBF(input_dim=w.shape[0], ARD=True)
        gp_model = GPy.models.GPRegression(z, self._data['Y'], ker)
        gp_model.kern.variance[:] = self._gp_params['var']
        gp_model.kern.lengthscale[:] = self._gp_params['lengthscale']
        gp_model.likelihood.variance[:] = self._gp_params['var_like']

        L_s = np.linalg.cholesky(gp_model.kern.K(z, z) + self._gp_params['var_like'] * np.eye(z.shape[0]))
        alpha_s = np.linalg.solve(L_s.T, np.linalg.solve(L_s, self._data['Y']))
        inv_K_s = np.linalg.solve(L_s.T, np.linalg.solve(L_s, np.eye(L_s.shape[0])))

        dL_dW = np.zeros(w.T.shape)
        for i in range(self._inp_dim):
            for j in range(z.shape[1]):
                dK_dw = self._data['X'][:, i].reshape(-1, 1) * gp_model.kern.dK_dX(z, z, j) + self._data['X'][:, i] * gp_model.kern.dK_dX2(z, z, j)
                dL_dW[i, j] = 0.5 * np.diag(np.dot(np.dot(alpha_s, alpha_s.T) - inv_K_s, dK_dw)).sum()

        return dL_dW.T


