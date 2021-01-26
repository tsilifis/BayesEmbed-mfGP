"""
Date: 05/05/2020
Author: Panagiotis Tsilifis
"""

import GPy
import pandas as pd
from scipy.optimize import minimize
import autograd.numpy as np
import numpy as np
from autograd import value_and_grad
from autograd import grad

from bayes import *


__all__ = ['GPLatent', 'MFGPLatent']


class GPLatent(object):
    """
    Class for GP with subspace embedding
    """

    _training_data = None
    _gp_model = None
    _projW = None

    def __init__(self, X, Y, W):

        assert X.ndim == 2  # Make sure X is two dimensional
        assert W.ndim == 2  # Make sure W is two dimensional as well
        assert X.shape[1] == W.shape[1]  # Make sure input data dimensionality is equal to number of projection columns
        assert X.shape[0] == Y.shape[0]  # Make sure number of input and output points is the same (hey you never know)
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        Z = (W @ X.T).T

        # Initialize gp_model variable and set priors
        kernel = GPy.kern.RBF(input_dim=W.shape[0], ARD=True)
        self._gp_model = GPy.models.GPRegression(Z, Y, kernel)
        self._gp_model.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1., 50.))
        self._gp_model.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1., 50.))
        self._gp_model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1., 50.))

        # Initialize projection matrix
        self._projW = W

        # Initialize training data
        self._training_data = {"X": X, "Y": Y}

    def update_GP(self):
        Z_new = (self._projW @ self._training_data["X"].T).T
        ker = GPy.kern.RBF(input_dim=W.shape[0], ARD=True)
        self._gp_model = GPy.models.GPRegression(Z_new, self._training_data["Y"], ker)
        self._gp_model.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1., 50.))
        self._gp_model.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1., 50.))
        self._gp_model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1., 50.))

    def train_gp(self, n_samples=1500, burn_in=500):
        hmc = GPy.inference.mcmc.HMC(self._gp_model, stepsize=5e-2)
        #s = hmc.sample(num_samples=burn_in)
        s = hmc.sample(num_samples=n_samples)[burn_in:]
        self.param_samples = s
        self.param_statistics = {'var': s[:, 0].mean(), 'lengthscale': s[:, 1].mean(), 'var_like': s[:, 2].mean()}

    def predict(self, x):
        # Set GP parameter values
        self._gp_model.kern.variance[:] = self.param_statistics['var']
        self._gp_model.kern.lengthscale[:] = self.param_statistics['lengthscale']
        self._gp_model.likelihood.variance[:] = self.param_statistics['var_like']
        z = (self._projW @ x.T).T
        pred_m, pred_std = self._gp_model.predict(z)
        return (pred_m, pred_std)

    def predict_direct(self, z):
        # Set GP parameter values
        self._gp_model.kern.variance[:] = self.param_statistics['var']
        self._gp_model.kern.lengthscale[:] = self.param_statistics['lengthscale']
        self._gp_model.likelihood.variance[:] = self.param_statistics['var_like']
        pred_m, pred_std = self._gp_model.predict(z)
        return (pred_m, pred_std)


class MFGPLatent(object):
    """
    Class for multi-fidelity GP with subspace embedding
    """

    _input_dim = None
    _latent_dim = None
    _training_data = None
    _mfgp_model = None
    _projW = None
    _nlevels = None
    _idx_theta = None
    _hyp = None
    _jitter = 1e-4


    def __init__(self, data, W, levels=2, nested_data=False):
        Z = [(W @ data['X_'+str(i)].T).T for i in range(levels)]
        data.update({'Z_'+str(i): Z[i] for i in range(levels)})

        self._projW = W
        self._training_data = data
        self._input_dim = data['X_0'].shape[1]
        self._latent_dim = data['Z_0'].shape[1]
        self._nlevels = levels
        self.init_params()
        self._nested = nested_data

    def init_params(self):
        hyp = np.log(np.ones(self._latent_dim + 1))
        self._idx_theta = [np.arange(hyp.shape[0])]

        for i in range(1, self._nlevels):
            hyp = np.concatenate([hyp, np.log(np.ones(self._latent_dim+1))])
            self._idx_theta += [np.arange(self._idx_theta[i-1][-1]+1, hyp.shape[0])]

        rho = np.array([1.0]*(self._nlevels-1))
        logsigma_n = np.array([-4.0]*self._nlevels)
        hyp = np.concatenate([hyp, rho, logsigma_n])
        self._hyp = hyp


    def update_latent_data(self):
        Z_new = [(self._projW @ self._training_data['X_'+str(i)].T).T for i in range(self._nlevels)]
        self._training_data.update({'Z_'+str(i): Z_new[i] for i in range(self._nlevels)})


    def kernel(self, x, xp, hyp):
        output_scale = np.exp(hyp[0])
        lengthscales = np.exp(hyp[1:])
        diffs = np.expand_dims(x / lengthscales, 1) - np.expand_dims(xp / lengthscales, 0)
        return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))


    def log_likelihood(self, hyp):
        if self._nlevels==3 and self._nested==True:
            nl = [self._training_data['X_'+str(i)].shape[0] for i in range(self._nlevels)]
            rhos = hyp[-2*self._nlevels+1:-self._nlevels]
            sigmas = hyp[-self._nlevels:]

            theta00 = hyp[self._idx_theta[0]]
            theta11 = hyp[self._idx_theta[1]]
            theta22 = hyp[self._idx_theta[2]]

            L1 = 0.5 * ( self.loglike1(np.hstack([np.array([sigmas[0]]), theta00[1:]])) + 1/(nl[0]-1) )
            L2 = 0.5 * ( self.loglike2(np.hstack([np.array([sigmas[1]]), theta11[1:]])) + 1/(nl[1]-2) )
            L3 = 0.5 * ( self.loglike3(np.hstack([np.array([sigmas[2]]), theta22[1:]])) + 1/(nl[2]-2) )
            return L1 * L2 * L3
        else:
            nl = [self._training_data['X_'+str(i)].shape[0] for i in range(self._nlevels)]
            n = np.sum(nl)
            y = np.vstack([self._training_data['Y_'+str(i)] for i in range(self._nlevels)])

            rhos = hyp[-2 * self._nlevels+1:-self._nlevels]
            sigmas = np.exp(hyp[-self._nlevels:])

            theta00 = hyp[self._idx_theta[0]]
            theta11 = hyp[self._idx_theta[1]]

            K_00 = self.kernel(self._training_data['Z_0'], self._training_data['Z_0'], theta00) + np.eye(nl[0]) * sigmas[0]
            K_01 = rhos[0] * self.kernel(self._training_data['Z_0'], self._training_data['Z_1'], theta00)
            K_11 = rhos[0]**2 * self.kernel(self._training_data['Z_1'], self._training_data['Z_1'], theta00)
            K_11 = K_11 + self.kernel(self._training_data['Z_1'], self._training_data['Z_1'], theta11) + np.eye(nl[1]) * sigmas[1]
            K_full = np.vstack([np.hstack([K_00, K_01]), np.hstack([K_01.T, K_11])])
            if self._nlevels==3:
                theta22 = hyp[self._idx_theta[2]]
                K_02 = rhos[0] * self.kernel(self._training_data['Z_0'], self._training_data['Z_2'], theta00)
                K_12 = rhos[0] * rhos[1] * self.kernel(self._training_data['Z_1'], self._training_data['Z_2'], theta00)
                K_22 = (rhos[0] * rhos[1])**2 * self.kernel(self._training_data['Z_2'], self._training_data['Z_2'], theta00)
                K_12 = K_12 + rhos[1] * self.kernel(self._training_data['Z_1'], self._training_data['Z_2'], theta11)
                K_22 = K_22 + rhos[1]**2 * self.kernel(self._training_data['Z_2'], self._training_data['Z_2'], theta11)
                K_22 = K_22 + self.kernel(self._training_data['Z_2'], self._training_data['Z_2'], theta22) + np.eye(nl[2]) * sigmas[2]

                K_full = np.vstack([ np.hstack([ K_full, np.vstack([K_02, K_12])]), np.hstack([K_02.T, K_12.T, K_22]) ])

            L = np.linalg.cholesky(K_full + np.eye(n) * self._jitter)
            self._L = L

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            NLML = 0.5 * np.matmul(y.T, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * np.log(2 * np.pi) * n
            return NLML[0, 0]


    def loglike1(self, hyp1):
        n1 = self._training_data['X_0'].shape[0]
        par = np.hstack([0, hyp1[1:]._value])
        sig_e = np.exp(hyp1[0]._value)
        K_00 = self.kernel(self._training_data['Z_0'], self._training_data['Z_0'], par) + np.eye(n1) * sig_e
        L_00 = np.linalg.cholesky(K_00 + np.eye(n1) * self._jitter)
        alpha = np.linalg.solve(L_00.T, np.linalg.solve(L_00, self._training_data['Y_0']))
        NLL1 = (n1 - 1) * np.log(np.matmul(self._training_data['Y_0'].T, alpha) / (n1 - 1)) + 2 * np.sum(np.log(np.diag(L_00)))
        return NLL1[0, 0]

    def loglike2(self, hyp2):
        n2 = self._training_data['X_1'].shape[0]
        par = np.hstack([0, hyp2[1:]])
        sig_e = np.exp(hyp2[0]._value)
        K_11 = self.kernel(self._training_data['Z_1'], self._training_data['Z_1'], par) + np.eye(n2) * sig_e
        L_11 = np.linalg.cholesky(K_11 + np.eye(n2) * self._jitter)
        H2 = np.hstack([np.ones((n2, 1)), self._training_data['Y_0'][-n2:,0].reshape(-1, 1)])
        BIG = np.matmul(H2.T, np.linalg.solve(L_11.T, np.linalg.solve(L_11, H2)) )
        L_BIG = np.linalg.cholesky(BIG + np.eye(2) * self._jitter)
        rho_hat = np.linalg.solve(L_BIG.T, np.linalg.solve(L_BIG, np.matmul(H2.T, np.linalg.solve(L_11.T, np.linalg.solve(L_11, self._training_data['Y_1']))) ))[1, 0]
        alpha = np.linalg.solve(L_11.T, np.linalg.solve(L_11, self._training_data['Y_1'] - rho_hat * self._training_data['Y_0'][-n2:].reshape(-1, 1)))
        NLL2 = (n2 - 2) * np.log(np.matmul( (self._training_data['Y_1'] - rho_hat * self._training_data['Y_0'][-n2:].reshape(-1, 1)).T, alpha) / (n2-2)) + 2 * np.sum(np.log(np.diag(L_11)))
        return NLL2[0, 0]


    def loglike3(self, hyp3):
        n3 = self._training_data['X_2'].shape[0]
        par = np.hstack([0, hyp3[1:]])
        sig_e = np.exp(hyp3[0]._value)
        K_22 = self.kernel(self._training_data['Z_2'], self._training_data['Z_2'], par) + np.eye(n3) * sig_e
        L_22 = np.linalg.cholesky(K_22 + np.eye(n3) * self._jitter)
        H3 = np.hstack([np.ones((n3, 1)), self._training_data['Y_1'][-n3:,0].reshape(-1, 1)])
        BIG = np.matmul(H3.T, np.linalg.solve(L_22.T, np.linalg.solve(L_22, H3)) )
        L_BIG = np.linalg.cholesky(BIG + np.eye(2) * self._jitter)
        rho_hat = np.linalg.solve(L_BIG.T, np.linalg.solve(L_BIG, np.matmul(H3.T, np.linalg.solve(L_22.T, np.linalg.solve(L_22, self._training_data['Y_2']))) ))[1, 0]
        alpha = np.linalg.solve(L_22.T, np.linalg.solve(L_22, self._training_data['Y_2'] - rho_hat * self._training_data['Y_1'][-n3:].reshape(-1, 1)))
        NLL3 = (n3 - 2) * np.log(np.matmul( (self._training_data['Y_2'] - rho_hat * self._training_data['Y_1'][-n3:].reshape(-1, 1)).T, alpha) / (n3-2)) + 2 * np.sum(np.log(np.diag(L_22)))
        return NLL3[0, 0]


    def loglikelihood_W(self, W):
        self._projW = W
        self.update_latent_data()
        return - self.log_likelihood(self._hyp)

    def loglikelihood_gradW(self, W):
        grad_w = grad(self.loglikelihood_W)
        return grad_w(W)

    def train(self):
        if self._nlevels==3 and self._nested==True:
            sigmas = self._hyp[-self._nlevels:]
            theta00 = self._hyp[self._idx_theta[0]]
            hyp1_init = np.hstack([np.array([sigmas[0]]), theta00[1:]])
            result1 = minimize(value_and_grad(self.loglike1), hyp1_init, jac=True, method='L-BFGS-B', tol=1e-6, callback=self.callback)

            theta11 = self._hyp[self._idx_theta[1]]
            hyp2_init = np.hstack([np.array([sigmas[1]]), theta11[1:]])
            result2 = minimize(value_and_grad(self.loglike2), hyp2_init, jac=True, method='L-BFGS-B', tol=1e-6, callback=self.callback)

            theta22 = self._hyp[self._idx_theta[2]]
            hyp3_init = np.hstack([np.array([sigmas[2]]), theta22[1:]])
            result3 = minimize(value_and_grad(self.loglike3), hyp3_init, jac=True, method='L-BFGS-B', tol=1e-6, callback=self.callback)

            sigmas[0] = result1.x[0]
            sigmas[1] = result2.x[0]
            sigmas[2] = result3.x[0]
            theta00[1:] = result1.x[1:]
            theta11[1:] = result2.x[1:]
            theta22[1:] = result3.x[1:]

            n1 = self._training_data['X_0'].shape[0]
            par1 = np.hstack([0, theta00[1:]])
            sig_e1 = np.exp(sigmas[0])
            K_00 = self.kernel(self._training_data['Z_0'], self._training_data['Z_0'], par1) + np.eye(n1) * sig_e1
            L_00 = np.linalg.cholesky(K_00 + np.eye(n1) * self._jitter)
            alpha1 = np.linalg.solve(L_00.T, np.linalg.solve(L_00, self._training_data['Y_0']))
            theta00[0] = np.log(np.matmul(self._training_data['Y_0'].T, alpha1) / (n1-1))

            n2 = self._training_data['X_1'].shape[0]
            par2 = np.hstack([0, theta11[1:]])
            sig_e2 = np.exp(sigmas[1])
            K_11 = self.kernel(self._training_data['Z_1'], self._training_data['Z_1'], par2) + np.eye(n2) * sig_e2
            L_11 = np.linalg.cholesky(K_11 + np.eye(n2) * self._jitter)
            H2 = np.hstack([np.ones((n2, 1)), self._training_data['Y_0'][-n2:,0].reshape(-1, 1)])
            BIG2 = np.matmul(H2.T, np.linalg.solve(L_11.T, np.linalg.solve(L_11, H2)) )
            L_BIG2 = np.linalg.cholesky(BIG2 + np.eye(2) * self._jitter)
            rho_hat1 = np.linalg.solve(L_BIG2.T, np.linalg.solve(L_BIG2, np.matmul(H2.T, np.linalg.solve(L_11.T, np.linalg.solve(L_11, self._training_data['Y_1']))) ))[1, 0]
            alpha2 = np.linalg.solve(L_11.T, np.linalg.solve(L_11, self._training_data['Y_1'] - rho_hat1 * self._training_data['Y_0'][-n2:].reshape(-1, 1)))
            theta11[0] = np.log( np.matmul( (self._training_data['Y_1'] - rho_hat1*self._training_data['Y_0'][-n2:].reshape(-1, 1)).T, alpha2) / (n2 - 2))

            n3 = self._training_data['X_2'].shape[0]
            par3 = np.hstack([0, theta22[1:]])
            sig_e3 = np.exp(sigmas[2])
            K_22 = self.kernel(self._training_data['Z_2'], self._training_data['Z_2'], par3) + np.eye(n3) * sig_e3
            L_22 = np.linalg.cholesky(K_22, np.eye(n3) * self._jitter)
            H3 = np.hstack([np.ones((n3, 1)), self._training_data['Y_1'][-n3:,0].reshape(-1, 1)])
            BIG3 = np.matmul(H3.T, np.linalg.solve(L_22.T, np.linalg.solve(L_22, H3)))
            L_BIG3 = np.linalg.cholesky(BIG3 + np.eye(2)*self._jitter)
            rho_hat2 = np.linalg.solve(L_BIG3.T, np.linalg.solve(L_BIG3, np.matmul(H3.T, np.linalg.solve(L_22.T, np.linalgs.solve(L_22, self._training_data['Y_2']))) ))[1, 0]
            alpha3 = np.linalg.solve(L_22.T, np.linalg.solve(L_22, self._training_data['Y_2'] - rho_hat2 * self._training_data['Y_1'][-n3:].reshape(-1, 1)))
            theta22[0] = np.log(np.matmul( (self._training_data['Y_2'] - rho_hat2 * self._training_data['Y_1'][-n3:].reshape(-1, 1)).T, alpha3 ) / (n3-2))

            self._hyp[-self._nlevels:] = sigmas
            self._hyp[self._idx_theta[0]] = theta00
            self._hyp[self._idx_theta[1]] = theta11
            self._hyp[self._idx_theta[2]] = theta22
            self._hyp[-2 * self._nlevels+1:-self._nlevels] = np.array([rho_hat1, rho_hat2])

        else:
            result = minimize(value_and_grad(self.log_likelihood), self._hyp, jac=True, method='L-BFGS-B', tol=1e-4, callback=self.callback)
            self._hyp = result.x


    def predict(self, X_star):
        nl = [self._training_data['X_'+str(i)].shape[0] for i in range(self._nlevels)]
        y = np.vstack([self._training_data['Y_'+str(i)] for i in range(self._nlevels)])

        rhos = self._hyp[-2 * self._nlevels+1:-self._nlevels]
        sigmas = np.exp(self._hyp[-self._nlevels])

        theta00 = self._hyp[self._idx_theta[0]]
        theta11 = self._hyp[self._idx_theta[1]]
        if self._nlevels==3 and self._nested==True:
            theta22 = self._hyp[self._idx_theta[2]]
            K_00 = self.kernel(self._training_data['Z_0'], self._training_data['Z_0'], theta00) + np.eye(nl[0]) * sigmas[0]
            L_00 = np.linalg.cholesky(K_00 + np.eye(nl[0]) * self._jitter)
            R_0 = self.kernel(self._training_data['Z_0'], X_star, theta00)
            y_00 = np.matmul(R_0.T, np.linalg.solve(L_00.T, np.linalg.solve(L_00, self._training_data['Y_0'])))
            I0vI0 = np.matmul(np.ones((nl[0], 1)).T, np.linalg.solve(L_00.T, np.linalg.solve(L_00, np.ones((nl[0], 1)))))
            R0vR0 = np.matmul(R_0.T, np.linalg.solve(L_00.T, np.linalg.solve(L_00, R_0)))

            y_var_00 = self.kernel(X_star, X_star, theta00) - R0vR0

            data_11 = self._training_data['Y_1'] - rhos[0] * self._training_data['Y_0'][-nl[1]:].reshape(-1, 1)
            K_11 = self.kernel(self._training_data['Z_1'], self._training_data['Z_1'], theta11) + np.eye(nl[1]) * sigmas[1]
            L_11 = np.linalg.cholesky(K_11 + np.eye(nl[1]) * self._jitter)
            R_1 = self.kernel(self._training_data['Z_1'], X_star, theta11)
            y_11 = rhos[0] * y_00 + np.matmul(R_1.T, np.linalg.solve(L_11.T, np.linalg.solve(L_11, data_11)))
            I1vI1 = np.matmul(np.ones((nl[1], 1)).T, np.linalg.solve(L_11.T, np.linalg.solve(L_11, np.ones((nl[1], 1)))))
            R1vR1 = np.matmul(R_1.T, np.linalg.solve(L_11.T, np.linalg.solve(L_11, R_1)))

            y_var_11 = self.kernel(X_star, X_star, theta11) - R1vR1 + rhos[0]**2 * y_var_00

            data_22 = self._training_data['Y_2'] - rhos[1] * self._training_data['Y_1'][-nl[2]:].reshape(-1, 1)
            K_22 = self.kernel(self._training_data['Z_2'], self._training_data['Z_2'], theta22) + np.eye(nl[2]) * sigmas[2]
            L_22 = np.linalg.cholesky(K_22 + np.eye(nl[2]) * self._jitter)
            R_2 = self.kernel(self._training_data['Z_2'], X_star, theta22)
            y_22 = rhos[1] * y_11 + np.matmul(self.kernel(self._training_data['Z_2'], X_star, theta22).T, np.linalg.solve(L_22.T, np.linalg.solve(L_22, data_22)))
            I2vI2 = np.matmul(np.ones((nl[2], 1)).T, np.linalg.solve(L_22.T, np.linalg.solve(L_22, R_2)))

            y_var_22 = self.kernel(X_star, X_star, theta22) - R2vR2 + rhos[1]**2 * y_var_11
            return y_22, np.diag(y_var_22).reshape(-1, 1)

        else:

            L = self._L

            psi1 = rhos[0] * self.kernel(X_star, self._training_data['Z_0'], theta00)
            psi2 = rhos[0] ** 2 * self.kernel(X_star, self._training_data['Z_1'], theta00) + self.kernel(X_star, self._training_data['Z_1'], theta11)
            if self._nlevels==3:
                psi3 = rhos[1]**2 * rhos[0] * self.kernel(X_star, self._training_data['Z_2'], theta00) + rhos[1] * self.kernel(X_star, self._training_data['Z_2'], theta11) + self.kernel(X_star, self._training_data['Z_2'], theta22)
                psi = np.hstack([psi1, psi2, psi3])
            else:
                psi = np.hstack([psi1, psi2])

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            pred_u_star = np.matmul(psi, alpha)

            beta = np.linalg.solve(L.T, np.linalg.solve(L, psi.T))
            if self._nlevels==3:
                var_u_star = rhos[1] ** 2 * rhos[0] ** 2 * self.kernel(X_star, X_star, theta00) + rhos[1]**2 * self.kernel(X_star, X_star, theta11) + self.kernel(X_star, X_star, theta22) - np.matmul(psi, beta)
            else:
                var_u_star = rhos[0]**2 * self.kernel(X_star, X_star, theta00) + self.kernel(X_star, X_star, theta11) - np.matmul(psi, beta)

            return pred_u_star, var_u_star


    def callback(self, params):
        if self._nlevels==3 and self._nested==True:
            print("Let's print current the parameter values: " + str(params))
        else:
            print("Log likelihood {}".format(self.log_likelihood(params)))




