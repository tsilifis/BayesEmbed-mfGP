"""
Date: 07/05/2020
Author: Panagiotis Tsilifis
"""

import numpy as np


class Posterior(object):

    _likelihood = None
    _prior = None

    def __init__(self, likelihood, prior):
        """
        Initializes the object
        """
        self._likelihood = likelihood
        self._prior = prior

    def eval_logp(self, W):
        #print('Splitting the posterior: ')
        #print(self._likelihood.eval(W), self._prior.eval_logp(W))
        #print('-' * 50)
        return self._likelihood.eval(W) + self._prior.eval_logp(W)

    def eval_grad_logp(self, W, complement = None, fixcols = False):
        if not fixcols:
            grad_logp1 = self._likelihood.eval_grad_W(W)
            grad_logp2 = self._prior.eval_grad_logp(W)
            #print(grad_logp1, grad_logp2)
            return (grad_logp1 + grad_logp2).T
        else:
            grad = np.vstack([complement.T, self._likelihood.eval_grad_W(W, fixcols)])
            return (grad + self._prior.eval_grad_logp(W)).T


class MFPosterior(object):

    _mfgp = None 
    _prior = None

    def __init__(self, mfgp, prior):
        """
        Initialize the object 
        """
        self._mfgp = mfgp
        self._prior = prior

    def eval_logp(self, W, *args):
        return self._mfgp.loglikelihood_W(W) + self._prior.eval_logp(W)

    def eval_grad_logp(self, W, complement=None, fixcols=False, *args):
        grad_logp1 = self._mfgp.loglikelihood_gradW(W)
        grad_logp2 = self._prior.eval_grad_logp(W)
        return (grad_logp1 + grad_logp2).T


