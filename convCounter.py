#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from numpy.linalg import norm
from numpy import array
import numpy as np
__all__ = ['convCounter']

class convCounter:
    """
    convCounter monitories the residual convergence though the callback
                function

    """
    def __init__(self, callbackRes = lambda x : x, callbackNorm=norm,
                verbose=False, increment = 1, label='no_set'):
        self.iter_ = 0
        self.resVec = np.array([])
        self.callbackRes = callbackRes
        self.callbackNorm = callbackNorm
        self.verbose = verbose
        self.increment = increment
        self.label=label

    def __call__(self, x):
        self.callback(x)

    def callback(self, x):
        if self.callbackRes is not None:
            rnrm = self.callbackNorm(self.callbackRes(x))
            self.resVec = np.append(self.resVec, rnrm)
            if self.verbose:
                print("{0}\t{1}".format(self.iter_, rnrm))
        self.iter_ += self.increment

    def toFile(self, fname, header = "mv\t r\n"):
        try:
            with open(fname, 'w') as f:
                f.write(header)
                i = 0
                for res in self.resVec:
                    f.write("{0}\t{1}\n".format(i, res))
                    i += self.increment
        except:
            print("Unable to open file {0}".format(fname))

    def reset(self, callbackRes=None, callbackNorm=norm,
                verbose=False, increment=1):
        self.iter_ = 0
        self.resVec = []
        self.callbackRes = callbackRes
        self.callbackNorm = callbackNorm
        self.verbose = verbose

    def finalResidualNorm(self):
        if len(self.resVec) == 0:
            return -1
        return self.resVec[-1]

    def toArray(self):
        return array(self.resVec)

    def printInfo(self):
        print(self)

    def scaleResVec(self, alpha):
        self.resVec = alpha*self.resVec

    def __str__(self):
        return "Number of iter: {0}, final residual {1}".format(self.iter_, self.finalResidualNorm())

    def __len__(self):
        return len(self.resVec)

    def __getitem__(self, index):
        return self.resVec[index]
