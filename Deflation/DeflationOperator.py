#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.sparse.linalg import LinearOperator, aslinearoperator

import numpy as np

__all__ = ['DeflationOperator', 'DeflatedOperator']


def vec2mat(x):
    n = x.shape[0]
    if x.shape == (n,):
        return np.reshape(a=x, newshape=(n, 1))
    return np.array(x)


class DeflationOperator(LinearOperator):
    ''' Representation of the deflation operator
        for symmetric matrices.
    '''

    def __init__(self, A, Z):
        '''
        A: matrix to be deflated (sparse matrix | LinearOperator | ndarray)
        Z: deflation space (ndarray)
        '''
        super().__init__(shape=A.shape, dtype=np.result_type(A, Z))
        self.Z = vec2mat(Z)
        self.A = aslinearoperator(A)
        self.AZ = vec2mat(self.A.matmat(self.Z))
        self.Ei = np.linalg.inv(np.matmul(Z.T, self.AZ))

    def multPT(self, x):
        return x - self.multQ(self.A.matvec(x))

    def multQ(self, x):
        return np.matmul(self.Z, self.Ei.dot(np.matmul(self.Z.T, x)))

    def _matvec(self, x):
        '''
        Performs the deflation multiplication Px
        where P = I - AQ, Q = Z*inv(E)*Z.T, and
        E = Z.T*A*Z
        '''
        return x - self.AZ.dot(self.Ei.dot(self.Z.T.dot(x)))

    def project_back(self, b, x):
        ''' Returns the solution of the original system using the solution
            of the projected system.
        '''
        return self.multQ(b) + self.multPT(x)

    def toarray(self):
        ''' Returns a dense ndarray representation of this operator.'''
        return self.matmat(np.eye(self.shape[0]))


class DeflatedOperator(LinearOperator):

    def __init__(self, A, Z):
        super().__init__(shape=A.shape, dtype=np.result_type(A, Z))
        self.P = DeflationOperator(A, Z)
        self.A = aslinearoperator(A)

    def _matvec(self, x):
        return self.P.matvec(self.A.matvec(x))

    def toarray(self):
        ''' Returns a dense ndarray representation of this operator.'''
        return self.matmat(np.eye(self.shape[0]))
