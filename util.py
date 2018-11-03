from datetime import datetime


import numpy as np
import os
import scipy.io as sio
import scipy.sparse as sparse


class MatricesKMZ(object):
    def __init__(self, begin=0, end=30):
        """docstring for __init__"""
        self._begin = begin
        self._end = end
        self._index = begin
        mat_contents = sio.loadmat('matrices/KM.mat')
        self._KArray = mat_contents['KArray'][0][begin:end]
        self._MArray = mat_contents['MArray'][0][begin:end]
        self._ZArray = mat_contents['RBM'][0][begin:end]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._index < self._end:
            K, M, Z = self._KArray[self._index], self._MArray[self._index], self._ZArray[self._index]
            self._index += 1
            return K, M, Z
        else:
            raise StopIteration()

def print_array_real(array):
    for x in array:
        print (x)
    print('')

def rayleigh(A, x, B = None):
    """ Returns the (generalized) Rayleigh quotient
    $$ x^T A x / x^T B x $$
    """
    if B is None:
        return np.inner(x, A.dot(x)) / np.inner(x, x)

    return np.inner(x, A.dot(x)) / np.inner(x, B.dot(x))

def spdia_inv(M, format = 'm'):
    """ Returns the inverse an sparse diagonal matrix
    """
    Mi = sparse.dia_matrix(M)
    Mi.data[0] = np.reciprocal(Mi.data[0])
    if format[0] == 'v':
        return Mi.data[0]
    return Mi

def write_vector_txt(fname, vector, header = 'x y'):
    n = len(vector)
    X = np.zeros((n, 2))
    X [:, 0] = range(1, n + 1)
    X [:, 1] = vector
    np.savetxt(fname, X, fmt = ['%d','%.18e'], header = header, comments='')

def print_array(array):
    for x in np.sort(np.real(array)):
        print(x, '', end='')
    print('')


def print_error_i(evalues, i):
    x = np.load('base/evalues' + str(i) + '.npy')
    print('Absolute error', np.linalg.norm(x[:13] - evalues))
    print('Relative error', np.linalg.norm(x[:13] - evalues) / np.linalg.norm(x[:13]))


def make_dir():
    directory = os.path.join(os.getcwd(), 'logs',
                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(directory)

    except OSError as e:
        print('Error creating log/output directory', e)
        directory = None
    return directory
