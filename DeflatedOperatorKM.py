from Deflation import DeflatedOperator
from Deflation import DeflationOperator
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import LinearOperator
import util
import numpy as np

class DeflatedOperatorKM(LinearOperator):
    def __init__(self, K, M, sigma, Z):
        """docstring for __init__"""
        super().__init__(shape=K.shape, dtype=np.result_type(K, M, sigma, Z))
        self.Mi = aslinearoperator(util.spdia_inv(M))
        KM = aslinearoperator(K - sigma * M)
        self.A = LinearOperator(shape=K.shape,
                                matvec=lambda x: KM.matvec(self.Mi.matvec(x)))
        self.P = DeflationOperator(self.A, Z)

    def _matvec(self, x):
        return self.P.matvec(self.A.matvec(x))
