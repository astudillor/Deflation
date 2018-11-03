#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import unittest
from Deflation import DeflationOperator


class TestDeflation(unittest.TestCase):

    def setUp(self):
        n = 6
        self.A = np.array([[2, -1, 0, 0, 0, 0],
                           [-1, 2, -1, 0, 0, 0],
                           [0, -1, 2, -1, 0, 0],
                           [0, 0, -1, 2, -1, 0],
                           [0, 0, 0, -1, 2, -1],
                           [0, 0, 0, 0, -1, 2]])
        self.Z = np.zeros((n, 2))
        alpha = np.sqrt(2.0 / (n + 1))
        for j in range(0, 6):
            self.Z[j, 0] = -alpha * np.sin((j + 1) * 1 * np.pi / (n + 1))
            self.Z[j, 1] = -alpha * np.sin((j + 1) * 2 * np.pi / (n + 1))

        self.P = DeflationOperator(self.A, self.Z)

    def test_Zx(self):
        result = np.array([1.62569130185772,
                           1.76979143396546,
                           0.117361291282443,
                           -2.20184484796086,
                           -3.44141745773056,
                           -2.55337375755503])
        x = np.array([2, -5])
        np.testing.assert_array_almost_equal(result, self.P.Z.dot(x),
                                             decimal=14)

    def test_ZTx(self):
        result = np.array([-8.19663606324645,
                           3.88481581113951])
        x = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_almost_equal(result, self.P.Z.T.dot(x),
                                             decimal=14)

    def test_Eix(self):
        result = np.array([10.0978346790446,
                           -6.63992638802841])
        x = np.array([2, -5])
        np.testing.assert_array_almost_equal(result, self.P.Ei.dot(x),
                                             decimal=14)

    def test_Qx(self):
        result = np.array([7.44186412383325,
                           14.6062491816228,
                           20.36966537151,
                           22.7626121558209,
                           19.9831517987508,
                           11.753805234256])
        x = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_almost_equal(result, self.P.multQ(x),
                                             decimal=12)

    def test_Px(self):
        result = np.array([0.722520933956309,
                           0.599031132097594,
                           -0.370469405576205,
                           -1.17240714138105,
                           -0.449886207424725,
                           2.47554133023887])
        x = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_almost_equal(result, self.P.matvec(x),
                                             decimal=12)

    def test_project_back(self):
        result = np.array([3.46475941368335,
                           5.54037412929381,
                           5.79128452689964,
                           4.9893467910948,
                           4.49145678977151,
                           5.21777980996588])
        x = np.array([1, 2, 3, 4, 5, 6])
        b = np.array([1, 1, 1, 1, 1, 1])
        np.testing.assert_array_almost_equal(result,
                                             self.P.project_back(b, x),
                                             decimal=12)


if __name__ == '__main__':
    unittest.main()
