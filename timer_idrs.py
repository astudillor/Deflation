#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from convCounter import convCounter
from DeflatedOperatorKM import DeflatedOperatorKM
from idrs import idrs
from scipy.sparse.linalg import LinearOperator

import argparse
import numpy as np
import os
import util
import time
import matplotlib.pyplot as plt


def defineArgs():
    parser = argparse.ArgumentParser(
     description='Compute 13 eigenvalues of generalized eigenvalue problems.')
    parser.add_argument("-b", "--begin", help="Beginning", type=int, default=0)
    parser.add_argument(
        "-e", "--end", help="Number of experiments", type=int, default=1)
    parser.add_argument('-g', '--sigma', help='sigma',
                        type=float, default=1.0e8)
    parser.add_argument('-s', '--sidr', help='shadow size',
                        type=int, default=4)
    parser.add_argument('-t', '--tol', help='tol residual',
                        type=float, default=1e-8)
    parser.add_argument('-r', '--seed', help='shadow size',
                        type=int, default=0)
    parser.add_argument('-p', '--plotting', help='plotting graph',
                        dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    return parser.parse_args()


def print_info(A, b, x, counter, label, time):
    print("number of iteration", len(counter),
          "relative error", np.linalg.norm(b - A.dot(x)) / np.linalg.norm(b),
          "time", time, label)


def run_timer(K, M, Z, x0, b, sigma, tol, s):
    KM = K - sigma * M
    # without deflation
    counter1 = convCounter()
    start = time.time()
    x1, info1 = idrs(KM, b, callback=counter1, tol=tol, s=s)
    end = time.time()
    time_no_deflation = end - start
    print_info(KM, b, x1, counter1, label='(N/Deflation)',
               time=time_no_deflation)

    # with deflation
    start = time.time()
    A = DeflatedOperatorKM(K, M, sigma, Z)
    counter2 = convCounter()
    x2, info2 = idrs(A, A.P.matvec(b), callback=counter2, tol=tol, s=s)
    x2 = A.P.project_back(b, x2)
    x2 = A.Mi.dot(x2)
    end = time.time()
    time_deflation = end - start
    print_info(KM, b, x2, counter2, label='(W/Deflation)',
               time=time_deflation)

    # Eigenvectors in G0
    start = time.time()
    options = dict()
    options['U0'] = np.random.randn(K.shape[0], s)
    options['U0'][:, s-3:s] = Z
    Mi = util.spdia_inv(M)
    A = LinearOperator(shape=K.shape, matvec=lambda x: Mi.dot(KM.dot(x)))
    counter3 = convCounter()
    x3, info3 = idrs(A, Mi.dot(b), callback=counter3, tol=tol, s=s,
                     options=options)
    end = time.time()
    time_deflation = end - start
    print_info(KM, b, x=x3, counter=counter3, label='(With eig G0)',
               time=time_deflation)


def write_log(filename, sigma, tol, s):
    try:
        with open(filename, 'w') as fhandle:
            fhandle.write('idr(' + str(s) + ')')
            fhandle.write('tol ' + str(tol))
            fhandle.write('sigma ' + str(sigma))
    except IOError as e:
        print(e)


def run_plotter(K, M, Z, x0, b, sigma, tol, s):
    KM = K - sigma * M
    # without deflation
    counter1 = convCounter(callbackRes=lambda x: b - KM.dot(x))
    x1, info1 = idrs(KM, b, callback=counter1, tol=tol, s=s)
    counter1.scaleResVec(1.0 / np.linalg.norm(b))

    # with deflation
    A = DeflatedOperatorKM(K, M, sigma, Z)
    bhat = A.P.matvec(b)
    counter2 = convCounter(callbackRes=lambda x: bhat - A.matvec(x))
    x2, info2 = idrs(A, bhat, callback=counter2, tol=tol, s=s)
    x2 = A.P.project_back(b, x2)
    x2 = A.Mi.dot(x2)
    counter2.scaleResVec(1.0 / np.linalg.norm(bhat))

    # Eigenvectors in G0
    options = dict()
    options['U0'] = np.random.randn(K.shape[0], s)
    options['U0'][:, s-3:s] = Z
    Mi = util.spdia_inv(M)
    A = LinearOperator(shape=K.shape, matvec=lambda x: Mi.dot(KM.dot(x)))
    bhat = Mi.dot(b)
    counter3 = convCounter(callbackRes=lambda x: bhat - A.matvec(x))
    x3, info3 = idrs(A, bhat, callback=counter3, tol=tol, s=s,
                     options=options)
    counter3.scaleResVec(1.0 / np.linalg.norm(bhat))

    directory = util.make_dir()
    if directory is None:
        print("Couldn't write the results to files")
    else:
        fname1 = os.path.join(directory, 'idrs' + str(s) + '_Ndef.dat')
        fname2 = os.path.join(directory, 'idrs' + str(s) + '_Wdef.dat')
        fname3 = os.path.join(directory, 'idrs' + str(s) + '_G0def.dat')
        counter1.toFile(fname=fname1)
        counter2.toFile(fname=fname2)
        counter3.toFile(fname=fname3)
        plt.title('Residual norm with IDR(' + str(s) + ')')
        plt.semilogy(counter1, label='No deflation')
        plt.semilogy(counter2, label='Deflation')
        plt.semilogy(counter3, label=r'Enriched $G_0$')
        plt.legend()
        plt.xlabel('matvecs')
        plt.ylabel(r'$\|r_i\|_2$')
        fname_fig = os.path.join(directory, 'figure.pdf')
        print('Plotting in', directory)
        plt.savefig(fname_fig)
        write_log(directory + '/info.log', sigma, tol, s)


def main():
    args = defineArgs()
    print("Sigma", args.sigma)
    tol = args.tol
    s = args.sidr
    sigma = args.sigma
    np.random.seed(seed=args.seed)
    print('Solving the systems using idrs({:0})'.format(s))
    for K, M, Z in util.MatricesKMZ(args.begin, args.end):
        b = np.zeros((K.shape[0],))
        b[0] = 1.0
        x0 = np.zeros((K.shape[0],))

        run_timer(K, M, Z, x0, b, sigma, tol, s)
        if args.plot:
            run_plotter(K, M, Z, x0, b, sigma, tol, s)


if __name__ == '__main__':
    main()
