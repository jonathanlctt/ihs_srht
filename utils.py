import numpy as np

from time import time


def timeit(method):
    def timed(*args):
        start = time()
        result = method(*args)
        end = time()
        return result, end-start
    return timed


def generate_example(n=1024, d=64, nu=1.):
	A = 1./np.sqrt(n)*np.random.randn(n, d)
	U, _, V = np.linalg.svd(A, full_matrices=False)
	Sigma = np.array([0.9/(ii+1) for ii in range(d)])
	A = np.dot(U, Sigma*V.T)
	xpl = 1./np.sqrt(d)*np.random.randn(d,)
	b = A.dot(xpl) + 1./np.sqrt(n)*np.random.randn(n,)
	de = np.sum( Sigma**2 / (Sigma**2 + nu**2) )
	return A, b, de