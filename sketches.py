import numpy as np 
from numpy.linalg import qr


def _is_power_2(n):
    return int(2**int(np.log(n) / np.log(2)) == n)


def _srht(indices, v):
    n = v.shape[0]
    v = np.random.choice([-1,1], n, replace=True).reshape((-1,1)) * v
    if n == 1:
        return v
    i1 = indices[indices < n//2]
    i2 = indices[indices >= n//2]
    if len(i1) == 0:
        return _srht(i2-n//2, v[:n//2,::]-v[n//2:,::])
    elif len(i2) == 0:
        return _srht(i1, v[:n//2,::]+v[n//2:,::])
    else:
        return np.vstack([_srht(i1, v[:n//2,::]+v[n//2:,::]), _srht(i2-n//2, v[:n//2,::]-v[n//2:,::])])


def srht(matrix, sketch_size):
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1,1))
    # pad matrix with 0 if first dimension is not a power of 2
    n = matrix.shape[0]
    if not _is_power_2(n):
        new_dim = 2**(int(np.log(n) / np.log(2))+1)
        matrix = np.vstack([matrix, np.zeros((new_dim - n, matrix.shape[1]))])
    n = matrix.shape[0]
    indices = np.sort(np.random.choice(np.arange(n), sketch_size, replace=False))
    return 1./np.sqrt(n) * _srht(indices, matrix)



def haar(matrix, sketch_size):
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1,1))
    n = matrix.shape[0]
    A = 1./np.sqrt(n)*np.random.randn(n, sketch_size)
    S = qr(A, mode='reduced')[0].T
    return S @ matrix


def gaussian(matrix, sketch_size):
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1,1))
    n = matrix.shape[0]
    S = 1./np.sqrt(sketch_size)*np.random.randn(sketch_size, n)
    return S @ matrix























