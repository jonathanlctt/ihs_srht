import numpy as np
from sketches import srht, haar, gaussian


class LeastSquares:

	def __init__(self, A, b, mode='evaluation'):
		'''
		A : n x d ndarray -- data matrix
		b : (n,) or (n,1) ndarray -- target vector
		'''
		self.A = A 
		self.b = b.squeeze()
		self.n, self.d = A.shape
		self.mode = mode 
		if self.mode == 'evaluation':
			self.xstar = np.linalg.inv(np.dot(self.A.T, self.A)).dot(np.dot(self.A.T, self.b))
		self.sketch_method = {'haar': haar, 'gaussian': gaussian, 'srht': srht}


	def compute_error(self, x):
		if self.mode == 'solver':
			return 1./2*np.sum((self.A.dot(x)-self.b)**2)
		elif self.mode == 'evaluation':
			return 1./2*np.sum((self.A.dot(x-self.xstar))**2)
		else:
			raise NotImplementedError



