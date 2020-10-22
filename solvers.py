import numpy as np 
from scipy.linalg import cholesky, solve_triangular, qr

from ls import LeastSquares
from sketches import srht, haar, gaussian

from utils import timeit



class IHS(LeastSquares):

	def __init__(self, A, b, mode='solver'):
		
		LeastSquares.__init__(self, A, b, mode=mode)


	def compute_params(self, sketch, sketch_size):
		self.step_sizes = {}
		if sketch == 'srht' or sketch == 'haar':
			gamma = self.d / self.n 
			xi = sketch_size / self.n
			th1 = (1-gamma)/(xi-gamma)
			th2 = (1-gamma)*(gamma**2+xi-2*gamma*xi)/(xi-gamma)**3
			eig_min = 0.98*(np.sqrt((1-gamma)*xi) - np.sqrt((1-xi)*gamma))**2
			eig_max = 1.02*(np.sqrt((1-gamma)*xi) + np.sqrt((1-xi)*gamma))**2
		elif sketch == 'gaussian':
			th1 = sketch_size / (sketch_size-self.d-1)
			th2 = sketch_size**2 * (sketch_size-1) / ((sketch_size-self.d)*(sketch_size-self.d-1)*(sketch_size-self.d-3))
			eig_min = (1-np.sqrt(rho*1.1))**2
			eig_max = (1+np.sqrt(rho*1.1))**2
		else:
			raise NotImplementedError
		self.step_size_fixed = 2 / (1/eig_min + 1/eig_max)
		self.step_size_fresh = th1 / th2
		self.cv_rate = 1 - th1**2 / th2


	def _sketch(self, sketch, sketch_size):
		return self.sketch_method[sketch](self.A, sketch_size)


	def _compute_direction(self, SA, x):
		g = np.dot(self.A.T, np.dot(self.A, x) - self.b)
		HS = np.dot(SA.T, SA)
		dx = - np.dot(np.linalg.pinv(HS), g)
		return dx


	def _update(self, x, dx):
		return x + dx


	def solve(self, sketch_size, sketch='srht', n_iterations=50, freq_update=1):
		self.compute_params(sketch, sketch_size)
		x = np.zeros((self.d,))
		errors = [self.compute_error(x)]
		th_errors = [1.]
		for iter_ in range(n_iterations):
			if iter_ % freq_update == 0:
				SA = self._sketch(sketch, sketch_size)
			dx = self._compute_direction(SA, x)
			if iter_ % freq_update == 0:
				x = self._update(x, self.step_size_fresh*dx)
			else:
				x = self._update(x, self.step_size_fixed*dx)
			errors.append(self.compute_error(x))
			th_errors.append(self.cv_rate**(iter_+1))
		return x, np.array(th_errors), np.array(errors) / errors[0]





class pCG(LeastSquares):

	def __init__(self, A, b, mode='solver'):
		
		LeastSquares.__init__(self, A, b)


	def _init_pcg(self, *argv):
		if self.sketch == 'srht':
			E = srht(np.hstack([self.A, self.b.reshape((-1,1))]), self.m)
		elif self.sketch == 'gaussian':
			E = 1./np.sqrt(self.m) * np.dot(np.random.randn(self.m, self.n), np.hstack([self.A, self.b.reshape((-1,1))]))
		else:
			raise NotImplementedError
		z = E[::,-1].squeeze()
		E = E[::,:-1]
		Q, R, pi = qr(E, pivoting=True, mode='economic')
		Pi = np.zeros((pi.shape[0], pi.shape[0]))
		Pi[range(pi.shape[0]), pi] = 1
		z = np.dot(Pi.T, solve_triangular(R, np.dot(Q.T, z)))
		y = np.copy(z)
		Atb = solve_triangular(R.T, np.dot(Pi, np.dot(self.A.T, self.b)), lower=True)
		AtAy = np.dot(Pi.T, solve_triangular(R, y))
		AtAy = solve_triangular(R.T, np.dot(Pi, np.dot(self.A.T, np.dot(self.A, AtAy))), lower=True)
		r, p = Atb - AtAy, Atb - AtAy
		x = np.dot(Pi.T, solve_triangular(R, y))
		return x, y, r, p, R, Pi


	def _pcg_iteration(self, *argv):
		R, Pi, x, y, r, p = argv[:6]
		Ap = np.dot(Pi.T, solve_triangular(R, p))
		Ap = solve_triangular(R.T, np.dot(Pi, np.dot(self.A.T, np.dot(self.A, Ap))), lower=True)
		rtr = np.linalg.norm(r)**2
		alpha = np.float(rtr / np.sum(p*Ap))
		y = y + alpha * p 
		r = r - alpha * Ap 
		beta = np.linalg.norm(r)**2 / rtr 
		p = r + beta * p
		x = np.dot(Pi.T, solve_triangular(R, y))	
		return x, y, r, p 


	def solve(self, sketch_size, sketch='srht', n_iterations=50):
		self.sketch, self.m = sketch, sketch_size
		x, y, r, p, R, Pi = self._init_pcg()
		errors = [self.compute_error(x)]
		for iteration in range(n_iterations):
			x, y, r, p = self._pcg_iteration(R, Pi, x, y, r, p)
			errors.append(self.compute_error(x))
		return x, np.array(errors)/errors[0]





class CG(LeastSquares):

	def __init__(self, A, b, mode='solver'):
		
		LeastSquares.__init__(self, A, b)


	def _init_cg(self):
		x = np.zeros((self.d,))
		Atb = self.A.T.dot(self.b)
		r, p = Atb.copy(), Atb.copy()
		return x, r, p 


	def _cg_iteration(self, *argv):
		x, r, p = argv[:3]
		Ap = np.dot(self.A.T, np.dot(self.A,p))
		alpha = np.sum(r**2) / np.sum(p*Ap)
		x = x + alpha*p
		_r = r - alpha*Ap 
		beta = np.sum(_r**2) / np.sum(r**2)
		r = np.copy(_r)
		p =  r + beta*p 	
		return x, r, p	


	def solve(self, n_iterations=50):
		x, r, p = self._init_cg()
		errors = [self.compute_error(x)]
		for iteration in range(n_iterations):
			x, r, p = self._cg_iteration(x, r, p)
			errors.append(self.compute_error(x))
		return x, np.array(errors)/errors[0]





























