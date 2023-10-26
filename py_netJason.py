import numpy as np
from numpy.linalg import norm 
from numpy.linalg import inv

class BaseMinimizer():
  """A class to house basic results from minimizing functions."""

  def __init__(self, flavor):
    """Initialize attributes."""
    self.flavor = 'basic'


class QuadMinimizer(BaseMinimizer):

  def __init__(self):
    self.form = 'quadratic'
    print('Assuming a ' + self.form + ' functional form.\n')

  def quad_gradient(self, A, b, xk):
    return np.matmul(A, xk) + b
  
  def quad_hessian(self, A):
    return A

  def quad_sd(self, A, b, alpha = 0.01, epsilon = 1e-6):

    # A = 0.5 * np.array([[10, -6], [-6, 10]])
    # b = np.array([4, 4])
    # alpha = 0.01
    # epsilon = 1e-6

    xk = np.array(A.shape[0] * [0])
    i = 0
    while True: 
      xk1 = xk - alpha * self.quad_gradient(A, b, xk)
      error = norm(xk1 - xk, 2)
      if error < epsilon:
        self.sd_min = xk1
        self.sd_steps = i
        self.sd_tolerance = error
        break
      i += 1
      xk = xk1
    return self
  
  def quad_newton(self, A, b, alpha = 0.01, epsilon = 1e-6):
    xk = np.array(A.shape[0] * [0])
    i = 0
    while True:
      xk1 = xk - np.matmul(inv(self.quad_hessian(A)), self.quad_gradient(A, b, xk))
      error = norm(xk1 - xk, 2)
      if error < epsilon:
        self.newton_min = xk1
        self.newton_steps = i
        self.newton_tolerance = error
        break
      i += 1
      xk = xk1
    return self
  
  def __repr__(self):
    return ""



A = 0.5 * np.array([[10, -6], [-6, 10]])
b = np.array([4, 4])

quad = QuadMinimizer()
quad.quad_sd(A = A, b = b)
quad.quad_newton(A = A, b = b)