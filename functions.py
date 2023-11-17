import numpy as np
from numpy.linalg import norm 
from numpy.linalg import inv
from numpy.linalg import eig
import random




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
    k = 0
    while True: 
      xk1 = xk - alpha * self.quad_gradient(A, b, xk)
      error = norm(xk1 - xk, 2)
      if error < epsilon:
        self.sd_min = xk1
        self.sd_steps = k
        self.sd_tolerance = error
        break
      k += 1
      xk = xk1
    return self
  
  def quad_newton(self, A, b, alpha = 0.01, epsilon = 1e-6):
    xk = np.array(A.shape[0] * [0])
    k = 0
    while True:
      xk1 = xk - np.matmul(inv(self.quad_hessian(A)), self.quad_gradient(A, b, xk))
      error = norm(xk1 - xk, 2)
      if error < epsilon:
        self.newton_min = xk1
        self.newton_steps = k
        self.newton_tolerance = error
        self.alpha_max = QuadMinimizer.quad_max_alpha(self, A = A)
        break
      k += 1
      xk = xk1
    return self
  
  # def quad_conj_grad(self, A, b):
  
  def quad_max_alpha(self, A):
    vals, vecs = eig(A)
    alpha_max = 2 / max(vals)
    return alpha_max
  
  def __repr__(self):
    return ""





class Adaline(BaseMinimizer):
  """A class to house ADALINE fits."""

  def __init__(self):
    """Initialize attributes."""
    self.flavor = 'adaline'

  def adaline(self, P, T, W0 = None, b0 = None, alpha = 0.01, epsilon = 1e-6):

    # P = np.array([[1, -1, -1], [1, 1, -1]])
    # T = np.array([[-1, 1]])
    # W0 = None 
    # bo = None 
    # alpha = 0.01
    # epsilon = 1e-6

    R = P.shape[1]   # Input dimension.
    S = T.shape[0]   # Output dimension = #(neurons)
    N = P.shape[0]   # Number of observations.

    if W0 is None:
      W0 = np.zeros((S, R))
    if b0 is None: 
      b0 = np.zeros(S)

    # Initialize steepest descent. 
    Wk = W0
    bk = b0
    k = 0

    while True:

      # Stochastic.  Choose a datum. 
      n = random.randint(0, N - 1)
      # Wk = W[n]
      # bk = b[n]
      pk = P[n]
      tk = T[0, n]

      # Take a step. 
      ak = np.matmul(Wk, pk) + bk
      ek = tk - ak
      Wk1 = Wk + 2 * alpha * np.outer(ek, pk)
      bk1 = bk + 2 * alpha * ek

      error_W = norm(np.squeeze(Wk1 - Wk), 2)
      error_b = norm(np.expand_dims(bk1 - bk, 0), 2)
      if ((error_W < epsilon) & (error_b < epsilon)):
        self.adaline_W_min = Wk1
        self.adaline_b_min = bk1
        self.adaline_steps = k
        self.adaline_tolerance_W = error_W
        self.adaline_tolerance_b = error_b
        self.alpha_max = Adaline.adaline_max_alpha(self, P = P)
        break
      k += 1
      Wk = Wk1
      bk = bk1
    return self


  def adaline_max_alpha(self, P):

    # P = np.array([[1, -1, -1], [1, 1, -1]])

    # Assume input vectors p have equal probability of selection to 
    # estimate expected value of input correlation matrix R.  
    R = P.shape[1]   # Input dimension.
    N = P.shape[0]   # Number of observations.
    RR = np.zeros((R, R))
    for i in range(N):
      RR += np.outer(P[i], P[i])
    RR = (1 / N) * RR

    vals, vecs = eig(RR)
    alpha_max = 1 / max(vals)
    return alpha_max

  def __repr__(self):
    return ""


# A = 0.5 * np.array([[10, -6], [-6, 10]])
# b = np.array([4, 4])

# -1 * np.matmul(inv(A), b)

# quad = QuadMinimizer()
# quad.quad_sd(A = A, b = b)
# quad.quad_newton(A = A, b = b)


P = np.array([[1, -1, -1], [1, 1, -1]])
T = np.array([[-1, 1]])
adaline = Adaline()
adaline.adaline(P = P, T = T)
# adaline.adaline_max_alpha(P = P)