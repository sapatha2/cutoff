import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as integrate
from scipy.misc import derivative

def f(x, epsilon): 
  c = 7./(epsilon**6)
  b = (-1. -2.*c*epsilon**6)/epsilon**4
  a = (-2*b*epsilon**2 - 3*c*epsilon**4)

  return a*x**2 + b*x**4 + c*x**6

#Testing 
for epsilon in [1e-3, 1, 10, 100]:
  print(f(epsilon, epsilon))
  print(f(-epsilon, epsilon))

  integral = integrate.quad(lambda x: f(x, epsilon), -epsilon, epsilon)
  print(integral[0]/epsilon) 

  deriv = derivative(lambda x: f(x, epsilon), epsilon, dx = 1e-10)
  print(deriv)

#Plot 
x = np.linspace(-epsilon, epsilon, 1000)
plt.suptitle('epsilon = '+str(epsilon))

plt.subplot(311)
plt.ylabel('f(x)')
plt.plot(x, f(x, epsilon), '-')
plt.axhline(1, c='k', ls='--')
plt.axvline(epsilon, c='k', ls='--')
plt.axvline(-epsilon, c='k', ls='--')

plt.subplot(312)
plt.ylabel('f(x) - 1')
plt.plot(x, f(x, epsilon) - 1, '-')
plt.axvline(epsilon, c='k', ls='--')
plt.axvline(-epsilon, c='k', ls='--')

plt.subplot(313)
plt.ylabel('f(x)/^2x^2')
plt.xlabel('x')
plt.plot(x, f(x, epsilon)**2/x**2, '-')
plt.axvline(epsilon, c='k', ls='--')
plt.axvline(-epsilon, c='k', ls='--')

plt.show()
