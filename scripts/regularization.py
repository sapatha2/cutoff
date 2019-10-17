import numpy as np 
import matplotlib.pyplot as plt 

epsilon = 1
def f(x, epsilon): 
  c = (105./8.)/epsilon**6 - (49./8.)/epsilon**2
  b = (-1. -2.*c*epsilon**6)/epsilon**4
  a = (-2*b*epsilon**2 - 3*c*epsilon**4)

  return a*x**2 + b*x**4 + c*x**6

x = np.linspace(-epsilon, epsilon, 1000)

plt.suptitle('epsilon = 1')

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
plt.ylabel('f(x)/x^2')
plt.xlabel('x')
plt.plot(x, f(x, epsilon)/x**2, '-')
plt.axvline(epsilon, c='k', ls='--')
plt.axvline(-epsilon, c='k', ls='--')

plt.show()
