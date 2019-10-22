import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats 

dpH_total = []
weight_total = []
for i in range(1,100):
  print(i)
  df = pd.read_json('evals'+str(i)+'.json')

  dpH = df['dpH'].values
  wfval = df['wfval'].values
  wfpval = df['wfpval'].values

  logweight = 2 * (wfval - wfpval)
  weight = np.exp(logweight)

  dpH_total += list(dpH)
  weight_total += list(weight)

  '''
  ind_max = np.argsort(-np.abs(dpH))[0]
  ind_min = np.argsort(-np.abs(dpH))[-1]
  print(dpH[ind_max], dpH[ind_min])
  print(wfval[ind_max], wfval[ind_min])
  print(wfpval[ind_max], wfpval[ind_min])
  print(logweight[ind_max], logweight[ind_min])
  print(weight[ind_max], weight[ind_min])
  '''

data = np.log10(np.abs(dpH_total))

hist, bin_edges = np.histogram(data, bins = 100, weights = weight_total, density = True)
plt.plot(bin_edges[:-1], np.log10(hist), 'k.')
plt.plot(bin_edges[1:], np.log10(hist), 'k.')

x = bin_edges[:-1][(bin_edges[:-1] > 5)*(bin_edges[:-1] < 10)]
y = np.log10(hist)[(bin_edges[:-1] > 5)*(bin_edges[:-1] < 10)]
p = np.polyfit(x, y, 1)
print(p)
plt.plot(bin_edges[:-1], p[0]*bin_edges[:-1] + p[1], 'r-')

'''
kde = stats.gaussian_kde(data, weights = weight_total)
x = np.linspace(min(data), max(data), 500)
plt.plot(x, np.log10(kde(x)), 'm-')
'''
plt.savefig('test.pdf',bbox_inches='tight')
