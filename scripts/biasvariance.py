import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_pickle('integratecutoff.pickle')
ind = np.argsort(df['cutoff'])

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (3,6), sharex=True)
x = df['cutoff'].iloc[ind]
y = df['bias'].iloc[ind]/1e-12
poly = 1e-2
p = np.polyfit(x[x>poly], y[x>poly], 3)
print(p)
xfit = np.linspace(min(x[x>poly]), max(x), 1000)
fit = p[0]*xfit**3 + p[1]*xfit**2 + p[2]*xfit + p[3]
ax[0].plot(np.log10(x), y, 'o')
ax[0].plot(np.log10(xfit), fit, '--')
ax[0].set_ylabel(r'Bias/$10^{-12}$')

x = np.log10(df['cutoff'].iloc[ind])
y = np.log10(df['variance'].iloc[ind])
poly = -2 #Log10 not Log1
p = np.polyfit(x[x<=poly], y[x<=poly], 1)
print(p[0], p[1])
xfit = np.logspace(min(x),max(x[x<=poly]), 1000)
fit = p[0]*np.log10(xfit) + p[1]
ax[1].plot(x, y, 'o')
ax[1].plot(np.log10(xfit), fit, '--')
ax[1].set_xlabel(r'log$_{10}(\epsilon)$')
ax[1].set_ylabel(r'log$_{10}$(Variance)')
plt.savefig('biasvariance.pdf',bbox_inches='tight')
plt.close()
