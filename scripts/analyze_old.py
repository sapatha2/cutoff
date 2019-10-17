import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_pickle('test_old.pickle')

plt.subplot(231)
plt.title('Wfval')
plt.plot(df['zcoord'],df['wfval'],'b.')
plt.axvline(0.52, c='k',ls='--')
plt.axhline(0)
plt.ylim((-5e-5, 5e-5))

plt.subplot(232)
plt.title('Energy')
plt.plot(df['zcoord'],df['total'],'b.')
plt.axvline(0.52, c='k',ls='--')

plt.subplot(233)
plt.title('dpH')
for i in range(df.shape[0]):
  plt.plot(df.iloc[i]['zcoord'],df.iloc[i]['dpH'][0],'b.')
plt.axvline(0.52, c='k',ls='--')

plt.subplot(234)
plt.title('dppsi')
for i in range(df.shape[0]):
  plt.plot(df.iloc[i]['zcoord'],df.iloc[i]['dppsi'][0],'b.')
plt.axvline(0.52, c='k',ls='--')
plt.show()

plt.subplot(235)
plt.title('dpH^2*Psi^2')
for i in range(df.shape[0]):
  plt.plot(df.iloc[i]['zcoord'],df.iloc[i]['dpH'][0]**2*df.iloc[i]['wfval']**2,'b.')
plt.axvline(0.52, c='k',ls='--')

plt.subplot(236)
plt.title('dppsi^2*Psi^2')
for i in range(df.shape[0]):
  plt.plot(df.iloc[i]['zcoord'],df.iloc[i]['dppsi'][0]**2*df.iloc[i]['wfval']**2,'b.')
plt.axvline(0.52, c='k',ls='--')
plt.show()
