import numpy as np
from pyqmc.energy import energy

class PGradTransform:
    """   """

    def __init__(self, enacc, transform, nodal_cutoff=1e-5):
        self.enacc = enacc
        self.transform = transform
        self.nodal_cutoff = nodal_cutoff

    def _node_cut(self, configs, wf):
        """ Return true if a given configuration is within nodal_cutoff 
        of the node """
        ne = configs.configs.shape[1]
        d2 = 0.0
        for e in range(ne):
            d2 += np.sum(wf.gradient(e, configs.electron(e)) ** 2, axis=0)
        r = 1.0 / d2
        mask = r  < self.nodal_cutoff ** 2
        return mask, r 

    def __call__(self, configs, wf):
        pgrad = wf.pgradient()
        d = self.enacc(configs, wf)
        energy = d["total"]
        dp = self.transform.serialize_gradients(pgrad)
  
        d["dpH"] = np.einsum("i,ij->ij", energy, dp)
        d["dppsi"] = dp
        d["dpidpj"] = np.einsum("ij,ik->ijk", dp, dp)
        
        node_cut, r2 = self._node_cut(configs, wf)
        
        c = 7./(self.nodal_cutoff**6)
        b = (-1. -2.*c*self.nodal_cutoff**6)/self.nodal_cutoff**4
        a = (-2*b*self.nodal_cutoff**2 - 3*c*self.nodal_cutoff**4)
        
        l2 = r2[node_cut, np.newaxis]
        d["dpH"][node_cut, :] *= a * l2 + b * l2**2 + c * l2**3
        return d

    def avg(self, configs, wf):
        nconf = configs.configs.shape[0]
        pgrad = wf.pgradient()
        den = self.enacc(configs, wf)
        energy = den["total"]
        dp = self.transform.serialize_gradients(pgrad)

        node_cut = self._node_cut(configs, wf)
        dp[node_cut, :] = 0.0
        # print('number cut off',np.sum(node_cut))

        d = {}
        for k, it in den.items():
            d[k] = np.mean(it, axis=0)
        d["dpH"] = np.einsum("i,ij->j", energy, dp) / nconf
        d["dppsi"] = np.mean(dp, axis=0)
        d["dpidpj"] = np.einsum("ij,ik->jk", dp, dp) / nconf

        return d
