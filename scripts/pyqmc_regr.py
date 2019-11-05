import numpy as np
from pyqmc.energy import energy

class PGradTransform_new:
    """   """

    def __init__(self, enacc, transform, nodal_cutoff=1e-3):
        self.enacc = enacc
        self.transform = transform
        self.nodal_cutoff = nodal_cutoff

    def _node_regr(self, configs, wf):
        """ Return true if a given configuration is within nodal_cutoff 
        of the node """
        ne = configs.configs.shape[1]
        d2 = 0.0
        for e in range(ne):
            d2 += np.sum(wf.gradient(e, configs.electron(e)) ** 2, axis=0)
        r = 1.0 / d2
        mask = r < self.nodal_cutoff ** 2
       
        c = 7./(self.nodal_cutoff ** 6)
        b = -15./(self.nodal_cutoff ** 4)
        a = 9./(self.nodal_cutoff ** 2)
        
        f = a * r + b * r ** 2 + c * r ** 3
        f[np.logical_not(mask)] = 1.

        return mask, f

    def __call__(self, configs, wf):
        pgrad = wf.pgradient()
        d = self.enacc(configs, wf)
        energy = d["total"]
        dp = self.transform.serialize_gradients(pgrad)
 
        node_cut, f = self._node_regr(configs, wf)

        d["dpH"] = np.einsum("i,ij->ij", energy, dp * f[:, np.newaxis])
        d["dppsi"] = dp 
        d["dpidpj"] = np.einsum("ij,ik->ijk", dp, dp * f[:, np.newaxis])
        return d

    def avg(self, configs, wf):
        nconf = configs.configs.shape[0]
        pgrad = wf.pgradient()
        den = self.enacc(configs, wf)
        energy = den["total"]
        dp = self.transform.serialize_gradients(pgrad)

        node_cut, f = self._node_regr(configs, wf)

        d = {}
        for k, it in den.items():
            d[k] = np.mean(it, axis=0)
        d["dpH"] = np.einsum("i,ij->j", energy, dp * f[:, np.newaxis]) / nconf
        d["dppsi"] = np.mean(dp, axis=0)
        d["dpidpj"] = np.einsum("ij,ik->jk", dp, dp * f[:, np.newaxis]) / nconf

        return d
