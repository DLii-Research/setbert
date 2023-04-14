import multiprocessing
import numpy as np
from sklearn.manifold import MDS

# Metric Multidimensional-scaling (MDS) ------------------------------------------------------------

def mds(dist_mat, ndim=2, metric=True, seed=None, n_jobs=None, **kwargs):
    """
    Compute MDS given the distance matrix.

    Returns: (embedded space, stress)
    """
    mds = MDS(n_components=ndim, metric=metric, dissimilarity="precomputed", random_state=seed,
              n_jobs=n_jobs, **kwargs)
    pca = mds.fit_transform(dist_mat)
    return pca, mds.stress_

def mds_stress_analysis(dist_mat, dims, metric=True, seed=None, workers=1, **kwargs):
    """
    Compute the MDS stress value for the given possible range of components.
    """
    if isinstance(dims, int):
        dims = range(1, dims+1)
    dims = list(dims)
    with multiprocessing.Pool(workers) as pool:
        stresses = pool.map(MdsStressAnalysisProcess(
            dist_mat, metric=metric, random_state=seed, **kwargs
        ), dims)
    return dims, (1 - np.cumsum(stresses) / np.sum(stresses))

# Pool Processing Helper Classes  ------------------------------------------------------------------

class ChamferDistanceProcessor:
    def __init__(self, fn, sets, p=1):
        self.fn = fn
        self.sets = sets
        self.p = p

    def __call__(self, indices):
        a, b = indices
        return self.fn(self.sets[a], self.sets[b], self.p)

class MdsStressAnalysisProcess:
    def __init__(self, dist_mat, **kwargs):
        self.dist_mat = dist_mat
        self.kwargs = kwargs

    def __call__(self, dim):
        mds = MDS(n_components=dim, dissimilarity="precomputed", **self.kwargs)
        mds.fit(self.dist_mat)
        return mds.stress_
