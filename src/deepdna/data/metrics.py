import multiprocessing
import numpy as np
from sklearn.manifold import MDS

def binary_clf_curve(y_true, y_pred):
    indices = np.argsort(y_pred)[::-1]
    thresholds = y_pred[indices]
    tps = np.cumsum(y_true[indices])
    fps = np.cumsum(1 - y_true[indices])
    return fps.astype(np.float64), tps.astype(np.float64), thresholds

def ppv_npv_curve(y_true, y_pred, pad: bool = False):
    if pad:
        if not np.isclose(y_pred[0], 1.0):
            y_true = np.concatenate(([1], y_true))
            y_pred = np.concatenate(([1.0], y_pred))
        if not np.isclose(y_pred[-1], 0.0):
            y_true = np.concatenate((y_true, [0]))
            y_pred = np.concatenate((y_pred, [0.0]))

    fps, tps, thresholds = binary_clf_curve(y_true, y_pred)
    fns, tns, _ = binary_clf_curve(1 - y_true, 1.0 - y_pred)

    fns, tns = fns[::-1], tns[::-1]

    ps = tps + fps
    ns = tns + fns

    # Initialize the result array with zeros to make sure that precision[ps == 0]
    # does not contain uninitialized values.
    ppv = np.zeros_like(tps)
    np.divide(tps, ps, out=ppv, where=(ps != 0))

    npv = np.zeros_like(tns)
    np.divide(tns, ns, out=npv, where=(ps != 0))

    # reverse the outputs so recall is decreasing
    sl = slice(None, None, -1)
    return np.hstack((ppv[sl], 1)), np.hstack((npv[sl], 0)), thresholds[sl]

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
