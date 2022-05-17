import itertools
import multiprocessing
import numpy as np
from scipy.spatial import KDTree
from sklearn.manifold import MDS

# Chamfer Distances --------------------------------------------------------------------------------

def chamfer_distance(a, b, p=1):
	"""
	Compute the Chamfer distance between two sets
	"""
	d1 = np.mean([np.min(np.linalg.norm(a - b[j], axis=1, ord=p)) for j in range(len(b))])
	d2 = np.mean([np.min(np.linalg.norm(b - a[j], axis=1, ord=p)) for j in range(len(a))])
	return (d1 + d2)/2.0

def chamfer_distance_kdtree(a: KDTree, b: KDTree, p=1, workers=1):
	"""
	Compute the Chamfer distance between two KDTrees
	"""
	d_ab = np.mean(a.query(b.data, p=p, workers=workers)[0])
	d_ba = np.mean(b.query(a.data, p=p, workers=workers)[0])
	return (d_ab + d_ba)/2.0

def chamfer_distance_matrix(sets, p=1, workers=1, indices=None, fn=chamfer_distance_kdtree, result=None):
	"""
	Compute the distance matrix between sets using Chamfer distance
	"""
	n = len(sets)
	if indices is None:
		indices = np.array(list(itertools.combinations(np.arange(n), 2)))
	if result is None:
		result = np.zeros((n, n))
	with multiprocessing.Pool(workers) as pool:
		result[tuple(indices.T[::-1])] = \
		result[tuple(indices.T)] = pool.map(ChamferDistanceProcessor(fn, sets, p), indices)
	return result

def chamfer_distance_extend_matrix(sets_a, sets_b, dist_mat=None, p=1, workers=1, fn=chamfer_distance_kdtree):
	"""
	Efficiently extend a Chamfer distance matrix by adding the list of sets in set_b
	"""
	sets = sets_a + sets_b
	n = len(sets)

	if len(dist_mat) != n:
		m = n - len(dist_mat)
		dist_mat = np.pad(dist_mat, ((0, m), (0, m)))

	indices_a = np.arange(len(sets_a))
	indices_b = np.arange(len(sets_b)) + len(sets_a)
	indices = np.array(
		list(itertools.product(indices_a, indices_b)) +
		list(itertools.combinations(indices_b, 2)))
	return chamfer_distance_matrix(sets, p=p, workers=workers, indices=indices, fn=fn, result=dist_mat)

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
		mds.fit_transform(self.dist_mat)
		return mds.stress_
