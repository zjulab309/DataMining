"""
typically very useful k-means clustering algorithm
"""
import warnings

import numpy as np
import scipy.sparse as sp

from ..utils import validation

################################################################################
#
# This function initialization heuristic
#
################################################################################

def _k_init_seeds(x,n_clusters,n_local_trails = None,
                  random_state = None,x_squared_norms = None):
    """
    Init n_cluster seeds according to k-means prameters
    x: array or sparse matrix, shape(n_sample,n_features)

    n_cluster: integer

    n_local_trails: integer, optional
        The number of seeding trails for each center(except the first) of which
        the one reducing inertia the most is greedily chosen.  Set to None to
        make the number of trails depend logarithmically on the numer of seeds;
        this is the default

    random_state: integer, optional
    The generator used to initialize the centers.  If an integer is given, it
    fixes the seed.  Defaults to the global numpy random number generator

    x_squared_norms: array, shape(n_samples,), optional

    """
    n_samples, n_features = x.shape
    #random_state = check_random_state(random_state)

    centers = np.empty(n_clusters,n_features)

    # Set the number of local seeding trials if none is given
    if n_local_trails is None:
        n_local_trails = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(x):
        centers[0] = x[center_id].toarray()
    else:
        centers[0] = x[center_id]

    # Initialize list of closest distance and calculate current potential
    if x_squared_norms is None:
        closest_dist_sq = euclidean_distance()


def _squared_norms(x):
    """
    Compute the squared euclidean norms the rows of x
    """
    #if sp.issparse(x):
       # return _k_means.
    #else:
    return (x ** 2).sum(axis = 1)
        #



class KMeans():
    """K-Means Clustering

    Parameters
    ----------
    n_clusters: int, optional,default:8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter: int
        Maximum number of interations of the k-means algorithm for
        a single run.

    n_init: int ,optional,default:10
        Number of time the k-means algorithm will be run with different
        centroid seeds.  The final results will be the best output of n_init
        consecutive runs in terms of inertia.

    init: {'k-means++','random' or adarray}
        Method for initialization, defaults to 'k-means++'
        Clustering in a smart way to speed up convergence.

    'random': choose k observations(rows) at random from data for the initial
     centroids.

    If an ndarray is passed, it should be of shape(n_cluster, n_features) and
    gives the initial centers.

    precompute_distances: boolen
       Percompute distance

    tol: float, optional default:1e-4
       Relative tolerance w.r.t:inertia to declare convergence

    n_jobs:int
       The number of jobs to use for the computation.  This works by breaking
       down the pairwise matrix into n_jobs even slices and computing them in
       parallel.

    random_state: integer or numpy.  RandomState, optional
       The generator used to initialize the centers.  If an integer is used
       given, it fixess the seed,Defaults to the global numpy random number
       generator.

    Attributes
    ----------
    'cluster_centers_': array,[n_clusters,n_features]
        Coordinates of cluster centers

    'labels_':
        Labels of each point

    'inertia_':float
        The value of the inertia criterion associated with the chosen partition.

    Note:
    ----
    The k-means problem is solved using Lloyd's algorithm.

    The average complexity is given by 0(k n t). t is the number of iteration
    and n is the number of samples.

    The worst case complexity is given by 0((n^(k+2/p))

    In practice, the k-means algorithm is very fast clustering algorithm, but it
    fall in local minima.  That's why it can be useful to restart it several
    times.
    """
    def __init__(self, n_clusters = 8, init = 'k-menas++', n_init = 10,
                 max_iter = 300,tol = 1e-4, precompute_distances = True,
                 verbose = 0, random_state = None,copy_x = True, n_jobs = 1):
        if hasattr(init,'__array__'):
            n_cluster = init.shape[0]
            init = np.asarray(init,dtype = np.float64)

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs

    def _check_fit_data(self, x):
        """Verify that the number of samples given is larger than k cluster
        """
        x = validation.atleast2d_or_csr(x,dtype=np.float64)
        if x.shape[0] < self.n_clusters:
            raise ValueError("n_sample = %d should be >= n_clusters = %d"%(
                x.shape[0],self.n_clusters))

        return x

    def _check_test_data(self, x):
        x = validation.atleast2d_or_csr(x, dtype = np.float64)
        n_samples, n_features = x.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features."
                             "Got %d features,but expected %d"%(
                                 n_features,expected_n_features))

        if not x.dtype.kind is 'f':
            warnings.warn("Got data type %s, converted to float"
                          "to avoid overflows"%x.dtype,
                          RuntimeWarning,stacklevel = 2)
            x = x.astype(np.float)

        return x

    def _check_fitted(self):
        if not hasattr(self,"cluster_centers_"):
            raise AttributeError("The Model has not been trained yet.")

    def fit(self, x, y = None):
        """Train the K-means Mode.  Compute k-means clustering in other words.

        Parameters:
        ----------
        x: array-like or sparse matrix, shape = (n_samples, n_features)
        """
