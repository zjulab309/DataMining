"""
typically very useful k-means clustering algorithm
"""
import warnings

import numpy as np
import scipy.sparse as sp

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
    n_sample, n_features = x.shape
    #random_state = check_random_state(random_state)

    centers = np.empty(n_clusters,n_features)

    # Set the number of local seeding trials if none is given
    if n_local_trails is None:
        n_local_trails = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    #if sp.is
