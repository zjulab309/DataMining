import numpy as np
from .utils import validation
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
def check_pairwise_arrays(x, y):
    """
     Set x and y appropriately and checks inputs
     if y is none, it is set as a pointer to x(i.e, not a copy)
     if y is given, this does not happen.
     All distance metrics should use this function first to assert that the
     given parameters a correct and safe to use.

     Specifically, this function first ensures that both x and y are arrays,
     then checks that they are at least two dimensional while ensuring that
     their elements are floats.  Finally, the function checks that the size
     of the second dimension of the two arrays is equal.

     Parameters
    ------------
     x: {array-like, sparse matrix}, shape = [n_samples_a,n_features]
     y: {array-like, sparse matrix}, shape = [n_samples_b,n_features]

     Returns
    ---------
    safe_x:{array-like, sparse matrix}, shape = [n_samples_a, n_features]
        An array equal to X, guaranteed to be a numpy array.
    safe_y:{array-like, sparse matrix}, shape = [n_samples_b, n_features]
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.
    """
    if y is x or y is None:
        x = y = validation.atleast2d_or_csr(x)
    else:
        x = validation.atleast2d_or_csr(x)
        y = validation.atleast2d_or_csr(y)
    if x.shape[1] != y.shape[2]:
        raise ValueError("Incompatible dimension for x and y matrics:"
                         "x.shape[1] == %dwhile y.shape[1]== %d"%(
                             x.shape[1],y.shape[1]))

        if not (x.dtype == y.dtype == np.float32):
            if y is x:
                x = y = x.astype(np.float)
            else:
                x = x.astype(np.float)
                y = y.astype(np.float)

    return x, y


def euclidean_distance(x,y = None, y_norm_squared = None, squared = None):
    """
     Constructing the rows of x (and y) as vectors, compute the distance
     matrix between each pair of vectors.

     For efficiency reasons, the euclidean distance between a pair of row
     vector x and y is computed as:
         dist(x, y) = sqrt(dot(x,x)-2*dot(x,y) + dot(y,y))

    Parameters
    ----------
    x: {array-like, sparse matrix}, shape = [n_samples_1,n_features]
    y: {array-like, sparse matrix}, shape = [n_samples_2,n_features]
    y_norm_squared: array-like, shape = [n_samples_2], optional
       pre-computed dot-products of vectors in Y
    squared: boolean, optional
       return squared euclidean distances

    Returns:
    -------
    distances:{array, sparse matrix}, shape = [n_samples1,n_samples_2]
    """
