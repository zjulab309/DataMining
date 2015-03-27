"""
Utilities for validating input values
"""
import warnings
import numbers

import numpy as np
from scipy import sparse

def _assert_all_finite(x):
    """Like assert_all_finite, but only for ndarray"""
    if(x.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(x.sum())
       and not np.isfinite(x).all()):
        raise ValueError("Array contains NaN or infinity.")


def _atleast2d_or_sparse(x, dtype, order, copy, sparse_class,
                          convmenthod, force_all_finite):
    if sparse.issparse(x):
        if dtype is None or x.type == dtype:
            x = getattr(x, convmenthod)()
        else:
            x = sparse_class(x,dtype = dtype)
        if force_all_finite:
            _assert_all_finite(x.data)
    else:
        x = array2d(x, dtype = dtype, order = order,copy = copy,
                    force_all_finite = force_all_finite)
        if force_all_finite:
            _assert_all_finite(x)
    return x



def atleast2d_or_csr(x,dtype = None, order = None, copy = False,
                     force_all_finite = True):
    """Like numpy.atleast_2d, but converts sparse matrices to CSR format.
    Also, converts np.matrix to np.ndarray.
    """
    return _atleast2d_or_sparse(x,dtype,order,copy,sparse.csr_matrix,
                                "tocsr",force_all_finite)



def array2d(x, dtype = None, order = None, copy = False,
            force_all_finite = True):
    """Returns at least 2-d array with data from x"""
    if sparse.issparse(x):
        raise TypeError('A sparse matrix was passed, but dense data is'
                        'required. Use x.toarray() to convert to dense')
    x_2d = np.asarray(np.atleast_2d(x),dtype = dtype, order = order)
    if force_all_finite:
        _assert_all_finite(x_2d)
    #if x is x_2d and copy:
        #x_2d = safe_copy(x_2d)
    return x_2d
