"""
Dissimilarity measures for clustering
"""

import numpy as np


def matching_dissim(a, b, **_):
    """Simple matching dissimilarity function"""
    return np.sum(a != b, axis=1)

def distance(a, b):
    result = []
    dist = 0
# =============================================================================
#     for centroidi in a:
#         if centroidi[0] != b[0]:
#             dist += 1
#         if str(centroidi[1])[0] == str(b[1])[0]:
#             if str(centroidi[1])[1] == str(b[1])[1]:
#                 if str(centroidi[1])[2] == str(b[1])[2]:
#                     if str(centroidi[1])[3] == str(b[1])[3]:
#                         dist += 0
#                     else:
#                         dist += 0.25
#                 else:
#                     dist += 0.5
#             else:
#                 dist += 0.75
#         else:
#             dist += 1
#         result.append(dist)
#         dist = 0
# =============================================================================
    for centroidi in a:
        if centroidi[0] != b[0]:
            dist += 1
        if str(centroidi[1])[0:2] == str(b[1])[0:2]:
            if str(centroidi[1])[2:4] == str(b[1])[2:4]:
                dist += 0.
                
            else:
                dist += 0.5
        else:
            dist += 1
        result.append(dist)
        dist = 0
    return result

def new_dissim(a, b, **_):
    """有序分类属性"""
    
    
    return distance(a, b)
    

def euclidean_dissim(a, b, **_):
    """Euclidean distance dissimilarity function"""
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    return np.sum((a - b) ** 2, axis=1)


def ng_dissim(a, b, X=None, membship=None):
    """Ng et al.'s dissimilarity measure, as presented in
    Michael K. Ng, Mark Junjie Li, Joshua Zhexue Huang, and Zengyou He, "On the
    Impact of Dissimilarity Measure in k-Modes Clustering Algorithm", IEEE
    Transactions on Pattern Analysis and Machine Intelligence, Vol. 29, No. 3,
    January, 2007

    This function can potentially speed up training convergence.

    Note that membship must be a rectangular array such that the
    len(membship) = len(a) and len(membship[i]) = X.shape[1]

    In case of missing membship, this function reverts back to
    matching dissimilarity (e.g., when predicting).
    """
    # Without membership, revert to matching dissimilarity
    if membship is None:
        return matching_dissim(a, b)

    def calc_cjr(b, X, memj, idr):
        """Num objects w/ category value x_{i,r} for rth attr in jth cluster"""
        xcids = np.where(memj == 1)
        return float((np.take(X, xcids, axis=0)[0][:, idr] == b[idr]).sum(0))

    def calc_dissim(b, X, memj, idr):
        # Size of jth cluster
        cj = float(np.sum(memj))
        return (1.0 - (calc_cjr(b, X, memj, idr) / cj)) if cj != 0.0 else 0.0

    if len(membship) != a.shape[0] and len(membship[0]) != X.shape[1]:
        raise ValueError("'membship' must be a rectangular array where "
                         "the number of rows in 'membship' equals the "
                         "number of rows in 'a' and the number of "
                         "columns in 'membship' equals the number of rows in 'X'.")

    return np.array([np.array([calc_dissim(b, X, membship[idj], idr)
                               if b[idr] == t else 1.0
                               for idr, t in enumerate(val_a)]).sum(0)
                     for idj, val_a in enumerate(a)])
