import numpy as np


def recover_data(Z, U, K):
    """
    recover the original data from the projection
    """    
    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # ===================== Your Code Here =====================
    # Instructions: Compute the approximation of the data by projecting back
    #               onto the original space using the top K eigenvectors in U.
    #
    #               For the i-th example Z[i], the approximate
    #               recovered data for dimension j is given as follows:
    #                   v = Z(i, :)';
    #                   recovered_j = v' * U(j, 1:K)';
    #                   (above is octave code)
    #

    Ureduce = U[:, np.arange(K)]
    X_rec = np.dot(Z, Ureduce.T)

    # ==========================================================

    return X_rec