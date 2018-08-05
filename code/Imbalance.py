"""
Several methods for over-sampling to deal with imbalance
in supervised learning
"""

from annoy import AnnoyIndex # Annoy (Approximate Nearest Neighbors Oh Yeah) 
import numpy as np

"""
An implementation of:
* Chawla, Nitesh V., et al. "SMOTE: synthetic minority over-sampling technique." 
  Journal of artificial intelligence research 16 (2002): 321-357.
"""
def SMOTE(X, k, oversample_times, aknn_positive):
    # load AnnoyIndex
    feature_dim = X.shape[1]
    index = AnnoyIndex(feature_dim)
    index.load(aknn_positive)
    # generate synthetic examples
    X_new = []
    for i in range(X.shape[0]):
        x = X[i]
        knn = list( set(index.get_nns_by_vector(x, k+1)) - set([i]) ) # get NNs excluding the element itself
        for j in range(int(oversample_times)):
            x2 = X[knn[np.random.randint(k)]]
            x_new = x + (x2-x)*np.random.rand()
            X_new.append(x_new)
        if np.random.rand()<=(oversample_times-int(oversample_times)): # dealing with fractions
            x2 = X[knn[np.random.randint(k)]]
            x_new = x + (x2-x)*np.random.rand()
            X_new.append(x_new)
    X_new = np.asarray(X_new, dtype=np.float)
    X = np.concatenate((X, X_new))
    return X

"""
An implementation of:
* He, Haibo, et al. "ADASYN: Adaptive synthetic sampling approach for imbalanced learning." 
  Neural Networks, 2008. IJCNN 2008.(IEEE World Congress on Computational Intelligence). IEEE 
  International Joint Conference on. IEEE, 2008.
"""
def ADASYN(X, k, oversample_times, aknn_full, aknn_positive):
    # load AnnoyIndex for full dataset
    feature_dim = X.shape[1]
    index = AnnoyIndex(feature_dim)
    index.load(aknn_full)
    # figure amount of samples per point
    n_in = X.shape[0]
    n_out = oversample_times*n_in
    knns = list()
    rs = list() # list of "r"s from the ADASYN paper
    for i in range(n_in):
        x = X[i]
        knn = list( set(index.get_nns_by_vector(x, k+1)) - set([i]) ) # get NNs excluding the element itself
        knns.append(knn)
        rs.append( len([neighbor for neighbor in knn if neighbor>=n_in]) )
    # compute the normalization factor
    normalizer = n_out/sum(rs)
    # load AnnoyIndex for positive examples
    index = AnnoyIndex(feature_dim)
    index.load(aknn_positive)
    # generate synthetic examples
    X_new = []
    for i in range(n_in):
        r = rs[i]
        if r==0:
            continue
        n_i = r*normalizer
        x = X[i]
        knn = list( set(index.get_nns_by_vector(x, k+1)) - set([i]) ) # get NNs excluding the element itself
        for j in range(int(n_i)):
            x2 = X[knn[np.random.randint(k)]]
            x_new = x + (x2-x)*np.random.rand()
            X_new.append(x_new)
        if np.random.rand()<=(n_i-int(n_i)): # dealing with fractions
            x2 = X[knn[np.random.randint(k)]]
            x_new = x + (x2-x)*np.random.rand()
            X_new.append(x_new)
    X_new = np.asarray(X_new, dtype=np.float)
    X = np.concatenate((X, X_new))
    return X
