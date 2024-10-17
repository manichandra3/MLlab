"""Fuzzy K-means clustering"""

# ==============================================================================
# Author: Ammar Sherif <ammarsherif90 [at] gmail [dot] com >
# License: MIT
# ==============================================================================

# ==============================================================================
# The file includes an implementation of a fuzzy version of kmeans with sklearn-
# like interface.
# ==============================================================================

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state


class FuzzyKMeans(KMeans):
    """The class implements the fuzzy version of kmeans
    ----------------------------------------------------------------------------
    Args: same arguments as in SKlearn in addition to
        - m: the fuzziness index to determine how fuzzy our boundary is
        - eps: the tolerance value for convergence
    """

    def __init__(self, m, eps=0.001, *args, **kwargs):
        self.__m = m
        self.__eps = eps
        self.fmm_ = None
        self.__fitted = False
        super(FuzzyKMeans, self).__init__(*args, **kwargs)

    # --------------------------------------------------------------------------

    def __is_fitted(self):
        return self.__fitted

    # --------------------------------------------------------------------------

    def _check_params(self, X):
        if (self.__m <= 1):
            raise ValueError(
                "the fuzziness index m should be more than 1"
                f", got '{self.__m}' instead."
            )
        super(FuzzyKMeans, self)._check_params(X)

    # --------------------------------------------------------------------------

    def __compute_dist(self, data, centroids):
        """The method computes the distance matrix for each data point with respect to each cluster centroid.
        ------------------------------------------------------------------------

        Inputs:
            - data: the input data points
            - centroids: the clusters' centroids

        Output:
            - distance_m: the distance matrix
        """
        n_points = data.shape[0]
        n_clusters = centroids.shape[0]

        distance_m = np.zeros((n_points, n_clusters))

        for i in range(n_clusters):
            diff = data - centroids[i, :]
            distance_m[:, i] = np.sqrt((diff * diff).sum(axis=1))

        return distance_m

    # --------------------------------------------------------------------------

    def __update_centroids(self, data, fmm):
        """The method computes the updated centroids according the  computed
        fuzzy membership matrix <fmm> of the previous centroids.
        ------------------------------------------------------------------------

        Inputs:
            - data: the input data points
            -  fmm: fuzzy membership matrix of each data point

        Output:
            - centroids: the newly computed centroids
        """
        # ----------------------------------------------------------------------
        # We start computing the normalizing denominator terms
        # ----------------------------------------------------------------------
        norm = np.sum(fmm ** self.__m, axis=0)

        # ----------------------------------------------------------------------
        # Initialize the new centroids with zeros
        # ----------------------------------------------------------------------
        n_clusters = fmm.shape[1]
        n_features = data.shape[1]

        new_centroids = np.zeros((n_clusters, n_features))
        # ----------------------------------------------------------------------
        # Loop computing each one
        # ----------------------------------------------------------------------
        for i in range(n_clusters):
            # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            # Notice that we multiply the data points by the ith column of <fmm>
            # which represent the probabilities of being assigned to the ith cluster.
            # After that, we sum all the weighted points and average them
            # by dividing over the norm of that cluster.
            # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            new_centroids[i, :] = np.sum(data * (fmm[:, i] ** self.__m)[:, None],
                                         axis=0) / norm[i]

        return new_centroids

    # --------------------------------------------------------------------------

    def _compute_membership(self, data, centroids):
        """The method computes the membership matrix of the data according to
        the clusters specified by the given centroids

        Inputs:
            - data: the input data points being clustered
            - centroids: numpy array including the cluster centroids;
                its shape is (n_clusters, n_features)

        Outputs:
            - fmm: fuzzy membership matrix"""
        # ----------------------------------------------------------------------
        # First, compute  the distance between the point and the other centroids
        # ======================================================================
        # Note we also add alpha,  1e-10 very little value, as  we are computing
        # 1 over the distances, and there might be 0 distance
        # ----------------------------------------------------------------------

        dist = self.__compute_dist(data, centroids) + 1e-10
        # ----------------------------------------------------------------------
        # We are computing the below value once because we need it in both the
        # numerator and the denominator of the value to be computed
        # ----------------------------------------------------------------------
        sqr_dist = dist ** (-2 / (self.__m - 1))

        # ----------------------------------------------------------------------
        # We compute the normalizing term (denominator)
        # ----------------------------------------------------------------------
        norm_dist = np.expand_dims(np.sum(sqr_dist, axis=1), axis=1)

        fmm = sqr_dist / norm_dist
        return fmm

    # --------------------------------------------------------------------------

    def __converged(self, centroids, new_centroids):
        """The method checks convergence"""

        # ----------------------------------------------------------------------
        # We compute the squared difference between both; indicate convergence
        # if the total distance of the centroids is below the eps
        # ----------------------------------------------------------------------
        diff = (centroids - new_centroids) ** 2

        return np.sum(diff) <= self.__eps

    # --------------------------------------------------------------------------

    def fit(self, X, y=None, sample_weight=None):
        """The method computes the fuzzy k-means clustering algorithm

        Inputs:
            - X: training data
            - y: ignored
            - sample_weight: weights of each data point
        """
        # ----------------------------------------------------------------------
        # Generate a random_state and do some initializations
        # ----------------------------------------------------------------------
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params(X)
        random_state = check_random_state(self.random_state)

        # ----------------------------------------------------------------------
        # Initialize the centroids
        # ----------------------------------------------------------------------
        centroids = self._init_centroids(X, x_squared_norms=None, init=self.init,
                                         random_state=random_state)
        # ----------------------------------------------------------------------
        # Do the first iteration
        # ----------------------------------------------------------------------
        fmm = self._compute_membership(X, centroids)
        new_centroids = self.__update_centroids(X, fmm)

        itr = 0
        while (not self.__converged(centroids, new_centroids) \
               and itr < self.max_iter):
            centroids = new_centroids
            # ------------------------------------------------------------------
            # compute the new fuzzy membership matrix, fmm; then, update the new
            # centroids.
            # ------------------------------------------------------------------
            fmm = self._compute_membership(X, centroids)
            new_centroids = self.__update_centroids(X, fmm)
            itr += 1
        # ----------------------------------------------------------------------
        # Save the results
        # ----------------------------------------------------------------------
        self.cluster_centers_ = new_centroids
        self.labels_ = fmm.argmax(axis=1)
        self.fmm_ = fmm

        self.__fitted = True

        return self

    # --------------------------------------------------------------------------

    def compute_membership(self, data, centroids=None):
        """The method computes  the membership matrix  of the data  according to
        the fitted points.

        Inputs:
            - data: the input data points being clustered
            - centroids: numpy array including the cluster centroids;
                its shape is (n_clusters, n_features)
        Outputs:
            - fmm: fuzzy membership matrix of each data point"""

        if centroids is not None:
            return self._compute_membership(data, centroids)
        elif not self.__is_fitted():
            raise RuntimeError("You did not fit the estimator yet.")
        else:
            return self._compute_membership(data, self.cluster_centers_)

    # --------------------------------------------------------------------------

    def predict(self, X, sample_weight=None):
        """The method clusters each data point according to previously fitted
        data."""

        if not self.__is_fitted():
            raise RuntimeError("You did not fit the estimator yet.")

        X = self._check_test_data(X)

        return self.compute_membership(X).argmax(axis=1)

    # --------------------------------------------------------------------------

    def score(self, X, y=None, sample_weight=None):
        """Not supported by this implementation"""
        pass

    # --------------------------------------------------------------------------
