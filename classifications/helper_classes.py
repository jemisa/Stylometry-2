"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


class Feature_Preprocessor:
    """
        Pre-processes the features.
        Takes a feature matrix to learn from, can then be used to pre-processes unseen vectors.
            - Prunes invariable features
            - Centralizes data using Z-score
            - Performs PCA to reduce data
    """
    def __init__(self, matrix, centralize=True, pca=True, components=30):
        self.set_prune_indexes(matrix)

        pruned_matrix = []
        for i in xrange(len(matrix)):
            pruned_matrix.append(self.prune(matrix[i]))

        if centralize:
            self.set_centralize_params(pruned_matrix)
        else:
            self.centralize = lambda x : x

        if pca:
            self.set_pca_params(pruned_matrix, components)
        else:
            self.pca = lambda x : x

    def batch_normalize(self, matrix):
        new_matrix = []
        for row in matrix:
            new_matrix.append(self.normalize(row))
        return new_matrix

    def normalize(self, vector):
        return self.pca(self.centralize(self.prune(vector)))

    def prune(self, vector):
        return [ vector[i] for i in xrange(len(vector)) if i not in self.unseen ]

    def set_prune_indexes(self, matrix):
        seen = [ False for _ in xrange(len(matrix[0])) ]
        for vector in matrix:
            seen = [ seen[i] or vector[i] != 0.0 for i in xrange(len(vector)) ]
        unseen = [ i for i in xrange(len(seen)) if not seen[i] ]
        self.unseen = set(unseen)

    def set_centralize_params(self, matrix):
        import sys
        from math import sqrt
        mu_v = [ 0.0 for _ in xrange(len(matrix[0])) ]
        for i in xrange(len(matrix)):
            vector = matrix[i]
            mu_v = [ x[0] + x[1] for x in zip(vector, mu_v) ]

        mu_v = [ x / len(matrix) for x in mu_v ]
        self.mu_v = mu_v

        sigma_agg = [ 0.0 for _ in xrange(len(matrix[0])) ]
        for vector in matrix:
            for i in xrange(len(vector)):
                sigma_agg[i] += (vector[i] - mu_v[i])**2

        sigma_v = [ sqrt(agg / len(matrix)) for agg in sigma_agg ]
        self.sigma_v = map(lambda x : x if x != 0 else sys.float_info.epsilon, sigma_v)

    def centralize(self, vector):
        return [ (vector[i] - self.mu_v[i]) / self.sigma_v[i] for i in xrange(len(vector)) ]

    def set_pca_params(self, matrix, components):
        from sklearn.decomposition import PCA
        analizer = PCA(n_components=components)
        analizer.fit(matrix)
        self.pca = lambda x : analizer.transform([x])[0]