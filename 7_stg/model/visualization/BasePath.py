import numpy as np
from find_Gauss_max import find_maximum


class BasePath:

    def __init__(self, L=50, pdf=None, dim=None):
        """
        Initialize base path

        :param L: scalar, int, length of path
        :param pdf: delfi.distribution.MoG object
        """
        self.L = L
        self.pdf = pdf
        self.path_coords = []
        self.path_probs = None
        self.dists = []
        self.num_dim = dim
        if pdf is not None:
            try:
                self.num_dim = len(pdf.xs[0].m)
                self.dist_type = 'mog'
            except:
                assert dim is not None, 'If you do not give a MoG, you have to specify dimensionsality'
                self.dist_type = 'maf'

    def set_length(self, L):
        """
        Set the length of the path

        :param L: scalar, int
        :return: scalar, int
        """
        self.L = L

    def set_pdf(self, pdf, dim=None):
        """
        Set the probability distribution the path is defined on

        :param pdf: delfi.distribution.MoG object
        :param dim: int, dimensionality of problem
        :return: delfi.distribution.MoG object
        """
        try:
            self.num_dim = len(pdf.xs[0].m)
            self.dist_type = 'mog'
        except:
            assert dim is not None, 'If you do not give a MoG, you have to specify dimensionsality'
            self.dist_type = 'maf'
            self.num_dim = dim
        self.pdf = pdf

    def get_probability_along_path(self, log=False, normalize=True):
        """
        Evaluate the probability of the path on the pdf

        :param log: bool. If True, we evaluate ratio of log-probabilities
        :return:probability of the path on the pdf
        """
        max_params = find_maximum(self.pdf, self.num_dim)
        if self.dist_type == 'mog':
            max_params = [max_params]
        if log:
            if normalize: max_value = self.pdf.eval(max_params)
            else: max_value = 0.0
            probabilites_along_path = self.pdf.eval(self.path_coords)
            graded_probabilites_along_path = probabilites_along_path - max_value
        else:
            if normalize: max_value = np.exp(self.pdf.eval(max_params))
            else: max_value = 1.0
            probabilites_along_path = np.exp(self.pdf.eval(self.path_coords))
            graded_probabilites_along_path = probabilites_along_path / max_value


        self.path_probs = graded_probabilites_along_path

    def get_travelled_distance(self):
        """
        Calculated the cumulated travelled distance

        :return: list, cumulated distances
        """
        dists = []
        overall_d = 0
        dists.append(0)
        for i in range(self.L-1):
            d = np.linalg.norm(self.path_coords[i+1] - self.path_coords[i])
            overall_d += d# / np.log(10)
            dists.append(overall_d)
        self.dists = dists

    def find_closest_index_to_dist(self, dists):
        """
        Calcualte the indizes of the path increment that are closest to the elements in dists

        :param dists: list, gives the distances of which we search the closest elements
        :return: list, contains the indices
        """
        indizes = []
        for dist in dists:
            index = np.argmin(np.abs(np.asarray(self.dists) - np.asarray(dist)))
            indizes.append(index)
        return indizes