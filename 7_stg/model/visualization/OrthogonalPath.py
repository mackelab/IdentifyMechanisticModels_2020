import numpy as np
from scipy import optimize
from BasePath import BasePath
from utils import inv_logistic_fct


class OrthogonalPath(BasePath):

    def __init__(self, high_prob_path, start_point_ind, L=50, pdf=None, params_mean=None, params_std=None):
        """
        Initialize the orthogonal path

        Inputs:
        -------
        high_prob_path: list of lists. Each list contains coordinates of the point on a path
        start_point_ind: scalar, int. Index of starting point
        """
        super().__init__(L=L, pdf=pdf)
        self.high_prob_path = high_prob_path
        self.start_point_ind = start_point_ind
        self.start_point = self.high_prob_path[self.start_point_ind]
        #self.start_point[-6] = self.start_point[-6] - 0.01
        self.num_dim = len(self.start_point)
        self.params_mean = params_mean
        self.params_std = params_std
        self.n = None

    def set_start_point(self, start_point_ind):
        """
        Set the starting point for the orthgonal path

        Inputs:
        -------
        start_point_ind: scalar, int. Index of starting point
        """
        self.start_point_ind = start_point_ind
        self.start_point = self.high_prob_path[self.start_point_ind]

    def set_high_prob_path(self, high_prob_path):
        """
        Set the high probability path that we are walking orthogonally to

        Inputs:
        -------
        high_prob_path: list of lists. Each list contains coordinates of the point on a path
        """
        self.high_prob_path = high_prob_path
        self.start_point = self.high_prob_path[self.start_point_ind]

    def get_orthonormal_vector(self):
        """
        Get the orthonormal vector to the disk that is orthogonal to the high probability path
        """
        next_point = self.high_prob_path[self.start_point_ind + 1]
        prev_point = self.high_prob_path[self.start_point_ind - 1]
        self.n = (next_point - prev_point) / np.linalg.norm(next_point - prev_point)

    def eval_log_grad(self, pdf, point, prior, tf):
        """
        Evaluate gradient of the log-likelihood

        Inputs:
        -------
        pdf: delfi.distribution.MoG object
        point: list, coordinates of point where the gradient is evaluated

        Returns:
        --------
        grad: list, gradients of log-likelihood
        """
        point = np.asarray(point)
        grad = optimize.approx_fprime(point, self.eval_log_pdf, 1e-4, pdf, prior, tf)
        return grad


    def eval_log_pdf(self, point, pdf, prior, tf):
        """
        Evaluate the log-likelihood

        Inputs:
        -------
        pdf: delfi.distribution.MoG object
        point: list, coordinates of point where the gradient is evaluated

        Returns:
        --------
        scalar, log-likelihood
        """
        if self.dist_type == 'mog':   return pdf.eval([point])
        elif self.dist_type == 'maf':
            if tf:
                mu1 = prior.lower
                sigma1 = prior.upper - prior.lower
                mu2 = self.params_mean
                sigma2 = self.params_std

                #term1 = 1 / np.prod(sigma1) / np.prod(sigma2)
                #a = inv_logistic_fct((point - mu1) / sigma1)
                #b = (point - mu1) / sigma1
                #term2 = np.exp(pdf.eval((a - mu2) / sigma2))
                #term3 = np.prod(np.abs(1 / (b - b ** 2)))
                term1 = 1 / np.prod(sigma2)
                a = inv_logistic_fct(point)
                b = point
                term2 = np.exp(pdf.eval((a - mu2) / sigma2))
                term3 = np.prod(np.abs(1 / (b - b ** 2)))

                p = np.log(term1 * term2 * term3)
            else:
                p = pdf.eval(point)

            return p


    def find_orthogonal_path(self, pdf, prior, dim=None, L=None, max_distance=None, tf=False):
        """
        Gind orthogonal path using gradient projection method

        Inputs:
        -------
        pdf: delfi.distribution.MoG object
        L: scalar, int, If provided, number of steps the path consists of
        max_distance: scalar, double. If provided, the path walks until its length exceeds max_distance
        """

        #TODO: when getting out of prior range, stop and terminate optimization

        self.set_pdf(pdf, dim)

        if L is not None:
            self.L = L
        current_point = self.start_point.tolist()
        test_if_bounded = np.ones_like(current_point, dtype=bool)
        self.path_coords.append(current_point)

        self.get_orthonormal_vector()
        learning_rate = 1e-2

        if L is not None:
            if max_distance is not None: assert "Only L or max_distance can be not None."
            for _ in range(self.L-1):
                current_point, travelled_dist, test_if_bounded, finished = \
                    self.run_gradient_step(pdf, current_point, prior, test_if_bounded, learning_rate, tf)
                if finished: break
            self.path_coords = np.asarray(self.path_coords)

        elif max_distance is not None:
            current_dist = 0
            while current_dist < max_distance:
                current_point, travelled_dist, test_if_bounded, finished = \
                    self.run_gradient_step(pdf, current_point, prior, test_if_bounded, learning_rate, tf)
                if finished: break
                current_dist += travelled_dist# / np.log(10)
            self.L = len(self.path_coords)
            self.path_coords = np.asarray(self.path_coords)

        else:
            assert "Provide either a length of the path or a maximal distance for the path"


    def run_gradient_step(self, pdf, current_point, prior, test_if_bounded, learning_rate, tf):
        """
        Run a single gradient step

        :param pdf: dd.distribution
        :param current_point: list or array
        :param prior: dd.distribution
        :param test_if_bounded: list of bools
        :param learning_rate: int
        :return: new point, distance travelled during step, bounded?, local minimum found?
        """
        finished = False
        grad = self.eval_log_grad(pdf, current_point, prior, tf)
        grad = np.asarray(grad)
        grad[np.invert(test_if_bounded)] = 0.0
        grad[np.isnan(grad)] = 0.0

        P = np.diag(np.ones(self.num_dim)) - 1 / np.dot(self.n, self.n) \
            * np.dot(np.transpose([self.n]), [self.n])
        P_grad = np.dot(P, grad)
        beta = 0.5 * np.sqrt(np.dot(grad, P_grad))

        if beta == 0:
            print("Found local minimum")
            finished=True
        else:
            d_bar = -P_grad / np.sqrt(np.dot(grad, P_grad))

        current_point += learning_rate * d_bar
        test_if_bounded = np.logical_and(prior.lower < current_point, prior.upper > current_point)
        current_point[prior.lower > current_point] = prior.lower[prior.lower > current_point] + 1e-4
        current_point[prior.upper < current_point] = prior.upper[prior.upper < current_point] - 1e-4

        self.path_coords.append(current_point.tolist())
        travelled_dist = np.linalg.norm(learning_rate * d_bar)

        return current_point, travelled_dist, test_if_bounded, finished