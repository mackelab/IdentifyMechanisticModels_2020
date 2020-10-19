import numpy as np
import numpy.matlib
from scipy.optimize import minimize
from BasePath import BasePath
import delfi.distribution as dd
from find_pyloric import params_are_bounded
import numpy.random as nr

class HighProbabilityPath(BasePath):

    def __init__(self, N=3, L=50, pdf=None, dim=None, start_point=None, end_point=None, use_sine_square=False):
        """
        Inputs:
        -------
        N: int, number of basis functions
        L: int, number of timesteps
        start_point: list of dimension num_dim = len(pdf.xs[0].m). Coordinates of starting point.
        end_point: list of dimension num_dim = len(pdf.xs[0].m). Coordinates of end point.
        """
        super().__init__(L=L, pdf=pdf, dim=dim)
        self.N = N
        self.start_point = start_point
        self.end_point = end_point
        if self.start_point is not None:
            self.find_coord_trafo()
        else:
            self.R = None
            self.d = None
        self.velocity_coords = None
        self.alpha = None
        self.use_sine_square = use_sine_square


    ####################################################################################################################
    # Helper functions
    def set_start_end(self, start_point, end_point):
        """
        start_point: list of dimension num_dim = len(pdf.xs[0].m). Coordinates of starting point.
        end_point: list of dimension num_dim = len(pdf.xs[0].m). Coordinates of end point.
        """
        self.start_point = np.asarray(start_point)
        self.end_point   = np.asarray(end_point)
        self.num_dim = len(self.start_point)
        self.find_coord_trafo()


    ####################################################################################################################
    # Coordinate Transfrom
    def find_coord_trafo(self):
        """
        Calculate the coordinate transform x = f(gamma) = R * gamma + d
        """
        d = self.start_point
        R_diag = self.end_point - self.start_point
        R = np.diag(R_diag)
        self.R = R
        self.d = d


    def apply_coord_trafo(self, gamma):
        """
        Apply the coordinate transform x = f(gamma) = R * gamma + d

        Inputs:
        -------
        gamma: numpy array, path

        Returns:
        --------
        x_: numpy array, transformed path x = f(gamma)
        """
        d_ = np.matlib.repmat(self.d, len(gamma), 1)
        x_ = np.dot(gamma, self.R) + d_
        return x_


    ####################################################################################################################
    # Evaluating the path
    def evaluate_path(self, alpha_k, K, S):
        """
        Calculate the path

        Inputs:
        -------
        alpha_k: numpy array, of length (N x num_dim)
        K: int, number of time steps
        S: numpy array, from 0 to 1 with T steps

        Returns:
        --------
        sin_term: numpy array, path gamma
        """
        sin_term = np.zeros((self.L, self.num_dim))
        for dim in range(self.num_dim):
            for k in K:
                sin_term[:, dim] += alpha_k[k - 1][dim] * np.sin(np.pi * k * S)
            if self.use_sine_square:
                for k in K:
                    sin_term[:, dim] += alpha_k[K[-1] + k - 1][dim] * \
                                        (np.sin(np.pi * k * S)) ** 2
            sin_term[:, dim] += S
        return sin_term

    #  Evaluating the path derivative
    def evaluate_path_derivative(self, alpha_k, K, S):
        """
        Calculate the euclidean norm of second derivative of the normalized path

        Inputs:
        -------
        alpha_k: numpy array, of length (N x num_dim)
        K: int, number of time steps
        S: numpy array, from 0 to 1 with T steps

        Returns:
        --------
        sin_term: numpy array, euclidean norm of second derivative of normalized path gamma
        """
        sin_term = np.zeros((self.L, self.num_dim))

        for dim in range(self.num_dim):
            for k in K:
                sin_term[:, dim] += (k * np.pi * alpha_k[k - 1][dim] * np.cos(np.pi * k * S))
            if self.use_sine_square:
                for k in K:
                    sin_term[:, dim] += (k * np.pi * alpha_k[K[-1] + k - 1][dim] *
                                         np.cos(np.pi * k * S) *
                                         2 * np.sin(np.pi * k * S))
            sin_term[:, dim] += 1

        return sin_term


    ####################################################################################################################
    # Define error function
    def calculate_loss(self, alpha_k, pdf, K, S, prior, non_linearity=None, non_lin_param=2.0, multiply_posterior=1.0):
        """
        Calculate the loss

        Inputs:
        -------
        alpha_k: numpy array, of length (N x num_dim)
        pdf: delfi.distribution.MoG object
        K: int, number of time steps
        S: numpy array, from 0 to 1 with T steps
        prior: delfi.distribution object
        non_linearity: string. 'exp' or None
        non_lin_param: scalar, double, used only if non_linearity=='exp'

        Returns:
        --------
        L: scalar, loss
        """

        # coefficients for basis functions: alpha_k * sin(pi*k*s)
        if self.use_sine_square:
            alpha_k = np.reshape(alpha_k, (self.N * 2, self.num_dim))
        else:
            alpha_k = np.reshape(alpha_k, (self.N, self.num_dim))
        gamma = self.evaluate_path(alpha_k, K, S)

        # coordinate trafo
        x_ = self.apply_coord_trafo(gamma)

        # get cost as negtive log-posterior
        cost = -pdf.eval(x_)  # gives back the value in log space already
        cost = np.log(np.exp(cost / multiply_posterior))
        bounded = params_are_bounded(x_, prior, normalized=True)
        cost[not bounded] = 1e20

        # multiply by veclocity term
        cost *= np.linalg.norm(np.dot(self.evaluate_path_derivative(alpha_k, K, S), self.R), axis=1)

        # evaluate integral
        L = np.sum(cost)
        return L


    ####################################################################################################################
    # Full opimization procedure
    def find_path(self, pdf, prior, non_linearity=None, multiply_posterior=1.0, non_lin_param=2.0):
        """
        Calculate the optimal coefficients alpha_k and find the path coordinates (and the velocity)

        Inputs:
        -------
        pdf: delfi.distribution.MoG object
        prior: delfi.distribution object
        non_linearity: string. 'exp' or None
        non_lin_param: scalar, double
        """

        # line vector
        S = np.linspace(0.0, 1.0, self.L)
        K = np.arange(1, self.N+1)

        # coefficients for basis functions: alpha_k * sin(pi*k*s)
        if self.use_sine_square:
            alpha_k = np.zeros(self.num_dim * self.N * 2)
        else:
            alpha_k = np.zeros(self.num_dim * self.N)

        result = minimize(self.calculate_loss, alpha_k,
                          args=(pdf, K, S, prior, non_linearity, non_lin_param, multiply_posterior),
                          method="BFGS", options={'maxiter': 30}) # 30

        optimal_alpha = result.x
        if self.use_sine_square:
            self.alpha = np.reshape(optimal_alpha, (self.N * 2, self.num_dim))
        else:
            self.alpha = np.reshape(optimal_alpha, (self.N, self.num_dim))
        path_gamma = self.evaluate_path(self.alpha, K, S)

        # apply coordinate transformation
        self.path_coords = self.apply_coord_trafo(path_gamma)
        self.velocity_coords = np.dot(self.evaluate_path_derivative(self.alpha, K, S), self.R)