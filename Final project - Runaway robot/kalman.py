import numpy as np


class Kalman:
    """
    Kalman filter for estimating location on a 2D plane.
    Location predictions are made in 2 positional dimensions: x and y (observed).
    Velocities dx and dy and optional further derivatives are unobserved,
    but included as dimensions in the model.

    Args:
        loc (array of int or float): array that contains the initial location
        estimates x and y, followed by any number of derivative terms.
        The estimates for derivatives may be set to zeros, for example
        `[x, y, 0, 0]` for a model with x, y, dx and dy.
        The length of this array will dictate the dimensionality of the
        Kalman filter. The above example will result in a 4D model.

    Optional args:
        var (array of int or float): initial variance estimate for location
            of size `dim`, where `dim` is the number of dimensions in the
            model. Default is 1000.
            error).
        dt (int or float): time step between states. Default is 1.
        motion_err (array of int or float): displacement effect by an unknown
            external source, such as external motion. Default is zeros, size
            is `dim`.
        sense_err (array of int or float): measurement noise. Default is
            [0.1, 0.1], size is `m` where `m` is the number of measurements per
            time point. `m` = 2 for the 2D case, as x and y are observed,
            but not dx and dy.
        covar_transition (int or float): uncertainty of the state prediction.
    """      

    def __init__(self, loc, **kwargs):
        # number of model dimensions
        dim = len(loc)

        # location estimate
        self.w = np.array(loc).astype(float).reshape((dim, 1))

        # variance estimate (variances along the diagonal)
        self.P = np.zeros((dim, dim))
        for i, v in enumerate(kwargs.get('var', [1000] * dim)):
            self.P[i, i] = v
        
        # state transition matrix
        self.F = np.zeros((dim, dim))
        dt = kwargs.get('dt', 1.)
        # set all dt diagonals for the derivative terms. 
        for i in range(dim):
            for j in range(0, dim, 2):
                if (i + j) < dim:
                    self.F[i, (i + j)] = dt**(j / 2)

        # covariance matrix for the state transition matrix gives
        # the uncertainty of the prediction
        self.Q = kwargs.get('covar_transition', 0.001) * np.dot(self.F, self.F.T)

        # Extraction matrix (measurement function) maps states to
        # measurements. Size m x dim. Here, m = 2 because x and y
        # positions are measured, but not velocity.
        self.H = np.zeros((2, dim))
        self.H[0, 0] = 1
        self.H[1, 1] = 1

        # motion error (external motion)
        self.u = np.array(kwargs.get('motion_err', [0.] * dim)).reshape((dim, 1))

        # measurement uncertainty
        self.R = np.eye(2) * kwargs.get('sense_err', [0.1, 0.1])


    def predict(self):
        """
        Predict next state.
        """
        # location update
        self.w = np.dot(self.F, self.w) + self.u

        # variance update
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement):
        """
        Update the estimates of the current state after measurement.

        Args:
            measurement (array of int or float): measured positions x and y.
        """
        Z = np.array(measurement).reshape((-1, 1))

        # prediction error with respect to measurement
        y = Z - np.dot(self.H, self.w)

        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S.astype(float)))

        I = np.eye(self.P.shape[0])

        # update estimates
        self.w = self.w + np.dot(K, y)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def filter(self, measurement):
        """
        Perform one Kalman filter step with a measurement.

        Args:
            measurement (array of int or float): measured positions x and y.

        Returns:
            w (array of float): location estimate.
        """
        self.predict()
        self.update(measurement)

        return self.w

    def get_loc(self):
        return self.w


class ExtendedKalman(Kalman):

    def __init__(self, **kwargs):
        super().__init__(loc=[0, 0, 0, 0, 0], **kwargs)

        # overwrite linear state transition matrix
        # with the non-linear state transition function
        self.F = self.get_state_transition_matrix()

        # initialise the Jacobian2
        self.F_jac = self.get_state_transition_jacobian()

    def get_state_transition_matrix(self):
        # state estimation vector w has 5 values:
        # [x, y, heading, rotation, distance]
        x, y, h, r, d = self.w.flatten()

        # state transition matrix
        F = np.array([
            [1, 0, (-d * np.cos(h) + d * np.cos(r + h)), (d * np.cos(r + h)), (-np.sin(h) + np.sin(h + r))],
            [0, 1, (-d * np.sin(h) - d * np.sin(r - h)), (d * np.sin(r - h)), (np.cos(h) - np.cos(h - r))],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])

        return F
        
    def get_state_transition_jacobian(self):
        # state estimation vector w has 5 values:
        # [x, y, heading, rotation, distance]
        x, y, h, r, d = self.w.flatten()

        # partial derivatives of state transition matrix
        F_jac = np.array([
            x - d * np.sin(h) + d * np.sin(h + r),    # dx
            y + d * np.cos(h) - d * np.cos(h - r),    # dy
            h + r,                                    # dh
            r,                                        # dr
            d                                         # dd
        ]).reshape((-1, 1))

        return F_jac

    def predict(self):
        # get Jacobian of the state transition function
        self.F_jac = self.get_state_transition_jacobian()

        # location update
        self.w = self.F_jac + self.u

        # get state transition matrix
        self.F = self.get_state_transition_matrix()

        # variance update
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
