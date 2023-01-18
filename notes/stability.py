import numpy as np
from scipy import integrate as spint
from matplotlib import pyplot as plt


def sigmoid(z):
    """sigmoid function 1/(1+e^{-z})

    :arg z: input argument z
    """
    return 1.0 / (1.0 + np.exp(-z))


class ODESystem:
    """Auxilliary class for integrating Hamiltonian or hyperbolic CNNs"""

    def __init__(self, dim):
        """Initialise new instance

        :arg dim: dimension d of system
        """
        self.dim = dim
        self.rng = np.random.default_rng(seed=21417)
        # weight matrix K
        self.weight = self.rng.uniform(low=-1.0, high=+1.0, size=(self.dim, self.dim))
        # bias vector b
        self.bias = self.rng.uniform(low=-1.0, high=+1.0, size=self.dim)
        # maximal eigenvalue Lambda of K K^T
        evals, _ = np.linalg.eig(self.weight @ self.weight.T)
        evals.sort()
        self.Lambda = evals[-1]


class HamiltonianSystem(ODESystem):
    """Auxilliary class for integrating Hamiltonian CNN

    dx/dt = + K sigma(K^T p + b)
    dp/dt = - K sigma(K^T x + b)
    """

    def __init__(self, dim):
        """Initialise new instance

        :arg dim: dimension d of system
        """
        super().__init__(dim)

    def call(self, y, t):
        """Evaluate RHS for a given input y = (x,p)

        :arg y: input vector (x,p) of length 2*d
        :arg t: time t
        """
        return np.concatenate(
            [
                +self.weight @ sigmoid(self.weight.T @ y[self.dim :] + self.bias),
                -self.weight @ sigmoid(self.weight.T @ y[: self.dim] + self.bias),
            ]
        )

    def bound(self, t):
        """Return bound function on deviation if ||y^epsilon(0) - y(0)||_2 < epsilon.

        The bound on ||y^epsilon(t)-y(t)||_2 is given by epsilon * C_1(t) with

            C_1(t) = sqrt( 2 ) * exp[2*L*Lambda*t]

        where L = 1/4 is the Lipschitz constant of the sigmoid function.

        This method returns the value of the function C_1(t)

        :arg t: time t
        """
        LLipschitz = 0.25
        return np.sqrt(2.0) * np.exp(2 * LLipschitz * self.Lambda * t)


class HyperbolicSystem(ODESystem):
    """Auxilliary class for integrating hyperbolic CNN

    dx/dt = + p
    dp/dt = - K sigma(K^T x + b)
    """

    def __init__(self, dim):
        """Initialise new instance

        :arg dim: dimension d of system
        """
        super().__init__(dim)

    def call(self, y, t):
        """Evaluate RHS for a given input y = (x,p)

        :arg y: input vector (x,p) of length 2*d
        :arg t: time t
        """
        return np.concatenate(
            [
                +y[self.dim :],
                -self.weight @ sigmoid(self.weight.T @ y[: self.dim] + self.bias),
            ]
        )

    def bound(self, t):
        """Return bound function on deviation if ||y^epsilon(0) - y(0)||_2 < epsilon.

        The bound on ||y^epsilon(t)-y(t)||_2 is given by epsilon * C_2(t) with

            C_2(t) = sqrt( 2 ) * exp[(1+L*Lambda)*t]

        where L = 1/4 is the Lipschitz constant of the sigmoid function.

        This method returns the value of the function C_1(t)

        :arg t: time t
        """
        LLipschitz = 0.25
        return np.sqrt(2.0) * np.exp((1 + LLipschitz * self.Lambda) * t)


rng = np.random.default_rng(seed=3219417)


def plot_trajectories(ode_system, epsilon, t_final, filename="trajectories.pdf"):
    t = np.arange(0, t_final, 1.0e-3)
    y0 = rng.normal(size=2 * dim)
    dy = rng.normal(size=2 * dim)
    dy /= np.linalg.norm(dy)
    y = spint.odeint(ode_system.call, y0, t)
    plt.clf()
    plt.plot(y[:, 0], y[:, 1], color="blue", label="true")
    y_epsilon = spint.odeint(ode_system.call, y0 + epsilon * dy, t)
    plt.plot(y_epsilon[:, 0], y_epsilon[:, 1], color="red", label="perturbed")
    plt.legend(loc="upper left")
    plt.savefig(filename, bbox_inches="tight")


def plot_error(dim, t_final, filename="error.pdf"):
    log_eps = np.arange(-10, 0, 0.1)
    eps = np.exp(log_eps)
    error = np.empty(len(eps))
    ode_systems = {
        "Hamiltonian": HamiltonianSystem(dim),
        "hyperbolic": HyperbolicSystem(dim),
    }
    plt.clf()
    for label, ode_system in ode_systems.items():
        for j, epsilon in enumerate(eps):
            t = [0, t_final]
            y0 = rng.normal(size=2 * dim)
            dy = rng.normal(size=2 * dim)
            dy /= np.linalg.norm(dy)
            y = spint.odeint(ode_system.call, y0, t)
            y_epsilon = spint.odeint(ode_system.call, y0 + epsilon * dy, t)
            error[j] = np.linalg.norm(y_epsilon[-1, :] - y[-1, :])
        plt.plot(eps, error, label=label)
        ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$||y^\varepsilon(0)-y(0)||_2 = \varepsilon$")
    ax.set_ylabel(r"$||y^\varepsilon(t)-y(t)||_2$")
    plt.legend(loc="upper left")
    plt.savefig(filename, bbox_inches="tight")


dim = 4
epsilon = 1.0e-4
t_final = 10.0
ode_system = HyperbolicSystem(dim)
plot_trajectories(ode_system, epsilon, t_final)
plot_error(dim, t_final)
