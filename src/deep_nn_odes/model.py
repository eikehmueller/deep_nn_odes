import numpy as np
import jax
from jax import numpy as jnp


def init_mlp_parameters(layer_widths):
    """Initialise parameters of MLP model

    :arg layer_widths: List with number of nodes in each layer, e.g. (2,16,32,3)
    """
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(
                weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2 / n_in),
                biases=np.ones(shape=(n_out,)),
            )
        )
    return params


def mlp_model(params, x):
    """MLP model for classification

    The final layer returns logits which can be used in the optax.softmax_cross_entropy loss

    :arg params: model parameters, as created with init_mlp_parameters()
    :arg x: input state, tensor of shape (n_data,d)

    returns logits
    """
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.sigmoid(x @ layer["weights"] + layer["biases"])
    return x @ last["weights"] + last["biases"]


def init_hamiltonian_parameters(dim, n_class, n_steps=8):
    """Initialise parameters of Hamiltonian model

    Returns a dictionary with weights and biases for the following components:

    * (K_j,b_j) for the j-the Verlet update step,
        where j=0,1,...,n-1; the corresponding keys in the dictionary are "K" and "b";
        K_j are d x d matrices and b_j are scalars.
    * (K,b) for the final classification step, the corresponding key
        is "classification". K is a n_c x d matrix and b is a n_c vector.

    :arg dim: dimension d
    :arg n_class: number of classes n_c
    """
    params = dict(
        K=np.random.normal(size=(n_steps, dim, dim)) * np.sqrt(2 / dim),
        b=np.ones(shape=(n_steps,)),
        classification=dict(
            weights=np.random.normal(size=(dim, n_class)) * np.sqrt(2 / dim),
            biases=np.ones(shape=(n_class,)),
        ),
    )
    return params


def hamiltonian_model(params, x, n_steps=8):
    """Hamiltonian model for classification

    starting with x_0 = x, evolve the state forward in
    time according to the Verlet integration

        p_0 = - h/2*K.sigmoid(x_0.K_0+b)

    and

        x_{j+1} = x_j + h*K.sigmoid(p_j.K_j+b_j)
        p_{j+1} = p_j - h*K.sigmoid(x_j.K_{j+1}+b_{j+1})

    for j = 0,...,n-1.
    (note that this assumes that the initial momentum is zero)

    The logits are obtained from x_n as

        logits = x_n.K_{class} + b_{class}

    :arg params: model parameters, as created with init_mlp_parameters()
    :arg x: input state, tensor of shape (n_data,d)
    :arg n_steps: number of steps

    returns logits
    """
    h = 20.0 / n_steps
    K, b = params["K"], params["b"]
    activation = jax.nn.sigmoid
    p = -0.5 * h * activation(x @ K[0, :, :] + b[0]) @ K[0, :, :].T
    for j in range(n_steps):
        x += h * activation(p @ K[j, :, :] + b[j]) @ K[j, :, :].T
        if j < n_steps - 1:
            p -= h * activation(x @ K[j + 1, :, :] + b[j + 1]) @ K[j + 1, :, :].T
    K, b = (
        params["classification"]["weights"],
        params["classification"]["biases"],
    )
    return p @ K + b


def hamiltonian_regulariser(params, alpha=1.0e-3):
    """Regulariser for Hamiltonian formulation

    Returns the sum R(K) + r(b) where

        R(K) = 1/(2*h) sum_{j=1}^{n-1} ||K_j-K_{j-1}||_F^2

    with the Frobenius norm

        ||A||_F = ( sum_{a,b=0}^{d-1} K_{a,b}^2 )^{1/2}

    and

        r(b) = alpha/(2*h) sum_{j=1}^{n-1} |b_j-b_{j-1}|^2

    h = 1/n_steps is the stepsize of the Verlet integrator.

    :arg params: model parameters, as created by init_hamiltonian_parameters()
    :arg alpha: scaling factor
    """
    n_steps = float(params["K"].shape[0])
    h = 20.0 / n_steps
    R = lambda A: 0.5 * h * jnp.sum((A[1:, ...] - A[:-1, ...]) ** 2)
    return alpha * (R(params["K"]) + R(params["b"]))
