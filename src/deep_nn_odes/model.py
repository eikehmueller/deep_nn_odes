import numpy as np
import jax


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


def init_hamiltonian_parameters(dim, latent_dim, n_class):
    """Initialise parameters of Hamiltonian model

    :arg dim: dimension d
    :arg n_class: number of classes
    """
    params = []
    layer_widths = [dim, latent_dim, latent_dim, latent_dim, n_class]
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(
                weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2 / n_in),
                biases=np.ones(shape=(n_out,)),
            )
        )
    return params


def hamiltonian_model(params, x, n_steps=8):
    """Hamiltonian model for classification

    starting with x_0 = x, evolve the state forward in
    time according to the Verlet integration

        p_0 = h/2*K_p.sigmoid(x_0.K_p+b_p)

    and

        x_j = x_{j-1} - h*K_x.sigmoid(p_{j-1}.K_x+b_x)
        p_j = p_{j-1} + h*K_p.sigmoid(x_j.K_p+b_p)

    for j = 1,...,n.
    (note that this assumes that the initial momentum is zero)

    The logits are obtained from x_n as

        logits = x_n.K + b

    :arg params: model parameters, as created with init_mlp_parameters()
    :arg x: input state, tensor of shape (n_data,d)
    :arg n_steps: number of steps

    returns logits
    """
    h = 1.0 / n_steps
    first, layer_x, layer_p, last = params
    x = x @ first["weights"] + first["biases"]
    K_x, b_x = layer_x["weights"], layer_x["biases"]
    K_p, b_p = layer_p["weights"], layer_p["biases"]

    p = 0.5 * h * jax.nn.sigmoid(x @ K_p + b_p) @ K_p.T
    for _ in range(n_steps):
        x -= h * jax.nn.sigmoid(p @ K_x + b_x) @ K_x.T
        p += h * jax.nn.sigmoid(x @ K_p + b_p) @ K_p.T
    return x @ last["weights"] + last["biases"]
