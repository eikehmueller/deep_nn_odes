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

    returns
    """
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.sigmoid(x @ layer["weights"] + layer["biases"])
    return x @ last["weights"] + last["biases"]
