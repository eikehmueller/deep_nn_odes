from functools import partial
import numpy as np
from jax import numpy as jnp
import jax
import optax
from matplotlib import pyplot as plt
from matplotlib import cm
from deep_nn_odes.data_generator import (
    DataGeneratorCircular,
    DataGeneratorMoons,
    DataGeneratorSwissRoll,
)
from deep_nn_odes.model import (
    init_mlp_parameters,
    mlp_model,
    init_hamiltonian_parameters,
    hamiltonian_model,
    hamiltonian_regulariser,
)


def fit(model, params, optimizer, data_generator, regulariser=None, nepoch=10):
    """Fit the MLP classifier using cross-entropy loss

    :arg model: the model to fit
    :arg params: fit parameters
    :arg optimizer: optimizer
    :arg data_generator: data generator
    :arg regulariser: regulariser function, which is passed params as sole parameter
    :arg nepoch: number of epochs
    """
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(768123)

    def loss(params, model, x, y):
        """Cross-entropy loss function"""
        logits = model(params, x)
        loss_value = optax.softmax_cross_entropy(logits, y).mean()
        if regulariser is not None:
            loss_value += regulariser(params)
        return loss_value

    @jax.jit
    def step(params, opt_state, X_batch, y_batch):
        loss_value, grads = jax.value_and_grad(loss)(params, model, X_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # Loop over epochs
    for epoch in range(nepoch):
        # iterate over all batches
        for j, (X_batch, y_batch) in enumerate(
            zip(*data_generator.get_shuffled_batched_train_data())
        ):
            params, opt_state, loss_value = step(params, opt_state, X_batch, y_batch)
            if j == 0:
                print(f"epoch {epoch:4d}: loss = {loss_value:8.3e}")
    return params


def visualise(data_generator, model, params, nx=64, ny=64, filename="fit.pdf"):
    """Visualise the data and save to file

    Create contour plot of binary classifier and overlay scatter plot of test data
    :arg data_generator: data generator for creating the test data
    :arg model: that was used for fitting
    :arg final fit parameters:
    :arg nx: number of points in the x-direction for countour plot
    :arg ny: number of points in the y-direction for countour plot
    :arg filename: name of file for plot
    """
    X_test, y_test = data_generator.get_test_data()
    x = np.linspace(np.min(X_test[:, 0]) - 0.1, np.max(X_test[:, 0]) + 0.1, nx)
    y = np.linspace(np.min(X_test[:, 1]) - 0.1, np.max(X_test[:, 1]) + 0.1, ny)
    XY = np.stack(np.meshgrid(x, y), axis=-1)

    Z = jax.nn.softmax(model(params, XY))[..., 0]
    cmap = cm.get_cmap("Greys")
    plt.contourf(XY[..., 0], XY[..., 1], Z, cmap=cmap, alpha=0.7)
    plt.colorbar()
    cmap = cm.get_cmap("bwr")
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test[:, 0],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=4,
    )
    plt.savefig(filename, bbox_inches="tight")


# Number of samples used for training
n_train = 16384
# Number of samples used for testing
n_test = 1024
# number of epochs
nepoch = 1000
# batch size
batchsize = 256
# Hyperparameters
learning_rate = 1.0e-2

# choose dataset
dataset_label = "swissroll"

# choose model
model_label = "Hamiltonian"

if dataset_label == "circular":
    radius = 0.25
    data_generator = DataGeneratorCircular(n_train, n_test, batchsize, radius=radius)
elif dataset_label == "moons":
    sigma = 0.2
    data_generator = DataGeneratorMoons(n_train, n_test, batchsize, noise=sigma)
elif dataset_label == "swissroll":
    sigma = 0.025
    data_generator = DataGeneratorSwissRoll(n_train, n_test, batchsize, noise=sigma)
else:
    raise Exception(f"unknown dataset: '{dataset_label}'")


optimizer = optax.adam(learning_rate)

if model_label == "MLP":
    layer_widths = (2, 16, 16, 2)
    params = init_mlp_parameters(layer_widths)
    model = mlp_model
    regulariser = None
elif model_label == "Hamiltonian":
    n_steps = 64
    alpha = 1.0e-3
    params = init_hamiltonian_parameters(2, 2, n_steps=n_steps)
    model = partial(hamiltonian_model, n_steps=n_steps)
    regulariser = partial(hamiltonian_regulariser, alpha=alpha)
    # regulariser = None
else:
    raise Exception(f"unknown dataset: '{dataset_label}'")

# Fit the model
params = fit(
    model, params, optimizer, data_generator, regulariser=regulariser, nepoch=nepoch
)

# Visualise fitted model
visualise(data_generator, model, params)
