from functools import partial
import numpy as np
from jax import numpy as jnp
import jax
import optax
from matplotlib import pyplot as plt
import matplotlib as mpl
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
)


def fit(model, params, optimizer, data_generator, nepoch=10):
    """Fit the MLP classifier using cross-entropy loss

    :arg model: the model to fit
    :arg params: fit parameters
    :arg optimizer: optimizer
    :arg data_generator: data generator
    :arg nepoch: number of epochs
    """
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(768123)

    def loss(params, model, x, y):
        """Cross-entropy loss function"""
        logits = model(params, x)
        return optax.softmax_cross_entropy(logits, y).mean()

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
    plt.contourf(XY[..., 0], XY[..., 1], Z, cmap=mpl.colormaps["Greys"], alpha=0.7)
    plt.colorbar()
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test[:, 0],
        cmap=mpl.colormaps["bwr"],
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
batchsize = 128
# Hyperparameters
learning_rate = 1.0e-3

# choose dataset
dataset_label = "swissroll"

# choose model
model_label = "MLP"

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
elif model_label == "Hamiltonian":
    n_steps = 16
    latent_dim = 8
    params = init_hamiltonian_parameters(2, latent_dim, 2)
    model = partial(hamiltonian_model, n_steps=n_steps)
else:
    raise Exception(f"unknown dataset: '{dataset_label}'")

# Fit the model
params = fit(model, params, optimizer, data_generator, nepoch=nepoch)

# Visualise fitted model
visualise(data_generator, model, params)
