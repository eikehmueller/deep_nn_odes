"""Train a simple Multilayer Perceptron (MLP) model to fit a set of points uniformly scattered
in the unit square

All points that lie within a circle of radius 0.3 centred at (0.5,0.5) belong to one category,
whereas all points outside this circle belong to the other category.

For simplicity, I haven't been careful with splitting the data into training/validation/tests 
sets - the model is fitted to the entire data.

The code requires JAX (https://github.com/google/jax) and optax (https://github.com/deepmind/optax).

"""

import numpy as np
import jax
import optax
import matplotlib as mpl
from matplotlib import pyplot as plt

# Seed for random number generator (to make results reproducible)
seed = 2151517
# Random number generator object
rng = np.random.default_rng(seed)

# Number of data points
n_data = 1024
# Batch size
batch_size = 64
# Number of epochs for training
nepoch = 1000
# Learning rate
learning_rate = 1.0e-3
# Layer widths of the MLP network.
# The first layer has size 2, since the input of the NN is a 2d point.
# The final layer has size 2, since the network returns the probabilities of the two
# categories, e.g. (0.72, 0.28)
layer_widths = (2, 16, 16, 2)

# ==== Create the data ====
# input to model = array of shape n_data x 2, containing 2d position vectors
# that are uniformly distributed over the unit square.
X = rng.uniform(low=0, high=1, size=(n_data, 2))
# labels = array of shape n_data x 2, containing one-hot encodings. All points that
# lie within a circle of radius 0.3 centred at (0.5,0.5) have the one-hot label [1,0],
# all points outside this circle have the one-hot label [0,1]
y = np.eye(2)[1 * ((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2 < 0.3**2)]


def init_parameters(layer_widths):
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


def model(params, x):
    """MLP model for classification. This model takes as input a two-dimensional point
    and outputs the logits, which can then be used for classification

    The final layer returns logits which can be used in the optax.softmax_cross_entropy loss

    :arg params: model parameters, as created with init_mlp_parameters()
    :arg x: input state, tensor of shape (n_data,d)

    returns logits
    """
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.sigmoid(x @ layer["weights"] + layer["biases"])
    return x @ last["weights"] + last["biases"]


def loss(params, model, x, y):
    """Cross-entropy loss function

    :arg x: model inputs (points in 2d)
    :arg y: targets (one-hot encodings of probabilities)
    """
    logits = model(params, x)
    return optax.softmax_cross_entropy(logits, y).mean()


params = init_parameters(layer_widths)
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)


@jax.jit
def step(params, opt_state, X_batch, y_batch):
    """Single optimisation step

    :arg params: model parameters
    :arg opt_state: state of the optimizer
    :arg X_batch: input minibatch
    :arg y_batch: target minibatch (one-hot encodings)
    """
    loss_value, grads = jax.value_and_grad(loss)(params, model, X_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


# ==== Train network ====
for epoch in range(nepoch):
    # iterate over all batches
    for j, (X_batch, y_batch) in enumerate(
        zip(np.split(X, n_data // batch_size), np.split(y, n_data // batch_size))
    ):
        params, opt_state, loss_value = step(params, opt_state, X_batch, y_batch)
    print(f"epoch {epoch:4d}: loss = {loss_value:8.3e}")

# ==== Plotting ====
# Create scatter plot of the labelled data points and save to the file "logistic.pdf".
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap=mpl.colormaps["bwr"])

# Create a contour plot of the model function
nx = ny = 100
x = np.linspace(np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1, nx)
y = np.linspace(np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1, ny)
XY = np.stack(np.meshgrid(x, y), axis=-1)

Z = jax.nn.softmax(model(params, XY))[..., 0]
plt.contourf(XY[..., 0], XY[..., 1], Z, cmap=mpl.colormaps["Greys"], alpha=0.7)
plt.colorbar()
plt.savefig("logistic.pdf", bbox_inches="tight")
