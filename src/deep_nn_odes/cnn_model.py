"""ResNet based CNN model for image classification

Implements simplified ResNet model for classification on small image datasets
duch as MNIST or CIFAR10.

The input is an array of the shape (n_images,32,32,C), where each image has width h_x 
and height n_y and C channels (=colours).

The model output are the logits of the class probabilities, i.e. an array of shape
(n_images,) where each entry in an integer number 0,1,...,M-1 and M is the number of
categories.
"""

from functools import partial
import jax
from jax import numpy as jnp
from jax import random
from jax import lax
from flax.linen import max_pool, avg_pool


def dropout(input, state, p_dropout):
    """Apply dropout to input and return result.

    The behaviour of this layer is controlled by the 'train' flag in the state
    dictionary: during training, a fraction of p_dropout nodes is set to zero and the
    output is scaled by (1-p_dropout)^{-1}. Otherwise (i.e. during inference),
    the input is returned unchanged.

    :arg input: neurons to which the dropout is to be applied
    :arg state: model state
    :arg p_dropout: dropout probability
    """

    # extract key for random number generation
    state["rngkey"], subkey = random.split(state["rngkey"])

    def apply_dropout(x):
        """Implementation of the actual dropout function.

        Multiply each element of the input array by

            1/p_keep*Bernoulli(p_keep)

        with p_keep = 1-p_dropout being the probability of *not* dropping the node

        :arg x: input array
        """
        p_keep = 1.0 - p_dropout
        mask = random.bernoulli(subkey, p=p_keep, shape=x.shape)
        return jax.lax.select(mask, x / p_keep, jnp.zeros_like(x))

    # using lax.cond avoids if-statements which can not be jit-ed
    return jax.lax.cond(
        state["train"],
        apply_dropout,  # apply dropout during training...
        lambda x: x,  # ...otherwise just return the unmodified input
        input,
    )


def init_residualblock_parameters(key, input_channels, n_conv=2):
    """Initialise parameters for the convolutions in a single
    residual block.

    :arg key: key for random number generator
    :arg input_channels: number of input channels
    :arg n_conv: number of convolutions
    """
    # The number of channels does not change
    C_ins = n_conv * [input_channels]
    C_outs = n_conv * [input_channels]
    params = dict(weights=[], biases=[])
    for C_in, C_out in zip(C_ins, C_outs):
        key, subkey = random.split(key)
        scale = jnp.sqrt(2 / C_in)
        params["weights"].append(
            random.uniform(
                subkey,
                (3, 3, C_in, C_out),
                minval=-scale,
                maxval=+scale,
                dtype=jnp.float32,
            )
        )
        params["biases"].append(jnp.ones((C_out,), dtype=jnp.float32))
    return params


def init_downsampling_residualblock_parameters(key, input_channels, n_conv=2):
    """Initialise parameters for the convolutions and the projection in a single
    residual block with downsampling

    :arg key: key for random number generator
    :arg input_channels: number of input channels
    :arg n_conv: number of convolutions
    """
    # The subsequent layers have twice the number of channels as the first layer
    C_ins = [input_channels] + (n_conv - 1) * [2 * input_channels]
    C_outs = n_conv * [2 * input_channels]
    params = dict(weights=[], biases=[])
    # Convolutions
    for C_in, C_out in zip(C_ins, C_outs):
        scale = jnp.sqrt(2 / C_in)
        key, subkey = random.split(key)
        params["weights"].append(
            random.uniform(
                subkey,
                (3, 3, C_in, C_out),
                minval=-scale,
                maxval=+scale,
                dtype=jnp.float32,
            )
        )
        params["biases"].append(jnp.ones((C_out,), dtype=jnp.float32))
    # Projection
    scale = jnp.sqrt(2 / input_channels)
    params["projection_weights"] = random.uniform(
        subkey,
        (1, 1, input_channels, 2 * input_channels),
        minval=-scale,
        maxval=+scale,
        dtype=jnp.float32,
    )
    return params


def init_resnet_parameters(input_channels=3, n_categories=10):
    """Initialise parameters of the entire ResNet model

    :arg input_channels: number of input channels
    :arg n_categories: number of categories
    """
    from collections import defaultdict

    seed = 47
    key = random.PRNGKey(seed)
    params = defaultdict(list)
    # Initial 5x5 convolution with 8 output channels
    scale = jnp.sqrt(2 / input_channels)
    key, subkey = random.split(key)
    params["initial_conv_weights"] = random.uniform(
        subkey,
        (5, 5, input_channels, 16),
        minval=-scale,
        maxval=scale,
        dtype=jnp.float32,
    )
    params["initial_conv_biases"] = jnp.ones((16,), dtype=jnp.float32)
    # 5 residual blocks on 16x16 image with 16 channels
    params["residualblocks_16x16x16"] = []
    for _ in range(5):
        key, subkey = random.split(key)
        params["residualblocks_16x16x16"].append(
            init_residualblock_parameters(subkey, input_channels=16, n_conv=2)
        )
    # Downsampling residual block:
    #     16x16 image with 16 channels -> 8x8 image with 32 channels
    key, subkey = random.split(key)
    params["downsampling_residualblock"] = init_downsampling_residualblock_parameters(
        key, input_channels=16, n_conv=2
    )
    # 5 residual blocks on 8x8 image with 32 channels
    params["residualblocks_8x8x32"] = []
    for _ in range(5):
        key, subkey = random.split(key)
        params["residualblocks_8x8x32"].append(
            init_residualblock_parameters(key, input_channels=32, n_conv=2)
        )
    # Classification block (32 nodes -> n_categories nodes)
    C_in, C_out = (32, n_categories)
    key, subkey = random.split(key)
    scale = jnp.sqrt(2 / C_in)
    params["classification_weights"] = random.uniform(
        subkey,
        (C_in, C_out),
        minval=-scale,
        maxval=scale,
        dtype=jnp.float32,
    )
    params["classification_biases"] = jnp.ones(
        (C_out,),
        dtype=jnp.float32,
    )
    return params


def residual_block(params, x):
    """Apply a single residual block

    :arg params: parameters of the residual block (= weights and biases of the convolutions)
    :arg x: input
    """
    z = x  # copy state before block
    for weights, biases in zip(params["weights"], params["biases"]):
        z = (
            lax.conv_general_dilated(
                z,
                weights,
                (1, 1),
                "SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            + biases
        )
        z = jax.nn.relu(z)
    return x + z


def downsampling_residual_block(params, state, x):
    """Apply a single downsampling residual block, including dropout

    :arg params: parameters of the residual block (= weights and biases of the convolutions
        and the projection matrix
    :arg state: model state, required for dropout
    :arg x: input array
    """
    z = x  # copy state before block
    # Maxpool layer
    z = max_pool(z, (2, 2), strides=(2, 2))
    z = dropout(z, state, 0.2)
    # Residual layers
    for weights, biases in zip(params["weights"], params["biases"]):
        z = (
            lax.conv_general_dilated(
                z,
                weights,
                (1, 1),
                "SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            + biases
        )
        z = jax.nn.relu(z)
    # Add projection of input state
    z += lax.conv_general_dilated(
        x,
        params["projection_weights"],
        (2, 2),
        "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return z


def resnet_model(params, state, x):
    """Apply simple ResNet model

    :arg params: model parameters
    :arg state: model state (required for dropout implementation)
    :arg x: input image
    """
    batch_size = x.shape[0]
    # Initial convolution with 5x5 kernel and 8 features
    x = (
        lax.conv_general_dilated(
            x,
            params["initial_conv_weights"],
            (1, 1),
            "SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["initial_conv_biases"]
    )
    # Initial downsampling
    x = max_pool(x, (2, 2), strides=(2, 2))

    # Residual blocks on 16x16 image with 16 channels
    for residual_block_params in params["residualblocks_16x16x16"]:
        x = residual_block(residual_block_params, x)
    # Downsampling residual block:
    #     16x16 image with 16 channels -> 8x8 image with 32 channels
    x = downsampling_residual_block(params["downsampling_residualblock"], state, x)
    # Residual blocks on 8x8 image with 32 channels
    for residual_block_params in params["residualblocks_8x8x32"]:
        x = residual_block(residual_block_params, x)
    # Global average pooling
    x = avg_pool(x, x.shape[1:3])
    # Flatten
    x = jnp.reshape(x, (batch_size, -1))
    x = dropout(x, state, 0.5)
    # Classification layer: compute logits
    return x @ params["classification_weights"] + params["classification_biases"]
