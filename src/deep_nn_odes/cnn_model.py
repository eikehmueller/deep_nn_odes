"""CNN for image classification"""

import jax
from jax import numpy as jnp
from jax import random
from jax import lax
from flax.linen import avg_pool


def init_cnn_parameters(seed, input_channels=3):
    """Initialise parameters for simple CNN

    :arg key: key for random number generation
    :arg input_channels: number C of input channels

      The parameters have the following shapes for the weight matrices and biases:
      Convolution 0: (3,3,C,16) and (16,)    # 144*C + 16
      Convolution 1: (3,3,16,16)   and (16,)    # 2320
      MaxPool2D
      Convolution 2: (3,3,16,32)   and (32,)    # 4640
      Convolution 3: (3,3,32,32)   and (32,)    # 9248
      MaxPool2D
      Convolution 4: (3,3,32,64)   and (64,)    # 18496
      Convolution 5: (3,3,64,64)   and (64,)    # 36928
      MaxPool2D -> size is 4x4x64 = 1024 for 32x32 input images
      Dense Relu: (1024,64)        and (64,)    # 65600
      Classification: (64,10)      and (10,)    # 650
                          TOTAL (for C = 3)  : 175258 parameters
    """

    from collections import defaultdict

    key = random.PRNGKey(seed)
    params = defaultdict(list)
    # weights and biases for CNN layers
    for C_in, C_out in (
        (input_channels, 16),
        (16, 16),
        (16, 32),
        (32, 32),
        (32, 64),
        (64, 64),
    ):
        scale = jnp.sqrt(2 / C_in)
        key, subkey = random.split(key)
        params["cnn_weights"].append(
            scale * random.normal(subkey, (3, 3, C_in, C_out), dtype=jnp.float32)
        )
        params["cnn_biases"].append(jnp.ones((C_out,), dtype=jnp.float32))
    # weights and biases of intermediate dense layer
    scale = jnp.sqrt(2 / 1024)
    key, subkey = random.split(key)
    params["dense_weights"] = scale * random.normal(
        subkey, (1024, 64), dtype=jnp.float32
    )
    params["dense_biases"] = jnp.ones((64,), dtype=jnp.float32)
    # weights and biases of dense classification layer
    key, subkey = random.split(key)
    scale = jnp.sqrt(2 / 64)
    params["classification_weights"] = scale * random.normal(
        subkey, (64, 10), dtype=jnp.float32
    )
    params["classification_biases"] = jnp.ones((10,), dtype=jnp.float32)
    return params


def cnn_model(params, x):
    """Simple CNN model

    The model consists of the following layers:

    CNN Superlayer 0
      Convolution 0, kernelsize = (3,3) output channels = 16
      RELU nonlinearity
      Convolution 1, kernelsize = (3,3) output channels = 16
      RELU nonlinearity
      MaxPool2D, kernelsize (2,2) and stride (2,2)
    CNN Superlayer 1
      Convolution 2, kernelsize = (3,3) output channels = 32
      RELU nonlinearity
      Convolution 3, kernelsize = (3,3) output channels = 32
      RELU nonlinearity
      MaxPool2D, kernelsize (2,2) and stride (2,2)
    CNN Superlayer 2
      Convolution 4, kernelsize = (3,3) output channels = 64
      RELU nonlinearity
      Convolution 5, kernelsize = (3,3) output channels = 64
      RELU nonlinearity
      MaxPool2D, kernelsize (2,2) and stride (2,2)
    Dense layers
      Flattening from (4,4,64) to 1024 (for 32x32 input images)
      Dense RELU layer from 1024 to 64
      Dense layer from 64 to 10 to compute logits
    """
    batch_size = x.shape[0]
    # CNN layers
    for super_layer in range(3):
        for sub_layer in range(2):
            x = (
                lax.conv_general_dilated(
                    x,
                    params["cnn_weights"][2 * super_layer + sub_layer],
                    (1, 1),
                    "SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                )
                + params["cnn_biases"][2 * super_layer + sub_layer]
            )
            x = jax.nn.relu(x)
        x = avg_pool(x, (2, 2), strides=(2, 2))

    # Dense layers
    x = jnp.reshape(x, (batch_size, -1))
    # RELU layer
    x = jax.nn.relu(x @ params["dense_weights"] + params["dense_biases"])
    # Compute logits
    return x @ params["classification_weights"] + params["classification_biases"]
