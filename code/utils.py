from pathlib import Path
import time
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import jax
from jax import jit
import jax.numpy as jnp
import jax.random as jr
from functools import partial


# ------------------------------ FILE MANAGEMENT ------------------------------
def get_syllabus_path():
    return Path(__file__).resolve().parent / "syllabus.xlsx"


def get_name_data_path():
    return Path(__file__).resolve().parent / "name_data.npy"


def get_name_images_path():
    return Path(__file__).resolve().parent / "name_images.npy"


def get_name_image_example_path(name):
    image_example_path = Path(__file__).resolve().parent / "name_patterns"
    if not image_example_path.exists():
        image_example_path.mkdir(parents=True, exist_ok=True)
    return image_example_path / f"{name}"


def get_hopfield_network_path():
    return Path(__file__).resolve().parent / "hopfield_network.npy"


def get_name_gif_path(name):
    gif_path = Path(__file__).resolve().parent / "name_gifs"
    if not gif_path.exists():
        gif_path.mkdir(parents=True, exist_ok=True)
    return gif_path / f"{name}.gif"


def get_weight_matrix_path():
    return Path(__file__).resolve().parent / "weight_matrix.png"


# ------------------------------ NAME VISUALIZATION ----------------------------
def get_offsets(names_list):
    font = ImageFont.load_default()
    left, top, right, bottom = map(np.array, zip(*[font.getbbox(n) for n in names_list]))
    return left, top, right, bottom


def generate_saturated_color(min_saturation=100):
    """Make a saturated RGB color"""
    r = random.randint(min_saturation, 255)
    g = random.randint(min_saturation, 255)
    b = random.randint(min_saturation, 255)
    use_colors = np.random.rand(3) > 0.66  # 1/3 chance for each color
    if np.all(~use_colors):
        # Make sure at least one is always used
        use_colors[np.random.randint(0, 3)] = True
    r = r * use_colors[0]
    g = g * use_colors[1]
    b = b * use_colors[2]
    return r, g, b


def generate_name_image(name, width, height, bg_color=(0, 0, 0), text_color=None, x_offset=0, y_offset=0):
    """Add a name to a PIL image with a random color, black background, and potential x/y offset"""
    # Create a new image with the given size and background color
    if text_color is None:
        img = Image.new("1", (width, height))
    else:
        img = Image.new("RGB", (width, height), bg_color)

    draw = ImageDraw.Draw(img)

    # Load a font (you may need to specify the path to a font file on your system)
    font = ImageFont.load_default()

    # Calculate text size
    left, top, right, bottom = font.getbbox(name)
    text_width = right - left
    text_height = bottom - top
    position = ((width - text_width) // 2 + x_offset, (height - text_height) // 2 + y_offset)

    # Draw the text on the image
    if text_color is None:
        draw.text(position, name, font=font, fill=1)
    else:
        draw.text(position, name, font=font, fill=text_color)

    # Return the image
    return np.array(img)


def create_gif_from_array(array, output_filename, duration=100):
    "Array should be (n_pixel_rows, n_pixel_columns, n_frames). Duration is in ms."

    if not output_filename.lower().endswith(".gif"):
        output_filename += ".gif"

    # Ensure the array is in uint8 format
    if array.dtype != np.uint8:
        array = (array - array.min()) / (array.max() - array.min()) * 255
        array = array.astype(np.uint8)

    # Create a list to store individual frames
    frames = []

    # Convert each frame to an image and append to frames list
    for i in range(array.shape[2]):
        frame = array[:, :, i]
        img = Image.fromarray(frame)
        frames.append(img)

    # Save the frames as an animated GIF
    frames[0].save(output_filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)


# -------------------------------- BINARY HANDLING -----------------------------------
def image_to_binary(image):
    """
    Convert a (..., width x height x 3) int8 image array to a (..., width x height x 24) binary array.
    """
    # Ensure the input is the correct shape and type
    assert image.shape[-1] == 3 and image.dtype == np.uint8, "Input must be (..., width, height, 3) with dtype int8"
    assert image.ndim >= 3, "Image must have at least 3 dimensions"

    # Create a 24-bit integer representation
    int32 = image.astype(np.uint32)
    int32 = (int32[..., 0] << 16) | (int32[..., 1] << 8) | int32[..., 2]

    # Convert to binary
    binary = np.unpackbits(int32.view(np.uint8), axis=-1, bitorder="big")

    # Reshape to (width, height, 24)
    new_shape = [*image.shape[:-1], 32]
    binary = binary.reshape(*new_shape)
    binary = binary[..., :24][..., ::-1]  # Remove the extra bits and reverse the order
    return binary


def binary_to_image(binary):
    """
    Convert a (..., width x height x 24) binary array back to a (..., width x height x 3) int8 image array.
    """
    # Ensure the input is the correct shape and type
    assert binary.shape[-1] == 24 and np.array_equal(binary, binary.astype(bool)), "Input must be (width, height, 24) with 1/0"

    # Reshape and pack bits
    int24 = np.packbits(binary[..., ::-1], axis=-1, bitorder="big")
    pad = np.zeros_like(int24[..., [0]])
    int24 = np.concatenate([int24, pad], axis=-1)
    int32 = int24.view(np.uint32)

    # Extract RGB channels
    r = (int32 >> 16) & 255
    g = (int32 >> 8) & 255
    b = int32 & 255

    # Combine channels and convert to int8
    return np.stack([r, g, b], axis=-1).astype(np.uint8).squeeze(-2)


# -------------------------------- EXTRAS (Useful for testing mostly) ---------------------------
def rgb_to_binary(r, g, b):
    """
    Convert RGB values (0-255) to a binary string.
    Each color channel is represented by 8 bits.
    """
    return int(f"{r:08b}{g:08b}{b:08b}")


def binary_to_rgb(binary):
    """
    Convert a 24-bit binary string back to RGB values.
    """
    binary_str = str(binary)
    r = int(binary_str[:8], 2)
    g = int(binary_str[8:16], 2)
    b = int(binary_str[16:], 2)
    return r, g, b


def list_to_number_string(digit_list):
    """
    Convert a list of digits to a single number string.

    :param digit_list: List or NumPy array of digits
    :return: String representation of the number
    """
    return "".join(map(str, digit_list))


def binary_array_to_decimal(binary_array):
    """
    Convert a multi-dimensional NumPy array of binary digits (0s and 1s) to its decimal representation.
    The conversion is applied to the last axis.

    :param binary_array: NumPy array of binary digits (0s and 1s)
    :return: Decimal (base 10) representation of the binary numbers
    """
    # Ensure the input contains only 0s and 1s
    assert np.all(np.isin(binary_array, [0, 1])), "Input must contain only 0s and 1s"

    # Create an array of powers of 2, reversed
    powers_of_2 = 2 ** np.arange(binary_array.shape[-1])[::-1]

    # Use dot product along the last axis
    decimal = np.dot(binary_array, powers_of_2)

    return decimal


# ------------------------------------- HOPFIELD CODE -------------------------------------
def sign(x):
    # unlike numpy sign function, returns 1 for values of 0
    return jnp.sign(x) + (x == 0)


def get_nearest_pattern(query, patterns):
    i = jnp.argmax(patterns @ query)
    p_star = patterns[i]
    return p_star


def check_pattern(query, patterns):
    return jnp.any(jnp.all(patterns == query, -1)) or np.any(jnp.all(patterns == -1 * query, -1))


def generate_random_pattern(key, shape):
    if isinstance(shape, int):
        shape = (shape,)
    return sign(jr.uniform(key, shape, minval=-1, maxval=1))


def hebbian_weights(patterns):
    W = patterns.T @ patterns
    return W


@partial(jit, static_argnames="num_iters")
def simulate(x_0, W, b=0, beta=0, num_iters=10):
    def _step(x_t, b_t):
        x_tp1 = sign(jnp.einsum("ij,...i->...j", W, x_t) - b_t)
        return x_tp1, x_tp1

    b = b * jnp.ones((num_iters, *x_0.shape))
    _, x = jax.lax.scan(_step, x_0, b)
    x = jnp.r_[x_0[jnp.newaxis], x]
    return x.astype(np.int8)


@partial(jit, static_argnames="num_iters")
def simulate_probabilistic(x_0, W, b=0, beta=0, end_beta=None, num_iters=10):
    key = jax.random.key(int(time.time()))  # Random seed

    def _step(x_t, b_t):
        bias, beta = b_t
        logit = jnp.einsum("ij,...i->...j", W, x_t) - bias
        prob = 1.0 / (1.0 + jnp.exp(-logit / beta))
        random_array = jax.random.uniform(key, shape=prob.shape)
        binary_array = (prob > random_array).astype(jnp.int32)
        x_tp1 = binary_array * 2.0 - 1.0
        return x_tp1, x_tp1

    b = b * jnp.ones((num_iters, *x_0.shape))
    if end_beta is None:
        beta = beta * jnp.ones((num_iters, *x_0.shape))
    else:
        beta = 0.3
        end_beta = 0.01
        beta = jnp.linspace(beta, end_beta, num_iters)
        beta = beta[:, *[jnp.newaxis] * x_0.ndim]
        beta = jnp.broadcast_to(beta, (beta.shape[0], *x_0.shape))
    bb = jnp.stack((b, beta), axis=1)
    _, x = jax.lax.scan(_step, x_0, bb)
    x = jnp.r_[x_0[jnp.newaxis], x]
    return x.astype(np.int8)


@jit
def pseudo_inv(memory_patterns):
    N = memory_patterns.shape[1]
    C = jnp.einsum("mi,vi->mv", memory_patterns, memory_patterns) / N
    Q = jnp.linalg.pinv(C)
    W = jnp.einsum("mi,mv,vj->ij", memory_patterns, Q, memory_patterns)
    return W


@jit
def pseudo_inverse_learning(memory_patterns):
    # Compute the pseudo-inverse
    P_pinv = jnp.linalg.pinv(memory_patterns)

    # Compute the weight matrix
    W = jnp.dot(memory_patterns.T, P_pinv.T)

    # Set diagonal elements to zero (no self-connections)
    W = jnp.fill_diagonal(W, 0, inplace=False)

    return W
