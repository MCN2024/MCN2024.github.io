from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image
import utils


def get_name_patterns(filter_by_used=False):
    """
    Load images from stored path

    Notes: depending on how they were saved, images are either black/white or in color
    This will detect that and handle them appropriately (will convert to 24bit binary if in color)

    Converts data to Hopfield style {-1, 1}

    Will always detect which pixels are always set to -1 and filter those out so the Hopfield network
    can be smaller. The ones that are used are =True in idx_used. This is required for visualizing the
    result of Hopfield simulations (need to organize the Hopfield data to be the right pixels etc)
    - That's why we return the original image array - we need the shape to reconstruct accurately
    """
    images = np.load(utils.get_name_images_path())
    using_color = (images.shape[-1] == 3) and (images.ndim == 4)
    if using_color:
        images = utils.image_to_binary(images)

    images = 2.0 * images - 1.0
    images = jnp.array(images)

    P = images.shape[0]
    full_patterns = images.reshape((P, -1)).astype(jnp.float32)
    if using_color:
        if filter_by_used:
            idx_used = np.any(full_patterns == 1, axis=0)
        else:
            images_plus = np.copy(images)
            add_bits_idx = [7, 15, 23]
            images_plus[:, :, :, add_bits_idx] = 1
            full_patterns_plus = images_plus.reshape((P, -1)).astype(jnp.float32)
            idx_used = np.any(full_patterns_plus == 1, axis=0)
    else:
        idx_used = np.ones(full_patterns.shape[-1], dtype=bool)
    memory_patterns = full_patterns[:, idx_used]

    return memory_patterns, images, idx_used


def show_weights(W):
    w_max = jnp.max(jnp.abs(W)) / 10
    plt.imshow(W, interpolation="none", vmin=-w_max, vmax=w_max, cmap="coolwarm")
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.savefig(utils.get_weight_matrix_path(), dpi=300)
    plt.show()


def run_simulations(memory_patterns, W, sigma=5, b=1e-2, start_beta=0.5, end_beta=0.0001, num_per_name=5, max_attempts=50, max_T=10):
    P, N = memory_patterns.shape
    name_simulations = []
    name_success = []
    name_attempts = []
    # Run simulations for each name independently
    for p in tqdm(range(P)):
        # Requires num_per_name successful simulations, but it's helpful to parallelize
        success = False
        attempts = 0
        # Store the results in a 3D array (T, num_per_name, N) and check for successful convergence
        successful_results = np.zeros((max_T + 1, num_per_name, N))
        converged = np.all(successful_results[-1] == memory_patterns[p], axis=1)
        while not success:
            # Reduce sigma to increase the probability of convergence as we try more times for each name
            c_sigma = sigma * (1.0 - 0.999 * attempts / max_attempts)
            # Measure how many more successful attempts are required
            num_required = num_per_name - np.sum(converged)
            # Add noise to the memory pattern and run the simulation however many times are still needed
            noise_patterns = np.repeat(memory_patterns[p][np.newaxis], num_required, axis=0) + c_sigma * np.random.randn(num_required, N)
            c_results = utils.simulate_probabilistic(noise_patterns, W, b=b, beta=start_beta, end_beta=end_beta, num_iters=max_T)
            # Check which ones worked, add them to the successful results, and check if we're done
            c_converged = np.all(c_results[-1] == memory_patterns[p], axis=1)
            update_idx = np.where(~converged)[0][c_converged]
            successful_results[:, update_idx] = c_results[:, c_converged]
            converged = np.all(successful_results[-1] == memory_patterns[p], axis=1)
            success = np.all(converged).item()
            attempts += 1
            if attempts > max_attempts:
                print(f"Failed to converge on {p} after {max_attempts} attempts. Total success: {np.sum(converged)}")
                break
        if success:
            # Concatenate the successful results and store them into a single movie
            stacked_results = np.transpose(successful_results, axes=(1, 0, 2)).reshape(-1, N)
            name_simulations.append(stacked_results)
        else:
            name_simulations.append(successful_results)
        name_attempts.append(attempts)
        name_success.append(success)
    return name_simulations, name_success, name_attempts


def reconstruct_rgbs(output, height, width, idx_used):
    """Restore flattened Hopfield output to RGB image with intended shape"""
    # Initialize output array to include all dimensions in image (some are ignored by the Hopfield network)
    full_output = -1.0 * np.ones((output.shape[0], len(idx_used)))
    # Put the hopfield output where it's supposed to go and convert to binary
    full_output[:, idx_used] = output
    full_binary = ((full_output + 1.0) // 2).astype(bool)

    # Reshape to image dimensions, and convert 24bit binary to 3-channel RGB
    binary_images = np.reshape(full_binary, (full_binary.shape[0], height, width, 24))
    rgb_images = utils.binary_to_image(binary_images)
    return rgb_images


def create_gif(filename, images, scaleup=1, duration=100):
    num_frames = images.shape[0]
    frames = []
    for i in range(num_frames):
        frame = Image.fromarray(np.array(images[i]))
        frames.append(frame)

    if scaleup is not None:
        original_shape = images[0].shape
        new_shape = (original_shape[1] * scaleup, original_shape[0] * scaleup)
        frames = [f.resize(new_shape) for f in frames]

    # Save the frames as an animated GIF
    frames[0].save(filename, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a Hopfield network on names and save gifs of the results")
    parser.add_argument("--redo-training", default=False, action="store_true", help="Whether to redo the training")
    parser.add_argument("--filter-by-used", default=False, action="store_true", help="Whether to filter out unused pixels")
    parser.add_argument("--visualize-weights", default=False, action="store_true", help="Whether to visualize the weights (if used, won't save gifs)")
    parser.add_argument("--sigma", type=float, default=5, help="Standard deviation of noise to add to memory patterns")
    parser.add_argument("--b", type=float, default=1e-2, help="Bias term for the Hopfield network")
    parser.add_argument("--start_beta", type=float, default=0.5, help="Starting beta for the Hopfield network")
    parser.add_argument("--end_beta", type=float, default=0.0001, help="Ending beta for the Hopfield network")
    parser.add_argument("--num_per_name", type=int, default=5, help="Number of simulations to run per name")
    parser.add_argument("--max_attempts", type=int, default=20, help="Maximum number of attempts to converge")
    parser.add_argument("--max_T", type=int, default=10, help="Maximum number of iterations to run")
    parser.add_argument("--scaleup", type=int, default=1, help="Scale up the images for the gif")
    parser.add_argument("--duration", type=int, default=100, help="Duration of each frame in the gif")
    args = parser.parse_args()

    # Load names and general data
    name_data = np.load(utils.get_name_data_path(), allow_pickle=True).item()

    # Load the network if it exists and we're not redoing the training
    loaded = False
    if not args.redo_training:
        if utils.get_hopfield_network_path().exists():
            network = np.load(utils.get_hopfield_network_path(), allow_pickle=True).item()
            loaded = True

    # If we're redoing the training or the network doesn't exist, train the network, run simulations, save it
    if not loaded:
        # Load memory patterns, original images, and idx to relevant dimensions in full images
        memory_patterns, images, idx_used = get_name_patterns(args.filter_by_used)
        height, width = images.shape[1], images.shape[2]

        # Train the Hopfield network (just need to learn the weights)
        W = utils.pseudo_inverse_learning(memory_patterns)

        # Run the simulations
        name_simulations, name_success, name_attempts = run_simulations(
            memory_patterns,
            W,
            sigma=args.sigma,
            b=args.b,
            start_beta=args.start_beta,
            end_beta=args.end_beta,
            num_per_name=args.num_per_name,
            max_attempts=args.max_attempts,
            max_T=args.max_T,
        )

        # Save network
        network = dict(
            W=W,
            name_simulations=name_simulations,
            name_success=name_success,
            name_attempts=name_attempts,
            idx_used=idx_used,
            height=height,
            width=width,
            args=vars(args),
        )

        np.save(utils.get_hopfield_network_path(), network)

    # Visualize the weights if requested
    if args.visualize_weights:
        show_weights(network["W"])

    else:
        # Save the results as gifs
        for iname, (simulation, success) in enumerate(zip(network["name_simulations"], network["name_success"])):
            if not success:
                print(f"Failed to converge on {iname} after {args.max_attempts} attempts")
                continue
            name = name_data["first_names"][iname]
            filename = utils.get_name_gif_path(name)
            rgb_images = reconstruct_rgbs(simulation, network["height"], network["width"], network["idx_used"])
            create_gif(filename, rgb_images, scaleup=args.scaleup, duration=args.duration)
            print(f"Saved {name} simulation to {filename}")
