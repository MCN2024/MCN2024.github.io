from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image
import utils


def get_name_patterns():
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
        idx_used = np.any(full_patterns == 1, axis=0)
    else:
        idx_used = np.ones(full_patterns.shape[1], dtype=bool)
    memory_patterns = full_patterns[:, idx_used]

    return memory_patterns, images, idx_used


def show_weights(W):
    plt.imshow(jnp.abs(W) > 5, vmin=0, vmax=1, interpolation="none", cmap="Greys")
    plt.show()

    plt.imshow(W, interpolation="none", cmap="jet")
    plt.show()


def run_simulations(memory_patterns, W, sigma=5, b=1e-2, start_beta=0.5, end_beta=0.0001, num_per_name=5, max_attempts=50, max_T=10):
    P, N = memory_patterns.shape
    name_simulations = []
    name_success = []
    name_attempts = []
    for p in tqdm(range(P)):
        success = False
        attempts = 0
        while not success:
            noise_patterns = np.repeat(memory_patterns[p][np.newaxis], num_per_name, axis=0) + sigma * np.random.randn(num_per_name, N)
            results = utils.simulate_probabilistic(noise_patterns, W, b=b, beta=start_beta, end_beta=end_beta, num_iters=max_T)
            converged = np.all(results[-1] == memory_patterns[p], axis=1)
            success = np.all(converged).item()
            attempts += 1
            if attempts > max_attempts:
                break
        if success:
            stacked_results = np.transpose(results, axes=(1, 0, 2)).reshape(-1, N)
            name_simulations.append(stacked_results)
        else:
            name_simulations.append(results)
        name_attempts.append(attempts)
        name_success.append(success)
    return name_simulations, name_success, name_attempts


def reconstruct_rgbs(images, height, width, idx_used):
    asim_activity = -1.0 * np.ones((images.shape[0], len(idx_used)))
    asim_activity[:, idx_used] = images
    asim_binary = ((asim_activity + 1.0) // 2).astype(bool)

    sim_images = np.reshape(asim_binary, (asim_binary.shape[0], height, width, 24))
    rgb_images = utils.binary_to_image(sim_images)
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
    parser.add_argument("--visualize-weights", default=False, action="store_true", help="Whether to visualize the weights")
    parser.add_argument("--sigma", type=float, default=5, help="Standard deviation of noise to add to memory patterns")
    parser.add_argument("--b", type=float, default=1e-2, help="Bias term for the Hopfield network")
    parser.add_argument("--start_beta", type=float, default=0.5, help="Starting beta for the Hopfield network")
    parser.add_argument("--end_beta", type=float, default=0.0001, help="Ending beta for the Hopfield network")
    parser.add_argument("--num_per_name", type=int, default=5, help="Number of simulations to run per name")
    parser.add_argument("--max_attempts", type=int, default=50, help="Maximum number of attempts to converge")
    parser.add_argument("--max_T", type=int, default=10, help="Maximum number of iterations to run")
    parser.add_argument("--scaleup", type=int, default=1, help="Scale up the images for the gif")
    parser.add_argument("--duration", type=int, default=100, help="Duration of each frame in the gif")
    args = parser.parse_args()

    # Load names and general data
    name_data = np.load(utils.get_name_data_path(), allow_pickle=True).item()

    loaded = False
    if not args.redo_training:
        if utils.get_hopfield_network_path().exists():
            network = np.load(utils.get_hopfield_network_path(), allow_pickle=True).item()
            loaded = True

    if not loaded:
        # Load memory patterns, original images, and idx to relevant dimensions in full images
        memory_patterns, images, idx_used = get_name_patterns()
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
        )

        np.save(utils.get_hopfield_network_path(), network)

    # Visualize the weights if requested
    if args.visualize_weights:
        show_weights(W)

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
