from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import utils


if __name__ == "__main__":
    parser = ArgumentParser(description="Build name images")
    parser.add_argument("--use-color", default=False, action="store_true", help="If used, will save names as colors")
    args = parser.parse_args()

    # Load data
    name_data = np.load(utils.get_name_data_path(), allow_pickle=True).item()

    # Construct names (usually first, sometimes last also)
    names = name_data["first_names"]
    if name_data["args"]["full_name"]:
        last_names = name_data["last_names"]
        names = [f"{first} {last}" for first, last in zip(names, last_names)]

    # Get other data
    image_width = name_data["image_width"]
    image_height = name_data["image_height"]
    x_offsets = name_data["x_offsets"]
    y_offsets = name_data["y_offsets"]
    roles = name_data["roles"]
    r = name_data["r"]
    g = name_data["g"]
    b = name_data["b"]

    # Create images for each name
    num_names = len(names)
    name_images = []
    for iname in range(num_names):
        cimage = utils.generate_name_image(
            names[iname],
            image_width,
            image_height,
            bg_color=(0, 0, 0),
            text_color=(r[iname], g[iname], b[iname]) if args.use_color else None,
            x_offset=x_offsets[iname],
            y_offset=y_offsets[iname],
        )
        name_images.append(cimage)

        # Also save a png
        cpath = utils.get_name_image_example_path(names[iname])
        plt.imshow(cimage)
        plt.savefig(cpath)
        plt.close("all")

    # Save images
    name_images = np.array(name_images)
    np.save(utils.get_name_images_path(), name_images)
