import os
import json


def update_gif_manifest():
    gif_dir = "code/name_gifs"
    manifest_file = "gif-manifest.json"

    # Get all .gif files in the directory
    gif_files = [f for f in os.listdir(gif_dir) if f.endswith(".gif")]

    # Create manifest data
    manifest_data = {"gifs": gif_files}

    # Write to JSON file
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Updated {manifest_file} with {len(gif_files)} GIFs.")


if __name__ == "__main__":
    update_gif_manifest()
