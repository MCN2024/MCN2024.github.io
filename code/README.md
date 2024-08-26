---
layout: page
title: Instructions for building and visualizing the Hopfield network
permalink: /code/
---

To view this page and the relevant code on GitHub, go [here](https://github.com/MCN2024/MCN2024.github.io/tree/main/code).

## Constructing the patterns
The patterns that our Hopfield network stores are our names (usually first, but can save last too). The names are represented as text in an image with a unique color and spatial offset for each person. This is fun, but also helpful because it decreases correlations in the Hopfield network's attractor states so makes for a much more stable representation. 

### Choosing colors and offsets for the name representations
To register the names, colors, and spatial offsets, run the ``save_names_and_params.py`` script. You can do it like this:
```python
python code/save_names_and_params.py # {options: --full-name, --min-saturation, --horizontal/vertical-padding}
```

This will save a file called ``name_data.npy`` which contains a dictionary of all the relevant parameters in the code directory. 

### Saving the images
To build the name images and save as a numpy array, run the ``build_name_images.py`` script.
```python
python code/build_name_images.py
```

This will save a numpy array in a file called ``name_images.npy`` which is a (names x height x width, 3) array of RGB images for each name. 

## Training the Hopfield network
#### Note on representing RGB data
Hopfield networks require data with values equal to 1 or -1. This is a challenge for RGB data with values scaling from 0-255. Solution? Binary! First, we assume that a -1 is equal to 0, then represent each of the R/G/B channels as an 8 bit binary word stacked, such that an RGB image with (height, width, 3) will be transformed to a binary image of shape (height, width, 24). 

To do this transformation, use the methods in utils called ``image_to_binary`` and ``binary_to_image``. 

### Training, Visualizing, Saving GIFs
All of these steps are done in one script: ``train_classic_hopfield.py``. There are options, look at the argument parser in the script to explore them. To use it, run:
```python
python code/train_classic_hopfield.py
```

This will:
1. Load the name data
2. Load the name images (which are the memory patterns to learn)
3. Train the Hopfield network (which corresponds to learning the weights with the pseudoinverse method)
4. Store several iterations of the network converging to each name
5. Save all iterations per name as a single GIF in the code/name_gifs folder. 

Notes on the simulations:
Hopfield dynamics are apparently pretty finnicky due to false stable points, competition, and constructive interference. To make the visualizations interesting and work, we used a few tricks:
- The activity updates are probabilistic with an annealing temperature parameter, such that updates are noisy at the beginning and deterministic at the end of each run. See the ``simulate_probabilistic()`` method in ``utils.py`` for details.
- We used a small positive bias, which helped a little bit (this wasn't a systematic check, so it might be imperfect, but it works, so...).




