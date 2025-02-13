# Hiker Location Assignment

## Overview

This assignment involves finding the location of a lost hiker on a 4x4 map. Each grid in the map contains an image, and the goal is to identify the hiker's location using the AlexNet model for image classification.

## Files

1. **world.py**: Contains the `create_map()` function that generates a 4x4 map with image filenames. The hiker's image is randomly placed on the map.

2. **hiker_locator.py**: The main script that loads the map, identifies the hiker's location, and uses a pre-trained AlexNet model to classify the hiker's image.

## Instructions

1. Install dependencies:

   ```bash
   pip install torch torchvision

2. Run the solve.py script:

    ```bash
    python solve.py

3. View the output indicating the hiker's location
    - I included some code that prints out the map as it was generated to verify the correctness of the algorithm

## Code Details
- The `classify_image` function loads the pre-trained AlexNet model and classifies the provided image using PyTorch and torchvision.

- The `find_hiker_location` function iterates through the map, identifies the hiker's location based on the classified image, and prints the result.

- The script prints a formatted representation of the map with centered image filenames for visual clarity.

## Dependencies
- PyTorch
- torchvision
- Pillow(PIL)

## Note
In playing around with the code, I found that the class index `795` appears to correspond to the hiker class in the AlexNet model.

It is also assumed that when running the code, solve.py is placed within the same directory as the rest of the starter code.