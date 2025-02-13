# Hiker Location Assignment

## Overview
This assignment focuses on implementing basic shape detection using the PyImageSearch ShapeDetector. The objective is to read an input image, preprocess it to simplify shape detection, utilize the ShapeDetector to identify shapes, and finally draw the names of the shapes on the image.

## Files

File Structure:

    ```plaintext
    |-- pyimagesearch/
    |   |-- shapedetector.py
    |-- detect_shapes.py
    |-- shapes_and_colors.png

1. **shapedetector.py**: contains the `ShapeDetector` class, which is utilized for detecting basic shapes such as triangles, squares, rectangles, pentagons, and circles based on the contours of objects in an image.

2. **detect_shapes.py**: is the main driver for the assignment. It reads an input image, resizes it for better shape approximation, converts it to grayscale, applies Gaussian blur and thresholding, and then utilizes the `PyImageSearch ShapeDetector` to identify and label shapes on the image. The final output image is displayed, showing contours and shape labels.

## Instructions
1. Install dependencies:

    ```bash
    pip install opencv-python imutils

2. Run the `detect_shapes.py` script:

    ```bash
    python detect_shapes.py --image shapes_and_colors.png

3. The output image will be displayed, showing contours and labeled shapes.

## Output
The final labeled shapes image will be displayed, with contours drawn around detected shapes, and the corresponding shape names labeled on the image.

## Note
The `shapes_and_colors.png` file is provided as a test image. You can replace it with your own images for shape detection.

The output images are displayed one at a time (with each new image containing one more labeled shape than the last). In order to see the next image, close the previous one.

The `labeled.png` file is provided as the final output from running detect_shapes on the provided test image.