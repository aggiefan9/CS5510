import os
import cv2
import numpy as np

# Define the size of the calibration pattern (e.g., chessboard)
pattern_size = (4, 5)  # Adjust based on your calibration pattern

# Set termination criteria for the corner subpixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)

# Arrays to store object points and image points from all the calibration images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Create a grid of 3D points for the calibration pattern
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Capture images for calibration
# Replace 'your_image_folder' with the actual path to your image folder
# image_folder = os.path.join(os.getcwd(), 'img')
image_folder = 'img'
image_files = [f'image_{i}.jpg' for i in range(500, 520)]  # Change to real names after 

for image_file in image_files:
    # Read the image
    img = cv2.imread(image_folder + '/' + image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if not img.any():
        print("Image not found")
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If corners are found, refine them
    if ret:
        objpoints.append(objp)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)  # Adjust the wait time as needed
    else:
        print("No corners found")

cv2.destroyAllWindows()

# Calibrate the camera to obtain intrinsic parameters
image_size = gray.shape[::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

# Print the intrinsic parameters
print("Intrinsic Matrix (Camera Matrix):")
print(mtx)

print("\nDistortion Coefficients:")
print(dist)

# Compute extrinsic parameters for a specific image
image_index = 5  # Adjust based on the image you want to use for computing extrinsic parameters
rvec, tvec = rvecs[image_index], tvecs[image_index]

# Transformation matrix (rotation and translation)
rotation_matrix, _ = cv2.Rodrigues(rvec)
transformation_matrix = np.hstack((rotation_matrix, tvec.reshape(-1, 1)))
transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))

print("Extrinsic Parameters (Transformation Matrix):")
print(transformation_matrix)