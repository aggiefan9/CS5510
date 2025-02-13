import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import AlexNet_Weights
from world import create_map
from PIL import Image

# Step 1: Use a pre-trained model to classify the hiker's image
def classify_image(image_path):
    # Load the pre-trained AlexNet model
    model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(image)

    # Get the predicted class index
    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

# Step 2: Return the coordinates of the hiker on the map
def find_hiker_location(map):
    hiker_location = None

    for i in range(4):
        for j in range(4):
            image_path = os.path.join(os.getcwd(), map[i][j])
            predicted_class = classify_image(image_path)
            # print(f"[{i},{j}] = {predicted_class}")

            # Assuming class index 1 corresponds to the hiker class (adjust if needed)
            if predicted_class == 795:
                hiker_location = [i, j]

    if hiker_location is not None:
        print(f"Found hiker on map{hiker_location}")
    else:
        print("Hiker not found in the provided map.")

if __name__ == "__main__":
    map = create_map()
    find_hiker_location(map)
    print("Map as generated:")
    print("  |           0            |           1            |           2            |           3            |")
    print("--+------------------------+------------------------+------------------------+------------------------+")
    for i in range(len(map)):
        print(f"{i} |", end=" ")
        for col in map[i]:
            print(col.center(22), end=" | ")
        print("\n--+------------------------+------------------------+------------------------+------------------------+")