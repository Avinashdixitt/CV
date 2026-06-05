import cv2
import numpy as np

def edge_density(image_path):

    image = cv2.imread(image_path, 0)

    edges = cv2.Canny(
        image,
        100,
        200
    )

    edge_pixels = np.sum(edges > 0)

    total_pixels = edges.shape[0] * edges.shape[1]

    density = edge_pixels / total_pixels

    return density