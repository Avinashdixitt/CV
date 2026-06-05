import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_density(real_path, test_path):

    real = cv2.imread(real_path, 0)
    test = cv2.imread(test_path, 0)

    real_edges = cv2.Canny(real, 100, 200)
    test_edges = cv2.Canny(test, 100, 200)

    density = np.sum(test_edges > 0) / (
        test_edges.shape[0] * test_edges.shape[1]
    )

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18,6))

    axes[0].imshow(real, cmap='gray')
    axes[0].set_title("REAL Currency")
    axes[0].axis("off")

    axes[1].imshow(test, cmap='gray')
    axes[1].set_title("TEST Currency")
    axes[1].axis("off")

    axes[2].imshow(test_edges, cmap='gray')
    axes[2].set_title("Canny Edge Detection")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return density