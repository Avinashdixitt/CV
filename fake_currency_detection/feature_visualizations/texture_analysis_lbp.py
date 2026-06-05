import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern

def extract_lbp_score(real_path, test_path):

    # Read images
    real = cv2.imread(real_path)
    test = cv2.imread(test_path)

    real_gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    # LBP Parameters
    radius = 1
    n_points = 8 * radius

    # Apply LBP
    real_lbp = local_binary_pattern(
        real_gray,
        n_points,
        radius,
        method="uniform"
    )

    test_lbp = local_binary_pattern(
        test_gray,
        n_points,
        radius,
        method="uniform"
    )

    # Texture Score
    texture_score = np.mean(test_lbp)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18,6))

    axes[0].imshow(real_gray, cmap='gray')
    axes[0].set_title("REAL Currency")
    axes[0].axis("off")

    axes[1].imshow(test_gray, cmap='gray')
    axes[1].set_title("TEST Currency")
    axes[1].axis("off")

    axes[2].imshow(test_lbp, cmap='gray')
    axes[2].set_title("LBP Texture Pattern")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return texture_score