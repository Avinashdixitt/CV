import cv2
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

def calculate_ssim(real_path, test_path):

    # Read images
    img1 = cv2.imread(real_path, 0)
    img2 = cv2.imread(test_path, 0)

    if img1 is None or img2 is None:

        print("Error loading images")

        return 0

    # Resize
    img1 = cv2.resize(img1, (500,300))
    img2 = cv2.resize(img2, (500,300))

    # Calculate SSIM
    score, diff = ssim(
        img1,
        img2,
        full=True
    )

    diff = (diff * 255).astype("uint8")

    # ======================================
    # VISUALIZATION
    # ======================================

    fig, axes = plt.subplots(1, 3, figsize=(18,6))

    # REAL IMAGE
    axes[0].imshow(img1, cmap='gray')

    axes[0].set_title("REAL Currency")

    axes[0].axis("off")

    # TEST IMAGE
    axes[1].imshow(img2, cmap='gray')

    axes[1].set_title("TEST / FAKE Currency")

    axes[1].axis("off")

    # DIFFERENCE MAP
    axes[2].imshow(diff, cmap='gray')

    axes[2].set_title(f"SSIM Difference Map\nScore: {score:.4f}")

    axes[2].axis("off")

    plt.tight_layout()

    plt.show()

    return score