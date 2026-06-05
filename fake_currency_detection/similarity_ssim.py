import cv2

from skimage.metrics import structural_similarity as ssim

def calculate_ssim(real_path, test_path):

    img1 = cv2.imread(real_path, 0)
    img2 = cv2.imread(test_path, 0)

    img1 = cv2.resize(img1, (500, 300))
    img2 = cv2.resize(img2, (500, 300))

    score, _ = ssim(
        img1,
        img2,
        full=True
    )

    return score