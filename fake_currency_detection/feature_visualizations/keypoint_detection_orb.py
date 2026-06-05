import cv2
import matplotlib.pyplot as plt

def extract_orb_matches(real_path, test_path):

    img1 = cv2.imread(real_path, 0)
    img2 = cv2.imread(test_path, 0)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    real_kp = cv2.drawKeypoints(
        img1,
        kp1,
        None,
        color=(0,255,0)
    )

    test_kp = cv2.drawKeypoints(
        img2,
        kp2,
        None,
        color=(0,255,0)
    )

    # Match
    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    matched_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:50],
        None,
        flags=2
    )

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20,6))

    axes[0].imshow(real_kp)
    axes[0].set_title("REAL ORB Features")
    axes[0].axis("off")

    axes[1].imshow(test_kp)
    axes[1].set_title("TEST ORB Features")
    axes[1].axis("off")

    axes[2].imshow(matched_img)
    axes[2].set_title("ORB Feature Matching")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return len(matches)