import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_thread(real_path, test_path):

    real = cv2.imread(real_path)
    test = cv2.imread(test_path)

    gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    output = test.copy()

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            cv2.line(
                output,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18,6))

    axes[0].imshow(cv2.cvtColor(real, cv2.COLOR_BGR2RGB))
    axes[0].set_title("REAL Currency")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
    axes[1].set_title("TEST Currency")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Detected Thread Lines")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    if lines is None:
        return 0

    return len(lines)