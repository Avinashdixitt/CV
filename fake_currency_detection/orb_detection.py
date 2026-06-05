import cv2

def extract_orb_matches(real_path, test_path):

    img1 = cv2.imread(real_path, 0)
    img2 = cv2.imread(test_path, 0)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    return len(matches)