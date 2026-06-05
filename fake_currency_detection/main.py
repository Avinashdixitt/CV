from texture_lbp import extract_lbp_score
from orb_detection import extract_orb_matches
from edge_detection import edge_density
from thread_detection import detect_thread
from similarity_ssim import calculate_ssim

# Image paths
REAL_NOTE = "images/real1.jpg"
TEST_NOTE = "images/fake11.avif"

# ----------------------------
# LBP Texture
# ----------------------------

lbp_score = extract_lbp_score(TEST_NOTE)

print("\nLBP Texture Score:", lbp_score)

# ----------------------------
# ORB Keypoints
# ----------------------------

orb_matches = extract_orb_matches(
    REAL_NOTE,
    TEST_NOTE
)

print("ORB Matches:", orb_matches)

# ----------------------------
# Edge Density
# ----------------------------

edge_score = edge_density(TEST_NOTE)

print("Edge Density:", edge_score)

# ----------------------------
# Security Thread
# ----------------------------

thread_score = detect_thread(TEST_NOTE)

print("Thread Lines:", thread_score)

# ----------------------------
# SSIM Similarity
# ----------------------------

ssim_score = calculate_ssim(
    REAL_NOTE,
    TEST_NOTE
)

print("SSIM Score:", ssim_score)

# ----------------------------
# Final Hybrid Decision
# ----------------------------

final_score = (
    (ssim_score * 50)
    +
    (orb_matches / 10)
    +
    (thread_score / 10)
    +
    (edge_score * 100)
)

print("\nFinal Score:", final_score)

# Threshold
if final_score > 60:

    print("\nPrediction: REAL Currency")

else:

    print("\nPrediction: FAKE Currency")