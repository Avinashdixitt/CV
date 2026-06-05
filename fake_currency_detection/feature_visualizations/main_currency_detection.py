from texture_analysis_lbp import extract_lbp_score
from keypoint_detection_orb import extract_orb_matches
from edge_detection_canny import edge_density
from security_thread_hough import detect_thread
from similarity_analysis_ssim import calculate_ssim

# ==========================================
# IMAGE PATHS
# ==========================================

REAL_NOTE = "./images/real1.jpg"
TEST_NOTE = "./images/images.jpeg"

# ==========================================
# PROJECT HEADER
# ==========================================

print("\n========================================")
print("   FAKE CURRENCY DETECTION SYSTEM")
print("========================================\n")

# ==========================================
# LBP TEXTURE ANALYSIS
# ==========================================

print("Running LBP Texture Analysis...\n")

lbp_score = extract_lbp_score(
    REAL_NOTE,
    TEST_NOTE
)

print(f"LBP Texture Score: {lbp_score:.4f}\n")

# ==========================================
# ORB KEYPOINT DETECTION
# ==========================================

print("Running ORB Keypoint Detection...\n")

orb_matches = extract_orb_matches(
    REAL_NOTE,
    TEST_NOTE
)

print(f"ORB Matches: {orb_matches}\n")

# ==========================================
# CANNY EDGE DETECTION
# ==========================================

print("Running Canny Edge Detection...\n")

edge_score = edge_density(
    REAL_NOTE,
    TEST_NOTE
)

print(f"Edge Density: {edge_score:.4f}\n")

# ==========================================
# SECURITY THREAD DETECTION
# ==========================================

print("Running Security Thread Detection...\n")

thread_score = detect_thread(
    REAL_NOTE,
    TEST_NOTE
)

print(f"Detected Thread Lines: {thread_score}\n")

# ==========================================
# SSIM SIMILARITY ANALYSIS
# ==========================================

print("Running Structural Similarity Analysis...\n")

ssim_score = calculate_ssim(
    REAL_NOTE,
    TEST_NOTE
)

print(f"SSIM Score: {ssim_score:.4f}\n")

# ==========================================
# FINAL HYBRID SCORE
# ==========================================

print("Calculating Final Hybrid Score...\n")

final_score = (
    (ssim_score * 50)
    +
    (orb_matches / 10)
    +
    (thread_score / 10)
    +
    (edge_score * 100)
)

print(f"Final Score: {final_score:.2f}\n")

# ==========================================
# FINAL DECISION
# ==========================================

THRESHOLD = 60

if final_score > THRESHOLD:

    prediction = "REAL Currency"

else:

    prediction = "FAKE Currency"

# ==========================================
# FINAL OUTPUT
# ==========================================

print("========================================")
print(f" FINAL PREDICTION : {prediction}")
print("========================================\n")

# ==========================================
# FEATURE SUMMARY
# ==========================================

print("=============== FEATURE SUMMARY =================")

print(f"Texture Score (LBP)        : {lbp_score:.4f}")

print(f"ORB Feature Matches        : {orb_matches}")

print(f"Edge Density               : {edge_score:.4f}")

print(f"Security Thread Lines      : {thread_score}")

print(f"Structural Similarity      : {ssim_score:.4f}")

print(f"Final Hybrid Score         : {final_score:.2f}")

print("=================================================\n")