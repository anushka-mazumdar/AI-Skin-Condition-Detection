import os
import cv2
import numpy as np

# === Paths ===
RAW_DIR = "data/raw"
ENHANCED_DIR = "data/enhanced"

os.makedirs(ENHANCED_DIR, exist_ok=True)


def enhance_image(img):
    """
    Enhances image clarity and tone balance.
    Steps:
        1. Denoise
        2. Convert to LAB color space and equalize lightness
        3. Sharpen
        4. Adjust contrast slightly
    """
    # Denoise (useful for low-light webcam captures)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # LAB equalization
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    eq_img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(eq_img, -1, kernel)

    # Slight contrast and brightness adjustment
    enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
    return enhanced


def enhance_all_images(raw_dir=RAW_DIR, enhanced_dir=ENHANCED_DIR):
    """
    Enhances all images in `raw_dir` and saves them to `enhanced_dir`.
    Preserves filenames.
    """
    for fname in os.listdir(raw_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(raw_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Could not read {fname}")
                continue

            enhanced = enhance_image(img)
            save_path = os.path.join(enhanced_dir, fname)
            cv2.imwrite(save_path, enhanced)
            print(f"[INFO] Enhanced: {fname}")

    print(f"\n✅ Enhancement complete. Enhanced images saved to: {enhanced_dir}")


if __name__ == "__main__":
    enhance_all_images()
